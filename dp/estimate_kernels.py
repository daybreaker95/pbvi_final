"""Estimate ALL kernels of the decision model from the two real-engine cohorts
(results/dp/runs/nh_quarterly = never screened, results/dp/runs/screen_random_q
= randomised colonoscopy schedules), both with quarterly 18-state recording.

Annual WAIT kernel (decision time y -> decision time y+1), conditional on
    sex, risk class, memory cell (tau group x last finding), age y
    Tw[sex, class, mem, y, from(11), to(11)]   alive & undiagnosed -> same
    Ew[sex, class, mem, y, from(11), exit(6)]  diagnosis at stage k / other death / complication death
SCREEN kernel (the colonoscopy itself), conditional on sex, class, memory, age band
    K[sex, class, mem, band, pre(11), obs(4), post(11)]
    X[sex, class, mem, band, pre(11), exit(6)]
Exit values (after diagnosis), initial belief at 40, class thresholds.

Estimation is direct maximum likelihood on annual person-year windows: the
decision-time state is the quarter-2-start snapshot, and an exit inside the
window is read from the intermediate quarterly snapshots (the first D-state
or death code), so within-year onset -> diagnosis -> death is never missed.
Sparse cells back off hierarchically (wider age window -> pool last finding
-> pool tau groups -> pool class -> pool sex).
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

from .common import (
    RES, RUNS, NC, NX, MAP18_TO_CLIN, MAP18_TO_EXIT, DEFAULT_CLASS_CUTS, risk_thresholds,
    risk_class_of, load_settings_pool, X_D1, X_DCOMP, X_DOTH, O_NORMAL, O_ADENOMA, O_MULTI, O_ADVAD,
    CLIN_NAMES, SCREEN_OBS,
)
from .kernels import (
    N_MEM, N_OL, N_TAU_G, TAU_NEVER, mem_index, OL_OF_OBS, tau_group, mem_name,
)
from .estimate_screen import BANDS, NB, band_of

Y0, Y1 = 40, 99                  # decision ages with a WAIT kernel
NY = Y1 - Y0 + 1
NO4 = len(SCREEN_OBS)
NT_W = NC + NX                   # WAIT targets: 11 clinical + 6 exits
NT_K = NO4 * NC + NX             # SCREEN targets: (obs, post) + exits
MIN_ROW_W = 150
MIN_ROW_K = 150

_TO_CODE = np.where(MAP18_TO_CLIN >= 0, MAP18_TO_CLIN, NC + MAP18_TO_EXIT).astype(np.int16)  # 18 -> 0..16


def _obs_from_removed(er, ar):
    return np.where(ar > 0, O_ADVAD, np.where(er >= 3, O_MULTI, np.where(er > 0, O_ADENOMA, O_NORMAL)))


def _memory_per_year(n, log_z, log_y, log_obs_ol, log_exit):
    """Per person and decision age y in [Y0, Y1]: (tau, ol) of the natural-
    history window starting at y. tau=0 & ol=obs if screened (non-exit) at y.
    Returns tau (NY, n) int16, ol (NY, n) int8, screened (NY, n) bool."""
    scr = np.full((101, n), -2, dtype=np.int8)        # -2 none, -1 exit screen, >=0 obs ol
    scr[log_y, log_z] = np.where(log_exit, -1, log_obs_ol).astype(np.int8)
    tau = np.full((NY, n), TAU_NEVER, dtype=np.int16)
    ol = np.zeros((NY, n), dtype=np.int8)
    screened = np.zeros((NY, n), dtype=bool)
    last_y = np.zeros(n, dtype=np.int16)
    last_o = np.zeros(n, dtype=np.int8)
    for iy, y in enumerate(range(Y0, Y1 + 1)):
        s = scr[y]
        sc = s >= 0
        has_prev = last_y > 0
        tau[iy] = np.where(sc, 0, np.where(has_prev, y - last_y, TAU_NEVER))
        ol[iy] = np.where(sc, np.maximum(s, 0), last_o)
        screened[iy] = sc
        last_y[sc] = y
        last_o[sc] = s[sc]
    return tau, ol, screened


def accumulate(paths, thr, n_class, with_logs, verbose=True):
    W = np.zeros((2, n_class, N_MEM, NY, NC, NT_W), dtype=np.int64)
    Kc = np.zeros((2, n_class, N_MEM, NB, NC, NT_K), dtype=np.int64)
    occ40 = np.zeros((2, n_class, NC), dtype=np.int64)
    dx = np.zeros((2, 101, 4, 3), dtype=np.float64)
    n_tot = 0
    t0 = time.time()
    for ip, p in enumerate(paths):
        z = np.load(p, allow_pickle=True)
        qr = z['qr']; n = qr.shape[1]; n_tot += n
        risk = z['risk'].astype(float); sex = z['sex'].astype(int) - 1
        cls = risk_class_of(risk, thr).astype(int)
        if with_logs:
            lz = z['log_z'].astype(int); ly = z['log_y'].astype(int)
            er = z['log_early_removed']; ar = z['log_adv_removed']
            cdet = z['log_cancer_detected']; comp = z['log_comp_death']
            obs = _obs_from_removed(er, ar)
            obs_ol = np.array([OL_OF_OBS[int(o)] for o in obs], dtype=np.int8) if len(obs) else np.zeros(0, np.int8)
            is_exit = cdet | comp
            tau, ol, screened = _memory_per_year(n, lz, ly, obs_ol, is_exit)
            # pre-colonoscopy state of every screen: the WAIT window y -> y+1
            # must END at the decision time of y+1, which is BEFORE a
            # colonoscopy performed at y+1 (the quarter-2 snapshot is after it)
            pre18 = np.full((101, n), -1, dtype=np.int8)
            pre18[ly, lz] = z['log_s_pre']
            # ---- SCREEN kernel rows: memory at the decision = BEFORE this screen
            spre = MAP18_TO_CLIN[z['log_s_pre']]; spost18 = z['log_s_post']
            cst = z['log_cancer_stage'].astype(int)
            bnd = band_of(ly)
            # memory before the screen: tau = ly - previous screen year (or never)
            order = np.lexsort((ly, lz))
            lz_o, ly_o, ol_o = lz[order], ly[order], obs_ol[order]
            same = np.r_[False, lz_o[1:] == lz_o[:-1]] if len(order) else np.zeros(0, bool)
            prev_y_s = np.where(same, np.r_[0, ly_o[:-1]] if len(order) else 0, 0)
            prev_o_s = np.where(same, np.r_[0, ol_o[:-1]] if len(order) else 0, 0)
            tau_k_s = np.where(same, ly_o - prev_y_s, TAU_NEVER)
            tau_k = np.empty(len(lz), np.int64); tau_k[order] = tau_k_s
            prev_o = np.empty(len(lz), np.int64); prev_o[order] = prev_o_s
            memK = mem_index(tau_k, prev_o)
            ok = (spre >= 0) & (bnd >= 0)
            tgt = np.full(len(lz), -1, dtype=np.int64)
            exc = cdet & ok
            tgt[exc] = NO4 * NC + X_D1 + np.clip(cst[exc] - 7, 0, 3)
            exd = comp & ~cdet & ok
            tgt[exd] = NO4 * NC + X_DCOMP
            rest = ok & ~cdet & ~comp
            sp = MAP18_TO_CLIN[spost18]
            good = rest & (sp >= 0)
            tgt[good] = obs_ol[good].astype(np.int64) * NC + sp[good]
            bad = rest & (sp < 0)
            if bad.any():
                ex = MAP18_TO_EXIT[spost18[bad]]
                tgt[bad] = NO4 * NC + np.where(ex >= 0, ex, X_DCOMP)
            keep = tgt >= 0
            idx = ((((sex[lz[keep]] * n_class + cls[lz[keep]]) * N_MEM + memK[keep]) * NB + bnd[keep]) * NC
                   + spre[keep]) * NT_K + tgt[keep]
            Kc += np.bincount(idx, minlength=Kc.size).reshape(Kc.shape)
        else:
            tau = np.full((NY, n), TAU_NEVER, dtype=np.int16)
            ol = np.zeros((NY, n), dtype=np.int8)
            pre18 = None
        mem = mem_index(tau, ol)                                   # (NY, n)
        # ---- WAIT kernel rows (annual windows)
        for iy, y in enumerate(range(Y0, Y1 + 1)):
            qi0 = (y - 1) * 4 + 1
            frm = MAP18_TO_CLIN[qr[qi0]]
            valid = frm >= 0
            if not valid.any():
                continue
            seg = qr[qi0 + 1:qi0 + 5].copy()                         # (4, n) following snapshots
            if pre18 is not None and y + 1 <= 100:
                ov = pre18[y + 1] >= 0                                # screened at y+1: end = pre-colonoscopy state
                seg[3, ov] = pre18[y + 1][ov]
            codes = _TO_CODE[seg]                                    # 0..10 clinical, 11..16 exits
            is_exit = codes >= NC
            any_exit = is_exit.any(axis=0)
            first = np.where(any_exit, is_exit.argmax(axis=0), 3)
            tgt = codes[first, np.arange(n)]                         # exit code or final clinical
            v = valid
            idx = ((((sex[v] * n_class + cls[v]) * N_MEM + mem[iy][v]) * NY + iy) * NC + frm[v]) * NT_W + tgt[v]
            W += np.bincount(idx, minlength=W.size).reshape(W.shape)
        # ---- age-40 occupancy (never-screened persons only)
        if not with_logs:
            s40 = MAP18_TO_CLIN[qr[157]]
            okk = s40 >= 0
            np.add.at(occ40, (sex[okk], cls[okk], s40[okk]), 1)
        # ---- diagnosis outcomes, keyed by the DECISION EPOCH the exit is
        # charged to, with post-diagnosis life-years counted in the model's
        # own convention (number of later decision-time snapshots reached
        # alive), so that survivors and exits accrue life-years identically.
        isD = (qr >= 11) & (qr <= 14)
        hasD = isD.any(axis=0)
        fq = isD.argmax(axis=0)                    # first D snapshot index
        fy, fqq = fq // 4 + 1, fq % 4              # year (1-based) and quarter-start index of that snapshot
        y_dec = np.where(fqq >= 2, fy, fy - 1)     # seen at Q3/Q4 start -> window fy; at Q1/Q2 start -> window fy-1
        if with_logs and cdet.any():
            y_dec[lz[cdet]] = ly[cdet]             # screen-detected: charged to the SCREEN decision year
        dxs = z['dx_stage'].astype(int) - 7
        cd = z['crc_death']
        alive_q2 = qr[1::4] < 15                   # (100, n): alive at Q2-start of year t+1
        csum = np.cumsum(alive_q2[::-1], axis=0)[::-1]   # csum[t] = # alive Q2-start snapshots of years >= t+1
        m = hasD & (y_dec >= 1) & (y_dec <= 100)
        idx_y = np.clip(y_dec[m], 0, 99)           # snapshots of years > y_dec  -> row index y_dec
        lyv = csum[idx_y, np.arange(n)[m]].astype(float)
        np.add.at(dx, (sex[m], y_dec[m], dxs[m], 0), 1.0)
        np.add.at(dx, (sex[m], y_dec[m], dxs[m], 1), cd[m].astype(float))
        np.add.at(dx, (sex[m], y_dec[m], dxs[m], 2), lyv)
        if verbose:
            print(f'  [{ip + 1}/{len(paths)}] n={n_tot:,} ({time.time() - t0:.0f}s)', flush=True)
    return W, Kc, occ40, dx, n_tot


# ----------------------------------------------------------------------
# hierarchical back-off
# ----------------------------------------------------------------------
def _age_pool(C, half):
    """Moving sum over the age axis (axis=3) with +-half window."""
    n = C.shape[3]
    half = min(half, n - 1)
    if half <= 0:
        return C
    out = np.zeros_like(C)
    for k in range(-half, half + 1):
        lo, hi = max(0, -k), min(n, n - k)
        out[:, :, :, lo:hi] += C[:, :, :, lo + k:hi + k]
    return out


def _tau_ol_pools(C):
    """Coarser versions of C along the memory axis (axis=2):
    'tau_keep_ol': pooled over tau groups (post-screen cells) keeping the last finding
    'ol'         : pooled over last finding within each tau group
    'post'       : all post-screen cells pooled (never kept separate)
    'all'        : every memory cell pooled."""
    Ctk = np.zeros_like(C)
    Ctk[:, :, 0] = C[:, :, 0]
    for ol in range(N_OL):
        idx = [1 + (g - 1) * N_OL + ol for g in range(1, N_TAU_G)]
        Ctk[:, :, idx] = C[:, :, idx].sum(axis=2, keepdims=True)
    Col = np.zeros_like(C)
    Col[:, :, 0] = C[:, :, 0]
    for g in range(1, N_TAU_G):
        sl = slice(1 + (g - 1) * N_OL, 1 + g * N_OL)
        Col[:, :, sl] = C[:, :, sl].sum(axis=2, keepdims=True)
    Cp = np.zeros_like(C)
    Cp[:, :, 0] = C[:, :, 0]
    Cp[:, :, 1:] = C[:, :, 1:].sum(axis=2, keepdims=True)
    Ca = np.broadcast_to(C.sum(axis=2, keepdims=True), C.shape)
    return {'tau_keep_ol': Ctk, 'ol': Col, 'post': Cp, 'all': Ca}


def backoff_probs(C, min_row, age_axis=True):
    """C: counts [sex, class, mem, age/band, from, to]. Returns probabilities of
    the same shape, each row estimated from the first level of the back-off
    ladder with >= min_row observations:
      own memory cell with widening age window (0, 1, 2, 4, 8 years)
      -> pooled over tau groups keeping the last finding (age 2, 4, 8)
      -> pooled over last finding keeping tau group (age 2, 4, 8)
      -> all post-screen cells (age 4, 8) -> all memory (age 8)
      -> class pooled -> sex pooled."""
    halves = [0, 1, 2, 4, 8] if age_axis else [0]
    aged = {h: _age_pool(C, h) for h in halves}
    pools = {h: _tau_ol_pools(aged[h]) for h in halves}
    levels = [aged[h] for h in halves]
    mid = [h for h in (2, 4, 8) if h in aged] or [halves[-1]]
    levels += [pools[h]['tau_keep_ol'] for h in mid]
    levels += [pools[h]['ol'] for h in mid]
    levels += [pools[h]['post'] for h in mid[-2:]]
    levels += [pools[mid[-1]]['all']]
    Cc = np.broadcast_to(pools[mid[-1]]['all'].sum(axis=1, keepdims=True), C.shape)
    levels.append(Cc)
    Cs = np.broadcast_to(Cc.sum(axis=0, keepdims=True), C.shape)
    levels.append(Cs)
    P = np.full(C.shape, np.nan)
    filled = np.zeros(C.shape[:-1], dtype=bool)
    level_used = np.full(C.shape[:-1], -1, dtype=np.int8)
    for li, L in enumerate(levels):
        tot = L.sum(axis=-1)
        take = (~filled) & (tot >= min_row)
        if take.any():
            P[take] = L[take] / tot[take][:, None]
            filled |= take
            level_used[take] = li
    left = ~filled
    if left.any():
        tot = levels[-1].sum(axis=-1)
        has = left & (tot > 0)
        P[has] = levels[-1][has] / tot[has][:, None]
    return P, level_used


def finalize(W, Kc):
    Pw, fw = backoff_probs(W, MIN_ROW_W, age_axis=True)
    Pk, fk = backoff_probs(Kc, MIN_ROW_K, age_axis=True)
    # rows never observed anywhere: WAIT -> stay; SCREEN -> cancer detected at own stage / normal
    for idx in zip(*np.where(np.isnan(Pw[..., 0]))):
        Pw[idx] = 0.0; Pw[idx + (idx[-1],)] = 1.0
    for idx in zip(*np.where(np.isnan(Pk[..., 0]))):
        s = idx[-1]
        Pk[idx] = 0.0
        if s >= NC - 4:
            Pk[idx + (NO4 * NC + X_D1 + (s - (NC - 4)),)] = 1.0
        else:
            Pk[idx + (0 * NC + s,)] = 1.0
    Tw = Pw[..., :NC]; Ew = Pw[..., NC:]
    K = Pk[..., :NO4 * NC].reshape(Pk.shape[:-1] + (NO4, NC)); X = Pk[..., NO4 * NC:]
    return Tw, Ew, K, X, fw, fk


def smooth_dx(dx, band=5):
    ny = dx.shape[1]
    Vd = np.full((2, ny, 4), np.nan); Vl = np.full((2, ny, 4), np.nan)
    for sx in range(2):
        for k in range(4):
            for y in range(ny):
                lo, hi = max(0, y - band), min(ny, y + band + 1)
                w = dx[sx, lo:hi, k]; n = w[:, 0].sum()
                if n > 0:
                    Vd[sx, y, k] = w[:, 1].sum() / n; Vl[sx, y, k] = w[:, 2].sum() / n
            for V in (Vd, Vl):
                row = V[sx, :, k]; bad = np.isnan(row)
                if bad.any() and (~bad).any():
                    gi = np.where(~bad)[0]
                    for y in np.where(bad)[0]:
                        row[y] = row[gi[np.argmin(np.abs(gi - y))]]
                elif bad.all():
                    row[:] = 0.0
    return Vd, Vl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nh-runs', default=os.path.join(RUNS, 'nh_quarterly'))
    ap.add_argument('--screen-runs', default=os.path.join(RUNS, 'screen_random_q'),
                    help='comma-separated list of run directories')
    ap.add_argument('--cuts', type=float, nargs='*', default=list(DEFAULT_CLASS_CUTS))
    ap.add_argument('--tag', default=None)
    ap.add_argument('--max-chunks', type=int, default=None)
    a = ap.parse_args()
    cuts = tuple(a.cuts); n_class = len(cuts) + 1
    tag = a.tag or f'c{n_class}'
    thr = risk_thresholds(load_settings_pool(), cuts) if cuts else np.zeros(0)
    nh_paths = sorted(glob.glob(os.path.join(a.nh_runs, '*.npz')))
    sc_paths = sorted(sum((glob.glob(os.path.join(d, '*.npz')) for d in a.screen_runs.split(',')), []))
    if a.max_chunks:
        nh_paths = nh_paths[:a.max_chunks]; sc_paths = sc_paths[:a.max_chunks]
    print(f'classes={n_class} thr={thr}; NH chunks={len(nh_paths)} screen chunks={len(sc_paths)}', flush=True)
    W1, K1, occ40, dx1, n1 = accumulate(nh_paths, thr, n_class, with_logs=False)
    W2, K2, _, dx2, n2 = accumulate(sc_paths, thr, n_class, with_logs=True)
    W = W1 + W2; Kc = K1 + K2; dx = dx1 + dx2
    Tw, Ew, K, X, fw, fk = finalize(W, Kc)
    b0 = occ40.reshape(2, n_class * NC).astype(float); b0 /= b0.sum(axis=1, keepdims=True)
    Vd, Vl = smooth_dx(dx)
    out = os.path.join(RES, f'kernels_{tag}.npz')
    np.savez_compressed(out, Tw=Tw, Ew=Ew, K=K, X=X, b0=b0, thr=thr, cuts=np.array(cuts), n_class=n_class, nc=NC,
                        y0=Y0, y1=Y1, bands=np.array(BANDS), Vexit_death=Vd, Vexit_ly=Vl,
                        n_nh=n1, n_screen=n2, W_rowcounts=W.sum(axis=-1), K_rowcounts=Kc.sum(axis=-1),
                        W_level=fw, K_level=fk)
    print('saved', out)
    # diagnostics
    rc = W.sum(axis=-1)
    print('WAIT rows by memory cell (sex0,class1,age60, from=N..P6):')
    for m in range(N_MEM):
        print(f'  {mem_name(m):18s}', rc[0, 1, m, 60 - Y0, :7])
    iy = 60 - Y0
    for m in (0, mem_index(np.array([0]), np.array([1]))[0], mem_index(np.array([5]), np.array([1]))[0]):
        T = Tw[0, 1, m, iy]
        print(f'male class1 age60 mem={mem_name(m):14s}: N->P {T[0, 1:7].sum():.4f}  P3->P5-6 {T[3, 5:7].sum():.4f}  P4->P5-6 {T[4, 5:7].sum():.4f}')
    print('Vexit_death age60 (male)', Vd[0, 60], ' LY', Vl[0, 60])


if __name__ == '__main__':
    main()
