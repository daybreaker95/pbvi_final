"""Estimate the natural-history (WAIT) kernel of the reduced POMDP from a
quarterly-recorded no-screening cohort simulated in the REAL engine
(dp.run_cohorts -> results/dp/runs/nh_quarterly/*.npz).

Timing convention
-----------------
The engine calls the policy hook at quarter q==1 AFTER that quarter's
natural-history events, i.e. the decision-time state equals the state at
the START of quarter 2 of age y (quarter index qi = (y-1)*4 + 1). One
decision step therefore spans qi -> qi+4. The annual kernel is composed from
the four quarter-to-quarter transition matrices on that window, with the
exits (diagnosis at stage k, other-cause death) made absorbing.

Outputs (results/dp/nh_kernel_<tag>.npz)
  Tw[sex, class, y, 7, 7]     alive&undiagnosed -> alive&undiagnosed (decision age y -> y+1)
  Ew[sex, class, y, 7, 6]     exits within the year (D1..D4, DeadOther, DeadComp)
  b0[sex, class*7]            joint (class, clinical) distribution at the age-40 decision time
  Vexit_death[sex, y, 4]      P(CRC death | diagnosed at stage k during age y)
  Vexit_ly[sex, y, 4]         E[life-years after diagnosis | stage k, age y]
  plus metadata (class thresholds, counts)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from .common import (
    RES, RUNS, NC, NX, MAP18_TO_CLIN, MAP18_TO_EXIT, E_D, DEFAULT_CLASS_CUTS,
    risk_thresholds, risk_class_of, load_settings_pool, X_DOTH,
)

NQ = 400
AGES = np.arange(1, 100)          # decision ages for which an annual kernel is built (1..99)


def quarter_counts(paths, thr, n_class, verbose=True):
    """Accumulate quarter-to-quarter transition counts.
    counts[sex, class, qi, from(7), to(13)] with to = 0..6 clinical, 7..12 exits."""
    counts = np.zeros((2, n_class, NQ - 1, NC, NC + NX), dtype=np.int64)
    occ40 = np.zeros((2, n_class, NC), dtype=np.int64)     # decision-time occupancy at age 40
    # diagnosis outcomes: rows (sex, age_y, stage_k) -> [n, n_crc_death, sum_ly]
    dx = np.zeros((2, 101, 4, 3), dtype=np.float64)
    to_code = np.where(MAP18_TO_CLIN >= 0, MAP18_TO_CLIN, NC + MAP18_TO_EXIT).astype(np.int16)   # 18 -> 0..12
    n_tot = 0
    t0 = time.time()
    for ip, p in enumerate(paths):
        z = np.load(p, allow_pickle=True)
        qr = z['qr']                              # (400, n) int8
        risk = z['risk'].astype(float)
        sex = z['sex'].astype(int) - 1            # 0 male, 1 female
        cls = risk_class_of(risk, thr).astype(int)
        n = qr.shape[1]
        n_tot += n
        grp = sex * n_class + cls                 # (n,)
        frm = MAP18_TO_CLIN[qr[:-1]]              # (399, n), -1 if not alive&undiagnosed
        to = to_code[qr[1:]]                      # (399, n)
        valid = frm >= 0
        qi = np.broadcast_to(np.arange(NQ - 1)[:, None], frm.shape)
        g = np.broadcast_to(grp[None, :], frm.shape)
        idx = ((g[valid] * (NQ - 1) + qi[valid]) * NC + frm[valid]) * (NC + NX) + to[valid]
        bc = np.bincount(idx, minlength=counts.size)
        counts += bc.reshape(counts.shape)
        # occupancy at the age-40 decision time (qi = 39*4+1 = 157)
        s40 = MAP18_TO_CLIN[qr[157]]
        ok = s40 >= 0
        np.add.at(occ40, (sex[ok], cls[ok], s40[ok]), 1)
        # diagnosis outcomes from the per-person arrays
        dxy = z['dx_year'].astype(int)            # 1-based year of first diagnosis, 0 = never
        dxs = z['dx_stage'].astype(int) - 7       # 0..3
        dy = z['death_year'].astype(int)          # 101 = survivor
        cd = z['crc_death']
        m = dxy > 0
        ly = np.maximum(dy[m] - 1 - dxy[m], 0).astype(float)
        np.add.at(dx, (sex[m], dxy[m], dxs[m], 0), 1.0)
        np.add.at(dx, (sex[m], dxy[m], dxs[m], 1), cd[m].astype(float))
        np.add.at(dx, (sex[m], dxy[m], dxs[m], 2), ly)
        if verbose:
            print(f'  [{ip + 1}/{len(paths)}] n={n_tot:,} ({time.time() - t0:.0f}s)', flush=True)
    return counts, occ40, dx, n_tot


def normalise_rows(c, eps=0.0):
    """Row-normalise counts; rows with no data are returned as NaN."""
    tot = c.sum(axis=-1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = (c + eps) / (tot + eps * c.shape[-1])
    P[np.broadcast_to(tot == 0, P.shape)] = np.nan
    return P


def compose_annual(Pq):
    """Pq: (NQ-1, 7, 13) quarter transition probs (rows may be NaN where no
    data). Returns Tw (99, 7, 7) and Ew (99, 7, 6) for decision ages 1..99
    (decision age y uses quarters (y-1)*4+1 .. (y-1)*4+4)."""
    ny = len(AGES)
    Tw = np.zeros((ny, NC, NC))
    Ew = np.zeros((ny, NC, NX))
    # fill NaN rows by carrying the nearest valid quarter forward/backward
    Pq = Pq.copy()
    for s in range(NC):
        rows = Pq[:, s, :]
        bad = np.isnan(rows[:, 0])
        if bad.all():
            rows[:] = 0.0
            rows[:, s] = 1.0          # no data at all: stay
            continue
        good_idx = np.where(~bad)[0]
        for qi in np.where(bad)[0]:
            j = good_idx[np.argmin(np.abs(good_idx - qi))]
            rows[qi] = rows[j]
        Pq[:, s, :] = rows
    # absorbing augmented matrices
    for iy, y in enumerate(AGES):
        A = np.eye(NC + NX)
        for k in range(4):
            qi = (y - 1) * 4 + 1 + k
            if qi >= NQ - 1:
                break
            Q = np.eye(NC + NX)
            Q[:NC, :] = Pq[qi]
            A = A @ Q
        Tw[iy] = A[:NC, :NC]
        Ew[iy] = A[:NC, NC:]
    return Tw, Ew


def smooth_dx(dx, band=5):
    """dx[sex, y, k, (n, n_crc, sum_ly)] -> Vdeath[sex, y, k], Vly[sex, y, k]
    using a moving +-band-year window (in age) for stability."""
    ny = dx.shape[1]
    Vd = np.zeros((2, ny, 4)); Vl = np.zeros((2, ny, 4))
    for sx in range(2):
        for k in range(4):
            for y in range(ny):
                lo, hi = max(0, y - band), min(ny, y + band + 1)
                w = dx[sx, lo:hi, k]
                n = w[:, 0].sum()
                if n > 0:
                    Vd[sx, y, k] = w[:, 1].sum() / n
                    Vl[sx, y, k] = w[:, 2].sum() / n
                else:
                    Vd[sx, y, k] = np.nan; Vl[sx, y, k] = np.nan
            # fill any remaining NaN with nearest
            for V in (Vd, Vl):
                row = V[sx, :, k]
                bad = np.isnan(row)
                if bad.any() and (~bad).any():
                    gi = np.where(~bad)[0]
                    for y in np.where(bad)[0]:
                        row[y] = row[gi[np.argmin(np.abs(gi - y))]]
                elif bad.all():
                    row[:] = 0.0
    return Vd, Vl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(RUNS, 'nh_quarterly'))
    ap.add_argument('--cuts', type=float, nargs='*', default=list(DEFAULT_CLASS_CUTS),
                    help='quantile cuts of individual_risk defining classes (empty = 1 class)')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--max-chunks', type=int, default=None)
    a = ap.parse_args()
    cuts = tuple(a.cuts)
    n_class = len(cuts) + 1
    tag = a.tag or f'c{n_class}'
    paths = sorted(glob.glob(os.path.join(a.runs, '*.npz')))
    if a.max_chunks:
        paths = paths[:a.max_chunks]
    pool = load_settings_pool()
    thr = risk_thresholds(pool, cuts) if cuts else np.zeros(0)
    print(f'{len(paths)} chunks, classes={n_class}, thresholds={thr}', flush=True)
    counts, occ40, dx, n_tot = quarter_counts(paths, thr, n_class)

    Tw = np.zeros((2, n_class, len(AGES), NC, NC))
    Ew = np.zeros((2, n_class, len(AGES), NC, NX))
    for sx in range(2):
        for c in range(n_class):
            Pq = normalise_rows(counts[sx, c].astype(float))
            Tw[sx, c], Ew[sx, c] = compose_annual(Pq)
    b0 = occ40.reshape(2, n_class * NC).astype(float)
    b0 /= b0.sum(axis=1, keepdims=True)
    Vdeath, Vly = smooth_dx(dx)
    frac_class = occ40.sum(axis=2) / occ40.sum(axis=(1, 2), keepdims=True)[:, :, 0]

    out = os.path.join(RES, f'nh_kernel_{tag}.npz')
    np.savez_compressed(out, Tw=Tw, Ew=Ew, b0=b0, ages=AGES, thr=thr, cuts=np.array(cuts),
                        n_class=n_class, n_persons=n_tot, counts_per_class=occ40.sum(axis=2),
                        frac_class_at40=frac_class, Vexit_death=Vdeath, Vexit_ly=Vly,
                        dx_raw=dx, occ40=occ40)
    print('saved', out)
    # quick diagnostics
    for sx, nm in enumerate(['male', 'female']):
        for c in range(n_class):
            iy = list(AGES).index(60)
            T = Tw[sx, c, iy]
            print(f'{nm} class{c} age60: N->P1 {T[0, 1]:.4f}  P4->P5 {T[4, 5]:.4f}  P6->U {T[6, 7:11].sum():.4f}  '
                  f'N->deadOther {Ew[sx, c, iy, 0, X_DOTH]:.4f}  b0 class share {frac_class[sx, c]:.3f}')
    print('Vexit_death age60 (male) by stage', Vdeath[0, 60])
    print('Vexit_ly    age60 (male) by stage', Vly[0, 60])


if __name__ == '__main__':
    main()
