"""Estimate the empirical colonoscopy (SCREEN) kernel from the randomised-
schedule cohort (results/dp/runs/screen_random/*.npz, decision logs).

K[sex, class, band, s_pre(11), o(4), s_post(11)]  -- probability that a colonoscopy
    performed on a person in clinical state s_pre yields observation o in
    {normal, adenoma, multi, advad} AND leaves the person in state s_post
X[sex, class, band, s_pre(11), exit(6)]         -- probability of an exit at the
    colonoscopy itself: cancer detected at stage k (D1..D4) or complication death

Rows sum to 1 over (o, s_post) + exits. Sparse cells are pooled
hierarchically: (sex,class,band) -> (class,band) -> (band) -> all.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .common import (
    RES, RUNS, NC, NX, MAP18_TO_CLIN, MAP18_TO_EXIT, DEFAULT_CLASS_CUTS, risk_thresholds,
    risk_class_of, load_settings_pool, X_D1, X_DCOMP, O_NORMAL, O_ADENOMA, O_MULTI, O_ADVAD,
    CLIN_NAMES,
)

BANDS = [(40, 44), (45, 49), (50, 54), (55, 59), (60, 64), (65, 69), (70, 74), (75, 80)]
NB = len(BANDS)
NO3 = 4                      # non-exit observations: normal / adenoma / multi / advad
MIN_CELL = 300               # minimum rows before a cell stands on its own


def band_of(y):
    y = np.asarray(y)
    b = np.full(y.shape, -1, dtype=np.int8)
    for i, (lo, hi) in enumerate(BANDS):
        b[(y >= lo) & (y <= hi)] = i
    return b


def collect(paths, thr, n_class, verbose=True):
    # counts[sex, class, band, s_pre, target] where target = o*7 + s_post (21) or 21+exit (6)
    NT = NO3 * NC + NX
    counts = np.zeros((2, n_class, NB, NC, NT), dtype=np.int64)
    n_rows = 0
    for ip, p in enumerate(paths):
        z = np.load(p, allow_pickle=True)
        risk = z['risk'].astype(float); sex = z['sex'].astype(int) - 1
        cls = risk_class_of(risk, thr).astype(int)
        lz = z['log_z'].astype(int); ly = z['log_y'].astype(int)
        spre = MAP18_TO_CLIN[z['log_s_pre']]
        spost18 = z['log_s_post']
        er = z['log_early_removed']; ar = z['log_adv_removed']
        cdet = z['log_cancer_detected']; cst = z['log_cancer_stage'].astype(int)
        comp = z['log_comp_death']
        bnd = band_of(ly)
        ok = (spre >= 0) & (bnd >= 0)
        # target code
        tgt = np.full(len(lz), -1, dtype=np.int64)
        # exits first
        ex_c = cdet & ok
        tgt[ex_c] = NO3 * NC + X_D1 + np.clip(cst[ex_c] - 7, 0, 3)
        ex_d = comp & ~cdet & ok
        tgt[ex_d] = NO3 * NC + X_DCOMP
        rest = ok & ~cdet & ~comp
        o = np.where(ar > 0, O_ADVAD, np.where(er >= 3, O_MULTI, np.where(er > 0, O_ADENOMA, O_NORMAL))) - O_NORMAL   # 0..3
        sp = MAP18_TO_CLIN[spost18]
        good = rest & (sp >= 0)
        tgt[good] = o[good] * NC + sp[good]
        # a non-exit row whose post-state is not alive&undiagnosed cannot happen
        # (engine only kills via complication, which we route to DCOMP) -- guard:
        bad = rest & (sp < 0)
        if bad.any():
            ex = MAP18_TO_EXIT[spost18[bad]]
            tgt[bad] = NO3 * NC + np.where(ex >= 0, ex, X_DCOMP)
        keep = tgt >= 0
        idx = (((sex[lz[keep]] * n_class + cls[lz[keep]]) * NB + bnd[keep]) * NC + spre[keep]) * NT + tgt[keep]
        counts += np.bincount(idx, minlength=counts.size).reshape(counts.shape)
        n_rows += int(keep.sum())
        if verbose:
            print(f'  [{ip + 1}/{len(paths)}] colonoscopies={n_rows:,}', flush=True)
    return counts


def pooled_probs(counts):
    """Hierarchical pooling -> probabilities P[sex, class, band, s_pre, target]."""
    n_sex, n_class, nb, nc, nt = counts.shape
    P = np.zeros(counts.shape, float)
    c_cb = counts.sum(axis=0)                 # (class, band, s, t)
    c_b = c_cb.sum(axis=0)                    # (band, s, t)
    c_all = c_b.sum(axis=0)                   # (s, t)
    for sx in range(n_sex):
        for c in range(n_class):
            for b in range(nb):
                for s in range(nc):
                    for cell in (counts[sx, c, b, s], c_cb[c, b, s], c_b[b, s], c_all[s]):
                        if cell.sum() >= MIN_CELL:
                            P[sx, c, b, s] = cell / cell.sum()
                            break
                    else:
                        cell = c_all[s]
                        if cell.sum() > 0:
                            P[sx, c, b, s] = cell / cell.sum()
                        else:   # never observed (e.g. U4 pre-state): assume detected at that stage
                            P[sx, c, b, s, NO3 * NC + X_D1 + max(0, s - 7)] = 1.0
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(RUNS, 'screen_random'))
    ap.add_argument('--cuts', type=float, nargs='*', default=list(DEFAULT_CLASS_CUTS))
    ap.add_argument('--tag', default=None)
    ap.add_argument('--max-chunks', type=int, default=None)
    a = ap.parse_args()
    cuts = tuple(a.cuts); n_class = len(cuts) + 1
    tag = a.tag or f'c{n_class}'
    paths = sorted(glob.glob(os.path.join(a.runs, '*.npz')))
    if a.max_chunks:
        paths = paths[:a.max_chunks]
    thr = risk_thresholds(load_settings_pool(), cuts) if cuts else np.zeros(0)
    counts = collect(paths, thr, n_class)
    P = pooled_probs(counts)
    K = P[..., :NO3 * NC].reshape(2, n_class, NB, NC, NO3, NC)
    X = P[..., NO3 * NC:]
    out = os.path.join(RES, f'screen_kernel_{tag}.npz')
    np.savez_compressed(out, K=K, X=X, counts=counts, bands=np.array(BANDS), thr=thr, cuts=np.array(cuts), n_class=n_class)
    print('saved', out)
    # diagnostics: pooled over everything
    tot = counts.sum(axis=(0, 1, 2))
    Pt = tot / np.maximum(tot.sum(axis=1, keepdims=True), 1)
    for s in range(NC):
        po = Pt[s, :NO3 * NC].reshape(NO3, NC)
        post_found = po[1:].sum(axis=0); post_found = post_found / max(post_found.sum(), 1e-12)
        print(f'pre={CLIN_NAMES[s]:3s} n={tot[s].sum():8d}  P(normal)={po[0].sum():.3f} P(adenoma)={po[1].sum():.3f} '
              f'P(multi)={po[2].sum():.3f} P(advad)={po[3].sum():.3f} P(cancer found)={Pt[s, NO3 * NC:NO3 * NC + 4].sum():.3f} '
              f'P(compdeath)={Pt[s, NO3 * NC + X_DCOMP]:.5f} | post|found: ' + ' '.join(f'{CLIN_NAMES[j]}={post_found[j]:.2f}' for j in range(NC) if post_found[j] > 0.005))


if __name__ == '__main__':
    main()
