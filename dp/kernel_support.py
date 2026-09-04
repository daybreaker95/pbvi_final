"""Where do the kernels the headline policy actually uses come from?

For every decision-relevant kernel row -- weighted by the exact occupancy the
headline policy (lambda = 0.001561) places on it -- reports the level of the
back-off ladder at which that row was estimated (own cell with age window
+-0/1/2/4/8, pooled tau groups keeping the finding, pooled findings keeping
the tau group, all post-screen cells, all memory cells, class pooled, sex
pooled), overall and by tau group, together with the own-cell sample sizes of
the open-ended tau >= 13 cells and the tau composition of the person-years
behind them (from `tau_rowcounts`, written by dp.estimate_kernels).

python -m dp.kernel_support
"""
from __future__ import annotations

import json
import os

import numpy as np

from .common import RES, NC, WAIT, SCREEN, SCREEN_OBS
from .kernels import TAU_NEVER, TAU_GROUPS, N_TAU_G, N_OL, mem_index_scalar, tau_group, OL_OF_OBS
from .estimate_screen import band_of
from .model import policy_tree
from .solver import load_policy

LEVEL_NAMES = ['own cell, +-0 y', 'own cell, +-1 y', 'own cell, +-2 y', 'own cell, +-4 y', 'own cell, +-8 y',
               'tau pooled (finding kept), +-2 y', 'tau pooled (finding kept), +-4 y', 'tau pooled (finding kept), +-8 y',
               'finding pooled (tau kept), +-2 y', 'finding pooled (tau kept), +-4 y', 'finding pooled (tau kept), +-8 y',
               'all post-screen cells, +-4 y', 'all post-screen cells, +-8 y', 'all memory cells, +-8 y',
               'class pooled', 'sex pooled']
LEVEL_NAMES_K = [s.replace(' y', ' bands') for s in LEVEL_NAMES]
GROUP_NAMES = ['never', 'tau 0', 'tau 1', 'tau 2', 'tau 3', 'tau 4-5', 'tau 6-8', 'tau 9-12', 'tau 13+']
KERNELS = os.path.join(RES, 'kernels_c6b.npz')
LAM = 0.001561


def occupancy_by_cell(kz, prune=1e-9):
    """Occupancy mass on WAIT rows [sex, class, mem, age, from] and SCREEN rows
    [sex, class, mem, band, pre] under the headline policy (ages 40-80), plus
    WAIT rows of the WAIT-only continuation (81-99)."""
    n_class = int(kz['n_class']); y0 = int(kz['y0'])
    W_occ = np.zeros(kz['W_rowcounts'].shape); K_occ = np.zeros(kz['K_rowcounts'].shape)
    W_occ_post80 = np.zeros(kz['W_rowcounts'].shape)
    for sex in (1, 2):
        sx = sex - 1
        pol = load_policy(os.path.join(RES, 'policies', f'c6bhi_death_lam{LAM:.6g}_sex{sex}.npz'))
        m = pol.model
        _, tree = policy_tree(m, pol.best_action_batch, collect=True, prune=prune)
        leftovers = {}
        for (y, tau, ol), (U, acts) in tree.items():
            iy = y - y0
            mem = mem_index_scalar(tau, ol)
            for a in (WAIT, SCREEN):
                sel = acts == a
                if not sel.any():
                    continue
                tot = U[sel].sum(axis=0)                      # (S,)
                occ = tot.reshape(n_class, NC)
                if a == WAIT:
                    W_occ[sx, :, mem, iy, :] += occ
                else:
                    b = int(band_of(np.array([y]))[0])
                    K_occ[sx, :, mem, b, :] += occ
                    # the natural-history year right after the screen (tau = 0 cell)
                    Ko = m.Ks(y, tau, ol)
                    for o in SCREEN_OBS:
                        post = (tot @ Ko[o]).reshape(n_class, NC)
                        W_occ[sx, :, mem_index_scalar(0, OL_OF_OBS[o]), iy, :] += post
            if y == m.age_max:
                # carry every row (whatever the action) to the boundary
                for a in (WAIT, SCREEN):
                    sel = acts == a
                    if not sel.any() or a not in m.M[(y, tau, ol)]:
                        continue
                    for o, M in m.M[(y, tau, ol)][a].items():
                        nk = m.succ[(y, tau, ol)][a][o]
                        leftovers[nk] = leftovers.get(nk, 0) + (U[sel] @ M).sum(axis=0)
        # WAIT-only continuation 81..99
        nodes = dict(leftovers)
        for y in range(m.age_max + 1, m.life_max + 1):
            nxt = {}
            for (tau, ol), u in nodes.items():
                mem = mem_index_scalar(tau, ol)
                W_occ_post80[sx, :, mem, y - y0, :] += u.reshape(n_class, NC)
                key = (y, tau, ol)
                for o, M in m.M[key][WAIT].items():
                    nk = m.succ[key][WAIT][o]
                    nxt[nk] = nxt.get(nk, 0) + u @ M
            nodes = nxt
    return W_occ, K_occ, W_occ_post80


def level_hist(occ, level, n_levels=16):
    h = np.zeros(n_levels)
    for l in range(n_levels):
        h[l] = occ[level == l].sum()
    return h / max(occ.sum(), 1e-300)


def mem_group_of(mem):
    return 0 if mem == 0 else (mem - 1) // N_OL + 1


def main():
    kz = np.load(KERNELS, allow_pickle=True)
    Wl, Kl = kz['W_level'], kz['K_level']; Wc, Kc = kz['W_rowcounts'], kz['K_rowcounts']
    W_occ, K_occ, W_post = occupancy_by_cell(kz)
    out = dict(kernels=KERNELS, lam=LAM)
    # ---- overall level usage
    out['wait_levels_40_80'] = level_hist(W_occ, Wl).tolist()
    out['wait_levels_81_99'] = level_hist(W_post, Wl).tolist()
    out['screen_levels'] = level_hist(K_occ, Kl).tolist()
    # ---- by tau group (WAIT rows, ages 40-80)
    by_group = {}
    for g in range(N_TAU_G):
        mems = [0] if g == 0 else [1 + (g - 1) * N_OL + ol for ol in range(N_OL)]
        occ = W_occ[:, :, mems]; lev = Wl[:, :, mems]; cnt = Wc[:, :, mems]
        tot = occ.sum()
        h = level_hist(occ, lev)
        own = float(h[:5].sum())
        # occupancy-weighted own-cell person-years and the share of mass on cells below the threshold
        wcnt = float((occ * cnt).sum() / max(tot, 1e-300))
        thin = float(occ[cnt < 150].sum() / max(tot, 1e-300))
        by_group[GROUP_NAMES[g]] = dict(mass_share=float(tot / W_occ.sum()), own_cell_share=own,
                                        levels=h.tolist(), weighted_own_cell_person_years=wcnt,
                                        mass_on_cells_below_150=thin)
    out['wait_by_tau_group'] = by_group
    # ---- SCREEN rows by tau group
    by_group_k = {}
    for g in range(N_TAU_G):
        mems = [0] if g == 0 else [1 + (g - 1) * N_OL + ol for ol in range(N_OL)]
        occ = K_occ[:, :, mems]; lev = Kl[:, :, mems]; cnt = Kc[:, :, mems]
        tot = occ.sum()
        if tot <= 0:
            continue
        h = level_hist(occ, lev)
        by_group_k[GROUP_NAMES[g]] = dict(mass_share=float(tot / K_occ.sum()), own_cell_share=float(h[:5].sum()),
                                          levels=h.tolist(),
                                          weighted_own_cell_colonoscopies=float((occ * cnt).sum() / tot),
                                          mass_on_cells_below_150=float(occ[cnt < 150].sum() / tot))
    out['screen_by_tau_group'] = by_group_k
    # ---- tau >= 13 cells in detail: own-cell counts by finding and age band (class-pooled, sex-pooled)
    g13 = N_TAU_G - 1
    det = {}
    for ol in range(N_OL):
        mem = 1 + (g13 - 1) * N_OL + ol
        occ = W_occ[:, :, mem]; cnt = Wc[:, :, mem]; lev = Wl[:, :, mem]
        tot = occ.sum()
        det[['normal', 'adenoma', 'multi', 'advad'][ol]] = dict(
            mass_share_of_all_wait=float(tot / W_occ.sum()),
            own_cell_person_years_total=int(cnt.sum()),
            own_cell_person_years_ages_53_80=int(cnt[:, :, 13:41].sum()),
            weighted_own_cell_person_years=float((occ * cnt).sum() / max(tot, 1e-300)),
            own_cell_share=float(level_hist(occ, lev)[:5].sum()),
            levels=level_hist(occ, lev).tolist())
    out['tau13_cells'] = det
    # ---- rows filled at no level (never observed) / sex pooled
    out['n_wait_rows'] = int(Wl.size); out['n_wait_rows_sex_pooled'] = int((Wl == 15).sum())
    out['n_wait_rows_unfilled'] = int((Wl < 0).sum())
    out['n_screen_rows'] = int(Kl.size); out['n_screen_rows_sex_pooled'] = int((Kl == 15).sum())
    out['n_screen_rows_unfilled'] = int((Kl < 0).sum())
    # ---- tau composition of the person-years (if a re-estimated kernel file carries it)
    for cand in ('kernels_c6b_re.npz', 'kernels_c6b_tau20.npz'):
        p = os.path.join(RES, cand)
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            if 'tau_rowcounts' in z.files:
                th = z['tau_rowcounts']          # (NY, 62): col 0 never, col k+1 = tau k (61 = 60+)
                tot13 = th[:, 14:].sum()          # tau >= 13
                comp = dict(source=cand, tau_max=int(z['tau_max']),
                            person_years_tau_ge13=int(tot13),
                            share_tau_13_20=float(th[:, 14:22].sum() / max(tot13, 1)),
                            share_tau_gt20=float(th[:, 22:].sum() / max(tot13, 1)),
                            share_tau_gt30=float(th[:, 32:].sum() / max(tot13, 1)),
                            by_age_share_gt20={int(40 + iy): float(th[iy, 22:].sum() / max(th[iy, 14:].sum(), 1))
                                               for iy in range(0, 60, 5)})
                out['tau_composition'] = comp
                break
    with open(os.path.join(RES, 'kernel_support.json'), 'w') as f:
        json.dump(out, f, indent=1)
    # ---- markdown
    L = ['# Kernel support under the headline policy (lambda = 0.001561)', '',
         '## Back-off level of the kernel rows, weighted by policy occupancy', '',
         '| level | WAIT rows, ages 40-80 | WAIT rows, ages 81-99 | SCREEN rows |', '|---|---|---|---|']
    for l in range(16):
        L.append(f"| {LEVEL_NAMES[l]} | {100 * out['wait_levels_40_80'][l]:.1f} % | {100 * out['wait_levels_81_99'][l]:.1f} % | {100 * out['screen_levels'][l]:.1f} % ({LEVEL_NAMES_K[l]}) |")
    L += ['', '## WAIT rows by tau group (ages 40-80)', '',
          '| tau group | share of occupancy | own-cell (+-0..8 y) share | occupancy-weighted own-cell person-years | mass on cells with < 150 own person-years |',
          '|---|---|---|---|---|']
    for g, r in out['wait_by_tau_group'].items():
        L.append(f"| {g} | {100 * r['mass_share']:.1f} % | {100 * r['own_cell_share']:.1f} % | {r['weighted_own_cell_person_years']:,.0f} | {100 * r['mass_on_cells_below_150']:.1f} % |")
    L += ['', '## SCREEN rows by tau group', '',
          '| tau group | share of colonoscopies | own-cell (+-0..8 bands) share | occupancy-weighted own-cell colonoscopies | mass on cells with < 150 own colonoscopies |',
          '|---|---|---|---|---|']
    for g, r in out['screen_by_tau_group'].items():
        L.append(f"| {g} | {100 * r['mass_share']:.1f} % | {100 * r['own_cell_share']:.1f} % | {r['weighted_own_cell_colonoscopies']:,.0f} | {100 * r['mass_on_cells_below_150']:.1f} % |")
    L += ['', '## The open-ended tau >= 13 cells', '',
          '| last finding | share of all WAIT occupancy | own-cell person-years (all ages) | own-cell person-years (ages 53-80) | occupancy-weighted own-cell person-years | own-cell share |',
          '|---|---|---|---|---|---|']
    for k, r in out['tau13_cells'].items():
        L.append(f"| {k} | {100 * r['mass_share_of_all_wait']:.2f} % | {r['own_cell_person_years_total']:,} | {r['own_cell_person_years_ages_53_80']:,} | {r['weighted_own_cell_person_years']:,.0f} | {100 * r['own_cell_share']:.1f} % |")
    L += ['', f"Rows estimated at the sex-pooled level: WAIT {out['n_wait_rows_sex_pooled']} of {out['n_wait_rows']}, SCREEN {out['n_screen_rows_sex_pooled']} of {out['n_screen_rows']}; "
          f"rows never observed at any level (defaulted): WAIT {out['n_wait_rows_unfilled']}, SCREEN {out['n_screen_rows_unfilled']}."]
    if 'tau_composition' in out:
        c = out['tau_composition']
        L += ['', f"## tau composition of the person-years in the tau >= 13 group ({c['source']})", '',
              f"- {c['person_years_tau_ge13']:,} person-years with tau >= 13; {100 * c['share_tau_13_20']:.1f} % at tau 13-20 (inside the randomised design's interval support), "
              f"{100 * c['share_tau_gt20']:.1f} % at tau > 20, {100 * c['share_tau_gt30']:.1f} % at tau > 30.",
              '- share at tau > 20 by decision age: ' + ', '.join(f'{a}: {100 * v:.0f} %' for a, v in c['by_age_share_gt20'].items())]
    md = os.path.join(RES, 'kernel_support.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('saved', md)


if __name__ == '__main__':
    main()
