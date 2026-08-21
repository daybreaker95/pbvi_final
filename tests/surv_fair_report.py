"""surv_fair_report.py
======================
Reads everything tests/surv_fair_compare.py wrote into results/survfair/ and
answers the comparison question: is the PBVI schedule more efficient than a
fixed 10-year or 5-year schedule -- under surveillance OFF (primary) and ON
(sensitivity), overall and split by disclosed risk class.

"More efficient" can mean three things, and only the third needs no
trade-off weight, so that is the one worth leading with if it holds:

  MATCHED BUDGET  at the comparator's colonoscopies per person, what CRC
                  mortality does the policy reach? (read off the policy's
                  own lambda curve)
  MATCHED EFFECT  at the comparator's CRC mortality, how many colonoscopies
                  does the policy need?
  DOMINANCE       is there a lambda where the policy is better on BOTH axes
                  at once? No interpolation, no weighting.

Every arm is run at several seeds, so the error bars here are EMPIRICAL
(spread across independent cohorts), not binomial approximations -- with
~900 CRC deaths per 100k at n=2e5 the binomial SE alone is ~21 per 100k,
which is the whole reason the lambda curve is read as a frontier rather
than point by point. The binomial SE is printed next to the empirical one
as a sanity check that the two agree.

Run: python tests/surv_fair_report.py
"""
import os
import sys
import json
import glob
import math
from collections import defaultdict

PBVI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(PBVI_ROOT, 'results', 'survfair')
FIGDIR = os.path.join(PBVI_ROOT, 'paper', 'figures')

import numpy as np

METRICS = ['avg_colonoscopies_per_person', 'crc_death_per_100k', 'incidence_per_100k',
           'screen_detected_per_100k', 'symptom_detected_per_100k',
           'surveillance_detected_per_100k', 'life_years', 'life_years_disc3pct']
TOP_ONLY = ['cost_per_person_usd', 'cost_per_person_disc3pct_usd',
            'screening_colo_total', 'followup_colo_total', 'symptom_colo_total']
RISKS = ['high_risk', 'low_risk']


def load():
    """(surveillance, arm, lam) -> seed-averaged metrics with empirical SEM."""
    files = [json.load(open(f)) for f in glob.glob(os.path.join(RES, '*.json'))]
    # Keep one cohort size only. A stray run at another n -- an abandoned pilot,
    # a job left over from an earlier launch -- would otherwise be averaged in
    # with a different Monte-Carlo variance and a different weight, silently.
    sizes = defaultdict(int)
    for d in files:
        sizes[d['n']] += 1
    n_keep = max(sizes, key=lambda k: sizes[k])
    dropped = [d for d in files if d['n'] != n_keep]
    if dropped:
        print(f'ignoring {len(dropped)} run(s) at n != {n_keep:,}: '
              + ', '.join(sorted({f"{d['arm']}@n={d['n']}" for d in dropped})))

    raw = defaultdict(list)
    for d in files:
        if d['n'] != n_keep:
            continue
        key = ('surv' if d['surveillance'] else 'nosurv', d['arm'],
               round(d['lam'], 6) if d['arm'] == 'policy' else None)
        raw[key].append(d)

    agg = {}
    for key, ds in raw.items():
        k = len(ds)
        e = {'n_seeds': k, 'n': ds[0]['n'], 'n_total': ds[0]['n'] * k,
             'seeds': sorted(d['seed'] for d in ds), 'by_risk': {}}
        for m in METRICS + TOP_ONLY:
            v = np.array([d[m] for d in ds], float)
            e[m] = float(v.mean())
            # ddof=1 across independent cohorts; k=1 leaves it undefined, and
            # nan is the honest value there rather than a silent 0.
            e[m + '_sem'] = float(v.std(ddof=1) / math.sqrt(k)) if k > 1 else float('nan')
        for r in RISKS:
            er = {'n': sum(d['by_risk'][r]['n'] for d in ds) / k}
            for m in METRICS:
                v = np.array([d['by_risk'][r][m] for d in ds], float)
                er[m] = float(v.mean())
                er[m + '_sem'] = float(v.std(ddof=1) / math.sqrt(k)) if k > 1 else float('nan')
            e['by_risk'][r] = er
        agg[key] = e
    return agg


def binom_se(rate_per_100k, n):
    p = rate_per_100k / 100_000.0
    return math.sqrt(max(p * (1 - p), 0.0) / n) * 100_000.0


# --------------------------------------------------------------- frontier
def policy_curve(agg, surv, metric='crc_death_per_100k', risk=None):
    pts = []
    for (sv, arm, lam), e in agg.items():
        if sv != surv or arm != 'policy':
            continue
        src = e['by_risk'][risk] if risk else e
        pts.append((src['avg_colonoscopies_per_person'], src[metric], lam))
    pts.sort()
    return (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]),
            [p[2] for p in pts])


def frontier(x, y, maximize=False):
    """Monotone envelope: at each volume, the best outcome achieved at that
    volume or any smaller one. Strips the Monte-Carlo dips that make a raw
    lambda curve non-monotone without inventing points nobody simulated."""
    o = np.argsort(x)
    xs, ys = np.asarray(x, float)[o], np.asarray(y, float)[o]
    return xs, (np.maximum.accumulate(ys) if maximize else np.minimum.accumulate(ys))


def interp_at(x, y, x0, maximize=False):
    xs, ys = frontier(x, y, maximize)
    if x0 < xs[0] or x0 > xs[-1]:
        return None
    return float(np.interp(x0, xs, ys))


def volume_for(x, y, y0, maximize=False):
    """Smallest colonoscopy volume at which the frontier reaches outcome y0."""
    xs, ys = frontier(x, y, maximize)
    hit = np.nonzero(ys >= y0)[0] if maximize else np.nonzero(ys <= y0)[0]
    if len(hit) == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return float(xs[0])
    x1, x2, y1, y2 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return float(x2 if y2 == y1 else x1 + (y0 - y1) * (x2 - x1) / (y2 - y1))


# --------------------------------------------------------------- reporting
HEAD = (f"{'arm':<22} {'colo/pp':>15} {'CRC death/100k':>19} "
        f"{'incid/100k':>11} {'LY':>8} {'LY(3%)':>8}")


def row(name, e, risk=None):
    src = e['by_risk'][risk] if risk else e
    n_eff = src.get('n', e['n']) * e['n_seeds']
    bse = binom_se(src['crc_death_per_100k'], n_eff)
    return (f"{name:<22} "
            f"{src['avg_colonoscopies_per_person']:>8.3f}"
            f"+-{src['avg_colonoscopies_per_person_sem']:<5.3f} "
            f"{src['crc_death_per_100k']:>9.1f}+-{src['crc_death_per_100k_sem']:<5.1f}"
            f"[{bse:>4.1f}] "
            f"{src['incidence_per_100k']:>11.1f} "
            f"{src['life_years']:>8.4f} {src['life_years_disc3pct']:>8.4f}")


def arms_in(agg, surv):
    lams = sorted(l for (sv, a, l) in agg if sv == surv and a == 'policy')
    return ([('no_screen', None), ('q10y', None), ('q5y', None)]
            + [('policy', l) for l in lams])


def section(agg, surv, out):
    out.append('\n' + '=' * 108)
    out.append('SENSITIVITY -- both surveillance streams ON' if surv == 'surv'
               else 'PRIMARY -- both surveillance streams OFF')
    out.append('=' * 108)
    out.append('(+- is the empirical SEM across seeds; [..] is the binomial SE '
               'on the pooled cohort, for comparison)')

    for risk in (None, 'high_risk', 'low_risk'):
        out.append('\n-- ' + {None: 'WHOLE COHORT',
                              'high_risk': 'HIGH RISK (top 20% of individual_risk)',
                              'low_risk': 'LOW RISK (bottom 80%)'}[risk] + ' --')
        out.append(HEAD)
        for arm, lam in arms_in(agg, surv):
            e = agg.get((surv, arm, lam))
            if e:
                out.append(row(arm if lam is None else f'policy lam={lam:+.4f}', e, risk))

    # -------------------------------------------------- efficiency questions
    out.append('\n-- EFFICIENCY vs THE FIXED SCHEDULES (whole cohort) --')
    x, y, lams = policy_curve(agg, surv)
    xl, yl, _ = policy_curve(agg, surv, 'life_years_disc3pct')
    ns = agg.get((surv, 'no_screen', None))
    for arm in ('q10y', 'q5y'):
        e = agg.get((surv, arm, None))
        if not e:
            continue
        cx, cy = e['avg_colonoscopies_per_person'], e['crc_death_per_100k']
        csem = e['crc_death_per_100k_sem']
        out.append(f'\n  {arm}: {cx:.3f} colonoscopies/person, '
                   f'{cy:.1f}+-{csem:.1f} CRC deaths/100k')

        v = interp_at(x, y, cx)
        out.append(f'    matched budget : policy reaches {v:.1f} deaths/100k at the same '
                   f'{cx:.3f} colo/pp  ({cy - v:+.1f}/100k, {(cy - v) / cy * 100:+.1f}%)'
                   if v is not None else
                   f'    matched budget : policy curve does not span {cx:.3f} colo/pp '
                   f'(covers {x.min():.3f}-{x.max():.3f})')

        w = volume_for(x, y, cy)
        out.append(f'    matched effect : policy needs {w:.3f} colo/pp for the same '
                   f'{cy:.1f} deaths/100k  ({w - cx:+.3f}, {(w - cx) / cx * 100:+.1f}%)'
                   if w is not None else
                   f'    matched effect : policy frontier never reaches {cy:.1f} deaths/100k')

        dom = [(lam, xi, yi) for xi, yi, lam in zip(x, y, lams) if xi <= cx and yi <= cy]
        if dom:
            b = min(dom, key=lambda t: t[2])
            out.append(f'    dominance      : {len(dom)} lambda(s) better on BOTH axes; '
                       f'best lam={b[0]:+.4f} -> {b[1]:.3f} colo/pp, {b[2]:.1f} deaths/100k '
                       f'({b[1] - cx:+.3f} colo, {b[2] - cy:+.1f} deaths)')
        else:
            out.append('    dominance      : no lambda is better on both axes')

        vly = interp_at(xl, yl, cx, maximize=True)
        if vly is not None:
            out.append(f'    disc. LY       : policy {vly:.4f} vs {arm} '
                       f'{e["life_years_disc3pct"]:.4f} at matched budget '
                       f'({vly - e["life_years_disc3pct"]:+.4f} LY)')

        if ns:
            av = ns['crc_death_per_100k'] - cy
            out.append(f'    per-colo yield : {arm} averts {av:.0f} deaths/100k with '
                       f'{cx:.3f} colo/pp = {av / cx:.1f} per colonoscopy-unit')

    if ns:
        out.append('')
        for xi, yi, lam in zip(x, y, lams):
            av = ns['crc_death_per_100k'] - yi
            out.append(f'    policy lam={lam:+.4f}: averts {av:.0f} deaths/100k with '
                       f'{xi:.3f} colo/pp = {av / xi:.1f} per colonoscopy-unit')

    # -------------------------------------------------- risk targeting
    out.append('\n-- WHERE EACH ARM SPENDS ITS COLONOSCOPIES --')
    out.append(f"{'arm':<22} {'high colo/pp':>13} {'low colo/pp':>12} {'ratio':>7} "
               f"{'high death':>11} {'low death':>10} {'high LY(3%)':>12} {'low LY(3%)':>11}")
    for arm, lam in arms_in(agg, surv):
        e = agg.get((surv, arm, lam))
        if not e:
            continue
        hi, lo = e['by_risk']['high_risk'], e['by_risk']['low_risk']
        nm = arm if lam is None else f'policy lam={lam:+.4f}'
        r = hi['avg_colonoscopies_per_person'] / max(lo['avg_colonoscopies_per_person'], 1e-9)
        out.append(f"{nm:<22} {hi['avg_colonoscopies_per_person']:>13.3f} "
                   f"{lo['avg_colonoscopies_per_person']:>12.3f} {r:>7.2f} "
                   f"{hi['crc_death_per_100k']:>11.1f} {lo['crc_death_per_100k']:>10.1f} "
                   f"{hi['life_years_disc3pct']:>12.4f} {lo['life_years_disc3pct']:>11.4f}")

    # -------------------------------------------------- colonoscopy sources
    out.append('\n-- COLONOSCOPY VOLUME BY SOURCE (per person) --')
    out.append(f"{'arm':<22} {'screening':>10} {'follow-up':>10} {'symptom':>9} "
               f"{'total':>8} {'surv-detected/100k':>19}")
    for arm, lam in arms_in(agg, surv):
        e = agg.get((surv, arm, lam))
        if not e:
            continue
        nm = arm if lam is None else f'policy lam={lam:+.4f}'
        n = e['n']
        out.append(f"{nm:<22} {e['screening_colo_total']/n:>10.3f} "
                   f"{e['followup_colo_total']/n:>10.3f} {e['symptom_colo_total']/n:>9.3f} "
                   f"{e['avg_colonoscopies_per_person']:>8.3f} "
                   f"{e['surveillance_detected_per_100k']:>19.1f}")


def compare_primary_vs_sensitivity(agg, out):
    if not any(k[0] == 'surv' for k in agg) or not any(k[0] == 'nosurv' for k in agg):
        return
    out.append('\n' + '=' * 108)
    out.append('DOES THE CONCLUSION SURVIVE TURNING SURVEILLANCE ON?')
    out.append('=' * 108)
    for arm in ('q10y', 'q5y'):
        out.append(f'\n  vs {arm}')
        for surv, tag in (('nosurv', 'primary   (surveillance off)'),
                          ('surv', 'sensitivity (surveillance on)')):
            e = agg.get((surv, arm, None))
            if not e:
                continue
            x, y, lams = policy_curve(agg, surv)
            cx, cy = e['avg_colonoscopies_per_person'], e['crc_death_per_100k']
            v = interp_at(x, y, cx)
            w = volume_for(x, y, cy)
            dom = sum(1 for xi, yi in zip(x, y) if xi <= cx and yi <= cy)
            out.append(f'    {tag:<30} matched-budget deaths '
                       f'{("%.1f" % v) if v is not None else "n/a":>7} vs {cy:.1f}   '
                       f'matched-effect colo '
                       f'{("%.3f" % w) if w is not None else "n/a":>6} vs {cx:.3f}   '
                       f'dominating lambdas {dom}')


def make_figure(agg):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable, skipping figure')
        return
    panels = [s for s in ('nosurv', 'surv') if any(k[0] == s for k in agg)]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 5.4), squeeze=False)
    titles = {'nosurv': 'Primary: surveillance OFF',
              'surv': 'Sensitivity: surveillance ON'}
    for ax, surv in zip(axes[0], panels):
        x, y, lams = policy_curve(agg, surv)
        err = [agg[(surv, 'policy', l)]['crc_death_per_100k_sem'] for l in lams]
        ax.errorbar(x, y, yerr=err, fmt='o-', color='#1f77b4', ms=4, lw=1.3,
                    capsize=2, label='PBVI policy (lambda sweep)')
        fx, fy = frontier(x, y)
        ax.plot(fx, fy, color='#1f77b4', lw=1.0, ls='--', alpha=0.55, label='policy frontier')
        for arm, c, m in (('no_screen', '#7f7f7f', 's'), ('q10y', '#d62728', '^'),
                          ('q5y', '#2ca02c', 'v')):
            e = agg.get((surv, arm, None))
            if not e:
                continue
            ax.errorbar([e['avg_colonoscopies_per_person']], [e['crc_death_per_100k']],
                        yerr=[e['crc_death_per_100k_sem']], fmt=m, color=c, ms=9,
                        capsize=3, label=arm)
            ax.axvline(e['avg_colonoscopies_per_person'], color=c, lw=0.7, ls=':', alpha=0.5)
        for xi, yi, lam in zip(x, y, lams):
            ax.annotate(f'{lam:+.3f}', (xi, yi), fontsize=6.5, color='#1f77b4',
                        xytext=(3, 5), textcoords='offset points')
        ax.set_xlabel('colonoscopies per person (all sources)')
        ax.set_ylabel('CRC deaths per 100,000')
        ax.set_title(titles[surv], fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    any_e = next(iter(agg.values()))
    fig.suptitle('CRC mortality vs colonoscopy volume: PBVI policy against fixed schedules '
                 f'(CMOST, ages 40-80, {any_e["n_seeds"]} x {any_e["n"]:,} per point)',
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, 'surv_fair_frontier.png')
    fig.savefig(out, dpi=160)
    print('saved', out)


def make_risk_figure(agg):
    """The mechanism behind the frontier: the policy does not screen better,
    it screens a different set of people. Panel (a) plots each arm's high-risk
    against its low-risk colonoscopy volume -- a uniform schedule sits on the
    diagonal by construction, and distance above it is exactly the risk
    reallocation. Panel (b) shows what that buys, per stratum."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    panels = [s_ for s_ in ('nosurv', 'surv') if any(k[0] == s_ for k in agg)]
    fig, axes = plt.subplots(2, len(panels), figsize=(6.6 * len(panels), 9.4), squeeze=False)
    titles = {'nosurv': 'Primary: surveillance OFF', 'surv': 'Sensitivity: surveillance ON'}

    for col, surv in enumerate(panels):
        # ---- (a) where the volume goes
        ax = axes[0][col]
        hx, hy, lams = [], [], []
        for lam in sorted(l for (sv, a, l) in agg if sv == surv and a == 'policy'):
            e = agg[(surv, 'policy', lam)]
            hx.append(e['by_risk']['low_risk']['avg_colonoscopies_per_person'])
            hy.append(e['by_risk']['high_risk']['avg_colonoscopies_per_person'])
            lams.append(lam)
        ax.plot(hx, hy, 'o-', color='#1f77b4', ms=4, lw=1.3, label='PBVI policy')
        for xi, yi, lam in zip(hx, hy, lams):
            ax.annotate(f'{lam:+.3f}', (xi, yi), fontsize=6.5, color='#1f77b4',
                        xytext=(4, 3), textcoords='offset points')
        for arm, c, m in (('no_screen', '#7f7f7f', 's'), ('q10y', '#d62728', '^'),
                          ('q5y', '#2ca02c', 'v')):
            e = agg.get((surv, arm, None))
            if e:
                ax.plot([e['by_risk']['low_risk']['avg_colonoscopies_per_person']],
                        [e['by_risk']['high_risk']['avg_colonoscopies_per_person']],
                        m, color=c, ms=9, label=arm)
        top = max(max(hx + hy), 6)
        ax.plot([0, top], [0, top], ls='--', lw=0.9, color='#999999',
                label='uniform (high = low)')
        ax.set_xlabel('colonoscopies per person, LOW risk (bottom 80%)')
        ax.set_ylabel('colonoscopies per person, HIGH risk (top 20%)')
        ax.set_title(titles[surv] + ' -- volume allocation', fontsize=10.5)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        # ---- (b) what it buys, per stratum
        ax = axes[1][col]
        for risk, c in (('high_risk', '#b5651d'), ('low_risk', '#1f77b4')):
            x, y, lm = policy_curve(agg, surv, 'crc_death_per_100k', risk=risk)
            err = [agg[(surv, 'policy', l)]['by_risk'][risk]['crc_death_per_100k_sem']
                   for l in lm]
            ax.errorbar(x, y, yerr=err, fmt='o-', color=c, ms=4, lw=1.3, capsize=2,
                        label=f'policy, {risk.replace("_", " ")}')
        for arm, m in (('q10y', '^'), ('q5y', 'v')):
            e = agg.get((surv, arm, None))
            if not e:
                continue
            for risk, c in (('high_risk', '#b5651d'), ('low_risk', '#1f77b4')):
                r = e['by_risk'][risk]
                ax.plot([r['avg_colonoscopies_per_person']], [r['crc_death_per_100k']],
                        m, color=c, ms=10, mec='k', mew=0.7,
                        label=f'{arm}, {risk.replace("_", " ")}')
        ax.set_xlabel("colonoscopies per person within that stratum")
        ax.set_ylabel('CRC deaths per 100,000 in that stratum')
        ax.set_title(titles[surv] + ' -- outcome per stratum', fontsize=10.5)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, ncol=2)

    fig.suptitle('How the PBVI policy buys its advantage: it moves colonoscopies '
                 'from low-risk to high-risk people', fontsize=12.5)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, 'surv_fair_risk_targeting.png')
    fig.savefig(out, dpi=160)
    print('saved', out)


def write_csv(agg):
    import csv
    path = os.path.join(RES, 'summary.csv')
    base = ['surveillance', 'arm', 'lam', 'n', 'n_seeds']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(base + METRICS + [m + '_sem' for m in METRICS] + TOP_ONLY
                   + [f'{r}_{m}' for r in RISKS for m in METRICS])
        for key in sorted(agg, key=lambda k: (k[0], k[1], k[2] if k[2] is not None else -9)):
            e = agg[key]
            w.writerow(list(key) + [e['n'], e['n_seeds']]
                       + [e[m] for m in METRICS] + [e[m + '_sem'] for m in METRICS]
                       + [e[m] for m in TOP_ONLY]
                       + [e['by_risk'][r][m] for r in RISKS for m in METRICS])
    print('saved', path)


def main():
    agg = load()
    if not agg:
        sys.exit(f'no results in {RES}')
    any_e = next(iter(agg.values()))
    out = ['PBVI vs fixed-schedule colonoscopy screening in CMOST',
           f'{len(agg)} arm-points, {any_e["n_seeds"]} seeds x {any_e["n"]:,} '
           f'individuals each, ages 40-80',
           'risk label: top 20% of CMOST individual_risk, disclosed to the agent at t=0']
    for surv in ('nosurv', 'surv'):
        if any(k[0] == surv for k in agg):
            section(agg, surv, out)
    compare_primary_vs_sensitivity(agg, out)
    txt = '\n'.join(out)
    print(txt)
    with open(os.path.join(RES, 'report.txt'), 'w') as f:
        f.write(txt + '\n')
    write_csv(agg)
    make_figure(agg)
    make_risk_figure(agg)


if __name__ == '__main__':
    main()
