"""Poster-legible re-plots of two manuscript figures (same data, same layout,
same pixel size as dp/figures.py output; poster typography).

  dp_frontier_poster.png   1760 x 736  (figsize 11 x 4.6 in @ 160 dpi)
  dp_adherence_poster.png  1760 x 704  (figsize 11 x 4.4 in @ 160 dpi)

Reads only results/dp/*.json; writes only the two PNGs above.
python paper/ksaim_poster/assets/meta/make_poster_figures.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
RES = os.path.join(ROOT, 'results', 'dp')
OUT = os.path.join(ROOT, 'paper', 'ksaim_poster', 'assets', 'figures')
TAG, N_HEAD, N_GRID = 'c6b', 1_000_000, 200_000
DPI = 160

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 17,
    'axes.titlesize': 19,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14.5,
    'lines.linewidth': 2.4,
    'lines.markersize': 8,
})
ELW = 1.6          # error-bar line width (thinner than the series line)
REF_LW = 1.6       # 'no screening' dashed reference line
REF_FS = 14        # its label


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def frontier(rows, metric='death', bins=np.arange(0.5, 8.51, 0.5)):
    """Best fixed schedule (lowest metric) per colonoscopy-volume bin
    (verbatim copy of dp.fixed_search.frontier, kept local on purpose)."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [r for r in rows if lo <= r['colos'] < hi]
        if sel:
            out.append(min(sel, key=lambda r: r[metric]))
    return out


def _ref_line(ax, y, label='no screening'):
    ax.axhline(y, color='gray', lw=REF_LW, ls='--', zorder=1)
    ax.text(0.985, y, label, color='gray', fontsize=REF_FS, ha='right', va='top',
            transform=ax.get_yaxis_transform())


def _save(fig, name, legend, w_px, h_px):
    """Place a figure-level legend below both panels, then pack the axes above it."""
    fig.canvas.draw()
    bb = legend.get_window_extent().transformed(fig.transFigure.inverted())
    fig.tight_layout(rect=[0, bb.y1 + 0.012, 1, 1], w_pad=1.2)
    out = os.path.join(OUT, name)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    from PIL import Image
    assert Image.open(out).size == (w_px, h_px), Image.open(out).size
    return out


# ---------------------------------------------------------------------------
def frontier_figure():
    head = _load(os.path.join(RES, f'eval_headline_{TAG}_n{N_HEAD}.json'))
    fixed = _load(os.path.join(RES, f'fixed_search_{TAG}.json'))
    surv = _load(os.path.join(RES, f'eval_surveillance_n{N_HEAD}.json'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    H = {}   # legend handles by key

    # adaptive policy families (grid, n = 200,000 per point)
    for obj, col, lab0, lab1 in (
            ('death', 'C0', 'adaptive policy (mortality objective), engine n = 200,000',
             'adaptive policy (mortality objective)'),
            ('inc', 'C2', 'adaptive policy (incidence objective)', 'adaptive policy (incidence objective)')):
        grid = _load(os.path.join(RES, f'eval_grid_{TAG}_{obj}_n{N_GRID}.json'))
        pts = [(r['colos_per_person'], r['crc_death_per_100k'], r['incidence_per_100k'],
                r.get('crc_death_se', 0), r.get('incidence_se', 0))
               for k, r in grid.items() if k.startswith(f'{TAG}_{obj}_lam')]
        pts.sort()
        x = [p[0] for p in pts]
        H[obj] = axes[0].errorbar(x, [p[1] for p in pts], yerr=[p[3] for p in pts], fmt='o-',
                                  color=col, elinewidth=ELW, label=lab0, zorder=3)
        axes[1].errorbar(x, [p[2] for p in pts], yerr=[p[4] for p in pts], fmt='o-',
                         color=col, elinewidth=ELW, label=lab1, zorder=3)

    src = head
    # guideline fixed schedules (headline, n = 1,000,000)
    for key, mk, lab in (('q10y', 's', 'fixed 10-yearly (50/60/70)'), ('q5y', 'D', 'fixed 5-yearly (50–75)')):
        r = src[key]
        H[key] = axes[0].errorbar([r['colos_per_person']], [r['crc_death_per_100k']], yerr=[r['crc_death_se']],
                                  fmt=mk, ms=13, color='k', elinewidth=ELW, label=lab, zorder=5)
        axes[1].errorbar([r['colos_per_person']], [r['incidence_per_100k']], yerr=[r['incidence_se']],
                         fmt=mk, ms=13, color='k', elinewidth=ELW, zorder=5)
    # best fixed schedules from the in-model search, engine-verified
    for k, r in src.items():
        if k.startswith('bestfixed'):
            axes[0].plot([r['colos_per_person']], [r['crc_death_per_100k']], '^', ms=13, color='C3', zorder=6)
            axes[1].plot([r['colos_per_person']], [r['incidence_per_100k']], '^', ms=13, color='C3', zorder=6)
    H['bestfixed'] = Line2D([], [], marker='^', ms=13, color='C3', ls='none',
                            label='best fixed schedule (engine-verified)')
    # surveillance-augmented programmes and the adaptive policy at their volume
    agg = surv['aggregate']
    for key, mk, lab in (('q10y_surv', 'P', '10-yearly + CMOST surveillance'),
                         ('q5y_surv', 'X', '5-yearly + CMOST surveillance')):
        r = agg[key]
        H[key] = axes[0].errorbar([r['colos_per_person']], [r['crc_death_per_100k']], yerr=[r['crc_death_se']],
                                  fmt=mk, ms=14, color='C6', elinewidth=ELW, label=lab, zorder=5)
        axes[1].errorbar([r['colos_per_person']], [r['incidence_per_100k']], yerr=[r['incidence_se']],
                         fmt=mk, ms=14, color='C6', elinewidth=ELW, zorder=5)
    for k, r in agg.items():
        if k.startswith('dp_death_lam') and k.endswith('_surv'):
            axes[0].plot([r['colos_per_person']], [r['crc_death_per_100k']], '*', ms=19, color='C1', zorder=7)
            axes[1].plot([r['colos_per_person']], [r['incidence_per_100k']], '*', ms=19, color='C1', zorder=7)
    H['dp_surv'] = Line2D([], [], marker='*', ms=19, color='C1', ls='none',
                          label="adaptive policy at the surveillance programme's volume")
    # in-model fixed frontier (faint reference)
    fr = frontier(fixed['rows'], 'death')
    H['inmodel'], = axes[0].plot([r['colos'] for r in fr], [r['death'] * 1e5 for r in fr], ':', color='gray',
                                 label='best fixed schedules, in-model', zorder=2)
    fr = frontier(fixed['rows'], 'inc')
    axes[1].plot([r['colos'] for r in fr], [r['inc'] * 1e5 for r in fr], ':', color='gray', zorder=2)
    # no-screening reference
    _ref_line(axes[0], src['none']['crc_death_per_100k'])
    _ref_line(axes[1], src['none']['incidence_per_100k'])

    for ax in axes:
        ax.set_xlabel('colonoscopies per person (ages 40–80)')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('CRC deaths\nper 100,000')
    axes[1].set_ylabel('CRC diagnoses\nper 100,000')
    axes[0].set_title('Mortality frontier (engine)')
    axes[1].set_title('Incidence frontier (engine)')

    # column-major: col 1 = adaptive + best-fixed entries, col 2 = guideline + surveillance programmes
    order = ['death', 'inc', 'dp_surv', 'bestfixed', 'inmodel', 'q10y', 'q5y', 'q10y_surv', 'q5y_surv']
    handles = [H[k] for k in order]
    leg = fig.legend(handles, [h.get_label() for h in handles], loc='lower center', ncol=2,
                     frameon=True, columnspacing=2.5, handlelength=1.8, handletextpad=0.6,
                     labelspacing=0.28, borderpad=0.55, borderaxespad=0.2)
    return _save(fig, 'dp_frontier_poster.png', leg, 1760, 736)


# ---------------------------------------------------------------------------
def adherence_figure():
    d = _load(os.path.join(RES, f'eval_adherence_{TAG}_n{N_GRID}.json'))
    alphas = [1.0, 0.7, 0.5, 0.3]
    series = (  # (label, colour, keys at alpha = 1.0, 0.7, 0.5, 0.3)
        ('fixed 10-yearly, missed slot lost', 'k', ['q10y', 'q10y_a07', 'q10y_a05', 'q10y_a03']),
        ('fixed 10-yearly + annual recall', 'C3', ['q10y', 'q10y_recall_a07', 'q10y_recall_a05', 'q10y_recall_a03']),
        ('adaptive policy', 'C1', ['dp_death_lam0.001561_q10y', 'dp_a07', 'dp_a05', 'dp_a03']),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    handles = []
    for lab, col, keys in series:
        rows = [d[k] for k in keys]
        h = axes[0].errorbar(alphas, [r['crc_death_per_100k'] for r in rows], yerr=[r['crc_death_se'] for r in rows],
                             fmt='o-', color=col, elinewidth=ELW, label=lab, zorder=3)
        axes[1].errorbar(alphas, [r['colos_per_person'] for r in rows], yerr=[r['colos_se'] for r in rows],
                         fmt='o-', color=col, elinewidth=ELW, zorder=3)
        handles.append(h)
    _ref_line(axes[0], d['none']['crc_death_per_100k'])
    for ax in axes:
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    axes[0].set_xlabel('adherence (probability an invitation is attended)')
    axes[0].set_ylabel('CRC deaths\nper 100,000')
    axes[0].set_title('CRC mortality vs adherence\n(engine, n = 200,000)')
    axes[1].set_xlabel('adherence')
    axes[1].set_ylabel('colonoscopies\nper person')
    axes[1].set_title('Colonoscopy use')
    leg = fig.legend(handles, [h.get_label() for h in handles], loc='lower center', ncol=3,
                     frameon=True, columnspacing=2.0, handlelength=1.8, handletextpad=0.6,
                     borderpad=0.4, borderaxespad=0.25)
    return _save(fig, 'dp_adherence_poster.png', leg, 1760, 704)


if __name__ == '__main__':
    print(frontier_figure())
    print(adherence_figure())
