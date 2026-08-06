"""
export_tauphase_vs_cmost_csv.py
================================
CSV export of the ORIGINAL (pre-session, untouched) 97-state tau-phase
transition-matrix vs CMOST comparison (transitions/verify_screening_
tauphase.py -- the script SCREENING_VALIDATION.md's findings table was
built from), so it can be compared numerically against compare_9state_
vs_cmost*.csv (this session's 9-state result).

Does NOT modify verify_screening_tauphase.py -- imports and reuses its
functions as-is, only adding a CSV-writing wrapper.

ONE deliberate deviation from the original: build_params pool size is
20000 here, not the original's 500 (that 500-pool has the individual_risk
resampling-noise bug found and fixed elsewhere this session; reusing the
buggy pool would unfairly penalise the tauphase side of the comparison
with noise that has nothing to do with the modelling approach itself).

Run: python export_tauphase_vs_cmost_csv.py -n 100000
"""
from __future__ import annotations
import os, sys, csv, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.params import build_params
from env.cmost_individual import CRCEngine
import verify_screening_tauphase as VST

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


def row_from_compare(name, micro, coh, rows_out):
    s_m, s_c = micro['stage_pct'], coh['stage_pct']
    specs = [
        ('Avg age at death (all)', micro['avg_age_death_all'], coh['avg_age_death_all']),
        ('Avg age at CRC death', micro['avg_age_crc_death'], coh['avg_age_crc_death']),
        ('CRC deaths /100k', micro['crc_deaths_100k'], coh['crc_deaths_100k']),
        ('CRC incidence /100k', micro['crc_incidence_100k'], coh['crc_incidence_100k']),
        ('Stage I %', s_m[0], s_c[0]), ('Stage II %', s_m[1], s_c[1]),
        ('Stage III %', s_m[2], s_c[2]), ('Stage IV %', s_m[3], s_c[3]),
    ]
    for label, a, b in specs:
        diff = f"{100*(b-a)/a:+.1f}%" if abs(a) > 1e-9 else ''
        rows_out.append({'scenario': name, 'metric': label,
                         'CMOST': a, 'matrix_tauphase97': b, 'diff_pct': diff})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=20240705)
    args = ap.parse_args()

    eng = CRCEngine(build_params('CMOST13', 20000, args.seed), rng=np.random.default_rng(0))
    d_EA, d_AA = VST.load_adenoma_detect()
    dPC = VST.stage_specific_dPC(eng)
    print(f"NS={VST.NS} states; d_EA={d_EA:.3f} d_AA={d_AA:.3f} d_PC[I..IV]={np.round(dPC,3).tolist()}")

    scenarios = {'No screening': [], 'Screen q10y (50,60,70)': [50, 60, 70],
                 'Screen q5y (50-75)': [50, 55, 60, 65, 70, 75]}
    micro = {}
    print(f"[1] no-screening microsim (quarterly, n={args.n}) ...")
    micro['No screening'], paths = VST.run_microsim_q(eng, args.n, [], True, args.seed)
    print("[2] estimating tau-phased quarterly transition matrix ...")
    T = VST.estimate_Tq(paths)
    for name, sa in scenarios.items():
        if name == 'No screening':
            continue
        print(f"[3] microsim {name} (n={args.n}) ...")
        micro[name], _ = VST.run_microsim_q(eng, args.n, sa, False, args.seed + 1)

    rows_out = []
    for name, sa in scenarios.items():
        coh = VST.run_cohort_q(T, sa, d_EA, d_AA, dPC)
        VST.print_compare(name, micro[name], coh)
        row_from_compare(name, micro[name], coh, rows_out)

    out_csv = os.path.join(RES, 'tauphase97_vs_cmost.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scenario', 'metric', 'CMOST', 'matrix_tauphase97', 'diff_pct'])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nSaved -> {out_csv}")


if __name__ == '__main__':
    main()
