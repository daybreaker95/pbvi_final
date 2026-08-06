"""Matrix-side-only recompute against the NEW real-engine-estimated Tn/d_symp
(transitions_9state_stratified.npz now swapped to the real-engine version),
compared against the ALREADY-SAVED real CMOST engine N=1M results from
tests/cmost_4way_eval.py (results/cmost_4way_{no_screen,q10y,q5y}.json) --
no need to re-run the slow real-engine simulation, matrix propagation itself
is fast (deterministic cohort, no MC)."""
import os
import sys
import json
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pomdp.model_v2 import CRCScreeningPOMDP9
from compare_model_v2_vs_cmost import run_cohort_modelv2, MAXY

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

pomdp = CRCScreeningPOMDP9(age_min=1, age_max=85, life_max=MAXY, gamma=0.97)
print(f"n_risk={pomdp.n_risk} NC={pomdp.NC} NS={pomdp.NS} T_det_source={pomdp._T_det_source} "
      f"frac_high={pomdp.frac_high:.4f}")

scenarios = {
    'No screening': ([], 'cmost_4way_no_screen.json'),
    'Screen q10y (50,60,70)': ([50, 60, 70], 'cmost_4way_q10y.json'),
    'Screen q5y (50-75)': ([50, 55, 60, 65, 70, 75], 'cmost_4way_q5y.json'),
}

rows = []
for name, (sa, fname) in scenarios.items():
    mx = run_cohort_modelv2(pomdp, sa)
    real = json.load(open(os.path.join(RES, fname)))

    def add(metric, real_val, matrix_val):
        if matrix_val == 0 and real_val == 0:
            diff = ''
        elif matrix_val == 0:
            diff = ''
        else:
            diff = f"{(real_val - matrix_val) / matrix_val * 100:+.1f}%"
        rows.append([name, metric, round(real_val, 2), round(matrix_val, 2), diff])

    add('CRC deaths /100k', real['crc_death_per_100k'], mx['crc_deaths_100k'])
    add('Incidence /100k', real['incidence_per_100k'], mx['incid_100k'])
    add('Screen-detected /100k', real['screen_detected_per_100k'], mx['screen_100k'])
    add('Symptom-detected /100k', real['symptom_detected_per_100k'], mx['symptom_100k'])
    for k, lbl in enumerate(['I', 'II', 'III', 'IV']):
        add(f'Stage {lbl} % (all)', real['stage_pct_all'][k], mx['stage_all_pct'][k])
    if mx['screen_100k'] > 0:
        for k, lbl in enumerate(['I', 'II', 'III', 'IV']):
            add(f'Stage {lbl} % (screen)', real['stage_pct_screen'][k], mx['stage_scr_pct'][k])
    for k, lbl in enumerate(['I', 'II', 'III', 'IV']):
        add(f'Stage {lbl} % (symptom)', real['stage_pct_symptom'][k], mx['stage_sym_pct'][k])

out_path = os.path.join(RES, 'compare_model_v2_vs_real_cmost_v2.csv')
with open(out_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['scenario', 'metric', 'real_cmost', 'matrix9state_v2', 'diff_pct'])
    w.writerows(rows)
print('saved', out_path)

print(f"\n{'scenario':<26}{'metric':<24}{'real':>10}{'matrix':>10}{'gap':>10}")
for r in rows:
    print(f"{r[0]:<26}{r[1]:<24}{r[2]:>10}{r[3]:>10}{r[4]:>10}")
