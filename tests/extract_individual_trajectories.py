"""extract_individual_trajectories.py
======================================
Real-CMOST individual trajectories under the trained policy -- state (sr)
+ WAIT/SCREEN decisions (action_recorder) + per-year QALY, for a handful of
illustrative people. Krijkamp et al. (Med Decis Making 2018) Figure 2/3
style: "this is what personalization actually looks like for one person",
not just population averages.

Small N (this is for hand-picking a few example people, not a precision
estimate) -- runs the real engine directly (not cmost_4way_eval.run_cohort,
which doesn't wire up action_recorder/decision_state_recorder) so we get
the actual WAIT/SCREEN call at every age, not just the disease state.
"""
import os
import sys
import io
import json
import contextlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import cmost_4way_eval as C4
from pomdp.model_v2 import age_weight, u_initial, u_continuing, u_term_crc, u_term_other
from compute_real_qaly import AGE_W, U_CONT, U_INIT, U_TCRC, U_TOTH

RES = os.path.join(os.path.dirname(__file__), '..', 'results')

STATE_GROUP = {  # 18-state -> readable group index
    0: 0,                          # Normal
    1: 1, 2: 1, 3: 1, 4: 1,         # EarlyPolyp
    5: 2, 6: 2,                    # AdvPolyp
    7: 3, 8: 3, 9: 3, 10: 3,       # Undetected Ca (any stage)
    11: 4, 12: 4, 13: 4, 14: 4,    # Detected Ca (any stage)
    15: 5, 16: 5, 17: 5,           # Dead
}
GROUP_NAMES = ['Normal', 'EarlyPolyp', 'AdvPolyp', 'UndetectedCa', 'DetectedCa', 'Dead']


def main():
    n = 20000
    seed = 4242
    np.random.seed(seed)
    p = C4.BNH.prepare_simulation_params(n)
    p['flag']['Polyp_Surveillance'] = False
    p['flag']['Cancer_Surveillance'] = False

    pomdp_m, solver_m = C4.train_policy(sex=1)
    pomdp_f, solver_f = C4.train_policy(sex=2)
    risk_class = (np.asarray(p['individual_risk']) >= C4.RISK_THRESHOLD).astype(int)
    sex_arr = np.asarray(p['gender_arr']).astype(int)  # 1=male, 2=female
    hook = C4.SexAwareEngineHook(pomdp_m, solver_m, pomdp_f, solver_f, risk_class, sex_arr, seed=seed)

    sr = np.zeros((100, n), dtype=np.int8)
    act = np.full((100, n), -1, dtype=np.int8)
    dstate = np.full((100, n), -1, dtype=np.int8)
    ncr = np.zeros(n, dtype=np.int32)
    args = [p['p'], p['stage_variables'], p['location'], p['cost'], p['cost_stage'],
            p['risc'], p['flag'], p['special_text'], p['female'], p['sensitivity'],
            p['screening_test'], p['screening_preference'], p['age_progression'],
            p['new_polyp'], p['colonoscopy_likelyhood'], p['individual_risk'],
            p['risk_dist'], p['gender_arr'], p['life_table'], p['mortality_matrix'],
            p['location_matrix'], p['stage_duration'], p['tx1'],
            p['direct_cancer_rate'], p['direct_cancer_speed'], p['dwell_speed']]
    print('running N=%d real-CMOST policy sim with action/state recorders...' % n, flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        out = C4.NumberCrunching_policy(*args, state_recorder=sr, policy_hook=hook,
                                         n_colo_recorder=ncr, action_recorder=act,
                                         decision_state_recorder=dstate,
                                         policy_hook_age_min=40, policy_hook_age_max=80)
    DeathYear = out[4]
    print('done. picking example people...', flush=True)

    risk_arr = risk_class
    # candidates: stayed fully Normal to age 85+; had polyp caught (detected
    # via screen then reverts to Normal-track... in this 18-state layout,
    # once cured a polyp patient's Ca_Cancer stays empty so state returns to
    # Normal-ish/low group); progressed to detected cancer.
    max_group_per_person = np.array([STATE_GROUP[int(x)] for x in sr.max(axis=0)])
    n_screens = (act == 1).sum(axis=0)

    def pick(mask, k=1, sort_key=None):
        idx = np.where(mask)[0]
        if sort_key is not None:
            idx = idx[np.argsort(sort_key[idx])]
        return idx[:k]

    # Pick for CONTRAST, not just category membership -- the whole point of
    # this figure is that the policy looks different for different people,
    # so favor examples whose n_screens diverges from the population-common
    # {54,66,75} pattern instead of whichever person happens to match it.
    low_risk_light = pick((max_group_per_person <= 1) & (risk_arr == 0),
                           k=1, sort_key=n_screens)  # fewest screens, ascending
    high_risk_heavy_idx = np.where((max_group_per_person <= 2) & (risk_arr == 1))[0]
    high_risk_heavy_idx = high_risk_heavy_idx[np.argsort(-n_screens[high_risk_heavy_idx])]  # most screens
    high_risk_heavy = high_risk_heavy_idx[:1]
    cancer_idx = pick(max_group_per_person == 4, k=1)
    # 4th example: screen-detected at an early stage (DetectedCa Stage I,
    # sr code 11) who then lives to an old age -- illustrating the "caught
    # early, survived" outcome alongside the missed/progressed one above.
    is_stageI_detected = (sr == 11).any(axis=0)
    survived_old = DeathYear >= 80
    early_caught_idx = pick(is_stageI_detected & survived_old, k=1)
    chosen = list(low_risk_light) + list(high_risk_heavy) + list(cancer_idx) + list(early_caught_idx)
    labels = ['low-risk, minimal screening', 'high-risk, frequent screening',
              'progressed to detected cancer', 'caught early by screening, survived']
    print('chosen person indices:', chosen)

    people = []
    for label, i in zip(labels, chosen):
        i = int(i)
        states18 = sr[:, i].tolist()
        actions = act[:, i].tolist()
        dyear = float(DeathYear[i])
        # per-year QALY, ages 1..100 -- reuse the SAME lookup tables as
        # compute_real_qaly.qaly_from_sr, but keep the per-year series
        # instead of summing.
        qaly_series = []
        first_det = None
        for yi, s in enumerate(states18):
            if s >= 11 and s <= 14 and first_det is None:
                first_det = yi
        for yi, s in enumerate(states18):
            if yi < 39:
                qaly_series.append(None)
                continue
            if s <= 6:
                u = AGE_W[yi]
            elif 7 <= s <= 10:
                u = U_CONT[yi, s - 7]
            elif 11 <= s <= 14:
                k = s - 11
                u = U_INIT[yi, k] if first_det == yi else U_CONT[yi, k]
            else:
                u = None  # dead -- filled below via death-year half-cycle
            qaly_series.append(u)
        people.append(dict(
            label=label, person_idx=i, risk_class=int(risk_arr[i]),
            sex=int(sex_arr[i]),  # 1=male, 2=female
            death_year=dyear, n_screens=int(n_screens[i]),
            states18=states18, actions=actions, qaly_series=qaly_series,
        ))

    out_path = os.path.join(RES, 'individual_trajectories.json')
    with open(out_path, 'w') as f:
        json.dump(dict(n=n, seed=seed, group_names=GROUP_NAMES, people=people), f)
    print('saved', out_path)
    for pe in people:
        scr_ages = [a + 1 for a, act_v in enumerate(pe['actions']) if act_v == 1]
        sex_lbl = 'M' if pe['sex'] == 1 else 'F'
        print(f"{pe['label']:<36} sex={sex_lbl} risk={pe['risk_class']} death_year={pe['death_year']:.1f} "
              f"screens@{scr_ages}")


if __name__ == '__main__':
    main()
