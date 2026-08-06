"""CMOST-in-the-loop evaluation of the FiVI-trained policy (pure 13-state
model_v2). Runs the REAL CMOST engine (NumberCrunching_policy, MATLAB-ported)
with an EngineHook13 that, at each age 40-85, consults the trained FiVI
policy's belief (over our 13-state x 2-risk-class POMDP) to decide WAIT vs
SCREEN. CMOST's own disease dynamics (not our transition-matrix
approximation) drive what actually happens to the patient; our POMDP only
supplies the decision-maker and the belief update.

Surveillance (Polyp_Surveillance / Cancer_Surveillance) is left ON, matching
Zaika 2024's own convention: surveillance runs automatically inside CMOST as
a guideline-driven background process, NOT a decision variable -- the trained
policy only controls primary screening (WAIT/SCREEN), exactly as designed.

Baseline for reduction% = same CMOST engine, no screening at all
(policy_hook=None, natural history), so gaps are pure policy effect, not
CMOST-vs-matrix approximation error (that's covered separately by
compare_model_v2_vs_cmost.py).

Clinical/cost metrics are read DIRECTLY from CMOST's own native outputs
(state_recorder for incidence/mortality, Money dict for dollar cost,
Number/n_colo_recorder for colonoscopy counts) -- NOT reconstructed via our
POMDP's blended (u - cost/wtp) reward, since raw cost isn't separately
exposed there. This treats CMOST as ground truth for clinical/cost outcomes;
our POMDP is only the policy being evaluated.
"""
import os
import sys
import io
import json
import time
import contextlib
import warnings

PBVI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CMOST_PY = os.path.abspath(os.path.join(
    PBVI_ROOT, '..', '..', '..', '..', 'cmost_experiment_final', 'CMOST_experiment', 'python'))
sys.path.insert(0, PBVI_ROOT)
sys.path.insert(0, CMOST_PY)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np

from pomdp.model_v2 import CRCScreeningPOMDP9, NORMAL, EARLY_POLYP, ADV_POLYP, \
    CA_STAGES, CRC_DEATH, OTHER_DEATH, DET_CA_STAGES
from pomdp.fivi import FiVI

import build_natural_history_transition_matrix as BNH
from NumberCrunching_policy import NumberCrunching_policy

# 18-state (CMOST engine's own internal _pomdp_state_idx convention) -> our
# 13-state local index. P1-P4 = early adenoma progression rows, P5-P6 =
# advanced (matches GenderProgression[0:4]=early / [4:6]=advanced in
# NumberCrunching_policy.py's own setup, i.e. CMOST's native early/adv split).
MAP18TO13 = {
    0: NORMAL,
    1: EARLY_POLYP, 2: EARLY_POLYP, 3: EARLY_POLYP, 4: EARLY_POLYP,
    5: ADV_POLYP, 6: ADV_POLYP,
    7: CA_STAGES[0], 8: CA_STAGES[1], 9: CA_STAGES[2], 10: CA_STAGES[3],
    11: DET_CA_STAGES[0], 12: DET_CA_STAGES[1], 13: DET_CA_STAGES[2], 14: DET_CA_STAGES[3],
    15: CRC_DEATH,       # Dead_CRC
    16: OTHER_DEATH,     # Dead_Comp (procedure complication death)
    17: OTHER_DEATH,     # Dead_Other
}


class EngineHook13:
    """policy_hook interface expected by NumberCrunching_policy:
    decide_one(z, y) -> action ; obs(a, s_true_18) -> observation ;
    step_update(z, a, o, y) -> None (updates internal belief[z])."""

    def __init__(self, pomdp, solver, risk_class, seed=0):
        self.p = pomdp
        self.solver = solver
        self.risk_class = risk_class          # (n,) int 0/1, per-patient TRUE risk bucket
        n = len(risk_class)
        b0 = pomdp.initial_belief()           # population-average prior (no family-history signal)
        self.belief = np.tile(b0, (n, 1))
        self.rng = np.random.default_rng(seed + 7)
        self._cur_z = None
        self._cur_y = None

    def decide_one(self, z, y):
        self._cur_z = z
        self._cur_y = y
        a = self.solver.best_action(y, self.belief[z])
        return int(a)

    def obs(self, a, s_true_18):
        z = self._cur_z
        y = self._cur_y
        r = int(self.risk_class[z])
        local13 = MAP18TO13[int(s_true_18)]
        s_full = self.p.fidx(r, local13)
        age = min(y, self.p.life_max)
        M = self.p.M[age][a]
        NO = 5
        p_o = np.array([M[o][s_full, :].sum() for o in range(NO)])
        tot = p_o.sum()
        if tot <= 1e-300:
            p_o = np.full(NO, 1.0 / NO)
        else:
            p_o = p_o / tot
        return int(self.rng.choice(NO, p=p_o))

    def step_update(self, z, a, o, y):
        age = min(y, self.p.life_max)
        M = self.p.M[age][a][o]
        bn = self.belief[z] @ M
        tot = bn.sum()
        if tot > 1e-300:
            self.belief[z] = bn / tot


def train_policy():
    pomdp = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97)
    solver = FiVI(pomdp, seed=0, max_sawtooth_points=300)
    solver.solve(max_iters=50, time_limit=300, precision=1e-3, verbose=False,
                 n_stochastic_trajectories=0, use_pbs=False, dbbu_interval=1)
    return pomdp, solver


def run_cohort(n, seed, policy_hook, surveillance=True, hook_age_max=85):
    np.random.seed(seed)
    p = BNH.prepare_simulation_params(n)
    if surveillance:
        p['flag']['Polyp_Surveillance'] = True
        p['flag']['Cancer_Surveillance'] = True
    sr = np.zeros((100, n), dtype=np.int8)
    ncr = np.zeros(n, dtype=np.int32)
    args = [p['p'], p['stage_variables'], p['location'], p['cost'], p['cost_stage'],
            p['risc'], p['flag'], p['special_text'], p['female'], p['sensitivity'],
            p['screening_test'], p['screening_preference'], p['age_progression'],
            p['new_polyp'], p['colonoscopy_likelyhood'], p['individual_risk'],
            p['risk_dist'], p['gender_arr'], p['life_table'], p['mortality_matrix'],
            p['location_matrix'], p['stage_duration'], p['tx1'],
            p['direct_cancer_rate'], p['direct_cancer_speed'], p['dwell_speed']]
    with contextlib.redirect_stdout(io.StringIO()):
        out = NumberCrunching_policy(*args, state_recorder=sr, policy_hook=policy_hook,
                                      n_colo_recorder=ncr,
                                      policy_hook_age_min=40, policy_hook_age_max=hook_age_max)
    Money = out[22]
    Number = out[23]
    return p, sr, ncr, Money, Number


def summarize(sr, ncr, Money, Number, n):
    # sr: (100, n) end-of-year 18-state index, yi=age-1. Full remaining
    # lifetime (age 40..100 -> yi 39..99), matching the matrix-side FiVI
    # training's lifetime crc_100k/incid_100k convention (life_max=100),
    # not clipped to the age-85 policy-decision horizon.
    win = sr[39:100]
    crc_death = (win == 15).any(axis=0)
    incid = np.isin(win, list(range(7, 15))).any(axis=0)
    total_cost = float(Money['AllCost'].sum())        # cohort total, $ (undiscounted)
    return {
        'n': n,
        'crc_death_per_100k': float(crc_death.mean() * 100_000),
        'incidence_per_100k': float(incid.mean() * 100_000),
        'avg_colonoscopies_per_person': float(ncr.mean()),
        'total_cost_usd': total_cost,
        'cost_per_person_usd': total_cost / n,
        'screening_colo_total': float(Number['Screening_Colonoscopy'].sum()),
        'followup_colo_total': float(Number['Follow_Up_Colonoscopy'].sum()),
    }


def main():
    N = 200_000
    SEED = 999
    print('[1/3] training FiVI policy (pure 13-state model)...', flush=True)
    pomdp, solver = train_policy()
    print(f'  gap={solver.gap_history[-1]["gap"]:.4f}', flush=True)

    print(f'[2/3] CMOST-in-loop cohort, policy-driven, N={N:,} ...', flush=True)
    t0 = time.time()
    p_pol = BNH.prepare_simulation_params(N)
    risk_class = (np.asarray(p_pol['individual_risk']) >= 3.527830006754019).astype(int)
    hook = EngineHook13(pomdp, solver, risk_class, seed=SEED)
    # surveillance OFF here too: our POMDP has no surveillance concept at all
    # (states/transitions/belief never represent it), so leaving CMOST's own
    # guideline surveillance on would let it silently inject extra
    # colonoscopies/benefit the trained policy neither decided nor is aware
    # of, contaminating the policy-attributable effect. Symmetric with the
    # no-screen baseline below.
    _, sr_pol, ncr_pol, money_pol, number_pol = run_cohort(N, SEED, hook, surveillance=False)
    stats_pol = summarize(sr_pol, ncr_pol, money_pol, number_pol, N)
    print(f'  done ({time.time()-t0:.0f}s): {stats_pol}', flush=True)

    print(f'[3/3] CMOST no-screening baseline cohort, N={N:,} ...', flush=True)
    t0 = time.time()
    _, sr_nos, ncr_nos, money_nos, number_nos = run_cohort(N, SEED, None, surveillance=False)
    stats_nos = summarize(sr_nos, ncr_nos, money_nos, number_nos, N)
    print(f'  done ({time.time()-t0:.0f}s): {stats_nos}', flush=True)

    def pct(a, b):
        return None if b == 0 else 100.0 * (a - b) / b

    result = {
        'n': N, 'seed': SEED,
        'policy': stats_pol, 'no_screen': stats_nos,
        'mortality_reduction_pct': -pct(stats_pol['crc_death_per_100k'], stats_nos['crc_death_per_100k']),
        'incidence_reduction_pct': -pct(stats_pol['incidence_per_100k'], stats_nos['incidence_per_100k']),
        'cost_increase_pct': pct(stats_pol['cost_per_person_usd'], stats_nos['cost_per_person_usd']),
    }
    out_path = os.path.join(PBVI_ROOT, 'results', 'cmost_in_loop_policy_eval.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print('\n=== SUMMARY ===')
    print(json.dumps(result, indent=2))
    print('saved', out_path)


if __name__ == '__main__':
    main()
