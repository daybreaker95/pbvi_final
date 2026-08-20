"""4-way CMOST-in-the-loop comparison, all with surveillance OFF (symmetric,
matching the pure-13-state POMDP's own no-surveillance world model):
  no_screen  -- pure natural history, no colonoscopy at all
  q10y       -- fixed colonoscopy at ages 50, 60, 70 (100% adherence)
  q5y        -- fixed colonoscopy at ages 50,55,...,75 (100% adherence)
  policy     -- FiVI-trained POMDP policy (belief-driven WAIT/SCREEN)

All four scenarios reuse the SAME policy_hook mechanism (decide_one/obs/
step_update) already wired into NumberCrunching_policy.py, so the only
difference between them is the decision LOGIC, not the code path -- this
keeps cost accounting, colonoscopy mechanics, and state recording perfectly
consistent across all four for a fair comparison. Fixed schedules use a
trivial hook (SCREEN iff age in a fixed set) instead of the native
ScreeningTest/adherence machinery, matching the same "100% adherence, exact
ages" convention already used by the matrix-vs-CMOST validation
(transitions/compare_9state_vs_cmost.py's run_microsim).

Run one scenario per process (meant to be launched in parallel):
    python3 cmost_4way_eval.py --scenario no_screen -n 10000000
    python3 cmost_4way_eval.py --scenario q10y      -n 10000000
    python3 cmost_4way_eval.py --scenario q5y       -n 10000000
    python3 cmost_4way_eval.py --scenario policy    -n 10000000
"""
import os
import sys
import io
import json
import time
import argparse
import contextlib
import warnings

PBVI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CMOST_PY = os.path.join(PBVI_ROOT, 'cmost_engine')  # self-contained copy of the
                                                     # real-engine dependency closure
                                                     # (NumberCrunching_policy.py,
                                                     # build_natural_history_transition_matrix.py,
                                                     # NumberCrunching_100000.py, state34.py,
                                                     # settings/CMOST13.py) -- previously an
                                                     # external sibling path outside this repo
sys.path.insert(0, PBVI_ROOT)
sys.path.insert(0, CMOST_PY)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np

from pomdp.model_v2 import CRCScreeningPOMDP9, NORMAL, EARLY_POLYP, ADV_POLYP, \
    CA_STAGES, CRC_DEATH, OTHER_DEATH, DET_CA_STAGES
from pomdp.fivi import FiVI

import build_natural_history_transition_matrix as BNH
from NumberCrunching_policy import NumberCrunching_policy

MAP18TO13 = {
    0: NORMAL,
    1: EARLY_POLYP, 2: EARLY_POLYP, 3: EARLY_POLYP, 4: EARLY_POLYP,
    5: ADV_POLYP, 6: ADV_POLYP,
    7: CA_STAGES[0], 8: CA_STAGES[1], 9: CA_STAGES[2], 10: CA_STAGES[3],
    11: DET_CA_STAGES[0], 12: DET_CA_STAGES[1], 13: DET_CA_STAGES[2], 14: DET_CA_STAGES[3],
    15: CRC_DEATH, 16: OTHER_DEATH, 17: OTHER_DEATH,
}

RISK_THRESHOLD = 3.974249328225908    # CRCEngine individual_risk, top-10% cut
                                       # (must match transitions_9state_sex_risk.npz's
                                       # own risk_threshold field exactly -- this is
                                       # what SexAwareEngineHook uses to classify each
                                       # simulated individual's risk_class, and it has
                                       # to line up with whatever threshold the loaded
                                       # policy was actually TRAINED on, or risk_class
                                       # routing silently mismatches the policy)


class EngineHook13:
    def __init__(self, pomdp, solver, risk_class, seed=0):
        self.p = pomdp
        self.solver = solver
        self.risk_class = risk_class
        n = len(risk_class)
        b0 = pomdp.initial_belief()
        self.belief = np.tile(b0, (n, 1))
        self.rng = np.random.default_rng(seed + 7)
        self._cur_z = None
        self._cur_y = None

    def decide_one(self, z, y):
        self._cur_z = z
        self._cur_y = y
        return int(self.solver.best_action(y, self.belief[z]))

    def obs(self, a, s_true_18):
        z, y = self._cur_z, self._cur_y
        r = int(self.risk_class[z])
        local13 = MAP18TO13[int(s_true_18)]
        s_full = self.p.fidx(r, local13)
        age = min(y, self.p.life_max)
        M = self.p.M[age][a]
        NO = 5
        p_o = np.array([M[o][s_full, :].sum() for o in range(NO)])
        tot = p_o.sum()
        p_o = p_o / tot if tot > 1e-300 else np.full(NO, 1.0 / NO)
        return int(self.rng.choice(NO, p=p_o))

    def step_update(self, z, a, o, y):
        age = min(y, self.p.life_max)
        M = self.p.M[age][a][o]
        bn = self.belief[z] @ M
        tot = bn.sum()
        if tot > 1e-300:
            self.belief[z] = bn / tot


class SexAwareEngineHook:
    """Like EngineHook13, but routes each individual to their OWN
    sex-specific (pomdp, solver) pair instead of one pooled pair -- sex is
    an observable factor (known at t=0, unlike risk_class which stays
    latent/belief-tracked), so it's handled by a deterministic per-person
    routing split rather than by expanding the belief-state itself.
    sex_arr: 1=male, 2=female per individual, using the exact convention
    already produced by build_natural_history_transition_matrix's own
    gender_arr (rand_gender < fraction_female -> 2 else 1), which matches
    CRCScreeningPOMDP9's sex=1/2 convention exactly -- no remapping needed."""

    def __init__(self, pomdp_m, solver_m, pomdp_f, solver_f, risk_class, sex_arr, seed=0):
        self.pm, self.sm = pomdp_m, solver_m
        self.pf, self.sf = pomdp_f, solver_f
        self.risk_class = risk_class
        self.sex_arr = np.asarray(sex_arr).astype(int)
        b0_m = pomdp_m.initial_belief()
        b0_f = pomdp_f.initial_belief()
        self.belief = np.where((self.sex_arr == 1)[:, None], b0_m[None, :], b0_f[None, :])
        self.rng = np.random.default_rng(seed + 7)
        self._cur_z = None
        self._cur_y = None

    def _pair(self, z):
        return (self.pm, self.sm) if self.sex_arr[z] == 1 else (self.pf, self.sf)

    def decide_one(self, z, y):
        self._cur_z = z
        self._cur_y = y
        _, s = self._pair(z)
        return int(s.best_action(y, self.belief[z]))

    def obs(self, a, s_true_18):
        z, y = self._cur_z, self._cur_y
        p, _ = self._pair(z)
        r = int(self.risk_class[z])
        local13 = MAP18TO13[int(s_true_18)]
        s_full = p.fidx(r, local13)
        age = min(y, p.life_max)
        M = p.M[age][a]
        NO = 5
        p_o = np.array([M[o][s_full, :].sum() for o in range(NO)])
        tot = p_o.sum()
        p_o = p_o / tot if tot > 1e-300 else np.full(NO, 1.0 / NO)
        return int(self.rng.choice(NO, p=p_o))

    def step_update(self, z, a, o, y):
        p, _ = self._pair(z)
        age = min(y, p.life_max)
        M = p.M[age][a][o]
        bn = self.belief[z] @ M
        tot = bn.sum()
        if tot > 1e-300:
            self.belief[z] = bn / tot


class FixedScheduleHook:
    """SCREEN iff the current year is in a fixed age set -- 100% adherence,
    exact ages, no belief tracking needed."""

    def __init__(self, screen_ages):
        self.screen_ages = set(int(a) for a in screen_ages)

    def decide_one(self, z, y):
        return 1 if y in self.screen_ages else 0

    def obs(self, a, s_true_18):
        return 0

    def step_update(self, z, a, o, y):
        pass


def train_policy(sex=None, age_min=40, age_max=80, sex_risk_npz=None, colo_penalty_qaly=0.0):
    # PBS alone (no DBBU): confirmed via ablation to converge to the same
    # policy/clinical outcome as plain FiVI while giving the tightest final
    # gap and clearest convergence trajectory (DBBU can shift the learned
    # policy without a reliable net-benefit improvement, see
    # tests/pbs_dbbu_robustness.py -- 5/10 seeds either way).
    # sex: None (pooled), 1 (male), 2 (female) -- see SexAwareEngineHook,
    # which trains one policy per sex and routes each simulated individual
    # to their own via CMOST's own gender_arr (1=male, 2=female).
    # sex_risk_npz/colo_penalty_qaly: see CRCScreeningPOMDP9's own docstring --
    # both default to the original (top-25%-high, lambda=0) behavior.
    pomdp = CRCScreeningPOMDP9(age_min=age_min, age_max=age_max, gamma=0.97, sex=sex,
                                sex_risk_npz=sex_risk_npz, colo_penalty_qaly=colo_penalty_qaly)
    solver = FiVI(pomdp, seed=0, max_sawtooth_points=300)
    solver.solve(max_iters=50, time_limit=300, precision=1e-3, verbose=False,
                 n_stochastic_trajectories=0, use_pbs=True, dbbu_interval=1)
    return pomdp, solver


def run_cohort(p, n, seed, policy_hook, hook_age_max=80, hook_age_min=40):
    # p must come from a SINGLE np.random.seed(seed) + prepare_simulation_params(n)
    # call made by the CALLER, before any risk_class/sex_arr used to build
    # policy_hook is derived from it -- previously this function drew its
    # OWN p internally (a second, separately-seeded draw), while callers
    # separately drew a p_tmp to compute risk_class for the hook. Those two
    # draws are uncorrelated (~50% per-individual match rate, confirmed by
    # direct comparison), so the hook's risk_class/sex_arr for individual z
    # did not correspond to individual z's actual simulated risk/sex. Fixed
    # by requiring exactly one p, shared by hook construction and the sim.
    # surveillance OFF (prepare_simulation_params already leaves it off by
    # default -- kept explicit here for clarity/symmetry across all 4 runs)
    p['flag']['Polyp_Surveillance'] = False
    p['flag']['Cancer_Surveillance'] = False
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
                                      policy_hook_age_min=hook_age_min, policy_hook_age_max=hook_age_max)
    Money = out[22]
    Number = out[23]
    TumorRecord = out[12]
    DeathYear = out[4]
    return p, sr, ncr, Money, Number, TumorRecord, DeathYear


def summarize(sr, ncr, Money, Number, TumorRecord, DeathYear, n):
    win = sr[39:100]
    crc_death = (win == 15).any(axis=0)

    # Avg age at death / life-years, same convention as run_100k_benchmark.py
    # (DeathYear is a 1-indexed year count -> subtract 1 for actual age).
    # DeathYear==0 survivor-bug fix (inherited from the original MATLAB
    # CMOST_original/NumberCrunching_*.m -- confirmed present there too):
    # the engine's own post-loop patch only sets NaturalDeathYear=100 for
    # survivors, never DeathYear, leaving it at its 0 init value -- silently
    # counted as "died at age -1" by run_100k_benchmark.py's own formula.
    DeathYear_corr = np.where(DeathYear == 0, 101, DeathYear)
    avg_age_at_death = float(np.sum(DeathYear_corr) / n - 1)
    life_years = avg_age_at_death - 40.0

    # Discounted LYG/cost, 3%/year (discount factor 0.97/year -- matches
    # the POMDP's own training-time gamma=0.97, so training and reporting
    # now use the same discount convention). sr/Money are indexed by year
    # i=0..99 -> age i+1, so age=40 is index 39; discount factor at index i
    # is 0.97**(i-39) for i>=39 (age_min=40), matching how LYG/cost are
    # already measured from age 40 onward everywhere else in this script.
    DISC = 0.97
    disc_factors = DISC ** np.arange(0, 100 - 39)  # length 61, index 0 -> age 40
    alive_frac = (sr[39:100] < 15).mean(axis=1)     # fraction alive at each age 40..100
    life_years_disc = float(np.sum(alive_frac * disc_factors))
    total_cost_disc = float(np.sum(Money['AllCost'][39:100] * disc_factors))

    # "Incidence" = DIAGNOSED cases only (screening + symptoms + surveillance
    # + baseline), matching compare_model_v2_vs_cmost.csv's definition and
    # run_100k_benchmark.py's own methodology -- NOT sr's undetected-cancer
    # states (7-10), which an earlier version of this script mistakenly
    # folded in, inflating "incidence" by counting still-undiagnosed cases.
    stage = TumorRecord['Stage']
    detect = TumorRecord['Detection']
    nz = stage != 0
    total_detected = int(nz.sum())
    # detection mode codes (Evaluation.py convention): 1=screening,
    # 2=symptoms, 3=surveillance, 4=baseline
    screen_detected = int((nz & (detect == 1)).sum())
    symptom_detected = int((nz & (detect == 2)).sum())
    surveillance_detected = int((nz & (detect == 3)).sum())
    baseline_detected = int((nz & (detect == 4)).sum())

    # Stage I-IV % breakdown (stage codes 7,8,9,10 = I,II,III,IV), matching
    # compare_model_v2_vs_cmost.csv's "Stage X % (all/screen/symptom)" columns.
    def stage_pct(mask):
        vals = stage[mask]
        tot = len(vals)
        if tot == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float((vals == code).sum() / tot * 100) for code in (7, 8, 9, 10)]

    stage_pct_all = stage_pct(nz)
    stage_pct_screen = stage_pct(nz & (detect == 1))
    stage_pct_symptom = stage_pct(nz & (detect == 2))

    total_cost = float(Money['AllCost'].sum())
    return {
        'n': n,
        'crc_death_per_100k': float(crc_death.mean() * 100_000),
        'incidence_per_100k': float(total_detected / n * 100_000),
        'screen_detected_per_100k': float(screen_detected / n * 100_000),
        'symptom_detected_per_100k': float(symptom_detected / n * 100_000),
        'surveillance_detected_per_100k': float(surveillance_detected / n * 100_000),
        'baseline_detected_per_100k': float(baseline_detected / n * 100_000),
        'stage_pct_all': stage_pct_all,
        'stage_pct_screen': stage_pct_screen,
        'stage_pct_symptom': stage_pct_symptom,
        'avg_age_at_death': avg_age_at_death,
        'life_years': life_years,
        'life_years_disc3pct': life_years_disc,
        'avg_colonoscopies_per_person': float(ncr.mean()),
        'total_cost_usd': total_cost,
        'cost_per_person_usd': total_cost / n,
        'total_cost_disc3pct_usd': total_cost_disc,
        'cost_per_person_disc3pct_usd': total_cost_disc / n,
        'screening_colo_total': float(Number['Screening_Colonoscopy'].sum()),
        'followup_colo_total': float(Number['Follow_Up_Colonoscopy'].sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True, choices=['no_screen', 'q10y', 'q5y', 'policy'])
    ap.add_argument('-n', type=int, default=10_000_000)
    ap.add_argument('--seed', type=int, default=999)
    # policy-only knobs (all default to the original age_min=40/age_max=80/
    # top-25%/lambda=0 behavior -- passing none of these reproduces
    # final_algorithm/csv/02_policy_vs_schedules_final.csv exactly)
    ap.add_argument('--age-min', type=int, default=40)
    ap.add_argument('--age-max', type=int, default=80)
    ap.add_argument('--risk-npz', type=str, default=None,
                     help='override transitions_9state_sex_risk.npz path (e.g. the '
                          'top25pct backup, or the current top-10-pct file)')
    ap.add_argument('--risk-threshold', type=float, default=RISK_THRESHOLD,
                     help='must match --risk-npz\'s own risk_threshold field, or '
                          'risk_class routing silently mismatches the loaded policy')
    ap.add_argument('--lam', type=float, default=0.0, help='colo_penalty_qaly (lambda)')
    ap.add_argument('--tag', type=str, default=None,
                     help='output filename suffix, e.g. results/cmost_4way_policy_<tag>.json')
    a = ap.parse_args()

    t0 = time.time()
    # Single seeded draw of p, shared by hook construction (risk_class/
    # sex_arr) AND the actual simulation -- see run_cohort's docstring note
    # for why this must not be two separate prepare_simulation_params calls.
    np.random.seed(a.seed)
    p = BNH.prepare_simulation_params(a.n)
    risk_class = (np.asarray(p['individual_risk']) >= a.risk_threshold).astype(int)
    sex_arr = np.asarray(p['gender_arr']).astype(int)  # 1=male, 2=female

    if a.scenario == 'no_screen':
        hook = None
    elif a.scenario == 'q10y':
        hook = FixedScheduleHook([50, 60, 70])
    elif a.scenario == 'q5y':
        hook = FixedScheduleHook([50, 55, 60, 65, 70, 75])
    else:  # policy -- one FiVI policy per sex (sex is observable at t=0,
        # unlike risk_class which stays latent/belief-tracked), each with
        # its own CMOST-original sex x risk transition dynamics
        print(f'training FiVI policy (male) age={a.age_min}-{a.age_max} '
              f'risk_npz={a.risk_npz} lam={a.lam} ...', flush=True)
        pomdp_m, solver_m = train_policy(sex=1, age_min=a.age_min, age_max=a.age_max,
                                          sex_risk_npz=a.risk_npz, colo_penalty_qaly=a.lam)
        print(f'  male gap={solver_m.gap_history[-1]["gap"]:.4f}  ({time.time()-t0:.1f}s)', flush=True)
        print('training FiVI policy (female)...', flush=True)
        pomdp_f, solver_f = train_policy(sex=2, age_min=a.age_min, age_max=a.age_max,
                                          sex_risk_npz=a.risk_npz, colo_penalty_qaly=a.lam)
        print(f'  female gap={solver_f.gap_history[-1]["gap"]:.4f}  ({time.time()-t0:.1f}s)', flush=True)
        hook = SexAwareEngineHook(pomdp_m, solver_m, pomdp_f, solver_f, risk_class, sex_arr, seed=a.seed)

    print(f'[{a.scenario}] running N={a.n:,} ...', flush=True)
    t1 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = run_cohort(
        p, a.n, a.seed, hook, hook_age_max=a.age_max, hook_age_min=a.age_min)
    stats = summarize(sr, ncr, money, number, tumor_record, death_year, a.n)
    print(f'[{a.scenario}] done ({time.time()-t1:.0f}s): {stats}', flush=True)

    suffix = f'_{a.tag}' if a.tag else ''
    out_path = os.path.join(PBVI_ROOT, 'results', f'cmost_4way_{a.scenario}{suffix}.json')
    with open(out_path, 'w') as f:
        json.dump({'scenario': a.scenario, 'n': a.n, 'seed': a.seed,
                    'age_min': a.age_min, 'age_max': a.age_max,
                    'risk_npz': a.risk_npz, 'risk_threshold': a.risk_threshold, 'lam': a.lam,
                    'elapsed_sec': time.time() - t0, **stats}, f, indent=2)
    print('saved', out_path)


if __name__ == '__main__':
    main()
