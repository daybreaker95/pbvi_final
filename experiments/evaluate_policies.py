"""
evaluate_policies.py -- evaluate screening policies in the true CMOST gym env.

Uses per-patient common random numbers (the engine RNG is reseeded per patient
index) so that the same simulated individual is faced by every policy up to the
point where a screening action diverges the random stream -- a strong variance
reduction for policy contrasts.

Outcomes reported (per person, averaged over the population):
  life_years         mean age at death
  ly_lost_to_crc     mean life-years lost to CRC / colonoscopy death
  crc_incidence      fraction ever clinically diagnosed with CRC
  crc_mortality      fraction dying of CRC
  n_colonoscopies    mean number of screening colonoscopies used
Efficiency metrics (vs no-screening) are added by compare().
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.params import build_params
from env.cmost_individual import CRCEngine
from env.crc_env import CRCScreeningEnv, EnvConfig, NoScreening


def evaluate(policy, n=20000, seed=1000, settings='CMOST13', config: EnvConfig = None):
    """Roll out `policy` over n paired patients. Returns aggregate outcomes."""
    config = config or EnvConfig()
    params = build_params(settings, n_patients=500, seed=777)
    eng = CRCEngine(params, rng=np.random.default_rng(0))
    env = CRCScreeningEnv(eng, config)

    rows = []
    for i in range(n):
        eng.rng = np.random.default_rng(seed + i)   # common random numbers
        rows.append(env.rollout(policy))
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    out = {
        'n': n,
        'life_years': arr['life_years'].mean(),
        'life_years_se': arr['life_years'].std(ddof=1) / np.sqrt(n),
        'ly_lost_to_crc': arr['ly_lost_to_crc'].mean(),
        'ly_lost_to_crc_se': arr['ly_lost_to_crc'].std(ddof=1) / np.sqrt(n),
        'crc_incidence': 100 * arr['ever_clinical'].mean(),
        'crc_mortality': 100 * arr['crc_death'].mean(),
        'n_colonoscopies': arr['n_colonoscopies'].mean(),
        'colo_deaths': int(arr['colo_death'].sum()),
        '_raw': arr,
    }
    return out


def compare(policies: dict, n=20000, seed=1000, settings='CMOST13',
            config: EnvConfig = None):
    """Evaluate a dict {name: policy}; the entry named 'No screening' (or the
    first one) is the reference for life-years-gained (LYG)."""
    config = config or EnvConfig()
    results = {}
    for name, pol in policies.items():
        results[name] = evaluate(pol, n=n, seed=seed, settings=settings, config=config)

    ref_name = 'No screening' if 'No screening' in results else list(results)[0]
    ref_lost = results[ref_name]['ly_lost_to_crc']
    ref_inc = results[ref_name]['crc_incidence']
    ref_mort = results[ref_name]['crc_mortality']

    for name, r in results.items():
        # LYG (per person) vs reference = reduction in life-years lost to CRC
        r['LYG'] = ref_lost - r['ly_lost_to_crc']
        r['LYG_per_1000'] = 1000 * r['LYG']
        r['inc_reduction'] = ref_inc - r['crc_incidence']
        r['mort_reduction'] = ref_mort - r['crc_mortality']
        nc = r['n_colonoscopies']
        r['LYG_per_colo'] = (1000 * r['LYG'] / nc) if nc > 1e-9 else 0.0
        r['mortred_per_colo'] = (r['mort_reduction'] / nc) if nc > 1e-9 else 0.0
    return results, ref_name


def print_table(results: dict, ref_name: str):
    hdr = (f"{'policy':<26}{'colo':>6}{'life-yrs':>10}{'CRCinc%':>9}"
           f"{'CRCmort%':>9}{'LYG/1k':>8}{'LYG/colo':>9}")
    print(hdr)
    print('-' * len(hdr))
    for name, r in results.items():
        print(f"{name:<26}{r['n_colonoscopies']:>6.2f}{r['life_years']:>10.3f}"
              f"{r['crc_incidence']:>9.2f}{r['crc_mortality']:>9.2f}"
              f"{r['LYG_per_1000']:>8.1f}{r['LYG_per_colo']:>9.2f}")


if __name__ == '__main__':
    from env.crc_env import UniformInterval, FixedAgeSchedule
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    cfg = EnvConfig(start_age=20, stop_age=90, budget=None, discount=0.0)
    pols = {
        'No screening': NoScreening(),
        'Colo q10y 50-70': FixedAgeSchedule([50, 60, 70]),
        'Colo q10y 45-75': FixedAgeSchedule([45, 55, 65, 75]),
        'Zaika 1x @55': FixedAgeSchedule([55]),
        'Zaika 2x @49,64': FixedAgeSchedule([49, 64]),
    }
    print(f"Evaluating {len(pols)} policies on n={n} paired patients ...\n")
    results, ref = compare(pols, n=n, seed=2024, config=cfg)
    print_table(results, ref)
