"""
smoke_engine.py -- sanity-check the individual engine reproduces CMOST's
natural history (no screening) before we build anything on top of it.

Reports lifetime CRC risk, cancer-death rate, mean age at death, clinical
stage distribution and adenoma prevalence, next to CMOST13 reference values.
"""

import os
import sys
import time as _time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.params import build_params
from env.cmost_individual import (
    CRCEngine, NORMAL, EARLY_ADENOMA, ADVANCED_ADENOMA,
    EARLY_CANCER, ADVANCED_CANCER, DIAGNOSED, DEAD_OTHER, STATE_NAMES,
)


def main(n=20000, seed=1):
    print(f"Building CMOST13 parameters (pool seed={seed}) ...")
    params = build_params('CMOST13', n_patients=500, seed=seed)
    rng = np.random.default_rng(seed)
    eng = CRCEngine(params, rng=rng)

    ever_cancer = 0
    ever_clinical = 0
    ca_death = 0
    other_death = 0
    colo_death = 0
    ages_at_death = []
    clinical_stage_counts = {7: 0, 8: 0, 9: 0, 10: 0}
    # adenoma prevalence snapshots at selected ages
    snap_ages = [50, 60, 65, 70]
    prev = {a: {ADVANCED_ADENOMA: 0, EARLY_ADENOMA: 0, EARLY_CANCER: 0, ADVANCED_CANCER: 0,
                'alive': 0} for a in snap_ages}

    t0 = _time.time()
    for i in range(n):
        pt = eng.new_patient()
        had_cancer = False
        for y in range(1, 101):
            eng.step_year(pt, y, screen=False)
            if y in snap_ages and pt.alive:
                st = eng.clinical_state(pt)
                prev[y]['alive'] += 1
                if st in (ADVANCED_ADENOMA, EARLY_ADENOMA, EARLY_CANCER, ADVANCED_CANCER):
                    prev[y][st] += 1
            if pt.ca_stage or pt.ever_clinical:
                had_cancer = True
            if not pt.alive:
                break
        if had_cancer:
            ever_cancer += 1
        if pt.ever_clinical:
            ever_clinical += 1
            # record stage at (first) clinical diagnosis
            if pt.det_stage:
                clinical_stage_counts[int(pt.det_stage[0])] += 1
        if pt.death_cause == 2:
            ca_death += 1
        elif pt.death_cause == 1:
            other_death += 1
        elif pt.death_cause == 3:
            colo_death += 1
        ages_at_death.append(pt.death_time if pt.death_time > 0 else 100)

    dt = _time.time() - t0
    print(f"Simulated {n} patients in {dt:.1f}s ({1000*dt/n:.2f} ms/patient)\n")

    print("=" * 60)
    print("NATURAL-HISTORY SUMMARY (no screening)")
    print("=" * 60)
    print(f"  Lifetime CRC (pre/clinical ever):  {100*ever_cancer/n:5.2f} %   "
          f"[US ref ~4.3-5%]")
    print(f"  Lifetime clinically diagnosed CRC: {100*ever_clinical/n:5.2f} %")
    print(f"  Died of CRC:                       {100*ca_death/n:5.2f} %   "
          f"[US ref ~1.8-2%]")
    print(f"  Died of other causes:              {100*other_death/n:5.2f} %")
    print(f"  Colonoscopy-complication deaths:   {colo_death}")
    print(f"  Mean age at death:                 {np.mean(ages_at_death):5.2f}  "
          f"[CMOST13 ref ~74.5]")

    tot = sum(clinical_stage_counts.values())
    print("\n  Stage distribution at clinical diagnosis:")
    for s in (7, 8, 9, 10):
        lbl = ['I', 'II', 'III', 'IV'][s - 7]
        pct = 100 * clinical_stage_counts[s] / tot if tot else 0
        print(f"    Stage {lbl:<3} {clinical_stage_counts[s]:5d}  ({pct:4.1f} %)")

    print("\n  Prevalence among the living (adenoma / undiagnosed cancer):")
    print(f"    {'age':>4} {'earlyAd%':>9} {'advAd%':>8} {'earlyCa%':>9} {'advCa%':>8}")
    for a in snap_ages:
        al = prev[a]['alive']
        if al == 0:
            continue
        ea = 100 * prev[a][EARLY_ADENOMA] / al
        aa = 100 * prev[a][ADVANCED_ADENOMA] / al
        ec = 100 * prev[a][EARLY_CANCER] / al
        ac = 100 * prev[a][ADVANCED_CANCER] / al
        print(f"    {a:>4} {ea:>9.2f} {aa:>8.2f} {ec:>9.2f} {ac:>8.2f}")

    print("\n  (Adenoma prevalence ref: advanced adenoma ~5-9% at 50-65;")
    print("   any adenoma ~25-30% at 50-65 in screening cohorts)")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    main(n=n)
