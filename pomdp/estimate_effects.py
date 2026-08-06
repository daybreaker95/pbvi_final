"""
estimate_effects.py
===================

Empirically estimate the screening-related quantities the POMDP needs, from the
same CMOST engine:

  * per-macro-state colonoscopy DETECTION probabilities
        d_EA, d_AA, d_PC
    obtained from the empirical distribution of the underlying lesion stage
    within each macro-state combined with CMOST's Colo_Detection sensitivity
    and mean location detection.

  * the life-year VALUE of a clinically diagnosed cancer, separately for
    screen-detected and symptom-detected cancers:
        V_clin_screen, V_clin_symp
    = mean (optionally discounted) remaining life-years after diagnosis.
    Early (screen) detection yields more remaining life-years; this is the
    mechanism through which screening produces benefit in the POMDP.

  * per-age remaining life expectancy for transient (cancer-free) patients,
    used only to sanity-check reward scaling.

Saves results/pomdp_effects.npz.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.params import build_params
from env.cmost_individual import CRCEngine
from env.crc_env import CRCScreeningEnv, EnvConfig, UniformInterval

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


def estimate_detection(settings='CMOST13', n=20000, seed=321):
    """Estimate lesion-stage distribution within each macro-state and convert
    to macro-state colonoscopy detection probabilities."""
    params = build_params(settings, n_patients=500, seed=seed)
    eng = CRCEngine(params, rng=np.random.default_rng(seed + 5))
    colo = eng.colo_detection                       # per-stage sensitivity (10,)
    mean_loc = float(np.average(eng.loc_coloDetection,
                                weights=None))       # mean location detection

    # Faithful cancer reach factor.  The CMOST engine detects a CANCER with
    #   rand < Colo_Detection[stage-1]  AND  reach_ok[loc]
    # i.e. cancers use NO Location_ColoDetection factor (unlike polyps).  We
    # therefore weight cancer (preclinical) detection by P(loc >= reach) over the
    # cancer location distribution, not by mean_loc (which wrongly lowered d_PC).
    rng2 = np.random.default_rng(seed + 11)
    M = 2_000_000
    li = (rng2.random(M) * 999).round().astype(int)
    cloc = eng.location_matrix[1, li].astype(int)          # cancer locations 1..13
    ri = (rng2.random(M) * 999).round().astype(int)
    reach = eng.colo_reach_matrix[ri].astype(int)
    R_cancer = float((cloc >= reach).mean())               # cancer reach factor

    # accumulate stage histograms within each macro-state (sampled over ages 45-80)
    ea_hist = np.zeros(6)     # polyp stages 1..6 (index 0..5)
    aa_hist = np.zeros(6)
    pc_hist = np.zeros(4)     # cancer stages 7..10
    for i in range(n):
        pt = eng.new_patient()
        for y in range(1, 86):
            eng.step_year(pt, y, screen=False)
            if not pt.alive:
                break
            if 45 <= y <= 80:
                mps = pt.max_polyp_stage()
                if pt.ca_stage:
                    pc_hist[max(pt.ca_stage) - 7] += 1
                elif mps >= 5:
                    aa_hist[mps - 1] += 1
                elif mps >= 1:
                    ea_hist[mps - 1] += 1

    def det_from_hist(hist, stage_offset, loc_factor):
        tot = hist.sum()
        if tot == 0:
            return 0.0
        p = hist / tot
        # detection = sum_stage p(stage) * Colo_Detection(stage) * loc_factor
        d = 0.0
        for k, pk in enumerate(p):
            stage_idx = stage_offset + k
            d += pk * colo[stage_idx] * loc_factor
        return float(d)

    # polyps use the mean Location_ColoDetection factor; cancers use the reach
    # factor only (faithful to the engine's per-lesion detection lines).
    d_ea = det_from_hist(ea_hist, 0, mean_loc)   # stages 1..6 -> colo idx 0..5 (EA uses 1-4)
    d_aa = det_from_hist(aa_hist, 0, mean_loc)
    d_pc = det_from_hist(pc_hist, 6, R_cancer)   # cancer stages 7..10 -> colo idx 6..9
    # stage-specific preclinical detection (used by the validation cohort)
    d_pc_stage = [float(colo[6 + k] * R_cancer) for k in range(4)]
    return {
        'd_EA': d_ea, 'd_AA': d_aa, 'd_PC': d_pc,
        'd_PC_stage': d_pc_stage,
        'mean_loc_detection': mean_loc, 'cancer_reach_factor': R_cancer,
        'ea_stage_dist': (ea_hist / ea_hist.sum()).tolist() if ea_hist.sum() else [],
        'aa_stage_dist': (aa_hist / aa_hist.sum()).tolist() if aa_hist.sum() else [],
        'pc_stage_dist': (pc_hist / pc_hist.sum()).tolist() if pc_hist.sum() else [],
    }


def estimate_terminal_values(settings='CMOST13', n=40000, seed=654, discount=0.03):
    """Remaining life-years after diagnosis, by detection mode, via a screening
    cohort (uniform every-5-years colonoscopy generates screen-detected cancers).

    We track the FIRST diagnosed cancer per patient and whether it was screen-
    or symptom-detected, and the (discounted) remaining life-years to death.
    """
    params = build_params(settings, n_patients=500, seed=seed)
    eng = CRCEngine(params, rng=np.random.default_rng(seed + 7))
    eng.deterministic_natural_death = True
    cfg = EnvConfig(start_age=20, stop_age=90, budget=None, discount=0.0)
    env = CRCScreeningEnv(eng, cfg)
    pol = UniformInterval(interval=5, start=45, stop=85)

    screen_ly, symp_ly = [], []
    screen_stage, symp_stage = [], []
    for i in range(n):
        eng.rng = np.random.default_rng(seed + 100000 + i)
        env.reset()
        # capture detection mode by monitoring det_stage growth
        prev_det = 0
        mode = None
        while not env._done:
            a = pol.act(env._make_obs(0))
            # was a screening colonoscopy just requested?
            screening_now = (a == 1)
            obs, r, term, trunc, info = env.step(a)
            pt = env.pt
            if mode is None and len(pt.det_stage) > prev_det:
                # a cancer was just diagnosed this step
                mode = 'screen' if (screening_now and obs['obs'] == 4) else 'symp'
                det_age = pt.det_year[0]
                det_stg = pt.det_stage[0]
            prev_det = len(pt.det_stage)
        pt = env.pt
        if mode is not None:
            death_age = pt.death_time if pt.death_time > 0 else cfg.max_age
            rem = max(0.0, death_age - det_age)
            disc = ((1.0 + discount) ** (-(det_age - cfg.start_age))) if discount > 0 else 1.0
            val = rem * disc
            if mode == 'screen':
                screen_ly.append(val); screen_stage.append(det_stg)
            else:
                symp_ly.append(val); symp_stage.append(det_stg)

    def summ(ly, stg):
        ly = np.array(ly); stg = np.array(stg)
        return {
            'n': int(len(ly)),
            'mean_remaining_ly': float(ly.mean()) if len(ly) else 0.0,
            'stage_dist': [float(np.mean(stg == s)) for s in (7, 8, 9, 10)] if len(stg) else [],
        }

    return {'screen': summ(screen_ly, screen_stage),
            'symp': summ(symp_ly, symp_stage),
            'discount': discount}


def main():
    print("Estimating colonoscopy detection probabilities ...")
    det = estimate_detection()
    print("   ", {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in det.items() if not k.endswith('dist')})
    print("Estimating terminal cancer values (screen vs symptomatic) ...")
    term = estimate_terminal_values()
    print(f"    screen-detected: n={term['screen']['n']}, "
          f"remaining LY={term['screen']['mean_remaining_ly']:.2f}, "
          f"stages={[round(x,2) for x in term['screen']['stage_dist']]}")
    print(f"    symptom-detected: n={term['symp']['n']}, "
          f"remaining LY={term['symp']['mean_remaining_ly']:.2f}, "
          f"stages={[round(x,2) for x in term['symp']['stage_dist']]}")

    np.savez(os.path.join(RES, 'pomdp_effects.npz'),
             d_EA=det['d_EA'], d_AA=det['d_AA'], d_PC=det['d_PC'],
             d_PC_stage=np.asarray(det['d_PC_stage'], float),
             V_clin_screen=term['screen']['mean_remaining_ly'],
             V_clin_symp=term['symp']['mean_remaining_ly'],
             screen_stage_dist=term['screen']['stage_dist'],
             symp_stage_dist=term['symp']['stage_dist'],
             detail=np.array([str(det), str(term)], dtype=object))
    print(f"Saved {os.path.join(RES, 'pomdp_effects.npz')}")


if __name__ == '__main__':
    main()
