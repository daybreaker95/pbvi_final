"""Chunked, paired-seed, parallel runs of the real CMOST engine.

Every "arm" (no_screen / fixed schedule / dp policy / recorder) is simulated
on the SAME sequence of chunk seeds. For a given seed the population
(individual_risk, sex, mortality permutations) and the RNG stream are
identical across arms until the first colonoscopy diverges them, so arm
differences are paired at the chunk level (and the person level for
outcomes decided before divergence).

Results of each (arm, chunk) are cached under results/dp/runs/<arm_tag>/
seed<seed>_n<n>.npz so sweeps are resumable and nothing is re-simulated.

Usage (library):
    from dp.engine_runner import run_arm
    out = run_arm({'kind': 'fixed', 'ages': [50, 60, 70]}, tag='q10y',
                  n_total=1_000_000, chunk=50_000, workers=6)
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import contextlib
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .common import (
    CMOST_PY, ROOT, RUNS, E_DEAD_CRC, E_DEAD_COMP, risk_class_of, load_settings_pool,
    risk_thresholds, DEFAULT_CLASS_CUTS,
)

BASE_SEED = 20260823


def chunk_seeds(n_chunks: int, base: int = BASE_SEED):
    return [base + 1000 * i for i in range(n_chunks)]


def _engine_args(p):
    return [p['p'], p['stage_variables'], p['location'], p['cost'], p['cost_stage'],
            p['risc'], p['flag'], p['special_text'], p['female'], p['sensitivity'],
            p['screening_test'], p['screening_preference'], p['age_progression'],
            p['new_polyp'], p['colonoscopy_likelyhood'], p['individual_risk'],
            p['risk_dist'], p['gender_arr'], p['life_table'], p['mortality_matrix'],
            p['location_matrix'], p['stage_duration'], p['tx1'],
            p['direct_cancer_rate'], p['direct_cancer_speed'], p['dwell_speed']]


def build_hook(arm: dict, p: dict, seed: int, n: int):
    """Instantiate the engine hook described by `arm` for population p."""
    from . import hooks as H
    kind = arm['kind']
    if kind == 'none':
        return None
    if kind == 'fixed':
        if not arm.get('no_dx', True):
            return H.FixedScheduleHook(arm['ages'])
        return H.FixedScheduleNoDxHook(arm['ages'], n, adherence=arm.get('adherence', 1.0),
                                       recall=arm.get('recall', False), seed=seed)
    if kind == 'record_screen':
        return H.RandomScheduleRecorderHook(n, seed, age_lo=arm.get('age_lo', 40),
                                            age_hi_first=arm.get('age_hi_first', 70),
                                            max_interval=arm.get('max_interval', 12),
                                            age_max=arm.get('age_max', 80))
    if kind == 'rule':
        return H.FindingRuleHook(n, start=arm.get('start', 52), intervals=tuple(arm.get('intervals', (10, 5, 3, 3))),
                                 age_max=arm.get('age_max', 80))
    if kind == 'policy':
        from .solver import load_policy
        pols = {1: load_policy(arm['policy_male']), 2: load_policy(arm['policy_female'])}
        sex_arr = np.asarray(p['gender_arr']).astype(int)
        rc = None
        if arm.get('observed_class', False):
            thr = np.asarray(arm['class_thr'], float)
            rc = risk_class_of(np.asarray(p['individual_risk'], float), thr)
        return H.BeliefPolicyHook(pols, sex_arr, risk_class=rc,
                                 observed_class=arm.get('observed_class', False),
                                 adherence=arm.get('adherence', 1.0), seed=seed)
    raise ValueError(f'unknown arm kind {kind}')


def run_chunk(job: dict) -> str:
    """Worker entry point. Simulates one chunk and writes the cache file.
    Returns the output path."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    for pth in (ROOT, CMOST_PY):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    import build_natural_history_transition_matrix as BNH
    from NumberCrunching_policy import NumberCrunching_policy

    arm, seed, n = job['arm'], int(job['seed']), int(job['n'])
    age_min, age_max = int(job.get('age_min', 40)), int(job.get('age_max', 80))
    out_path = job['out_path']
    if os.path.exists(out_path) and not job.get('force', False):
        return out_path
    t0 = time.time()

    np.random.seed(seed)
    p = BNH.prepare_simulation_params(n)
    p['flag']['Polyp_Surveillance'] = False
    p['flag']['Cancer_Surveillance'] = False
    hook = build_hook(arm, p, seed, n)

    sr = np.zeros((100, n), dtype=np.int8)
    ncr = np.zeros(n, dtype=np.int32)
    act = np.zeros((100, n), dtype=np.int8)
    qr = np.zeros((400, n), dtype=np.int8) if job.get('quarterly', False) else None
    with contextlib.redirect_stdout(io.StringIO()):
        out = NumberCrunching_policy(*_engine_args(p), state_recorder=sr, policy_hook=hook,
                                     action_recorder=act, n_colo_recorder=ncr,
                                     policy_hook_age_min=age_min, policy_hook_age_max=age_max,
                                     quarterly_recorder=qr)
    DeathCause = np.asarray(out[2]).astype(np.int8)
    DeathYear_f = np.asarray(out[4], float)                 # y + (q-1)/4 at death; 0 = survivor
    DeathYear = np.floor(DeathYear_f).astype(np.int16)
    TumorRecord = out[12]
    Money = out[22]
    Number = out[23]
    DiagnosedCancer = np.asarray(out[25])            # (100, n) stage code 7..10 at diagnosis year

    # ---- per-person outcomes ------------------------------------------------
    death_year = np.where(DeathYear == 0, 101, DeathYear).astype(np.int16)   # survivors -> 101
    crc_death = (DeathCause == 2)
    comp_death = (DeathCause == 3)
    # deaths at ages >= 40 (the decision window); the absorbing recorder
    # state cannot be used for this, the quarter-resolved DeathYear can
    crc_death_40 = crc_death & (DeathYear_f >= 40)
    comp_death_40 = comp_death & (DeathYear_f >= 40)
    dx_any = (DiagnosedCancer > 0)
    diagnosed = dx_any.any(axis=0)
    dx_year = np.where(diagnosed, dx_any.argmax(axis=0) + 1, 0).astype(np.int16)
    dx_stage = np.where(diagnosed, DiagnosedCancer[np.maximum(dx_year - 1, 0), np.arange(n)], 0).astype(np.int8)
    n_policy_colo = act.sum(axis=0).astype(np.int16)
    n_total_colo = ncr.astype(np.int16)
    risk = np.asarray(p['individual_risk'], float)
    sex = np.asarray(p['gender_arr']).astype(np.int8)

    stage = TumorRecord['Stage']
    detect = TumorRecord['Detection']
    nz = stage != 0
    summary = dict(
        n=n, seed=seed, arm=arm, age_min=age_min, age_max=age_max,
        crc_death=int(crc_death_40.sum()), comp_death=int(comp_death_40.sum()),
        crc_death_anyage=int(crc_death.sum()),
        diagnosed=int(diagnosed.sum()),
        screen_detected=int((nz & (detect == 1)).sum()),
        symptom_detected=int((nz & (detect == 2)).sum()),
        policy_colos=int(n_policy_colo.sum()),
        total_colos=int(n_total_colo.sum()),
        screening_colo_number=float(Number['Screening_Colonoscopy'].sum()),
        life_years_from40=float(np.sum(np.maximum(death_year - 1 - 40, 0))),
        total_cost=float(Money['AllCost'].sum()),
        elapsed_sec=time.time() - t0,
    )
    save = dict(
        summary=json.dumps(summary),
        crc_death=crc_death_40, comp_death=comp_death_40, diagnosed=diagnosed,
        dx_year=dx_year, dx_stage=dx_stage, death_year=death_year, death_time=DeathYear_f.astype(np.float32),
        n_policy_colo=n_policy_colo, n_total_colo=n_total_colo,
        risk=risk.astype(np.float32), sex=sex,
    )
    if job.get('keep_state_recorder', False):
        save['sr'] = sr
    if qr is not None:
        save['qr'] = qr
    if job.get('keep_actions', False):
        save['act'] = act
    if hasattr(hook, 'log_arrays'):
        for k, v in hook.log_arrays().items():
            save['log_' + k] = v
    if hasattr(hook, 'n_screen'):
        save['hook_n_screen'] = np.asarray(hook.n_screen)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + '.tmp.npz'
    np.savez_compressed(tmp, **save)
    os.replace(tmp, out_path)
    return out_path


def arm_dir(tag: str) -> str:
    return os.path.join(RUNS, tag)


def job_list(arm: dict, tag: str, n_total: int, chunk: int, seeds=None, **kw):
    n_chunks = int(np.ceil(n_total / chunk))
    seeds = seeds if seeds is not None else chunk_seeds(n_chunks)
    jobs = []
    for s in seeds[:n_chunks]:
        out_path = os.path.join(arm_dir(tag), f'seed{s}_n{chunk}.npz')
        jobs.append(dict(arm=arm, seed=s, n=chunk, out_path=out_path, **kw))
    return jobs


def run_jobs(jobs, workers=6, verbose=True):
    """Run jobs in a process pool (skipping cached ones). Returns paths."""
    todo = [j for j in jobs if not os.path.exists(j['out_path']) or j.get('force', False)]
    done = [j['out_path'] for j in jobs if j['out_path'] not in {t['out_path'] for t in todo}]
    if verbose:
        print(f'[runner] {len(done)} cached, {len(todo)} to run, workers={workers}', flush=True)
    if not todo:
        return [j['out_path'] for j in jobs]
    t0 = time.time()
    paths = list(done)
    if workers <= 1:
        for i, j in enumerate(todo):
            paths.append(run_chunk(j))
            if verbose:
                print(f'  [{i + 1}/{len(todo)}] {os.path.basename(j["out_path"])} ({time.time() - t0:.0f}s)', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_chunk, j): j for j in todo}
            for i, f in enumerate(as_completed(futs)):
                j = futs[f]
                try:
                    paths.append(f.result())
                except Exception as e:  # surface, keep going
                    print(f'  FAILED {j["out_path"]}: {e!r}', flush=True)
                    raise
                if verbose:
                    print(f'  [{i + 1}/{len(todo)}] {os.path.relpath(j["out_path"], RUNS)} ({time.time() - t0:.0f}s)', flush=True)
    return [j['out_path'] for j in jobs]


def run_arm(arm: dict, tag: str, n_total: int, chunk: int = 50_000, workers: int = 6,
            seeds=None, verbose=True, **kw):
    jobs = job_list(arm, tag, n_total, chunk, seeds=seeds, **kw)
    return run_jobs(jobs, workers=workers, verbose=verbose)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def load_chunk(path):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files if k != 'summary'}
    d['summary'] = json.loads(str(z['summary']))
    return d


def aggregate(paths, denominator='policy'):
    """Pool chunk files of one arm into per-100k rates with chunk-level SE.

    Recomputed from the PER-PERSON arrays (not the stored summaries) so that
    the death definition (CRC/complication deaths at ages >= 40) is identical
    across chunks written by different code revisions.
    denominator: 'policy' -> policy-initiated colonoscopies; 'total' -> all."""
    per = []
    n = 0
    for pth in paths:
        d = load_chunk(pth)
        dy = d['death_year'].astype(int)
        row = dict(
            n=len(dy),
            crc_death=int((d['crc_death'] & (dy >= 40)).sum()),
            comp_death=int((d['comp_death'] & (dy >= 40)).sum()),
            diagnosed=int(d['diagnosed'].sum()),
            policy_colos=int(d['n_policy_colo'].sum()),
            total_colos=int(d['n_total_colo'].sum()),
            life_years=float(np.maximum(dy - 1 - 40, 0).sum()),
            screen_detected=d['summary']['screen_detected'],
            symptom_detected=d['summary']['symptom_detected'],
            cost=d['summary']['total_cost'],
        )
        per.append(row)
        n += row['n']
    tot = lambda k: sum(r[k] for r in per)
    per100k = lambda k: tot(k) / n * 1e5
    colo_key = 'policy_colos' if denominator == 'policy' else 'total_colos'
    rates = dict(
        n=n, n_chunks=len(per),
        crc_death_per_100k=per100k('crc_death'),
        comp_death_per_100k=per100k('comp_death'),
        incidence_per_100k=per100k('diagnosed'),
        screen_detected_per_100k=per100k('screen_detected'),
        symptom_detected_per_100k=per100k('symptom_detected'),
        colos_per_person=tot(colo_key) / n,
        policy_colos_per_person=tot('policy_colos') / n,
        total_colos_per_person=tot('total_colos') / n,
        life_years_from40=tot('life_years') / n,
        cost_per_person=tot('cost') / n,
    )

    def se(k, scale):
        v = np.array([r[k] / r['n'] * scale for r in per])
        return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float('nan')
    rates['crc_death_se'] = se('crc_death', 1e5)
    rates['incidence_se'] = se('diagnosed', 1e5)
    rates['colos_se'] = se(colo_key, 1.0)
    rates['life_years_se'] = se('life_years', 1.0)
    return rates


def efficiency(arm_rates: dict, base_rates: dict) -> dict:
    """Reduction vs no-screening per colonoscopy (per 1000 colonoscopies)."""
    dd = base_rates['crc_death_per_100k'] - arm_rates['crc_death_per_100k']
    di = base_rates['incidence_per_100k'] - arm_rates['incidence_per_100k']
    c = arm_rates['colos_per_person']
    return dict(
        deaths_averted_per_100k=dd, cases_averted_per_100k=di,
        death_reduction_pct=100 * dd / base_rates['crc_death_per_100k'],
        incidence_reduction_pct=100 * di / base_rates['incidence_per_100k'],
        deaths_averted_per_1000_colos=(dd / 1e5 * 1e3 / c) if c > 0 else float('nan'),
        cases_averted_per_1000_colos=(di / 1e5 * 1e3 / c) if c > 0 else float('nan'),
        lyg_per_1000=(arm_rates['life_years_from40'] - base_rates['life_years_from40']) * 1e3,
    )


def paired_diff(paths_a, paths_b, key='crc_death'):
    """Chunk-paired difference (A - B) per 100k with SE over chunk pairs.
    paths must correspond to identical seeds in the same order."""
    def rows(paths):
        out = []
        for pth in paths:
            d = load_chunk(pth)
            dy = d['death_year'].astype(int)
            v = {'crc_death': int((d['crc_death'] & (dy >= 40)).sum()),
                 'diagnosed': int(d['diagnosed'].sum())}[key]
            out.append((d['summary']['seed'], v, len(dy)))
        return out
    A, B = rows(paths_a), rows(paths_b)
    assert [a[0] for a in A] == [b[0] for b in B]
    d = np.array([(a[1] - b[1]) / a[2] * 1e5 for a, b in zip(A, B)])
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan')


def class_thresholds(cuts=DEFAULT_CLASS_CUTS):
    return risk_thresholds(load_settings_pool(), cuts)
