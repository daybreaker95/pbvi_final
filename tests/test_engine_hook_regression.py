"""Regression test for the engine instrumentation.

The dp pipeline uses cmost_engine/NumberCrunching_policy.py, which adds a
quarter-resolved state recorder and a findings-carrying policy hook to the
MATLAB-ported engine. Both are record-only: with no hook attached (and with or
without the recorders attached) the instrumented engine must reproduce the
un-instrumented port cmost_engine/NumberCrunching_100000.py bit-for-bit on
the same seed -- every death cause and time, every diagnosis, every polyp
removed, every cost.

python tests/test_engine_hook_regression.py [n] [seed]
python -m pytest tests/test_engine_hook_regression.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for p in (ROOT, os.path.join(ROOT, 'cmost_engine')):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_natural_history_transition_matrix as BNH          # noqa: E402
from NumberCrunching_100000 import NumberCrunching_100000       # noqa: E402
from NumberCrunching_policy import NumberCrunching_policy       # noqa: E402
from dp.engine_runner import _engine_args                       # noqa: E402

OUT_NAMES = ['y', 'Gender', 'DeathCause', 'Last', 'DeathYear', 'NaturalDeathYear', 'DirectCancer', 'DirectCancerR',
             'DirectCancer2', 'DirectCancer2R', 'ProgressedCancer', 'ProgressedCancerR', 'TumorRecord',
             'DwellTimeProgression', 'DwellTimeFastCancer', 'HasCancer', 'NumPolyps', 'MaxPolyps', 'AllPolyps',
             'NumCancer', 'MaxCancer', 'PaymentType', 'Money', 'Number', 'EarlyPolypsRemoved', 'DiagnosedCancer',
             'AdvancedPolypsRemoved', 'YearIncluded', 'YearAlive']


def _run(fn, n, seed, **kw):
    np.random.seed(seed)
    p = BNH.prepare_simulation_params(n)
    p['flag']['Polyp_Surveillance'] = False
    p['flag']['Cancer_Surveillance'] = False
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*_engine_args(p), **kw)


def _assert_equal_outputs(a, b, label):
    diffs = []
    for i, name in enumerate(OUT_NAMES):
        x, y = a[i], b[i]
        if isinstance(x, dict):
            for k in x:
                if not np.array_equal(np.asarray(x[k]), np.asarray(y[k])):
                    diffs.append(f'{name}[{k}]')
        else:
            if not np.array_equal(np.asarray(x), np.asarray(y)):
                diffs.append(name)
    assert not diffs, f'{label}: outputs differ in {diffs}'


def test_no_hook_is_bit_identical(n=3000, seed=20260823):
    ref = _run(NumberCrunching_100000, n, seed)
    plain = _run(NumberCrunching_policy, n, seed)
    _assert_equal_outputs(ref, plain, 'instrumented engine without hook vs un-instrumented port')
    sr = np.zeros((100, n), dtype=np.int8); qr = np.zeros((400, n), dtype=np.int8)
    act = np.zeros((100, n), dtype=np.int8); ncr = np.zeros(n, dtype=np.int32)
    rec = _run(NumberCrunching_policy, n, seed, state_recorder=sr, quarterly_recorder=qr, action_recorder=act,
               n_colo_recorder=ncr, policy_hook=None)
    _assert_equal_outputs(ref, rec, 'instrumented engine with recorders attached vs un-instrumented port')
    assert act.sum() == 0 and (qr >= 0).all()
    return dict(n=n, seed=seed, crc_deaths=int((np.asarray(ref[2]) == 2).sum()),
                diagnosed=int((np.asarray(ref[25]) > 0).any(axis=0).sum()), identical=True)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260823
    print(test_no_hook_is_bit_identical(n, seed))
