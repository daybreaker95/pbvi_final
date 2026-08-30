"""Shared constants / paths for the dp pipeline."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CMOST_PY = os.path.join(ROOT, 'cmost_engine')
RES = os.path.join(ROOT, 'results', 'dp')
RUNS = os.path.join(RES, 'runs')
for _d in (RES, RUNS):
    os.makedirs(_d, exist_ok=True)

for _p in (ROOT, CMOST_PY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Engine 18-state codes (cmost_engine/NumberCrunching_policy.py _pomdp_state_idx)
#   0 Normal, 1-6 P1..P6 (max polyp stage), 7-10 U1..U4 (undetected cancer),
#   11-14 D1..D4 (diagnosed cancer), 15 Dead_CRC, 16 Dead_Comp, 17 Dead_Other
# ---------------------------------------------------------------------------
E_NORMAL = 0
E_P = (1, 2, 3, 4, 5, 6)
E_U = (7, 8, 9, 10)
E_D = (11, 12, 13, 14)
E_DEAD_CRC, E_DEAD_COMP, E_DEAD_OTHER = 15, 16, 17

# ---------------------------------------------------------------------------
# Reduced clinical axis used by the POMDP (11 "alive & undiagnosed" states):
#   0 N (no lesion), 1-6 P1..P6 (CMOST polyp stage of the most advanced polyp;
#   P1-4 = early adenoma, P5-6 = advanced adenoma), 7-10 U1..U4 (undetected
#   cancer stage I-IV). Keeping CMOST's own 6 polyp stages (instead of pooling
#   them into early/advanced) preserves the progression "memory" that
#   determines how much a polypectomy prevents -- pooling under-predicted the
#   engine's q10y incidence reduction by ~10 percentage points.
# Exits (terminal for the decision process; value folded into reward):
#   diagnosis at stage k (k=0..3), other-cause death, colonoscopy-complication death
# ---------------------------------------------------------------------------
C_N = 0
C_P = (1, 2, 3, 4, 5, 6)
C_U1, C_U2, C_U3, C_U4 = 7, 8, 9, 10
NC = 11
CLIN_NAMES = ['N', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'U1', 'U2', 'U3', 'U4']
X_D1, X_D2, X_D3, X_D4, X_DOTH, X_DCOMP = range(6)   # exit codes
NX = 6
EXIT_NAMES = ['D1', 'D2', 'D3', 'D4', 'DeadOther', 'DeadComp']

MAP18_TO_CLIN = np.full(18, -1, dtype=np.int8)
MAP18_TO_CLIN[E_NORMAL] = C_N
for s in C_P:
    MAP18_TO_CLIN[s] = s
for k, s in enumerate(E_U):
    MAP18_TO_CLIN[s] = C_U1 + k

MAP18_TO_EXIT = np.full(18, -1, dtype=np.int8)
for k, s in enumerate(E_D):
    MAP18_TO_EXIT[s] = X_D1 + k
MAP18_TO_EXIT[E_DEAD_OTHER] = X_DOTH
MAP18_TO_EXIT[E_DEAD_COMP] = X_DCOMP
MAP18_TO_EXIT[E_DEAD_CRC] = X_DOTH   # never reachable without prior diagnosis; guard only

# observations emitted by the engine-facing hook
#   notest   : WAIT year passed without diagnosis
#   normal   : colonoscopy found nothing
#   adenoma  : 1-2 early (stage 1-4) polyps removed, no advanced polyp
#   multi    : >=3 early polyps removed, no advanced polyp
#   advad    : >=1 advanced (stage 5-6) polyp removed
#   exit     : diagnosis (symptomatic or at colonoscopy) / complication death
O_NOTEST, O_NORMAL, O_ADENOMA, O_MULTI, O_ADVAD, O_EXIT = range(6)
NO = 6
SCREEN_OBS = (O_NORMAL, O_ADENOMA, O_MULTI, O_ADVAD)      # non-exit colonoscopy outcomes
OBS_NAMES = ['notest', 'normal', 'adenoma', 'multi', 'advad', 'exit']

WAIT, SCREEN = 0, 1

# default risk-class definition: quantile cuts of CMOST's individual_risk pool
DEFAULT_CLASS_CUTS = (0.5, 0.9)    # -> 3 classes: <50%, 50-90%, >=90%


def risk_thresholds(individual_risk_pool: np.ndarray, cuts=DEFAULT_CLASS_CUTS):
    """Quantile thresholds of the (population-weighted) individual_risk
    distribution. Returns an increasing array of thresholds; class =
    np.searchsorted(thr, risk, side='right')."""
    pool = np.asarray(individual_risk_pool, float)
    return np.array([np.quantile(pool, c) for c in cuts], float)


def risk_class_of(individual_risk: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(thr, float), np.asarray(individual_risk, float), side='right').astype(np.int8)


def load_settings_pool():
    """CMOST13 individual_risk source pool (the 500-ish value array the
    engine samples from). Used to define class thresholds once, globally,
    instead of per chunk."""
    import build_natural_history_transition_matrix as BNH
    variables = BNH.load_cmost13()
    return np.asarray(variables['IndividualRisk'], float)
