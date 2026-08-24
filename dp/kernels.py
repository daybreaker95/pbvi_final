"""Observed "memory" component of the decision model and the kernel cell
indexing shared by the estimator (dp.estimate_kernels), the model
(dp.model) and the engine hook (dp.hooks).

Mixed-observability structure (MOMDP):
  observed : age y, tau = years since the last colonoscopy (never / 0 / 1 / ...),
             ol  = finding at the last colonoscopy (normal / adenoma / multi / advad)
  hidden   : risk class x clinical state (belief)

tau = 0 denotes the natural-history year immediately following a colonoscopy
(the WAIT kernel applied after SCREEN). At a decision epoch tau >= 1 or never.

Why: the real engine's post-polypectomy dynamics are not those of the
never-screened cross-section -- new polyps arise faster (within-class risk
heterogeneity) and progress faster (the unscreened cross-section is enriched
for slow-growing polyps). Conditioning the kernels on (tau, last finding),
both observed by any screening programme, captures this memory without
enlarging the hidden state.
"""
from __future__ import annotations

import numpy as np

from .common import O_NORMAL, O_ADENOMA, O_MULTI, O_ADVAD, SCREEN_OBS

TAU_NEVER = -1
TAU_CAP = 13                      # exact tau tracked up to 13 (13 = "13 or more")

# tau groups used for kernel estimation (index 0 = never)
TAU_GROUPS = [(TAU_NEVER, TAU_NEVER), (0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 999)]
N_TAU_G = len(TAU_GROUPS)
N_OL = 4                          # last finding: normal / adenoma / multi / advad
OL_OF_OBS = {o: i for i, o in enumerate(SCREEN_OBS)}      # observation -> ol index
OL_NAMES = ['normal', 'adenoma', 'multi', 'advad']
N_MEM = 1 + (N_TAU_G - 1) * N_OL  # 0 = never, else 1 + (g-1)*N_OL + ol


def tau_group(tau):
    """tau (int, -1 = never) -> group index."""
    tau = np.asarray(tau)
    g = np.zeros(tau.shape, dtype=np.int8)
    for i, (lo, hi) in enumerate(TAU_GROUPS):
        g[(tau >= lo) & (tau <= hi)] = i
    return g


def mem_index(tau, ol):
    """(tau, ol) -> memory cell index (vectorised)."""
    tau = np.asarray(tau); ol = np.asarray(ol)
    g = tau_group(tau).astype(np.int64)
    m = np.where(g == 0, 0, 1 + (g - 1) * N_OL + ol)
    return m.astype(np.int64)


def mem_index_scalar(tau: int, ol: int) -> int:
    return int(mem_index(np.array([tau]), np.array([ol]))[0])


def mem_name(m: int) -> str:
    if m == 0:
        return 'never'
    g = (m - 1) // N_OL + 1
    ol = (m - 1) % N_OL
    lo, hi = TAU_GROUPS[g]
    return f'tau{lo}' + (f'-{hi}' if hi != lo and hi < 999 else ('+' if hi >= 999 else '')) + f'/{OL_NAMES[ol]}'


def next_tau_wait(tau: int) -> int:
    if tau == TAU_NEVER:
        return TAU_NEVER
    return min(tau + 1, TAU_CAP)


def mem_cells_after_screen():
    """Memory cells a natural-history year can be in right after a screen."""
    return [mem_index_scalar(0, ol) for ol in range(N_OL)]
