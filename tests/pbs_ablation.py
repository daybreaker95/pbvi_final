"""Controlled PBS on/off ablation: everything else held fixed (same
max_iters=39 fixed pass count, same n_stochastic_trajectories=0, same
dbbu_interval=1/exact, same wtp=50000, same seed) -- only use_pbs is
toggled. precision is set unreachably tight so neither run stops early on
the gap criterion; both always run the full 39 iterations, giving a clean
per-iteration gap/vl/vu trajectory to compare.

This isolates PBS's effect on the gap TRAJECTORY (shape/rate of v_l's
non-monotonic wobble) from the earlier, confounded comparison (which also
changed n_stochastic_trajectories 0->5 at the same time)."""
import os
import sys
import json
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore', category=RuntimeWarning)

from pomdp.model_v2 import CRCScreeningPOMDP9
from pomdp.fivi import FiVI

MAX_ITERS = 39


def run(use_pbs):
    pomdp = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97)
    solver = FiVI(pomdp, seed=0, max_sawtooth_points=300)
    solver.solve(max_iters=MAX_ITERS, time_limit=None, precision=1e-9, verbose=False,
                 n_stochastic_trajectories=0, use_pbs=use_pbs, dbbu_interval=1)
    return solver.gap_history


def main():
    print('running use_pbs=False (matches original report run) ...', flush=True)
    hist_no_pbs = run(False)
    print(f'  {len(hist_no_pbs)} iters, final gap={hist_no_pbs[-1]["gap"]:.4f}', flush=True)

    print('running use_pbs=True (only PBS toggled, nothing else) ...', flush=True)
    hist_pbs = run(True)
    print(f'  {len(hist_pbs)} iters, final gap={hist_pbs[-1]["gap"]:.4f}', flush=True)

    print('\niter  gap(no_pbs)  gap(pbs)   vl(no_pbs)  vl(pbs)    vu(no_pbs)  vu(pbs)')
    for a, b in zip(hist_no_pbs, hist_pbs):
        print(f"{a['iter']:>4}  {a['gap']:>10.5f}  {b['gap']:>8.5f}   "
              f"{a['vl']:>9.5f}  {b['vl']:>8.5f}   {a['vu']:>9.5f}  {b['vu']:>8.5f}")

    out = {'no_pbs': hist_no_pbs, 'pbs': hist_pbs, 'max_iters': MAX_ITERS}
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'pbs_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('\nsaved', out_path)


if __name__ == '__main__':
    main()
