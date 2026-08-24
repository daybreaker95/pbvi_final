"""Simulate the two estimation cohorts in the real engine (resumable).

  nh      : no screening, quarterly 18-state recorder  -> WAIT kernels
  screen  : randomised colonoscopy schedules + decision log -> SCREEN kernel

python -m dp.run_cohorts --nh-n 2000000 --screen-n 1000000 --chunk 50000 --workers 6
"""
from __future__ import annotations

import argparse
import time

from .engine_runner import run_arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nh-n', type=int, default=2_000_000)
    ap.add_argument('--screen-n', type=int, default=1_000_000)
    ap.add_argument('--chunk', type=int, default=50_000)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--only', choices=['nh', 'screen'], default=None)
    a = ap.parse_args()
    t0 = time.time()
    if a.only in (None, 'nh'):
        print(f'== natural-history cohort n={a.nh_n:,}', flush=True)
        run_arm({'kind': 'none'}, 'nh_quarterly', a.nh_n, chunk=a.chunk, workers=a.workers,
                quarterly=True, keep_state_recorder=False)
        print(f'   done ({time.time() - t0:.0f}s)', flush=True)
    if a.only in (None, 'screen'):
        print(f'== randomised-screening cohort n={a.screen_n:,}', flush=True)
        # quarterly recording + long intervals so that WAIT kernels conditional
        # on (years since last colonoscopy, last finding) can be estimated
        run_arm({'kind': 'record_screen', 'age_lo': 40, 'age_hi_first': 75, 'max_interval': 20, 'age_max': 80},
                'screen_random_q', a.screen_n, chunk=a.chunk, workers=a.workers, quarterly=True)
        print(f'   done ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
