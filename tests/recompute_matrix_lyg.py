"""
recompute_matrix_lyg.py
=========================
The report's matrix-predicted DeltaLYG (+0.268yr) came from an OLDER training
run (tests/gather_fivi_full_trajectory.py's "faithful paper-as-written"
plain-FiVI, no PBS) on the PRE-tau-phase matrix. Since then the matrix's
d_symp/P_undetected tables were rebuilt (tau-phase correction) and the
active policy is now trained via cmost_4way_eval.py's PBS-based train_policy().
Before hunting for exotic explanations of the matrix-vs-real ΔLYG gap, check
the cheap thing first: does the matrix's OWN self-consistent ΔLYG prediction
even still say +0.268yr on the CURRENT setup, or has it moved?

Reuses gather_fivi_full_trajectory.simulate_fivi (fast, vectorized, matrix-
only MC -- not the slow real-CMOST engine) with the CURRENT tau-phase pomdp
and the CURRENT PBS-trained solver from cmost_4way_eval.train_policy().
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import cmost_4way_eval as C4
from gather_fivi_full_trajectory import simulate_fivi
from mc_eval_policy9 import summarize

MC_N = 2_000_000
SEED = 42

t0 = time.time()
print('training PBS policy (current tau-phase matrix, same as cmost_4way_eval)...', flush=True)
pomdp, solver = C4.train_policy()
print(f'  gap={solver.gap_history[-1]["gap"]:.4f}  ({time.time()-t0:.1f}s)', flush=True)

t0 = time.time()
r_nos = simulate_fivi(pomdp, solver, MC_N, seed=SEED, policy=False)
life_years_nos = float((r_nos['death_age'] - pomdp.age_min).mean())
s_nos = summarize(r_nos)
print(f'no-screen: life_years={life_years_nos:.4f}  crc_100k={s_nos["crc_100k"]:.1f}  '
      f'incid_100k={s_nos["incid_100k"]:.1f}  ({time.time()-t0:.0f}s)', flush=True)

t0 = time.time()
r_pol = simulate_fivi(pomdp, solver, MC_N, seed=SEED, policy=True)
life_years_pol = float((r_pol['death_age'] - pomdp.age_min).mean())
s_pol = summarize(r_pol)
print(f'policy:    life_years={life_years_pol:.4f}  crc_100k={s_pol["crc_100k"]:.1f}  '
      f'incid_100k={s_pol["incid_100k"]:.1f}  ({time.time()-t0:.0f}s)', flush=True)

print(f'\nDelta LYG (matrix, current tau-phase + PBS policy) = {life_years_pol - life_years_nos:+.4f} yr')
print(f'Delta CRC mortality = {(s_pol["crc_100k"]-s_nos["crc_100k"])/s_nos["crc_100k"]*100:+.1f}%')
print(f'Delta incidence     = {(s_pol["incid_100k"]-s_nos["incid_100k"])/s_nos["incid_100k"]*100:+.1f}%')
