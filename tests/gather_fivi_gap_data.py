import os, sys, json, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore', category=RuntimeWarning)
from pomdp.model_v2 import CRCScreeningPOMDP9
from pomdp.fivi import FiVI

p = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97)
solver = FiVI(p, seed=0, max_sawtooth_points=300)
solver.solve(max_iters=200, precision=1e-3, verbose=True, time_limit=180, n_stochastic_trajectories=20)

out = os.path.join(os.path.dirname(__file__), '..', 'results', 'fivi_gap_history.json')
with open(out, 'w') as f:
    json.dump(solver.gap_history, f)
print('saved', out)
