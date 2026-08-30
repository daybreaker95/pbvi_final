import json, os
from dp.sweep import pooled_rows
from dp.run_pipeline import step_engine, arm_policy, FRAC_FEMALE
from dp.common import RES
if __name__ == '__main__':
    sw = json.load(open(os.path.join(RES, 'sweep_c6b_death.json')))
    rows = pooled_rows(sw['rows'], FRAC_FEMALE)
    arms = {'none': {'kind': 'none'}, 'q10y': {'kind': 'fixed', 'ages': [50, 60, 70]},
            'q5y': {'kind': 'fixed', 'ages': [50, 55, 60, 65, 70, 75]}}
    for r in rows:
        arms[f'c6b_death_lam{r["lam"]:.6g}'] = arm_policy(r)
    step_engine(arms, 200_000, 4, 'grid_c6b_death')
