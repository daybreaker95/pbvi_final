"""Does the finer lesion axis / the memory buy anything for the POLICY (not
just for predicting fixed schedules)? Solve the same lambda on ablation
kernels and evaluate in the engine."""
import os, json
from dp.sweep import run_sweep, policy_path, pooled_rows
from dp.run_pipeline import step_engine
from dp.common import RES

LAM = 0.001561
VARIANTS = ['pooled', 'nomem', 'c1']
if __name__ == '__main__':
    arms = {'none': {'kind': 'none'}, 'q10y': {'kind': 'fixed', 'ages': [50, 60, 70]},
            'dp_death_lam0.001561_q10y': {'kind': 'policy',
                'policy_male': os.path.join(RES, 'policies', 'c6bhi_death_lam0.001561_sex1.npz'),
                'policy_female': os.path.join(RES, 'policies', 'c6bhi_death_lam0.001561_sex2.npz'),
                'observed_class': False}}
    for v in VARIANTS:
        kern = os.path.join(RES, f'kernels_abl_{v}.npz')
        run_sweep(kern, 'death', lams=[LAM], tag=f'abl{v}', workers=2, cap=600, rounds=3)
        arms[f'abl_{v}_lam{LAM:g}'] = {'kind': 'policy',
            'policy_male': policy_path('death', LAM, 1, f'abl{v}'),
            'policy_female': policy_path('death', LAM, 2, f'abl{v}'),
            'observed_class': False}
    step_engine(arms, 200_000, 4, 'ablation_policy')
