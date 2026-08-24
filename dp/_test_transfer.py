import os, json, time, numpy as np
os.environ.setdefault('OMP_NUM_THREADS','2')
from dp.model import ReducedPOMDP, evaluate_policy, evaluate_fixed
from dp.solver import PBVISolver
from dp.engine_runner import run_arm, aggregate, efficiency
kz='results/dp/kernels_c3.npz'
if __name__ == '__main__':
    paths={}
    for sex in (1,2):
        p=f'results/dp/policies/test_death_lam0.002_sex{sex}.npz'
        if not os.path.exists(p):
            m=ReducedPOMDP(sex, kz, weights=dict(death=1.0), lam=0.002)
            pol=PBVISolver(m, cap=600, seed=0, verbose=False).solve(rounds=4, rollouts=150)
            ev=pol.meta['eval']; print(f'sex{sex} in-model: death={ev["death"]*1e5:.0f} inc={ev["inc"]*1e5:.0f} colos={ev["colos"]:.3f}', flush=True)
            pol.save(p)
        paths[sex]=p
    arm={'kind':'policy','policy_male':paths[1],'policy_female':paths[2],'observed_class':False}
    t0=time.time()
    pp=run_arm(arm,'test_policy_lam0.002',100000,chunk=50000,workers=2)
    pn=run_arm({'kind':'none'},'none',100000,chunk=50000,workers=2)   # shares cache with baseline arm 'none'
    pq=run_arm({'kind':'fixed','ages':[50,60,70]},'q10y',100000,chunk=50000,workers=2)
    rn=aggregate(pn); rp=aggregate(pp); rq=aggregate(pq)
    for nm,r in (('none',rn),('q10y',rq),('policy',rp)):
        e=efficiency(r,rn) if nm!='none' else {}
        print(f'{nm:7s} colos={r["colos_per_person"]:.3f} death={r["crc_death_per_100k"]:.0f} inc={r["incidence_per_100k"]:.0f} LY={r["life_years_from40"]:.2f} comp={r["comp_death_per_100k"]:.1f}', {k:round(v,3) for k,v in e.items()})
