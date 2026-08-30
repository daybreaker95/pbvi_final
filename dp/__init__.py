"""dp -- dynamic-programming colonoscopy policy pipeline grounded in the real
CMOST engine (see docs/DP_PLAN.md).

Modules
-------
engine_runner   chunked, paired-seed, parallel runs of cmost_engine/NumberCrunching_policy
hooks           engine policy hooks (fixed schedules, recording hooks, belief-policy hook)
estimate_nh     natural-history (WAIT) kernels per sex x risk class from a quarterly-recorded cohort
estimate_screen empirical colonoscopy (SCREEN) kernel from a randomised-schedule cohort
model           reduced finite-horizon POMDP with deaths / incidence / life-year objectives
solver          point-based value iteration on reachable beliefs + exact in-model evaluation
fixed_search    exhaustive in-model search over fixed schedules
sweep           lambda sweep -> in-model efficiency frontier
evaluate        real-engine evaluation of arms with paired CIs
"""
