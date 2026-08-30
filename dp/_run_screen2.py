from dp.engine_runner import run_arm, chunk_seeds
if __name__ == '__main__':
    seeds = [s + 500 for s in chunk_seeds(20)]   # distinct seeds from screen_random_q
    run_arm({'kind': 'record_screen', 'age_lo': 40, 'age_hi_first': 75, 'max_interval': 20, 'age_max': 80},
            'screen_random_q2', 1_000_000, chunk=50_000, workers=4, quarterly=True, seeds=seeds)
