from dp.engine_runner import run_arm
if __name__ == '__main__':
    run_arm({'kind':'record_screen','age_lo':40,'age_hi_first':75,'max_interval':20,'age_max':80}, 'screen_dbg_q', 40000, chunk=20000, workers=2, quarterly=True)
