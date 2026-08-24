from dp.engine_runner import run_arm
if __name__ == '__main__':
    run_arm({'kind':'fixed','ages':[50,60,70]}, 'q10y_dbg', 40000, chunk=20000, workers=2, quarterly=True)
