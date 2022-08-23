import multiprocessing
import os

def sh_script(idx):
    os.system('bash train/plot_congestion.sh {}'.format(idx))

pool = multiprocessing.Pool(5)
for ckpt in range(100):
    pool.apply_async(sh_script, (ckpt,))
pool.close()
pool.join()