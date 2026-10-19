cd /home/claude/dss
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nohup sh -c 'python3 point3.py 6.5 31 solD_6.500_kc31.npy 39 2.2957835 res_kc.jsonl; python3 point3.py 6.5 31 solD_6.500_kc31.npy 23 2.2957835 res_kc.jsonl' > conv.log 2>&1 &
