cd /home/claude/dss
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nohup python3 scan_D2.py 23 "8.0:/home/claude/dss/solD_8.000_kc23.npy:2.238556917;8.25:/home/claude/dss/solD_8.250_kc23.npy:2.231694473" /home/claude/dss/scan_up_kc23.jsonl 40 1.0 2.0 > scan_fast.log 2>&1 &
