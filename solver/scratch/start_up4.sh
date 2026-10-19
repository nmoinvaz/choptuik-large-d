cd /home/claude/dss
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nohup python3 scan_D2.py 31 "7.75:/home/claude/dss/solD_7.750_kc31.npy:2.245998084;8.0:/home/claude/dss/solD_8.000_kc31.npy:2.238556917" /home/claude/dss/scan_up_kc31.jsonl 30 0.25 > scan_up4.log 2>&1 &
