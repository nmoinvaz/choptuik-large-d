cd /home/claude/dss
OPENBLAS_NUM_THREADS=1 nohup nice -n 5 python3 scan_profile.py scan_prof.jsonl 40 0.3 0.5 "30.51032140000001:longB_D30.5103.npy;30.84832140000001:longB_D30.8483.npy;31.04832140000001:longB_D31.0483.npy;31.30832140000001:longB_D31.3083.npy" > scan_prof1.log 2>&1 &
