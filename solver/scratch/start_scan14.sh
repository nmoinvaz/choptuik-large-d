cd /home/claude/dss
NOGAMMA=1 nohup nice -n 5 python3 scan_long3.py scan_longB.jsonl 32 0.25 0.4 "26.892321400000007:longB_D26.8923.npy:23;27.192321400000008:longB_D27.1923.npy:23;27.38732140000001:longB_D27.3873.npy:23" > scan_long14.log 2>&1 &
