cd /home/claude/dss
NOGAMMA=1 nohup nice -n 5 python3 scan_long3.py scan_longB.jsonl 32 0.3 0.6 "19.489024999999998:longB_D19.4890.npy:23;19.846037499999998:longB_D19.8460.npy:23;20.146037499999998:step_D20.1460.npy:23" > scan_long9.log 2>&1 &
