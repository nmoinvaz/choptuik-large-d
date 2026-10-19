cd /home/claude/dss
NOGAMMA=1 nohup nice -n 5 python3 scan_long3.py scan_longB.jsonl 32 0.3 0.6 "20.146037499999998:longB_D20.1460.npy:23;20.4460375:longB_D20.4460.npy:23;20.6410375:longB_D20.6410.npy:23" > scan_long10.log 2>&1 &
