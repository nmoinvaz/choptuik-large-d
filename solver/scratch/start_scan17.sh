cd /home/claude/dss
NOGAMMA=1 nohup nice -n 5 python3 scan_long4.py scan_longB.jsonl 40 0.3 0.4 "28.762321400000005:longB_D28.7623.npy:23;29.022321400000006:longB_D29.0223.npy:23;29.360321400000007:longB_D29.3603.npy:23" > scan_long17.log 2>&1 &
