cd /home/claude/dss
NOGAMMA=1 nohup nice -n 5 python3 scan_long3.py scan_longB.jsonl 32 0.2 0.4 "27.962321400000008:longB_D27.9623.npy:23;28.362321400000006:longB_D28.3623.npy:23;28.562321400000005:longB_D28.5623.npy:23" > scan_long15.log 2>&1 &
