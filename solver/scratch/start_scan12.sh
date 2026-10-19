cd /home/claude/dss
NOGAMMA=1 nohup nice -n 5 python3 scan_long3.py scan_longB.jsonl 32 0.4 0.6 "23.728037500000006:longB_D23.7280.npy:23;24.235037500000008:longB_D24.2350.npy:23;24.53503750000001:longB_D24.5350.npy:23" > scan_long12.log 2>&1 &
