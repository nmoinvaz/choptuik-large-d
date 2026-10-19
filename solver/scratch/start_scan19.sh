cd /home/claude/dss
NOGAMMA=1 nohup nice -n 5 python3 scan_long5.py scan_long7.jsonl 40 0.3 0.4 "23.038037500000005:longB_D23.0380.npy:23;23.338037500000006:longB_D23.3380.npy:23;23.728037500000006:longB_D23.7280.npy:23;24.235037500000008:longB_D24.2350.npy:23" > scan_long19.log 2>&1 &
