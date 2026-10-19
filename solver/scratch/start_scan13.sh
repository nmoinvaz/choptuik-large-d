cd /home/claude/dss
NOGAMMA=1 nohup nice -n 5 python3 scan_long3.py scan_longB.jsonl 32 0.3 0.5 "26.038347500000008:longB_D26.0383.npy:23;26.40964050000001:longB_D26.4096.npy:23;26.892321400000007:longB_D26.8923.npy:23" > scan_long13.log 2>&1 &
