cd /home/claude/dss
NOGAMMA=1 nohup nice -n 5 python3 scan_long3.py scan_longB.jsonl 32 0.6 0.6 "21.838037500000002:longB_D21.8380.npy:23;22.438037500000004:longB_D22.4380.npy:23;23.038037500000005:longB_D23.0380.npy:23" > scan_long11.log 2>&1 &
