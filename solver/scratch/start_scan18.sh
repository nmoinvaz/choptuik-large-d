cd /home/claude/dss
NOGAMMA=1 nohup nice -n 5 python3 scan_long4.py scan_longB.jsonl 40 0.3 0.4 "30.51032140000001:longB_D30.5103.npy:23;30.84832140000001:longB_D30.8483.npy:23;31.04832140000001:longB_D31.0483.npy:23;31.30832140000001:longB_D31.3083.npy:23" > scan_long18.log 2>&1 &
