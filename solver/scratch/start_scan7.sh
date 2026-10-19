cd /home/claude/dss
NOGAMMA=1 nohup nice -n 5 python3 scan_long2.py scan_longB.jsonl 32 0.3 0.5 "19.214399999999998:long_D19.2144.npy:23;19.489024999999998:long_D19.4890.npy:23;19.846037499999998:long_D19.8460_svd2.npy:23" > scan_long7.log 2>&1 &
