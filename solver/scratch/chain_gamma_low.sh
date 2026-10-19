cd /home/claude/dss
while [ -d /proc/4848 ]; do sleep 30; done
nice -n 8 python3 gamma_root.py gamma_cs.jsonl 0.6 60 "$(cat gseeds_low.txt)" 2.2386 > gamma_cs_low.log 2>&1
