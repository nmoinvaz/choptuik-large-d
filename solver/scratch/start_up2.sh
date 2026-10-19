cd /home/claude/dss
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nohup python3 scan_D.py 31 "5.75:/home/claude/dss/solD_5.750_kc31.npy:2.3432329195;6.0:/home/claude/dss/solD_6.000_kc31.npy:2.3251516016" 6.25,6.5,6.75,7.0,7.25,7.5,7.75,8.0,8.5,9.0,9.5,10.0,11.0,12.0,13.0,14.0,16.0,18.0,20.0 /home/claude/dss/scan_up_kc31.jsonl > scan_up3.log 2>&1 &
