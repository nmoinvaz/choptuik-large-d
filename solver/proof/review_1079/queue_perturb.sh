#!/bin/bash
# Review of #1079, check 5: the perturbation runs one after the other (one process).
# usage: bash review_1079/queue_perturb.sh RAW    from solver/proof
{
RAW=$1
R=review_1079/run_perturb.sh
bash $R "$RAW" x1 none
bash $R "$RAW" x1 qr 1079
bash $R "$RAW" x2 perron 1079 4096
bash $R "$RAW" x1 qr 1
bash $R "$RAW" x2 qr 1
bash $R "$RAW" x1 qr 2
bash $R "$RAW" x2 qr 2
bash $R "$RAW" x1 qr 3
bash $R "$RAW" x2 qr 3
echo "$(date -u '+%F %T') queue done" >> "$RAW/I1079-mac-review/steps.txt"
exit 0
}
