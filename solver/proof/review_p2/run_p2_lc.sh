#!/bin/sh
# P2: the independent Lemma LC check and its mutation controls
cd "$(dirname "$0")"
nice -n 10 python3 p2_lc.py --lip > p2_lc.log 2>&1
for m in rho2 gF d0r4 phid; do
  echo "== --mutate=$m" >> p2_lc_mutations.log
  nice -n 10 python3 p2_lc.py --lip --mutate=$m 2>&1 | grep -E "=> \(A\)|Lipschitz|MUTATION" >> p2_lc_mutations.log
done
