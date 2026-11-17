#!/bin/bash
# The runs of the review of smdriver.newton_band (#922), one process at a time, each under a time cap.
#   SITE=<the sigmams build's site directory> RAWR=<clone of choptuik-large-d-raw> PY=<python 3.14> bash run_i922.sh
# Run from matter/sigma_ms. Writes into $RAWR/I922-mac/ (logs, JSON records, exits.txt).
set -u
: "${SITE:?the site directory of the sigmams build}" "${RAWR:?the raw-data clone}"
PY=${PY:-python3}
RAW=$RAWR/I922-mac
export PYTHONPATH=$SITE VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$RAW/planted" "$RAW/mut" "$RAW/newton" "$RAW/states"

run() {  # run LOG WANT CAP command...: the exit status and the seconds go to exits.txt
    local log=$1 want=$2 cap=$3
    shift 3
    local t0
    t0=$(date +%s)
    nice -n 10 gtimeout "$cap" "$@" > "$RAW/$log.txt" 2>&1
    echo "$log: exit $? ($want), $(( $(date +%s) - t0 )) s" >> "$RAW/exits.txt"
}

# R3: the author's planted test and its four mutations
run planted/clean "must be 0" 580 "$PY" planted_band.py --out "$RAW/planted/clean.json"
for m in noband easy root gamma; do
    run planted/mut_$m "must be 1" 580 "$PY" planted_band.py --mutate $m --out "$RAW/planted/mut_$m.json"
done
# R4: the reviewer's mutations of newton_band through the same test
run mut/none "must be 0" 580 "$PY" review_922/rv_band_mut.py --mutate none --out "$RAW/mut/none.json"
run mut/sonly "must be 1: named on the issue before any run" 580 "$PY" review_922/rv_band_mut.py --mutate sonly --out "$RAW/mut/sonly.json"
run mut/noshort "must be 0" 580 "$PY" review_922/rv_band_mut.py --mutate noshort --out "$RAW/mut/noshort.json"
run mut/loose "must be 1: written before the run" 580 "$PY" review_922/rv_band_mut.py --mutate loose --out "$RAW/mut/loose.json"
for m in nodamp nofloor noheld anyB nochord; do
    run mut/$m "0 or 1: reported" 580 "$PY" review_922/rv_band_mut.py --mutate $m --out "$RAW/mut/$m.json"
done
# R2: one Newton step from #913's states at D 10.5 (K1 and K3 with --fd: 10 minutes, and 18 from two step sizes)
S913=$RAWR/I913-mac/out
run newton/hi_fd "must be 0" 3000 "$PY" -u review_922/rv_newton.py --state "$S913/sm_hi_D10.5_M32.npz" --fd --out "$RAW/newton/hi_fd.json"
run newton/hi_fd2 "must be 0" 3600 "$PY" -u review_922/rv_newton.py --state "$S913/sm_hi_D10.5_M32.npz" --fd --richardson --out "$RAW/newton/hi_fd2.json"
run newton/fdcol "must be 0" 500 "$PY" -u review_922/rv_fdcol.py --state "$S913/sm_hi_D10.5_M32.npz" --cols 158 --random 40 --out "$RAW/newton/fdcol.json"
for t in hi up march; do
    run newton/$t "must be 0" 500 "$PY" -u review_922/rv_newton.py --state "$S913/sm_${t}_D10.5_M32.npz" --out "$RAW/newton/$t.json"
done
# R6: the residual at every stored state of the band solver's runs, and the control
F=I845-mac,I913-mac,I915-mac,I916-mac,I917-mac
run states/all "must be 0" 900 "$PY" -u review_922/rv_states.py "$RAWR" --folders $F --out "$RAW/states/all.json"
run states/mut_state "must be 1" 900 "$PY" -u review_922/rv_states.py "$RAWR" --folders $F --mutate state --out "$RAW/states/mut_state.json"
cat "$RAW/exits.txt"
