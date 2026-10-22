#!/bin/bash
# Mode-count runs proposed for the local machine (notes/MODE_COUNT.md, section 7). One thread each, below-normal
# priority; the jobs are independent and may run side by side (one process per thread). Each is resumable: evaluations
# are cached in hp/data/modecount_<tag>_evals.jsonl and re-converged states in hp/data/modecount_state_*.npz.
# Stop conditions: modecount_common.setup raises if the background does not converge (|F| > --fmax); a count with
# wind(g) != 0 in any box is a finding: stop, locate it (hp/modecount_moment.py TAG STATE), check it on a circle at a
# second M (hp/modecount_circle.py), and report. Push hp/data/modecount_* after each job.
#
#   bash hp/modecount_local.sh N        runs job N (1-9); without N lists them
cd "$(dirname "$0")/.." || exit 1
J=(
 ""
 # 1. D = 16.76 at production resolution (M 32, nodes 0.1-0.7)
 "python3 hp/modecount_wind.py hp/data/states/up2_D16.7600.npz --lam=2.127809048219261 --h0=0.6"
 # 2. D = 8.25 resampled to M 32 (native M 48)
 "python3 hp/modecount_wind.py hp/data/states/up_D8.2500.npz --M=32 --lam=2.2316917257584605 --h0=0.6 --fmax=1e-15"
 # 3. D = 4 in double-double: seed from the f64 kc 39 state (pilot: |F| < 1e-20 at M 32, Delta within 1e-6 of
 #    3.4454555), then count
 "python3 hp/modecount_lowD.py solver/data/neB_kc39_D4.0000.npy --kc=39 --M=32,48 && python3 hp/modecount_wind.py hp/data/modecount_state_D4.0000_M48.npz --h0=0.6 --lam=2.6740838"
 # 4. the D = 30 M 48 pair at a third resolution: circles at M 64 without and with the cut
 "python3 hp/modecount_circle.py hp/data/states/up6_D30.0688.npz 4.267,1.324,0.35 --M=64 --fmax=1e-15 && python3 hp/modecount_circle.py hp/data/states/up6_D30.0688.npz 4.267,1.324,0.35 --M=64 --fmax=1e-15 --kf=13"
 # 5. full M 48 count at D = 30 with the cut (state saved, |F| 8e-28; shares the circle's cache)
 "python3 hp/modecount_wind.py hp/data/states/up6_D30.0688.npz --M=48 --kf=13 --h0=0.6 --lam=2.079747703853 --fmax=1e-15 --tag=D30.0688_M48_kf13"
 # 6. M 48 at D = 61 with the cut: the saved state is at |F| 2.7e-15; this resumes the convergence, then counts
 "python3 hp/modecount_wind.py hp/data/seeds/local/up23_D61.0264.npz --M=48 --h0=0.6 --lam=2.043532530673195 --fmax=1e-15"
 # 7. M 48 at D = 45 with the cut
 "python3 hp/modecount_wind.py hp/data/states/up7_D45.0507.npz --M=48 --h0=0.6 --lam=2.056738610147195 --fmax=1e-15"
 # 8. D = 61, M 32: a taller strip, Im up to 3.5 omega (the 2i omega copies of 0, lam_g, lam_phys inside)
 "python3 hp/modecount_wind.py hp/data/seeds/local/up23_D61.0264.npz --lam=2.043532530673195 --Im2=3.5 --tag=D61.0264_M32_Im3.5 --cache=D61.0264_M32"
 # 9. D = 61, M 32: Re up to 12
 "python3 hp/modecount_wind.py hp/data/seeds/local/up23_D61.0264.npz --lam=2.043532530673195 --aR=12.07 --tag=D61.0264_M32_Re12 --cache=D61.0264_M32"
)
if [ $# -eq 0 ]; then
  for i in $(seq 1 9); do echo "$i: ${J[$i]}"; done
  exit 0
fi
nice -n 10 bash -c "${J[$1]}"
