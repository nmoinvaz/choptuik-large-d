#!/bin/bash
# The mode-count runs of notes/MODE_COUNT.md, in order (resumable: start it again after an interruption; finished
# jobs are skipped, see hp/modecount_queue.sh).
#   - the first D = 61 count ran with lam_g +- i omega in the list of expected zeros (they are not zeros: the other
#     sector's gauge mode is the same perturbation as lam_g, see MODE_COUNT.md); the "--v2" job recounts it from its
#     cached evaluations with the corrected list (the windings of det M are the same, only the bookkeeping changes);
#   - "--Im2=3.5 --tag=..._Im3.5" counts a taller strip, up to 3.5 omega, sharing the M = 16 evaluations.
cd "$(dirname "$0")/.." || exit 1
exec bash hp/modecount_queue.sh \
 "check hp/data/seeds/local/up23_D61.0264.npz --quick" \
 "hp/data/seeds/local/up23_D61.0264.npz --lam=2.043532530673195" \
 "hp/data/states/up7_D45.0507.npz" \
 "hp/data/states/up6_D30.0688.npz --lam=2.079747703853" \
 "hp/data/seeds/local/up23_D61.0264.npz --M=16" \
 "hp/data/states/up7_D45.0507.npz --M=16" \
 "hp/data/states/up6_D30.0688.npz --M=16" \
 "f64 --check" \
 "f64" \
 "hp/data/states/up2_D16.7600.npz --lam=2.127809048219261" \
 "hp/data/states/up_D8.2500.npz --M=32 --lam=2.2316917257584605" \
 "hp/data/states/up2_D16.7600.npz --M=16" \
 "hp/data/seeds/local/up23_D61.0264.npz --lam=2.043532530673195 --v2" \
 "hp/data/seeds/local/up23_D61.0264.npz --M=16 --Im2=3.5 --tag=D61.0264_M16_Im3.5 --cache=D61.0264_M16" \
 >> hp/data/modecount_queue.log 2>&1
