#!/bin/bash
# The cloud mode-count runs of notes/MODE_COUNT.md, in order (resumable: start it again after an interruption;
# finished jobs are skipped, see hp/modecount_queue.sh). The longer runs proposed for the local machine are listed in
# MODE_COUNT.md, section "Not done here".
#   - the first D = 61 count ran with lam_g +- i omega in the list of expected zeros (they are not zeros: the other
#     sector's gauge mode is the same perturbation as lam_g, see MODE_COUNT.md); the "--v2" job recounts it from its
#     cached evaluations with the corrected list (the windings of det M are the same, only the bookkeeping changes);
#   - resolution: M = 48 (resampled from the M = 32 states and re-converged). M = 16 was tried at D = 61 and did not
#     converge (|F| 1.3e-9 after Levenberg-Marquardt), so that run is void; modecount_common.setup now refuses such a
#     background;
#   - the f64 Python mode matrix at complex lambda (modecount_f64.py without --realscan) failed its real-lambda check
#     against the Rust kernel (central differences in the horizon block) and took 430 s per matrix; it is not used.
cd "$(dirname "$0")/.." || exit 1
exec bash hp/modecount_queue.sh \
 "check hp/data/seeds/local/up23_D61.0264.npz --quick" \
 "hp/data/seeds/local/up23_D61.0264.npz --lam=2.043532530673195" \
 "hp/data/states/up7_D45.0507.npz" \
 "hp/data/states/up6_D30.0688.npz --lam=2.079747703853" \
 "hp/data/seeds/local/up23_D61.0264.npz --lam=2.043532530673195 --v2" \
 "f64 --realscan=0.05" \
 "hp/data/states/up6_D30.0688.npz --M=48 --h0=0.6 --lam=2.079747703853" \
 "hp/data/seeds/local/up23_D61.0264.npz --M=48 --h0=0.6 --lam=2.043532530673195" \
 "hp/data/states/up2_D16.7600.npz --lam=2.127809048219261 --h0=0.6" \
 >> hp/data/modecount_queue.log 2>&1
