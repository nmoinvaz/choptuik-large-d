#!/bin/bash
# The mode-count runs of notes/MODE_COUNT.md, in order (resumable: start it again after an interruption).
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
 "hp/data/states/up2_D16.7600.npz" \
 "hp/data/states/up_D8.2500.npz --M=32" \
 "hp/data/states/up2_D16.7600.npz --M=16" \
 >> hp/data/modecount_queue.log 2>&1
