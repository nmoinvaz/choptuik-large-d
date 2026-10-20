"""Heuristic test of the S_nu = 0 (nu >= 3) relations at finite D: evaluate the D = infinity inner-problem prediction of
F3/F1 and F5/F1 at each D's own amplitude A(D) (README: first harmonic of sqrt(n) dp/dx(0)) and omega(D) = 2 pi/Delta(D),
and compare with the measured ratios. (Ignores 1/n corrections of the relation itself.)
    python3 harm_pred.py [K J]"""
import sys
import numpy as np
from stokes_amp2 import run
K = int(sys.argv[1]) if len(sys.argv) > 1 else 7
J = int(sys.argv[2]) if len(sys.argv) > 2 else 140
DINF = 1.64389375
rows = [(20, 2.112080154580, 3.480, 0.130, 0.024), (30, 1.964957899258, 3.212, 0.105, 0.016),
        (40, 1.888568197910, 3.086, 0.093, 0.012), (50, 1.841659503007, 3.011, 0.085, 0.0097),
        (60, 1.809889459975, 2.962, 0.080, 0.0085), (1e9, DINF, 2.72, None, None)]
for D, Dl, A, m3, m5 in rows:
    wf = DINF / Dl
    sig, f = run(wf, K, J, verbose=False)
    f3 = sum(f.get((k, 3), 0) * A ** k for k in range(3, K + 1, 2))
    f5 = sum(f.get((k, 5), 0) * A ** k for k in range(5, K + 1, 2))
    F1 = A / 2
    terms3 = [abs(f.get((k, 3), 0) * A ** k) for k in range(3, K + 1, 2)]
    print(f"D {D:>6g}: omega {2 * np.pi / Dl:.4f} A {A}: predicted |F3/F1| {abs(f3) / F1:.4f} |F5/F1| {abs(f5) / F1:.5f}"
          f"  (f3 terms {', '.join(f'{t:.2e}' for t in terms3)})   measured {m3} {m5}", flush=True)
