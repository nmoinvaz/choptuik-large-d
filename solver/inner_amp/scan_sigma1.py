"""sigma_1(omega) = S_1^(3)/S_1^(1): first amplitude correction to the harmonic-1 inner Stokes multiplier (all harmonics
exact). Leading-order selection 1 + sigma_1 eps^2 = 0 needs sigma_1 real and negative.
    python3 scan_sigma1.py wf_lo wf_hi n [J]"""
import sys
import numpy as np
from stokes_amp2 import prep, fit_nu, W0
lo, hi, n = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
J = int(sys.argv[4]) if len(sys.argv) > 4 else 140
for wf in np.linspace(lo, hi, n):
    d, E, delta = prep(wf * W0, 3, J, {})
    r, _ = fit_nu(wf * W0, 3, J, d, E, delta, 1, J // 3)
    sg = r[3] / r[1]
    print(f"wf {wf:.4f} omega {wf * W0:.5f}: sigma_1 = {sg.real:+.6f} {sg.imag:+.6f}i   |s| {abs(sg):.4f} arg/pi {np.angle(sg)/np.pi:+.4f}"
          + (f"   -> eps^2 = {-1/sg.real:.4f}" if abs(sg.imag) < 0.05 * abs(sg) else ""), flush=True)
