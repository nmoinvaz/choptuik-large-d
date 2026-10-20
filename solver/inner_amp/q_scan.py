"""Leading-order harmonic structure forced by S_nu = 0 (nu >= 3): ratios f_3/F_1, f_5/f_3 at leading order in eps,
as a function of omega (F_1 = -i eps/2 is the e^{i theta} coefficient of eps sin theta).
    python3 q_scan.py wf1 wf2 ...      (K = 5, J = 140)"""
import sys
from stokes_amp2 import run
for wf in map(float, sys.argv[1:]):
    sig, fx = run(wf, 5, 140, verbose=False)
    q3 = fx[(3, 3)] / (-0.5j)
    q5 = fx[(5, 5)] / fx[(3, 3)]
    w = wf * 2 * 3.141592653589793 / 1.64389375
    print(f"wf {wf:.3f} omega {w:.5f}: f3/F1 = {q3:.7g}  f5/f3 = {q5:.7g}  1/(f3/F1) = {1/q3:.6g}  1/(f5/f3) = {1/q5:.6g}"
          f"  8w^2 = {8*w*w:.4f}  sigma1 {sig[1]:.6g} sigma2 {sig[2]:.6g}", flush=True)
