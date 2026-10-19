"""Are the junk directions weakly damped perturbation modes? Smallest singular values of the growth-mode matrix
M(lambda) along real lambda (a mode e^{lambda tau} times a periodic profile; lambda = -delta + i m omega is the same
periodic class as lambda = -delta). Dips of s_min at lambda = -delta close to 0 make J = M(0) nearly singular."""
import numpy as np, sys, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from rust_solver import make
lams = np.r_[np.linspace(-3, -0.5, 26), np.linspace(-0.45, 0.3, 16)]
for spec in sys.argv[1:]:
    f, D = spec.split(":"); D = float(D); z = np.load(f)
    S, R = make(D, z[-1])
    out = []
    for l in lams:
        s = np.linalg.svd(R.mode_matrix(z, l), compute_uv=False)
        out.append(s[-6:] / s[0])
    out = np.array(out)
    print(f"D={D}: log10 s_min/s_max (6 smallest at each lambda)")
    for l, row in zip(lams, out):
        print(f"  lam {l:+.2f}: " + " ".join(f"{np.log10(x):6.1f}" for x in row[::-1]), flush=True)
