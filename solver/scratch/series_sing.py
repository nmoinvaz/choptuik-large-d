"""Locate the nearest complex singularity of the horizon series in u = 1 - x from the late coefficients:
fit |c_K| ~ R^-K and sign/phase oscillation c_K ~ R^-K cos(K theta + phi) for single Fourier harmonics."""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
D = float(sys.argv[1]); f = sys.argv[2]; KS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
S = make_long(23, D - 2, nodes=[0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6], Ks=KS); z = np.load(f)
parts, Dl = S.split(z[None, :]); om = np.array([2 * np.pi / Dl[0]])
P, M, A, C = S.ssh_series(S.phys(parts[-2], "odd"), S.phys(parts[-1], "even"), om)
n = D - 2
def harm(Q, k):
    return np.array([np.fft.rfft(q[0])[k] / S.N for q in Q])
for name, Q, k in (("C", C, 0), ("A", A, 0), ("P", P, 1), ("M", M, 1), ("C", C, 2)):
    c = harm(Q, k); K = np.arange(len(c))
    lo = 30
    # modulus: linear fit of log|c_K| on K for K >= lo  (envelope via running max over a window)
    env = np.array([np.abs(c[max(0, i - 4):i + 1]).max() for i in range(len(c))])
    sl, ic = np.polyfit(K[lo:], np.log(env[lo:] + 1e-300), 1)
    R = np.exp(-sl)
    # phase: ratio method on complex coefficients (Fourier harmonic is complex): c_{K+1}/c_K -> 1/u_c if single singularity;
    # for a conjugate pair, use real/imag parts separately: fit 2-term linear recurrence c_{K+1} = a1 c_K + a2 c_{K-1}
    X = np.stack([c[lo:-1], c[lo - 1:-2]], 1); y = c[lo + 1:]
    (a1, a2), *_ = np.linalg.lstsq(X, y, rcond=None)
    roots = np.roots([1, -a1, -a2])            # = 1/u_c candidates
    us = 1 / roots
    print(f"{name} harmonic {k}: envelope radius R = {R:.3f};  recurrence singularities u_c = " + ", ".join(f"{u.real:+.3f}{u.imag:+.3f}i (|u|={abs(u):.3f}, arg={np.degrees(np.angle(u)):+.1f} deg)" for u in us))
lg = float(np.mean(1 - (n - 1) * (S.phys(parts[-1], 'even')[0] ** 2 - 1)))
print(f"D={D}: s0 = {n/(2*lg)-1:.3f}, lambda_g = {lg:.4f}, Delta = {Dl[0]:.6f}")
