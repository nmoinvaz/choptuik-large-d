"""Test Ecker-Ecker-Grumiller's NLO condition beta'' + 3 beta' = 0 at beta = 0 (their Eq. 20) on the exact DSS
solutions. Our amplitude B(tau) = <c q / x> over x in (0.05, 0.95) relates to beta by 2B^2 = beta^2/(1+beta^2),
so at a zero of B: beta'/B' = beta''/B'' = sqrt(2). Condition => B''/B' = -3 at the zero crossings."""
import json, numpy as np
d = json.load(open("largeD_B.json"))
def crossings(B, Dl):
    N = len(B); F = np.fft.rfft(B) / N; k = np.arange(len(F)); om = 2 * np.pi / Dl
    ev = lambda t, m: np.real(sum((2 if j else 1) * F[j] * (1j * j * om) ** m * np.exp(1j * j * om * t) for j in range(len(F)) if j < N / 2))
    tt = np.linspace(0, Dl, 4000, endpoint=False); b = ev(tt, 0)
    out = []
    for i in np.where(np.sign(b) != np.sign(np.roll(b, -1)))[0]:
        t0 = tt[i]
        for _ in range(30): t0 -= ev(t0, 0) / ev(t0, 1)
        out.append((t0 % Dl, ev(t0, 1), ev(t0, 2)))
    return out
for k in sorted(d, key=float):
    r = d[k]; B = np.array(r["B"]); Dl = r["Delta"]
    cr = crossings(B, Dl)
    print(f"D={float(k):4.1f} Delta={Dl:.4f}: " + "; ".join(f"tau0={t:.3f} B'={b1:+.3f} B''/B' = {b2/b1:+.3f}" for t, b1, b2 in cr))
