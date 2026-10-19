"""Exact DSS solutions vs the large-D zero-crossing analysis.
B_c(tau) = Q(tau, x->0) (centre amplitude, EEG's beta/sqrt2),  B_h(tau) = Q(tau, 1) = q(tau, 1) (horizon edge).
Predictions:  EEG condition B_c''/B_c' = -3 at zeros (claimed);  our analysis: no such condition, and
              tau_z(horizon) - tau_z(centre) = -(eps/2) (B_c''/B_c' + 3) + O(eps^2),  eps = 1/(D-2)."""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from dense import dense
cases = [(4.0, 31, "spec_kc31.npy"), (5.0, 31, "solD_5.000_kc31.npy"), (6.0, 31, "solD_6.000_kc31.npy"), (7.0, 31, "solD_7.000_kc31.npy"),
         (8.0, 31, "solD_8.000_kc31.npy"), (9.203125, 23, "solD_9.203_kc23.npy"), (10.106142578125, 23, "solD_10.106_kc23.npy")]
def fourier(f, Dl):
    N = len(f); F = np.fft.rfft(f) / N; om = 2 * np.pi / Dl
    return lambda t, mm=0: np.real(sum((2 if j else 1) * F[j] * (1j * j * om) ** mm * np.exp(1j * j * om * t) for j in range(len(F)) if j < N / 2))
def zeros(ev, Dl):
    tt = np.linspace(0, Dl, 3000, endpoint=False); b = ev(tt)
    out = []
    for i in np.where(np.sign(b) != np.sign(np.roll(b, -1)))[0]:
        t0 = tt[i]
        for _ in range(40): t0 -= ev(t0) / ev(t0, 1)
        out.append(t0 % Dl)
    return np.array(sorted(out))
res = []
for D, kc, f in cases:
    eps = 1 / (D - 2)
    X, Fd, Dl, N = dense(D, kc, f)
    p, m, a, c = Fd[:, 0], Fd[:, 1], Fd[:, 2], Fd[:, 3]
    q = 0.5 * (p - m)
    ic = np.argmin(np.where(X > 0, X, 9))            # smallest x > 0
    Bc = c[ic] * q[ic] / X[ic]; Bh = q[-1]            # X[-1] = 1
    evc, evh = fourier(Bc, Dl), fourier(Bh, Dl)
    zc, zh = zeros(evc, Dl), zeros(evh, Dl)
    z0 = zc[0]; ratio = evc(z0, 2) / evc(z0, 1)
    # matching horizon zero (same sign of slope)
    cand = [z for z in zh if np.sign(evh(z, 1)) == np.sign(evc(z0, 1))]
    dz = min(((z - z0 + Dl / 2) % Dl - Dl / 2 for z in cand), key=abs)
    pred = -(eps / 2) * (ratio + 3)
    res.append(dict(D=D, ratio=ratio, shift=dz, pred=pred))
    print(f"D={D:7.3f} eps={eps:.3f}: B_c''/B_c' at zero = {ratio:+.3f} (EEG claims -3);  zero shift horizon-centre = {dz:+.4f},  first-order prediction {pred:+.4f}  (x_min={X[ic]:.0e})", flush=True)
json.dump(res, open("zero_shift.json", "w"), indent=1)
