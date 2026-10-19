import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from dense import dense
cases = [(float(a), int(b), c) for a, b, c in (x.split(":") for x in sys.argv[1:])]
res = {}
for D, kc, f in cases:
    n = D - 2
    X, Fd, Dl, N = dense(D, kc, f)
    p, m, a, c = Fd[:, 0], Fd[:, 1], Fd[:, 2], Fd[:, 3]
    q = 0.5 * (p - m); sel = (X > 0.05) & (X < 0.95)
    B = (c * q / X[:, None])[sel].mean(axis=0)
    F = np.fft.rfft(B) / N; om = 2 * np.pi / Dl
    ev = lambda t, mm: np.real(sum((2 if j else 1) * F[j] * (1j * j * om) ** mm * np.exp(1j * j * om * t) for j in range(len(F)) if j < N / 2))
    tt = np.linspace(0, Dl, 4000, endpoint=False); b = ev(tt, 0)
    out = []
    for i in np.where(np.sign(b) != np.sign(np.roll(b, -1)))[0]:
        t0 = tt[i]
        for _ in range(30): t0 -= ev(t0, 0) / ev(t0, 1)
        out.append(ev(t0, 2) / ev(t0, 1))
    res[D] = dict(Delta=Dl, ratio=float(np.mean(out)), Bmax=float(np.abs(B).max()))
    print(f"D={D:.4f} Delta={Dl:.5f} B''/B' at zeros = {np.mean(out):+.4f}  Bmax={np.abs(B).max():.4f}", flush=True)
json.dump(res, open("eeg_ratio_hiD.json", "w"))
