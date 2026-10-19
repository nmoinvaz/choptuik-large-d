"""Robustness of the Delta-selection strength: Tikhonov-weighted Schur complement
   s(mu) = min_y |J_o y + J_D|^2 + mu^2 |W y|^2   (W = harmonic weights, penalising the weakly determined top harmonics)
at several mu, and at two truncations kc, for the same converged solutions."""
import sys, json, numpy as np, warnings, time
warnings.filterwarnings("ignore")
from run_spec import make
from newton_tsvd import harmonic_weights
cases = [(6.0, 31, "solD_6.000_kc31.npy"), (8.0, 31, "solD_8.000_kc31.npy"), (8.0, 23, "solD_8.000_kc23.npy"),
         (8.953125, 31, "solD_8.953_kc31.npy"), (9.203125, 23, "solD_9.203_kc23.npy")]
for D, kc, f in cases:
    t0 = time.time()
    S = make(kc, n=D - 2); z = np.load(f)
    F, J = S.jacobian(z)
    w = harmonic_weights(S, 4)[:-1]
    Jd, Jo = J[:, -1], J[:, :-1]
    smax = np.linalg.norm(J, 2)
    res = []
    for mur in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6):
        mu = mur * smax
        A = np.vstack([Jo, mu * np.diag(w)]); b = np.concatenate([Jd, np.zeros(len(w))])
        y = np.linalg.lstsq(A, b, rcond=None)[0]
        res.append(np.linalg.norm(Jd - Jo @ y))
    print(f"D={D:6.3f} kc={kc}: |F|={np.linalg.norm(F):.1e}  s(mu_rel=1e-10..1e-6) = " + "  ".join(f"{r:.3e}" for r in res) + f"   ({time.time()-t0:.0f}s)", flush=True)
