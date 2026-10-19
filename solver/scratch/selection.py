"""How strongly do the regularity conditions select Delta?  For converged solutions at several D, compute the
Jacobian J of the full DSS system and measure
  s_Delta = | (I - P_other) dF/dDelta |  : change of the residual per unit change of Delta when every other unknown
                                          is re-optimised (Schur complement; 0 would mean Delta is not selected),
and the smallest singular values of J with the Delta-weight of their right singular vectors."""
import json, numpy as np, warnings, time
warnings.filterwarnings("ignore")
from run_spec import make
KC = 23
cases = [(4.1, 31, "solD_4.100_kc31.npy"), (5.0, 31, "solD_5.000_kc31.npy"), (6.0, 31, "solD_6.000_kc31.npy"),
         (7.0, 31, "solD_7.000_kc31.npy"), (8.0, 23, "solD_8.000_kc23.npy"), (8.5, 23, "solD_8.500_kc23.npy"),
         (9.203125, 23, "solD_9.203_kc23.npy")]
out = []
for D, kc, f in cases:
    t0 = time.time()
    S = make(KC, n=D - 2); z = np.load(f)
    if kc != KC: z = S.regrid(z, make(kc, n=D - 2))
    F, J = S.jacobian(z)
    Jd = J[:, -1]; Jo = J[:, :-1]
    y, *_ = np.linalg.lstsq(Jo, Jd, rcond=1e-12)
    sD = np.linalg.norm(Jd - Jo @ y)
    U, s, Vt = np.linalg.svd(J)
    vD = np.abs(Vt[-4:, -1])
    rec = dict(D=D, Delta=float(z[-1]), F=float(np.linalg.norm(F)), sDelta=float(sD), Jd=float(np.linalg.norm(Jd)),
               smax=float(s[0]), smin=[float(v) for v in s[-4:]], vDelta=[float(v) for v in vD], shape=list(J.shape))
    out.append(rec)
    print(f"D={D:6.3f} Delta={z[-1]:.6f} |F|={rec['F']:.1e}  s_Delta={sD:.3e}  |dF/dDelta|={rec['Jd']:.3e}  "
          f"smin(last4)={np.array2string(s[-4:], precision=2)}  |v_Delta|={np.array2string(vD, precision=2)}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open("selection.json", "w"), indent=1)
