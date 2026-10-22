"""Noise floor of the Schur-corrected residuals of hier.py at M levels, at a point where the levels above 3 are tuned.

Why: the M = 7 pilot (logs/hier_nb_M7_pilot_fixA_at_M5zero.log) stopped its level-3 Newton at |r_3| = 9 with a fresh
Jacobian no longer halving |r_3| (hier's noise-floor test), and its r_1 was ~1e5. Either the level-3 finite-difference
Jacobian (step 1e-6 relative, |dr_3| ~ 0.08) is below the noise, or the corrected r_3 itself is that noisy. This script
measures it: re-tune levels 5..M at fixed (A, w, B3, C3) (LazyHier, as in the run), compute the leakage Jacobian K, then
evaluate the corrected residual with the integrator's rtol at 1e-12 (hier's) and at 0.8e-12, and with B_M moved by a few
ulps; the spread is the noise floor of each corrected r_l, against which the level-3 FD steps (1e-6, 1e-5, 1e-4 relative,
with the higher levels moved by K's linear prediction) are compared.

    python3 m7_noise.py M --x=A,wf,B3,C3,...      (compiled backend)
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hier_nb as HN                                     # noqa: E402  (installs the compiled backend and LazyHier)
import hier as H                                         # noqa: E402
import inner_np as I                                     # noqa: E402

T0 = time.time()


def main():
    M = int(sys.argv[1])
    opts = {a.split("=")[0]: a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--") and "=" in a}
    x = np.array([float(t) for t in opts["--x"].split(",")])
    h = H.Hier(M, verbose=1, top=3)
    h.schur = True
    h.ltop = 3
    x, r = h.solve_from(5, x)                       # tune levels 5..M at fixed (A, w, B3, C3)
    print(f"levels 5..{M} tuned: " + "  ".join(f"|r_{nu}| {np.linalg.norm(r[H.idx(nu)]):.2e}" for nu in range(5, M + 1, 2))
          + f"  [{time.time() - T0:.0f} s]", flush=True)
    for nu in range(5, M + 1, 2):
        h.mode[nu] = "top"
    h.predict = True
    h.leak_jac(x)
    K = h.K
    base = h.res(x)
    raw = h.res_raw(x)
    print(f"corrected r_1 {base[0]:+.4e}{base[1]:+.4e}i  r_3 {base[2]:+.4e}{base[3]:+.4e}i   raw r_1 {raw[0]:+.3e}, "
          f"raw r_3 {raw[2]:+.3e}{raw[3]:+.3e}i", flush=True)
    # 1. integration noise: rtol 1e-12 -> 0.8e-12 in the compiled integrator
    orig = I.Inner

    class InnerRt(orig):
        def integrate(self, Y, kind, a, b, par, rtol=1e-12, atol=1e-16):
            return orig.integrate(self, Y, kind, a, b, par, 0.8e-12, atol)
    I.Inner = InnerRt
    alt = h.res(x)
    I.Inner = orig
    d = alt - base
    print("integration noise (rtol 1e-12 vs 0.8e-12), corrected: " +
          "  ".join(f"r_{nu} {np.linalg.norm(d[H.idx(nu)]):.2e}" for nu in range(1, M + 1, 2)), flush=True)
    # 2. ulp noise of the top harmonic, corrected
    for k in (1, 3):
        xp = x.copy()
        xp[-2] = np.nextafter(xp[-2], np.inf) if k == 1 else xp[-2] + 3 * (np.nextafter(xp[-2], np.inf) - xp[-2])
        rp = h.res(xp)
        dd = rp - base
        rr = h.res_raw(xp) - raw
        print(f"B_{M} + {k} ulp: corrected change " + "  ".join(f"r_{nu} {np.linalg.norm(dd[H.idx(nu)]):.2e}" for nu in
                                                                range(1, M + 1, 2))
              + f"   (raw r_1 {np.linalg.norm(rr[:2]):.2e}, raw r_3 {np.linalg.norm(rr[2:4]):.2e})", flush=True)
    # 3. level-3 finite differences with the higher levels moved by K's linear prediction
    Hr = list(range(4, M + 1))
    sch = K[2:4, 2] - K[2:4, Hr] @ np.linalg.solve(K[np.ix_(Hr, Hr)], K[Hr, 2])
    for rel in (1e-6, 1e-5, 1e-4):
        hs = rel * abs(x[2])
        xp = x.copy()
        xp[2] += hs
        xp[Hr] = x[Hr] - np.linalg.solve(K[np.ix_(Hr, Hr)], K[np.ix_(Hr, [2])] @ np.array([hs]))
        rp = h.res(xp)
        col = (rp[2:4] - base[2:4]) / hs
        print(f"dB3 = {rel:g} rel: d r_3 = {np.linalg.norm(rp[2:4] - base[2:4]):.3e}  column d r_3/dB3 {col[0]:+.4e} "
              f"{col[1]:+.4e}   K's Schur column {sch[0]:+.4e} {sch[1]:+.4e}   |r_H| after prediction "
              f"{np.linalg.norm(rp[4:]):.1e}  [{time.time() - T0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
