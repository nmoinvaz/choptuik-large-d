"""Growth-rate condition of the one-harmonic inner model by ratios instead of fits. For profiles f = cos theta and
f = sin theta the late orders share the fast-mode factor 2^j Gamma(j + gamma_k + lam) (1 + c_1/j + ...), so the ratio
of their mode-k components (transport eigenvector basis) tends to Tc_k / Ts_k; a least-squares fit in 1/j with the other mode's oscillating leakage j^(+-(gamma_1 - gamma_0))/j sharpens it. A growth
rate is a lam where that limit is real (then x cos + y sin with y/x = -Tc/Ts has no fast part).
Exact checks: lam = 0 gives Tc = 0 (theta-translation), lam = 1 gives Tc/Ts = -1/omega (time translation, X Z).
    python3 pert_ratio.py lam [lam ...]       (env J, default 200)"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inner_borel"))
import mpmath as mp
from flint import acb
from pert import series_pert
from inner_stokes import W0, to_mp, transport_modes
mp.mp.dps = 120
A = mp.mpf(os.environ.get("A", "2.54214805")); wf = mp.mpf(os.environ.get("WF", "0.77699401")); w = W0 * wf
J = int(os.environ.get("J", "200"))
Fh = {1: -0.5j * A, -1: 0.5j * A}
nus, E, V = transport_modes(Fh, w, 1)
Vi = V ** -1
Fa = {1: acb(0, -0.5) * acb(str(A)), -1: acb(0, 0.5) * acb(str(A))}
prof = {"c": {1: acb(0.5), -1: acb(0.5)}, "s": {1: acb(0, -0.5), -1: acb(0, 0.5)}}


def ratios(lam):
    comp = {}
    for key, fh in prof.items():
        _, a, _, _ = series_pert(Fa, fh, str(w), str(lam), J, 2)
        comp[key] = [Vi * mp.matrix([to_mp(a[j][nu + 2]) for nu in nus]) for j in range(J + 1)]
    out = []
    gam = [mp.mpf(1) / 2 - E[k] / 2 for k in range(2)]
    for k in range(2):
        R = {j: comp["c"][j][k] / comp["s"][j][k] for j in range(J - 80, J + 1)}
        dl = gam[1 - k] - gam[k]                  # leakage of the other mode: relative factor j^(dl), dl imaginary here
        cols = lambda j: [mp.mpf(1), mp.mpf(1) / j, mp.mpf(1) / j ** 2, mp.power(j, dl) / j, mp.power(j, dl) / j ** 2,
                          mp.power(j, -dl) / j, mp.power(j, -dl) / j ** 2]
        res = []
        for j0 in (J - 80, J - 50):
            js = list(range(j0, J + 1))
            x = mp.qr_solve(mp.matrix([cols(j) for j in js]), mp.matrix([R[j] for j in js]))[0]
            res.append(x[0])
        out.append((R[J], res[0], res[1]))
    return out


if __name__ == "__main__":
    print(f"A {A} omega {mp.nstr(w, 10)} J {J}; -1/omega = {mp.nstr(-1 / w, 10)}", flush=True)
    for lam in sys.argv[1:]:
        o = ratios(mp.mpf(lam))
        print(f"lam {lam}: mode0 R_J {mp.nstr(o[0][0], 8)}  fit {mp.nstr(o[0][1], 10)} / {mp.nstr(o[0][2], 10)}"
              f"   mode1 fit {mp.nstr(o[1][1], 10)}   arg(R0)/pi {mp.nstr(mp.arg(o[0][1]) / mp.pi, 6)}", flush=True)
