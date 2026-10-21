#!/usr/bin/env python3
"""Neighbour-family amplitudes in the late orders of harmonic 1 (gap (b)): run the amplitude expansion (phys fits),
then refit the late orders of harmonic 1 at selected orders k and print each family's coefficient at relative order 1/j
(the basis column Gamma(j+g')/Gamma(j+g0)/(j+g'-1) has modulus ~1/j, so the coefficient is the family's amplitude
relative to the main family's constant term, i.e. relative to S_1^(k)'s own scale).

    python3 family_amps.py [K] [J] [wf]
"""
import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inner_amp"))
import amp_flint_par as P  # noqa: E402
from flint import acb, acb_mat, ctx  # noqa: E402


def fit_coeffs(A, nu, k, Mf):
    ys = A._ys(nu, k, A.fit_data(nu)[3], A.fit_step)
    p0 = ctx.prec
    ctx.prec = A.phys_prec
    try:
        w = A.w
        g0 = acb(0, nu) * w + acb(1) / 2
        fams = P.phys_families(nu, k)
        js = list(range(A.jlo, A.J + 1, A.fit_step))
        rows = []
        for j in js:
            main = [acb(1)]
            t = acb(1)
            for m in range(1, A.M + 1):
                t = t / (j + g0 - m)
                main.append(t)
            lg0 = (j + g0).lgamma()
            ts = [(g0 + acb(0, tsh) * w, ((j + g0 + acb(0, tsh) * w).lgamma() - lg0).exp()) for tsh in fams]
            levels = []
            for m in range(1, Mf + 1):
                for i, (gs, t) in enumerate(ts):
                    t = t / (j + gs - m)
                    ts[i] = (gs, t)
                    levels.append(t)
            rows.append(main + levels)
        n = len(rows[0])
        sc = [max(abs(r[c]).mid() for r in rows) for c in range(n)]
        Am = acb_mat([[r[c].mid() / sc[c] for c in range(n)] for r in rows])
        y = acb_mat([[acb(v.mid())] for v in ys])
        AH = Am.transpose().conjugate()
        c = (AH * Am).solve(AH * y, algorithm="approx")
        S = c[0, 0] / sc[0]
        nf = len(fams)
        lev1 = {fams[i] + nu: c[A.M + 1 + i, 0] / sc[A.M + 1 + i] for i in range(nf)}
        return S, lev1
    finally:
        ctx.prec = p0


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    J = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    wf = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    A = P.AmpPar(wf, K, J, prec=640, procs=1, fit="phys", mf=(6, 10), fit_step=1)
    A.run(verbose=False)
    out = {}
    for k in range(5, K + 1, 2):
        S, lev = fit_coeffs(A, 1, k, 8)
        print(f"k {k}: S_1^(k) = {complex(S.mid()):.6e}", flush=True)
        for nup in sorted(lev, key=lambda x: (abs(x), x)):
            v = complex(lev[nup].mid())
            print(f"     family nu' = {nup:+3d}: level-1 coefficient {v.real:+.4e}{v.imag:+.4e}i   |.|/|S_1^(k)| {abs(v) / abs(complex(S.mid())):.3e}")
        out[k] = {"S": [complex(S.mid()).real, complex(S.mid()).imag],
                  "fam": {str(n): [complex(lev[n].mid()).real, complex(lev[n].mid()).imag] for n in lev}}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"family_amps_K{K}_J{J}.json"), "w"))
