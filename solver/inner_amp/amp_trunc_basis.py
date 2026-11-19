#!/usr/bin/env python3
"""Which late-order basis fits a truncated model's sequence? (#1016) Runs amp_flint_par.AmpPar with --trunc M (phys fit,
no log columns) to order K, then fits the normalised sequence of S_nu^(K) (the one the run's own fit used, after its
tuning) with several bases and prints, for Mf = 6, 8, 10, the fitted constant and its steps. A basis that describes
the sequence gives steps that fall fast (the phys fit's plateau); one that misses a term gives steps that stay.

Bases: the families present (|nu'| <= min(K, M)) or every odd |nu'| <= K; ln j times the main family's terms; ln j
times every neighbour family's terms; ln j on one neighbour family at a time.

    python3 amp_trunc_basis.py M K NU [J] [wf]       e.g. 3 9 1 1000 1.0156 (S_1 at k = 9), 3 7 3 1000 1.0156
"""
import sys
import time

from flint import acb, acb_mat, arb, ctx

import amp_flint_par as AP


def main():
    M, K, NU = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    J = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    wf = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    t0 = time.time()
    A = AP.AmpPar(wf, K, J, prec=640, procs=1, fit="phys", mf=(6, 12), trunc=M)
    A._run(False, True)
    ys = A._ys(NU, K, A.fit_data(NU)[3])
    print(f"M {M}, S_{NU}^({K}), J {J}, wf {wf}: run {time.time() - t0:.0f} s", flush=True)

    def fit(fams, Mf, mainlog=False, famlog=(), Mmain=12, prec=1920):
        ctx.prec = prec
        w = A.w
        g0 = acb(0, NU) * w + acb(1) / 2
        rows = []
        for j in range(A.jlo, J + 1):
            main = [acb(1)]
            t = acb(1)
            for m in range(1, Mmain + 1):
                t = t / (j + g0 - m)
                main.append(t)
            lj = arb(j).log()
            lm = [x * lj for x in main] if mainlog else []
            lg0 = (j + g0).lgamma()
            fr = []
            for tsh in fams:
                gs = g0 + acb(0, tsh) * w
                t = ((j + gs).lgamma() - lg0).exp()
                for m in range(1, Mf + 1):
                    t = t / (j + gs - m)
                    fr.append(t)
                    if tsh in famlog:
                        fr.append(t * lj)
            rows.append(main + lm + fr)
        n = len(rows[0])
        sc = [max(abs(r[c]).mid() for r in rows) for c in range(n)]
        Am = acb_mat([[r[c].mid() / sc[c] for c in range(n)] for r in rows])
        y = acb_mat([[acb(v.mid())] for v in ys])
        AH = Am.transpose().conjugate()
        c = (AH * Am).solve(AH * y, algorithm="approx")
        S = (c[0, 0] / sc[0]).mid()
        ctx.prec = 640
        return S, n

    sh = lambda kmax: [nup - NU for nup in range(-kmax, kmax + 1, 2) if nup != NU]
    pres, allf = sh(min(K, M)), sh(K)
    cases = [("present", pres, {}), (f"every odd |nu'| <= {K}", allf, {}), ("present + ln j main", pres, dict(mainlog=True)),
             ("present + ln j on every neighbour", pres, dict(famlog=tuple(pres))),
             ("present + ln j main and neighbours", pres, dict(mainlog=True, famlog=tuple(pres)))]
    cases += [(f"present + ln j on nu' = {s + NU} only", pres, dict(famlog=(s,))) for s in pres]
    for lab, fams, kw in cases:
        out = [fit(fams, Mf, **kw) for Mf in (6, 8, 10)]
        steps = [float(abs(out[i + 1][0] - out[i][0]).mid()) for i in range(2)]
        print(f"{lab:40s} cols {out[-1][1]:4d}  S at Mf 6, 8, 10: " + ", ".join(AP.mp.nstr(AP.to_mp(o[0]), 14) for o in out)
              + f"  steps {steps[0]:.1e}, {steps[1]:.1e}", flush=True)


if __name__ == "__main__":
    main()
