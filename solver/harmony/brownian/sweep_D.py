"""alpha_max(D) for a list of D, by continuation in D from the first entry (sp_bvp2.py).
Each D: planted alpha = 2 check (S(0) = 4/(D-2)), then the maximum of alpha over z.
X is chosen so that the inward error suppression e^{-(f(X)-f(xm))} beats the target digits,
f(x) = x^2/2 - (D+2-2 alpha) ln x, with xm = max(2, sqrt(D)). Appends to sweep_D.jsonl."""
import sys, json, time, math
import flint
from flint import arb
import sp_bvp2 as B

prec = int(sys.argv[1]); flint.ctx.prec = prec
Ds = [float(d) for d in sys.argv[2].split(',')]
out = sys.argv[3] if len(sys.argv) > 3 else 'sweep_D.jsonl'
xpad = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
digits = prec*0.30103
zprev = None
for D in Ds:
    t0 = time.time()
    xm = max(2.0, math.sqrt(D))
    f = lambda x: x*x/2 - (D + 2 - 4.4)*math.log(x)
    X = xm + 1
    while f(X) - f(xm) < (digits + 6)*math.log(10):
        X += 0.25
    X += xpad
    Ds_ = repr(D) if D != int(D) else str(int(D))
    P = B.Prob(Ds_, '%.2f' % X, '%.4f' % xm, 28)
    tol = arb(2)**(-prec + 20)
    Dv = arb(Ds_)
    zc = 2*Dv/(Dv - 2)
    # planted truth: alpha(z) at the canonical z = 2D/(D-2) must be exactly 2
    ap, Ap = B.alpha_of_z(P, zc, P.A_guess(zc, arb('2.001')), arb('2.001'), tol)
    rel = abs(ap - 2)
    if zprev is None:
        z0 = 4 + (arb(41)/2 - 6*arb(2).log())/Dv
    else:
        z0 = zprev
    al0 = 2 + 1/(2*Dv) + arb(11)/(16*Dv*Dv)
    am, zs, As, d2 = B.alpha_max(P, z0, P.A_guess(z0, al0), al0, tol, verbose=False)
    zprev = zs
    rec = dict(D=Ds_, prec=prec, X='%.2f' % X, xm='%.4f' % xm, N=28, alpha_max=am.str(50, radius=False),
               z=zs.str(30, radius=False), planted_rel=rel.str(4, radius=False), secs=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True)
    with open(out, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')
