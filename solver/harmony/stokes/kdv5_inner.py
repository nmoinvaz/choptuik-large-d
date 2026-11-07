"""Late-order constants of two inner equations of embedded-soliton problems, by exact recursion in ball arithmetic
(python-flint arb) and Richardson extrapolation in 1/N.
 (Q) phi'''' + phi'' + c phi^2 = 0, phi ~ -2/z^2 (c = 3): inner problem of eps^2 u'''' + u'' + 3u^2 - cu = 0 (Trinh
     arXiv:1410.4124, quoting Grimshaw's Lambda ~ -19.97). phi = sum a_N z^(-2N-2); Lambda_N = a_N/((-1)^N Gamma(2N+2)).
 (C) psi'''' + psi'' - 2 psi^3 = 0, psi ~ 1/z: the modified-Kawahara inner equation of Baldoma-Guardia-Pelinovsky
     (CMP 2025, arXiv:2412.01563) after psi = sqrt|gamma| phi. psi = sum b_N z^(-2N-1); mu_N = b_N/((-1)^N Gamma(2N+1)).
Criteria (fixed before the run): two ladders (N = 500 and 1000) agree to >= 30 digits; (Q) has |Lambda| within 0.01 of
19.97; the mutation c = 3.03 in (Q) must move |Lambda| outside that window (|Lambda| scales like 1/c at leading
order, so it fails by about 0.2). Exit 1 if a criterion fails."""
import sys
from flint import arb, ctx
ctx.prec = 3400

def quad_series(N, c=3):
    a = [arb(-6) / c]                      # 6 a0 + c a0^2 = 0
    sq = []                                # running sums handled directly (O(N^2) total)
    for n in range(1, N + 1):
        conv = sum((a[i] * a[n - i] for i in range(1, n)), arb(0))
        rhs = -a[n - 1] * (2 * n) * (2 * n + 1) * (2 * n + 2) * (2 * n + 3) - c * conv
        a.append(rhs / ((2 * n + 2) * (2 * n + 3) + 2 * c * a[0]))
    return a

def cubic_series(N):
    b = [arb(1)]
    s = [arb(1)]                           # s = psi^2 coefficients (in z^(-2N-2))
    for n in range(1, N + 1):
        s_part = sum((b[i] * b[n - i] for i in range(1, n)), arb(0))        # s_n without the 2 b0 b_n term
        cube_part = sum((s[i] * b[n - i] for i in range(1, n)), arb(0)) + s_part * b[0]   # without b_n terms
        # cube_n = s0 b_n + s_n b0 + sum_{1<=i<=n-1} s_i b_{n-i};  s_n = 2 b0 b_n + s_part
        rhs = -b[n - 1] * (2 * n - 1) * (2 * n) * (2 * n + 1) * (2 * n + 2) + 2 * cube_part
        bn = rhs / ((2 * n + 1) * (2 * n + 2) - 2 * (s[0] + 2 * b[0] * b[0]))
        b.append(bn)
        s.append(2 * b[0] * bn + s_part)
    return b

def richardson(vals, Ns, order):
    """seq L_N = L + c1/N + ... : Richardson on the last order+1 points"""
    from flint import arb_mat
    pts = list(zip(Ns, vals))[-(order + 1):]
    M = arb_mat([[arb(1) / arb(n) ** j for j in range(order + 1)] for n, _ in pts])
    v = arb_mat([[val] for _, val in pts])
    return M.solve(v)[0, 0]

def estimate(seq, shift, Nmax, order=60, step=4):
    Ns = list(range(Nmax - step * order, Nmax + 1, step))
    vals = [seq[n] * (-1) ** n / arb.gamma(arb(2 * n + shift)) for n in Ns]
    return richardson(vals, Ns, order)

ok = True
out = {}
for lab, build, shift in (("Q", lambda N: quad_series(N), 2), ("C", lambda N: cubic_series(N), 1)):
    seq = build(1000)
    e1 = estimate(seq, shift, 500); e2 = estimate(seq, shift, 1000)
    d = abs(e1 - e2)
    print(f"({lab}) N=500: {e1.mid().str(45)}\n    N=1000: {e2.mid().str(45)}\n    difference {d.mid().str(3)}")
    out[lab] = e2.mid().str(45)
    ok &= d < arb(10) ** -30
    if lab == "Q":
        ok &= abs(abs(e2) - arb("19.97")) < arb("0.01")
mut = quad_series(600, c=arb("3.03"))
em = estimate(mut, 2, 600, order=40)
print(f"(Q) control, c = 3.03: {em.mid().str(20)}  (must leave |Lambda| in 19.96..19.98)")
ok &= abs(abs(em) - arb("19.97")) > arb("0.01")
print("PASS" if ok else "FAIL")
import json; json.dump(out, open("../inner_constants.json", "w"), indent=1)
sys.exit(0 if ok else 1)
