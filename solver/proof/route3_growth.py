#!/usr/bin/env python3
"""P6 route 3: how the j^-2 remainder of the one-A^2-step model grows with nu (notes/P6_route3.md section 9, #986).

route3_cancel.py shows that the O(1/j) parts of DR_j/K_nu cancel identically in nu. What is left is
sum_lam (c2_lam / j^2 + c3_lam / j^3 + ...) rho^lam_j. This script gives

  --sym      c2_lam(nu, w) exactly (sympy) for every family and perturbation amplitude at the A sin theta background,
             the 1/j coefficient of the cancelling piece (de-mixing or psi) of each family, and at the physical w and
             nu = 3, 5, ..., 21: |c2|, j* = |c2| / |piece| (the j below which the remainder exceeds that piece), and
             sup over odd nu >= 3 of |c2(nu)| e^(-sigma nu) at sigma = 0.736 (proof/B_tail_M7.md section 8.4).
  --fit NU   the exact recursion (route3_cancel.dr_builder) at nu = NU, J = 1600, one perturbation at a time, with a
             least-squares fit y_j = j^2 DR_j / K = sum_lam rho^lam_j sum_k a_{lam,k} j^-k over j in [J/4, J] on the
             families whose |rho| is within 1e-12 of the largest. a_{lam,0} is the measured c2, a_{lam,1} the measured
             c3. Its error estimate is the spread over K = 6, 8, 10 terms and the ranges [J/4, J], [J/2, J].
             Pass (fixed before the runs): |a_0 - c2_sym| <= 10 x spread + 1e-30, and spread <= 1e-3 |c2_sym|.
  --planted  the same fit on the harmonic-1 background (nu = 1, labels +-1, no perturbation): it must recover
             D_entire.md section 6.2's j^2 F -> i(7 - w^2 - 2iw)/8 and j^2 G -> i(7 - w^2 + 5iw)/16 by the same pass rule.
  --mutate=demix or psi with --fit or --planted scales E or h2 by 3/2 in the recursion only: must fail.

    python3 route3_growth.py --sym
    python3 route3_growth.py --planted [--mutate=psi]
    python3 route3_growth.py --fit NU [J] [--pert=m0p] [--others] [--wf=X] [--mutate=demix]
                                        --others also checks every other visible family where its fit is resolved;
                                        at small wf (0.05) the families +-1 are visible
    python3 route3_growth.py --all       the planted check, then --fit at nu = 3, 5, ..., 21 (prints a table)
"""
import sys

import mpmath as mp
import sympy as sp

import route3_cancel as R

SIGMA = 0.736
NUS = list(range(3, 23, 2))
MUT = R.MUT
NAMES = {"nu": sp.expand(R.nu), "nu-2": sp.expand(R.nu - 2), "nu+2": sp.expand(R.nu + 2), "+1": sp.Integer(1),
         "-1": sp.Integer(-1)}
AMPS = {"m": R.pm, "0": R.p0, "p": R.pp}
WF = 1.0
mp.mp.dps = 80


def wphys():
    return 2 * mp.pi / mp.mpf("1.64389375") * WF


def sym_coeffs():
    """{(family name, perturbation): (c2(nu, w), cancelling piece's 1/j coefficient)} at b_p = -i, b_m = i"""
    inp, tgt, deg = R.cases()["nu"]
    bg = {R.bp: -sp.I, R.bm: sp.I}
    fam = R.tail(inp, tgt, deg, mutate="")
    famd = R.tail(inp, tgt, deg, mutate="demix")             # E x 3/2: lim j C = (1/2) x the de-mixing piece
    famp = R.tail(inp, tgt, deg, mutate="psi")
    out = {}
    for name, key in NAMES.items():
        c2 = sp.expand(R.laurent(sp.Add.make_args(fam[key]))[3].subs(bg))
        cd = sp.expand(2 * R.laurent(sp.Add.make_args(famd[key]))[2].subs(bg))
        cp = sp.expand(2 * R.laurent(sp.Add.make_args(famp[key]))[2].subs(bg))
        for pk, a in AMPS.items():
            ca = sp.factor(sp.cancel(c2.coeff(a)))
            if ca != 0:
                out[(name, pk)] = (ca, sp.cancel(cd.coeff(a) + cp.coeff(a)))
    return out


def sym():
    mp.mp.dps = 30
    co = sym_coeffs()
    w0 = wphys()
    print(f"w = {mp.nstr(w0, 12)} (wf = {WF}); sigma = {SIGMA}\n")
    print("| family | perturbation | c2(nu, w), exact | degree in nu | cancelling piece, 1/j |")
    print("|---|---|---|---|---|")
    for (name, pk), (c2, piece) in co.items():
        print(f"| {name} | {pk} | {c2} | {sp.Poly(sp.numer(sp.together(c2)), R.nu).degree()} | {piece} |")
    print("\n|c2| at the physical w, and j* = |c2|/|piece|:\n")
    print("| family | perturbation | " + " | ".join(f"nu {n}" for n in NUS) + " |")
    print("|---|---|" + "---|" * len(NUS))
    for (name, pk), (c2, piece) in co.items():
        f = sp.lambdify((R.nu, R.w), c2, "mpmath")
        vals = [abs(f(n, w0)) for n in NUS]
        print(f"| {name} | {pk} | " + " | ".join(mp.nstr(v, 4) for v in vals) + " |")
        print(f"| {name} | j* | " + " | ".join(mp.nstr(v / abs(complex(piece)), 4) for v in vals) + " |")
    print(f"\nsup over odd nu >= 3 of |c2(nu)| exp(-{SIGMA} nu), and its argmax (nu up to 401):\n")
    for (name, pk), (c2, piece) in co.items():
        f = sp.lambdify((R.nu, R.w), c2, "mpmath")
        best = max((abs(f(n, w0)) * mp.exp(-SIGMA * n), n) for n in range(3, 403, 2))
        print(f"  {name:5s} {pk}: {mp.nstr(best[0], 4)} at nu = {best[1]}")
    return True


# ---------------------------------------------------------------- the fit on the exact recursion
def fit(y, rho, js, K):
    """least squares y_j = sum_lam rho_lam(j) sum_{k<K} a_{lam,k} j^-k; returns {lam: [a_0, a_1]}"""
    lams = list(rho)
    rows = []
    for idx, jv in enumerate(js):
        rows.append([rho[lam][idx] * mp.mpf(jv) ** (-k) for lam in lams for k in range(K)])
    A = mp.matrix(rows)
    sc = [max(abs(A[i, c]) for i in range(A.rows)) for c in range(A.cols)]
    for c in range(A.cols):
        for i in range(A.rows):
            A[i, c] /= sc[c]
    x, _ = mp.qr_solve(A, mp.matrix(y))
    return {lam: [x[n * K] / sc[n * K], x[n * K + 1] / sc[n * K + 1]] for n, lam in enumerate(lams)}


def measure(labels, NU, J, deg, mut):
    """the fitted a_0, a_1 of every visible family, their spreads over the fit variants, and the max relative radius"""
    from flint import acb, arb, ctx
    ctx.prec = 512
    mp.mp.dps = 80
    wv = acb(arb(str(WF)) * 2 * arb.pi() / arb("1.64389375"))
    lab = [(lam, acb(*d0), g) for lam, d0, g in labels]
    fams = [lam for lam, _, _ in labels]
    if NU not in fams:                                            # T^NU is needed; a zero input adds nothing
        lab.append((NU, acb(0), 1))
    DR, T, TN = R.dr_builder(lab, NU, J, wv, deg, mut)
    js = sorted({int(round(J / 4 * 4 ** (k / 127))) for k in range(128)})
    vals = {jv: DR(jv) for jv in js}
    rad = max(float(vals[jv].rad()) / max(float(abs(vals[jv]).mid()), 1e-300) for jv in js)
    tomp = lambda z: mp.mpc(mp.mpf(z.real.str(70, radius=False)), mp.mpf(z.imag.str(70, radius=False)))
    rho_all = {lam: [tomp(T[lam][jv] / TN[jv]) for jv in js] for lam in fams}
    top = max(abs(rho_all[lam][0]) for lam in rho_all)
    vis = {lam: r for lam, r in rho_all.items() if abs(r[0]) > 1e-12 * top}
    y_all = [tomp(vals[jv]) * jv * jv for jv in js]
    res = []
    for K in (6, 8, 10):
        for lo in (J // 4, J // 2):
            sel = [i for i, jv in enumerate(js) if jv >= lo]
            res.append(fit([y_all[i] for i in sel], {lam: [r[i] for i in sel] for lam, r in vis.items()},
                           [js[i] for i in sel], K))
    ref = res[2]                                                  # K = 8 on [J/4, J]
    spread = {lam: [max(abs(r[lam][k] - ref[lam][k]) for r in res) for k in (0, 1)] for lam in vis}
    return ref, spread, rad, vis


def check(name, a0, sp_, want):
    ok = abs(a0 - want) <= 10 * sp_ + mp.mpf(10) ** -30 and sp_ <= 1e-3 * abs(want)
    print(f"  {name}: fit {mp.nstr(a0, 12)}  symbolic {mp.nstr(want, 12)}  |diff| {mp.nstr(abs(a0 - want), 3)}"
          f"  spread {mp.nstr(sp_, 3)}  {'ok' if ok else '** FAIL **'}", flush=True)
    return ok


def planted(J=1600):
    w0 = wphys()
    ref, spread, rad, vis = measure([(1, (0, -1), 0), (-1, (0, 1), 0)], 1, J, 0, MUT)
    print(f"planted: harmonic 1 at order 3, J = {J}, max relative radius {rad:.1e}")
    ok = check("j^2 F (family +1)", ref[1][0], spread[1][0], 1j * (7 - w0**2 - 2j * w0) / 8)
    ok &= check("j^2 G (family -1)", ref[-1][0], spread[-1][0], 1j * (7 - w0**2 + 5j * w0) / 16)
    return ok


def expected(co, pk, NU, lab, w0):
    """symbolic c2 per unit amplitude of the family with label lab at this nu (coincident labels summed)"""
    return sum(sp.lambdify((R.nu, R.w), co[(nm, q)][0], "mpmath")(NU, w0) for (nm, q) in co
               if q == pk and NAMES[nm].subs(R.nu, NU) == lab)


def fitnu(NU, J=1600, table=None, perts=("m", "0", "p"), others=False):
    """the dominant family's c2 against the symbolic one for each perturbation; with others, every other visible
    family too, which is checked by the same rule where its fit is resolved (spread <= 1e-3 of the symbolic value)"""
    co = sym_coeffs()
    w0 = wphys()
    ok = True
    spec = {"m": (-2, (1, 0)), "0": (0, ("0.7", "0.3")), "p": (2, ("-0.4", "1.1"))}
    for pk in perts:
        sh, amp = spec[pk]
        labels = [(1, (0, -1), 0), (-1, (0, 1), 0), (NU + sh, amp, 1)]
        ref, spread, rad, vis = measure(labels, NU, J, 1, MUT)
        ampv = mp.mpc(mp.mpf(str(amp[0])), mp.mpf(str(amp[1])))
        dom = NU + sh
        name = {-2: "nu-2", 0: "nu", 2: "nu+2"}[sh]
        print(f"nu = {NU}, wf = {WF}, perturbation {pk} (label {dom}), J = {J}, families fitted {sorted(vis)}, max "
              f"relative radius {rad:.1e}")
        c2 = expected(co, pk, NU, dom, w0) * ampv
        good = check(f"c2, family {name} (label {dom})", ref[dom][0], spread[dom][0], c2)
        print(f"  c3, family {name}: fit {mp.nstr(ref[dom][1] / ampv, 8)} per unit amplitude, spread "
              f"{mp.nstr(spread[dom][1] / abs(ampv), 3)}", flush=True)
        if others:
            for lam in sorted(vis):
                if lam == dom:
                    continue
                want = expected(co, pk, NU, lam, w0) * ampv
                if spread[lam][0] <= 1e-3 * abs(want):
                    good &= check(f"c2, family {lam} (resolved)", ref[lam][0], spread[lam][0], want)
                else:
                    print(f"  c2, family {lam}: not resolved (spread {mp.nstr(spread[lam][0], 3)}, symbolic "
                          f"{mp.nstr(abs(want), 3)})")
        ok &= good
        if table is not None:
            table.append((NU, name, ref[dom][0] / ampv, c2 / ampv, spread[dom][0] / abs(ampv), ref[dom][1] / ampv,
                          spread[dom][1] / abs(ampv), good))
    return ok


def slopes(table):
    """least-squares slope of log|c| against log nu per family, for c2 (symbolic) and c3 (fit), over nu >= 9"""
    out = {}
    for name in ("nu-2", "nu", "nu+2"):
        rows = [(mp.log(NU), mp.log(abs(c2)), mp.log(abs(a1))) for NU, nm, a0, c2, s0, a1, s1, g in table
                if nm == name and NU >= 9]
        n = len(rows)
        mx = sum(r[0] for r in rows) / n
        sl = []
        for col in (1, 2):
            my = sum(r[col] for r in rows) / n
            sl.append(sum((r[0] - mx) * (r[col] - my) for r in rows) / sum((r[0] - mx) ** 2 for r in rows))
        out[name] = sl
    return out


def allnu(J=1600):
    ok = planted(J)
    table = []
    for NU in NUS:
        ok &= fitnu(NU, J, table)
    sl = slopes(table)
    print("\nslope of log|c| against log nu, nu = 9 to 21 (numerical evidence for c3):")
    for name, (s2, s3) in sl.items():
        print(f"  {name:5s}: c2 (symbolic) {mp.nstr(s2, 4)},  c3 (fit) {mp.nstr(s3, 4)}")
    print("\n| ν | family | c2 symbolic | c2 fit − symbolic | fit spread | c3 fit | c3 spread | pass |")
    print("|---|---|---|---|---|---|---|---|")
    for NU, name, a0, c2, s0, a1, s1, good in table:
        print(f"| {NU} | {name} | {mp.nstr(c2, 8)} | {mp.nstr(abs(a0 - c2), 2)} | {mp.nstr(s0, 2)} | {mp.nstr(a1, 6)} | "
              f"{mp.nstr(s1, 2)} | {'yes' if good else 'no'} |")
    return ok


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--mutate=")]
    WF = float(next((a.split("=", 1)[1] for a in args if a.startswith("--wf=")), "1"))
    if "--sym" in args:
        good = sym()
    elif "--planted" in args:
        good = planted()
    elif "--fit" in args:
        k = args.index("--fit")
        NU = int(args[k + 1])
        J = int(args[k + 2]) if len(args) > k + 2 and args[k + 2].isdigit() else 1600
        perts = next((a.split("=", 1)[1] for a in args if a.startswith("--pert=")), "m0p")
        good = fitnu(NU, J, perts=tuple(perts), others="--others" in args)
    elif "--all" in args:
        good = allnu()
    else:
        print(__doc__)
        sys.exit(2)
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
