#!/usr/bin/env python3
"""Integer-relation (LLL) test of sigma_1 against a frozen basket of constants, with planted and random controls.

sigma_1 = S_1^(3)/S_1^(1) is computed at omega = 4 exactly (sigma1_hp.py), so that any rational function of omega in a
closed form becomes a Gaussian rational and the basket needs no powers of omega. The basket is frozen here, before the
target is read; its choice follows the structure of the exact order-3 recursion (solver/proof/exact_sigma.py: digamma
normalisation psi(a) + psi(b), the de-mixing 1/(i omega), the rotating ratio T^-/T^+, Gauss-type sums of Gamma ratios),
with z = i omega, a = (1 + z)/2:

    1, ln 2, pi, psi(1 + z), psi(a), psi'(1 + z), pi coth(pi omega), Gamma(1 - z)/Gamma(1 + z),
    4^z Gamma(a)^2/Gamma(1 + z) (Gauss/Kummer-type value), pi/Gamma(a)^2 ... (see BASKET below)

A Gaussian-integer relation sum_k g_k x_k = 0 among complex x_k is sought by LLL (python-flint fmpz_mat.lll; Lenstra,
Lenstra, Lovasz, Math. Ann. 261 (1982) 515) on the lattice of rows (e_k, round(10^d Re y_k), round(10^d Im y_k)) with
y = (x_1, i x_1, x_2, i x_2, ...). Criterion, fixed before the run: a relation is accepted only if it involves the target,
its coefficients are at most H = 10^3 in modulus, its residual is below 10^(5 - d), and the same relation is returned at
d - 6 digits. Controls through the same code and the same digits:

    planted:  a random Gaussian-integer combination (height <= 30) of the basket, rounded to d digits: must be found;
    linear:   log of the fitted linear multiplier S^(1) (from the same late-order fit) against 1, i pi, ln 2, ln pi,
              lnGamma(1 + z): the exact S^(1) = -i 2^z/(sqrt(pi) Gamma(1 + z)) must be recovered;
    random:   a random complex number in place of sigma_1: must give no accepted relation.

    python3 sigma_lll.py RUN_A.json RUN_B.json [--mutate=perturb]

RUN_A, RUN_B: two sigma1_hp.py outputs at omega = 4 with different J; their agreement sets d. Exit status 1 if any
control misbehaves. --mutate=perturb adds 10^(-d/2) to the planted targets, which are then not found, so the run must
exit 1.
"""
import json
import random
import sys

import mpmath as mp
from flint import fmpz_mat

H_MAX = 10 ** 3


def baskets(w):
    """two frozen baskets. A: polygamma values (psi(1 - z), psi(conj a), psi'(1 - z) are in its span through the
    reflection and duplication formulas, the reflection terms giving pi coth(pi w), pi coth(pi w/2) and pi^2);
    B: Gamma ratios of the kind Gauss- and Kummer-type sums of T^+-/T^- produce, with digamma partners."""
    z = mp.mpc(0, w)
    a = (1 + z) / 2
    b = (2 + z) / 2
    r1 = mp.gamma(1 - z) / mp.gamma(1 + z)
    r2 = mp.power(2, z) * mp.gamma(a) / mp.gamma(b)
    A = [
        ("1", mp.mpc(1)),
        ("ln2", mp.log(2)),
        ("pi", mp.pi),
        ("pi^2", mp.pi ** 2),
        ("psi(1+z)", mp.digamma(1 + z)),
        ("psi(a)", mp.digamma(a)),
        ("psi'(1+z)", mp.psi(1, 1 + z)),
        ("pi coth(pi w)", mp.pi * mp.coth(mp.pi * w)),
        ("pi coth(pi w/2)", mp.pi * mp.coth(mp.pi * w / 2)),
    ]
    B = [
        ("1", mp.mpc(1)),
        ("ln2", mp.log(2)),
        ("pi", mp.pi),
        ("psi(1+z)", mp.digamma(1 + z)),
        ("G(1-z)/G(1+z)", r1),
        ("2^z G(a)/G(b)", r2),
        ("G(1-z)/G(1+z) psi(1-z)", r1 * mp.digamma(1 - z)),
        ("2^z G(a)/G(b) psi(a)", r2 * mp.digamma(a)),
    ]
    return {"A (polygamma)": A, "B (Gamma ratios)": B}


def lll_relation(xs, d):
    """shortest Gaussian-integer relation for the complex list xs at d digits: (coeffs, residual)"""
    ys = []
    for x in xs:
        ys += [x, mp.mpc(0, 1) * x]
    n = len(ys)
    scale = mp.mpf(10) ** d
    rows = []
    for k in range(n):
        r = [0] * n
        r[k] = 1
        r += [int(mp.nint(scale * ys[k].real)), int(mp.nint(scale * ys[k].imag))]
        rows.append(r)
    M = fmpz_mat(rows).lll()
    best = None
    for i in range(M.nrows()):
        v = [int(M[i, k]) for k in range(n)]
        if not any(v):
            continue
        g = [complex(v[2 * k], v[2 * k + 1]) for k in range(len(xs))]
        res = abs(sum(mp.mpc(gg.real, gg.imag) * x for gg, x in zip(g, xs)))
        h = max(abs(gg) for gg in g)
        if best is None or h < best[2]:
            best = (g, res, h)
        break                                   # first row of an LLL-reduced basis is the short one
    return best


def accept(xs, d, label):
    g, res, h = lll_relation(xs, d)
    g2, res2, h2 = lll_relation(xs, d - 6)
    same = all(abs(a - b) == 0 for a, b in zip(g, g2)) or all(abs(a + b) == 0 for a, b in zip(g, g2))
    involves = abs(g[0]) > 0
    ok = involves and h <= H_MAX and res < mp.mpf(10) ** (5 - d) and same
    print(f"   {label}: height {h:.3g}, residual {mp.nstr(res, 3)}, stable at d-6: {same}, involves target: "
          f"{involves} -> {'RELATION' if ok else 'none'}")
    return ok, g


def main(argv):
    mutate = "--mutate=perturb" in argv
    files = [a for a in argv[1:] if not a.startswith("--")]
    runs = [json.load(open(f)) for f in files]
    mp.mp.dps = 90
    w = mp.mpf(runs[0]["omega"])
    assert all(mp.mpf(r["omega"]) == w for r in runs)
    s = [mp.mpc(mp.mpf(r["sigma1"][0]), mp.mpf(r["sigma1"][1])) for r in runs]
    S1 = [mp.mpc(mp.mpf(r["S1"][0]), mp.mpf(r["S1"][1])) for r in runs]
    agree = -mp.log10(abs(s[0] - s[1]) / abs(s[0]))
    d = int(agree) - 2
    best = max(range(len(runs)), key=lambda i: runs[i]["J"])
    sig = s[best]
    print(f"omega = {mp.nstr(w, 20)}; sigma_1 = {mp.nstr(sig, 40)}")
    print(f"two fit windows (J = {[r['J'] for r in runs]}) agree to {mp.nstr(agree, 4)} digits; using d = {d}")
    bad = 0
    random.seed(20261106)
    found = []
    for bname, B in baskets(w).items():
        names = [n for n, _ in B]
        vals = [v for _, v in B]
        print(f"basket {bname}: {', '.join(names)}")
        g = [complex(random.randint(-30, 30), random.randint(-30, 30)) for _ in vals]
        t = sum(mp.mpc(x.real, x.imag) * v for x, v in zip(g, vals))
        if mutate:
            t += mp.mpf(10) ** (-d // 2)
        t = mp.mpc(mp.nstr(t.real, d + 1), mp.nstr(t.imag, d + 1))
        ok, _ = accept([t] + vals, d, "planted (height 30)")
        if not ok:
            bad += 1
        r = mp.mpc(mp.rand(), mp.rand())
        r = mp.mpc(mp.nstr(r.real, d + 1), mp.nstr(r.imag, d + 1))
        ok, _ = accept([r] + vals, d, "random")
        if ok:
            bad += 1
        sg = mp.mpc(mp.nstr(sig.real, d + 1), mp.nstr(sig.imag, d + 1))
        ok, gs = accept([sg] + vals, d, "sigma_1")
        if ok:
            found.append((bname, list(zip(["sigma_1"] + names, gs))))
            print(f"      CANDIDATE relation: {found[-1][1]}")
        print(f"   a random target gives LLL height ~ 10^{mp.nstr(d / (len(vals) + 1), 3)} at d = {d}; "
              f"the null covers heights up to {H_MAX}")

    print("linear-multiplier control: log S^(1) from the late-order fit")
    z = mp.mpc(0, w)
    lin = [mp.log(S1[best]), mp.mpc(1), mp.mpc(0, 1) * mp.pi, mp.log(2), mp.log(mp.pi), mp.loggamma(1 + z)]
    lin[0] = mp.mpc(mp.nstr(lin[0].real, d + 1), mp.nstr(lin[0].imag, d + 1))
    ok, gl = accept(lin, d, "linear")
    print(f"      relation {gl}  (exact: 2 log S1 = -i pi + 2 z ln2 - ln pi - 2 lnGamma(1+z), up to 2 pi i k)")
    if not ok:
        bad += 1
    print(f"sigma_1 relations accepted: {len(found)}")
    print(f"controls misbehaving: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
