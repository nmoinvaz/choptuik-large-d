#!/usr/bin/env python3
"""P2b review, comparison step: the certificate's atom coefficients (b8m_atoms.atoms, python-flint) against p2b_series
(exact polynomials in the parameters, enclosed over the box with mpmath intervals), at the box centre and over B8's
box. Both are upper bounds of |coefficient| (summed over harmonics); at the centre they must agree; over the box the
certificate's must not be smaller than the exact range by more than rounding (else its enclosure would miss values).

    python3 p2b_atoms_cmp.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.dirname(HERE)
sys.path.insert(0, PROOF)
ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
import b8m_atoms as AT  # noqa: E402
import p2b_series as Ser  # noqa: E402

NAMES = {"V": "V", "S": "S", "beta": "beta", "c-1": "cm1", "Y": "Y", "Dl": "Dl", "tauD": "tauD", "tauv": "tauv",
         "taub": "taub", "rho4": "rho4", "cb": "cb"}


def compare(rA, rW):
    D, _ = AT.atoms(4, rA, rW)
    T = Ser.tables(float(rA), float(rW))
    worst_low, worst_hi, n, rows = 0.0, 0.0, 0, []
    for cn, mn in NAMES.items():
        if cn not in D:
            continue
        mine = {(j, m): float(C) for (j, m, C) in T[mn][0]}
        cert = {k: float(v) for k, v in D[cn]["coef"].items()}
        for k in set(mine) | set(cert):
            a, b = cert.get(k, 0.0), mine.get(k, 0.0)
            if max(a, b) == 0:
                continue
            n += 1
            rel = (a - b) / max(a, b)
            if rel < worst_low:
                worst_low = rel
            if rel > worst_hi:
                worst_hi = rel
            rows.append((rel, cn, k, a, b))
    rows.sort()
    return n, worst_low, worst_hi, rows


def sample(npts=200, seed=11):
    """the exact polynomials of the four entries where the certificate's bound is closest to (below) p2b's, sampled at
    the 16 corners of the box and npts random points (40 digits): both bounds must exceed the sampled maximum"""
    import itertools
    import random
    from mpmath import mp, mpf, mpc, pi
    mp.dps = 40
    rnd = random.Random(seed)
    fs = Ser.formal(4)
    Vt = Ser.trunc(fs["V"], 4, Ser.ODD)
    Yt = Ser.bv_add(Vt, Ser.trunc(fs["S"], 4, Ser.ODD))

    def ev(r, A, W):
        tot = mpc(0)
        for (i, j), (a, b) in r.t.items():
            tot += mpc(mpf(a.numerator) / a.denominator, mpf(b.numerator) / b.denominator) * A ** i * W ** j
        return tot
    Ac, wc, rr = mpf("2.5421480475"), mpf("0.7769940136"), mpf("2e-6")
    winf = 2 * pi / mpf("1.64389375")
    D, _ = AT.atoms(4, "2e-6", "2e-6")
    T = Ser.tables()
    pts = list(itertools.product([-1, 1], repeat=4)) + [tuple(rnd.uniform(-1, 1) for _ in range(4)) for _ in range(npts)]
    lines = [f"  sampled over the box ({len(pts)} points): entry: sampled max | certificate bound | p2b bound"]
    ok = True
    for nm, obj, key in (("V", Vt, (3, 6)), ("V", Vt, (4, 8)), ("V", Vt, (2, 4)), ("Y", Yt, (2, 4))):
        best = mpf(0)
        for s in pts:
            A = mpc(Ac + s[0] * rr, s[1] * rr)
            W = winf * mpc(wc + s[2] * rr, s[3] * rr)
            best = max(best, sum(abs(ev(obj[h][key], A, W)) for h in Ser.ODD if key in obj[h]))
        mine = {(j, m): float(C) for (j, m, C) in T[nm][0]}[key]
        cert = float(D[nm]["coef"][key])
        ok &= best < cert and best < mine
        lines.append(f"    {nm} {key}: {mp.nstr(best, 12)} | {cert:.12g} | {mine:.12g}")
    lines.append(f"  both bounds above the sampled maximum: {ok}")
    return lines


def main():
    out = []
    for rA, lab in (("0", "centre"), ("2e-6", "box")):
        n, lo, hi, rows = compare(rA, rA)
        out.append(f"{lab}: {n} coefficients; (cert - p2b)/max: smallest {lo:.3e}, largest {hi:.3e}")
        both = [r for r in rows if r[3] > 0 and r[4] > 0]
        only = [r for r in rows if r[4] == 0]
        out.append(f"  nonzero in both: {len(both)}, largest |relative difference| "
                   f"{max(abs(r[0]) for r in both):.3e}; zero in p2b (exactly, by structure) but carried by the "
                   f"certificate: {len(only)}, largest such entry {max((r[3] for r in only), default=0):.3e}")
        if lab == "box":
            out.append("  largest excesses of the certificate's enclosure (cert | p2b):")
            for rel, cn, k, a, b in rows[::-1][:8]:
                out.append(f"    {cn} {k}: {a:.6g} | {b:.6g}  ({rel:+.3f})")
            out.append("  entries where the certificate's bound is below p2b's:")
            for rel, cn, k, a, b in rows[:5]:
                out.append(f"    {cn} {k}: {a:.10g} | {b:.10g}  ({rel:+.2e})")
    out.extend(sample())
    print("\n".join(out))
    with open(os.path.join(HERE, "p2b_atoms_cmp.log"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
