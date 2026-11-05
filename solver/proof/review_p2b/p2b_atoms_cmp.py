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


def main():
    out = []
    for rA, lab in (("0", "centre"), ("2e-6", "box")):
        n, lo, hi, rows = compare(rA, rA)
        out.append(f"{lab}: {n} coefficients; (cert - p2b)/max: smallest {lo:.3e}, largest {hi:.3e}")
        if lab == "box":
            out.append("  largest excesses of the certificate's enclosure (cert | p2b):")
            for rel, cn, k, a, b in rows[::-1][:8]:
                out.append(f"    {cn} {k}: {a:.6g} | {b:.6g}  ({rel:+.3f})")
            out.append("  entries where the certificate's bound is below p2b's:")
            for rel, cn, k, a, b in rows[:5]:
                out.append(f"    {cn} {k}: {a:.10g} | {b:.10g}  ({rel:+.2e})")
    print("\n".join(out))
    with open(os.path.join(HERE, "p2b_atoms_cmp.log"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
