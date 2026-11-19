#!/usr/bin/env python3
"""The c2 table of notes/L2_MODE.md 9 from l2_c2.py's result files, with the verdicts fixed before the runs (#1010).

    planted   lambda_1 of the l = 0 forcing through the same code (l0, and SymForce: sym) at M = 5 within 1% of the
              double-double fit 3.0324 (FIRST_ORDER.md 6); lambda* of M = 1 = 1.769122008 to 1e-8
    control   the forcing with the leading background only (l2_ctrl) moves c2 by more than ten times the slice spread
              and |c2 + lambda_1| at M = 3 and 5
    sign      read only if M = 3 and M = 5 agree on it
    mode      d_s Z0 is an exact mode of each model: its corrected jump below 1e-5 of |Sch| |f2|

    python3 l2_c2_summary.py          exit 0 iff the planted checks, the control and the mode check hold
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
FIT = 3.0324


def load(M, names):
    fn = os.path.join(D, f"l2_c2_M{M}_xi0.2_{names}.json")
    return json.load(open(fn)) if os.path.exists(fn) else None


def main():
    ok = True
    rows = []
    for M in (1, 3, 5):
        a = load(M, "l2-l2_ctrl-l0-sym")
        b = load(M, "l2-l0-G-sym")
        if a is None:
            print(f"M = {M}: no results")
            ok = False
            continue
        c2, l1 = a["l2"]["ell"], a["l0"]["ell"]
        sp_ = [a["l2"][k] for k in ("e1", "e2", "f2")]
        rows.append(dict(M=M, c2=c2, c2lo=min(sp_ + [c2]), c2hi=max(sp_ + [c2]), l1=l1, sym=a["sym"]["ell"],
                         sumv=c2 + l1, G=(b["G"]["ell"] if b else None), ctrl=a["l2_ctrl"]["ell"],
                         mode=a["mode_jump"] / a["mode_scale"], lamc=(b["lamcheck"] if b else None),
                         lame=(b["lamcheck_ell"] if b else None)))
    print("| M | c₂ (ℓ of Sch) | c₂ over φ | λ₁, l0 (same code) | λ₁, SymForce | c₂ + λ₁ | G alone | c₂, control |"
          " mode jump | F_λ relation | F_λ, ℓ projection |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    f = lambda v, d=4: "—" if v is None else f"{v:.{d}f}"
    e = lambda v: "—" if v is None else f"{v:.1e}"
    for r in rows:
        print(f"| {r['M']} | {f(r['c2'])} | {f(r['c2lo'])} to {f(r['c2hi'])} | {f(r['l1'])} | {f(r['sym'])} | "
              f"{f(r['sumv'])} | {f(r['G'])} | {f(r['ctrl'], 2)} | {e(r['mode'])} | {e(r['lamc'])} | {e(r['lame'])} |")
    by = {r["M"]: r for r in rows}
    # verdicts
    pl = json.load(open(os.path.join(D, "l2_c2_M1_xi0.2_planted_lam.json")))
    v = abs(pl["lam"] - pl["want"]) < 1e-8
    ok &= v
    print(f"planted lambda* (M = 1): {pl['lam']:.10f} against {pl['want']}: {'OK' if v else 'BAD'} (bar 1e-8)")
    if 5 in by:
        for key, lab in (("l1", "l0 (the converted forcing, the l = 2 code path)"), ("sym", "SymForce")):
            rel = abs(by[5][key] - FIT) / FIT
            v = rel <= 0.01
            ok &= v
            print(f"planted lambda_1 (M = 5), {lab}: {by[5][key]:.4f} against {FIT}, {100 * rel:.2f}%: "
                  f"{'OK' if v else 'BAD'} (bar 1%)")
    for M in (3, 5):
        if M in by:
            r = by[M]
            bar = 10 * max(r["c2hi"] - r["c2lo"], abs(r["sumv"]))
            v = abs(r["ctrl"] - r["c2"]) > bar
            ok &= v
            print(f"control (M = {M}): |c2_ctrl - c2| = {abs(r['ctrl'] - r['c2']):.3f} against 10 x max(slice spread, "
                  f"|c2 + lambda_1|) = {bar:.3f}: {'OK' if v else 'BAD'}")
    for r in rows:
        v = r["mode"] < 1e-5
        ok &= v
        print(f"mode (M = {r['M']}): corrected jump of d_s Z0 / (|Sch| |f2|) = {r['mode']:.1e}: {'OK' if v else 'BAD'} "
              f"(bar 1e-5)")
    if 3 in by and 5 in by:
        print(f"step M = 3 to 5: c2 {by[5]['c2'] - by[3]['c2']:+.4f}, lambda_1 {by[5]['l1'] - by[3]['l1']:+.4f}; "
              f"their sum {by[5]['c2'] - by[3]['c2'] + by[5]['l1'] - by[3]['l1']:+.4f}")
        s3, s5 = by[3]["c2hi"] < 0, by[5]["c2hi"] < 0
        same = (by[3]["c2lo"] > 0 and by[5]["c2lo"] > 0) or (s3 and s5)
        print(f"sign: M = 3 range {by[3]['c2lo']:.4f} to {by[3]['c2hi']:.4f}, M = 5 range {by[5]['c2lo']:.4f} to "
              f"{by[5]['c2hi']:.4f}: {'both negative' if s3 and s5 else 'both positive' if same else 'do not agree'}")
    return ok


if __name__ == "__main__":
    good = main()
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
