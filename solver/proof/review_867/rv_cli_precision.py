#!/usr/bin/env python3
"""Review of fixed_point_M_b.py (#867), check K4: the precision at which the command line reads its point.

main() builds x = [acb(arb(v)) ...] from --x before Setup sets the working precision, so on the plain path
(python3 fixed_point_M_b.py --M=5 --x=...) the point is read at python-flint's default 53 bits and becomes a ball of
radius about 1e-16. --start, --planted and --two set the precision first. This script runs Setup and prove at the
M = 5 certificate's centre both ways and prints the truncation N and the proved radius of eta. With --ray= it does
the same at the centre that the stored start set records, which is the certificate's cut to 38 decimals. It also
runs the module's guard on the weight exponent.

    python3 rv_cli_precision.py [--ray=<raw clone>/I38-macm3/lemma6b1_runs_M5_cert/ray.json]
                                    exit 0 if every run proves the contraction and the guard refuses k = N - 1
"""
import json
import os
import sys

from flint import acb, arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import fixed_point_M_b as B  # noqa: E402  (the module under review)


def run(x, label):
    st = B.Setup(x, 5, "0.07", "3pi/8")
    o = B.prove(st, False)
    rad = {t: max(float(B.eta_radius(o, cp, st.R)) for cp in st.comps if cp[0] == t) for t in "pvc"}
    print(f"{label}: N = {st.N}, k = {st.k}, contraction constant {float(o['lam']):.4f}, proved {o['ok']}, |eta| at "
          f"|z| = R: " + ", ".join(f"{t} {v:.2e}" for t, v in rad.items()))
    return o["ok"], st.N


def main():
    with open(os.path.join(HERE, "..", "data", "A_M2_box_M5_1e-10.json")) as fh:
        X = json.load(fh)["p"]
    print(f"python-flint's precision when main() reads --x: {ctx.prec} bits")
    x53 = [acb(arb(v)) for v in X]                     # as main() does on the plain path
    print(f"  the centre's A read at {ctx.prec} bits is a ball of radius {float(x53[0].real.rad()):.1e}")
    ok1, n1 = run(x53, "  read at 53 bits (the plain command line)")
    ctx.prec = 512
    x512 = [acb(arb(v)) for v in X]
    print(f"  read at 512 bits its radius is {float(x512[0].real.rad()):.1e}")
    ok2, n2 = run(x512, "  read at 512 bits (--start, --planted, --two)")
    ok3 = True
    ray = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--ray=")]
    if ray:
        with open(ray[0]) as fh:
            xr = [acb(arb(v)) for v in json.load(fh)["x"]]
        d = max(float(abs(a.real - b.real).upper()) for a, b in zip(xr, x512))
        ok3, _ = run(xr, f"  the stored start set's centre ({d:.1e} from the certificate's), read at 512 bits")
    refused = False
    try:
        B.Setup(x512, 5, "0.07", "3pi/8", None, 105, 104)
    except ValueError as e:
        refused = True
        print(f"the guard on the weight exponent, N = 105 and k = 104: refused ({e})")
    ok = ok1 and ok2 and ok3 and refused
    print(f"RV_CLI_PRECISION: {'both proved' if ok1 and ok2 else 'NOT both proved'}; N = {n1} at 53 bits against "
          f"{n2} at 512 bits; guard {'fires' if refused else 'DOES NOT FIRE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
