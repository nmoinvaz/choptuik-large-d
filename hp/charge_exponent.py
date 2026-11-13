#!/usr/bin/env python3
"""The charge exponent delta(D) = (D - 2 - mu) gamma of a weakly charged scalar near the critical solution (#621), from
the slowest decaying real test-field rate mu (hp/modecount_test_axis.py's records) and our gamma (the data files that
hp/compare_gamma_published.py reads). The formula is Mittal, Gundlach and Martel's (arXiv:2609.17207, as quoted in
matter/LITERATURE.md section 8); at D = 5 they give mu = -0.448, gamma = 0.41325 and delta_Q = 1.425.

    cd hp && python3 charge_exponent.py RAWDIR [--mutate=d3]

RAWDIR is a clone of choptuik-large-d-raw; the records are I621-macm3/axis_*.jsonl (M 48) and, where present,
I632-macm3/axis_*.jsonl (M 64). Checks (exit 1 on failure): the formula on Mittal, Gundlach and Martel's own mu and
gamma reproduces their delta_Q = 1.425 to its last digit; and #621's control, mu(4) within 1e-5 of -0.36512 (notes/
GHOST_ADMIXTURE.md section 4). Control: --mutate=d3 puts D - 3 in place of D - 2, and the first check must fail.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_gamma_published import ours_sources  # noqa: E402

MGM = dict(D=5.0, mu=-0.448, gamma=0.41325, delta=1.425)
MU4 = -0.36512
FOLDERS = (("I621-macm3", 48), ("I632-macm3", 64))


def delta(D, mu, gamma, mutate=""):
    """(D - 2 - mu) gamma; the control d3 takes D - 3"""
    return ((D - 3 if mutate == "d3" else D - 2) - mu) * gamma


def slowest(rec):
    """the slowest decaying real zero of a record: the largest Re among its zeros on the real axis"""
    z = [q["re"] for q in rec.get("zeros", []) if abs(q.get("im", 0.0)) < 1e-12]
    return max(z) if z else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    if len(args) != 1:
        sys.exit(__doc__)
    raw, mutate = args[0], opts.get("mutate", "")
    if mutate not in ("", "d3"):
        sys.exit(f"--mutate={mutate}: d3")
    mus = {}
    for folder, M in FOLDERS:
        for f in sorted(glob.glob(os.path.join(raw, folder, "axis_*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                mu = slowest(r)
                if mu is not None:
                    mus[(round(r["D"], 6), r["M"])] = (mu, f"{folder}/{os.path.basename(f)}")
    gam, _ = ours_sources()
    ok = True
    d_mgm = delta(MGM["D"], MGM["mu"], MGM["gamma"], mutate)
    ok1 = abs(d_mgm - MGM["delta"]) <= 5e-4
    print(f"the formula on Mittal, Gundlach and Martel's own mu {MGM['mu']} and gamma {MGM['gamma']} at D 5: "
          f"{d_mgm:.4f} against their {MGM['delta']} (to its last digit): {'pass' if ok1 else 'FAIL'}"
          + (" [control d3: D - 3 in place of D - 2]" if mutate else ""))
    ok &= ok1
    m4 = [v for (D, M), v in mus.items() if D == 4.0]
    ok2 = bool(m4) and all(abs(v[0] - MU4) <= 1e-5 for v in m4)
    print(f"#621's control, mu(4) within 1e-5 of {MU4}: {', '.join(f'{v[0]:.10f}' for v in m4) or 'none'}: "
          f"{'pass' if ok2 else 'FAIL'}")
    ok &= ok2
    print("\n| D | M | mu | gamma | delta = (D - 2 - mu) gamma | mu from | gamma from |")
    print("|---|---|---|---|---|---|---|")
    for (D, M), (mu, src) in sorted(mus.items()):
        g = gam.get(D)
        dl = f"{delta(D, mu, g[0], mutate):.6f}" if g else "no gamma"
        print(f"| {D:g} | {M} | {mu:.10f} | {f'{g[0]:.9f}' if g else '-'} | {dl} | {src} | {g[1] if g else '-'} |")
    for (D, M), (mu, _) in sorted(mus.items()):
        if D == MGM["D"] and D in gam:
            dd = delta(D, mu, gam[D][0], mutate)
            print(f"\nD 5, M {M}: our mu {mu:.6f} is {mu - MGM['mu']:+.2e} from their {MGM['mu']}, our delta {dd:.6f} is "
                  f"{dd - MGM['delta']:+.2e} from their {MGM['delta']} (a discrepancy, reported)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
