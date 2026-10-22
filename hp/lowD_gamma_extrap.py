#!/usr/bin/env python3
"""gamma(3.5) by extrapolation from our points at 3.64 <= D <= 4 (notes/LOW_D.md, Sec. 4), with the extrapolation
tested first on a planted curve.

    python3 lowD_gamma_extrap.py ECC [--target=3.5]

ECC: a clone of github.com/EckerChristian/CriticalSpacetimeCrystal (its GammaList.dat is the planted curve: a smooth,
independently computed gamma(D) through the same range).
Estimator: polynomials of degree 2 and 3 in D through our points (least squares when over-determined), evaluated at the
target; the quoted value is the mean of the two fits and the quoted range twice their spread (rule fixed in advance).
Planted test: the same rule applied to the planted curve's values at our D must contain the planted value at the
target; if it does not, the script exits non-zero and no extrapolated value is quoted. --mutate shifts the planted
target by 0.01 (a curve that turns below our range) and the test must fail.
Our points: gamma from the f64 solver (kc 47 RK x2, Rust growth-mode matrix: solver/data/scan_lowD3_kc47.jsonl at
3.6425, 3.675, 3.7938; solver/data/lowD_f64.jsonl at 3.7) and D = 4 (README value 0.3739610). Their own error is about
3e-7 (kc 39 -> 47 change at 3.7: 2.6e-6, contracting about 9-fold per 8 modes), far below the extrapolation error.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OURS = [(4.0, 0.3739610), (3.7938, 0.35792558), (3.7, 0.34864959), (3.675, 0.34590386), (3.6425, 0.34213711)]


def fits(Ds, gs, target):
    out = []
    for deg in (2, 3):
        c = np.polyfit(np.array(Ds) - 3.75, gs, deg)
        out.append(float(np.polyval(c, target - 3.75)))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    target = float(opts.get("target", 3.5))
    ecc = args[0] if args else "ecc"
    GL = np.array([[float(a), float(b)] for a, b in (l.split() for l in open(os.path.join(ecc, "complete_data", "GammaList.dat")) if l.strip())])
    GL = GL[np.argsort(GL[:, 0])]
    _, i = np.unique(np.round(GL[:, 0], 9), return_index=True)
    GL = GL[i]

    def planted(D):
        j = np.argsort(np.abs(GL[:, 0] - D))[:4]
        return float(np.polyval(np.polyfit(GL[j, 0] - D, GL[j, 1], 3), 0.0))

    Ds = [d for d, _ in OURS]
    pv = [planted(d) for d in Ds]
    pt = planted(target) + (0.01 if "--mutate" in sys.argv else 0.0)   # control: a curve that turns below our range
    pf = fits(Ds, pv, target)
    # rule fixed in advance: value = mean of the degree-2 and degree-3 fits, range = twice their spread
    pc, pr = float(np.mean(pf)), 2 * (max(pf) - min(pf))
    ok = abs(pc - pt) <= pr
    print(f"planted curve (Ecker et al.'s GammaList) at our D, extrapolated to {target}: fits {', '.join(f'{v:.5f}' for v in pf)} "
          f"-> {pc:.5f} +- {pr:.5f}; planted value {pt:.5f}: error {pc - pt:+.1e} -> {'within' if ok else 'OUTSIDE'} the range")
    if not ok:
        print("planted test FAILED: no value quoted")
        sys.exit(1)
    gs = [g for _, g in OURS]
    of = fits(Ds, gs, target)
    centre, rng = float(np.mean(of)), 2 * (max(of) - min(of))
    print(f"ours, extrapolated to {target}: fits {', '.join(f'{v:.5f}' for v in of)} -> gamma({target}) = {centre:.4f} +- {rng:.4f} "
          f"(twice the spread of the two fits; an estimator range, not a bound)")
    print(f"our points minus the planted curve at our D: {', '.join(f'{g - p:+.1e}' for g, p in zip(gs, pv))}")


if __name__ == "__main__":
    main()
