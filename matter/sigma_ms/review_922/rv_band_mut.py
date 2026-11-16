#!/usr/bin/env python3
"""Review of smdriver.newton_band (#922): the reviewer's mutations of the function, run through the author's planted
test. planted_band.py is run unchanged, with smdriver.newton_band replaced by a copy made from the current source with
one piece of text changed. The exit status is planted_band.py's: 0 if its checks A to D all pass (the mutation is not
noticed), 1 if one fails (noticed).

    PYTHONPATH=<site> python3 rv_band_mut.py --mutate NAME [--out FILE.json]
      sonly    a state is returned as converged when the resolved part S of its residual is below tol, whatever the
               band part B is (and every iteration goes through the split). This is the mutation named on the issue
               before any run, as the one the planted test should notice (A, by max |z - z*|).
      noshort  sonly's first half alone: the whole-step path switched off. The function is then still correct, so
               the test should pass: this shows that sonly fails by its convergence test.
      loose    max |F| < tol replaced by max |F| < 1e10 tol in the convergence test: the function returns a state that
               is not converged. Written here before its run: the planted test should notice (A, by max |z - z*|).
      nodamp   the band step scaled as a whole to the radius (weights rad/|a| in every band direction): the version
               that SIGMA_DSS section 11 says stalled at start A
      nofloor  the floor of the split set to zero: E = 0 and E(held) = 0
      noheld   the chord corrector without taking the held-back band step off first
      anyB     a band trial accepted without the fall of B
      nochord  no chord corrections (the corrector loop skipped)
      none     no change (the harness itself: must exit 0)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SM = os.path.dirname(HERE)
sys.path.insert(0, SM)

MUT = {
    "none": [],
    "loose": [("        if nF < tol:\n            rec[\"converged\"] = True\n            if verbose:\n"
               "                log(f\"  band {it}: |F| {nF:.2e} converged, Delta {zh[-1]:.15f}\")",
               "        if nF < 1e10 * tol:\n            rec[\"converged\"] = True\n            if verbose:\n"
               "                log(f\"  band {it}: |F| {nF:.2e} converged, Delta {zh[-1]:.15f}\")")],
    # sonly: every iteration goes through the split (the whole-step path is switched off), and a state whose resolved
    # part S is below tol is returned as converged, whatever its band part B is
    "noshort": [("        if nN <= radius:\n", "        if False:\n")],
    "sonly": [("        if nN <= radius:\n", "        if False:\n"),
              ("        S0, B0 = split(Fv)\n",
               "        S0, B0 = split(Fv)\n        if k and S0 < tol:\n            rec[\"converged\"] = True\n"
               "            return zh, zl, hist\n")],
    "nodamp": [("            return sg ** 2 / (sg ** 2 + np.exp(2 * hi)), float(np.exp(hi))",
                "            return np.full(k, rad / nb), 1.0")],
    "nofloor": [("            E = float(np.linalg.norm(L @ a))", "            E = 0.0"),
                ("            Eh = float(np.linalg.norm(L @ ((1 - w) * a)))", "            Eh = 0.0")],
    "noheld": [("                c = dd_add(c[0], c[1], -held, np.zeros_like(held))\n", "")],
    "anyB": [("            ok = np.isfinite(S1) and B1 < B0 - 1e-4 * (B0 - Bp) and S1 <=",
              "            ok = np.isfinite(S1) and S1 <=")],
    "nochord": [("            while nc < correctors and np.isfinite(S1) and S1 > max(kappa * B1, 1.5 * Eh, 0.1 * tol):",
                 "            while False:")],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutate", required=True, choices=sorted(MUT))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    path = os.path.join(SM, "smdriver.py")
    src = open(path).read()
    for old, new in MUT[a.mutate]:
        assert src.count(old) == 1, f"mutation {a.mutate}: the text to change occurs {src.count(old)} times"
        src = src.replace(old, new)
    ns = {"__name__": "smdriver_mutated", "__file__": path}
    exec(compile(src, path, "exec"), ns)
    import planted_band as PB
    PB.sd.newton_band = ns["newton_band"]
    print(f"rv_band_mut: newton_band with mutation {a.mutate} ({len(MUT[a.mutate])} change(s)); planted_band.py follows")
    sys.argv = ["planted_band.py"] + (["--out", a.out] if a.out else [])
    PB.main()


if __name__ == "__main__":
    main()
