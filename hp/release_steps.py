#!/usr/bin/env python3
"""The junk-frozen full-system step (full_release.py --octo 1 --junk 4) from chain_fit's states, collected from the
release logs of the raw-data repository (notes/QUAD_DOUBLE.md 13.29). Per state: D, the residual and Delta at the
pinned state, the first step's dDelta (the held harmonics' bias, a linear estimate, not a bound), and the pinned Delta
minus the 1/D series of degree 20, with the log it came from. The logs of the control state (T137's state at 79.77,
anm-mac/) are checked instead: each must give T137's +5.03e-8 within 2%. Writes data/release_steps.csv and prints the
table. The controls of the checks, which must fail: --mutate=control scales the first control's dDelta by 1.03, and
--mutate=duplicate the second log's of a state in two logs.

    cd hp && python3 release_steps.py RAWDIR [--tasks=T141-win2,T142-macm3,T143-win2,T145-win2] [--mutate=control|duplicate]

Exits 1 when a control fails, or when two logs of one state differ in dDelta by more than 2%.
"""
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "release_steps.csv")
CONTROL, CONTROL_STEP, TOL = "anm-mac/run_trend_pin3_qd3/anm_pin3_D79.7657.npz", 5.03e-8, 0.02
NUM = r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
FIRST = re.compile(rf"^(\S+): D ({NUM}), M \d+, kfreeze 13, octo-double: residual at the pinned state ({NUM}), "
                   rf"Delta ({NUM})")
FINAL = re.compile(rf"first step's dDelta ({NUM})\); series 1/D 20 ({NUM}): pinned - series ({NUM})")


def read(path):
    """(state, D, residual, Delta, dDelta, pinned - series) of a release log, or None while it has no final line"""
    lines = open(path, errors="replace").read().splitlines()
    m1 = FIRST.match(lines[0]) if lines else None
    m2 = next((FINAL.search(x) for x in reversed(lines) if FINAL.search(x)), None)
    if not (m1 and m2):
        return None
    return dict(state=m1[1], D=float(m1[2]), residual=float(m1[3]), Delta=m1[4], dDelta=float(m2[1]),
                pinned_minus_series=float(m2[3]))


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    opts = dict((a[2:] + "=").split("=")[:2] for a in argv if a.startswith("--"))
    if len(args) != 1:
        sys.exit(__doc__)
    raw = os.path.abspath(args[0])
    tasks = opts.get("tasks", "T141-win2,T142-macm3,T143-win2,T145-win2").split(",")
    rows, controls, ok = {}, [], True
    for t in tasks:
        for p in sorted(glob.glob(os.path.join(raw, t, "release_D*.log"))):
            r = read(p)
            rel = os.path.relpath(p, raw)
            if r is None:
                print(f"  incomplete: {rel}")
                continue
            r["log"] = rel
            if r["state"] == CONTROL:
                controls.append(r)
                continue
            if r["state"] in rows:
                a, b = rows[r["state"]]["dDelta"], r["dDelta"]
                if opts.get("mutate") == "duplicate":
                    b *= 1.03
                    print(f"mutated: {rel}'s dDelta scaled by 1.03")
                same = abs(a - b) <= TOL * max(abs(a), abs(b))
                ok &= same
                print(f"{'agree' if same else 'DIFFERS'}: {r['state']}: dDelta {a:+.4e} ({rows[r['state']]['log']}) and "
                      f"{b:+.4e} ({rel})")
                continue
            rows[r["state"]] = r
    if opts.get("mutate") == "control" and controls:
        controls[0]["dDelta"] *= 1.03
        print(f"mutated: the control of {controls[0]['log']} scaled by 1.03")
    for c in controls:
        good = abs(c["dDelta"] / CONTROL_STEP - 1) <= TOL
        ok &= good
        print(f"control {c['log']}: dDelta {c['dDelta']:+.4e}, T137 {CONTROL_STEP:+.2e}: {'pass' if good else 'FAIL'}")
    if not controls:
        ok = False
        print("FAIL: no control log")
    out = sorted(rows.values(), key=lambda r: r["D"])
    print(f"{'D':>7} {'dDelta':>11} {'pinned - series':>16}  log")
    for r in out:
        print(f"{r['D']:7.2f} {r['dDelta']:+11.3e} {r['pinned_minus_series']:+16.3e}  {r['log']}")
    if not opts.get("mutate"):
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["D", "state", "residual", "Delta", "dDelta", "pinned_minus_series", "log"])
            w.writeheader()
            w.writerows(out)
        print(f"{len(out)} states -> {OUT}")
    print(f"{'pass' if ok else 'FAIL'}: {len(controls)} controls, {len(out)} states")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
