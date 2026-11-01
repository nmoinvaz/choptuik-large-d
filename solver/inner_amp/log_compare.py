#!/usr/bin/env python3
"""Compare two amp_flint_par.py run logs order by order. The lines "k N: S_1^(N) = ... sigma = ... [Ts]" must agree in
every printed digit at every order both logs have, the elapsed-seconds tag aside. K does not enter below the top order,
so a K 43 run reproduces a K 41 run's lines at the same wf, J and precision (the Mac's T94 logs equal its T89 logs at
orders 1-35), and a run on another machine must reproduce them too: this is the early cross-machine control of the
section 13.0 amendment of 2026-11-01 (notes/PLANTED_SELECTION.md), run before a node run on a new machine is trusted.
Logs at different J differ in the fit's err and resid fields from order 3 on, so the comparison is only for equal J.

    python3 log_compare.py LOG_A LOG_B      exit 0: identical at every common order (the line printed says how many);
                                            exit 1: a difference (the differing lines are printed);
                                            exit 2: no common order yet (a log without a "k N:" line)
    python3 log_compare.py --selftest       two planted logs: equal ones pass, one changed digit is reported (exit 1
                                            inside), a log without orders gives exit 2; the selftest exits 1 if not
"""
import re
import sys

LINE = re.compile(r"^k (\d+): (.*?)\s*\[\d+s\]\s*$")


def orders(path):
    """{order: the line without its seconds tag} for every 'k N:' line of the log"""
    d = {}
    with open(path) as f:
        for line in f:
            m = LINE.match(line)
            if m:
                d[int(m.group(1))] = m.group(2)
    return d


def compare(a, b, out=print):
    """0 identical at every common order, 1 a difference, 2 no common order"""
    common = sorted(set(a) & set(b))
    if not common:
        out("no common order yet")
        return 2
    diff = [k for k in common if a[k] != b[k]]
    if diff:
        for k in diff:
            out(f"DIFF at order {k}:\n  A: {a[k]}\n  B: {b[k]}")
        out(f"{len(diff)} of {len(common)} common orders differ ({common[0]}..{common[-1]})")
        return 1
    out(f"identical at all {len(common)} common orders ({common[0]}..{common[-1]})")
    return 0


def selftest():
    import os
    import tempfile
    base = ["header line\n",
            "k 1: S_1^(1) = (25.6726510515863 - 38.9178209700835j)  fit Mf 12 err inf resid 0.0e+00  "
            "sigma = (1.0 + 0.0j)  [3s]\n",
            "   k 3 nu 1: |S_gen/linS| 1.0e-01 fit Mf 12 err 1e-25 resid 1e-70 f (1 + 1j)\n",
            "k 3: S_1^(3) = (39.5012058707245 - 15.4740316772773j)  fit Mf 12 err 1.4e-24 resid 1.2e-72  "
            "sigma = (0.743584579922219 + 0.52447485276042j)  [20s]\n"]
    other = list(base)
    other[1] = other[1].replace("[3s]", "[7s]")                  # seconds differ: still identical
    bad = list(base)
    bad[3] = bad[3].replace("0.743584579922219", "0.743584579922210")  # one digit changed at order 3
    empty = ["header line\n", "   k 3 nu 1: partial\n"]
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for name, lines in (("a", base), ("b", other), ("bad", bad), ("empty", empty)):
            paths[name] = os.path.join(d, name + ".log")
            with open(paths[name], "w") as f:
                f.writelines(lines)
        quiet = lambda *_: None
        got = (compare(orders(paths["a"]), orders(paths["b"]), quiet),
               compare(orders(paths["a"]), orders(paths["bad"]), quiet),
               compare(orders(paths["a"]), orders(paths["empty"]), quiet))
    want = (0, 1, 2)
    print(f"selftest: equal logs {got[0]} (want 0), one changed digit {got[1]} (want 1), no orders {got[2]} (want 2)")
    return 0 if got == want else 1


def main(argv):
    if "--selftest" in argv:
        return selftest()
    if len(argv) != 3:
        print(__doc__)
        return 2
    return compare(orders(argv[1]), orders(argv[2]))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
