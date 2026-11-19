#!/usr/bin/env python3
"""The runs that amp_trunc_checks.py reads (#1016): amp_flint_par.py with and without --trunc, and the legacy fit's pair
for the J check. Two single-process jobs at a time (the two procs-2 planted runs alone), nice 10, each under a 4 h cap;
each writes NAME.json and NAME.log into DIR (a raw-data task folder) and skips a NAME whose .json exists.

    python3 run_trunc_checks.py DIR [--only=planted,full,trunc,control,legacy]
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PHYS = ["--fit", "phys", "--mf", "6:12"]


def jobs():
    j = []
    P = lambda name, K, J, extra, procs=1, grp="planted": j.append(
        (grp, name, ["amp_flint_par.py", "1.0", str(K), str(J), "640", "--procs", str(procs)] + PHYS + extra, procs))
    P("P_K7_J500_full", 7, 500, [])
    P("P_K7_J500_t7", 7, 500, ["--trunc", "7"])
    P("P_K7_J500_t9", 7, 500, ["--trunc", "9"])
    P("P_K7_J500_t5", 7, 500, ["--trunc", "5"])
    P("P_K9_J500_p2_full", 9, 500, [], 2)
    P("P_K9_J500_p2_t9", 9, 500, ["--trunc", "9"], 2)
    for J in (1000, 1500):
        P(f"F_K11_J{J}", 11, J, [], grp="full")
        P(f"T3_K9_J{J}", 9, J, ["--trunc", "3"], grp="trunc")
        P(f"T5_K11_J{J}", 11, J, ["--trunc", "5"], grp="trunc")
    P("C3_K9_J1000", 9, 1000, ["--trunc", "3", "--trunc-even", "2"], grp="control")
    P("C5_K11_J1000", 11, 1000, ["--trunc", "5", "--trunc-even", "4"], grp="control")
    for J in (1000, 1500):
        j.append(("legacy", f"L_full_K9_J{J}", ["amp_flint_par.py", "1.0", "9", str(J), "640", "--procs", "1"], 1))
        j.append(("legacy", f"L3_K9_J{J}", ["-c", HIER.format(J=J)], 1))
    return j


# the legacy hier model (solver/proof/amp_trunc_hiermodel.py), its JSON written to the task folder instead of beside
# the script
HIER = ("import sys, json, mpmath as mp; sys.path.insert(0, '../proof'); from amp_trunc_hiermodel import AmpHierModel; "
        "a = AmpHierModel(1.0, 9, {J}, 3); sig, f = a.run(); "
        "json.dump({{'M': 3, 'H': 4, 'wf': 1.0, 'K': 9, 'J': {J}, 'dps': 60, 'fit': 'legacy-hier', "
        "'sigma_str': [[mp.nstr(z.real, 20), mp.nstr(z.imag, 20)] for z in sig], 'f': f}}, open(sys.argv[1], 'w'))")


def run(D, name, args, py):
    out = os.path.join(D, name + ".json")
    if os.path.exists(out):
        return name, "skipped"
    cmd = ["timeout", "14400", "nice", "-n", "10", py, "-u"] + args
    cmd += [out] if args[0] == "-c" else ["--out", out]
    with open(os.path.join(D, name + ".log"), "w") as fh:
        rc = subprocess.call(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
        fh.write(f"exit {rc}\n")
    return name, rc


def main():
    D = os.path.abspath(sys.argv[1])
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    py = sys.executable
    js = [x for x in jobs() if only is None or x[0] in only]
    single = [x for x in js if x[3] == 1]
    double = [x for x in js if x[3] == 2]
    with ThreadPoolExecutor(2) as ex:
        for name, rc in ex.map(lambda x: run(D, x[1], x[2], py), single):
            print(name, rc, flush=True)
    for x in double:
        print(*run(D, x[1], x[2], py), flush=True)


if __name__ == "__main__":
    main()
