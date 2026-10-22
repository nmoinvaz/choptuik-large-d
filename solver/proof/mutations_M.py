#!/usr/bin/env python3
"""Mutation tests and controls for the M-harmonic lemmas (proof/A_lemmas_M.md section 8). Each entry runs one script
with one input broken on purpose (or a control) and compares the outcome with the verdict written down here before the
runs. Exit status 0 only if every verdict matches. Results: lemmas_M_runs/mutations_M.json.

    python3 mutations_M.py               the cheap ones (seconds to a few minutes each)
    python3 mutations_M.py --all         also those that need the saved runs (the 6b1 covering of M = 3 and the 6a
                                         pilot's saved backward enclosure)
    python3 mutations_M.py --only=name1,name2
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# (name, command, expected outcome, what is broken); "caught" means the script's exit status reports the check failing
# (each script exits 0 when its mutation is caught, see its docstring), "fails" that a control's lemma fails (exit 2/3),
# "passes" that a control passes (exit 0)
CHEAP = [
    ("fp_series_M3", ["fixed_point_M.py", "--M=3", "--mutate=series"], "caught",
     "step (i): one coefficient of the harmonic-3 D series off by a relative 1e-6"),
    ("fp_series_M5", ["fixed_point_M.py", "--M=5", "--mutate=series", "--r0=0.0622", "--arg=5pi/16", "--diag"], "caught",
     "step (i), M = 5 on inner_np's ray: one coefficient of the harmonic-5 D series off by 1e-6"),
    ("fp_cdrop_M3", ["fixed_point_M.py", "--M=3", "--mutate=cdrop"], "caught",
     "step (i): the series built without the c harmonics +-4 (an inconsistent harmonic set)"),
    ("fp_diagpath_M3", ["fixed_point_M.py", "--M=3", "--mutate=diagpath"], "caught",
     "guard, not a mutation: the diag splitting with a non-ray kernel bound (invalid there) must be refused"),
    ("fp_control_r0", ["fixed_point_M.py", "--M=3", "--r0=0.3", "--diag"], "fails",
     "control: start radius 0.3, the contraction must fail"),
    ("fp_m1", ["fixed_point_M.py", "--check-m1"], "passes",
     "control: M = 1 reproduces fixed_point.py (constants, start values, start data)"),
    ("usys_label", ["lemma6_usys_M.py", "--mutate=label"], "caught",
     "system: s_mu = i abs(mu) w in the u-form (the label sign lost); the comparisons with the xi-form and with "
     "fixed_point's encoding must fire"),
    ("usys_clean", ["lemma6_usys_M.py"], "passes", "control: the system self-tests at M = 3, 5"),
    ("S_series_M3", ["lemma6_sector_M.py", "--M=3", "--rp=0.1", "--k=20", "--mutate=series"], "caught",
     "lemma S: one coefficient of the harmonic-3 series off by 1e-6"),
    ("S_control_rp", ["lemma6_sector_M.py", "--M=3", "--rp=0.13", "--k=10"], "fails",
     "control: lemma S at R+ = 0.13^2, beyond the contraction (0.83 at 0.1, label terms grow like R+)"),
    ("disc_m1", ["lemma6b1_disc_M.py"], "passes", "control: lemma D at M = 1 reproduces lemma6b1_disc.certify"),
]
SAVED = [
    ("cover_drop_M3", ["lemma6b1_cover_M.py", "check", "--M=3", "--h=1e-8", "--drop=5,6"], "fails",
     "6b1 gluing check with tile (5, 6) left out: must print FAIL"),
    ("cover_sign_M3", ["lemma6b1_cover_M.py", "mutate", "--M=3", "--h=1e-8", "--j=11"], "caught",
     "6b1: every disc of the innermost row with the fast term of the wrong sign: none may be certified"),
    ("jump_partial_M3", ["lemma6_jump_M.py", "--M=3", "--pilot=1e-10", "--mutate=partial",
                         "--load-B=lemmas_M_runs/lemma6a_pilot_M3_B.json"], "caught",
     "6a: only J_D,1 = 0 assumed (J_D,3 dropped): G_slow is not square, 6a must not be established"),
    ("jump_swap_M3", ["lemma6_jump_M.py", "--M=3", "--pilot=1e-10", "--mutate=swap"], "caught",
     "6a: y- = conj(y+) without the reflection mu -> -mu: must fail"),
    ("jump_coords_M3", ["lemma6_jump_M.py", "--M=3", "--pilot=1e-10", "--mutate=coords",
                        "--load-B=lemmas_M_runs/lemma6a_pilot_M3_B.json"], "report",
     "6a: (1 - i mu w) in the change of variables at xi_a; at M = 1 not caught (second order); reported only"),
]


def run(name, cmd, expect, what):
    t0 = time.time()
    p = subprocess.run([PY, "-u"] + cmd, cwd=HERE, capture_output=True, text=True)
    tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
    code = p.returncode
    outcome = {0: "caught" if expect == "caught" else "passes"}.get(code, "fails")
    if expect == "caught":
        ok = code == 0
    elif expect == "fails":
        ok = code != 0
    elif expect == "passes":
        ok = code == 0
    else:
        ok = True
    r = dict(name=name, cmd=" ".join(cmd), expected=expect, exit=code, matches=ok, what=what, tail=tail,
             sec=round(time.time() - t0, 1))
    print(f"{name:16s} expected {expect:7s} exit {code}: {'OK' if ok else 'MISMATCH'}   ({r['sec']} s)  "
          f"{tail[-1] if tail else ''}", flush=True)
    return r


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    todo = CHEAP + (SAVED if "all" in opts else [])
    if "only" in opts:
        names = opts["only"].split(",")
        todo = [t for t in CHEAP + SAVED if t[0] in names]
    path = os.path.join(HERE, "lemmas_M_runs", "mutations_M.json")
    old = {}
    if os.path.exists(path):
        old = {r["name"]: r for r in json.load(open(path))}
    res = []
    for t in todo:
        r = run(*t)
        res.append(r)
        old[r["name"]] = r
        with open(path, "w") as fh:
            json.dump(list(old.values()), fh, indent=1)
    allok = all(r["matches"] for r in res)
    print(f"ALL VERDICTS AS EXPECTED: {allok}")
    return 0 if allok else 3


if __name__ == "__main__":
    sys.exit(main())
