#!/usr/bin/env python3
"""Is lemma LC's saved Lipschitz matrix the same from run to run? (#891)

An entry of the matrix is a secant (M(eta) - M(0))/eta of two 256-bit evaluations (b8m_lc.lipschitz). Where the bound
does not depend on the perturbed input, the secant is rounding noise over eta, about 1e-69, and it follows the last
digits of the inputs. b8m_lc.py writes every entry below its floor (--lipfloor, 1e-50) as the floor. This script runs
the certificate's command twice, with the centre A* of the box as it is and moved by --dA, and compares the two
saved matrices entry by entry. A third run at the centre with --lipfloor=0 checks the floor itself (#894): every
floored entry must be at least its unfloored value, equal to it where that value is above the floor, and the floor
where it is not. So a matrix whose entries were lowered, all written as the floor for example, cannot pass, though it
agrees with itself from run to run and gives theta near 0.

    python3 b8m_lc_liprep.py [--out=DIR] [--dA=1e-17] [--tol=1e-9] [--mutate=nofloor|cap]
        three runs of  b8m_lc.py 4 73.89 --rA=2e-6 --rW=2e-6 --lip , about 3 min each, written to DIR (a new
        temporary directory by default; not solver/proof, since the runs must not touch the stored files). Exit 0
        if every entry agrees to --tol relative, both matrices give theta < 1, and the floor check passes.
        --mutate=nofloor: both runs with --lipfloor=0. The noise entries then differ, and the script must exit 1.
        --mutate=cap: the centre run's saved matrix with every entry written as the floor (a copy, _cap). It agrees
        with itself and gives theta near 0, so only the floor check can catch it, and the script must exit 1.
    python3 b8m_lc_liprep.py --compare=a.json,b.json [--tol=1e-9]
        compares two saved matrices only.

The run with the centre moved is a test run. It does not certify the lemma's box, and its files are named _dA.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
CMD = ["4", "73.89", "--rA=2e-6", "--rW=2e-6", "--lip"]
# the certificate's script as __main__, after the centre of b8m_formal.params is moved by the decimal dA
RUNNER = ("import runpy, sys\nfrom decimal import Decimal\nimport b8m_formal as MF\ndA = sys.argv.pop(1)\n"
          "MF.A_STAR_S = str(Decimal(MF.A_STAR_S) + Decimal(dA))\nsys.argv[0] = 'b8m_lc.py'\n"
          "runpy.run_path('b8m_lc.py', run_name='__main__')\n")


def compare(pa, pb, tol):
    """(entries, entries that differ by more than tol relative, the largest relative difference and its key)"""
    A, B = json.load(open(pa))["M"], json.load(open(pb))["M"]
    assert set(A) == set(B), "the two matrices have different rows or columns"
    worst, wkey, bad = 0.0, None, 0
    for key in A:
        a, b = A[key], B[key]
        d = abs(a - b) / max(a, b) if max(a, b) > 0 else 0.0
        bad += d > tol
        if d > worst:
            worst, wkey = d, key
    return len(A), bad, worst, wkey


def floor_check(pf, pu, tol):
    """(entries lowered below their unfloored value, entries above the floor that differ from it, entries at or below
    the floor that are not the floor) of the floored matrix pf against the unfloored pu"""
    F, U = json.load(open(pf)), json.load(open(pu))
    floor, A, B = F["floor"], F["M"], U["M"]
    assert set(A) == set(B), "the floored and unfloored matrices have different rows or columns"
    lowered = sum(A[k] < B[k] * (1 - tol) for k in A)
    above = sum(B[k] > floor and abs(A[k] - B[k]) > tol * B[k] for k in A)
    atfloor = sum(B[k] <= floor and A[k] != floor for k in A)
    return lowered, above, atfloor


def theta(path):
    import b8m_lc as LC
    return float(LC.lipcheck_file(path)[1].upper())


def run(out, tag, dA, extra):
    lip, log = os.path.join(out, f"lip{tag}.json"), os.path.join(out, f"lc{tag}.log")
    cmd = [sys.executable, "-c", RUNNER, dA] + CMD + [f"--lipout={lip}", f"--log={log}"] + extra
    with open(os.path.join(out, f"lc{tag}.out"), "w") as fh:
        rc = subprocess.run(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"run{tag or ' as written'}: centre moved by {dA}, exit {rc} -> {os.path.basename(lip)}", flush=True)
    return rc, lip


def main():
    tol = float(OPTS.get("tol", "1e-9"))
    if "compare" in OPTS:
        pa, pb = OPTS["compare"].split(",")
        rcs = [0, 0]
    else:
        out = os.path.abspath(OPTS["out"]) if "out" in OPTS else tempfile.mkdtemp(prefix="b8m_lc_liprep_")
        if out == HERE:
            sys.exit("b8m_lc_liprep: --out must be a directory other than solver/proof")
        os.makedirs(out, exist_ok=True)
        print(f"output directory {out}", flush=True)
        mut = OPTS.get("mutate", "")
        assert mut in ("", "nofloor", "cap"), f"unknown mutation {mut}"
        extra = ["--lipfloor=0"] if mut == "nofloor" else []
        tag = "_nofloor" if mut == "nofloor" else ""
        (rc0, pa), (rc1, pb) = run(out, tag, "0", extra), run(out, tag + "_dA", OPTS.get("dA", "1e-17"), extra)
        rcs = [rc0, rc1]
        if mut != "nofloor":
            rcu, pu = run(out, "_unfloored", "0", ["--lipfloor=0"])
            rcs.append(rcu)
        if mut == "cap":
            d = json.load(open(pa))
            d["M"] = {k: d["floor"] for k in d["M"]}
            pa = pa.replace(".json", "_cap.json")
            json.dump(d, open(pa, "w"))
            pb = pa      # the capped matrix agrees with itself, as a run-to-run comparison of it would
    n, bad, worst, wkey = compare(pa, pb, tol)
    th = [theta(p) for p in (pa, pb)]
    ok = bad == 0 and max(th) < 1 and all(rc == 0 for rc in rcs)
    if "compare" not in OPTS and OPTS.get("mutate", "") != "nofloor":
        lowered, above, atfloor = floor_check(pa, pu, tol)
        fok = lowered == 0 and above == 0 and atfloor == 0
        print(f"floor check against the unfloored run: {lowered} entries lowered, {above} above the floor that differ, "
              f"{atfloor} at or below it that are not the floor: {'PASS' if fok else 'FAIL'}")
        ok = ok and fok
    print(f"{n} entries; {bad} differ by more than {tol:g} relative; the largest difference is {worst:.1e} at {wkey}")
    print(f"theta = {th[0]:.4f} and {th[1]:.4f}")
    print(f"LIPREP: {'PASS' if ok else 'FAIL'} (the two matrices agree to {tol:g} in every entry: {bad == 0})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
