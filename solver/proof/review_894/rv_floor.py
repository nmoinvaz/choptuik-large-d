#!/usr/bin/env python3
"""Review of lemma LC's matrix floor (#894): b8m_lc.lipschitz writes every entry of the Lipschitz matrix below
--lipfloor (1e-50) as the floor (#891). The reviewer's checks, from saved matrices.

    python3 rv_floor.py --dir=DIR [--floored=F.json] [--stored=../b8m_lc_lip.json]
        DIR holds the outputs of  b8m_lc_liprep.py --out=DIR  and of its --mutate=nofloor: lip.json (floor 1e-50),
        lip_nofloor.json (no floor), lip_dA.json (floor, centre moved by 1e-17), all from the same code.
        K1  floored against unfloored, same centre: every entry of the floored matrix is >= the unfloored one, equal to
            it in every digit where the unfloored entry is above the floor, and equal to the floor elsewhere
        K2  what the entries at the floor add to a weighted row sum, in exact rational arithmetic, with the
            certificate's weights and with the reviewer's own (power iteration); theta from both matrices
        K3  the two centres: the same entries at the floor; the entries above it within 5.3e-11 (stated in B_chain.md)
        K4  the saved fields: floor, floored (recounted), eta, the note
        K5  the stored matrix (solver/proof/b8m_lc_lip.json) against the rerun, entry by entry
        Exit 0 only if K1 to K5 hold. --floored names another file to take the place of lip.json in K1 (the mutation).
    python3 rv_floor.py cap --out=DIR [--dA=1e-17]
        THE REVIEWER'S MUTATION. The certificate's command on a copy of b8m_lc.py in memory whose flooring line is
        reversed for the entries above the floor: every entry is written as the floor, so each true entry above
        1e-50 is replaced by a number below it. Writes DIR/lip_cap.json (and lip_cap_dA.json with --dA). Written
        before the run: K1 with --floored=DIR/lip_cap.json must fail (exit 3).
"""
import json
import os
import subprocess
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.dirname(HERE)
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
CMD = ["4", "73.89", "--rA=2e-6", "--rW=2e-6", "--lip"]
LINE = "(v if v > fl else fl)"
MUT = "(fl)"
RUNNER = ("import sys\nfrom decimal import Decimal\nimport b8m_formal as MF\ndA = sys.argv.pop(1)\n"
          "MF.A_STAR_S = str(Decimal(MF.A_STAR_S) + Decimal(dA))\nsrc = open('b8m_lc.py').read()\n"
          f"assert src.count({LINE!r}) == 1, 'the flooring line was not found'\nsrc = src.replace({LINE!r}, {MUT!r})\n"
          "sys.argv[0] = 'b8m_lc.py'\ng = {'__name__': '__main__', '__file__': 'b8m_lc.py'}\n"
          "exec(compile(src, 'b8m_lc.py', 'exec'), g)\n")


def load(p):
    with open(p) as fh:
        return json.load(fh)


def weights_own(M, keys, it=4000):
    """a positive vector by power iteration on M + 1 (floats); any positive vector gives a valid bound"""
    v = {k: 1.0 for k in keys}
    for _ in range(it):
        w = {r: v[r] + sum(M[f"{r}<-{c}"] * v[c] for c in keys) for r in keys}
        m = max(w.values())
        v = {k: x / m for k, x in w.items()}
    return v


def theta_exact(M, keys, v):
    """max_r sum_c M[r<-c] v_c / v_r in exact rational arithmetic (the floats are taken as exact numbers)"""
    fv = {k: Fraction(x) for k, x in v.items()}
    best = Fraction(0)
    for r in keys:
        s = sum(Fraction(M[f"{r}<-{c}"]) * fv[c] for c in keys) / fv[r]
        best = max(best, s)
    return best


def added(M, keys, v, floor):
    """max_r of the part of row r's weighted sum that comes from the entries equal to the floor"""
    fv = {k: Fraction(x) for k, x in v.items()}
    best = Fraction(0)
    for r in keys:
        s = sum(Fraction(floor) * fv[c] for c in keys if M[f"{r}<-{c}"] == floor) / fv[r]
        best = max(best, s)
    return best


def check():
    d = OPTS["dir"]
    Fd = load(OPTS.get("floored", os.path.join(d, "lip.json")))
    Ud = load(os.path.join(d, "lip_nofloor.json"))
    F, U = Fd["M"], Ud["M"]
    floor = load(os.path.join(d, "lip.json"))["floor"]
    keys = sorted({k.split("<-")[0] for k in U})
    assert set(F) == set(U) and len(U) == len(keys) ** 2
    ok = {}
    # ---- K1
    low = [k for k in U if F[k] < U[k]]
    above = [k for k in U if U[k] > floor]
    below = [k for k in U if U[k] <= floor]
    neq = [k for k in above if F[k] != U[k]]
    nfl = [k for k in below if F[k] != floor]
    b60 = sum(1 for k in below if U[k] < 1e-60)
    ok["K1"] = not low and not neq and not nfl
    print(f"K1 floored against unfloored ({len(U)} entries, floor {floor:g}): entries lowered {len(low)}; above the "
          f"floor {len(above)}, of which not equal in every digit {len(neq)}; at or below it {len(below)} "
          f"({b60} below 1e-60, {len(below) - b60} between 1e-60 and the floor), of which not written as the floor "
          f"{len(nfl)}; smallest unfloored entry above the floor {min(U[k] for k in above):.3e}, largest at or below "
          f"it {max(U[k] for k in below):.3e}: {'OK' if ok['K1'] else 'FAIL'}")
    if low:
        w = max(low, key=lambda k: U[k] - F[k])
        print(f"    largest lowering: {w}: {U[w]:.6e} written as {F[w]:.6e}")
    # ---- K2
    sys.path.insert(0, PROOF)
    import b8m_lc as LC
    tF, tU = [], []
    for name, Mx, acc in (("floored", F, tF), ("unfloored", U, tU)):
        vo = weights_own(Mx, keys)
        acc.append(theta_exact(Mx, keys, vo))
        acc.append(vo)
    _, thF, vc = LC.lipcheck_file(OPTS.get("floored", os.path.join(d, "lip.json")))
    _, thU, _ = LC.lipcheck_file(os.path.join(d, "lip_nofloor.json"))
    add_c = added(F, keys, vc, floor)
    add_o = added(F, keys, tF[1], floor)
    th_c = theta_exact(F, keys, vc)
    ok["K2"] = float(add_c) <= 2.4e-48 and float(tF[0]) < 1 and float(thF.upper()) < 1 and float(th_c) < 1
    print(f"K2 theta: the certificate's check {float(thF.upper()):.12f} (floored) and {float(thU.upper()):.12f} "
          f"(unfloored); the reviewer's exact row sums with the certificate's weights {float(th_c):.12f}, with own "
          f"weights {float(tF[0]):.12f} (floored) and {float(tU[0]):.12f} (unfloored)")
    dth = th_c - theta_exact(U, keys, vc)
    print(f"   with the certificate's weights the entries at the floor add at most {float(add_c):.3e} to a weighted row "
          f"sum (stated 2.4e-48), and theta rises by {float(dth):.3e} from the unfloored matrix: "
          f"{'OK' if ok['K2'] else 'FAIL'}")
    print(f"   (with the reviewer's weights, an approximate Perron vector, the rows that hold only floored entries have "
          f"weights near the floor, and the floored share of a row sum is not small there: {float(add_o):.3e}; theta "
          f"is the same with and without the floor to {abs(float(tF[0] - tU[0])):.1e})")
    # ---- K3
    Ad = load(os.path.join(d, "lip_dA.json"))
    A = Ad["M"]
    Fr = load(os.path.join(d, "lip.json"))["M"]
    same_floor = {k for k in Fr if Fr[k] == floor} == {k for k in A if A[k] == floor}
    rel = {k: abs(Fr[k] - A[k]) / max(Fr[k], A[k]) for k in Fr if Fr[k] > floor or A[k] > floor}
    wk = max(rel, key=rel.get)
    n12 = sum(1 for x in rel.values() if x > 1e-12)
    ok["K3"] = same_floor and rel[wk] <= 5.4e-11
    print(f"K3 centres 1e-17 apart: the same entries at the floor: {same_floor}; entries above it differ by at most "
          f"{rel[wk]:.2e} (at {wk}), {n12} by more than 1e-12; stated 5.3e-11 and 37: {'OK' if ok['K3'] else 'FAIL'}")
    # ---- K4
    Fo = load(os.path.join(d, "lip.json"))
    cnt = sum(1 for x in Fo["M"].values() if x == Fo["floor"])
    ok["K4"] = Fo["floor"] == 1e-50 and Fo["floored"] == cnt and Fo["eta"] == 1e-6 and "floor" in Fo["note"]
    print(f"K4 saved fields: floor {Fo['floor']:g}; floored {Fo['floored']} (recounted {cnt}); eta {Fo['eta']:g}; note "
          f"{Fo['note']!r}: {'OK' if ok['K4'] else 'FAIL'}")
    # ---- K5
    Sd = load(OPTS.get("stored", os.path.join(PROOF, "b8m_lc_lip.json")))
    S = Sd["M"]
    dif = [k for k in S if S[k] != Fo["M"][k]]
    worst = max((abs(S[k] - Fo["M"][k]) / max(S[k], Fo["M"][k]) for k in S), default=0.0)
    ok["K5"] = set(S) == set(Fo["M"]) and worst <= 1e-9 and Sd.get("floor") == Fo["floor"] and Sd.get("floored") == cnt
    print(f"K5 the stored matrix against the rerun: {len(dif)} of {len(S)} entries not equal in every digit, largest "
          f"relative difference {worst:.2e}; stored floor {Sd.get('floor')}, floored {Sd.get('floored')}: "
          f"{'OK' if ok['K5'] else 'FAIL'}")
    good = all(ok.values())
    print(f"RV_FLOOR: {'PASS' if good else 'FAIL (' + ', '.join(k for k, v in ok.items() if not v) + ')'}")
    return good


def cap():
    out = os.path.abspath(OPTS["out"])
    os.makedirs(out, exist_ok=True)
    for tag, dA in [("", "0")] + ([("_dA", OPTS["dA"])] if "dA" in OPTS else []):
        lip, log = os.path.join(out, f"lip_cap{tag}.json"), os.path.join(out, f"lc_cap{tag}.log")
        cmd = [sys.executable, "-c", RUNNER, dA] + CMD + [f"--lipout={lip}", f"--log={log}"]
        with open(os.path.join(out, f"lc_cap{tag}.out"), "w") as fh:
            rc = subprocess.run(cmd, cwd=PROOF, stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"mutated run{tag}: centre moved by {dA}, exit {rc} -> {os.path.basename(lip)}", flush=True)
    return True


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args and args[0] == "cap":
        sys.exit(0 if cap() else 3)
    sys.exit(0 if check() else 3)
