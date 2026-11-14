"""Cross-check of #82's K 45, J 1500 coefficients sigma_0..sigma_16 against K 33, J 1000 files of other runs (#395).

The comparison reads only the 20-digit `sigma_str` of amp_flint_par.py's JSON files from the raw-data repository and
shares no code with amp_flint_par.py. Every relative difference is r_n = |sigma_n(A) - sigma_n(B)| / |sigma_n(A)|, in
mpmath at 50 digits, with A the K 45 file.

Tolerance rule, fixed and committed before the target comparison (`tol` writes it, `compare` reads it):
    R_n      max over wf = 0.99, 1.00, 1.01 of r_n(T99-win2 K45 J1500, T83-mac K33 J1000), the same K, J and
             precisions as the targets, at the only wf where both exist.
    floor_n  one unit in the 20th significant digit of each component of both strings, over |sigma_n(A)|. It covers
             the printing and the summation-order effect of --procs on pre-#246 files (1.5e-19 relative, #229).
    verdict  OK     r_n <= max(R_n, floor_n)            (within the reference truncation difference)
             WATCH  r_n <= F max(R_n, floor_n), F = 10  (within a decade: the truncation varies with wf, #256)
             BAD    otherwise, a possible fault.
The J truncation dominates these differences (sigma_n for n <= 15 does not depend on K at J 1000, #256).

Smoothness (`smooth`): at each target node t, the K 45 value is predicted by the degree-5 interpolant through the
six other nodes of the ladder wf = 1.06, 1.08, 1.09, 1.10, 1.11, 1.12, 1.14, with K 45 at the other two targets and
K 33 J 1000 (T155-win2, T153-macm3) at 1.06, 1.08, 1.12, 1.14. Its reference is the same interpolant through the pure
K 33 ladder, predicting the K 33 value at t. Same verdict levels, against that reference.

Controls (`selftest`; each must fire):
    replica   T103-mac against T99-win2 (wf 1.00, K 45, J 1500) through the same check with tolerance 0: 23 of 23
              equal; one digit of T103's sigma_k changed is found at k and nowhere else.
    reproduce the posted values of #82's table (two digits) are recomputed before any verdict.
    planted   a relative change of 3 F max(R_n, floor_n) planted in the 1.10 K 45 sigma_n is BAD at n only; one of
              0.3 max(R_n, floor_n) is OK; in `smooth`, a change of 100 times the reference is BAD.

    python3 sigma_xcheck.py tol RAW --out TOL.json
    python3 sigma_xcheck.py compare RAW --tol TOL.json [--plant=N:REL] [--out FILE]
    python3 sigma_xcheck.py smooth RAW [--plant=N:REL] [--out FILE]
    python3 sigma_xcheck.py replica RAW [--mutate=K]
    python3 sigma_xcheck.py reproduce RAW
    python3 sigma_xcheck.py selftest RAW --tol TOL.json
    python3 sigma_xcheck.py jseq RAW [--out FILE]      (J 1000, 1250, 1500 at each node: where a difference sits)
RAW is a checkout of choptuik-large-d-raw. Exit 1 when a verdict is BAD or a control does not fire.
"""
import hashlib
import json
import os
import sys

import mpmath as mp

mp.mp.dps = 50
F = 10
NMAX = 16

REF = {wf: (f"T99-win2/scan45J1500/sigF_w{wf}_K45_J1500_p1280.json",
            f"T83-mac/scan33J1000/sigF_w{wf}_K33_J1000_p960.json") for wf in ("0.9900", "1.0000", "1.0100")}
TARGET = {"1.09": ("I82-win2/sigF_w1.09_K45_J1500_p1280.json", "T155-win2/sigF_w1.09_K33_J1000.json"),
          "1.10": ("I82-win2/sigF_w1.10_K45_J1500_p1280.json", "T153-macm3/sigF_w1.10_K33_J1000.json"),
          "1.11": ("I82-win2/sigF_w1.11_K45_J1500_p1280.json", "T155-win2/sigF_w1.11_K33_J1000.json")}
LADDER = {"1.06": "T155-win2/sigF_w1.06_K33_J1000.json", "1.08": "T153-macm3/sigF_w1.08_K33_J1000.json",
          "1.09": "T155-win2/sigF_w1.09_K33_J1000.json", "1.10": "T153-macm3/sigF_w1.10_K33_J1000.json",
          "1.11": "T155-win2/sigF_w1.11_K33_J1000.json", "1.12": "T153-macm3/sigF_w1.12_K33_J1000.json",
          "1.14": "T155-win2/sigF_w1.14_K33_J1000.json"}
REPLICA = ("T99-win2/scan45J1500/sigF_w1.0000_K45_J1500_p1280.json",
           "T103-mac/scan45J1500/sigF_w1.0000_K45_J1500_p1280.json")
# #82's closing summary, K 45 J 1500 against K 33 J 1000, |dsigma_n|/|sigma_n| to two digits.
POSTED = {"1.09": {7: 6.7e-20, 8: 8.5e-20, 9: 3.2e-17, 10: 7.8e-18, 14: 9.3e-17, 15: 2.5e-16, 16: 3.5e-16},
          "1.11": {7: 3.6e-19, 8: 9.9e-20, 9: 3.3e-18, 10: 1.0e-17, 14: 1.3e-16, 15: 2.8e-16, 16: 4.4e-16},
          "1.10": {6: 9.7e-21, 7: 1.2e-18, 8: 6.1e-20, 9: 9.4e-18, 10: 8.9e-18, 14: 1.3e-16, 15: 4.0e-16,
                   16: 1.2e-15}}


def load(raw, rel):
    path = os.path.join(raw, rel)
    with open(path, "rb") as fh:
        blob = fh.read()
    d = json.loads(blob)
    return {"path": rel, "sha256": hashlib.sha256(blob).hexdigest(), "K": d["K"], "J": d["J"], "prec": d["prec"],
            "procs": d.get("procs"), "wf": d["wf"], "s": [list(p) for p in d["sigma_str"]]}


def val(pair):
    return mp.mpc(mp.mpf(pair[0]), mp.mpf(pair[1]))


def unit20(x):
    """One unit in the 20th significant digit of the printed number x."""
    x = mp.mpf(x)
    return mp.mpf(0) if x == 0 else mp.mpf(10) ** (mp.floor(mp.log10(abs(x))) - 19)


def rel(a, b, n):
    za, zb = val(a["s"][n]), val(b["s"][n])
    return abs(za - zb) / abs(za)


def floor20(a, b, n):
    u = sum(unit20(c) for c in a["s"][n] + b["s"][n])
    return u / abs(val(a["s"][n]))


def check(a, b, nmax, ref, fac=F):
    """The one check every comparison goes through: rows of (n, r_n, level, verdict)."""
    rows = []
    for n in range(nmax + 1):
        r = rel(a, b, n)
        lev = max(ref[n], floor20(a, b, n)) if ref is not None else mp.mpf(0)
        v = "OK" if r <= lev else ("WATCH" if r <= fac * lev else "BAD")
        rows.append((n, r, lev, v))
    return rows


def planted(f, n, relchg):
    """A copy of file f with sigma_n multiplied by (1 + relchg), printed to 20 digits as amp_flint_par.py does."""
    g = dict(f, s=[list(p) for p in f["s"]])
    z = val(f["s"][n]) * (1 + mp.mpf(relchg))
    g["s"][n] = [mp.nstr(z.real, 20), mp.nstr(z.imag, 20)]
    return g


def fmt(x):
    return "0" if x == 0 else mp.nstr(x, 2, min_fixed=0, max_fixed=0)


def cmd_tol(raw, out):
    per = {}
    files = {}
    for wf, (a, b) in REF.items():
        fa, fb = load(raw, a), load(raw, b)
        files[wf] = [{k: f[k] for k in ("path", "sha256", "K", "J", "prec", "procs")} for f in (fa, fb)]
        per[wf] = [rel(fa, fb, n) for n in range(NMAX + 1)]
    R = [max(per[wf][n] for wf in per) for n in range(NMAX + 1)]
    print("reference r_n = |sigma(K45 J1500) - sigma(K33 J1000)| / |sigma|, T99-win2 against T83-mac")
    print("| n | wf 0.99 | wf 1.00 | wf 1.01 | R_n (max) |")
    print("|---|---|---|---|---|")
    for n in range(NMAX + 1):
        print(f"| {n} | " + " | ".join(fmt(per[wf][n]) for wf in per) + f" | {fmt(R[n])} |")
    rec = {"rule": "OK r<=max(R,floor); WATCH r<=F*max(R,floor); BAD otherwise; floor = one unit in the 20th digit "
                   "of each component of both strings over |sigma(A)|", "F": F, "nmax": NMAX,
           "R": [mp.nstr(x, 6) for x in R], "per_wf": {wf: [mp.nstr(x, 6) for x in v] for wf, v in per.items()},
           "files": files}
    if out:
        with open(out, "w") as fh:
            json.dump(rec, fh, indent=1)
        print("wrote", out)
    return 0


def read_tol(path):
    with open(path) as fh:
        t = json.load(fh)
    assert t["F"] == F and t["nmax"] == NMAX
    return [mp.mpf(x) for x in t["R"]]


def target_rows(raw, R, plant=None):
    out = {}
    for wf, (a, b) in TARGET.items():
        fa, fb = load(raw, a), load(raw, b)
        if plant and wf == "1.10":
            fa = planted(fa, *plant)
        out[wf] = (fa, fb, check(fa, fb, NMAX, R))
    return out


def cmd_compare(raw, tolpath, plant=None, out=None, quiet=False):
    R = read_tol(tolpath)
    res = target_rows(raw, R, plant)
    bad = []
    if not quiet:
        print("target r_n = |sigma(K45 J1500, #82) - sigma(K33 J1000)| / |sigma|; ratio = r_n / max(R_n, floor_n)")
        print("| n | R_n | 1.09 vs T155-win2 | ratio | 1.10 vs T153-macm3 | ratio | 1.11 vs T155-win2 | ratio |")
        print("|---|---|---|---|---|---|---|---|")
    for n in range(NMAX + 1):
        cells = []
        for wf in TARGET:
            _, r, lev, v = res[wf][2][n]
            cells += [f"{fmt(r)} {v}", fmt(r / lev) if lev else "-"]
            if v == "BAD":
                bad.append((wf, n))
        if not quiet:
            print(f"| {n} | {fmt(R[n])} | " + " | ".join(cells) + " |")
    if not quiet:
        cnt = {v: sum(1 for wf in TARGET for row in res[wf][2] if row[3] == v) for v in ("OK", "WATCH", "BAD")}
        print("verdicts over 3 x 17 entries:", cnt, "BAD at", bad or "none")
    if out:
        rec = {wf: {"files": [{k: f[k] for k in ("path", "sha256", "K", "J", "prec", "procs")} for f in res[wf][:2]],
                    "rows": [[n, mp.nstr(r, 6), mp.nstr(lev, 6), v] for n, r, lev, v in res[wf][2]]}
               for wf in TARGET}
        rec["tol"] = tolpath
        rec["plant"] = plant
        with open(out, "w") as fh:
            json.dump(rec, fh, indent=1)
    return bad


def interp(xs, ys, x):
    """Lagrange interpolation in mpmath."""
    tot = mp.mpc(0)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        w = mp.mpf(1)
        for j, xj in enumerate(xs):
            if j != i:
                w *= (x - xj) / (xi - xj)
        tot += w * yi
    return tot


def smooth_rows(raw, plant=None):
    lad = {wf: load(raw, p) for wf, p in LADDER.items()}
    k45 = {wf: load(raw, TARGET[wf][0]) for wf in TARGET}
    if plant:
        k45["1.10"] = planted(k45["1.10"], *plant)
    res = {}
    for t in TARGET:
        others = [w for w in LADDER if w != t]
        xs = [mp.mpf(w) for w in others]
        rows = []
        for n in range(NMAX + 1):
            ref_y = [val(lad[w]["s"][n]) for w in others]
            mix_y = [val((k45[w] if w in TARGET else lad[w])["s"][n]) for w in others]
            e = abs(val(lad[t]["s"][n]) - interp(xs, ref_y, mp.mpf(t))) / abs(val(lad[t]["s"][n]))
            s = abs(val(k45[t]["s"][n]) - interp(xs, mix_y, mp.mpf(t))) / abs(val(k45[t]["s"][n]))
            lev = max(e, floor20(k45[t], lad[t], n))
            v = "OK" if s <= lev else ("WATCH" if s <= F * lev else "BAD")
            rows.append((n, s, e, v))
        res[t] = rows
    return res


def cmd_smooth(raw, plant=None, out=None, quiet=False):
    res = smooth_rows(raw, plant)
    bad = [(t, n) for t in res for n, _, _, v in res[t] if v == "BAD"]
    if not quiet:
        print("smoothness: s = |K45(t) - P(t)| / |sigma|, P the degree-5 interpolant through the six other ladder nodes")
        print("(K45 at the other two targets, K33 J1000 at 1.06, 1.08, 1.12, 1.14); e = the same for the pure K33 ladder")
        print("| n | s 1.09 | e 1.09 | s 1.10 | e 1.10 | s 1.11 | e 1.11 |")
        print("|---|---|---|---|---|---|---|")
        for n in range(NMAX + 1):
            print(f"| {n} | " + " | ".join(f"{fmt(res[t][n][1])} {res[t][n][3]} | {fmt(res[t][n][2])}"
                                           for t in TARGET) + " |")
        cnt = {v: sum(1 for t in res for row in res[t] if row[3] == v) for v in ("OK", "WATCH", "BAD")}
        print("verdicts over 3 x 17 entries:", cnt, "BAD at", bad or "none")
    if out:
        with open(out, "w") as fh:
            json.dump({t: [[n, mp.nstr(s, 6), mp.nstr(e, 6), v] for n, s, e, v in res[t]] for t in res}
                      | {"plant": plant}, fh, indent=1)
    return bad


def cmd_replica(raw, mutate=None, quiet=False):
    a, b = load(raw, REPLICA[0]), load(raw, REPLICA[1])
    if mutate is not None:
        s = b["s"][mutate][0]
        last = s.split("e")[0][-1]
        b = dict(b, s=[list(p) for p in b["s"]])
        b["s"][mutate][0] = s.replace(s.split("e")[0], s.split("e")[0][:-1] + str((int(last) + 1) % 10), 1)
    n = len(a["s"]) - 1
    rows = check(a, b, n, None)
    eq = [r[0] for r in rows if r[3] == "OK"]
    diff = [r[0] for r in rows if r[3] != "OK"]
    if not quiet:
        print(f"replica {REPLICA[1]} against {REPLICA[0]}: {len(eq)} of {n + 1} equal; differing n: {diff or 'none'}")
    return eq, diff


def cmd_reproduce(raw, quiet=False):
    bad = []
    for wf, vals in POSTED.items():
        fa, fb = load(raw, TARGET[wf][0]), load(raw, TARGET[wf][1])
        for n, v in vals.items():
            got = rel(fa, fb, n)
            if mp.nstr(got, 2) != mp.nstr(mp.mpf(v), 2):
                bad.append((wf, n, v, mp.nstr(got, 3)))
    if not quiet:
        print(f"reproduce #82's table: {sum(map(len, POSTED.values())) - len(bad)} of "
              f"{sum(map(len, POSTED.values()))} posted values recomputed; mismatches: {bad or 'none'}")
    return bad


JSEQ = {wf: (TARGET[wf][1], f"I213-win2/sigF_w{wf}_K45_J1250_p1280.json", TARGET[wf][0]) for wf in TARGET}
JSEQ["1.00"] = ("T83-mac/scan33J1000/sigF_w1.0000_K33_J1000_p960.json",
                "T99-win2/scan45J1250/sigF_w1.0000_K45_J1250_p1280.json", REF["1.0000"][0])


def cmd_jseq(raw, nlist=None, out=None):
    """Where a K45 J1500 / K33 J1000 difference sits: J 1000 -> 1250 against J 1250 -> 1500 (K 45 J 1250 of #213)."""
    nlist = nlist or list(range(NMAX + 1))
    rec = {}
    print("J sequence, r relative to |sigma(J1500)|: a = J1000 vs J1500, b = J1000 vs J1250, c = J1250 vs J1500")
    print("| n | " + " | ".join(f"{wf} a | {wf} b | {wf} c" for wf in JSEQ) + " |")
    print("|---|" + "---|" * (3 * len(JSEQ)))
    files = {wf: [load(raw, p) for p in ps] for wf, ps in JSEQ.items()}
    for n in nlist:
        cells = []
        for wf, (f1000, f1250, f1500) in files.items():
            den = abs(val(f1500["s"][n]))
            a = abs(val(f1000["s"][n]) - val(f1500["s"][n])) / den
            b = abs(val(f1000["s"][n]) - val(f1250["s"][n])) / den
            c = abs(val(f1250["s"][n]) - val(f1500["s"][n])) / den
            cells += [fmt(a), fmt(b), fmt(c)]
            rec.setdefault(wf, []).append([n, mp.nstr(a, 6), mp.nstr(b, 6), mp.nstr(c, 6)])
        print(f"| {n} | " + " | ".join(cells) + " |")
    if out:
        with open(out, "w") as fh:
            json.dump({"files": {wf: [f["path"] for f in fs] for wf, fs in files.items()}, "rows": rec}, fh, indent=1)
    return 0


def cmd_selftest(raw, tolpath):
    ok = True
    eq, diff = cmd_replica(raw)
    ok &= len(eq) == 23 and not diff
    for k in (3, 16, 22):
        eq, diff = cmd_replica(raw, mutate=k)
        ok &= diff == [k]
    ok &= not cmd_reproduce(raw)
    R = read_tol(tolpath)
    base = cmd_compare(raw, tolpath, quiet=True)
    print("compare without a plant: BAD at", base or "none")
    for n in (2, 9, 16):
        fa, fb = load(raw, TARGET["1.10"][0]), load(raw, TARGET["1.10"][1])
        lev = max(R[n], floor20(fa, fb, n))
        hi = cmd_compare(raw, tolpath, plant=(n, 3 * F * lev), quiet=True)
        lo = cmd_compare(raw, tolpath, plant=(n, lev * mp.mpf("0.3")), quiet=True)
        fired = ("1.10", n) in hi and set(hi) - {("1.10", n)} == set(base)
        quiet_lo = set(lo) == set(base)
        print(f"planted 3F x level at n={n}: BAD at {hi}; fires: {fired}. planted 0.3 x level: BAD at {lo or 'none'}; "
              f"silent: {quiet_lo}")
        ok &= fired and quiet_lo
    sbase = cmd_smooth(raw, quiet=True)
    res = smooth_rows(raw)
    for n in (2, 9, 16):
        e = res["1.10"][n][2]
        hi = cmd_smooth(raw, plant=(n, 100 * e), quiet=True)
        fired = ("1.10", n) in hi
        print(f"smooth: planted 100 x e at n={n}: BAD at {hi}; fires: {fired}")
        ok &= fired
    print("smooth without a plant: BAD at", sbase or "none")
    print("selftest", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main(argv):
    argv = list(argv)

    def opt(name):
        for i, a in enumerate(argv):
            if a.startswith(name + "="):
                del argv[i]
                return a.split("=", 1)[1]
            if a == name:
                v = argv[i + 1]
                del argv[i:i + 2]
                return v
        return None
    out, tol, plant, mutate = opt("--out"), opt("--tol"), opt("--plant"), opt("--mutate")
    plant = (int(plant.split(":")[0]), mp.mpf(plant.split(":")[1])) if plant else None
    cmd, raw = argv[1], argv[2]
    if cmd == "tol":
        return cmd_tol(raw, out)
    if cmd == "compare":
        bad_rep = cmd_reproduce(raw)
        if bad_rep:
            print("reproduction control failed, no verdict")
            return 1
        return 1 if cmd_compare(raw, tol, plant, out) else 0
    if cmd == "smooth":
        return 1 if cmd_smooth(raw, plant, out) else 0
    if cmd == "replica":
        eq, diff = cmd_replica(raw, None if mutate is None else int(mutate))
        return 1 if diff else 0
    if cmd == "reproduce":
        return 1 if cmd_reproduce(raw) else 0
    if cmd == "selftest":
        return cmd_selftest(raw, tol)
    if cmd == "jseq":
        return cmd_jseq(raw, out=out)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
