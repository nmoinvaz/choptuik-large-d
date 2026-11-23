"""#1025: every root of the l = 1 multiplet's growth-rate determinant found so far, as CSV, from the raw-data repository.

    python3 ell1_roots.py RAWDIR --commit=HASH [--out=matter/ell1_roots.csv] [--best=matter/ell1_best.csv]
                          [--kinds=matter/ell1_root_kinds.csv]
    python3 ell1_roots.py selftest [--mutate=shift|robust]     exit 0 only if every planted case reads as written

RAWDIR is a checkout (or an extract) of choptuik-large-d-raw at --commit. Two sources:
- hybrid sign scans (d18_hybsign.py rows: every jsonl under I1025-*/ whose rows carry its fields; planted, control
  and copied folders (in/, refcheck/, ref_*) left out):
  grouped by the machine folder that evaluated them and the state they evaluated; repeats must agree in sign; each
  sign change between adjacent points is a root, located as d185_scan_read.changes does (the det-linear estimate,
  and the range of it with the quadratic and cubic fits), robust when both points are noise-robust;
- ell_refine.py records with a `lam` (brentq on the float64 det, up to D 17.5): one root each, method `brentq`.
A root's kind (lambda_1, lambda_1_superseded, discretization) comes from KINDS, the coordinator's rulings by position (D, k_c, a value
within 0.002); every other root is `unidentified`. BEST has one row per D with a lambda_1 root: lambda_1 at the
highest k_c (Linux first), and err the largest of the k_c step on one build, the spread between builds at one k_c,
the spread between states of one build at one k_c (re-converged, other thread counts), the estimator half-range, and
from D 19.5 on a floor of 0.0026 for one regrid's reproducibility, named in err_kind."""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), None)
BUILD = {"dh170": "linux", "dh370": "linux", "dh610": "linux", "rpi5": "linux-arm", "mac": "macos", "macm3": "macos",
         "win": "windows", "win2": "windows", "win2b": "windows", "win2c": "windows"}
SKIP = ("plant", "control", "mutat", "killed", "selftest")
NEED = ("D", "kc", "lam", "sign", "ln", "noise_robust", "f64_sign", "state")
# past D 19.5 one regrid fixes lambda_1 only to about this much: half the span 0.0051 of four states equally converged
# at the |F| floor at D 19.75, k_c 87 (re-convergence, 6 and 8 threads; raw I1025-dh610 reconv_d1975, repro_d1975)
REPRO_FROM, REPRO = 19.5, 0.0026
OTHER = defaultdict(int)


def changes(S):
    """d185_scan_read.changes (raw I1025-mac/d18, blob 7c9d400f), unchanged but for the mutations"""
    x = np.array([r["lam"] for r in S])
    ref = max(r["ln"] for r in S)
    f = np.array([r["sign"] * np.exp(r["ln"] - ref) for r in S])
    out = []
    for i in range(len(S) - 1):
        if S[i]["sign"] == S[i + 1]["sign"]:
            continue
        ok = (S[i]["noise_robust"] and S[i + 1]["noise_robust"]) or MUT == "robust"
        h = x[i + 1] - x[i]
        est = [x[i] + h * f[i] / (f[i] - f[i + 1])]
        for j, k in ((i - 1, 3), (i, 3), (i - 1, 4)):
            if j >= 0 and j + k <= len(S):
                p = np.polyfit(x[j:j + k] - x[i], f[j:j + k], k - 1)
                est += [r.real + x[i] for r in np.roots(p) if abs(r.imag) < 1e-12 and 0 <= r.real <= h]
        if MUT == "shift":
            est = [e + 0.003 for e in est]
        out.append(dict(br=(float(x[i]), float(x[i + 1])), lin=float(est[0]), lo=float(min(est)), hi=float(max(est)),
                        robust=bool(ok), orient=("+" if S[i]["sign"] > 0 else "-") + ("+" if S[i + 1]["sign"] > 0 else "-")))
    return out


def machine_of(path, raw):
    top = os.path.relpath(path, raw).split(os.sep)[0]
    return top.split("-", 1)[1] if top.startswith("I1025-") else top


def scan_groups(raw):
    groups = defaultdict(list)
    files = defaultdict(set)
    for f in sorted(glob.glob(os.path.join(raw, "I1025-*", "**", "*.jsonl"), recursive=True)):
        rel = os.path.relpath(f, raw)
        if (any(s in rel.lower() for s in SKIP) or "/in/" in "/" + rel or "/refcheck/" in rel
                or os.path.basename(f).startswith("ref_")):   # another machine's rows copied as input or reference
            continue
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if not all(k in r for k in NEED):
                    OTHER[os.path.dirname(rel)] += 1          # an older format (not d18_hybsign's rows): left out, counted
                    continue
                key = (machine_of(f, raw), r["state"], float(r["D"]), int(r["kc"]))
                groups[key].append(r)
                files[key].add(os.path.dirname(rel))
    return groups, files


def records(raw):
    """ell_refine / ell_leap records: (tag, D) -> the record, preferring a machine's own copy over an in/ copy"""
    recs = {}
    for f in sorted(glob.glob(os.path.join(raw, "I1025-*", "**", "l1kc*.jsonl"), recursive=True)):
        rel = os.path.relpath(f, raw)
        if any(s in rel.lower() for s in SKIP):
            continue
        tag = os.path.basename(f)[:-6]
        own = "/in/" not in "/" + rel
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if "D" not in r:
                continue
            k = (tag, round(float(r["D"]), 6))
            if k not in recs or (own and not recs[k][1]):
                recs[k] = (r, own, rel)
    return recs


def load_kinds(path):
    if not path or not os.path.exists(path):
        return []
    return [dict(D=float(r["D"]), kc=int(r["kc"]), value=float(r["value"]), build=r.get("build", ""),
                 kind=r["kind"], reason=r["reason"]) for r in csv.DictReader(open(path))]


def kind_of(kinds, D, kc, build, root):
    for k in kinds:
        if abs(k["D"] - D) < 1e-9 and k["kc"] == kc and abs(k["value"] - root) <= 0.002 and k["build"] in ("", build):
            return k["kind"]
    return "unidentified"


def rows_from(raw, commit, kinds):
    groups, files = scan_groups(raw)
    recs = records(raw)
    rows = []
    for (mach, state, D, kc), R in sorted(groups.items()):
        by = {}
        rep_ok = True
        for r in R:
            k = round(r["lam"], 4)
            if k in by:
                a = by[k]
                rep_ok &= a["sign"] == r["sign"] and a.get("noise") == r.get("noise") and abs(a["ln"] - r["ln"]) <= 1e-6
            else:
                by[k] = r
        S = sorted(by.values(), key=lambda r: r["lam"])
        tag = state[4:].rsplit("_D", 1)[0] if state.startswith("sol_") else state
        rec = recs.get((tag, round(D, 6)), (None, None, None))[0] or {}
        base = dict(D=D, kc=kc, eval_machine=mach, build=BUILD.get(mach, "unknown"), state=state,
                    Delta=rec.get("Delta"), F=rec.get("F"), steps=rec.get("steps"), dDelta_dF=rec.get("dDelta_dF"),
                    method="hybrid_scan", n_points=len(S), lam_min=S[0]["lam"], lam_max=S[-1]["lam"],
                    repeats_agree=rep_ok, f64_disagree=sum(1 for r in S if r["sign"] != r["f64_sign"]),
                    raw_folder=";".join(sorted(files[(mach, state, D, kc)])), raw_commit=commit)
        for c in changes(S):
            rows.append(dict(base, root=c["lin"], range_lo=c["lo"], range_hi=c["hi"], bracket_lo=c["br"][0],
                             bracket_hi=c["br"][1], orientation=c["orient"], robust=c["robust"],
                             kind=kind_of(kinds, D, kc, base["build"], c["lin"])))
    for (tag, D), (rec, own, rel) in sorted(recs.items()):
        if not rec.get("lam"):
            continue
        mach = machine_of(os.path.join(raw, rel), raw)
        b = BUILD.get(mach, "unknown")
        rows.append(dict(D=D, kc=int(rec.get("kc", 0)), eval_machine=mach, build=b, state=f"sol_{tag}_D{D:.4f}.npy",
                         Delta=rec.get("Delta"), F=rec.get("F"), steps=rec.get("steps"), dDelta_dF=rec.get("dDelta_dF"),
                         method="brentq", n_points=rec.get("n_eval"), lam_min=None, lam_max=None, repeats_agree=None,
                         f64_disagree=None, raw_folder=os.path.dirname(rel), raw_commit=commit, root=float(rec["lam"]),
                         range_lo=None, range_hi=None, bracket_lo=None, bracket_hi=None, orientation=None,
                         robust=(rec.get("n_signchanges") == 1), kind=kind_of(kinds, D, int(rec.get("kc", 0)), b, float(rec["lam"]))))
    return rows


def best_rows(rows):
    out = []
    lam1 = [r for r in rows if r["kind"] == "lambda_1" and r["robust"]]
    for D in sorted({r["D"] for r in lam1}):
        R = [r for r in lam1 if r["D"] == D]
        kmax = max(r["kc"] for r in R)
        top = sorted([r for r in R if r["kc"] == kmax], key=lambda r: (r["build"] != "linux", r["eval_machine"]))[0]
        errs = {}
        same = [r for r in R if r["build"] == top["build"] and r["kc"] < kmax]
        if same:
            nxt = max(same, key=lambda r: r["kc"])
            errs[f"k_c step {nxt['kc']}->{kmax}"] = abs(top["root"] - nxt["root"])
        for kc in {r["kc"] for r in R}:
            v = [r["root"] for r in R if r["kc"] == kc]
            if len({r["build"] for r in R if r["kc"] == kc}) > 1:
                errs[f"build spread at k_c {kc}"] = max(v) - min(v)
            for b in {r["build"] for r in R if r["kc"] == kc}:          # states of one build at one k_c (re-convergence,
                vb = {r["state"]: r["root"] for r in R if r["kc"] == kc and r["build"] == b}   # threads, machines)
                if len(vb) > 1:
                    errs[f"state spread at k_c {kc} ({b})"] = max(vb.values()) - min(vb.values())
        if top["range_lo"] is not None:
            errs["estimator half-range"] = (top["range_hi"] - top["range_lo"]) / 2
        if D >= REPRO_FROM:                         # one regrid's reproducibility at the |F| floor (#1025, 10:15 UTC)
            errs["regrid reproducibility (half the D 19.75 state span)"] = REPRO
        ek = max(errs, key=errs.get) if errs else "none: one k_c and one build"
        out.append(dict(D=D, lambda1=top["root"], err=errs.get(ek), err_kind=ek, kc=kmax, build=top["build"],
                        Delta=top["Delta"], source=f"{top['raw_folder']} @ {top['raw_commit']}"))
    return out


def write(path, rows, cols):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (f"{r[k]:.10g}" if isinstance(r[k], float) else r[k])) for k in cols})


COLS = ["D", "kc", "build", "eval_machine", "state", "Delta", "F", "steps", "dDelta_dF", "method", "root", "range_lo",
        "range_hi", "bracket_lo", "bracket_hi", "orientation", "robust", "kind", "n_points", "lam_min", "lam_max",
        "repeats_agree", "f64_disagree", "raw_folder", "raw_commit"]
BCOLS = ["D", "lambda1", "err", "err_kind", "kc", "build", "Delta", "source"]


def selftest():
    def rows(grid, f, nr=()):
        return [dict(lam=l, sign=1.0 if f(l) > 0 else -1.0, ln=float(np.log(abs(f(l)))), noise_robust=(l not in nr))
                for l in grid]
    ok = []
    a = 1.2865                                           # one root, det nearly linear through it
    c = changes(rows([round(1.20 + 0.02 * i, 2) for i in range(18)], lambda l: (l - a) * (1 + 0.5 * (l - 1.4))))
    ok.append(len(c) == 1 and c[0]["br"] == (1.28, 1.3) and abs(c[0]["lin"] - a) < 5e-4 and c[0]["robust"])
    a1, a2 = 1.4954, 1.5293                              # two roots 0.034 apart on a 0.005 grid
    g = [round(1.47 + 0.005 * i, 3) for i in range(17)]
    c = changes(rows(g, lambda l: (l - a1) * (l - a2) * 40 * (1 + 0.3 * (l - 1.5))))
    ok.append(len(c) == 2 and all(abs(x["lin"] - t) < 5e-4 for x, t in zip(c, (a1, a2))))
    c = changes(rows(g, lambda l: (l - a1) * (l - a2) * 40, nr=(1.495,)))   # a non-robust endpoint
    ok.append(len(c) == 2 and not c[0]["robust"] and c[1]["robust"])
    print(f"selftest: {sum(ok)} of {len(ok)} planted cases read as written")
    return 0 if all(ok) else 1


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    o = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if args and args[0] == "selftest":
        return selftest()
    if not args:
        print(__doc__)
        return 2
    raw = args[0]
    kinds = load_kinds(o.get("kinds"))
    rows = rows_from(raw, o.get("commit", ""), kinds)
    write(o.get("out", "ell1_roots.csv"), rows, COLS)
    best = best_rows(rows)
    write(o.get("best", "ell1_best.csv"), best, BCOLS)
    n = defaultdict(int)
    for r in rows:
        n[(r["method"], r["kind"])] += 1
    if OTHER:
        print(f"left out (rows without d18_hybsign's fields): {sum(OTHER.values())} rows in {len(OTHER)} folders: "
              + ", ".join(sorted(OTHER)[:6]) + (" ..." if len(OTHER) > 6 else ""))
    print(f"{len(rows)} roots: " + ", ".join(f"{m} {k} {c}" for (m, k), c in sorted(n.items())) + f"; best rows {len(best)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
