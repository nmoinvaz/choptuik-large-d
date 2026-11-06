#!/usr/bin/env python3
"""Every computed value of Delta in one table, hp/data/delta_all.csv, one row per D from the low-D runs to the large-D
chain's frontier, with the kind of computation, an error estimate and what that estimate is.

    cd hp && python3 export_delta_all.py RAWDIR [--verify] [--mutate=dd|qd|pinned|chain]

RAWDIR is a clone of the raw-data repository (git clone https://github.com/nmoinvaz/choptuik-large-d-raw). --verify
re-reads every row of the written CSV from the source and state its row names and exits 1 on any difference;
--mutate changes one row of that kind by one unit in its second part before the verification (the control, which
must fail).

Columns: D; Delta in up to four parts, Delta = q0 + q1 + q2 + q3 (each an f64; q2 and q3 are empty where the value is
double-double); kind; err and err_kind; the source record; the state file (relative to RAWDIR, empty where none was
stored). The kinds, in order of precedence where two share a D:
  qd      the full system converged in quad-double (data/delta_qd.csv, export_qd.py). err: the spread between two
          starts with their own Taylor plans (err_kind "starts"), not a bound.
  dd      the full system in double-double (data/delta_dd.csv, export_dd.py). err and err_kind are that file's: "dM",
          the change from the next lower tau resolution M, or "check", the size of the resolution and
          harmonic-freezing checks near this D.
  pinned  a pinned double-double run (T73, T75: pin 7 and pin 5; T85-mac: last node 0.8 and 0.85) above the dd
          points' last D: Delta_full1, the pinned Delta plus one linearised full-system correction, as the README
          graph draws them (solver/plot_delta.py). Where both pins (layouts) reached a D, the row is pin 7's (0.8's) and
          err is the difference (err_kind "pins", "layouts"), a spread between two systems, not a bound.
  chain   the asymptotic numerical method's chain (anm_run.py; harmonics 5-13 of p1 held on T75's trend, notes/
          QUAD_DOUBLE.md 13.17-13.24), its backbone traced from the highest-D state back to T75's pin 7 state at
          73.27 through each run's start state; double-double states to 79.27, quad-double after. Where another run
          reached the same D with the same held trend and precision (another step history, corrector or machine), err is
          the largest difference (err_kind "runs"), a spread, not a bound. The held harmonics bias every chain value
          by about 1e-8 to 1e-7 at D 79-86, growing with D (13.19); err does not include it.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import mpmath as mp
import numpy as np

mp.mp.dps = 80
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "delta_all.csv")
COLS = ["D", "Delta_q0", "Delta_q1", "Delta_q2", "Delta_q3", "kind", "err", "err_kind", "source", "state"]
RANK = {"qd": 0, "dd": 1, "pinned": 2, "chain": 3}
PINNED = (("T73/t73/pin7.jsonl", "T73/t73/pin5.jsonl", "pins"), ("T75/pin7.jsonl", "T75/pin5.jsonl", "pins"),
          ("T85-mac/x80.jsonl", "T85-mac/x85.jsonl", "layouts"))
TOL_D = 1e-9


def total(parts):
    """the exact sum of the parts (empty parts are 0), in mpmath"""
    return sum((mp.mpf(p) for p in parts if p != ""), mp.mpf(0))


def fmt(x):
    return repr(float(x))


def dd_rows():
    rows = []
    for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_dd.csv"))):
        rows.append(dict(D=r["D"], Delta_q0=r["Delta_hi"], Delta_q1=r["Delta_lo"], Delta_q2="", Delta_q3="", kind="dd",
                         err=r["Delta_err"], err_kind=r["err_kind"], source=f"data/delta_dd.csv:{r['run']}", state=""))
    return rows


def qd_state(src):
    """the converged state of a delta_qd.csv source"""
    f, _, start = src.partition(":")
    if start:
        return os.path.join(os.path.dirname(f), start[:-4] + "_full.npz")
    folder, name = os.path.split(f)
    r = json.loads(open(os.path.join(RAW, f)).readline())
    return os.path.join(folder, r["state"])


def qd_rows():
    rows = []
    for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_qd.csv"))):
        st = qd_state(r["source"])
        if not os.path.exists(os.path.join(RAW, st)):
            sys.exit(f"qd row D {r['D']}: state {st} missing in {RAW}")
        rows.append(dict(D=r["D"], Delta_q0=r["Delta_q0"], Delta_q1=r["Delta_q1"], Delta_q2=r["Delta_q2"],
                         Delta_q3=r["Delta_q3"], kind="qd", err=r["spread"], err_kind="starts" if r["spread"] else "",
                         source=f"data/delta_qd.csv:{r['source']}", state=st))
    return rows


def records(f):
    return [json.loads(line) for line in open(os.path.join(RAW, f)) if line.strip()]


def pinned_state(f, r):
    folder, name = os.path.split(f)
    lay = name[:-6]
    st = os.path.join(folder, f"pin{r['pin']}_D{r['D']:.4f}.npz" if lay.startswith("pin") else f"pin{r['pin']}_{lay}_D{r['D']:.4f}.npz")
    return st if os.path.exists(os.path.join(RAW, st)) else ""


def pinned_rows(Dmin):
    rows = []
    for main, other, ek in PINNED:
        sec = {round(r["D"], 6): r for r in records(other) if "Delta_full1" in r}
        used = set()
        for f in (main, other):
            for r in records(f):
                if "Delta_full1" not in r or r["D"] <= Dmin:
                    continue
                k = round(r["D"], 6)
                if f == other and k in used:
                    continue
                err = ""
                if f == main and k in sec:
                    s = sec[k]
                    err = f"{float(abs(total([s['Delta_full1'], s['Delta_full1_lo']]) - total([r['Delta_full1'], r['Delta_full1_lo']]))):.1e}"
                    used.add(k)
                rows.append(dict(D=fmt(r["D"]), Delta_q0=fmt(r["Delta_full1"]), Delta_q1=fmt(r["Delta_full1_lo"]), Delta_q2="",
                                 Delta_q3="", kind="pinned", err=err, err_kind=ek if err else "", source=f"{f}:D{r['D']}",
                                 state=pinned_state(f, r)))
    return rows


def load_state(p):
    z = np.load(os.path.join(RAW, p))
    m = json.loads(str(z["meta"]))
    parts = [z["zh"][-1], z["zl"][-1]] + ([z["z2"][-1], z["z3"][-1]] if "z3" in z.files else [])
    return m, [fmt(x) for x in parts] + [""] * (4 - len(parts))


def chain_rows():
    states = {}
    for p in glob.glob(os.path.join(RAW, "**", "anm_pin3_D*.npz"), recursive=True):
        rel = os.path.relpath(p, RAW)
        m, parts = load_state(rel)
        states[rel] = (m, parts)
    by_dir = defaultdict(list)
    for rel, (m, parts) in states.items():
        by_dir[os.path.dirname(rel)].append(rel)
    for d in by_dir:
        by_dir[d].sort(key=lambda r: states[r][0]["D"])
        srcs = {states[r][0].get("source") for r in by_dir[d]}
        if len(srcs) != 1:
            sys.exit(f"chain: {d} has states from several starts {srcs}")
    top = max(states, key=lambda r: states[r][0]["D"])
    backbone, cur, upper = [], os.path.dirname(top), states[top][0]["D"]
    while True:
        seg = [r for r in by_dir[cur] if states[r][0]["D"] <= upper + TOL_D]
        src = states[seg[0]][0]["source"]
        lineage_check(cur, seg, src, states)
        backbone = seg + backbone
        if src not in states:
            root = src
            break
        cur, upper = os.path.dirname(src), states[src][0]["D"]
    on = set(backbone)
    trends = {d: trend_of(d) for d in by_dir}
    if len({trends[os.path.dirname(r)] for r in backbone}) != 1:
        sys.exit(f"chain: the backbone mixes held trends {sorted({str(trends[os.path.dirname(r)]) for r in backbone})}")
    rows = []
    print(f"chain: {len(backbone)} states from {root} to {top}")
    for rel in backbone:
        m, parts = states[rel]
        v, err, others = total(parts), "", []
        sysk = (trends[os.path.dirname(rel)], parts[3] != "")
        for o, (mo, po) in states.items():
            if o not in on and abs(mo["D"] - m["D"]) < TOL_D and (trends[os.path.dirname(o)], po[3] != "") == sysk:
                others.append((float(total(po) - v), o))
        if others:
            dmax = max(others, key=lambda t: abs(t[0]))
            err = f"{abs(dmax[0]):.1e}"
            for d, o in others:
                print(f"  check D {m['D']:.4f}: {o} - {rel} = {d:+.3e}")
        rows.append(dict(D=fmt(m["D"]), Delta_q0=parts[0], Delta_q1=parts[1], Delta_q2=parts[2], Delta_q3=parts[3], kind="chain",
                         err=err, err_kind="runs" if err else "", source=os.path.join(os.path.dirname(rel), "anm_run.jsonl"),
                         state=rel))
    return rows


def trend_of(folder):
    """the held harmonics' trend of a run (None: held constant, the runs before --trend)"""
    f = os.path.join(RAW, folder, "anm_run.jsonl")
    return json.loads(open(f).readline()).get("trend") if os.path.exists(f) else None


def lineage_check(folder, seg, src, states):
    """each state of the segment is an accepted step of the folder's anm_run.jsonl from the state before it (the
    first from the run's start state)"""
    f = os.path.join(RAW, folder, "anm_run.jsonl")
    if not os.path.exists(f):
        sys.exit(f"chain: {folder} has no anm_run.jsonl")
    steps = []
    for r in records(os.path.relpath(f, RAW)):
        for a in r.get("attempts", []):
            if a.get("accepted"):
                steps.append((r["D0"], a["D"]))
    prev = mp.mpf(json.loads(str(np.load(os.path.join(RAW, src))["meta"]))["D"]) if os.path.exists(os.path.join(RAW, src)) else None
    for rel in seg:
        D = states[rel][0]["D"]
        ok = any(abs(d1 - D) < TOL_D and (prev is None or abs(d0 - prev) < TOL_D) for d0, d1 in steps)
        if not ok:
            sys.exit(f"chain: {rel} is not an accepted step from D {prev} in {f}")
        prev = D


def build():
    dd = dd_rows()
    dmax = max(float(r["D"]) for r in dd)
    rows = {}
    for r in qd_rows() + dd + pinned_rows(dmax) + chain_rows():
        k = round(float(r["D"]), 9)
        if k not in rows or RANK[r["kind"]] < RANK[rows[k]["kind"]]:
            rows[k] = r
    return [rows[k] for k in sorted(rows)]


def verify(rows, mutate=""):
    """each row against the source it names, read anew"""
    if mutate:
        r = next(r for r in rows if r["kind"] == mutate)
        r["Delta_q1"] = fmt(float(r["Delta_q1"]) + np.spacing(float(r["Delta_q1"])))
        print(f"mutated: {mutate} row at D {r['D']}")
    bad = 0
    dd = {r["D"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_dd.csv")))}
    qd = {r["D"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_qd.csv")))}
    recs = {}
    for r in rows:
        parts = [r[f"Delta_q{i}"] for i in range(4)]
        if r["kind"] == "dd":
            s = dd[r["D"]]
            want = [s["Delta_hi"], s["Delta_lo"], "", ""]
        elif r["kind"] == "qd":
            s = qd[r["D"]]
            want = [s[f"Delta_q{i}"] for i in range(4)]
        elif r["kind"] == "pinned":
            f, _, d = r["source"].partition(":D")
            if f not in recs:
                recs[f] = {repr(x["D"]): x for x in records(f) if "Delta_full1" in x}
            s = recs[f][repr(float(d))]
            want = [fmt(s["Delta_full1"]), fmt(s["Delta_full1_lo"]), "", ""]
        else:
            want = load_state(r["state"])[1]
        if parts != want:
            bad += 1
            print(f"DIFFERS: {r['kind']} D {r['D']}: {parts} against {want}")
    print(f"verify: {len(rows) - bad} of {len(rows)} rows equal their sources")
    return bad == 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:] + "=").split("=")[:2] for a in sys.argv[1:] if a.startswith("--"))
    if not args:
        sys.exit(__doc__)
    RAW = os.path.abspath(args[0])
    if "verify" in opts or "mutate" in opts:
        rows = list(csv.DictReader(open(OUT)))
        sys.exit(0 if verify(rows, opts.get("mutate", "")) else 1)
    rows = build()
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    n = defaultdict(int)
    for r in rows:
        n[r["kind"]] += 1
    print(f"{len(rows)} rows -> {OUT}: " + ", ".join(f"{k} {n[k]}" for k in RANK))
    for k in RANK:
        ks = [r for r in rows if r["kind"] == k]
        if ks:
            print(f"  {k:7s} D {ks[0]['D']} to {ks[-1]['D']}")
