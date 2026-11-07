#!/usr/bin/env python3
"""Every computed value of Delta in one table, hp/data/delta_all.csv, one row per D from the low-D runs to the large-D
chain's frontier, with the kind of computation, an error estimate and what that estimate is.

    cd hp && python3 export_delta_all.py RAWDIR [--verify] [--mutate=dd|qd|pinned|chain_fit|chain|held|frontier|t85]

RAWDIR is a clone of the raw-data repository (git clone https://github.com/nmoinvaz/choptuik-large-d-raw). --verify
re-reads every row of the written CSV from the source and state its row names and exits 1 on any difference;
--mutate changes one row of that kind by one unit in its second part before the verification (the control, which
must fail; it exits 2 when the table has no row of that kind); --mutate=held labels the last chain_fit row chain, and
--mutate=frontier moves the first chain row below the chain_fit frontier (or, when chain_fit replaces every chain row,
puts the chain's last row back), and --mutate=t85 gives the first chain_fit row a T85-mac source (the controls of the
chains' checks).

Columns: D; Delta in up to four parts, Delta = q0 + q1 + q2 + q3 (each an f64; q2 and q3 are empty where the value is
double-double); kind; err and err_kind; the source record; published_ref, published_Delta and published_diff, the most
precise value of data/published_delta.csv at this D (6 decimals), as printed, and this row's Delta minus it (empty
where none is listed); the state file (relative to RAWDIR, empty where none was stored). The kinds, in order of
precedence where two share a D:
  qd      the full system converged in quad-double (data/delta_qd.csv, export_qd.py). err: the spread between two
          starts with their own Taylor plans (err_kind "starts"), not a bound.
  dd      the full system in double-double (data/delta_dd.csv, export_dd.py). err and err_kind are that file's: "dM",
          the change from the next lower tau resolution M, or "check", the size of the resolution and
          harmonic-freezing checks near this D.
  pinned  a pinned double-double run (T73, T75: pin 7 and pin 5; T85-mac: last node 0.8 and 0.85) above the dd
          points' last D: Delta_full1, the pinned Delta plus one linearised full-system correction, as the README
          graph draws them (solver/plot_delta.py). Where both pins (layouts) reached a D, the row is pin 7's (0.8's) and
          err is the difference (err_kind "pins", "layouts"), a spread between two systems, not a bound. T85-mac's rows
          inside chain_fit's span are left out, since its layouts differ by 2.3e-7 to 8.3e-7 and chain_fit replaces
          them (notes/QUAD_DOUBLE.md 13.25).
  chain_fit  the asymptotic numerical method's chain (anm_run.py) with harmonics 5-13 of p1 held on a 1/n fit of the
          full-system states (--held_fit, notes/HELD_HARMONICS.md), its backbone traced from its highest-D state back
          through each run's start state to T81's quad-double state at 73.0. err as for chain. The fit's own error is
          not in err.
  chain   the chain with the harmonics held on T75's trend (notes/QUAD_DOUBLE.md 13.17-13.24), its backbone traced
          the same way back to T75's pin 7 state at 73.27; double-double states to 79.27, quad-double after. Its rows at
          or below the chain_fit frontier are left out, since chain_fit replaces them. Where another run reached the
          same D with the same held harmonics and precision (another step history, corrector or machine), err is the
          largest difference (err_kind "runs"), a spread, not a bound. The held trend biases every chain value low, by
          5e-8 at D 80 growing to 3.2e-6 at D 200 (13.29); err does not include it.
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
COLS = ["D", "Delta_q0", "Delta_q1", "Delta_q2", "Delta_q3", "kind", "err", "err_kind", "source", "published_ref",
        "published_Delta", "published_diff", "state"]
RANK = {"qd": 0, "dd": 1, "pinned": 2, "chain_fit": 3, "chain": 4}
HELD = {"chain_fit": "fit", "chain": "trend"}
# pinned runs whose rows chain_fit replaces inside its span (their layouts' spread, 13.25)
FIT_REPLACES = ("T85-mac/",)
PINNED = (("T73/t73/pin7.jsonl", "T73/t73/pin5.jsonl", "pins"), ("T75/pin7.jsonl", "T75/pin5.jsonl", "pins"),
          ("T85-mac/x80.jsonl", "T85-mac/x85.jsonl", "layouts"))
TOL_D = 1e-9


def total(parts):
    """the exact sum of the parts (f64 values or their repr strings; empty parts are 0), in mpmath"""
    return sum((mp.mpf(float(p)) for p in parts if p != ""), mp.mpf(0))


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


def chain_states():
    """every anm_run.py state of the raw-data repository, and the states of each folder in D"""
    states = {}
    for p in glob.glob(os.path.join(RAW, "**", "anm_pin3_D*.npz"), recursive=True):
        rel = os.path.relpath(p, RAW)
        states[rel] = load_state(rel)
    by_dir = defaultdict(list)
    for rel in states:
        by_dir[os.path.dirname(rel)].append(rel)
    for d in by_dir:
        by_dir[d].sort(key=lambda r: states[r][0]["D"])
        # a run resumed from a state of its own folder (T124-win) continues that folder's lineage
        srcs = {states[r][0].get("source") for r in by_dir[d]}
        if len({x for x in srcs if os.path.dirname(x) != d}) != 1:
            sys.exit(f"chain: {d} has states from several starts {srcs}")
    return states, by_dir


def backbone(states, by_dir, held, kind):
    """the states from the highest-D state of the runs whose harmonics are held as kind says, traced back through each
    run's start state"""
    mine = [r for r in states if held[os.path.dirname(r)][0] == HELD[kind]]
    if not mine:
        return []
    top = max(mine, key=lambda r: states[r][0]["D"])
    bb, cur, upper = [], os.path.dirname(top), states[top][0]["D"]
    while True:
        seg = [r for r in by_dir[cur] if states[r][0]["D"] <= upper + TOL_D]
        src = states[seg[0]][0]["source"]
        lineage_check(cur, seg, src, states)
        bb = seg + bb
        if src not in states:
            root = src
            break
        cur, upper = os.path.dirname(src), states[src][0]["D"]
    if len({held[os.path.dirname(r)] for r in bb}) != 1:
        sys.exit(f"{kind}: the backbone mixes held harmonics {sorted({str(held[os.path.dirname(r)]) for r in bb})}")
    print(f"{kind}: {len(bb)} states from {root} to {top}")
    return bb


def chain_rows(keep=False):
    """the chain_fit backbone's rows and the chain's above its frontier (every chain row with keep)"""
    states, by_dir = chain_states()
    held = {d: held_of(d) for d in by_dir}
    fit = backbone(states, by_dir, held, "chain_fit")
    trend = backbone(states, by_dir, held, "chain")
    on = set(fit) | set(trend)
    front = max((states[r][0]["D"] for r in fit), default=-np.inf)
    rows = []
    for kind, bb in (("chain_fit", fit), ("chain", [r for r in trend if keep or states[r][0]["D"] > front + TOL_D])):
        for rel in bb:
            m, parts = states[rel]
            v, err, others = total(parts), "", []
            sysk = (held[os.path.dirname(rel)], parts[3] != "")
            for o, (mo, po) in states.items():
                if o not in on and abs(mo["D"] - m["D"]) < TOL_D and (held[os.path.dirname(o)], po[3] != "") == sysk:
                    others.append((float(total(po) - v), o))
            if others:
                dmax = max(others, key=lambda t: abs(t[0]))
                err = f"{abs(dmax[0]):.1e}"
                for d, o in others:
                    print(f"  check D {m['D']:.4f}: {o} - {rel} = {d:+.3e}")
            rows.append(dict(D=fmt(m["D"]), Delta_q0=parts[0], Delta_q1=parts[1], Delta_q2=parts[2], Delta_q3=parts[3],
                             kind=kind, err=err, err_kind="runs" if err else "",
                             source=os.path.join(os.path.dirname(rel), "anm_run.jsonl"), state=rel))
    return rows


def held_of(folder):
    """how a run holds the harmonics: ("fit", its file) for --held_fit, else ("trend", its trend; None: held constant,
    the runs before --trend)"""
    f = os.path.join(RAW, folder, "anm_run.jsonl")
    r = json.loads(open(f).readline()) if os.path.exists(f) else {}
    return ("fit", r["held_fit"]) if r.get("held_fit") else ("trend", r.get("trend"))


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


def published():
    """data/published_delta.csv by D (6 decimals): its rows (reference, value as printed, stated uncertainty)"""
    pub = defaultdict(list)
    for line in open(os.path.join(HERE, "data", "published_delta.csv")):
        if line.startswith("#") or line.startswith("D,"):
            continue
        d, v, u, ref = line.strip().split(",")
        pub[round(float(d), 6)].append((ref, v, float(u)))
    return pub


def published_cols(r, pub):
    """the most precise published value at the row's D: reference, value, and this row's Delta minus it"""
    hit = pub.get(round(float(r["D"]), 6))
    if not hit:
        return dict(published_ref="", published_Delta="", published_diff="")
    ref, v, _ = min(hit, key=lambda t: t[2])
    diff = total([r[f"Delta_q{i}"] for i in range(4)]) - mp.mpf(v)
    return dict(published_ref=ref, published_Delta=v, published_diff=format(float(diff), ".2e"))


def fit_span(rows):
    """the D range of the chain_fit rows"""
    fit = [float(r["D"]) for r in rows if r["kind"] == "chain_fit"]
    return (min(fit), max(fit)) if fit else (np.inf, -np.inf)


def replaced(r, span):
    """a row that chain_fit replaces: the chain's at or below its frontier, and FIT_REPLACES' inside its span"""
    D = float(r["D"])
    if r["kind"] == "chain":
        return D <= span[1] + TOL_D
    return r["source"].startswith(FIT_REPLACES) and span[0] - TOL_D <= D <= span[1] + TOL_D


def build():
    dd = dd_rows()
    dmax = max(float(r["D"]) for r in dd)
    chain = chain_rows()
    span = fit_span(chain)
    rows = {}
    for r in qd_rows() + dd + [r for r in pinned_rows(dmax) if not replaced(r, span)] + chain:
        k = round(float(r["D"]), 9)
        if k not in rows or RANK[r["kind"]] < RANK[rows[k]["kind"]]:
            rows[k] = r
    pub = published()
    return [{**rows[k], **published_cols(rows[k], pub)} for k in sorted(rows)]


def verify(rows, mutate=""):
    """each row against the source it names, read anew"""
    if mutate == "held":
        r = [r for r in rows if r["kind"] == "chain_fit"][-1]
        r["kind"] = "chain"
        print(f"mutated: the chain_fit row at D {r['D']} labelled chain")
    elif mutate == "t85":
        r = next(r for r in rows if r["kind"] == "chain_fit")
        r["source"] = f"{FIT_REPLACES[0]}x80.jsonl:D{r['D']}"
        print(f"mutated: the chain_fit row at D {r['D']} given the source {r['source']}")
    elif mutate == "frontier" and any(r["kind"] == "chain" for r in rows):
        r = next(r for r in rows if r["kind"] == "chain")
        r["D"] = fmt(max(float(x["D"]) for x in rows if x["kind"] == "chain_fit") - 1)
        print(f"mutated: the first chain row moved to D {r['D']}")
    elif mutate == "frontier":
        # chain_fit replaces every chain row, so the chain's last row is put back
        r = [x for x in chain_rows(keep=True) if x["kind"] == "chain"][-1]
        rows.append({**r, **published_cols(r, published())})
        print(f"mutated: the chain row at D {r['D']}, below the frontier, put back")
    elif mutate and not any(r["kind"] == mutate for r in rows):
        print(f"no {mutate} rows, so nothing to mutate")
        sys.exit(2)
    elif mutate:
        r = next(r for r in rows if r["kind"] == mutate)
        r["Delta_q1"] = fmt(float(r["Delta_q1"]) + np.spacing(float(r["Delta_q1"])))
        print(f"mutated: {mutate} row at D {r['D']}")
    bad = 0
    pub = published()
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
            h = held_of(os.path.dirname(r["state"]))[0]
            if h != HELD[r["kind"]]:
                bad += 1
                print(f"DIFFERS: {r['kind']} D {r['D']}: its run holds the harmonics on a {h}")
        pc = published_cols({**r, **{f"Delta_q{i}": want[i] for i in range(4)}}, pub)
        if parts != want or any(r[k] != v for k, v in pc.items()):
            bad += 1
            print(f"DIFFERS: {r['kind']} D {r['D']}: {parts} against {want}, published {pc}")
    # chain_fit replaces the chain up to its frontier, and FIT_REPLACES' rows inside its span
    span = fit_span(rows)
    low = [f"{r['kind']} {r['D']} {r['source']}" for r in rows if replaced(r, span)]
    if low:
        bad += len(low)
        print(f"DIFFERS: {len(low)} rows that chain_fit replaces (span {span}): {low[:3]}")
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
