#!/usr/bin/env python3
"""Every computed value of Delta in one table, hp/data/delta_all.csv, one row per D from the low-D runs to the large-D
chain's frontier, with the kind of computation, an error estimate and what that estimate is.

    cd hp && python3 export_delta_all.py RAWDIR [--verify] [--mutate=dd|qd|pinned|chain_fit|chain|held|frontier|projector|t85
                                                                   |fits|fits_table]
    cd hp && python3 export_delta_all.py RAWDIR --interp_check [--mutate=interp]

RAWDIR is a clone of the raw-data repository (git clone https://github.com/nmoinvaz/choptuik-large-d-raw). --verify
re-reads every row of the written CSV from the source and state its row names, rebuilds the chains' err, err_kind and
data/delta_fits.csv, and exits 1 on any difference; --mutate changes one row of that kind by one unit in its second
part before the verification (the control, which must fail; it exits 2 when the table has no row of that kind);
--mutate=held labels the last chain_fit row chain, and --mutate=frontier moves the first chain row below the chain_fit
frontier (or, when chain_fit replaces every chain row, puts the chain's last row back), --mutate=projector drops the
first projector err (data/projector_shift.csv's), and --mutate=t85 gives the first
chain_fit row a T85-mac source (the controls of the chains' checks); --mutate=fits gives the first chain_fit row whose
err is interpolated the other fit's direct difference instead, the value an export that compares only at equal D
writes, and --mutate=fits_table drops that row's interpolated entry from data/delta_fits.csv.
--interp_check is the interpolation's control: at D 186 to 190, where every chain has a state, each comparison fit's
Delta interpolated from its other states (as below) must equal its state's to 1e-14. It exits 1 otherwise;
--mutate=interp interpolates in D instead of 1/(D - 2), whose truncation error the check must catch.

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
          through each run's start state to T81's quad-double state at 73.0. err as for chain, or, where a run with
          the harmonics held on a fit of another degree (T140: 6 and 10, against chain_fit's 8) differs by more, the
          largest such difference, a spread over the fit's degree, not a bound. Each fit's difference is taken where
          one of its runs reached the same D, or, from D 191.5 on (where the degree-6 chain stepped at integer D and
          the degree-10 chain at half-integer D), interpolated from the six states of that fit's chain nearest D by
          the polynomial of degree 5 in 1/(D - 2) (Lagrange, mpmath), when they bracket D. err_kind names the fit
          and how: "fits d6", "fits d10", "fits d6 interp", "fits d10 interp", with " one-sided" appended where
          only some of the comparison fits reach that D (the frontier of a comparison chain). From D 215 on, the
          freeze projector's rounding enters too: Delta's shift from the double-double projector's rows to the exact
          ones (data/projector_shift.csv, log-linear between its points; #589, #595), err_kind "projector", or
          "projector first order" where it rests on a first-order figure. The runs on the exact projector (EXACT:
          #603's rerun from D 225.13 and #608's comparison chains) carry no projector term and are compared with each
          other only; a run started from a base or a copy is traced through that file's own source. Above the
          comparison chains' frontier, where no fit's difference is measured, the fits term is extrapolated
          log-linearly in D from the last six rows that have one (err_kind "fits extrapolated", an estimate). Each fit's difference is also written
          to data/delta_fits.csv (D, then per fit the signed Delta of that fit minus chain_fit's, how and the states).
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
# the fits that chain_fit's (degree 8) is compared with, T140's, and their labels in err_kind
FITS = {"data/held_fit_d6.json": "d6", "data/held_fit_d10.json": "d10"}
FITS_OUT = os.path.join(HERE, "data", "delta_fits.csv")
FITS_COLS = ["D"] + [f"{c}_{lab}" for lab in FITS.values() for c in ("diff", "how", "states")]
# from this D on, a fit with no state at a chain_fit row's D is interpolated there (#478)
INTERP_FROM = 191.5
# the freeze projector's rounding: Delta's shift from the double-double projector's rows to the exact ones (#589, #595)
PROJ = os.path.join(HERE, "data", "projector_shift.csv")
# the runs on the exact freeze projector (main's kernel from #523 on): their rows carry no projector term, and they are
# compared only with each other (#603, #608; #595's polished bases)
EXACT = ("I595-mac/", "I603-win2/", "I608-dh610/")
HOPS = 10
# above the comparison chains' frontier, the fits term is extrapolated from the last EXTRAP_N rows that have one
EXTRAP_N = 6
INTERP_N = 6
# the interpolation's control: leave-one-out at these D, where every chain has a state
CONTROL_D = (186.0, 187.0, 188.0, 189.0, 190.0)
CONTROL_TOL = 1e-14


def total(parts):
    """the exact sum of the parts (f64 values or their repr strings; empty parts are 0), in mpmath"""
    return sum((mp.mpf(float(p)) for p in parts if p != ""), mp.mpf(0))


def projector_err(D):
    """(|shift|, label) of the freeze projector's rounding at D: log-linear in |shift| between the points of
    data/projector_shift.csv, the last interval's slope past its end; None below its first point (the shift there is
    below 5e-19 and falls about 0.8 in ln per unit of D). The label is "projector", or "projector first order" where a
    bracketing point is a first-order figure, not a measured shift."""
    if not os.path.exists(PROJ):
        return None
    pts = sorted((float(r["D"]), abs(float(r["shift"])), r["kind"]) for r in csv.DictReader(open(PROJ)))
    if D < pts[0][0] - TOL_D:
        return None
    for (d0, s0, k0), (d1, s1, k1) in zip(pts[:-1], pts[1:]):
        if D <= d1 + TOL_D:
            break
    t = (D - d0) / (d1 - d0)
    val = float(np.exp((1 - t) * np.log(s0) + t * np.log(s1)))
    first = (k0 != "measured" and t < 1 - 1e-12) or (k1 != "measured" and t > 1e-12)
    return val, "projector first order" if first else "projector"


def exact(rel):
    """a state of a run on the exact freeze projector (EXACT)"""
    return rel.replace(os.sep, "/").startswith(EXACT)


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
    """the states from the highest-D state of the runs whose harmonics are held as kind says (chain_fit: on a fit other
    than the comparison fits), traced back through each run's start state"""
    mine = [r for r in states if held[os.path.dirname(r)][0] == HELD[kind] and held[os.path.dirname(r)][1] not in FITS]
    if not mine:
        return []
    return trace(states, by_dir, held, max(mine, key=lambda r: states[r][0]["D"]), kind)


def fit_chains(states, by_dir, held):
    """each comparison fit's chain: (D, Delta, state) of the states from that fit's highest-D state traced back"""
    out = {}
    for f, lab in FITS.items():
        mine = [r for r in states if held[os.path.dirname(r)] == ("fit", f)]
        if mine:
            bb = trace(states, by_dir, held, max(mine, key=lambda r: states[r][0]["D"]), f"fit {lab}")
            out[f] = [(mp.mpf(states[r][0]["D"]), total(states[r][1]), r) for r in bb]
    return out


def lagrange(pts, x):
    """the polynomial through pts [(x_i, y_i)] at x"""
    out = mp.mpf(0)
    for i, (xi, yi) in enumerate(pts):
        w = mp.mpf(1)
        for j, (xj, _) in enumerate(pts):
            if j != i:
                w *= (x - xj) / (xi - xj)
        out += w * yi
    return out


def interp(chain, D, var="1/n"):
    """Delta of a fit's chain [(D, Delta, state)] at D: the polynomial of degree INTERP_N - 1 in 1/(D - 2) (var "D": in
    D) through its INTERP_N states nearest D, other than one at D; (value, first state, last state), or None where
    they do not bracket D"""
    D = mp.mpf(D)
    near = sorted(sorted((p for p in chain if abs(p[0] - D) >= TOL_D), key=lambda p: (abs(p[0] - D), p[0]))[:INTERP_N])
    if len(near) < INTERP_N or not near[0][0] < D < near[-1][0]:
        return None
    t = (lambda d: d) if var == "D" else (lambda d: 1 / (d - 2))
    return lagrange([(t(d), v) for d, v, _ in near], t(D)), near[0][2], near[-1][2]


def trace(states, by_dir, held, top, kind):
    """the states from top traced back through each run's start state"""
    bb, cur, upper = [], os.path.dirname(top), states[top][0]["D"]
    while True:
        seg = [r for r in by_dir[cur] if states[r][0]["D"] <= upper + TOL_D]
        src = states[seg[0]][0]["source"]
        lineage_check(cur, seg, src, states)
        bb = seg + bb
        # a run started from a base or a copy (#603's polished base_pin3, I589-mac's s225) continues from that file's
        # own source, up to HOPS files deep
        hop, root = 0, None
        while src not in states and hop < HOPS and os.path.exists(os.path.join(RAW, src)):
            nxt = json.loads(str(np.load(os.path.join(RAW, src))["meta"])).get("source")
            if not nxt:
                break
            src, hop = nxt, hop + 1
        if src not in states:
            root = src
            break
        cur, upper = os.path.dirname(src), states[src][0]["D"]
    if len({held[os.path.dirname(r)] for r in bb}) != 1:
        sys.exit(f"{kind}: the backbone mixes held harmonics {sorted({str(held[os.path.dirname(r)]) for r in bb})}")
    print(f"{kind}: {len(bb)} states from {root} to {top}")
    return bb


def chain_rows(keep=False, quiet=False):
    """the chain_fit backbone's rows and the chain's above its frontier (every chain row with keep), and the rows of
    data/delta_fits.csv"""
    states, by_dir = chain_states()
    held = {d: held_of(d) for d in by_dir}
    fit = backbone(states, by_dir, held, "chain_fit")
    trend = backbone(states, by_dir, held, "chain")
    chains = fit_chains(states, by_dir, held)
    on = set(fit) | set(trend)
    front = max((states[r][0]["D"] for r in fit), default=-np.inf)
    rows, fits = [], []
    for kind, bb in (("chain_fit", fit), ("chain", [r for r in trend if keep or states[r][0]["D"] > front + TOL_D])):
        for rel in bb:
            m, parts = states[rel]
            v, err, ek = total(parts), "", ""
            sysk = (held[os.path.dirname(rel)], parts[3] != "", exact(rel))
            # runs: the same held harmonics and precision; a comparison fit's (T140) states at this D
            spread = {"runs": [], **{f: [] for f in FITS}}
            for o, (mo, po) in states.items():
                if o in on or abs(mo["D"] - m["D"]) >= TOL_D:
                    continue
                osys = (held[os.path.dirname(o)], po[3] != "", exact(o))
                if osys == sysk:
                    spread["runs"].append((total(po) - v, o))
                elif kind == "chain_fit" and osys[0][0] == "fit" and osys[0][1] in FITS and osys[1:] == sysk[1:]:
                    spread[osys[0][1]].append((total(po) - v, o))
            big, frow = {}, {}
            if spread["runs"]:
                big["runs"] = max(abs(d) for d, _ in spread["runs"])
            pe = None if exact(rel) else projector_err(m["D"])
            if pe:
                big[pe[1]] = pe[0]
            for f, lab in (FITS.items() if kind == "chain_fit" else ()):
                if spread[f]:
                    d, o = max(spread[f], key=lambda t: abs(t[0]))
                    how, src = "direct", o
                elif m["D"] >= INTERP_FROM - TOL_D and f in chains and (ip := interp(chains[f], m["D"])):
                    d, how, src = ip[0] - v, "interp", f"{ip[1]} .. {ip[2]}"
                else:
                    continue
                big[f"fits {lab}" + (" interp" if how == "interp" else "")] = abs(d)
                frow.update({f"diff_{lab}": f"{float(d):+.4e}", f"how_{lab}": how, f"states_{lab}": src})
            if not quiet:
                for d, o in spread["runs"]:
                    print(f"  check D {m['D']:.4f} (runs): {o} - {rel} = {float(d):+.3e}")
                for lab in FITS.values():
                    if f"diff_{lab}" in frow:
                        print(f"  check D {m['D']:.4f} (fits {lab}, {frow[f'how_{lab}']}): {frow[f'states_{lab}']}"
                              f" - {rel} = {frow[f'diff_{lab}']}")
            if big:
                ek = max(big, key=big.get)
                err = f"{float(big[ek]):.1e}"
                # some comparison fits reach this D and others do not: the spread is over fewer fits than its neighbours'
                if kind == "chain_fit" and 0 < sum(f"diff_{lab}" in frow for lab in FITS.values()) < len(FITS):
                    ek += " one-sided"
            if frow:
                fits.append({c: "" for c in FITS_COLS} | frow | {"D": fmt(m["D"])})
            rows.append(dict(D=fmt(m["D"]), Delta_q0=parts[0], Delta_q1=parts[1], Delta_q2=parts[2], Delta_q3=parts[3],
                             kind=kind, err=err, err_kind=ek,
                             source=os.path.join(os.path.dirname(rel), "anm_run.jsonl"), state=rel))
    extrapolate_fits(rows)
    return rows, fits


def extrapolate_fits(rows):
    """above the last chain_fit row with a fits term (the comparison chains' frontier) no fit's difference is measured,
    and the largest term elsewhere is that one: there the fits term of the last EXTRAP_N rows that have one is
    extrapolated log-linearly in D, its slope at least 0, and taken where it exceeds the row's own err (err_kind "fits
    extrapolated", an estimate, not a measured spread)"""
    fr = [(float(r["D"]), float(r["err"])) for r in rows if r["kind"] == "chain_fit" and r["err_kind"].startswith("fits")]
    if len(fr) < EXTRAP_N:
        return
    xs, ys = np.array([d for d, _ in fr[-EXTRAP_N:]]), np.log([e for _, e in fr[-EXTRAP_N:]])
    slope, icpt = np.polyfit(xs, ys, 1)
    slope = max(slope, 0.0)
    for r in rows:
        if r["kind"] == "chain_fit" and float(r["D"]) > fr[-1][0] + TOL_D:
            est = float(np.exp(icpt + slope * float(r["D"])))
            if est > (float(r["err"]) if r["err"] else 0.0):
                r["err"], r["err_kind"] = f"{est:.1e}", "fits extrapolated"


def interp_check(var="1/n"):
    """the interpolation's control: at CONTROL_D each comparison fit's chain interpolated from its other states against
    its state there; True when every difference is within CONTROL_TOL"""
    states, by_dir = chain_states()
    held = {d: held_of(d) for d in by_dir}
    chains = fit_chains(states, by_dir, held)
    bad, n = 0, 0
    for f, lab in FITS.items():
        for D in CONTROL_D:
            at = [p for p in chains.get(f, []) if abs(p[0] - D) < TOL_D]
            ip = interp(chains.get(f, []), D, var)
            n += 1
            if not at or ip is None:
                bad += 1
                print(f"DIFFERS: {lab} D {D}: {'no state' if not at else 'not bracketed'}")
                continue
            d = float(ip[0] - at[0][1])
            ok = abs(d) <= CONTROL_TOL
            bad += not ok
            print(f"  {'ok  ' if ok else 'DIFFERS:'} {lab} D {D}: interpolated ({var}, {ip[1]} .. {ip[2]}) - state"
                  f" = {d:+.3e}")
    print(f"verify: {n - bad} of {n} interpolated values within {CONTROL_TOL:g} of their states")
    return bad == 0


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
    chain, fits = chain_rows()
    span = fit_span(chain)
    rows = {}
    for r in qd_rows() + dd + [r for r in pinned_rows(dmax) if not replaced(r, span)] + chain:
        k = round(float(r["D"]), 9)
        if k not in rows or RANK[r["kind"]] < RANK[rows[k]["kind"]]:
            rows[k] = r
    pub = published()
    return [{**rows[k], **published_cols(rows[k], pub)} for k in sorted(rows)], fits


def verify(rows, mutate=""):
    """each row against the source it names, read anew; the chains' err and data/delta_fits.csv rebuilt"""
    ftab = list(csv.DictReader(open(FITS_OUT))) if os.path.exists(FITS_OUT) else []
    if mutate in ("fits", "fits_table"):
        # the first row with an interpolated fit: the other fit's direct difference only, as an export comparing at
        # equal D only writes it
        hit = next(((r, t) for r in rows if r["kind"] == "chain_fit" and r["err_kind"].endswith(" interp")
                    for t in ftab if t["D"] == r["D"]), None)
        if hit is None:
            print("no interpolated fits rows, so nothing to mutate")
            sys.exit(2)
        r, t = hit
        lab = r["err_kind"].split()[1]
        if mutate == "fits":
            other = next(x for x in FITS.values() if x != lab and t[f"how_{x}"] == "direct")
            r["err"], r["err_kind"] = f"{abs(float(t[f'diff_{other}'])):.1e}", f"fits {other}"
            print(f"mutated: the chain_fit row at D {r['D']} given err {r['err']} ({r['err_kind']}), {lab} left out")
        else:
            t.update({f"{c}_{lab}": "" for c in ("diff", "how", "states")})
            print(f"mutated: the {lab} entry of delta_fits.csv at D {r['D']} dropped")
    elif mutate == "projector":
        # the first row whose err is the projector's: given its old err (the runs' or none), as without the shift
        r = next((r for r in rows if r["err_kind"].startswith("projector")), None)
        if r is None:
            print("no projector rows, so nothing to mutate")
            sys.exit(2)
        r["err"], r["err_kind"] = "", ""
        print(f"mutated: the chain_fit row at D {r['D']} without its projector err")
    elif mutate == "held":
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
        r = [x for x in chain_rows(keep=True, quiet=True)[0] if x["kind"] == "chain"][-1]
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
    # the chains' err and err_kind, and every row of data/delta_fits.csv, rebuilt from the states
    crow, cfits = chain_rows(quiet=True)
    want = {r["D"]: r for r in crow}
    for r in rows:
        w = want.get(r["D"]) if r["kind"] in HELD else None
        if w and (r["err"], r["err_kind"]) != (w["err"], w["err_kind"]):
            bad += 1
            print(f"DIFFERS: {r['kind']} D {r['D']}: err {r['err']} ({r['err_kind']}) against {w['err']}"
                  f" ({w['err_kind']})")
    have = {t["D"]: t for t in ftab}
    for t in cfits:
        if have.pop(t["D"], None) != t:
            bad += 1
            print(f"DIFFERS: delta_fits.csv D {t['D']}: against {t}")
    for d in have:
        bad += 1
        print(f"DIFFERS: delta_fits.csv D {d}: no chain_fit row has it")
    print(f"verify: {len(rows) - bad} of {len(rows)} rows equal their sources, {len(cfits)} rows of delta_fits.csv"
          " rebuilt")
    return bad == 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:] + "=").split("=")[:2] for a in sys.argv[1:] if a.startswith("--"))
    if not args:
        sys.exit(__doc__)
    RAW = os.path.abspath(args[0])
    if "interp_check" in opts:
        sys.exit(0 if interp_check("D" if opts.get("mutate") == "interp" else "1/n") else 1)
    if "verify" in opts or "mutate" in opts:
        rows = list(csv.DictReader(open(OUT)))
        sys.exit(0 if verify(rows, opts.get("mutate", "")) else 1)
    rows, fits = build()
    for path, cols, rs in ((OUT, COLS, rows), (FITS_OUT, FITS_COLS, fits)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rs)
    print(f"{len(fits)} rows -> {FITS_OUT}")
    n = defaultdict(int)
    for r in rows:
        n[r["kind"]] += 1
    print(f"{len(rows)} rows -> {OUT}: " + ", ".join(f"{k} {n[k]}" for k in RANK))
    for k in RANK:
        ks = [r for r in rows if r["kind"] == k]
        if ks:
            print(f"  {k:7s} D {ks[0]['D']} to {ks[-1]['D']}")
