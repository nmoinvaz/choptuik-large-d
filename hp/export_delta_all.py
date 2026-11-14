#!/usr/bin/env python3
"""Every computed value of Delta in one table, hp/data/delta_all.csv, one row per D from the low-D runs to the large-D
chain's frontier, with the kind of computation, an error estimate and what that estimate is.

    cd hp && python3 export_delta_all.py RAWDIR [--verify] [--mutate=dd|qd|pinned|pinned_qd|gap|neighbours|cross
                                                   |chain_fit|chain|held|frontier|projector|t85|fits|fits_table|half
                                                   |qd_neighbours|dm_neighbours|one_sided]
    cd hp && python3 export_delta_all.py RAWDIR --interp_check [--mutate=interp]
    cd hp && python3 export_delta_all.py RAWDIR --lam_screen [--mutate=lam]

RAWDIR is a clone of the raw-data repository (git clone https://github.com/nmoinvaz/choptuik-large-d-raw). --verify
re-reads every row of the written CSV from the source and state its row names, rebuilds the err and err_kind of the qd
rows (from data/delta_qd.csv), the low-D dd rows (from data/delta_dd_lowD.csv), the pinned and the chain rows, and
data/delta_fits.csv, and exits 1 on any difference; --mutate changes one row of that kind by one unit in
its second part before the verification (the control, which must fail; it exits 2 when the table has no row of that
kind); --mutate=pinned_qd drops the last two parts of the first quad-double pinned row, --mutate=gap the err of the
first "dd-qd gap" row, --mutate=neighbours the err of the first "pins neighbours" row, and --mutate=cross puts T73's pin
5 back at the first shared waypoint of T75's pin 7 and T73's pin 5, without an err (the pinned rules' controls);
--mutate=qd_neighbours, dm_neighbours and one_sided drop the err of the first "starts neighbours", "dM neighbours"
and "dM neighbours one-sided" row (the controls of the qd and low-D rules, #764);
--mutate=half drops the err of the first chain_fit row whose fits are all interpolated below D 191.5 (the half steps'
control); --mutate=held labels the last chain_fit row chain, and
--mutate=frontier moves the first chain row below the chain_fit
frontier (or, when chain_fit replaces every chain row, puts the chain's last row back), --mutate=projector drops the
first projector err (data/projector_shift.csv's), and --mutate=t85 gives the first
chain_fit row a T85-mac source (the controls of the chains' checks); --mutate=fits gives the first chain_fit row whose
err is interpolated the other fit's direct difference instead, the value an export that compares only at equal D
writes, and --mutate=fits_table drops that row's interpolated entry from data/delta_fits.csv.
--interp_check is the interpolation's control: at D 173, 174 and 186 to 191, where every chain has a state (those
beside the half steps 173.5 and 190.5 among them), each comparison fit's Delta interpolated from its other states (as
below) must equal its state's to 1e-14. It exits 1 otherwise;
--mutate=interp interpolates in D instead of 1/(D - 2), whose truncation error the check must catch.
--lam_screen is the gauge rate's smoothness screen (#622, #637): lam_g of each chain_fit backbone state (hp/lam_gauge.py,
from its parts) against the polynomial of degree SCREEN_DEG in 1/(D - 2) through the SCREEN_N nearest other backbone states (a cubic
through six leaves its own truncation, up to 1.4e-10 near D 77, above SCREEN_TOL); it exits 1 when one
departs by more than SCREEN_TOL. Every other chain_fit-held state is set against the backbone's polynomial at its D too, and
those above SCREEN_TOL are listed (information: they are not rows of the table). --mutate=lam adds 1e-10 to the lam_g of
the backbone state nearest D 240, and the screen must fail.

Columns: D; Delta in up to four parts, Delta = q0 + q1 + q2 + q3 (each an f64; q2 and q3 are empty where the value is
double-double); kind; err and err_kind; the source record; published_ref, published_Delta and published_diff, the most
precise value of data/published_delta.csv at this D (6 decimals), as printed, and this row's Delta minus it (empty
where none is listed); the state file (relative to RAWDIR, empty where none was stored). The kinds, in order of
precedence where two share a D:
  qd      the full system converged in quad-double (data/delta_qd.csv, export_qd.py). err: the spread between two
          starts with their own Taylor plans (err_kind "starts"), not a bound. At 71.0 and 72.8 the second start
          stalled (raw qd2start-mac/second.jsonl, |F| 2e-19 and 8e-18), so the row has one start, and err is the larger
          spread of the nearest rows below and above with two starts (err_kind "starts neighbours", #764), an
          estimate, not a bound. A row with one start and no such row on one side keeps no err.
  dd      the full system in double-double (data/delta_dd.csv, export_dd.py). err and err_kind are that file's: "dM",
          the change from the next lower tau resolution M; "dM neighbours" or "dM neighbours one-sided" at the low-D
          rows solved at one M only (3.72-3.8 at M 80, 4.25-8.2 at M 48), that M's departure from the best value at
          the nearest D below and above that have a larger M (export_lowD.py, #764); or "check", the size of the
          resolution and harmonic-freezing checks near this D. Each is an estimate, not a bound.
  pinned  a pinned double-double run (T73 then T75: pin 7 and pin 5; T85-mac: last node 0.8 and 0.85) above the dd
          points' last D: Delta_full1, the pinned Delta plus one linearised full-system correction, as the README
          graph draws them (solver/plot_delta.py). T75 continues T73's pins (each T75 pin starts 0.005 past the same
          pin's last T73 state), so each pin is the two runs joined. Where both pins (layouts) reached a D (a shared
          waypoint, also where T75's pin 7 meets T73's pin 5 at 70.65 to 70.95), the row is pin 7's (0.8's) and err is
          the difference (err_kind "pins", "layouts"), a spread between two systems, not a bound. T85-mac's rows
          inside chain_fit's span are left out, since its layouts differ by 2.3e-7 to 8.3e-7 and chain_fit replaces
          them (notes/QUAD_DOUBLE.md 13.25). Three rules set the other rows (#756, #759):
          (a) where T84-mac's quad-double Delta_full1 exists (PINNED_QD, notes/QUAD_DOUBLE.md 12.4: the pinned
          solution converged in quad-double plus the full-system Newton step's Delta component, four parts each), it
          replaces the double-double rows at that D. The row is the pin with the most free harmonics (pin 7, or 3 at
          the frontier D 73.265722), its eight parts summed exactly and written as four, and err the difference of
          the two pins' exact sums (err_kind "pins qd"), a spread between two systems, not a bound.
          (b) a double-double row that only one pin reached takes the larger of the dd-qd gaps at the nearest D below
          and above it with a quad-double value, that of (a) or the full system's of data/delta_qd.csv (68.5 to
          73.265722). A gap is the largest difference between a double-double Delta_full1 of either pin at that D and
          a quad-double value there (err_kind "dd-qd gap"). It is an estimate, not a bound.
          (c) such a row also takes the larger spread of the nearest shared waypoints below and above it, those below
          the dd points' last D included (err_kind "pins neighbours", "layouts neighbours"). It is an estimate, not a
          bound.
          The row's err is the larger of (b) and (c) where both exist, and err_kind names that term. A row with
          neither (no bracket on one side for either) keeps no err, and none exists. The rule is #759's: the
          double-double error scatters by an order of magnitude from one D to the next, so neither term alone covers
          the other (at 71.0 to 71.5 the gap is 7.6e-17 and the waypoint 71.177466's spread 8.7e-16; at 70.5 the gap
          is 1.9e-16 and the neighbouring spreads at most 1.2e-16).
  chain_fit  the asymptotic numerical method's chain (anm_run.py) with harmonics 5-13 of p1 held on a 1/n fit of the
          full-system states (--held_fit, notes/HELD_HARMONICS.md), its backbone traced from its highest-D state back
          through each run's start state to T81's quad-double state at 73.0. err as for chain, or, where a run with
          the harmonics held on a fit of another degree (T140: 6 and 10, against chain_fit's 8) differs by more, the
          largest such difference, a spread over the fit's degree, not a bound. Each fit's difference is taken where
          one of its runs reached the same D, or else interpolated from the six states of that fit's chain nearest D
          by the polynomial of degree 5 in 1/(D - 2) (Lagrange, mpmath), when they bracket D. That happens from D 191.5
          on, where the degree-6 chain stepped at integer D and the degree-10 chain at half-integer D, and at the
          backbone's two half steps below it, 173.5 and 190.5, where both stepped at integer D (#759). err_kind names
          the fit and how: "fits d6", "fits d10", "fits d6 interp", "fits d10 interp", with " one-sided" appended where
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
import bisect
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
# (the first pin's runs, the second pin's runs, err_kind): T75 continues T73's pins, so each pin joins the two runs
PINNED = ((("T73/t73/pin7.jsonl", "T75/pin7.jsonl"), ("T73/t73/pin5.jsonl", "T75/pin5.jsonl"), "pins"),
          (("T85-mac/x80.jsonl",), ("T85-mac/x85.jsonl",), "layouts"))
# the quad-double Delta_full1 of the pinned solutions (qd_fullcorr.py, notes/QUAD_DOUBLE.md 12.4), four parts each of
# Delta_pinned and of the full-system Newton step's Delta component (#756)
PINNED_QD = "T84-mac/*_fullcorr.jsonl"
TOL_D = 1e-9
# the fits that chain_fit's (degree 8) is compared with, T140's, and their labels in err_kind
FITS = {"data/held_fit_d6.json": "d6", "data/held_fit_d10.json": "d10"}
FITS_OUT = os.path.join(HERE, "data", "delta_fits.csv")
FITS_COLS = ["D"] + [f"{c}_{lab}" for lab in FITS.values() for c in ("diff", "how", "states")]
# the freeze projector's rounding: Delta's shift from the double-double projector's rows to the exact ones (#589, #595)
PROJ = os.path.join(HERE, "data", "projector_shift.csv")
# the runs on the exact freeze projector (main's kernel from #523 on): their rows carry no projector term, and they are
# compared only with each other (#603, #608; #595's polished bases)
EXACT = ("I595-mac/", "I603-win2/", "I608-dh610/")
HOPS = 10
# above the comparison chains' frontier, the fits term is extrapolated from the last EXTRAP_N rows that have one
EXTRAP_N = 6
# the gauge rate's smoothness screen (--lam_screen): neighbours in the local cubic and the largest departure accepted
SCREEN_N = 8
SCREEN_DEG = 5
SCREEN_TOL = 1e-11
INTERP_N = 6
# the interpolation's control: leave-one-out at these D, where every chain has a state (173, 174, 190 and 191 beside
# the half steps 173.5 and 190.5)
CONTROL_D = (173.0, 174.0, 186.0, 187.0, 188.0, 189.0, 190.0, 191.0)
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


def qd_err(qd):
    """{D: (err, err_kind)} of the rows of data/delta_qd.csv: the two starts' spread, or, for a row with one start, the
    larger spread of the nearest rows below and above with two (#764)"""
    two = sorted((float(r["D"]), float(r["spread"])) for r in qd if r["spread"])
    out = {}
    for r in qd:
        D = float(r["D"])
        lo, hi = [s for d, s in two if d < D][-1:], [s for d, s in two if d > D][:1]
        out[r["D"]] = ((r["spread"], "starts") if r["spread"] else
                       (f"{max(lo + hi):.1e}", "starts neighbours") if lo and hi else ("", ""))
    return out


def qd_rows():
    rows = []
    qd = list(csv.DictReader(open(os.path.join(HERE, "data", "delta_qd.csv"))))
    errs = qd_err(qd)
    for r in qd:
        st = qd_state(r["source"])
        if not os.path.exists(os.path.join(RAW, st)):
            sys.exit(f"qd row D {r['D']}: state {st} missing in {RAW}")
        rows.append(dict(D=r["D"], Delta_q0=r["Delta_q0"], Delta_q1=r["Delta_q1"], Delta_q2=r["Delta_q2"],
                         Delta_q3=r["Delta_q3"], kind="qd", err=errs[r["D"]][0], err_kind=errs[r["D"]][1],
                         source=f"data/delta_qd.csv:{r['source']}", state=st))
    return rows


def records(f):
    return [json.loads(line) for line in open(os.path.join(RAW, f)) if line.strip()]


def pinned_state(f, r):
    folder, name = os.path.split(f)
    lay = name[:-6]
    st = os.path.join(folder, f"pin{r['pin']}_D{r['D']:.4f}.npz" if lay.startswith("pin") else f"pin{r['pin']}_{lay}_D{r['D']:.4f}.npz")
    return st if os.path.exists(os.path.join(RAW, st)) else ""


def full1_qd(r):
    """the exact sum of a PINNED_QD record's eight parts, Delta_pinned_q and correction_q"""
    with mp.workprec(1200):
        return sum((mp.mpf(p) for p in r["Delta_pinned_q"] + r["correction_q"]), mp.mpf(0))


def parts4(x):
    """x as four f64 parts, each the nearest f64 to what the ones before leave of it"""
    out = []
    with mp.workprec(1200):
        for _ in range(4):
            out.append(float(x))
            x -= mp.mpf(out[-1])
    return [fmt(p) for p in out]


def pinned_qd():
    """the quad-double Delta_full1 by D (6 decimals): [(pin, exact sum, file, record)], the highest pin first"""
    out = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(RAW, PINNED_QD))):
        f = os.path.relpath(p, RAW).replace(os.sep, "/")
        for r in records(f):
            out[round(r["D"], 6)].append((r["pin"], full1_qd(r), f, r))
    for k in out:
        out[k].sort(key=lambda t: -t[0])
    return out


def full1(r):
    """the exact sum of a pinned record's double-double Delta_full1"""
    return total([r["Delta_full1"], r["Delta_full1_lo"]])


def nearest(keys, k):
    """the nearest of the sorted keys strictly below k and strictly above it (None where there is none)"""
    i = bisect.bisect_left(keys, k)
    j = i + 1 if i < len(keys) and keys[i] == k else i
    return (keys[i - 1] if i > 0 else None), (keys[j] if j < len(keys) else None)


def pinned_rows(Dmin):
    # dd: the double-double Delta_full1 by D; wp: each pair's spread at its shared waypoints, those at or below Dmin
    # included; grp: each run's pair
    rows, dd, wp, grp = [], defaultdict(list), {}, {}
    for mains, others, ek in PINNED:
        sec = {round(r["D"], 6): r for f in others for r in records(f) if "Delta_full1" in r}
        wp[ek] = {}
        for f in mains + others:
            grp[f] = ek
            for r in records(f):
                if "Delta_full1" not in r:
                    continue
                k = round(r["D"], 6)
                if f in mains and k in sec:
                    wp[ek][k] = abs(full1(sec[k]) - full1(r))
                if r["D"] <= Dmin:
                    continue
                dd[k].append(full1(r))
                # a shared waypoint's row is the first pin's
                if f in others and k in wp[ek]:
                    continue
                err = f"{float(wp[ek][k]):.1e}" if f in mains and k in sec else ""
                rows.append(dict(D=fmt(r["D"]), Delta_q0=fmt(r["Delta_full1"]), Delta_q1=fmt(r["Delta_full1_lo"]), Delta_q2="",
                                 Delta_q3="", kind="pinned", err=err, err_kind=ek if err else "", source=f"{f}:D{r['D']}",
                                 state=pinned_state(f, r)))
    # (a) the quad-double Delta_full1 replaces the double-double rows at its D, with the pins' spread as err
    qd = {k: v for k, v in pinned_qd().items() if k > Dmin}
    rows = [r for r in rows if round(float(r["D"]), 6) not in qd]
    # the quad-double values by D, (a)'s and the full system's
    ref = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_qd.csv"))):
        ref[round(float(r["D"]), 6)].append(total([r[f"Delta_q{i}"] for i in range(4)]))
    for k, sols in qd.items():
        pin, v, f, r = sols[0]
        err = max((abs(v - w) for _, w, _, _ in sols[1:]), default=None)
        st = f"{os.path.dirname(f)}/{r['state']}"
        rows.append(dict(D=fmt(r["D"]), **dict(zip(("Delta_q0", "Delta_q1", "Delta_q2", "Delta_q3"), parts4(v))),
                         kind="pinned", err="" if err is None else f"{float(err):.1e}",
                         err_kind="" if err is None else "pins qd", source=f"{f}:D{r['D']}",
                         state=st if os.path.exists(os.path.join(RAW, st)) else ""))
        ref[k] += [w for _, w, _, _ in sols]
    # the dd-qd gap: the largest difference between a double-double Delta_full1 and a quad-double value at a D
    gap = {k: max(abs(a - w) for a in dd[k] for w in ws) for k, ws in ref.items() if dd.get(k)}
    gk, wk = sorted(gap), {ek: sorted(v) for ek, v in wp.items()}
    # (b) and (c): a row that one pin reached takes the larger of the gaps and of the spreads that bracket it
    for r in rows:
        if r["err"] or r["Delta_q2"]:
            continue
        k, ek, terms = round(float(r["D"]), 6), grp[r["source"].partition(":D")[0]], {}
        lo, hi = nearest(gk, k)
        if lo is not None and hi is not None:
            terms["dd-qd gap"] = max(gap[lo], gap[hi])
        lo, hi = nearest(wk[ek], k)
        if lo is not None and hi is not None:
            terms[f"{ek} neighbours"] = max(wp[ek][lo], wp[ek][hi])
        if terms:
            r["err_kind"] = max(terms, key=terms.get)
            r["err"] = f"{float(terms[r['err_kind']]):.1e}"
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
                elif f in chains and (ip := interp(chains[f], m["D"])):
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
        # the first row with an interpolated fit and another fit's direct difference: that difference only, as an
        # export comparing at equal D only writes it
        hit = next(((r, t) for r in rows if r["kind"] == "chain_fit" and r["err_kind"].endswith(" interp")
                    for t in ftab if t["D"] == r["D"] and "direct" in (t[f"how_{x}"] for x in FITS.values())), None)
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
    elif mutate in ("pinned_qd", "gap", "neighbours", "qd_neighbours", "dm_neighbours", "one_sided"):
        # a quad-double pinned row as its first two parts only; a dd-qd gap or pins neighbours row without its err
        # (as before #756 and #759); a qd row with one start, a dd row at one M without its err (as before #764)
        test = {"pinned_qd": lambda r: r["kind"] == "pinned" and r["Delta_q2"],
                "gap": lambda r: r["err_kind"] == "dd-qd gap",
                "neighbours": lambda r: r["kind"] == "pinned" and r["err_kind"].endswith(" neighbours"),
                "qd_neighbours": lambda r: r["err_kind"] == "starts neighbours",
                "dm_neighbours": lambda r: r["err_kind"] == "dM neighbours",
                "one_sided": lambda r: r["err_kind"] == "dM neighbours one-sided"}[mutate]
        r = next((r for r in rows if test(r)), None)
        if r is None:
            print(f"no {mutate} rows, so nothing to mutate")
            sys.exit(2)
        if mutate == "pinned_qd":
            r["Delta_q2"] = r["Delta_q3"] = ""
        else:
            r["err"] = r["err_kind"] = ""
        print(f"mutated: the {mutate} row at D {r['D']}")
    elif mutate == "cross":
        # the first shared waypoint of T75's pin 7 and T73's pin 5 as T73's pin 5 alone, as before #759
        p7, p5 = PINNED[0][0][1], PINNED[0][1][0]
        r = next((r for r in rows if r["source"].startswith(p7 + ":") and r["err_kind"] == "pins"
                  and any(abs(x["D"] - float(r["D"])) < TOL_D for x in records(p5) if "Delta_full1" in x)), None)
        if r is None:
            print("no cross-run waypoints, so nothing to mutate")
            sys.exit(2)
        s = next(x for x in records(p5) if "Delta_full1" in x and abs(x["D"] - float(r["D"])) < TOL_D)
        r.update(Delta_q0=fmt(s["Delta_full1"]), Delta_q1=fmt(s["Delta_full1_lo"]), err="", err_kind="",
                 source=f"{p5}:D{s['D']}", state=pinned_state(p5, s))
        print(f"mutated: the cross-run waypoint at D {r['D']} as {r['source']} without an err")
    elif mutate == "half":
        # the first chain_fit row whose fits are all interpolated (a half step of the backbone) without its err, as an
        # export interpolating only from D 191.5 writes it
        half = {t["D"] for t in ftab if all(t[f"how_{x}"] == "interp" for x in FITS.values())}
        r = next((r for r in rows if r["kind"] == "chain_fit" and r["D"] in half), None)
        if r is None:
            print("no half-step rows, so nothing to mutate")
            sys.exit(2)
        r["err"] = r["err_kind"] = ""
        print(f"mutated: the chain_fit row at D {r['D']} without its err")
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
    # the err of the qd rows and of the low-D dd rows rebuilt from delta_qd.csv and the per-M rows of delta_dd_lowD.csv:
    # the change from the next lower M, or the rule for a D solved at one M (#764)
    qerr = qd_err(list(qd.values()))
    import export_lowD
    lowd = list(csv.DictReader(open(os.path.join(HERE, "data", "delta_dd_lowD.csv"))))
    lerr = {k: (f"{v:.1e}", ek) for k, (v, ek) in export_lowD.neighbour_err(lowd).items()}
    lerr.update({float(r["D"]): (format(abs(float(r["change_from_lower_M"])), ".1e"), "dM") for r in lowd
                 if r["best"] == "1" and r["change_from_lower_M"]})
    # the pinned rows rebuilt, for their err (rules (a) and (b)); where two runs share a D, the first, as build() keeps
    pin = {}
    for r in pinned_rows(max(float(x) for x in dd)):
        pin.setdefault(r["D"], r)
    recs = {}
    for r in rows:
        parts = [r[f"Delta_q{i}"] for i in range(4)]
        if r["kind"] in ("dd", "qd"):
            s = (dd if r["kind"] == "dd" else qd)[r["D"]]
            want = [s["Delta_hi"], s["Delta_lo"], "", ""] if r["kind"] == "dd" else [s[f"Delta_q{i}"] for i in range(4)]
            w = (qerr[r["D"]] if r["kind"] == "qd" else lerr.get(float(r["D"]), ("", ""))
                 if s["run"].startswith("lowD-") else (s["Delta_err"], s["err_kind"]))
            if (r["err"], r["err_kind"]) != w:
                bad += 1
                print(f"DIFFERS: {r['kind']} D {r['D']}: err {r['err']} ({r['err_kind']}) against {w}")
        elif r["kind"] == "pinned":
            f, _, d = r["source"].partition(":D")
            if f not in recs:
                recs[f] = {repr(x["D"]): x for x in records(f) if "Delta_full1" in x or "correction_q" in x}
            s = recs[f][repr(float(d))]
            want = parts4(full1_qd(s)) if "correction_q" in s else \
                [fmt(s["Delta_full1"]), fmt(s["Delta_full1_lo"]), "", ""]
            w = pin.get(r["D"])
            if w is None or (r["err"], r["err_kind"], r["source"]) != (w["err"], w["err_kind"], w["source"]):
                bad += 1
                print(f"DIFFERS: pinned D {r['D']}: err {r['err']} ({r['err_kind']}) from {r['source']} against"
                      f" {w and (w['err'], w['err_kind'], w['source'])}")
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


def lam_screen(mutate=""):
    """lam_g of each chain_fit backbone state against the polynomial of degree SCREEN_DEG in 1/(D - 2) through its
    SCREEN_N nearest other backbone states; True when none departs by more than SCREEN_TOL. Other chain_fit-held states are listed against the
    backbone's polynomial at their D (information)."""
    import lam_gauge as LG
    states, by_dir = chain_states()
    held = {d: held_of(d) for d in by_dir}
    bb = backbone(states, by_dir, held, "chain_fit")

    def lam(rel):
        parts, meta = LG.load(os.path.join(RAW, rel))
        return mp.mpf(LG.lam_g(parts, meta)), mp.mpf(meta.get("D_str", repr(float(meta["D"]))))
    pts = [(rel,) + lam(rel) for rel in bb]
    if mutate == "lam":
        k = min(range(len(pts)), key=lambda i: abs(pts[i][2] - 240))
        pts[k] = (pts[k][0], pts[k][1] + mp.mpf("1e-10"), pts[k][2])
        print(f"mutated: lam_g of {pts[k][0]} + 1e-10")

    def cubic(others, D):
        near = sorted(others, key=lambda p: abs(p[2] - D))[:SCREEN_N]
        t = [1 / (p[2] - 2) for p in near]
        c = np.polyfit([float(x - 1 / (D - 2)) for x in t], [float(p[1] - near[0][1]) for p in near], SCREEN_DEG)
        return near[0][1] + mp.mpf(float(c[-1]))
    bad = 0
    for i, (rel, lg, D) in enumerate(pts):
        d = float(lg - cubic(pts[:i] + pts[i + 1:], D))
        if abs(d) > SCREEN_TOL:
            bad += 1
            print(f"  DEPARTS: backbone {rel} D {float(D):.4f}: lam_g - local polynomial = {d:+.2e}")
    on = set(bb)
    flagged = []
    for rel in sorted(states, key=lambda r: states[r][0]["D"]):
        if rel in on or held[os.path.dirname(rel)] != held[os.path.dirname(bb[0])]:
            continue
        lg, D = lam(rel)
        if not pts[0][2] < D < pts[-1][2]:
            continue
        d = float(lg - cubic(pts, D))
        if abs(d) > SCREEN_TOL:
            flagged.append((float(D), d, rel))
    for D, d, rel in flagged:
        print(f"  off the backbone (not in the table): {rel} D {D:.4f}: {d:+.2e}")
    print(f"verify: lam screen, {len(pts) - bad} of {len(pts)} backbone states within {SCREEN_TOL:.0e} of their neighbours' polynomial;"
          f" {len(flagged)} other chain_fit-held states off it")
    return bad == 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:] + "=").split("=")[:2] for a in sys.argv[1:] if a.startswith("--"))
    if not args:
        sys.exit(__doc__)
    RAW = os.path.abspath(args[0])
    if "lam_screen" in opts:
        sys.exit(0 if lam_screen(opts.get("mutate", "")) else 1)
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
