#!/usr/bin/env python3
"""No-Newton probe of a mode-count layout (#141): a state resampled to M tau points as modecount_common.setup does,
its initial residual by block and field, and the truncation of the centre and horizon series at the end nodes.

    python3 hp/modecount_probe.py STATE.npz [--M=48,64,80] [--kf=none|N] [--kc=N] [--ks=N] [--rows=6] [--expect=M:F]
        [--json=F] [--centre_tol=T]
    python3 hp/modecount_probe.py STATE.npz --M=64 --setup [--fmax=1e-15] [--nofit] [--threads=1] [--setup_json=F]
        [--data=DIR] [--centre_tol=T]
    python3 hp/modecount_probe.py --selftest [--mutate=nofit]
    python3 hp/modecount_probe.py --gatecheck --raw=RAWDIR [--mutate=nogate_after|tight] [--json=F]
    python3 hp/modecount_probe.py --tolcheck [--mutate=nocentretol] [--json=F]

For each M: the layout of the state's nodes ("state nodes") and, if modecount_common.fit_nodes adds nodes, the fitted
layout ("fitted"). Printed per layout: |F| (max norm), for each residual block (the centre series minus node 0, the
Taylor segments, the horizon series minus the last node) and field (p, m, a, c) the largest entry with its tau index
and the dominant tau-harmonic of that field's residual, the phase row, the largest rows, and per series the truncation
estimate (modecount_common.series_tail), the terms' ratio per order and the radius it implies. Nothing is solved or
saved. --kf: as setup (default 13 from D = 35 on). --kc, --ks: the series orders instead of the state's (a divergent
series' |F| grows with its order, a converged one's does not).
--expect=M:F exits 1 unless the state-nodes |F| at M is within 10% of F (the planted control: the D = 30 state of M 32
  resampled to M = 48 starts at 1.632e-11, #68).
--selftest: (1) that control; (2) D = 30 from the M 48 state to M = 64 on the state's nodes, where the centre series
  must be flagged as divergent and |F| > 1; (3) the fitted layout at M = 64, which must add a node below 0.3, have
  both truncations <= SERIES_TAIL and |F| <= 1e-9; (4) the resample of (1) keeps its nodes (the M = 48 counts' layout);
  (5) the D = 61 seed (M 32) at M = 64 with kf 23, whose horizon series (2.4e-15 at ks 127) must be fitted by a higher
  ks with the nodes kept. --mutate=nofit fits nothing, so (3) and (5) must fail and the self-test exit 1.
--setup: after the probe, modecount_common.setup at each M with --fmax (default 1e-15, the counts' value) and
  --threads, which re-converges and saves the state in hp/data as a count does (needs hp/modecount_cx); exit 1 if a
  setup raises. --nofit passes fit=False (the state's nodes and ks), where setup must refuse a divergent series.
  --setup_json=F records each setup's |F|, nodes, ks, Delta and truncations. --data=DIR: where setup keeps the
  re-converged states (modecount_common.DATA, default hp/data).
--gatecheck (#265; needs RAWDIR, a directory laid out as the raw-data repository, holding at least the files in
  GATE_RAW): (1) setup at D 45, M 48, kf 21 from the M 32 state, with the solve replaced by its result in #170 (the
  I170-macm3 state, on the same layout), must refuse the solution, whose centre series is truncated at 2.4e-8, although
  its start passes (#254: 7.4e-74); (2) the series gate passes on the 22 states #141 checked (GATE_STATES), each on its
  stored layout and cut. --mutate=nogate_after drops the gate after the solve, so (1) must fail. --mutate=tight sets the
  gate to 1e-16, so (2) must fail (D 30, M 48 without the cut has 4.8e-16). Either mutation exits 1. Nothing is solved,
  and the state setup saves goes to a temporary directory.
--centre_tol=T: the centre series' tolerance in fit_nodes, for the probe's fitted layout and setup (default
  SERIES_TAIL). A smaller T forces inner nodes on a layout that passes the gate (#270).
--tolcheck (#270): the D = 61 seed (M 32) resampled to M = 48 with kf 23, as setup prepares it. (1) With
  --centre_tol=TOLCHECK_TOL it must get one node at 0.15 and keep ks, with both series within SERIES_TAIL, and setup's
  saved name must carry _ct<T>. (2) Without it the layout must stay as it is (the centre series passes at 0.3).
  --mutate=nocentretol drops the option on its way to fit_nodes, so (1) must fail and the check exit 1. Nothing is
  solved.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import load  # noqa: E402
import modecount_common as mc  # noqa: E402

FIELDS = "pmac"
ODD = (True, True, False, False)


def harmonic(f, odd):
    """dominant tau-harmonic of a field on the half grid (odd: antiperiodic over the half period)"""
    full = np.r_[f, -f] if odd else np.r_[f, f]
    return int(np.abs(np.fft.rfft(full)).argmax())


def probe(meta, zh, zl, kf, rows=6, log=print):
    M, nodes = meta["M"], meta["nodes"]
    K = len(nodes)
    s = mc.build_solver(meta, zh, zl, M, kf)
    fh, fl = s.residual(zh, zl)
    F = np.asarray(fh) + np.asarray(fl)
    rec = dict(M=M, nodes=list(nodes), kc=meta.get("kc"), ks=meta.get("ks"), F=float(np.abs(F).max()),
               steps=[len(x) - 1 for x in s.get_segs()], blocks=[])
    log(f"    |F| {rec['F']:.3e}   Taylor steps per segment {rec['steps']}   kc {rec['kc']} ks {rec['ks']}")
    names = ["centre"] + [f"seg {nodes[b - 1]:g}-{nodes[b]:g}" for b in range(1, K)] + ["horizon"]
    for b in range(K + 1):
        blk = dict(block=names[b], fields={})
        cells = []
        for q in range(4):
            o = b * 4 * M + q * M
            v = F[o:o + M]
            j = int(np.abs(v).argmax())
            blk["fields"][FIELDS[q]] = dict(max=float(abs(v[j])), tau=j, harmonic=harmonic(v, ODD[q]))
            cells.append(f"{FIELDS[q]} {abs(v[j]):.1e}@{j:<3d} k{harmonic(v, ODD[q]):<3d}")
        rec["blocks"].append(blk)
        log(f"    {b} {names[b]:>12s}: " + "  ".join(cells))
    rec["phase"] = float(F[-1])
    log(f"    phase row {F[-1]:.1e}")
    top = np.argsort(-np.abs(F[:-1]))[:rows]
    rec["rows"] = [dict(row=int(i), block=names[i // (4 * M)], field=FIELDS[(i % (4 * M)) // M], tau=int(i % M),
                        F=float(F[i])) for i in top]
    log("    largest rows: " + ", ".join(f"{r['block']} {r['field']}[{r['tau']}] {r['F']:+.1e}" for r in rec["rows"]))
    rec["tails"] = [mc.series_tail(s, zh, zl, meta, w) for w in (0, 1)]
    for name, t in zip(("centre", "horizon"), rec["tails"]):
        var = "u" if name == "horizon" else "x"
        log(f"    {name:7s} series at {var} = {t['x']:g}: truncation {t['tail']:.1e}, last term {t['last']:.1e}, "
            f"ratio {t['ratio']:.4f} per order, radius {t['radius']:.3f}")
    return rec


def run(path, Ms, kf="default", rows=6, log=print, nmax=mc.NODES_ADDED_MAX, orders=None, centre_tol=None):
    zh0, zl0, meta0 = load(path)
    meta0 = dict(meta0, **(orders or {}))
    D = meta0["D"]
    if kf == "default":
        kf = 13 if D >= 35 else None
    out = []
    for M in Ms or [meta0["M"]]:
        if M == meta0["M"]:
            zh, zl, meta = zh0, zl0, meta0
        else:
            zh, zl, meta = mc.resample_state(zh0, zl0, meta0, M, kf)
        log(f"{os.path.basename(path)}: D {D}, M {meta0['M']} -> {M}, kf {kf}, kc {meta.get('kc')}, "
            f"ks {meta.get('ks')}")
        log(f"  state nodes {meta['nodes']}")
        rec = dict(state=os.path.basename(path), D=D, M0=meta0["M"], M=M, kf=kf, layout="state nodes")
        rec.update(probe(meta, zh, zl, kf, rows, log))
        out.append(rec)
        fh, fl, fm, _ = mc.fit_nodes(meta, zh, zl, kf, log=lambda m: log("  " + m), nmax=nmax, tol_centre=centre_tol)
        if fm["nodes"] != meta["nodes"] or fm.get("ks") != meta.get("ks"):
            log(f"  fitted nodes {fm['nodes']}, ks {fm.get('ks')}")
            rec = dict(state=os.path.basename(path), D=D, M0=meta0["M"], M=M, kf=kf, layout="fitted")
            rec.update(probe(fm, fh, fl, kf, rows, log))
            out.append(rec)
    return out


def selftest(opts):
    nmax = 0 if opts.get("mutate") == "nofit" else mc.NODES_ADDED_MAX
    s32 = os.path.join(HERE, "data", "states", "up6_D30.0688.npz")
    s48 = os.path.join(HERE, "data", "modecount_state_D30.0688_M48.npz")
    checks = []

    def check(name, ok, detail):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    r1 = run(s32, [48], None, nmax=nmax)
    a = r1[0]
    check("(1) control: D 30, M 32 -> 48, state nodes", abs(a["F"] / 1.632e-11 - 1) < 0.1,
          f"|F| {a['F']:.3e}, expect 1.632e-11")
    check("(4) the M 48 resample keeps its nodes", len(r1) == 1, f"layouts {[r['layout'] for r in r1]}")
    r2 = run(s48, [64], None, nmax=nmax)
    b = r2[0]
    check("(2) M 48 -> 64 on the state nodes: centre series divergent, |F| > 1",
          b["tails"][0]["tail"] == float("inf") and b["F"] > 1,
          f"|F| {b['F']:.1e}, centre ratio {b['tails'][0]['ratio']:.4f}")
    c = r2[-1]
    ok = (c["layout"] == "fitted" and c["nodes"][0] < 0.3 and max(t["tail"] for t in c["tails"]) <= mc.SERIES_TAIL
          and c["F"] <= 1e-9)
    check("(3) fitted layout at M 64", ok, f"layout {c['layout']}, nodes {c['nodes']}, |F| {c['F']:.1e}, truncations "
          f"{c['tails'][0]['tail']:.1e} {c['tails'][1]['tail']:.1e}")
    s61 = os.path.join(HERE, "data", "seeds", "local", "up23_D61.0264.npz")
    r3 = run(s61, [64], 23, nmax=nmax)
    e = r3[-1]
    ok = (e["layout"] == "fitted" and e["nodes"] == r3[0]["nodes"] and e["ks"] > r3[0]["ks"]
          and max(t["tail"] for t in e["tails"]) <= mc.SERIES_TAIL)
    check("(5) D 61, kf 23, M 32 -> 64: horizon fitted by ks, nodes kept", ok,
          f"layout {e['layout']}, nodes {e['nodes']}, ks {r3[0]['ks']} -> {e['ks']}, truncations "
          f"{e['tails'][0]['tail']:.1e} {e['tails'][1]['tail']:.1e}")
    if "json" in opts:
        json.dump(dict(checks=checks, records=r1 + r2 + r3), open(opts["json"], "w"), indent=1, default=str)
    bad = [x for x in checks if not x["passed"]]
    mut = f" (mutate {opts['mutate']})" if "mutate" in opts else ""
    print(f"self-test: {len(checks) - len(bad)}/{len(checks)} passed{mut}")
    return 1 if bad else 0


# #141's 22 states (I141-mac/probe/gate_states.log): the code repository's (under hp/data) and the raw-data
# repository's (under RAWDIR, I141-mac/inputs/SOURCES.txt)
GATE_STATES = ["states/up_D8.2500.npz", "states/up2_D16.7600.npz", "states/up6_D30.0688.npz", "states/up7_D45.0507.npz",
               "states/up10_D53.7914.npz", "states/up10_D54.0414.npz", "states/up10_D55.0351.npz",
               "seeds/local/up23_D61.0264.npz", "modecount_state_D8.2500_M32.npz", "modecount_state_D30.0688_M48.npz",
               "modecount_state_D30.0688_M48_kf13.npz", "modecount_state_D61.0264_M48_kf13.npz",
               "lowD_states/lowDn_D4.0000_M48.npz"]
GATE_RAW = ["I211-macm3/hp_data_after/modecount_state_D45.0507_M48_kf23.npz",
            "I232-macm3/D50/hp_data_after/modecount_state_D50.0075_M48_kf23.npz",
            "I237-macm3/D52/hp_data_after/modecount_state_D52.0414_M48_kf23.npz",
            "I237-macm3/D53/hp_data_after/modecount_state_D53.7914_M48_kf23.npz",
            "I240-macm3/hp_data_after/modecount_state_D54.0414_M48_kf23.npz",
            "I232-macm3/D55/hp_data_after/modecount_state_D55.0351_M48_kf23.npz",
            "I159-macm3/hp_data_after_kf21/modecount_state_D61.0264_M48_kf21.npz",
            "I212-macm3/hp_data_after/modecount_state_D61.0264_M48_kf23.npz",
            "I68-macm3/hp_data_after_control_fresh/modecount_state_D30.0688_M48.npz"]
# #170's D 45, M 48, kf 21 solution (byte-identical copies in I178-macm3 and I190-macm3)
GATE_D45 = "I170-macm3/hp_data_after/modecount_state_D45.0507_M48_kf21.npz"
# --tolcheck's centre_tol. The D = 61 seed resampled to M = 48 with kf 23 has centre truncation 1.4e-92 at x = 0.3 and
# 3.0e-219 at 0.15 (#270), so this forces one node at 0.15 and no more.
TOLCHECK_TOL = 1e-150


def state_kf(path, D):
    """the cut setup gives a stored state: _kfN in its name, else setup's default (13 from D = 35 on)"""
    import re
    m = re.search(r"_kf(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else (13 if D >= 35 else None)


def gatecheck(opts):
    import tempfile
    raw = opts.get("raw")
    if not isinstance(raw, str) or not os.path.isdir(raw):
        print("--gatecheck needs --raw=RAWDIR (a directory laid out as choptuik-large-d-raw)")
        return 2
    mutate = opts.get("mutate", "")
    checks, recs = [], []

    def check(name, ok, detail):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    gate, newton, lm, data, tail = mc.series_gate, mc.newton_solve, mc.lm_solve, mc.DATA, mc.SERIES_TAIL
    if mutate == "nogate_after":
        def gate_mut(s, zh, zl, meta, when, log=None):
            if when != "solution":
                return gate(s, zh, zl, meta, when, log)
            return [mc.series_tail(s, zh, zl, meta, w) for w in (0, 1)]
        mc.series_gate = gate_mut
    elif mutate == "tight":
        mc.SERIES_TAIL = 1e-16
    try:
        # (1) the start passes and the solution does not; the solve is planted with its result in #170
        zs, zls, ms = load(os.path.join(raw, GATE_D45))
        seed = os.path.join(HERE, "data", "states", "up7_D45.0507.npz")
        solves = []

        def planted(s, zh, zl, **kw):
            solves.append(len(zh))
            return zs.copy(), zls.copy(), []
        mc.newton_solve = mc.lm_solve = planted
        with tempfile.TemporaryDirectory() as tmp:
            mc.DATA = tmp
            zh0, zl0, m0, _, _ = mc.prepare(seed, 48, kf=21, log=lambda m: print(m, flush=True))
            s0 = mc.build_solver(m0, zh0, zl0, 48, 21)
            t0 = [mc.series_tail(s0, zh0, zl0, m0, w) for w in (0, 1)]
            keys = ("M", "nodes", "kt", "kc", "ks")
            same = all(m0.get(k) == ms.get(k) for k in keys) and len(zh0) == len(zs)
            print(f"  D 45 kf 21 start: {mc.tails_text(t0)}; layout of #170's solution "
                  f"{'the same' if same else 'DIFFERENT'} ({ {k: ms.get(k) for k in keys} })", flush=True)
            err = None
            try:
                st = mc.setup(seed, M=48, kf=21, fmax=1e-15, log=lambda m: print(m, flush=True))
                print(f"  setup accepted: |F| {st.F:.1e}, {mc.tails_text(st.tails)}", flush=True)
            except RuntimeError as e:
                err = str(e)
                print(f"  setup raised: {err}", flush=True)
        recs.append(dict(check=1, start_tails=t0, same_layout=same, solves=len(solves), error=err))
        check("(1) D 45, M 48, kf 21: setup refuses #170's solution, whose start passes",
              same and solves and max(t["tail"] for t in t0) <= mc.SERIES_TAIL and err is not None
              and "(solution;" in err,
              f"start {t0[0]['tail']:.1e} / {t0[1]['tail']:.1e}, planted solves {len(solves)}, "
              + (f"refused: {err[:60]}..." if err else "accepted"))
        mc.newton_solve, mc.lm_solve = newton, lm
        # (2) the series gate on #141's 22 states, as setup's gate after the solve sees a stored solution
        paths = [os.path.join(HERE, "data", p) for p in GATE_STATES] + [os.path.join(raw, p) for p in GATE_RAW]
        worst, refused = 0.0, []
        for p in paths:
            zh, zl, meta = load(p)
            kf = state_kf(p, meta["D"])
            s = mc.build_solver(meta, zh, zl, meta["M"], kf)
            try:
                t = mc.series_gate(s, zh, zl, meta, "solution")
                ok = True
            except RuntimeError:
                t = [mc.series_tail(s, zh, zl, meta, w) for w in (0, 1)]
                ok = False
                refused.append(os.path.basename(p))
            w = max(x["tail"] for x in t)
            worst = max(worst, w)
            recs.append(dict(check=2, state=p, D=meta["D"], M=meta["M"], kf=kf, tails=t, passed=ok))
            print(f"    {os.path.basename(p):42s} D {meta['D']:8.4f} M {meta['M']:3d} kf {str(kf):4s}: "
                  f"{mc.tails_text(t)}  {'ok' if ok else 'REFUSED'}", flush=True)
        check(f"(2) the gate passes #141's {len(paths)} states", len(paths) == 22 and not refused,
              f"worst {worst:.1e}" + (f", refused {refused}" if refused else ""))
    finally:
        mc.series_gate, mc.newton_solve, mc.lm_solve, mc.DATA, mc.SERIES_TAIL = gate, newton, lm, data, tail
    if "json" in opts:
        json.dump(dict(checks=checks, records=recs), open(opts["json"], "w"), indent=1, default=str)
    bad = [x for x in checks if not x["passed"]]
    print(f"gate check: {len(checks) - len(bad)}/{len(checks)} passed" + (f" (mutate {mutate})" if mutate else ""))
    return 1 if bad else 0


def tolcheck(opts):
    import tempfile
    mutate = opts.get("mutate", "")
    checks, recs = [], []

    def check(name, ok, detail):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    fit, data = mc.fit_nodes, mc.DATA
    if mutate == "nocentretol":
        mc.fit_nodes = lambda *a, tol_centre=None, **k: fit(*a, **k)
    seed = os.path.join(HERE, "data", "seeds", "local", "up23_D61.0264.npz")
    log = lambda m: print(m, flush=True)  # noqa: E731
    try:
        with tempfile.TemporaryDirectory() as tmp:
            mc.DATA = tmp
            out = {}
            for name, ctol in (("option", TOLCHECK_TOL), ("default", None)):
                zh, zl, meta, kf, saved = mc.prepare(seed, 48, kf=23, log=log, centre_tol=ctol)
                s = mc.build_solver(meta, zh, zl, 48, kf)
                t = [mc.series_tail(s, zh, zl, meta, w) for w in (0, 1)]
                out[name] = dict(nodes=meta["nodes"], ks=meta.get("ks"), saved=os.path.basename(saved), tails=t)
                print(f"  centre_tol {ctol}: nodes {meta['nodes']}, ks {meta.get('ks')}, {mc.tails_text(t)}, saved as "
                      f"{os.path.basename(saved)}", flush=True)
    finally:
        mc.fit_nodes, mc.DATA = fit, data
    o, d = out["option"], out["default"]
    recs.append(out)
    check(f"(1) D 61, M 48, kf 23 with centre_tol {TOLCHECK_TOL:.0e}: one node at 0.15, ks kept",
          o["nodes"] == [0.15] + d["nodes"] and o["ks"] == d["ks"]
          and max(x["tail"] for x in o["tails"]) <= mc.SERIES_TAIL and o["saved"].endswith(f"_ct{TOLCHECK_TOL:.0e}.npz"),
          f"nodes {o['nodes']}, ks {o['ks']}, centre {o['tails'][0]['tail']:.1e} at {o['tails'][0]['x']:g}, saved "
          f"{o['saved']}")
    check("(2) the same without it: the layout kept", d["nodes"] == [0.3, 0.4, 0.5, 0.6, 0.7] and d["ks"] == 127,
          f"nodes {d['nodes']}, ks {d['ks']}, centre {d['tails'][0]['tail']:.1e} at {d['tails'][0]['x']:g}")
    if "json" in opts:
        json.dump(dict(checks=checks, records=recs), open(opts["json"], "w"), indent=1, default=str)
    bad = [x for x in checks if not x["passed"]]
    mut = f" (mutate {mutate})" if mutate else ""
    print(f"centre_tol check: {len(checks) - len(bad)}/{len(checks)} passed{mut}")
    return 1 if bad else 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:].partition("=")[0], a[2:].partition("=")[2] or True) for a in sys.argv[1:] if a.startswith("--"))
    if "selftest" in opts:
        return selftest(opts)
    if "gatecheck" in opts:
        return gatecheck(opts)
    if "tolcheck" in opts:
        return tolcheck(opts)
    if "data" in opts:
        mc.DATA = opts["data"]
    kf = "default" if "kf" not in opts else (None if opts["kf"] == "none" else int(opts["kf"]))
    Ms = [int(v) for v in opts["M"].split(",")] if "M" in opts else None
    orders = {k: int(opts[k]) for k in ("kc", "ks") if k in opts}
    ctol = float(opts["centre_tol"]) if "centre_tol" in opts else None
    out = run(args[0], Ms, kf, int(opts.get("rows", 6)), orders=orders, centre_tol=ctol)
    if "json" in opts:
        json.dump(out, open(opts["json"], "w"), indent=1, default=str)
    if "expect" in opts:
        Me, Fe = opts["expect"].split(":")
        r = next(r for r in out if r["M"] == int(Me) and r["layout"] == "state nodes")
        ok = abs(r["F"] / float(Fe) - 1) < 0.1
        print(f"expect |F| {float(Fe):.3e} at M {Me}: {r['F']:.3e} -> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    if "setup" in opts:
        return run_setup(args[0], Ms, kf, float(opts.get("fmax", 1e-15)), "nofit" not in opts,
                         int(opts.get("threads", 1)), opts.get("setup_json"), ctol)
    return 0


def run_setup(path, Ms, kf, fmax, fit, threads, out=None, centre_tol=None):
    """modecount_common.setup at each M, as a count runs it (Newton, then Levenberg-Marquardt; the re-converged state
    is saved in hp/data); exit 1 if any setup raises"""
    import time
    recs, bad = [], 0
    for M in Ms or [None]:
        t0 = time.time()
        print(f"setup {os.path.basename(path)} M {M} kf {kf} fmax {fmax:.0e} fit {fit} centre_tol {centre_tol}:",
              flush=True)
        try:
            st = mc.setup(path, M=M, threads=threads, kf=kf, fmax=fmax, log=lambda m: print(m, flush=True), fit=fit,
                          centre_tol=centre_tol)
        except RuntimeError as e:
            print(f"  setup raised: {e}", flush=True)
            recs.append(dict(M=M, kf=kf, fit=fit, error=str(e), seconds=round(time.time() - t0)))
            bad += 1
            continue
        rec = dict(M=st.M, kf=st.kf, fit=fit, nodes=st.meta["nodes"], ks=st.ks, F_in=st.F_in, F=st.F,
                   Delta=float(st.zh[-1]), Delta_lo=float(st.zl[-1]), tails=st.tails, seconds=round(time.time() - t0))
        recs.append(rec)
        print(f"  setup: |F| {st.F_in:.1e} -> {st.F:.1e}, nodes {rec['nodes']}, ks {st.ks}, Delta {rec['Delta']!r} "
              f"{rec['Delta_lo']:+.3e}, {rec['seconds']} s", flush=True)
    if out:
        json.dump(recs, open(out, "w"), indent=1, default=str)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
