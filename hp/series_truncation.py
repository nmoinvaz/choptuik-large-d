#!/usr/bin/env python3
"""Truncation of the centre and horizon series on the states of the high-M paths (#254): gamma_dd.py --M48 / --Mhi,
and the M >= 48 runs on the large-D layout (first node 0.3). The measure is modecount_common.series_tail, the gate
SERIES_TAIL (1e-15, relative to the series' largest term). The paths other than the mode count have no such gate.

    python3 hp/series_truncation.py OUT.jsonl STATE.npz [...] [--kf=path|none|N] [--polish=N] [--Mhi=N]
        [--resample_only] [--fit] [--lam] [--kc=N] [--ks=N] [--fmax=1e-20] [--threads=1] [--save=DIR] [--tag=T]
        [--allow_unraw]
    python3 hp/series_truncation.py --selftest --raw=RAWDIR [--mutate=half]

For each state: the layout as saved (nodes, kt, kc, ks) with the cut kf, its |F|, Delta, and per series the truncation
estimate, the ratio of terms per order and the radius it implies (series_tail), and the gate verdict. A state of four
parts (z2, z3; the quad- and octo-double chains) is read as double-double, its upper parts' sum.
--kf: path (default) takes the cut from the file name (_kfN), a mode-count state without one has none, a state that
  stores kfreeze has that, and any other is cut as gamma_dd.py cuts it (13 from D = 35 on).
--kc, --ks: the series orders instead of the state's (a run that changed them, e.g. ks 200 in large_d/m48_diag).
--polish=N: up to N Newton steps first (gamma_batch.py passes --polish=4 to gamma_dd.py).
--Mhi=N: the state resampled to N tau points as gamma_dd.py --Mhi does (gamma_dd.resample_hi: double-double, p1
  projected onto the cut) and re-converged as it does (gamma_dd.solve_hi: Newton to the floor, Levenberg-Marquardt when
  that stops above --fmax), then the same measures, Delta's change and the Newton step's Delta component at the
  solution (est). --mutate=old_rule restores gamma_dd's path before #271 (f64 resampling, Newton to max(10 |F|,
  1e-27)). --resample_only measures the resampled state without solving (for paths whose
  solve did not converge). --save=DIR keeps the re-converged state there.
--fit: where a truncation exceeds the gate, modecount_common.fit_nodes refits the layout (a node at half the first
  node, or ks x 1.5), the state is re-converged there, and Delta's change is reported: a measured estimate of how far
  the truncation moved Delta. --lam: lam = 1/gamma located on each layout as gamma_dd.py does (find_lam).
--selftest (needs the raw-data checkout, --raw): (1) #68's D 30 M 64 state on the first-node-0.3 layout
  (I68-macm3/hp_data_after) must be flagged, its centre truncation inf or > 1; (2) #141's fitted M 64 state
  (I141-mac/states, node 0.15) must pass both series; (3) the October D 30 M 48 state (hp/data) must reproduce #141's
  centre truncation 4.8e-16 (I141-mac/probe/gate_states.log) to 20%. --mutate=half evaluates both series at half
  their end node's distance, so (1) and (3) must fail and the self-test exit 1.
"""
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import load, save, newton_solve  # noqa: E402
import modecount_common as mc  # noqa: E402
from gamma_dd import resample_hi, solve_hi  # noqa: E402

GATE = mc.SERIES_TAIL


def load_dd(path):
    """(zh, zl, meta, parts): a four-part state is summed into double-double (the parts do not overlap)"""
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    zh, zl = np.asarray(d["zh"], float), np.asarray(d["zl"], float)
    parts = 2
    if "z2" in d.files:
        zl = zl + np.asarray(d["z2"], float) + (np.asarray(d["z3"], float) if "z3" in d.files else 0.0)
        parts = 4
    return np.ascontiguousarray(zh), np.ascontiguousarray(zl), meta, parts


def path_kf(path, meta, opt="path"):
    """the cut a path used for this state (see --kf)"""
    if opt == "none":
        return None
    if opt != "path":
        return int(opt)
    name = os.path.basename(path)
    m = re.search(r"_kf(\d+)", name)
    if m:
        return int(m.group(1))
    if "kfreeze" in meta:
        return meta["kfreeze"]
    if "modecount_state" in name or name.startswith("D") and "_M" in name:
        return None
    return 13 if meta["D"] >= 35 else None


def measure(meta, zh, zl, kf, threads=1, mutate=""):
    """solver on the layout, |F|, and both series' truncation (mutate=half: at half the end nodes' distance)"""
    s = mc.build_solver(meta, zh, zl, meta["M"], kf, threads)
    F = mc.resid(s, zh, zl)
    m = meta
    if mutate == "half":
        nodes = list(meta["nodes"])
        nodes[0] = nodes[0] / 2
        nodes[-1] = 1 - (1 - nodes[-1]) / 2
        m = dict(meta, nodes=nodes)
    tails = [mc.series_tail(s, zh, zl, m, w) for w in (0, 1)]
    return s, F, tails


def converge(s, zh, zl, F_ref, fmax, log, mutate=""):
    """gamma_dd.py's M48 solve (solve_hi): Newton to the floor, then Levenberg-Marquardt if above fmax. Returns zh, zl,
    a summary and solve_hi's record"""
    zh, zl, sol = solve_hi(s, zh, zl, F_ref, fmax, verbose=False, mutate=mutate)
    how = (f"Newton {sol['newton']} steps" + (f", LM {sol['lm']} steps" if "lm" in sol else "")
           + f", stop {sol['stop']}; Newton step's Delta {sol['est']:+.1e}")
    return zh, zl, how, sol


def tails_rec(tails):
    return [dict(tail=t["tail"], ratio=t["ratio"], last=t["last"], x=t["x"], radius=t["radius"]) for t in tails]


def verdict(tails):
    w = max(t["tail"] for t in tails)
    return "ok" if w <= GATE else "ABOVE"


def line(label, meta, kf, F, tails, extra=""):
    t0, t1 = tails
    return (f"  {label:9s} M {meta['M']:3d} kf {str(kf):4s} nodes {meta['nodes'][0]:g}..{meta['nodes'][-1]:g} "
            f"kc {meta.get('kc')} ks {meta.get('ks')}: |F| {F:.1e}; centre {t0['tail']:.1e} (ratio {t0['ratio']:.3f}, "
            f"radius {t0['radius']:.3f} at x {t0['x']:g}); horizon {t1['tail']:.1e} (ratio {t1['ratio']:.3f}, radius "
            f"{t1['radius']:.3f} at u {t1['x']:g})  {verdict(tails)}{extra}")


def lam_of(s, zh, zl, D):
    from gamma_dd import find_lam, gamma_formula
    lam, nch, nev = find_lam(s, zh, zl, 1.0 / gamma_formula(D))
    return lam


def fit_and_solve(meta, zh, zl, kf, F_ref, fmax, threads, log, want_lam):
    """modecount_common.fit_nodes on the state, re-converged there; returns a record with Delta and the tails"""
    t = time.time()
    fh, fl, fm, _ = mc.fit_nodes(meta, zh, zl, kf, threads, log=lambda m: log("    fit:" + m))
    s, F0, _ = measure(fm, fh, fl, kf, threads)
    fh, fl, how, _ = converge(s, fh, fl, F_ref, fmax, log)
    s, F, tails = measure(fm, fh, fl, kf, threads)
    rec = dict(nodes=fm["nodes"], ks=fm.get("ks"), F_start=F0, F=F, solve=how, tails=tails_rec(tails),
               Delta=float(fh[-1]), Delta_lo=float(fl[-1]), seconds=round(time.time() - t, 1))
    if want_lam and F <= fmax:
        rec["lam"] = lam_of(s, fh, fl, meta["D"])
    log(line("fitted", fm, kf, F, tails, f"  ({how}, {rec['seconds']} s)"))
    return rec, fh, fl, fm


def run_state(path, opts, log=print):
    t_start = time.time()
    zh, zl, meta, parts = load_dd(path)
    meta = dict(meta, **{k: int(opts[k]) for k in ("kc", "ks") if k in opts})
    D = meta["D"]
    kf = path_kf(path, meta, opts.get("kf", "path"))
    threads = int(opts.get("threads", 1))
    fmax = float(opts.get("fmax", 1e-20))
    mutate = opts.get("mutate", "")
    want_lam = "lam" in opts
    log(f"{path}: D {D}, M {meta['M']}, {parts} parts, kf {kf}")
    rec = dict(state=path, tag=opts.get("tag", ""), D=D, M=meta["M"], kf=kf, parts=parts, nodes=meta["nodes"],
               kc=meta.get("kc"), ks=meta.get("ks"), F_saved=meta.get("F"))
    s, F, tails = measure(meta, zh, zl, kf, threads, mutate)
    if "polish" in opts:
        rec["F_unpolished"] = F
        rec["tails_unpolished"] = tails_rec(tails)
        zh, zl, hist = newton_solve(s, zh, zl, tol=1e-29, it_max=int(opts["polish"]), verbose=False)
        zh, zl = np.ascontiguousarray(zh), np.ascontiguousarray(zl)
        s, F, tails = measure(meta, zh, zl, kf, threads, mutate)
    rec.update(F=F, tails=tails_rec(tails), verdict=verdict(tails), Delta=float(zh[-1]), Delta_lo=float(zl[-1]))
    log(line("own M", meta, kf, F, tails))
    if want_lam and F <= fmax:
        rec["lam"] = lam_of(s, zh, zl, D)
        log(f"    lam {rec['lam']!r}")
    if "fit" in opts and rec["verdict"] != "ok":
        rec["fitted"], *_ = fit_and_solve(meta, zh, zl, kf, F, fmax, threads, log, want_lam)
        rec["fitted"]["Delta_diff"] = (rec["fitted"]["Delta"] - rec["Delta"]) + (rec["fitted"]["Delta_lo"]
                                                                                 - rec["Delta_lo"])
        log(f"    fitted minus own layout: Delta {rec['fitted']['Delta_diff']:+.3e}")
    if "Mhi" in opts:
        Mhi = int(opts["Mhi"])
        t = time.time()
        K = len(meta["nodes"])
        hh, hl = resample_hi(zh, zl, meta["M"], Mhi, K, kf, mutate)
        mh = dict(meta, M=Mhi)
        s2, F0, tails0 = measure(mh, hh, hl, kf, threads, mutate)
        hi = dict(M=Mhi, F_start=F0, tails_start=tails_rec(tails0))
        log(line("resampled", mh, kf, F0, tails0))
        if "resample_only" not in opts:
            hh, hl, how, sol = converge(s2, hh, hl, F, fmax, log, mutate)
            F2 = mc.resid(s2, hh, hl)
            _, _, tails2 = measure(mh, hh, hl, kf, threads, mutate)
            hi.update(F=F2, solve=how, est=sol["est"], floor=sol["floor"],
                      resample="f64" if mutate == "old_rule" else "dd", tails=tails_rec(tails2), verdict=verdict(tails2), Delta=float(hh[-1]),
                      Delta_lo=float(hl[-1]), Delta_diff=float((hh[-1] - zh[-1]) + (hl[-1] - zl[-1])),
                      seconds=round(time.time() - t, 1))
            log(line(f"M {Mhi}", mh, kf, F2, tails2, f"  ({how}; Delta minus own M {hi['Delta_diff']:+.3e}, "
                                                    f"{hi['seconds']} s)"))
            if want_lam and F2 <= fmax:
                hi["lam"] = lam_of(s2, hh, hl, D)
                log(f"    lam at M {Mhi} {hi['lam']!r}" + (f", minus own M {hi['lam'] - rec['lam']:+.3e}"
                                                         if rec.get("lam") else ""))
            if "save" in opts:
                os.makedirs(opts["save"], exist_ok=True)
                out = os.path.join(opts["save"], os.path.basename(path)[:-4] + f"_M{Mhi}" +
                                   (f"_kf{kf}" if kf else "") + ".npz")
                save(out, hh, hl, dict(mh, F=F2, source=os.path.basename(path)))
                hi["saved"] = out
            if "fit" in opts and hi["verdict"] != "ok":
                hi["fitted"], *_ = fit_and_solve(mh, hh, hl, kf, F, fmax, threads, log, want_lam)
                hi["fitted"]["Delta_diff"] = ((hi["fitted"]["Delta"] - hi["Delta"])
                                              + (hi["fitted"]["Delta_lo"] - hi["Delta_lo"]))
                log(f"    fitted minus M {Mhi} layout: Delta {hi['fitted']['Delta_diff']:+.3e}")
        rec["Mhi"] = hi
    rec["seconds"] = round(time.time() - t_start, 1)
    return rec


def selftest(opts):
    raw = opts.get("raw")
    if not raw or not os.path.isdir(raw):
        print("--selftest needs --raw=RAWDIR (a checkout of choptuik-large-d-raw)")
        return 2
    mutate = opts.get("mutate", "")
    o = dict(mutate=mutate) if mutate else {}
    checks = []

    def check(name, ok, detail):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    r = run_state(os.path.join(raw, "I68-macm3", "hp_data_after", "modecount_state_D30.0688_M64.npz"), o)
    c = r["tails"][0]
    check("(1) #68's D 30 M 64 state, first node 0.3: centre flagged (inf or > 1)",
          c["tail"] == float("inf") or c["tail"] > 1, f"centre {c['tail']:.1e}, ratio {c['ratio']:.3f}")
    r = run_state(os.path.join(raw, "I141-mac", "states", "modecount_state_D30.0688_M64.npz"), o)
    check("(2) #141's fitted D 30 M 64 state (node 0.15): both series pass", r["verdict"] == "ok",
          f"centre {r['tails'][0]['tail']:.1e}, horizon {r['tails'][1]['tail']:.1e}")
    r = run_state(os.path.join(HERE, "data", "modecount_state_D30.0688_M48.npz"), o)
    c = r["tails"][0]
    check("(3) October's D 30 M 48 state: centre 4.8e-16 as in #141 (20%)", abs(c["tail"] / 4.8e-16 - 1) < 0.2,
          f"centre {c['tail']:.2e}, ratio {c['ratio']:.3f}")
    bad = [x for x in checks if not x["passed"]]
    print(f"self-test: {len(checks) - len(bad)}/{len(checks)} passed" + (f" (mutate {mutate})" if mutate else ""))
    return 1 if bad else 0


def main():
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("--")]
    opts = dict((a[2:].partition("=")[0], a[2:].partition("=")[2] or True) for a in argv if a.startswith("--"))
    if "selftest" in opts:
        return selftest(opts)
    out = args[0]
    if "allow_unraw" not in opts:
        from raw_out import require_raw
        require_raw(out, allow=False)
    for p in args[1:]:
        try:
            rec = run_state(p, opts, log=lambda m: print(m, flush=True))
        except Exception as e:  # a state the path could not measure is recorded, not skipped silently
            rec = dict(state=p, tag=opts.get("tag", ""), error=f"{type(e).__name__}: {e}")
            print(f"  error: {rec['error']}", flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
