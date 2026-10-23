#!/usr/bin/env python3
"""Benchmark of the Newton variants of the double-double solver over the whole range of D (notes/LARGE_D_MIXED.md
section 5). Runs on Linux and Windows (no shell; paths by os.path), single process, the kernel's thread pool set by
--threads.

    cd hp
    python bench_variants.py [--states=3.7,8.25,16.76,20,30,40,45,50,55,60] [--variants=dd,f64,f64h,f64g,f64hg]
                             [--repeat=1] [--threads=4] [--out=data/bench_variants/bench_<host>.jsonl] [--no-nice]
    python bench_variants.py --report [--out=...]          (table from the records)
    python bench_variants.py --calibrate [--states=...]    (the perturbation search, see below)
    python bench_variants.py --prepare                      (the two prepared states, below; done once, committed)

What one run is. A state (below) is perturbed toward its neighbouring saved solution: z + eps (z_nb - z), z_nb the
saved state with the same nodes nearest above in D (below for the last one), resampled to the state's M and frozen
like it, eps from data/bench_variants/eps.json per state. This is a smooth perturbation of the size and kind of a
continuation predictor's error. A random one (--perturb=random: every entry times 1 + eps xi, xi standard normal) is
no use at large D: the horizon block amplifies it (at D = 60, eps = 1e-14 gives |F| = 2.4e-5 and a first Newton step
|dz| = 0.24), and the double-double deflated Newton did not converge from eps = 1e-8 ... 1e-14 there
(notes/LARGE_D_MIXED.md section 5). From that start the solve runs
with one Jacobian variant (driver.newton_step: dd exact; f64 the f64 Jacobian; f64h with the horizon block in double-
double; f64g / f64hg with the GMRES correction loop of exact products) to |F| < tol: plain Newton (driver.newton_solve)
below D = 25, deflated Newton with w = 24 (driver.deflated_newton, as hp/scan_hp.py runs it) from D = 25, which is
from D = 35 with the frozen formulation (kfreeze 13), as the scan ran. The Taylor plan is the reference state's (plan(z, tol_step,
0.05)), the same for every variant. tol is 10 x the reference state's |F|, at least 1e-28.

The states (data/bench_variants/states/ when present, else the saved scan states):
    3.7    D = 3.7, M = 80 (lowDv_D3.7000_M80 converged to 1e-28 by the double-double Newton, as bench_D3.7000_M80.npz)
    8.25 .. 60   the scan states hp/data/states/up*_D<D>.npz nearest to D (M = 32; 48 at 8.25), kfreeze 13 from 35
    30     the D = 30.07 scan state is a junk-accepted point (|F| 3.3e-12, run up6); the bench uses it frozen at
           kfreeze 13 and converged by the double-double Newton (bench_D30.0688_kf13.npz)

Each (state, variant, repeat) writes one JSON line to --out as soon as it is done; a rerun skips the lines present, so
an interrupted run resumes. At each state the variants run in turn, repeat after repeat, so that the machine's load
hits them alike. The load (os.getloadavg, else psutil's, else none) is recorded before and after each run.

Recorded per run: converged, iterations (Newton steps, the last confirming |F| < tol), whole-solve seconds, the steps'
seconds by part summed (residual, Jacobian, LU, solves, products) and per step, the products, final |F|, Delta (hi,
lo) and its difference from the dd run of the same state and repeat.

--calibrate: for each state, eps = 1e-1, 1e-2, ... down to 1e-13 (1e-8 ... 1e-14 with --perturb=random): the
double-double variant from that start, at most 10 Newton steps (a start with |F| above 1e-4 is not tried: the dd
Newton failed from every such start at D = 55 and 60); the first eps (from the top) that converges, divided by 10, is
written to eps.json (a margin of one decade). Writes data/bench_variants/calibrate.jsonl with every try.
"""
import glob
import json
import os
import platform
import re
import socket
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpkernel  # noqa: E402
from hpkernel.driver import JACS, dd_norm, deflated_newton, load, make_solver, newton_solve, plow_odd, resample_z, save  # noqa: E402

BENCH = os.path.join(HERE, "data", "bench_variants")
DEFAULT_STATES = "3.7,8.25,16.76,20,30,40,45,50,55,60"
DEFLATED_FROM = 25.0     # deflated Newton (w = 24) from here, plain Newton below
CAL_ITS = 10             # --calibrate: a start that the double-double Newton does not bring below tol in 10 steps is too far
CAL_FMAX = 1e-4          # --calibrate: starts with |F| above this are not tried (the dd Newton failed from all such)
FREEZE_FROM = 35.0       # kfreeze 13 from here, as the scan ran (a prepared state's meta 'kfreeze' takes precedence)


def loadavg():
    try:
        if hasattr(os, "getloadavg"):
            return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        pass
    try:
        import psutil  # noqa: PLC0415
        return [round(x, 2) for x in psutil.getloadavg()] + [psutil.cpu_percent(interval=None)]
    except Exception:  # noqa: BLE001
        return None


def lower_priority():
    try:
        if hasattr(os, "nice"):
            os.nice(10)
            return "nice 10"
        import psutil  # noqa: PLC0415
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return "below normal"
    except Exception as e:  # noqa: BLE001
        return f"unchanged ({e})"


def state_path(spec):
    """(path, label, kfreeze, tol_step) for a state spec (a D value): a prepared state in data/bench_variants/states
    when there is one for that D, else the saved state (D = 3.7: lowDv_D3.7000_M80; else the scan state nearest to D)"""
    D = float(spec)
    if abs(D - 3.7) < 0.05:
        p = os.path.join(HERE, "data", "lowD_states", "lowDv_D3.7000_M80.npz")
    else:
        avail = {float(re.search(r"_D([0-9.]+)\.npz$", q).group(1)): q for q in glob.glob(os.path.join(HERE, "data", "states", "up*_D*.npz"))}
        p = avail[min(avail, key=lambda d: abs(d - D))]
    _, _, m = load(p)
    prep = glob.glob(os.path.join(BENCH, "states", f"bench_D{m['D']:.4f}_*.npz"))
    if prep:
        p = prep[0]
        _, _, m = load(p)
    kf = m.get("kfreeze", 13 if m["D"] >= FREEZE_FROM else None)
    tol_step = 1e-30 if m["D"] < 5 else 1e-32
    return p, f"D{m['D']:.4f}_M{m['M']}", kf, tol_step


def freeze(z, M, kf):
    """p1's harmonics above kf set to zero (scan_hp.py freeze_p1), on an f64 vector"""
    z = np.asarray(z, dtype=float).copy()
    if kf:
        lo, lo2 = plow_odd(M, kf)
        z[:M] = (lo + lo2) @ z[:M]
    return z


def neighbour(m, kf):
    """the secant direction z_nb - z of a state: z_nb the saved state with the same nodes nearest above in D (else
    nearest below), resampled to M if its M differs, frozen like the state; returns (direction (f64), its path)"""
    cands = []
    for q in glob.glob(os.path.join(HERE, "data", "states", "up*_D*.npz")) + glob.glob(os.path.join(HERE, "data", "lowD_states", "lowD*_D*.npz")):
        _, _, mq = load(q)
        if list(mq["nodes"]) == list(m["nodes"]) and abs(mq["D"] - m["D"]) > 1e-9:
            cands.append((mq["D"], q, mq))
    above = sorted([c for c in cands if c[0] > m["D"]], key=lambda c: c[0] - m["D"])
    below = sorted([c for c in cands if c[0] < m["D"]], key=lambda c: m["D"] - c[0])
    D_nb, q, mq = (above or below)[0]
    hh, hl, _ = load(q)
    z = np.asarray(hh) + np.asarray(hl)
    if mq["M"] != m["M"]:
        z, _ = resample_z(z, mq["M"], m["M"], len(m["nodes"]))
    return freeze(z, m["M"], kf), q


def setup(spec, threads):
    p, label, kf, tol_step = state_path(spec)
    zh, zl, m = load(p)
    s = make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads, kfreeze=kf)
    segs = s.plan(zh, zl, tol_step, 0.05)
    s.set_segs(segs)
    F0 = dd_norm(*s.residual(zh, zl))
    znb, pnb = neighbour(m, kf)
    sec = znb - (np.asarray(zh) + np.asarray(zl))
    return dict(path=p, label=label, D=m["D"], M=m["M"], meta=m, kf=kf, segs=segs, zh=np.asarray(zh), zl=np.asarray(zl),
                F0=F0, tol=max(1e-28, 10 * F0), secant=sec, neighbour=os.path.basename(pnb))


def start_of(st, eps, kind="secant", seed=1):
    """the perturbed start: secant (default): z + eps (z_nb - z), toward the neighbouring saved solution (a smooth
    perturbation, like a continuation predictor's error); random: z (1 + eps xi), xi standard normal"""
    if kind == "random":
        rng = np.random.default_rng(seed)
        return st["zh"] * (1 + eps * rng.standard_normal(len(st["zh"]))), st["zl"].copy()
    return st["zh"] + eps * st["secant"], st["zl"].copy()


def solve(st, jac, eps, threads, seed=1, it_max=None, kind="secant"):
    m = st["meta"]
    s = make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads, kfreeze=st["kf"])
    s.set_segs(st["segs"])
    zh, zl = start_of(st, eps, kind, seed)
    F_start = dd_norm(*s.residual(zh, zl))
    deflated = m["D"] >= DEFLATED_FROM
    t = time.time()
    try:
        if deflated:
            a, b, hist = deflated_newton(s, zh, zl, tol=st["tol"], it_max=it_max or 25, w=24, verbose=False, jac=jac)
        else:
            a, b, hist = newton_solve(s, zh, zl, tol=st["tol"], it_max=it_max or 12, verbose=False, jac=jac)
        err = None
    except Exception as e:  # noqa: BLE001
        a, b, hist, err = zh, zl, [], f"{type(e).__name__}: {e}"[:300]
    wall = time.time() - t
    Fend = dd_norm(*s.residual(a, b)) if err is None else float("nan")
    conv = err is None and Fend < st["tol"]
    parts = {k: sum(r.get(k, 0.0) for r in hist if r.get(k) == r.get(k)) for k in ("t_res", "t_jac", "t_lu", "t_solve", "t_prod")}
    steps = sum(1 for r in hist if "t_jac" in r)
    return dict(converged=bool(conv), its=len(hist), steps=steps, wall=round(wall, 3), F_start=F_start, F=Fend, Delta=float(a[-1]),
                Delta_lo=float(b[-1]), parts={k: round(v, 4) for k, v in parts.items()},
                per_step={k: round(v / max(steps, 1), 4) for k, v in parts.items()},
                products=int(sum(r.get("products", 0) for r in hist)), retry_dd=sum(1 for r in hist if r.get("retry_dd")),
                solver="deflated" if deflated else "newton", error=err,
                dDelta_est=next((r.get("dDelta_est") for r in reversed(hist) if r.get("dDelta_est") is not None), None))


def read(out):
    if not os.path.exists(out):
        return []
    return [json.loads(x) for x in open(out) if x.strip()]


def run(opts):
    threads = int(opts.get("threads", 2))
    specs = str(opts.get("states", DEFAULT_STATES)).split(",")
    variants = str(opts.get("variants", ",".join(JACS))).split(",")
    repeat = int(opts.get("repeat", 1))
    out = opts["out"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    eps_tab = json.load(open(os.path.join(BENCH, "eps.json"))) if os.path.exists(os.path.join(BENCH, "eps.json")) else {}
    prio = lower_priority() if not opts.get("no-nice") else "unchanged"
    machine = dict(host=socket.gethostname(), platform=platform.platform(), processor=platform.processor(),
                   cpus=os.cpu_count(), python=platform.python_version(), isa=hpkernel.isa(), threads=threads, priority=prio)
    print(f"bench_variants: {machine}", flush=True)
    done = {(r["state"], r["variant"], r["repeat"]): r for r in read(out)}
    t_all = time.time()
    for spec in specs:
        st = None
        for rep in range(repeat):
            for v in variants:
                if v not in JACS:
                    raise SystemExit(f"unknown variant {v}; use {JACS}")
                p, label, _, _ = state_path(spec)
                if (label, v, rep) in done:
                    continue
                if st is None:
                    st = setup(spec, threads)
                    print(f"{label}: reference |F| {st['F0']:.1e}, tol {st['tol']:.0e}, kfreeze {st['kf']}, "
                          f"{sum(len(g) - 1 for g in st['segs'])} Taylor steps ({os.path.basename(st['path'])})", flush=True)
                eps = float(opts.get("eps", eps_tab.get(label, {}).get("eps", 1e-4)))
                kind = opts.get("perturb", eps_tab.get(label, {}).get("perturb", "secant"))
                l0 = loadavg()
                r = solve(st, v, eps, threads, kind=kind)
                rec = dict(state=label, D=st["D"], M=st["M"], variant=v, repeat=rep, eps=eps, perturb=kind,
                           neighbour=st["neighbour"], tol=st["tol"], load_before=l0,
                           load_after=loadavg(), machine=machine, time=time.strftime("%Y-%m-%d %H:%M:%S"), **r)
                ref = done.get((label, "dd", rep))
                if ref is not None and ref.get("converged") and r["converged"]:
                    rec["dDelta_vs_dd"] = float((r["Delta"] - ref["Delta"]) + (r["Delta_lo"] - ref["Delta_lo"]))
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                done[(label, v, rep)] = rec
                ps = r["per_step"]
                print(f"  {label} {v:6s} rep {rep}: {'converged' if r['converged'] else 'NOT converged'} in {r['its']} its, "
                      f"{r['wall']:.1f} s; per step: Jacobian {ps['t_jac']:.2f} s, LU {ps['t_lu']:.2f} s, products "
                      f"{ps['t_prod']:.2f} s ({r['products']} in all); |F| {r['F']:.1e}"
                      + (f"; Delta - dd {rec['dDelta_vs_dd']:+.1e}" if "dDelta_vs_dd" in rec else "")
                      + (f"; load {l0}" if l0 else "") + (f"; {r['error']}" if r["error"] else ""), flush=True)
    print(f"total {time.time() - t_all:.0f} s", flush=True)
    report(opts)


def report(opts):
    recs = read(opts["out"])
    if not recs:
        print("no records")
        return
    states = []
    for r in recs:
        if r["state"] not in states:
            states.append(r["state"])
    variants = [v for v in JACS if any(r["variant"] == v for r in recs)]
    print(f"\nmedian whole-solve seconds (speedup over dd) / iterations / per-step Jacobian+LU+products seconds, from {opts['out']}")
    print("state".ljust(16) + "".join(v.rjust(26) for v in variants))
    for stt in states:
        line = stt.ljust(16)
        rd = [r for r in recs if r["state"] == stt and r["variant"] == "dd" and r["converged"]]
        tdd = float(np.median([r["wall"] for r in rd])) if rd else None
        for v in variants:
            rr = [r for r in recs if r["state"] == stt and r["variant"] == v]
            if not rr:
                line += "-".rjust(26)
                continue
            ok = [r for r in rr if r["converged"]]
            if not ok:
                line += f"FAILED ({len(rr)})".rjust(26)
                continue
            w = float(np.median([r["wall"] for r in ok]))
            its = int(np.median([r["its"] for r in ok]))
            sp = f" ({tdd / w:.1f}x)" if tdd else ""
            line += f"{w:7.1f}s{sp} {its:2d}it".rjust(26)
        print(line)
    dd = [abs(r["dDelta_vs_dd"]) for r in recs if r.get("dDelta_vs_dd") is not None]
    if dd:
        print(f"max |Delta - Delta_dd| over all converged runs: {max(dd):.1e}")


def calibrate(opts):
    threads = int(opts.get("threads", 2))
    specs = str(opts.get("states", DEFAULT_STATES)).split(",")
    path = os.path.join(BENCH, "eps.json")
    tab = json.load(open(path)) if os.path.exists(path) else {}
    log = os.path.join(BENCH, "calibrate.jsonl")
    tried = {(r["state"], r["eps"]): r for r in read(log)}
    for spec in specs:
        st = setup(spec, threads)
        label = st["label"]
        if label in tab:
            print(f"{label}: eps {tab[label]['eps']:.0e} (already calibrated)")
            continue
        found = None
        kind = opts.get("perturb", "secant")
        for k in (range(8, 15) if kind == "random" else range(1, 14)):
            eps = 10.0 ** -k
            r = tried.get((label, eps))
            if r is None:
                m = st["meta"]
                s0 = make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads, kfreeze=st["kf"])
                s0.set_segs(st["segs"])
                F_start = dd_norm(*s0.residual(*start_of(st, eps, kind)))
                if F_start > CAL_FMAX:
                    r = dict(state=label, eps=eps, tol=st["tol"], perturb=kind, neighbour=st["neighbour"], converged=False,
                             its=0, wall=0.0, F_start=F_start, F=F_start, skipped=f"start |F| above {CAL_FMAX:g}")
                else:
                    r = dict(state=label, eps=eps, tol=st["tol"], perturb=kind, neighbour=st["neighbour"],
                             **solve(st, "dd", eps, threads, it_max=CAL_ITS, kind=kind))
                with open(log, "a") as f:
                    f.write(json.dumps(r) + "\n")
            print(f"{label}: eps {eps:.0e} ({kind}, toward {st['neighbour']}): start |F| {r.get('F_start', float('nan')):.1e}, "
                  f"{'converged' if r['converged'] else 'not converged'} in {r['its']} its, {r['wall']:.0f} s, |F| {r['F']:.1e}",
                  flush=True)
            if r["converged"]:
                found = eps
                break
        if found is not None:
            tab[label] = dict(eps=found / 10, largest_converging=found, perturb=kind, neighbour=st["neighbour"],
                              solver="deflated" if st["D"] >= DEFLATED_FROM else "newton")
            json.dump(tab, open(path, "w"), indent=1)


def prepare(opts):
    """the prepared states (data/bench_variants/states): D = 3.7, M = 80 converged by the double-double Newton from
    lowDv_D3.7000_M80 (|F| 7.5e-6 on its own plan) to 1e-28; D = 30.07 frozen at kfreeze 13 (p1's harmonics above 13
    set to zero, as scan_hp.py freeze_p1) and converged by the double-double deflated Newton to 1e-27"""
    threads = int(opts.get("threads", 2))
    os.makedirs(os.path.join(BENCH, "states"), exist_ok=True)
    jobs = [(os.path.join(HERE, "data", "lowD_states", "lowDv_D3.7000_M80.npz"), None, 1e-30, 1e-28),
            (os.path.join(HERE, "data", "states", "up6_D30.0688.npz"), 13, 1e-32, 1e-27)]
    for p, kf, tol_step, tol in jobs:
        zh, zl, m = load(p)
        out = os.path.join(BENCH, "states", f"bench_D{m['D']:.4f}_M{m['M']}.npz")
        if os.path.exists(out):
            print(f"{out}: exists")
            continue
        M = m["M"]
        if kf:
            lo, lo2 = plow_odd(M, kf)
            zh, zl = np.asarray(zh).copy(), np.asarray(zl).copy()
            zh[:M] = (lo + lo2) @ (zh[:M] + zl[:M])
            zl[:M] = 0.0
        s = make_solver(m["D"], M, nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads, kfreeze=kf)
        s.set_segs(s.plan(zh, zl, tol_step, 0.05))
        print(f"{os.path.basename(p)}{' frozen at ' + str(kf) if kf else ''}: |F| {dd_norm(*s.residual(zh, zl)):.1e}", flush=True)
        if m["D"] >= DEFLATED_FROM:
            a, b, hist = deflated_newton(s, zh, zl, tol=tol, it_max=25, w=24, verbose=True)
        else:
            a, b, hist = newton_solve(s, zh, zl, tol=tol, it_max=12, verbose=True)
        F = dd_norm(*s.residual(a, b))
        if F < tol:
            meta = dict(m, F=F, source=os.path.basename(p), tol_step=tol_step)
            if kf:
                meta["kfreeze"] = kf
            save(out, a, b, meta)
            print(f"  saved {out}: |F| {F:.1e}, Delta {a[-1]!r} + {b[-1]:.3e}", flush=True)
        else:
            print(f"  NOT converged (|F| {F:.1e}); not saved", flush=True)


def main():
    opts = {}
    for a in sys.argv[1:]:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opts[k] = v if v else True
    opts.setdefault("out", os.path.join(BENCH, f"bench_{socket.gethostname()}.jsonl"))
    if opts.get("prepare"):
        return prepare(opts)
    if opts.get("calibrate"):
        return calibrate(opts)
    if opts.get("report"):
        return report(opts)
    return run(opts)


if __name__ == "__main__":
    main()
