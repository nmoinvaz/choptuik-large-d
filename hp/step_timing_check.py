#!/usr/bin/env python3
"""What Solver.step_timing's four entries hold, and the Jacobian read from the Newton call's wall clock (#1040).

    cd hp && python3 step_timing_check.py [--state=STATE.npz] [--reps=3] [--threads=2] [--tol=0.05]
        [--driver=checkout|installed] [--mutate=drop]

At one stored state (default data/lowD_states/highm15_D3.7000_M80.npz, on its own plan), --reps times in turn: one f64
Newton call in block storage through driver.newton_step (jac "f64", block=True), and one in dense storage
(Solver.newton_mixed(block=False)), each timed by the wall clock and the process CPU clock around it. Both form the same
columns (multishoot's f64_columns, f64x4 lanes). The dense call's second entry times the f64 Jacobian, its columns and
their writing into the dense matrix: it is the measured remainder C, the work that no entry of the block call holds.
Another process's load lengthens a call's wall time but hardly its CPU time, so C is carried from the dense calls to
each block call at the calls' ratios of wall to CPU seconds k: C_b = k_b median(C_d / k_d), the median over the dense
calls (on a loaded machine the fastest calls of the two storages can differ by tens of percent in wall time, while
their CPU seconds agree to a few percent). Checks, each of which fails the run (exit 1):
  1. dense closure: in every dense call the four entries sum to the call's wall time within 0.5%;
  2. block closure: a block call's four entries plus C_b equal its wall time within --tol (5%; the median over calls);
  3. the old reading falls short: the block call's second entry, which scripts read as the Jacobian before #1040, is
     under a tenth of C_b in every block call;
  4. the fixed reading: newton_step's t_jac (the call's wall time less the residual, factorization and solve) is within
     --tol of C_b (the median over calls), and its t_fill is step_timing's second entry;
  5. the control can fire: the dense call's factorization entry is over twice check 1's bar.
Control: --mutate=drop drops the factorization entry from the sums of checks 1 and 2; check 1 must then fail (exit 1).
On a build before #523 the block call's second entry held the Jacobian, so checks 2 and 3 fail there and check 4
passes: the wall-clock reading holds on both builds.
--driver=checkout (the default) loads hpkernel/driver.py from this checkout into the installed hpkernel package, so a
build made before #1040 runs the fixed reading; --driver=installed uses the installed driver. The state must have
M <= 128 (the dense path).
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "data", "lowD_states", "highm15_D3.7000_M80.npz")
DENSE_BAR = 0.005      # check 1: the dense call's time outside its four entries, relative to the call
SHORT = 0.1            # check 3: the block call's second entry against C


def load_driver(which):
    """the driver under test: this checkout's hpkernel/driver.py in the installed package, or the installed one"""
    import hpkernel
    if which == "installed":
        import hpkernel.driver as drv
        return drv
    path = os.path.join(HERE, "python", "hpkernel", "driver.py")
    spec = importlib.util.spec_from_file_location("hpkernel.driver", path)
    drv = importlib.util.module_from_spec(spec)
    sys.modules["hpkernel.driver"] = drv
    hpkernel.driver = drv
    spec.loader.exec_module(drv)
    return drv


def kernel_info():
    """the hpkernel in use: its folder and the crate versions in its binary"""
    import hpkernel
    d = os.path.dirname(hpkernel.__file__)
    vers = set()
    for so in glob.glob(os.path.join(d, "_hpkernel*")):
        with open(so, "rb") as f:
            vers |= {m.decode() for m in re.findall(rb"(multi(?:double|shoot|solver)-\d+\.\d+\.\d+)", f.read())}
    return d, sorted(vers)


def timed(fn):
    """fn's wall and process CPU seconds"""
    c0, t0 = time.process_time(), time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0, time.process_time() - c0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state", default=STATE)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--driver", choices=("checkout", "installed"), default="checkout")
    ap.add_argument("--mutate", choices=("drop",))
    a = ap.parse_args()
    drv = load_driver(a.driver)
    kdir, crates = kernel_info()
    print(f"hpkernel {kdir} ({', '.join(crates) or 'no crate versions found'}); driver {drv.__file__}", flush=True)
    d = np.load(a.state, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    zh, zl = np.asarray(d["zh"], float), np.asarray(d["zl"], float)
    M = int(meta["M"])
    if M > 128:
        raise SystemExit(f"{a.state}: M {M} > 128 has no dense path to measure the remainder with")
    s = drv.make_solver(meta["D"], M, nodes=list(meta["nodes"]), kt=meta["kt"], kc=meta["kc"], ks=meta["ks"],
                        threads=a.threads, segs=meta["segs"])
    steps = sum(len(g) - 1 for g in meta["segs"])
    print(f"{os.path.basename(a.state)}: D {meta['D']} M {M} N {len(zh)}, {steps} plan steps, {a.threads} threads, "
          f"{a.reps} reps" + (f"; --mutate={a.mutate}" if a.mutate else ""), flush=True)
    # the entries a sum adds: all four, or under --mutate=drop all but the factorization's
    keep = (0, 1, 3) if a.mutate == "drop" else (0, 1, 2, 3)

    def total(e):
        return sum(e[i] for i in keep)

    blk, den = [], []
    for r in range(a.reps):
        order = ("block", "dense") if r % 2 == 0 else ("dense", "block")
        for kind in order:
            if kind == "block":
                out, w, c = timed(lambda: drv.newton_step(s, zh, zl, "f64", dict(block=True)))
                blk.append(dict(wall=w, cpu=c, e=s.step_timing(), info=out[-1]))
            else:
                _, w, c = timed(lambda: s.newton_mixed(zh, zl, block=False))
                den.append(dict(wall=w, cpu=c, e=s.step_timing()))
            q = (blk if kind == "block" else den)[-1]
            print(f"  rep {r} {kind:5s}: wall {q['wall']:8.3f} s, cpu {q['cpu']:8.2f} s; entries (residual, Jacobian, "
                  f"factorization, solve) " + " ".join(f"{v:.4f}" for v in q["e"])
                  + f"; wall - entries {q['wall'] - sum(q['e']):.4f} s", flush=True)

    # C in CPU seconds, from the dense calls; C_b, its wall seconds at a block call's ratio k of wall to CPU seconds
    c_cpu = float(np.median([q["e"][1] * q["cpu"] / q["wall"] for q in den]))
    for q in blk:
        q["C"] = c_cpu * q["wall"] / q["cpu"]
    fails = []

    def verdict(n, ok, text):
        print(f"{n}. {'PASS' if ok else 'FAIL'}: {text}", flush=True)
        if not ok:
            fails.append(n)

    gap = max(abs(q["wall"] - total(q["e"])) / q["wall"] for q in den)
    verdict(1, gap <= DENSE_BAR, f"dense closure: the four entries miss at most {gap:.2%} of a dense call "
            f"(bar {DENSE_BAR:.1%})")
    clo = [(total(q["e"]) + q["C"] - q["wall"]) / q["wall"] for q in blk]
    mc = float(np.median(clo))
    verdict(2, abs(mc) <= a.tol, f"block closure: (entries + C_b - wall) / wall is {mc:+.2%}, the median over the "
            f"block calls (" + ", ".join(f"{v:+.1%}" for v in clo) + f"; bar {a.tol:.0%}); C is {c_cpu:.2f} CPU s")
    old = max(q["e"][1] / q["C"] for q in blk)
    verdict(3, old < SHORT, f"the old reading falls short: the block call's second entry is at most {old:.2e} of C_b "
            f"(bar {SHORT:g}); it misses {1 - old:.2%} of the Jacobian")
    info = [q["info"] for q in blk]
    if not all("t_call" in i and "t_fill" in i for i in info):
        verdict(4, False, f"the fixed reading: {drv.__file__} predates #1040 (newton_step's info has no t_call, "
                f"t_fill)")
    else:
        rel = [(i["t_jac"] - q["C"]) / q["wall"] for i, q in zip(info, blk)]
        mr = float(np.median(rel))
        same = all(i["t_fill"] == q["e"][1] for i, q in zip(info, blk))
        verdict(4, abs(mr) <= a.tol and same, f"the fixed reading: (newton_step's t_jac - C_b) / wall is {mr:+.2%}, "
                f"the median over the block calls (bar {a.tol:.0%}); t_fill is step_timing's second entry in every "
                f"call: {same}")
    f2 = min(q["e"][2] / q["wall"] for q in den)
    verdict(5, f2 > 2 * DENSE_BAR, f"the control can fire: the dense factorization is at least {f2:.2%} of the call "
            f"(over twice {DENSE_BAR:.1%})")
    cb, cd = np.median([q["cpu"] for q in blk]), np.median([q["cpu"] for q in den])
    print(f"information: median process CPU seconds, block {cb:.2f}, dense {cd:.2f} ({cb / cd - 1:+.1%}); the fastest "
          f"block call's wall less its entries {min(q['wall'] - sum(q['e']) for q in blk):.3f} s, the smallest dense "
          f"second entry {min(q['e'][1] for q in den):.3f} s", flush=True)
    print(f"step_timing_check: {'PASS' if not fails else 'FAIL (checks ' + ', '.join(map(str, fails)) + ')'}",
          flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
