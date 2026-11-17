#!/usr/bin/env python3
"""The stall rule of the low-D solves (#925): `stall` accepted Newton steps that together do not bring max |F| below
stall_factor (1/2) of its value before the first of them fail the solve, so that lowD_highM.py halves its step in D.
The rule is driver.newton_solve's option stall; lowD_dd.solve_point applies a copy of it when the installed hpkernel
predates the option.

    python3 test_stall.py [--cases 1,2,3] [--threads 1]   exit 0 when every check passes
    python3 test_stall.py --cases 4                        the planted leg (case 4)
    python3 test_stall.py --cases 1,2 --mutate nostall     the rule off where it must fire: exit 1 is the expected result
    python3 test_stall.py --cases 4 --mutate nostall       the same for the leg: exit 1 is the expected result
    python3 test_stall.py --cases 3 --mutate eager         stall_factor 1e-12 in case 3: exit 1 is the expected result

Every solve runs on two paths. "driver" passes the option to the installed driver.newton_solve (skipped, with a line
saying so, when that predates #925). "script" gives solve_point a newton_solve with the signature before #925, so that
solve_point applies the rule itself.
(1) Toy, through newton_solve and solve_point: F(z) = z on 2 unknowns, its Newton step turned by acos(0.26) (a wrong
    Jacobian). The line search takes lam 1/2 at every step and |F|_2 falls 0.5% a step, as at #343's D 3.04315, M 320.
    Rule on: failed "stall" after the 3rd accepted step. Off: it_max (8) accepted steps, no failure.
(2) Planted on the solver: the converged M 48 state at D 4 (data/lowD_states/lowD_D4.0000_M48.npz), perturbed as
    test_chord.py does, solved at D 4 with every Newton step scaled by 0.05 (|F| falls 5% a step at lam 1). Rule on:
    not converged, failed "stall" after 3 accepted steps, the same history on both paths. Off: it_max (5) steps.
(3) Unaffected: the converged M 80 state at D 3.4 (data/lowD_states/lowDhm_D3.4000_M80.npz) continued to D 3.39 from
    itself on its stored plan (a leg's "last state" start). It converges in more than 3 accepted steps, so the rule is
    evaluated, and the history (every record but its times), Delta and the state are bit-identical with the rule on and
    off, and between the two paths.
(4) Leg: lowD_highM.py from the converged pair at D 3.4, M 80 / 96 (lowDhm_D3.4000_M80.npz, _M96.npz), target D 3.35,
    --it_max 5, every Newton step of solve_point scaled by 0.05, and --hours 0.002, so that the run stops at the first
    point boundary. Pass: the M 80 solve gives up on the stall after 3 accepted steps, D 3.375 is inserted (the step
    halved) and is the next target in the progress file, and the notice of the script-level rule appears once if the
    installed driver predates #925, else never.
Minutes at one thread: cases 1-3 about 20 on the Mac under load, case 4 about 10."""
import argparse
import contextlib
import functools
import inspect
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from hpkernel import driver  # noqa: E402
from hpkernel.driver import load  # noqa: E402
import lowD_dd  # noqa: E402

STATES = os.path.join(HERE, "data", "lowD_states")
TIMES = ("t", "t_res", "t_jac", "t_lu", "t_solve", "t_prod", "script")
NEW = lowD_dd.newton_solve
HAS = "stall" in inspect.signature(NEW).parameters
PATHS = (["driver"] if HAS else []) + ["script"]


def old_signature(s, zh, zl, tol=1e-28, it_max=12, chord=0, verbose=True, ls=True, max_halvings=10, jac="dd",
                  mixed=None, gmres=None, dd_finish=True, dd_last=False):
    """driver.newton_solve with the signature before #925 (no stall option)"""
    return NEW(s, zh, zl, tol=tol, it_max=it_max, chord=chord, verbose=verbose, ls=ls, max_halvings=max_halvings,
               jac=jac, mixed=mixed, gmres=gmres, dd_finish=dd_finish, dd_last=dd_last)


class Toy:
    """F(z) = z on 2 unknowns; its Newton step -z turned by the angle acos(c)"""
    m, size = 1, 2

    def __init__(self, c=0.26):
        self.c, self.sn, self.segs = c, np.sqrt(1 - c * c), None

    def residual(self, zh, zl):
        return np.array(zh, float), np.array(zl, float)

    def newton_mixed(self, zh, zl, **k):
        z = np.asarray(zh) + np.asarray(zl)
        d = -np.array([self.c * z[0] - self.sn * z[1], self.sn * z[0] + self.c * z[1]])
        return d, np.zeros(2), np.array(zh, float), np.array(zl, float), 1.0, 1.0

    newton = newton_mixed

    def plan(self, *a):
        return [[0.0, 1.0]]

    def set_segs(self, segs):
        self.segs = segs

    def get_segs(self):
        return self.segs


class Short:
    """the solver, with every Newton step scaled by eps (planted: |F| falls by about eps a step at lam 1)"""
    def __init__(self, s, eps):
        self.s, self.eps = s, eps

    def __getattr__(self, name):
        return getattr(self.s, name)

    def _scaled(self, out):
        return (np.asarray(out[0]) * self.eps, np.asarray(out[1]) * self.eps) + tuple(out[2:])

    def newton_mixed(self, *a, **k):
        return self._scaled(self.s.newton_mixed(*a, **k))

    def newton(self, *a, **k):
        return self._scaled(self.s.newton(*a, **k))


_make = lowD_dd.make_solver
PLANT = dict(kind=None, eps=0.05)


def _planted_make(*a, **k):
    """lowD_dd.make_solver with the plant of the current case (none, the toy, or the scaled step)"""
    if PLANT["kind"] == "toy":
        return Toy()
    s = _make(*a, **k)
    return Short(s, PLANT["eps"]) if PLANT["kind"] == "short" else s


lowD_dd.make_solver = _planted_make


def ns(out, tag, meta, threads, stall, factor, it_max):
    return argparse.Namespace(out=out, tag=tag, resume=False, kt=meta["kt"], kcs=meta["kc"], ks=meta["ks"],
                              tol_step=1e-32, hmax=0.05, solver="newton", it_max=it_max, tol=1e-28, accept=1e-27, lm=0,
                              jac="f64", threads=threads, svd=0, dn_its=2, w=24, stop_lam=0.0, floor_ratio=0.0,
                              chord_from=0.0, stall=stall, stall_factor=factor)


def strip(hist):
    """the records without their times (and the script-level rule's marker)"""
    return [{k: v for k, v in r.items() if k not in TIMES} for r in hist]


def steps(rec):
    return sum("lam" in r for r in rec)


def solve(path, D, zh, zl, meta, d, threads, stall, factor, it_max, tag, segs=None):
    """solve_point on a path, every newton_solve history captured; returns (converged, zh, zl, newton_solve records,
    log lines, the points-file record without its times)"""
    rec = []
    if path == "driver":
        @functools.wraps(NEW)                     # (keeps the signature solve_point inspects)
        def call(*a, **k):
            out = NEW(*a, **k)
            rec.extend(out[2])
            return out
    else:
        def call(s, zh, zl, tol=1e-28, it_max=12, chord=0, verbose=True, ls=True, max_halvings=10, jac="dd",
                 mixed=None, gmres=None, dd_finish=True, dd_last=False):
            out = old_signature(s, zh, zl, tol, it_max, chord, verbose, ls, max_halvings, jac, mixed, gmres,
                                dd_finish, dd_last)
            rec.extend(out[2])
            return out
    lowD_dd.newton_solve = call
    lines = []
    out = os.path.join(d, tag + ".jsonl")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ok, h, l = lowD_dd.solve_point(D, meta["M"], zh.copy(), zl.copy(), "test",
                                           ns(out, tag, meta, threads, stall, factor, it_max), meta["nodes"], d,
                                           lines.append, segs=segs)
    finally:
        lowD_dd.newton_solve = NEW
    point = json.loads(open(out).read().splitlines()[-1])
    point.pop("seconds", None)
    point.pop("t_its", None)
    point.pop("peak_rss_mb", None)
    return ok, np.asarray(h), np.asarray(l), rec, lines, point


def decider(rec):
    """which code failed the solve on the stall: the driver (its record among newton_solve's) or the script"""
    return "driver" if any(r.get("failed") == "stall" for r in rec) else "script"


def steps_only(rec):
    return [r for r in strip(rec) if "failed" not in r]


def case1(fails, mutate):
    stall = 0 if mutate == "nostall" else 3
    z = np.array([1.0, 0.5])
    if HAS:
        # newton_solve itself, its own window (it_max 8 in one call)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, on = NEW(Toy(), z, np.zeros(2), it_max=8, jac="f64", stall=stall)
            _, _, off = NEW(Toy(), z, np.zeros(2), it_max=8, jac="f64", stall=0)
        print(f"(1) toy, newton_solve: rule on: {steps(on)} accepted steps at lam "
              f"{sorted({r['lam'] for r in on if 'lam' in r})}, last record {on[-1].get('failed', 'a step')}; off: "
              f"{steps(off)} accepted steps, failures {[r['failed'] for r in off if 'failed' in r]}")
        if not (on[-1].get("failed") == "stall" and steps(on) == 3):
            fails.append("(1) newton_solve: no stall after 3 accepted steps")
        if not (steps(off) == 8 and not any("failed" in r for r in off)):
            fails.append("(1) newton_solve, rule off: not 8 accepted steps")
    else:
        print("(1) toy, newton_solve: skipped (the installed driver predates #925)")
    PLANT["kind"] = "toy"
    meta = dict(kt=36, kc=60, ks=100, M=2, nodes=[0.5])
    with tempfile.TemporaryDirectory() as d:
        for path in PATHS:
            ok1, _, _, r1, l1, p1 = solve(path, 4.0, z, np.zeros(2), meta, d, 1, stall, 0.5, 8, f"toy_{path}_on")
            ok0, _, _, r0, _, p0 = solve(path, 4.0, z, np.zeros(2), meta, d, 1, 0, 0.5, 8, f"toy_{path}_off")
            gave_up = any("(a stall); giving up" in x for x in l1)
            print(f"(1) toy, solve_point, {path} path: rule on: converged {ok1}, {steps(r1)} accepted steps, failed "
                  f"{p1.get('failed')!r} ({decider(r1)}); off: converged {ok0}, {steps(r0)} accepted steps, failed "
                  f"{p0.get('failed')!r}")
            if not (gave_up and not ok1 and p1.get("failed") == "stall" and steps(r1) == 3):
                fails.append(f"(1) solve_point, {path} path: no stall after 3 accepted steps")
            elif decider(r1) != path:
                fails.append(f"(1) solve_point, {path} path: the stall was decided by the {decider(r1)}")
            if not (not ok0 and steps(r0) == 8 and "failed" not in p0):
                fails.append(f"(1) solve_point, {path} path, rule off: not 8 accepted steps")
    PLANT["kind"] = None


def case2(fails, mutate, threads):
    stall = 0 if mutate == "nostall" else 3
    zh, zl, meta = load(os.path.join(STATES, "lowD_D4.0000_M48.npz"))
    zh = zh + 1e-8 * np.random.default_rng(1).standard_normal(len(zh))
    zh[-1] += 1e-6
    PLANT["kind"] = "short"
    got = {}
    with tempfile.TemporaryDirectory() as d:
        for path in PATHS:
            for lab, st in (("on", stall), ("off", 0)):
                ok, h, l, rec, _, pt = solve(path, 4.0, zh, zl, meta, d, threads, st, 0.5, 5, f"pl_{path}_{lab}")
                got[path, lab] = (ok, h, l, rec, pt)
                acc = [r for r in rec if "lam" in r]
                print(f"(2) planted, {path} path, rule {lab}: converged {ok}, {len(acc)} accepted steps at lam "
                      f"{sorted({r['lam'] for r in acc})}, |F| " + " ".join(f"{r['F']:.4e}" for r in acc)
                      + f" -> {pt['F']:.4e}, failed {pt.get('failed')!r}"
                      + (f" ({decider(rec)})" if pt.get("failed") else ""))
    PLANT["kind"] = None
    for path in PATHS:
        ok, _, _, rec, pt = got[path, "on"]
        if not (not ok and pt.get("failed") == "stall" and steps(rec) == 3):
            fails.append(f"(2) {path} path: no stall after 3 accepted steps")
        elif decider(rec) != path:
            fails.append(f"(2) {path} path: the stall was decided by the {decider(rec)}")
        ok, _, _, rec, pt = got[path, "off"]
        if not (not ok and steps(rec) == 5 and "failed" not in pt):
            fails.append(f"(2) {path} path, rule off: not 5 accepted steps")
    if HAS:
        a, b = got["driver", "on"], got["script", "on"]
        same = (steps_only(a[3]) == steps_only(b[3]) and a[4] == b[4] and np.array_equal(a[1], b[1])
                and np.array_equal(a[2], b[2]))
        print(f"(2) driver and script paths, rule on: steps, final state and record "
              f"{'bit-identical' if same else 'DIFFERENT'}")
        if not same:
            fails.append("(2) the driver and script paths differ")


def case3(fails, mutate, threads):
    factor = 1e-12 if mutate == "eager" else 0.5
    zh, zl, meta = load(os.path.join(STATES, "lowDhm_D3.4000_M80.npz"))
    got = {}
    with tempfile.TemporaryDirectory() as d:
        for path in PATHS:
            for lab, st in (("on", 3), ("off", 0)):
                ok, h, l, rec, _, pt = solve(path, 3.39, zh, zl, meta, d, threads, st, factor, 14,
                                             f"un_{path}_{lab}", segs=meta["segs"])
                got[path, lab] = (ok, h, l, rec, pt)
                print(f"(3) unaffected, {path} path, rule {lab}: converged {ok}, Delta {h[-1]:.16f} ({l[-1]:+.2e}), "
                      f"{steps(rec)} accepted steps, |F| " + " ".join(f"{r['F']:.2e}" for r in rec)
                      + (f", failed {pt['failed']!r}" if "failed" in pt else ""))
    pairs = [(p, "on", p, "off") for p in PATHS] + ([("driver", "on", "script", "on")] if HAS else [])
    for key in pairs:
        a, b = got[key[:2]], got[key[2:]]
        same = (strip(a[3]) == strip(b[3]) and a[4] == b[4] and np.array_equal(a[1], b[1])
                and np.array_equal(a[2], b[2]))
        print(f"(3) {key[0]} path rule {key[1]} against {key[2]} path rule {key[3]}: history, record and state "
              f"{'bit-identical' if same else 'DIFFERENT'}")
        if not same:
            fails.append(f"(3) {key[0]} {key[1]} and {key[2]} {key[3]} differ")
    for path in PATHS:
        a, b = got[path, "on"], got[path, "off"]
        if not (a[0] and b[0]):
            fails.append(f"(3) {path} path: did not converge")
        # the rule is evaluated with |F| above tol after every accepted step but the last
        if steps(b[3]) < 4:
            fails.append(f"(3) {path} path: {steps(b[3])} accepted steps, the rule was not evaluated")


def case4(fails, mutate, threads):
    import lowD_highM
    stall = 0 if mutate == "nostall" else 3
    PLANT["kind"] = "short"
    lowD_dd.STALL_NOTICE.clear()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "leg.jsonl")
        sys.argv = ["lowD_highM.py", "--start_w", os.path.join(STATES, "lowDhm_D3.4000_M80.npz"), "--start_n",
                    os.path.join(STATES, "lowDhm_D3.4000_M96.npz"), "--Ms", "80,96", "--dD", "0.05", "--D_end", "3.3",
                    "--min_step", "0.01", "--diff_max", "1", "--hours", "0.002", "--tag", "legtest", "--out", out,
                    "--points", os.path.join(d, "points.jsonl"), "--states", os.path.join(d, "states"),
                    "--threads", str(threads), "--it_max", "5", "--stall", str(stall), "--allow_unraw"]
        os.makedirs(os.path.join(d, "states"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lowD_highM.main()
        text = buf.getvalue()
        prog = json.load(open(out + ".progress.json"))
    PLANT["kind"] = None
    sys.stdout.write(text)
    lines = text.splitlines()
    head = [i for i, x in enumerate(lines) if "D 3.35 M 80: L " in x]
    gave = [i for i, x in enumerate(lines) if "D 3.35 M 80:" in x and "(a stall); giving up" in x]
    ins = [i for i, x in enumerate(lines) if "D 3.35 failed at M 80; inserting D 3.375" in x]
    n = sum(" lam " in x for x in lines[head[0]:gave[0]]) if head and gave else None
    notices = sum("this hpkernel has no stall option" in x for x in lines)
    print(f"(4) leg: gave up on the stall {bool(gave)} after {n} accepted steps, D 3.375 inserted {bool(ins)}, next "
          f"target {prog['targets'][:1]}, notices {notices} (the installed driver {'has' if HAS else 'lacks'} the "
          f"option)")
    if not (gave and n == 3):
        fails.append("(4) the M 80 solve did not give up on the stall after 3 accepted steps")
    if not (ins and gave and ins[0] > gave[0] and prog["targets"][:1] == [3.375]):
        fails.append("(4) no midpoint D 3.375 inserted after the stall")
    if notices != (1 if stall and not HAS else 0):
        fails.append(f"(4) {notices} notices of the script-level rule")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,2,3")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--mutate", choices=["nostall", "eager"], default=None)
    a = ap.parse_args()
    cases = [int(c) for c in a.cases.split(",")]
    print(f"installed driver {driver.__file__}: {'with' if HAS else 'without'} the stall option; paths {PATHS}")
    fails = []
    if 1 in cases:
        case1(fails, a.mutate)
    if 2 in cases:
        case2(fails, a.mutate, a.threads)
    if 3 in cases:
        case3(fails, a.mutate, a.threads)
    if 4 in cases:
        case4(fails, a.mutate, a.threads)
    print((f"mutated ({a.mutate}): " if a.mutate else "") + ("; ".join(fails) if fails else "all checks pass"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
