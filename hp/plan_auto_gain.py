#!/usr/bin/env python3
"""The Taylor order chosen by the cost of the planned steps (Solver.plan_auto, multishoot's plan_segments_auto), checked
and measured (notes/LOW_D_SPEEDUP.md, "The Taylor order by the cost of planned steps").

    cd hp && python3 plan_auto_gain.py planted [--check data/plan_auto/planted.txt] [--write F] [--mutate n]
    cd hp && python3 plan_auto_gain.py ab STATE --out F.jsonl [--w W] [--threads N] [--tol_step T] [--hmax H] [--kmax K]
        [--mutate invert]
    cd hp && python3 plan_auto_gain.py abo STATE --out F.jsonl [--w W] [--threads N] [--reps 2] [--kmax K] [--mutate invert]
    cd hp && python3 plan_auto_gain.py report F.jsonl [F.jsonl ...] [--mutate delta]
    cd hp && python3 plan_auto_gain.py selftest

planted (no kernel, seconds): the truth of the planted problem that hp/src/plan_auto.rs's tests plan. With p = m = 0
the model's Taylor recursion (model.rs taylor_coeffs) keeps p and m at zero and reduces to x a' = (n-1)(a - a^3)/2 and
x c' = (n-1) c (a^2 - 1), whose solution is a^2 = u / (1 + u), u = K2 x^(n-1), and c = C (1 + u) / x^(n-1). Its Taylor
coefficients come here from the closed forms by power-series arithmetic in mpmath (c's from the binomial series of
x^-(n-1), a's from the series of u / (1 + u) and its square root), not from the model's recursion. The planner's rule is
applied to them at every order 12..120: at each point the step is min((tol / e_(k-1))^(1/(k-1)), (tol / e_k)^(1/k),
hmax), e_j = max(|a_j|, |c_j|), and a step above 60% of the rest becomes half the rest (multishoot's plan_path). The
file holds the plant, its node states in double-double and the steps per segment at each order; plan_auto.rs reads it
and requires its planner's steps to equal these at every order, and the order chosen to be this table's least cost.
Checks (exit 1 on failure): the series satisfy both equations' recursions and sum to the closed forms at a step of
0.004 (to 1e-28), and with --check the recomputed table equals the stored file. Control: --mutate n (the plant at n
+ 0.25) must fail --check.

ab (kernel): one f64 Newton call (driver.newton_step, block storage, as tau_map_gain.py cost) on a stored state with
each plan, in the same process: the fixed order's (Solver.plan at the state's kt, today's path) and plan_auto's (two
passes, step cost k^2 + w k), both from the state at --tol_step and --hmax. The state's stored plan is not an arm: it
was planned from another state (the pass-1 state of its leg, notes/LOW_D_SPEEDUP.md step 5), so its steps are not
comparable. Records each plan's order, steps, the estimated cost, the wall seconds of the call, |F| before and after
and Delta after. Pass criteria, fixed before the runs: the two Delta after the call agree to 1e-20 (the L 1
criterion of tau_map_gain.py: the same solution at the same M), |F| after the call is below --tol (1e-28) on both
plans, and the auto plan's estimated cost is at most the fixed plan's. The gain is the fixed plan's call seconds over the
auto plan's; the error is the Delta difference. Control: --mutate invert (the inverted step cost, model plan_auto's
MUT_INVERT) must fail the cost criterion.

abo (kernel, M 32 eight-part states of the chain, hp/anm_run.py): the octo-double Jacobian (Solver.jacobian_o, the
chain's corrector's) on the state's stored plan, which the chain keeps, and on plan_auto's at the chain's tol 1e-32
and hmax 0.05, --reps times each, alternating. The chain's corrector is pinned (pinning.PinnedSolver), so no Newton
step of the plain solver is compared; the plans' own error is the octo-double residual of the Taylor blocks (rows of
blocks 1..K-1) at the state. Pass: that residual on the auto plan is at most max(1e-30, 10 times the stored plan's),
and the auto plan's estimated cost is at most the stored plan's. The gain is the stored plan's median Jacobian
seconds over the auto plan's. Control: --mutate invert must fail the cost criterion.

report: the ab and abo records as a table and the pass criteria (exit 1 when one fails, or when there is no record).
Control: --mutate delta moves every auto Delta by 1e-18, which must fail.

selftest (no kernel, seconds): report on planted records. A passing pair of each kind exits 0, and each planted fault
exits 1: report --mutate delta, Delta 1e-19 apart, an auto plan of higher estimated cost, |F| 1e-27 after the call, the
auto plan's Taylor rows at 1e-29, an arm missing, no records.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, HERE)
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

PLANTED = os.path.join(HERE, "data", "plan_auto", "planted.txt")
# the plant: D 6 (n = 4), a^2 = u / (1 + u) with u = K2 x^3 (branch points at |x| = K2^(-1/3) = 0.5), c = C (x^-3 + K2)
# (a pole at x = 0; c < x on the nodes, so x (x - c) does not vanish), the chain's tolerance and a cap that binds only
# at the highest orders
PLANT = dict(n=4.0, K2=8.0, C=1e-8, nodes=[0.02, 0.05, 0.1, 0.2, 0.4], tol=1e-32, hmax=0.5, delta=3.0)
ORDERS = list(range(12, 121))
TOL_DELTA = 1e-20


# ------------------------------------------------------------------ planted (no kernel)
def series(x0, kmax, p):
    """Taylor coefficients 0..kmax of a and c about x0 (mpmath), from the closed forms"""
    n1 = mp.mpf(p["n"]) - 1
    K2, C = mp.mpf(p["K2"]), mp.mpf(p["C"])
    x0 = mp.mpf(x0)
    # x^-n1 about x0: binom(-n1, j) x0^(-n1-j)
    xm = [x0 ** -n1]
    for j in range(1, kmax + 1):
        xm.append(xm[-1] * (-n1 - j + 1) / j / x0)
    c = [C * (xm[0] * (1 + K2 * x0 ** n1))] + [C * xm[j] for j in range(1, kmax + 1)]
    # x^n1 about x0, then u, s = u / (1 + u) and a = sqrt(s)
    xp = [x0 ** n1]
    for j in range(1, kmax + 1):
        xp.append(xp[-1] * (n1 - j + 1) / j / x0)
    u = [K2 * v for v in xp]
    v = [1 + u[0]] + u[1:]
    s = []
    for j in range(kmax + 1):
        s.append((u[j] - mp.fsum(v[i] * s[j - i] for i in range(1, j + 1))) / v[0])
    a = [mp.sqrt(s[0])]
    for j in range(1, kmax + 1):
        a.append((s[j] - mp.fsum(a[i] * a[j - i] for i in range(1, j))) / (2 * a[0]))
    return a, c


def cauchy(x, y, k):
    return mp.fsum(x[i] * y[k - i] for i in range(k + 1))


def series_check(p, x0=0.07, kmax=40, h=0.004):
    """largest relative misfit of the series: the recursions (x0 + t) a' = (n-1)(a - a^3)/2 and (x0 + t) c' =
    (n-1) c (a^2 - 1) order by order, and the sums at x0 + h against the closed forms"""
    n1 = mp.mpf(p["n"]) - 1
    a, c = series(x0, kmax, p)
    x0 = mp.mpf(x0)
    a2 = [cauchy(a, a, k) for k in range(kmax + 1)]
    a3 = [cauchy(a2, a, k) for k in range(kmax + 1)]
    worst = mp.mpf(0)
    for k in range(kmax):
        # coefficient k of (x0 + t) f' = x0 (k+1) f_(k+1) + k f_k
        la = x0 * (k + 1) * a[k + 1] + k * a[k]
        ra = n1 * (a[k] - a3[k]) / 2
        lc = x0 * (k + 1) * c[k + 1] + k * c[k]
        rc = n1 * (cauchy(c, a2, k) - c[k])
        worst = max(worst, abs(la - ra) / (abs(ra) + abs(la)), abs(lc - rc) / (abs(rc) + abs(lc)))
    x = x0 + h
    u = mp.mpf(p["K2"]) * x ** n1
    exact_a, exact_c = mp.sqrt(u / (1 + u)), mp.mpf(p["C"]) * (1 + u) / x ** n1
    sa = mp.fsum(a[k] * mp.mpf(h) ** k for k in range(kmax + 1))
    sc = mp.fsum(c[k] * mp.mpf(h) ** k for k in range(kmax + 1))
    worst = max(worst, abs(sa / exact_a - 1), abs(sc / exact_c - 1))
    return float(worst)


def plan_steps(k, p):
    """steps per segment at order k by the planner's rule on the exact coefficients (multishoot's plan_path: the
    points in f64, the step from the hi parts' norms)"""
    out = []
    nodes = p["nodes"]
    for b in range(len(nodes) - 1):
        xa, xb = nodes[b], nodes[b + 1]
        x, n = xa, 0
        while xb - x > 0:
            a, c = series(x, k, p)
            e1 = max(abs(float(a[k - 1])), abs(float(c[k - 1])))
            e2 = max(abs(float(a[k])), abs(float(c[k])))
            h1 = float((mp.mpf(p["tol"]) / e1) ** (mp.mpf(1) / (k - 1)))
            h2 = float((mp.mpf(p["tol"]) / e2) ** (mp.mpf(1) / k))
            h = min(h1, h2, p["hmax"])
            rest = xb - x
            if h >= rest:
                h = rest
            elif h > 0.6 * rest:
                h = 0.5 * rest
            x = xb if h == rest else x + h
            n += 1
        out.append(n)
    return out


def dd(x):
    hi = float(x)
    return hi, float(x - hi)


def planted_table(p):
    """the file's lines: the plant, the node states (a and c in double-double) and the steps at every order"""
    lines = ["# hp/plan_auto_gain.py planted: p = m = 0, a^2 = u / (1 + u), u = K2 x^(n-1), c = C (1 + u) / x^(n-1)",
             f"n {p['n']!r}", f"K2 {p['K2']!r}", f"C {p['C']!r}", "nodes " + " ".join(repr(x) for x in p["nodes"]),
             f"tol {p['tol']!r}", f"hmax {p['hmax']!r}", f"delta {p['delta']!r}"]
    for x in p["nodes"]:
        a, c = series(x, 0, p)
        lines.append("state " + " ".join(repr(v) for v in (x, *dd(a[0]), *dd(c[0]))))
    for k in ORDERS:
        lines.append("steps " + " ".join(str(v) for v in [k] + plan_steps(k, p)))
    return lines


def planted(a):
    mp.mp.dps = 40
    p = dict(PLANT)
    if a.mutate == "n":
        p["n"] += 0.25
    worst = series_check(p)
    ok = worst < 1e-28
    print(f"planted: the series' largest relative misfit {worst:.1e} (recursions and closed forms; bound 1e-28): "
          f"{'pass' if ok else 'FAIL'}")
    t0 = time.time()
    lines = planted_table(p)
    steps = {int(r.split()[1]): [int(v) for v in r.split()[2:]] for r in lines if r.startswith("steps ")}
    cost = {k: sum(v) * (k * k + 20 * k) for k, v in steps.items()}
    best = min(cost, key=lambda k: (cost[k], k))
    print(f"  {len(steps)} orders in {time.time() - t0:.1f} s; least cost at k^2 + 20 k: order {best}, steps "
          f"{steps[best]}, cost {cost[best]}; order 36: steps {steps[36]}, cost {cost[36]}")
    if a.write:
        with open(a.write, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  written: {a.write}")
    if a.check:
        old = open(a.check).read().splitlines()
        same = old == lines
        print(f"  {a.check}: {'the same' if same else 'DIFFERS'} ({len(old)} lines, {len(lines)} recomputed)")
        if not same:
            diff = [(o, n) for o, n in zip(old, lines) if o != n][:3]
            for o, n in diff:
                print(f"    stored {o}\n    now    {n}")
        ok = ok and same
    return 0 if ok else 1


# ------------------------------------------------------------------ ab (kernel)
def ab(a):
    from raw_out import require_raw
    from tau_map_gain import MIXED, fmax, kernel_info, load_state, nsteps
    from hpkernel.driver import dd_add, make_solver, newton_step
    require_raw(a.out)
    zh, zl, meta = load_state(a.state)
    M, kt0 = int(meta["M"]), int(meta.get("kt", 36))
    s = make_solver(meta["D"], M, nodes=list(meta["nodes"]), kt=kt0, kc=meta["kc"], ks=meta["ks"], threads=a.threads)
    if not hasattr(s, "plan_auto"):
        raise SystemExit("this hpkernel has no Solver.plan_auto (built before it); rebuild hp")
    info = kernel_info()
    t0 = time.time()
    segs0 = s.plan(zh, zl, a.tol_step, a.hmax)
    t_plan0 = time.time() - t0
    t0 = time.time()
    kt, segs, passes, dropped = s.plan_auto(zh, zl, a.tol_step, a.hmax, w=a.w, kmax=a.kmax,
                                            mutate=1 if a.mutate == "invert" else 0)
    t_plan = time.time() - t0
    cost = lambda k, n: n * (k * k + a.w * k)  # noqa: E731
    arms = [("fixed", kt0, segs0), ("auto", kt, segs)]
    for name, k, sg in arms:
        s.set_kt(k)
        s.set_segs(sg)
        F0 = fmax(s, zh, zl)
        t1 = time.time()
        dh, dl, Fh, Fl, pmin, pmax, _ = newton_step(s, zh, zl, "f64", MIXED)
        t_call = time.time() - t1
        nh, nl_ = dd_add(zh, zl, dh, dl)
        F1 = fmax(s, nh, nl_)
        rec = dict(cmd="ab", state=a.state, D=meta["D"], M=M, arm=name, kt=k, steps=nsteps(sg),
                   cost=cost(k, nsteps(sg)), w=a.w, F0=F0, F1=F1, Delta=[float(nh[-1]), float(nl_[-1])],
                   dDelta=float(dh[-1] + dl[-1]), piv=[pmin, pmax], t_call=round(t_call, 2), threads=a.threads,
                   tol_step=a.tol_step, hmax=a.hmax, mutate=a.mutate, kernel=info)
        rec.update(stored_steps=nsteps(meta["segs"]) if meta.get("segs") else None)
        if name == "auto":
            rec.update(t_plan=round(t_plan, 2), passes=passes, dropped=dropped)
        else:
            rec.update(t_plan=round(t_plan0, 2))
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{name}: kt {k}, {nsteps(sg)} steps, est. cost {rec['cost']:.3e}; |F| {F0:.2e} -> {F1:.2e}, Delta "
              f"{nh[-1]:.17f}, call {t_call:.1f} s", flush=True)
    return 0


def abo(a):
    from raw_out import require_raw
    from tau_map_gain import kernel_info, nsteps
    from hpkernel.driver import make_solver
    from octo_corrector import load_o
    require_raw(a.out)
    z8, meta = load_o(a.state)
    M, kt0, segs0 = int(meta["M"]), int(meta.get("kt", 36)), meta.get("segs")
    if M != 32 or not segs0:
        raise SystemExit(f"{a.state}: abo takes an M 32 chain state with its stored plan (M {M})")
    mp.mp.dps = 60
    D = mp.mpf(meta["D_str"]) if "D_str" in meta else mp.mpf(meta["D"])
    s = make_solver(D, M, nodes=list(meta["nodes"]), kt=kt0, kc=meta["kc"], ks=meta["ks"], threads=a.threads,
                    kfreeze=meta.get("kfreeze"))
    if not hasattr(s, "plan_auto"):
        raise SystemExit("this hpkernel has no Solver.plan_auto (built before it); rebuild hp")
    info = kernel_info()
    zh, zl = np.ascontiguousarray(z8[0]), np.ascontiguousarray(z8[1])
    t0 = time.time()
    kt, segs, passes, dropped = s.plan_auto(zh, zl, a.tol_step, a.hmax, w=a.w, kmax=a.kmax,
                                            mutate=1 if a.mutate == "invert" else 0)
    t_plan = time.time() - t0
    K = len(meta["nodes"])
    arms = {"fixed": (kt0, segs0), "auto": (kt, segs)}
    res, times = {}, {"fixed": [], "auto": []}
    for name, (k, sg) in arms.items():
        s.set_kt(k)
        s.set_segs(sg)
        F = s.residual_o(list(z8))
        Fs = np.abs(np.sum(np.asarray(F), axis=0))         # the parts summed in f64: enough for a magnitude
        res[name] = float(Fs[4 * M:K * 4 * M].max())
    for _ in range(a.reps):
        for name, (k, sg) in arms.items():
            s.set_kt(k)
            s.set_segs(sg)
            t1 = time.time()
            s.jacobian_o(list(z8))
            times[name].append(time.time() - t1)
    for name, (k, sg) in arms.items():
        ts = sorted(times[name])
        rec = dict(cmd="abo", state=a.state, D=float(D), M=M, arm=name, kt=k, steps=nsteps(sg), cost=nsteps(sg) * (k * k + a.w * k),
                   w=a.w, F_taylor=res[name], t_jac=[round(t, 2) for t in times[name]], t_jac_median=ts[len(ts) // 2],
                   threads=a.threads, tol_step=a.tol_step, hmax=a.hmax, mutate=a.mutate, kernel=info)
        if name == "auto":
            rec.update(t_plan=round(t_plan, 2), passes=passes, dropped=dropped)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{name}: kt {k}, {nsteps(sg)} steps, est. cost {rec['cost']:.3e}; Taylor rows max|F| {res[name]:.2e}; "
              f"jacobian_o {rec['t_jac']} s", flush=True)
    return 0


def report(a):
    recs = [json.loads(line) for p in a.files for line in open(p) if line.strip()]
    ok = True
    byo = {}
    for r in recs:
        if r.get("cmd") == "abo":
            byo.setdefault(r["state"], {})[r["arm"]] = r
    for st, arms in byo.items():
        if "fixed" not in arms or "auto" not in arms:
            print(f"{os.path.basename(st)}: one arm missing: FAIL")
            ok = False
            continue
        f, u = arms["fixed"], arms["auto"]
        F_ok = u["F_taylor"] <= max(1e-30, 10 * f["F_taylor"])
        c_ok = u["cost"] <= f["cost"]
        ok = ok and F_ok and c_ok
        print(f"{os.path.basename(st)}: kt {f['kt']} -> {u['kt']}, steps {f['steps']} -> {u['steps']}; octo-double "
              f"Jacobian {f['t_jac_median']:.1f} -> {u['t_jac_median']:.1f} s, gain {f['t_jac_median'] / u['t_jac_median']:.3f} "
              f"(estimated Taylor-block cost ratio {u['cost'] / f['cost']:.3f}); Taylor rows max|F| {f['F_taylor']:.1e} -> "
              f"{u['F_taylor']:.1e}: {'pass' if F_ok else 'FAIL'}; auto cost <= fixed: {'pass' if c_ok else 'FAIL'}")
    if not byo and not any(r.get("cmd") == "ab" for r in recs):
        print("report: no ab or abo records: FAIL")
        return 1
    recs = [r for r in recs if r.get("cmd") == "ab"]
    by = {}
    for r in recs:
        by.setdefault(r["state"], {})[r["arm"]] = r
    print(f"{'state':42s} {'arm':5s} {'kt':>3s} {'steps':>6s} {'cost':>9s} {'|F| after':>9s} {'call s':>8s}")
    for st, arms in by.items():
        for name in ("fixed", "auto"):
            r = arms.get(name)
            if r:
                print(f"{os.path.basename(st)[:42]:42s} {name:5s} {r['kt']:3d} {r['steps']:6d} {r['cost']:9.3e} "
                      f"{r['F1']:9.2e} {r['t_call']:8.1f}")
        if "fixed" not in arms or "auto" not in arms:
            print("  one arm missing: FAIL")
            ok = False
            continue
        f, u = arms["fixed"], arms["auto"]
        # the control's move goes on the lo part: added to the hi part, 1e-18 would vanish below its ulp
        ul = u["Delta"][1] + (1e-18 if a.mutate == "delta" else 0.0)
        dD = (u["Delta"][0] - f["Delta"][0]) + (ul - f["Delta"][1])
        c_ok = u["cost"] <= f["cost"]
        d_ok = abs(dD) <= TOL_DELTA
        F_ok = max(u["F1"], f["F1"]) < a.tol
        gain = f["t_call"] / u["t_call"]
        ok = ok and c_ok and d_ok and F_ok
        print(f"  gain (fixed call s / auto call s) {gain:.3f}, estimated {f['cost'] / u['cost']:.3f}; Delta difference "
              f"{dD:+.1e} (bound {TOL_DELTA:.0e}): {'pass' if d_ok else 'FAIL'}; |F| after below {a.tol:.0e}: "
              f"{'pass' if F_ok else 'FAIL'}; auto cost <= fixed: {'pass' if c_ok else 'FAIL'}")
    return 0 if ok else 1


def selftest(a):
    """report on planted records (no kernel): a passing pair of each kind passes, and each planted fault fails"""
    import tempfile

    def ab_pair(dD=0.0, cost_auto=0.9, F1=1e-30):
        base = dict(cmd="ab", state="s.npz", D=3.0, M=320, w=20, piv=[1e-3, 1.0], threads=1)
        return [dict(base, arm="fixed", kt=36, steps=100, cost=1.0, F0=1e-21, F1=1e-30, Delta=[1.68, 1e-17], t_call=10.0),
                dict(base, arm="auto", kt=40, steps=80, cost=cost_auto, F0=1e-21, F1=F1, Delta=[1.68, 1e-17 + dD],
                     t_call=9.0)]

    def abo_pair(F_auto=5e-31, cost_auto=0.6):
        base = dict(cmd="abo", state="c.npz", D=256.5, M=32, w=20, threads=1)
        return [dict(base, arm="fixed", kt=36, steps=23, cost=1.0, F_taylor=6e-31, t_jac_median=100.0),
                dict(base, arm="auto", kt=14, steps=54, cost=cost_auto, F_taylor=F_auto, t_jac_median=90.0)]

    cases = [("both pass", ab_pair() + abo_pair(), None, 0), ("report --mutate delta", ab_pair(), "delta", 1),
             ("Delta 1e-19 apart", ab_pair(dD=1e-19), None, 1), ("auto costs more", ab_pair(cost_auto=1.1), None, 1),
             ("|F| after 1e-27", ab_pair(F1=1e-27), None, 1), ("Taylor rows 1e-29", abo_pair(F_auto=1e-29), None, 1),
             ("abo auto costs more", abo_pair(cost_auto=1.2), None, 1), ("an arm missing", ab_pair()[:1], None, 1),
             ("no records", [], None, 1)]
    ok = True
    for name, recs, mut, want in cases:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("".join(json.dumps(r) + "\n" for r in recs))
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            got = report(argparse.Namespace(files=[f.name], tol=1e-28, mutate=mut))
        os.unlink(f.name)
        ok = ok and got == want
        print(f"selftest {name}: exit {got} (want {want}): {'pass' if got == want else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("planted")
    p.add_argument("--check", default=None, help="compare with this stored file (exit 1 on a difference)")
    p.add_argument("--write", default=None)
    p.add_argument("--mutate", default=None, choices=["n"])
    p = sub.add_parser("ab")
    p.add_argument("state")
    p.add_argument("--out", required=True)
    p.add_argument("--w", type=int, default=20)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tol_step", type=float, default=1e-30)
    p.add_argument("--hmax", type=float, default=0.05)
    p.add_argument("--mutate", default=None, choices=["invert"])
    p.add_argument("--kmax", type=int, default=0, help="no order above this (0: no bound; the controls take 96)")
    p = sub.add_parser("abo")
    p.add_argument("state")
    p.add_argument("--out", required=True)
    p.add_argument("--w", type=int, default=20)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--tol_step", type=float, default=1e-32)
    p.add_argument("--hmax", type=float, default=0.05)
    p.add_argument("--mutate", default=None, choices=["invert"])
    p.add_argument("--kmax", type=int, default=0, help="no order above this (0: no bound; the controls take 96)")
    sub.add_parser("selftest")
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    p.add_argument("--tol", type=float, default=1e-28)
    p.add_argument("--mutate", default=None, choices=["delta"])
    a = ap.parse_args()
    return dict(planted=planted, ab=ab, abo=abo, report=report, selftest=selftest)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
