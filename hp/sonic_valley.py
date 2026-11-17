#!/usr/bin/env python3
"""The sonic valley of the low-D states: how close c comes to x inside the domain (notes/LOW_D.md 17).

The p equation, x (x - c) p_x = -omega x p_tau + G p - c p + (n/2) c (p + m), is singular where x = c. Near D 3.04
the converged states keep c above x, but min over tau of c - x has a narrow valley at x ~ 0.0143 in the segment
0.01 -> 0.02, whose floor, the gap g = min c - x, closes linearly as D falls. The x integration through it is stiff
(its rates scale like 1 / (c - x)), so a Taylor plan made for one gap blows up at a smaller one, and at g = 0 the
integration meets the singular point itself.

    cd hp && python3 sonic_valley.py gap OUT.jsonl STATE.npz [STATE.npz ...] [--D D] [--plan stored|own|FILE.json]
        [--refine R] [--below 1e-2] [--threads 4] [--mutate none|dshift] [--predict D_target]
    cd hp && python3 sonic_valley.py tangency OUT.jsonl [--M M] [--last N] [--mutate none|slope]
    cd hp && python3 sonic_valley.py planted [--mutate none|grid|slope]
    cd hp && python3 sonic_valley.py planplant OUT.jsonl TRUTH.npz COARSE.npz [--refine 4] [--levels 2]
        [--delta 1e-8] [--threads 4] [--mutate none|norefine]
    cd hp && python3 sonic_valley.py startcase OUT.jsonl STATE.npz [STATE.npz ...] --predict D [--plan own|stored|FILE]
        [--tol_step 1e-32] [--save_plan FILE.json] [--refine 4] [--levels 2] [--threads 4]

gap: each state's node states are integrated through every segment whose end nodes have min(c - x) below --below
(default 1e-2), with the kernel's Taylor stepper, step by step along the plan (stored: the state's own "segs"; own:
planned from the state at --tol_step 1e-32; FILE.json: segs saved as JSON), each step subdivided --refine times. One
record per state and segment: the smallest c - x over the grid points and over tau between them (trigonometric
interpolation of c, an even field with period pi on the half grid), where, and whether the integration stayed finite.
--D integrates with the equations at another D (a start for the step to D); without it, at the state's own D, the
integrated end must reproduce the next node to the state's residual (check end_ok, 1e-20; exit 1 otherwise), and
--mutate dshift, the equations at D + 1e-4, must make that check fail. --predict D_target: one record instead, for the
leg's predictor (hpkernel.driver.predict, Lagrange through the given states, oldest first) at D_target with its
equations, on the last state's plan.
tangency: a straight line through (D, gap) of the records of one M (the grid gap of each state's narrowest segment;
--last N: the N of smallest D) and its zero, the D where the valley closes; with 4 or more records also the
quadratic's zero. Exit 1 when the line's largest residual exceeds 1e-3 of the gaps' range (the extrapolation is then
not a line). --mutate slope adds 100 (D - D_min)^2 to every gap before the fit (the M 320 gaps' own quadratic
coefficient is -0.13 to -0.65) and must make it fail on real records spaced 2e-4 or more.
planted: the estimators on planted truth, no kernel. (1) cont_min on c = x + eps + g sin^2(tau - tau0) with tau0
between grid points returns eps to 1e-12 and tau0 to 1e-6 while the grid minimum is above eps; (2) the line through
planted gaps s (D - D_star) with noise 1e-9 returns D_star to 1e-6; (3) a planted quadratic is flagged as not a
line, and the quadratic's zero is its root. Mutations: grid (cont_min returns the grid minimum), slope (the line's zero
taken at gap 1e-5 instead of 0) must each fail it. Exit 1 on failure.
planplant: the refinement near x = c on planted truth, as lowD_highM.py --sonic_refine uses it (a gate, then the
admitted start's own plan). TRUTH is a converged state; COARSE a state at the same M and nodes whose gap is wider (a
larger D), whose plan is too coarse for TRUTH's valley. Full residuals: TRUTH on its own plan (the reference) and on
COARSE's plan; TRUTH with every node state moved by delta (relative to each field's largest value, random signs) on
COARSE's plan with the steps of the segments near x = c split R, R^2, ... times (refine_near) until finite (the gate);
TRUTH on that split plan (information: its truncation error); and the perturbed TRUTH on its own plan, planned from it.
Pass: COARSE's plan does not fit TRUTH (|F| nan or above 1e-20), the gate admits the perturbed TRUTH, TRUTH is finite
on the split plan, and the perturbed TRUTH's residual on its own plan lies between delta / 100 and 1e4 delta.
--mutate norefine (no split: the gate then admits nothing and no own plan is made) must make it fail. Exit 1 on
failure. (Its first form also required TRUTH's residual on the split plan within 100 times its own: at 3.04335 on the
plan of 3.0475 split 16 times it was 2.9e-21 against 1.2e-28, so lowD_highM.py converges on no split plan.)
startcase: lowD_highM.py's two starts at the target --predict from the given converged states (oldest first), the
predictor and the last state, read on the reused plan (--plan own: the last state's plan at its own D and --tol_step,
as lowD_highM's plan_prev makes it) and on it with the steps near x = c split R, R^2, ... times. A record per start
and split; no check (a measurement for #926's real case).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DSS_SOLVER_DIR", os.path.normpath(os.path.join(HERE, "..", "solver")))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from raw_out import FLAG, HELP, require_raw  # noqa: E402


def cont_min(c, pad=32, mutate=""):
    """(min over tau, tau index of it) of an even field c on the half grid (M samples of one period), by zero-padded
    trigonometric interpolation (pad times the samples; the M/2 harmonic split between +-M/2)"""
    M = len(c)
    if mutate == "grid":
        j = int(np.argmin(c))
        return float(c[j]), float(j)
    C = np.fft.rfft(c)
    Np = M * pad
    Cp = np.zeros(Np // 2 + 1, complex)
    Cp[:len(C)] = C
    if M % 2 == 0:
        Cp[M // 2] *= 0.5
    f = np.fft.irfft(Cp, n=Np) * pad
    j = int(np.argmin(f))
    # refine by a parabola through the three samples around the minimum
    fm, f0, fp = f[j - 1], f[j], f[(j + 1) % Np]
    den = fm - 2 * f0 + fp
    dj = 0.5 * (fm - fp) / den if den > 0 else 0.0
    return float(f0 - 0.25 * (fm - fp) * dj), (j + dj) / pad


def node_gap(z, M, nodes, i):
    """min over the grid of c - x at node i"""
    o = 2 * M + 4 * M * i
    return float(np.min(z[o + 3 * M:o + 4 * M] - nodes[i]))


def valley(s, zh, zl, nodes, M, b, xs, pad=32):
    """integrate node b's state of z (zh, zl) along the step points xs (xs[0] = nodes[b]) with the solver s; the smallest
    c - x on the grid and between grid points, where, and whether the path stayed finite"""
    o = 2 * M + 4 * M * b
    sh, sl = np.array(zh[o:o + 4 * M]), np.array(zl[o:o + 4 * M])
    out = dict(seg=b, x0=xs[0], x1=xs[-1], steps=len(xs) - 1, finite=True, x_fail=None)
    best = dict(gap=np.inf, x=None, i=None, gap_cont=np.inf, x_cont=None, tau_cont=None)
    st = (sh + sl).reshape(4, M)
    for j in range(len(xs)):
        if j:
            r = s.integrate(zh, zl, sh, sl, xs[j - 1:j + 1])
            sh, sl = np.asarray(r[0]), np.asarray(r[1])
            st = (sh + sl).reshape(4, M)
            if not np.all(np.isfinite(st)):
                out.update(finite=False, x_fail=xs[j])
                break
        d = st[3] - xs[j]
        i = int(np.argmin(d))
        if d[i] < best["gap"]:
            best.update(gap=float(d[i]), x=xs[j], i=i)
        cm, tc = cont_min(st[3], pad)
        if cm - xs[j] < best["gap_cont"]:
            best.update(gap_cont=cm - xs[j], x_cont=xs[j], tau_cont=tc)
    out.update(best)
    if out["finite"] and b + 1 < len(nodes):
        # in double-double: (hi - hi') + (lo - lo'), not a difference of the rounded sums (1 ulp of f64 is 1e-16)
        out["end_mismatch"] = float(np.abs((sh - zh[o + 4 * M:o + 8 * M]) + (sl - zl[o + 4 * M:o + 8 * M])).max())
    return out


def refine(xs, r):
    """each step of xs split into r equal steps"""
    if r <= 1:
        return list(xs)
    return [float(v) for a_, b_ in zip(xs[:-1], xs[1:]) for v in np.linspace(a_, b_, r + 1)[:-1]] + [xs[-1]]


def valley_segments(z, M, nodes, below=1e-2):
    """the segments b (node b -> b + 1) with min(c - x) below `below` at either end node"""
    return [b for b in range(len(nodes) - 1)
            if min(node_gap(z, M, nodes, b), node_gap(z, M, nodes, b + 1)) <= below]


def refine_near(segs, z, M, nodes, r, below=1e-2):
    """the plan segs with each step of the segments near x = c (valley_segments of z) split r times; the stiffness
    of the x integration there scales like 1 / (c - x), so a plan made for a wider gap is too coarse at a narrower one
    (notes/LOW_D.md 17)"""
    near = set(valley_segments(z, M, nodes, below))
    return [refine(g, r) if b in near else list(g) for b, g in enumerate(segs)]


def fit_line(Ds, ys):
    """least-squares line y = a (D - Dm) + b through the points; (D where y = 0, slope, largest residual)"""
    Ds, ys = np.asarray(Ds, float), np.asarray(ys, float)
    Dm = Ds.mean()
    a, b = np.polyfit(Ds - Dm, ys, 1)
    res = ys - (a * (Ds - Dm) + b)
    return float(Dm - b / a), float(a), float(np.abs(res).max())


def fit_quad(Ds, ys):
    """the quadratic's zero nearest the points' smallest D (None without a real zero)"""
    Ds, ys = np.asarray(Ds, float), np.asarray(ys, float)
    Dm = Ds.mean()
    c2, c1, c0 = np.polyfit(Ds - Dm, ys, 2)
    r = np.roots([c2, c1, c0])
    r = [float(v.real) + Dm for v in r if abs(v.imag) < 1e-14]
    return min(r, key=lambda v: abs(v - Ds.min())) if r else None


def gap_of(r):
    """a record's grid gap ("depth" in the records of the first run, I926-mac/depth_M320.jsonl)"""
    return r["gap"] if "gap" in r else r["depth"]


def tangency(recs, mutate="", last=0):
    """the line's zero through (D, grid gap) of records of one M (a state's narrowest segment each; only the `last`
    states of smallest D, when given)"""
    by = {}
    for r in recs:
        if r.get("finite") and r.get("D_eq") == r.get("D") and not r.get("predicted") and not r.get("mutate"):
            k = (r["state"], r["D"])
            if k not in by or gap_of(r) < gap_of(by[k]):
                by[k] = r
    pts = sorted((r["D"], gap_of(r)) for r in by.values())
    if last:
        pts = pts[:last]
    Ds = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if mutate == "slope":
        ys = ys + 100.0 * (Ds - Ds.min()) ** 2
    Dz, a, res = fit_line(Ds, ys)
    out = dict(points=[[float(d), float(y)] for d, y in zip(Ds, ys)], D_star=Dz, slope=a, max_resid=res,
               rel_resid=res / max(np.ptp(ys), 1e-300))
    if len(pts) >= 4:
        out["D_star_quad"] = fit_quad(Ds, ys)
    return out


def planted(mutate=""):
    """the estimators on planted truth (the module's docstring); returns the list of failures"""
    fails = []
    # (1) cont_min: a valley between grid points
    M = 320
    tau = np.pi * np.arange(M) / M
    for tau0, eps, g in ((0.2345678, 1.3e-5, 2.0), (1.0 + 0.4 * np.pi / M, -2e-6, 0.7)):
        x = 0.0143
        c = x + eps + g * np.sin(tau - tau0) ** 2
        cm, tc = cont_min(c, mutate=mutate)
        ok = abs(cm - x - eps) < 1e-12 and abs(tc * np.pi / M - tau0) < 1e-6 and np.min(c) - x > eps
        print(f"  planted valley eps {eps:.1e} tau0 {tau0:.7f}: cont_min {cm - x:.6e} at tau {tc * np.pi / M:.7f}, "
              f"grid min {np.min(c) - x:.6e}: {'pass' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"cont_min eps {eps}")
    # (2) the line's zero
    rng = np.random.default_rng(1)
    Ds = np.array([3.04375, 3.04355, 3.04335])
    Dstar, sl = 3.04312, 0.057
    ys = sl * (Ds - Dstar) + 1e-9 * rng.uniform(-1, 1, len(Ds))
    Dz, a, res = fit_line(Ds, ys - (1e-5 if mutate == "slope" else 0.0))
    ok = abs(Dz - Dstar) < 1e-6 and abs(a - sl) < 1e-4
    print(f"  planted line: D_star {Dz:.7f} (planted {Dstar}), slope {a:.5f} (planted {sl}): {'pass' if ok else 'FAIL'}")
    if not ok:
        fails.append("line")
    # (3) a quadratic is not a line: its residual is flagged
    Ds4 = np.array([3.0450, 3.04375, 3.04355, 3.04335])
    yq = sl * (Ds4 - Dstar) + 50.0 * (Ds4 - Dstar) ** 2
    _, _, resq = fit_line(Ds4, yq)
    rel = resq / np.ptp(yq)
    ok = rel > 1e-3 and abs(fit_quad(Ds4, yq) - Dstar) < 1e-9
    print(f"  planted quadratic: line residual {rel:.1e} of the range (must exceed 1e-3), quadratic zero "
          f"{fit_quad(Ds4, yq):.7f}: {'pass' if ok else 'FAIL'}")
    if not ok:
        fails.append("quadratic")
    return fails


def planplant(a, out):
    """a converged state (the truth) on the plan of a state with a shallower valley: its residual there, unrefined and
    with the steps near x = c split R, R^2, ... times; and the truth with its node states perturbed by delta on the
    refined plan and on the coarse one (the module's docstring). Returns the exit code."""
    from hpkernel.driver import load, make_solver
    zh, zl, m = load(a.args[1])
    zh, zl = np.asarray(zh), np.asarray(zl)
    mc = load(a.args[2])[2]
    M, nodes = m["M"], list(m["nodes"])
    if list(mc["nodes"]) != nodes or mc["M"] != M:
        raise SystemExit("the truth and the plan's state must have the same M and nodes")
    s = make_solver(m["D"], M, nodes=nodes, kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=a.threads)
    z = zh + zl

    def res(segs, h, l, what):
        s.set_segs(segs)
        t0 = time.time()
        F = np.abs(np.add(*s.residual(h, l)))
        r = dict(what=what, F=float(F.max()), steps=sum(len(g) - 1 for g in segs), seconds=round(time.time() - t0, 1),
                 truth=os.path.basename(a.args[1]), D=m["D"], plan_of=os.path.basename(a.args[2]), D_plan=mc["D"], M=M,
                 mutate=a.mutate or None)
        print(f"  {what}: |F| {r['F']:.3e} on {r['steps']} steps ({r['seconds']} s)", flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        return r

    near = [(nodes[b], nodes[b + 1]) for b in valley_segments(z, M, nodes, a.below)]
    print(f"truth {os.path.basename(a.args[1])} (D {m['D']}), plan of {os.path.basename(a.args[2])} (D {mc['D']}), "
          f"segments near x = c: {near}", flush=True)
    own = res(m["segs"], zh, zl, "truth, own plan")
    coarse = res(mc["segs"], zh, zl, "truth, coarse plan")
    # the planted perturbation: every node state (not the centre or horizon data, nor Delta) moved by
    # delta * u * max|field| with u uniform in +-[0.5, 1]
    rng = np.random.default_rng(1)
    ph = zh.copy()
    for b in range(len(nodes)):
        for f_ in range(4):
            o = 2 * M + 4 * M * b + f_ * M
            ph[o:o + M] += a.delta * rng.uniform(0.5, 1, M) * rng.choice([-1, 1], M) * np.abs(z[o:o + M]).max()
    # the gate, as lowD_highM.py --sonic_refine takes it: the coarse plan split R, R^2, ... times near x = c until the
    # perturbed truth is finite on it
    gate, r, sg = None, 1, mc["segs"]
    for lev in range(1, a.levels + 1):
        r = 1 if a.mutate == "norefine" else a.refine ** lev
        sg = refine_near(mc["segs"], z, M, nodes, r, a.below)
        gate = res(sg, ph, zl, f"truth + {a.delta:g}, coarse plan split {r} near x = c")
        if np.isfinite(gate["F"]):
            break
    # the truth on the gate's plan: information (the split plan's own truncation error)
    fine = res(sg, zh, zl, f"truth, coarse plan split {r} near x = c")
    pert_own = None
    if np.isfinite(gate["F"]):
        # an admitted start gets a plan of its own (lowD_highM.py plans it in its two passes)
        s.set_segs(mc["segs"])
        t0 = time.time()
        sp = s.plan(ph, zl, a.tol_step, a.hmax)
        print(f"  truth + {a.delta:g}: own plan {sum(len(g) - 1 for g in sp)} steps ({time.time() - t0:.0f} s)",
              flush=True)
        pert_own = res(sp, ph, zl, f"truth + {a.delta:g}, its own plan")
    else:
        print(f"  truth + {a.delta:g}: not finite on the split plan, so no plan of its own (lowD_highM.py plans no nan "
              f"start)", flush=True)
    fails = []
    if np.isfinite(coarse["F"]) and coarse["F"] < 1e-20:
        fails.append("the coarse plan already fits the truth (no planted case)")
    if not np.isfinite(gate["F"]):
        fails.append(f"the perturbed truth is not finite on the coarse plan split {r} times near x = c")
    if not (np.isfinite(fine["F"])):
        fails.append(f"the truth is not finite on the coarse plan split {r} times near x = c")
    if pert_own is None or not (np.isfinite(pert_own["F"]) and 0.01 * a.delta <= pert_own["F"] <= 1e4 * a.delta):
        fails.append("the perturbed truth's residual on its own plan is not of the size of delta ("
                     + (f"{pert_own['F']:.1e}" if pert_own else "no plan") + ")")
    print(f"  (information: the truth's residual on the split plan is {fine['F']:.1e}, on its own {own['F']:.1e})")
    print("planplant: " + ("pass" if not fails else "FAIL: " + "; ".join(fails)), flush=True)
    return 1 if fails else 0


def startcase(a, out):
    """lowD_highM.py's starts at the target --predict from the given converged states (oldest first): the predictor
    (Lagrange through them) and the last state, read on the reused plan (the last state's plan at its own D with
    step tolerance --tol_step, lowD_highM's --tol_step 1e-30 times --plan_margin 1e-2) and on it with the steps near
    x = c split R, R^2, ... times (refine_near); --plan FILE.json takes the reused plan from a file instead of planning
    it, and --save_plan FILE.json keeps it. Returns 0."""
    from hpkernel.driver import load, make_solver, predict
    hist, last = [], None
    for path in a.args[1:]:
        h_, l_, m = load(path)
        hist.append((m["D"], np.asarray(h_) + np.asarray(l_)))
        last = (np.asarray(h_), np.asarray(l_), m)
    lh, ll, m = last
    M, nodes = m["M"], list(m["nodes"])
    if a.plan not in ("stored", "own"):
        segs = json.load(open(a.plan))
    elif a.plan == "stored":
        segs = m["segs"]
    else:
        t0 = time.time()
        segs = make_solver(m["D"], M, nodes=nodes, kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=a.threads).plan(
            lh, ll, a.tol_step, a.hmax)
        print(f"reused plan (the last state's at D {m['D']}, tol {a.tol_step:g}): {sum(len(g) - 1 for g in segs)} "
              f"steps ({time.time() - t0:.0f} s)", flush=True)
    if a.save_plan:
        json.dump(segs, open(a.save_plan, "w"))
    zp = predict(hist, a.predict)
    # as lowD_highM.py forms them: f64 vectors (the last state's hi + lo rounded), lower parts zero
    starts = [("predictor", np.asarray(zp), np.zeros(len(zp))), ("last state", lh + ll, np.zeros(len(lh)))]
    s = make_solver(a.predict, M, nodes=nodes, kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=a.threads)
    for lev in range(0, a.levels + 1):
        r = a.refine ** lev
        sg = refine_near(segs, lh + ll, M, nodes, r, a.below)
        s.set_segs(sg)
        for lab, h_, l_ in starts:
            t0 = time.time()
            F = float(np.abs(np.add(*s.residual(h_, l_))).max())
            rec = dict(start=lab, D=a.predict, history=[d for d, _ in hist], M=M, split=r,
                       steps=sum(len(g) - 1 for g in sg), F=F, seconds=round(time.time() - t0, 1))
            print(f"  D {a.predict} M {M}: start '{lab}' |F| {F:.3e} on the reused plan split {r} near x = c "
                  f"({rec['steps']} steps, {rec['seconds']} s)", flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gap", "tangency", "planted", "planplant", "startcase"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--D", type=float, default=None)
    ap.add_argument("--predict", type=float, default=None)
    ap.add_argument("--plan", default="stored")
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--below", type=float, default=1e-2)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--mutate", default="")
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--delta", type=float, default=1e-8)
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--save_plan", default=None)
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    if a.cmd == "planplant":
        require_raw(a.args[0], allow=a.allow_unraw)
        return planplant(a, a.args[0])
    if a.cmd == "startcase":
        require_raw(a.args[0], allow=a.allow_unraw)
        return startcase(a, a.args[0])
    if a.cmd == "planted":
        f = planted(a.mutate)
        print("planted: " + ("pass" if not f else "FAIL: " + ", ".join(f)))
        return 1 if f else 0
    if a.cmd == "tangency":
        recs = [json.loads(ln) for ln in open(a.args[0]) if ln.strip()]
        if a.M:
            recs = [r for r in recs if r.get("M") == a.M]
        t = tangency(recs, a.mutate, a.last)
        print(json.dumps(t, indent=1))
        ok = t["rel_resid"] <= 1e-3
        print(f"tangency: D_star {t['D_star']:.6f} from {len(t['points'])} states, slope {t['slope']:.5f}, line residual "
              f"{t['rel_resid']:.1e} of the range: {'a line' if ok else 'NOT a line'}")
        return 0 if ok else 1
    from hpkernel.driver import load, make_solver, predict
    out = a.args[0]
    require_raw(out, allow=a.allow_unraw)
    bad = 0
    jobs = []
    if a.predict is not None:
        # the leg's predictor (lowD_highM.py: Lagrange through the given states, oldest first) at --predict, with
        # the last state's meta (nodes, plan)
        hist = []
        for path in a.args[1:]:
            h_, l_, m = load(path)
            hist.append((m["D"], np.asarray(h_) + np.asarray(l_)))
        zp = predict(hist, a.predict)
        lab = "predictor(" + ",".join(f"{d:g}" for d, _ in hist) + f")->{a.predict:g}"
        jobs.append((lab, np.asarray(zp), np.zeros(len(zp)), dict(m, D=a.predict)))
    else:
        for path in a.args[1:]:
            zh, zl, m = load(path)
            jobs.append((os.path.basename(path), np.asarray(zh), np.asarray(zl), m))
    for path, zh, zl, m in jobs:
        M, nodes = m["M"], list(m["nodes"])
        D_eq = a.D if a.D is not None else m["D"]
        if a.mutate == "dshift":
            D_eq += 1e-4
        s = make_solver(D_eq, M, nodes=nodes, kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=a.threads)
        if a.plan == "stored":
            segs = m["segs"]
        elif a.plan == "own":
            t0 = time.time()
            segs = s.plan(zh, zl, a.tol_step, a.hmax)
            print(f"{path}: own plan at D {D_eq}: {sum(len(g) - 1 for g in segs)} steps "
                  f"({time.time() - t0:.0f} s)", flush=True)
        else:
            segs = json.load(open(a.plan))
        z = zh + zl
        for b in valley_segments(z, M, nodes, a.below):
            t0 = time.time()
            r = valley(s, zh, zl, nodes, M, b, refine(segs[b], a.refine))
            r.update(state=path, D=m["D"], D_eq=D_eq, M=M, plan=a.plan,
                     plan_steps=sum(len(g) - 1 for g in segs), refine=a.refine, seconds=round(time.time() - t0, 1),
                     mutate=a.mutate or None, predicted=a.predict is not None)
            if a.D is None and a.predict is None and r["finite"] and b + 1 < len(nodes):
                r["end_ok"] = r["end_mismatch"] <= 1e-20
                bad += not r["end_ok"]
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
            dc = f", between grid points {r['gap_cont']:.6e} at x {r['x_cont']:.8f} tau index {r['tau_cont']:.3f}" \
                if r["gap_cont"] is not None else ""
            print(f"{r['state']} D {m['D']} (equations at {D_eq}) M {M} segment {nodes[b]} -> {nodes[b + 1]}, "
                  f"{r['steps']} steps: " + (f"min c - x {r['gap']:.6e} at x {r['x']:.8f} grid point {r['i']}{dc}"
                                             if r["finite"] else f"NOT finite at x {r['x_fail']:.8f} (last min c - x "
                                             f"{r['gap']:.3e} at x {r['x']:.8f})")
                  + (f"; end mismatch {r['end_mismatch']:.1e}" if "end_mismatch" in r else "") + f" ({r['seconds']} s)",
                  flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
