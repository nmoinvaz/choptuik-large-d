#!/usr/bin/env python3
"""J2 of notes/QUAD_DOUBLE.md 13.15: the residual's Taylor coefficients in the dimension (Solver.residual_jet, the jet
lanes of multidouble with n + h as the jet variable) against the residual itself evaluated at D + h, on one stored
state z held fixed. At fixed data the residual is not analytic far in n: the horizon series' data are then wrong for
n + h by O(h), and by the radius law (13.6) its apparent horizon enters the domain once h exceeds the basin, so F(z,
D + h) has a radius of about 1e-7 at D 73 (the branch z(n) itself has about 50, 13.12). The comparison is therefore
at h = 1e-14, 1e-13, 1e-12, inside the basin (the basin is 2.6e-12 at D 73 for the 0.7 layout, 13.6), where
sum_{k<=N} F_k h^k differs from F(z, D + h) by the arithmetic floor alone; the terms h^k F_k then test the orders up
to about 4 (at D 73 the coefficients grow by 1e7 per order, so the sum is the low orders' only). Pass: the difference at most 1e-20 of |F_1 h| at every h;
the central differences of orders 1 and 2 at the smallest h agree with F_1 and 2 F_2 to their eps^2 terms; the
controls (model.rs JET_MUT_KAPPA, kappa's n frozen; JET_MUT_FREEZE_N, the dimension frozen) fail the same comparison.

    python3 jet_check.py RAWDIR OUT.jsonl [--state T75/pin7_D73.2657.npz] [--N 16] [--hs 1e-14,1e-13,1e-12] [--threads 4] [--kfreeze 5]

RAWDIR is a clone of the raw-data repository (the state's path is relative to it); OUT.jsonl must lie in a task
folder of that repository (hp/raw_out.py). Exit status 1 when the check fails or a control does not fire.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import mpmath as mp  # noqa: E402
from hpkernel.driver import make_solver, rnorm  # noqa: E402
from raw_out import require_raw  # noqa: E402

JET_MUT_FREEZE_N, JET_MUT_KAPPA = 21, 22
TOL = 1e-20


def load_state(path):
    d = np.load(path)
    m = json.loads(str(d["meta"]))
    return d["zh"], d["zl"], m


def to_mp(h, l):
    return [mp.mpf(float(a)) + mp.mpf(float(b)) for a, b in zip(h, l)]


def norm_mp(v):
    return mp.sqrt(sum(x * x for x in v))


def main():
    args = list(sys.argv[1:])
    state, N, hs, threads, kfreeze = "T75/pin7_D73.2657.npz", 16, (1e-14, 1e-13, 1e-12), 4, 0
    for flag in ("--state", "--N", "--hs", "--threads", "--kfreeze"):
        if flag in args:
            i = args.index(flag); v = args[i + 1]; del args[i:i + 2]
            if flag == "--state":
                state = v
            elif flag == "--N":
                N = int(v)
            elif flag == "--hs":
                hs = tuple(float(x) for x in v.split(","))
            elif flag == "--kfreeze":
                kfreeze = int(v)
            else:
                threads = int(v)
    raw, out = args[0], args[1]
    require_raw(out)
    zh, zl, m = load_state(os.path.join(raw, state))
    D, M, nodes = float(m["D"]), int(m["M"]), list(m["nodes"])
    kt, kc, ks = m.get("kt", 36), m.get("kc", 146), m.get("ks", 130)
    kf = kfreeze or (13 if D >= 35 else None)
    mp.mp.dps = 50
    # D + h as an mpmath number: a float would round h = 1e-12 at D 73 to 1e-14 (make_solver forms n in double-double
    # from whatever precision it is given)
    mk = lambda Dx: make_solver(Dx, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=threads, kfreeze=kf)  # noqa: E731
    s0 = mk(mp.mpf(D))
    segs = s0.plan(zh, zl, 1e-32, 0.05)
    s0.set_segs(segs)
    t0 = time.time()
    Fh0, Fl0 = s0.residual(zh, zl)
    t_res = time.time() - t0
    print(f"{state}: D {D}, M {M}, nodes {nodes}, |F| {rnorm(Fh0, Fl0):.2e}; residual {t_res:.2f} s", flush=True)
    mp.mp.dps = 50
    rec = dict(state=state, D=D, M=M, N=N, hs=list(hs), kfreeze=kf, F0=float(rnorm(Fh0, Fl0)), t_residual=t_res)

    def coefficients(mutate):
        t0 = time.time()
        Hs, Ls = s0.residual_jet([zh], [zl], N + 1, mutate)
        return [to_mp(h, l) for h, l in zip(Hs, Ls)], time.time() - t0

    Fk, t_jet = coefficients(0)
    rec.update(t_jet=t_jet, coef_norms=[float(norm_mp(c)) for c in Fk])
    print(f"jet residual, {N + 1} coefficients: {t_jet:.2f} s ({t_jet / t_res:.0f} residuals); |F_k|: "
          + " ".join(f"{float(norm_mp(c)):.2e}" for c in Fk), flush=True)
    # the value coefficient against the plain residual
    d0 = norm_mp([a - b for a, b in zip(Fk[0], to_mp(Fh0, Fl0))])
    rec["value_vs_residual"] = float(d0 / max(norm_mp(Fk[0]), mp.mpf(1e-300)))
    print(f"coefficient 0 against the plain residual: {rec['value_vs_residual']:.1e} relative", flush=True)

    # the residual at D + h for every h and -h, with the same segment plan
    direct = {}
    for h in sorted(set([x for h in hs for x in (h, -h)])):
        s = mk(mp.mpf(D) + mp.mpf(h))
        s.set_segs(segs)
        Fh, Fl = s.residual(zh, zl)
        direct[h] = to_mp(Fh, Fl)

    def compare(Fk, label):
        rows = []
        worst = 0.0
        for h in direct:
            hm = mp.mpf(h)
            total = [mp.mpf(0)] * len(Fk[0])
            for k, c in enumerate(Fk):
                p = hm ** k
                total = [t + x * p for t, x in zip(total, c)]
            diff = norm_mp([a - b for a, b in zip(total, direct[h])])
            scale = norm_mp(Fk[1]) * abs(hm)
            trunc = norm_mp(Fk[-1]) * abs(hm) ** (len(Fk) - 1)
            ratio = float(diff / scale) if scale > 0 else (float("inf") if diff > 0 else 0.0)
            worst = max(worst, ratio)
            rows.append(dict(h=h, diff=float(diff), scale=float(scale), ratio=ratio, last_term=float(trunc)))
            print(f"  {label} h {h:+.1e}: |sum - F(D+h)| {float(diff):.2e}, |F_1 h| {float(scale):.2e}, ratio {ratio:.1e}, "
                  f"last term {float(trunc):.1e}", flush=True)
        return rows, worst

    rows, worst = compare(Fk, "jet")
    rec.update(rows=rows, worst_ratio=worst, passed=bool(worst <= TOL))
    print(f"worst ratio {worst:.1e} against {TOL:.0e}: {'PASS' if rec['passed'] else 'FAIL'}", flush=True)

    # central differences, orders 1 and 2, at the smallest h: the cruder cross-check
    e = min(hs)
    Fp, Fm, F0m = direct[e], direct[-e], to_mp(Fh0, Fl0)
    em = mp.mpf(e)
    d1 = [(a - b) / (2 * em) for a, b in zip(Fp, Fm)]
    d2 = [(a - 2 * c + b) / (em * em) for a, b, c in zip(Fp, Fm, F0m)]
    r1 = float(norm_mp([a - b for a, b in zip(d1, Fk[1])]) / norm_mp(Fk[1]))
    r2 = float(norm_mp([a - 2 * b for a, b in zip(d2, Fk[2])]) / (2 * norm_mp(Fk[2])))
    # expected: eps^2 |F_3| / |F_1| and eps^2 |F_4| / |F_2| (the next terms of the central differences)
    x1 = float(em * em * norm_mp(Fk[3]) / norm_mp(Fk[1])) if N >= 3 else None
    x2 = float(em * em * norm_mp(Fk[4]) / norm_mp(Fk[2])) if N >= 4 else None
    rec.update(central=dict(eps=e, rel1=r1, rel2=r2, expect1=x1, expect2=x2))
    print(f"central differences at eps {e}: order 1 relative {r1:.1e} (eps^2 |F_3|/|F_1| {x1:.1e}), "
          f"order 2 relative {r2:.1e} (eps^2 |F_4|/|F_2| {x2:.1e})", flush=True)
    # the floor of the differences: the residual's rounding (about 1e-30 of its terms) over eps |F_1|, eps^2 |F_2|
    f1 = float(mp.mpf(1e-30) * norm_mp(Fk[0] if norm_mp(Fk[0]) > 1 else [mp.mpf(1)]) / (em * norm_mp(Fk[1])))
    f2 = float(mp.mpf(1e-30) / (em * em * norm_mp(Fk[2])))
    cd_ok = r1 <= 10 * max(x1, f1, 1e-28) and r2 <= 10 * max(x2, f2, 1e-28)
    rec["central_ok"] = bool(cd_ok)

    # the controls must fail the same comparison
    controls = {}
    for name, code in (("kappa_frozen", JET_MUT_KAPPA), ("n_frozen", JET_MUT_FREEZE_N)):
        Fm_k, _ = coefficients(code)
        _, w = compare(Fm_k, name)
        controls[name] = dict(worst_ratio=w, fires=bool(w > TOL))
        print(f"control {name}: worst ratio {w:.1e}: {'fires' if w > TOL else 'DOES NOT FIRE'}", flush=True)
    rec["controls"] = controls
    ok = rec["passed"] and cd_ok and all(c["fires"] for c in controls.values())
    rec["ok"] = bool(ok)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("J2", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
