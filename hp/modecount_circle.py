#!/usr/bin/env python3
"""Winding number of det M(lambda) on small circles (notes/MODE_COUNT.md): a local test of a single zero or pole, and
the position of a single zero inside.

    python3 modecount_circle.py STATE.npz RE,IM,R [RE,IM,R ...] [--M=N] [--kf=none|13] [--tol=0.785] [--n0=16]
                                [--data=hp/data] [--res=<data>/modecount_circles.jsonl] [--allow_unraw]

The phase of det M(lambda) exp(-beta (lambda - c)) is followed around each circle lambda = c + R e^{i t}, with beta
the slope of ln det M between c - 2R and c + 2R (removing the exponential growth of det M; the winding is that of
det M), starting from n0 equal steps and bisecting until every step is at most tol, then every step above tol / 2 is
bisected once more and checked against the whole, repeated until a verification adds nothing. Alias rule (#242, as
modecount_wind.py's edges): an arc is also bisected while its length times the local scale of ln h exceeds pi/2,
the scale being the largest |Delta ln h| / arc of the arc itself and 3 arcs each side (cyclically), so that a close
pair of zeros between two samples, which turns the phase by nearly 2 pi and leaves a small step, is resolved (#257).
A circle is resolved when every step is at most tol, no halves mismatched and every arc meets the rule.
--mutate=flat restores the old rule (no alias rule, one verification pass; the control of
hp/modecount_circle_planted.py). --kf=13 gives the tag a "_kf13" suffix.
Each evaluation goes to <data>/modecount_<tag>_evals.jsonl (the same cache as modecount_wind.py for that state), and
the log and the record go to <data> too, which must be a task folder of the raw-data repository (hp/raw_out.py).

Positions. When the winding W is 1, 2 or 3, h(lam) = det M exp(-beta (lam - c)) / (lam - c)^W has no winding, so
ln h is periodic on the circle, and the power sums of the zeros z_j inside are (integration by parts)

    s_k = sum_j (z_j - c)^k = -(k / 2 pi i) closed int (lam - c)^(k-1) ln h d lam
        = -(k R^k / n) sum_j ln h(lam_j) e^{i k t_j}                        (t_j = 2 pi j / n, k = 1 .. W),

the second form the trapezoid rule on the n0 equally spaced points, which converges geometrically for a periodic
analytic integrand; Newton's identities give the zeros (for W = 1, z = c + s_1). The same sums on every other point
(n0 / 2) give second values; their distance is the error estimate (conservative, since the n0 / 2 sums are the less
accurate; hp/modecount_circle_planted.py tests it). --mutate=nodiv leaves out the division by (lam - c)^W (the
planted test's control).
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from modecount_wind import parse, wrap, DATA  # noqa: E402

# the alias rule's threshold on arc * scale and its neighbourhood, as in modecount_wind (#242)
ALIAS = math.pi / 2
NB = 3


def circle(det, c, R, tol=math.pi / 4, n0=16, mutate=None, alias=ALIAS):
    """winding of det M on |lam - c| = R, det(lam) -> (ln|det M|, arg det M); with winding 1 also the zero inside"""
    pt = lambda s: c + R * complex(math.cos(2 * math.pi * s), math.sin(2 * math.pi * s))
    # local detrending: beta = (ln det(c + 2R) - ln det(c - 2R)) / 4R, from two points outside the circle;
    # exp(-beta (lam - c)) is entire and zero-free, so the winding is unchanged
    lp, ap = det(c + 2 * R)
    lm, am = det(c - 2 * R)
    beta = complex(lp - lm, wrap(ap - am)) / (4 * R)

    def gdet(lam):
        la, arg = det(lam)
        v = complex(la, arg) - beta * (lam - c)
        return v.real, wrap(v.imag)
    vals = {i / n0: gdet(pt(i / n0)) for i in range(n0)}
    vals[1.0] = vals[0.0]

    def step(a, b):
        return wrap(vals[b][1] - vals[a][1])

    # alias rule (#242, as modecount_wind's edge): an arc is also held against the local scale |Delta ln h| / arc of
    # itself and NB arcs each side, cyclically; flat (the control) restores the old rule, one verification pass
    flat = mutate == "flat"
    alias = 0.0 if flat else alias
    L = 2 * math.pi * R

    def scales(ts):
        return [abs(complex(vals[b][0] - vals[a][0], step(a, b))) / ((b - a) * L) for a, b in zip(ts[:-1], ts[1:])]

    def local(sc, i):
        return max(sc[(i + j) % len(sc)] for j in range(-NB, NB + 1))

    checked, bad, nver, passes, nalias = set(), 0, 0, 0, 0
    while True:
        changed = True
        while changed:
            changed = False
            ts = sorted(vals)
            sc = scales(ts) if alias > 0 else None
            for i, (a, b) in enumerate(zip(ts[:-1], ts[1:])):
                if (b - a) * L <= 1e-6:
                    continue
                split = abs(step(a, b)) > tol
                if not split and alias > 0:
                    split = (b - a) * L * local(sc, i) > alias
                    nalias += split
                if split:
                    m = 0.5 * (a + b)
                    vals[m] = gdet(pt(m))
                    changed = True
        # verification: every arc with |step| > tol / 2 is bisected once and its halves must add up to it; a half
        # above tol, or one the alias rule flags, goes back to the bisection, until a pass adds nothing
        ts = sorted(vals)
        added = False
        passes += 1
        for a, b in zip(ts[:-1], ts[1:]):
            d = step(a, b)
            if abs(d) > tol / 2 and (a, b) not in checked:
                checked.add((a, b))
                m = 0.5 * (a + b)
                vals[m] = gdet(pt(m))
                nver += 1
                added = True
                if abs(wrap(vals[m][1] - vals[a][1]) + wrap(vals[b][1] - vals[m][1]) - d) > 1e-6:
                    bad += 1
        if not added or flat:
            break
    ts = sorted(vals)
    steps = [step(a, b) for a, b in zip(ts[:-1], ts[1:])]
    w = sum(steps) / (2 * math.pi)
    # arcs below the floor that do not meet the rule are counted (a zero within about 1e-6 of the circle)
    sc = scales(ts)
    hs = [(b - a) * L * local(sc, i) for i, (a, b) in enumerate(zip(ts[:-1], ts[1:]))]
    alias_left = sum(h > alias for h in hs) if alias > 0 else 0
    rec = dict(center=[c.real, c.imag], R=R, winding=w, points=len(ts) - 1, beta=[beta.real, beta.imag],
               max_step=max(abs(s) for s in steps), verify_mismatch=bad, min_lndet=min(v[0] for v in vals.values()),
               verified=nver, passes=passes, alias=alias, alias_bisections=nalias, alias_left=alias_left,
               max_scale_h=max(hs), resolved=max(abs(s) for s in steps) <= tol and bad == 0 and alias_left == 0)
    W = int(round(w))
    if abs(w - W) < 1e-6 and 1 <= W <= 3:
        # ln h on the n0 base points: the phase continued through every sample, less that of (lam - c)^W
        phase, cont = vals[0.0][1], {0.0: vals[0.0][1]}
        for a, b in zip(ts[:-1], ts[1:]):
            phase += step(a, b)
            cont[b] = phase
        lnh = []
        for i in range(n0):
            t = i / n0
            v = complex(vals[t][0], cont[t])
            if mutate != "nodiv":
                v -= W * complex(math.log(R), 2 * math.pi * t)
            lnh.append(v)

        def zeros(idx):
            # s_k = sum (z_j - c)^k = -(k R^k / n) sum_j ln h_j e^{i k t_j}, then Newton's identities
            from modecount_moment import newton_roots
            s = [-(k * R ** k / len(idx)) * sum(lnh[i] * complex(math.cos(2 * math.pi * k * i / n0),
                                                                  math.sin(2 * math.pi * k * i / n0)) for i in idx)
                 for k in range(1, W + 1)]
            return sorted(newton_roots(s, c), key=lambda z: (z.real, z.imag))
        z, z2 = zeros(range(n0)), zeros(range(0, n0, 2))
        from modecount_moment import match
        z2 = match(z, z2)
        rec.update(zeros=[[x.real, x.imag] for x in z], zeros_est=[abs(a - b) for a, b in zip(z, z2)],
                   zeros_half=[[x.real, x.imag] for x in z2], zeros_n=n0)
    return rec


def main():
    from modecount_common import setup
    from raw_out import require_raw
    args, opts = parse(sys.argv[1:])
    path = args[0]
    data = opts.get("data", DATA)
    res_file = opts.get("res", os.path.join(data, "modecount_circles.jsonl"))
    kf = "default" if "kf" not in opts else (None if opts["kf"] == "none" else int(opts["kf"]))
    st = setup(path, M=int(opts["M"]) if "M" in opts else None, threads=int(opts.get("threads", 1)), kf=kf,
               fmax=float(opts.get("fmax", 1e-20)))
    tag = opts.get("tag", f"D{st.D:.4f}_M{st.M}" + ("" if "kf" not in opts else f"_kf{opts['kf']}"))
    tol = float(opts.get("tol", math.pi / 4))
    n0 = int(opts.get("n0", 16))
    cache_path = os.path.join(data, f"modecount_{tag}_evals.jsonl")
    log_path = os.path.join(data, f"modecount_{tag}.log")
    require_raw(cache_path, log_path, res_file)
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            cache[(r["re"], r["im"])] = r
    cf = open(cache_path, "a")
    logf = open(log_path, "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    def det(lam):
        key = (float(lam.real), float(lam.imag))
        if key not in cache:
            t = time.time()
            la, arg, pmin, pmax = st.det(key[0], key[1])
            r = dict(re=key[0], im=key[1], lnabs=la, arg=arg, pmin=pmin, pmax=pmax, sec=round(time.time() - t, 2))
            cache[key] = r
            cf.write(json.dumps(r) + "\n")
            cf.flush()
        return cache[key]["lnabs"], cache[key]["arg"]

    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} circles, D {st.D}, M {st.M}, kfreeze {st.kf}, omega {st.omega:.6f}, "
        f"lam_g {st.lam_g:.10f}, |F| {st.F:.1e}")
    for spec in args[1:]:
        cr, ci, R = (float(v) for v in spec.split(","))
        rec = circle(det, complex(cr, ci), R, tol, n0, opts.get("mutate"))
        rec = dict(D=st.D, M=st.M, kfreeze=st.kf, F=st.F, tag=tag, **rec, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        pos = ""
        if "zeros" in rec:
            pos = "; zeros inside " + ", ".join(f"{a:.6f} {b:+.6f}i (estimate {e:.1e})"
                                                for (a, b), e in zip(rec["zeros"], rec["zeros_est"]))
        log(f"  circle |lam - ({cr:.6f} {ci:+.6f}i)| = {R}: winding of det M {rec['winding']:+.6f}, {rec['points']} points "
            f"(+2 for beta), max step {rec['max_step']:.3f} rad, verification mismatches {rec['verify_mismatch']}, alias "
            f"bisections {rec['alias_bisections']} (largest arc*scale {rec['max_scale_h']:.2f})"
            + ("" if rec["resolved"] else "  UNRESOLVED") + pos)
        with open(res_file, "a") as fo:
            fo.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
