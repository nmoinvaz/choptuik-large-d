#!/usr/bin/env python3
"""Count the zeros of the growth-mode determinant det M(lambda) in boxes of the right half plane by the argument
principle (notes/MODE_COUNT.md), for one saved double-double state.

    python3 modecount_wind.py STATE.npz [--M=N] [--kf=none|13] [--threads=N] [--tag=NAME] [--h0=0.4] [--tol=0.785]
                              [--Im2=1.5] [--aR=..] [--aL=..] [--aLL=..] [--cache=TAG] [--res=hp/data/modecount_results.jsonl]

--tag names the run (log, record); --cache names the evaluation cache (default: the tag), so that runs on the same
state and layout with other boxes can share evaluations.

det M(lambda) is evaluated at complex lambda with hp/modecount_cx (the hp kernel's mode matrix with a complex tangent,
double-double, complex double-double LU). With omega = 2 pi / Delta, y1 = omega / 2 and y2 = Im2 * omega (default
1.5 omega) the contour network is

  verticals    Re = aLL, aL, aR from Im 0 to y2;  Re = a1, a2 from Im 0 to y1
  horizontals  Im = y1 and Im = y2 from Re aR to aLL
  real axis    lambda = aL ... aR (the sign of det M; also the trend below)

with aLL ~ -0.1, aL ~ 0.1-0.3, a1 between the gauge zero lam_g and lam_phys, a2 = lam_phys + 0.6, aR ~ 6, each chosen
away from the candidate poles. det M is real on the real axis and det M(conj lam) = conj det M(lam), so a box symmetric
about the real axis has winding (1/pi) x the phase change along its upper half; the strips y1 < Im < y2 are counted on
their own (their mirror images hold the conjugate zeros).

The function whose phase is followed is

    g(lam) = det M(lam) * q(lam) / k(lam) * exp(-p(lam)),
    q = prod (lam - p_j)   over the candidate poles p_j near the network (det M's only possible poles, all simple),
    k = prod (lam - z_j)   over the zeros expected: 0 (tau translation), lam_g (the residual gauge mode), lam_phys,
                           and their copies + 2 i k omega (the folded layout's period),
    p = a real quadratic fitted to ln|det M q / k| on the real axis.

q, k and exp(-p) are known exactly, so on every closed contour

    winding(det M) = winding(g) + (expected zeros inside) - (candidate poles inside),
    zeros of det M inside <= winding(g) + (expected zeros inside)      (equality if every candidate is a pole).

With no zeros besides the expected ones, winding(g) = 0 for every box, and g varies slowly, so few samples are needed.

Boxes:  B0 (aLL, aL) x (-y1, y1)   expected zeros: lam = 0
        U0 (aLL, aL) x (y1, y2)     none
        B1a (aL, a1) x (-y1, y1)   lam_g          B1b (a1, a2) x (-y1, y1)  lam_phys       B1c (a2, aR) x (-y1, y1)  none
        B1 (aL, aR) x (-y1, y1)    lam_g, lam_phys
        U (aL, aR) x (y1, y2)      none
        full (aL, aR) x (-y2, y2)  lam_g, lam_phys

Sampling: each edge starts on a grid of spacing h0 and every interval is bisected until the phase change of g across it
is at most tol (default pi/4); then every interval whose phase change exceeds tol/2 is bisected once more and the two
halves are checked to add up to the whole (a hidden 2 pi turn would show as a mismatch). Every evaluation is appended
to hp/data/modecount_<tag>_evals.jsonl and reused on restart.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")


def wrap(a):
    return math.remainder(a, 2 * math.pi)


def parse(argv):
    args = [a for a in argv if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in argv if a.startswith("--"))
    return args, opts


def count(det_fn, info, poles, opts, log, tag, res_file):
    """The argument-principle count. det_fn(re, im) -> (ln|det|, arg det, pmin, pmax); info: D, M, omega, lam_g,
    lam_f (lam_phys, used only as an expected zero and for the geometry) and whatever else goes into the record;
    poles: the candidate poles of det M (complex)."""
    h0 = float(opts.get("h0", 0.4))
    tol = float(opts.get("tol", math.pi / 4))
    hmin = float(opts.get("hmin", 1e-4))
    im2 = float(opts.get("Im2", 1.5))
    omega, lam_g, lam_f = info["omega"], info["lam_g"], info["lam_f"]
    cache_path = os.path.join(DATA, f"modecount_{opts.get('cache', tag)}_evals.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            cache[(r["re"], r["im"])] = r
        log(f"  {len(cache)} cached evaluations")
    cf = open(cache_path, "a")
    nev = [0, 0.0]

    def det(lam):
        key = (float(lam.real), float(lam.imag))
        if key not in cache:
            t = time.time()
            la, arg, pmin, pmax = det_fn(key[0], key[1])
            dt = time.time() - t
            r = dict(re=key[0], im=key[1], lnabs=la, arg=arg, pmin=pmin, pmax=pmax, sec=round(dt, 2))
            cache[key] = r
            cf.write(json.dumps(r) + "\n")
            cf.flush()
            nev[0] += 1
            nev[1] += dt
            if nev[0] % 10 == 0:
                log(f"    {nev[0]} evaluations, {nev[1] / nev[0]:.1f} s each")
        r = cache[key]
        return r["lnabs"], r["arg"]

    # ---- geometry: vertical edges away from the candidate poles and the expected zeros
    y1, y2 = 0.5 * omega, im2 * omega
    # expected zeros: lam = 0, lam_g and lam_phys, and their copies 2 i k omega of the folded layout
    kmax = int((y2 + 2 * omega) / omega) + 1
    known = [complex(z, 2 * k * omega) for z in (0.0, lam_g, lam_f) for k in range(-kmax, kmax + 1)
             if abs(2 * k * omega) < y2 + 2 * omega]
    pre = sorted(set(p.real for p in poles if abs(p.imag) < y2 + 0.5 * omega))

    def far(lo, hi, avoid):
        cands = np.linspace(lo, hi, 201)
        d = [min([abs(x - a) for a in avoid] + [9.0]) for x in cands]
        return float(cands[int(np.argmax(d))])

    zre = [z.real for z in known]
    aR = float(opts["aR"]) if "aR" in opts else far(6.0, 6.7, pre)
    aL = float(opts["aL"]) if "aL" in opts else far(0.08, 0.3, pre + zre)
    aLL = float(opts["aLL"]) if "aLL" in opts else far(-0.15, -0.05, pre + zre)
    a1 = float(opts["a1"]) if "a1" in opts else far(lam_g + 0.35, lam_f - 0.35, pre + zre)
    a2 = float(opts["a2"]) if "a2" in opts else lam_f + 0.6
    log(f"  contour: aLL {aLL:.4f}, aL {aL:.4f}, a1 {a1:.4f}, a2 {a2:.4f}, aR {aR:.4f}; y1 = omega/2 = {y1:.4f}, "
        f"y2 = {im2} omega = {y2:.4f}")
    log(f"  candidate poles with |Im| < y2, Re in (-1, 9): "
        f"{[(round(p.real, 4), round(p.imag, 4)) for p in sorted(poles, key=lambda p: (p.imag, p.real)) if -1 < p.real < 9 and abs(p.imag) < y2]}")
    log(f"  expected zeros: {[(round(z.real, 6), round(z.imag, 6)) for z in known]}")

    # all candidates (every horizon order kk = 0..ks) in a band around the network: the rows of poles at Im = +-omega
    # reach Re = n/2 - lam_g (ks + 1), far to the left, and leaving their tails out makes g drift along vertical edges
    Q = np.array([p for p in poles if abs(p.imag) < y2 + 3 * omega], complex)
    K = np.array(known, complex)

    def lnqk(lam):
        return complex(np.sum(np.log(lam - Q)) - np.sum(np.log(lam - K)))

    # ---- real axis: sign of det M and the trend
    xs = sorted(set([aL, a1, a2, aR] + list(np.arange(0.5, aR, 0.5))))
    real = []
    for x in xs:
        la, arg = det(complex(x, 0.0))
        real.append((x, la, 1 if abs(arg) < 1.0 else -1))
    X = np.array([r[0] for r in real])
    Y = np.array([r[1] + lnqk(complex(r[0], 0)).real for r in real])
    pc = np.polyfit(X, Y, 2)
    log("  real axis, sign of det M: " + ", ".join(f"{x:.3f}:{'+' if s > 0 else '-'}" for x, _, s in real))
    sign_changes = [(real[i][0], real[i + 1][0]) for i in range(len(real) - 1) if real[i][2] != real[i + 1][2]]
    log(f"  sign changes of det M on the real axis: {[(round(a, 3), round(b, 3)) for a, b in sign_changes]}")
    log(f"  {len(Q)} candidate poles multiplied out, {len(K)} expected zeros divided out; trend p(lam) = {pc[0]:.4f} lam^2 "
        f"+ {pc[1]:.4f} lam + {pc[2]:.3f} (residual on the real axis max {np.abs(np.polyval(pc, X) - Y).max():.3f})")

    def g(lam):
        la, arg = det(lam)
        v = complex(la, arg) + lnqk(lam) - complex(np.polyval(pc, lam))
        return v.real, wrap(v.imag), la, arg

    # ---- adaptive edges
    def edge(z0, z1, name):
        L = abs(z1 - z0)
        n = max(2, int(math.ceil(L / h0)))
        vals = {i / n: g(z0 + (i / n) * (z1 - z0)) for i in range(n + 1)}
        changed = True
        while changed:
            changed = False
            ts = sorted(vals)
            for a, b in zip(ts[:-1], ts[1:]):
                if (b - a) * L < hmin:
                    continue
                if abs(wrap(vals[b][1] - vals[a][1])) > tol:
                    m = 0.5 * (a + b)
                    vals[m] = g(z0 + m * (z1 - z0))
                    changed = True
        # verification: bisect every interval with |dphi| > tol / 2 once more; the halves must add up to the whole
        ts = sorted(vals)
        bad = nver = 0
        for a, b in zip(ts[:-1], ts[1:]):
            dphi = wrap(vals[b][1] - vals[a][1])
            if abs(dphi) > tol / 2 and (b - a) * L >= hmin:
                m = 0.5 * (a + b)
                vals[m] = g(z0 + m * (z1 - z0))
                nver += 1
                if abs(wrap(vals[m][1] - vals[a][1]) + wrap(vals[b][1] - vals[m][1]) - dphi) > 1e-6:
                    bad += 1
        ts = sorted(vals)
        steps = [wrap(vals[b][1] - vals[a][1]) for a, b in zip(ts[:-1], ts[1:])]
        raw = [wrap(vals[b][3] - vals[a][3]) for a, b in zip(ts[:-1], ts[1:])]
        e = dict(name=name, z0=[z0.real, z0.imag], z1=[z1.real, z1.imag], points=len(ts), dphi=sum(steps),
                 max_step=max(abs(s) for s in steps), max_raw_step=max(abs(s) for s in raw),
                 max_hspace=max(b - a for a, b in zip(ts[:-1], ts[1:])) * L,
                 min_hspace=min(b - a for a, b in zip(ts[:-1], ts[1:])) * L,
                 min_lng=min(v[0] for v in vals.values()), max_lng=max(v[0] for v in vals.values()),
                 min_lndet=min(v[2] for v in vals.values()), verified=nver, verify_mismatch=bad,
                 samples=[[round(t, 10), round(vals[t][0], 6), round(vals[t][1], 6), round(vals[t][2], 6),
                           round(vals[t][3], 6)] for t in ts])
        log(f"  edge {name}: {len(ts)} points, phase change of g {e['dphi']:+.6f}, max step {e['max_step']:.3f} rad "
            f"(det M itself: {e['max_raw_step']:.3f}), spacing {e['min_hspace']:.2e}..{e['max_hspace']:.2e}, ln|g| "
            f"{e['min_lng']:.2f}..{e['max_lng']:.2f}, verified {nver} (mismatch {bad})")
        return e

    E = {}
    E["R1"] = edge(complex(aR, 0), complex(aR, y1), "Re=aR, Im 0->y1")
    E["V2"] = edge(complex(a2, 0), complex(a2, y1), "Re=a2, Im 0->y1")
    E["V1"] = edge(complex(a1, 0), complex(a1, y1), "Re=a1, Im 0->y1")
    E["L1"] = edge(complex(aL, 0), complex(aL, y1), "Re=aL, Im 0->y1")
    E["V0"] = edge(complex(aLL, 0), complex(aLL, y1), "Re=aLL, Im 0->y1")
    E["H1"] = edge(complex(aR, y1), complex(a2, y1), "Im=y1, Re aR->a2")
    E["H2"] = edge(complex(a2, y1), complex(a1, y1), "Im=y1, Re a2->a1")
    E["H3"] = edge(complex(a1, y1), complex(aL, y1), "Im=y1, Re a1->aL")
    E["H0"] = edge(complex(aL, y1), complex(aLL, y1), "Im=y1, Re aL->aLL")
    E["R2"] = edge(complex(aR, y1), complex(aR, y2), "Re=aR, Im y1->y2")
    E["L2"] = edge(complex(aL, y1), complex(aL, y2), "Re=aL, Im y1->y2")
    E["V0b"] = edge(complex(aLL, y1), complex(aLL, y2), "Re=aLL, Im y1->y2")
    E["T"] = edge(complex(aR, y2), complex(aL, y2), "Im=y2, Re aR->aL")
    E["T0"] = edge(complex(aL, y2), complex(aLL, y2), "Im=y2, Re aL->aLL")

    def ph(name):
        return E[name]["dphi"]

    # counter-clockwise upper half paths (symmetric boxes, / pi) and closed strips (/ 2 pi)
    boxes = {
        "B0": ((ph("L1") + ph("H0") - ph("V0")) / math.pi, (aLL, aL, -y1, y1), ["L1", "H0", "V0"]),
        "U0": ((-ph("H0") + ph("L2") + ph("T0") - ph("V0b")) / (2 * math.pi), (aLL, aL, y1, y2), ["H0", "L2", "T0", "V0b"]),
        "B1a": ((ph("V1") + ph("H3") - ph("L1")) / math.pi, (aL, a1, -y1, y1), ["V1", "H3", "L1"]),
        "B1b": ((ph("V2") + ph("H2") - ph("V1")) / math.pi, (a1, a2, -y1, y1), ["V2", "H2", "V1"]),
        "B1c": ((ph("R1") + ph("H1") - ph("V2")) / math.pi, (a2, aR, -y1, y1), ["R1", "H1", "V2"]),
        "B1": ((ph("R1") + ph("H1") + ph("H2") + ph("H3") - ph("L1")) / math.pi, (aL, aR, -y1, y1), ["R1", "H1", "H2", "H3", "L1"]),
        "U": ((ph("R2") + ph("T") - ph("L2") - ph("H1") - ph("H2") - ph("H3")) / (2 * math.pi), (aL, aR, y1, y2),
              ["R2", "T", "L2", "H1", "H2", "H3"]),
        "full": ((ph("R1") + ph("R2") + ph("T") - ph("L2") - ph("L1")) / math.pi, (aL, aR, -y2, y2), ["R1", "R2", "T", "L2", "L1"]),
    }

    def dist(p, r0, r1, i0, i1):
        """distance from p to the boundary of the rectangle"""
        if r0 < p.real < r1 and i0 < p.imag < i1:
            return min(p.real - r0, r1 - p.real, p.imag - i0, i1 - p.imag)
        return math.hypot(max(r0 - p.real, 0, p.real - r1), max(i0 - p.imag, 0, p.imag - i1))

    out = {}
    for b, (w, (r0, r1, i0, i1), eds) in boxes.items():
        pin = [p for p in Q if r0 < p.real < r1 and i0 < p.imag < i1]
        kin = [z for z in K if r0 < z.real < r1 and i0 < z.imag < i1]
        dmin = min([dist(p, r0, r1, i0, i1) for p in Q] + [9.0])
        dz = min([dist(z, r0, r1, i0, i1) for z in K] + [9.0])
        ed = [E[e] for e in eds]
        wi = int(round(w))
        out[b] = dict(rect=[r0, r1, i0, i1], winding_g=w, winding_g_int=wi, expected_zeros_inside=len(kin),
                      expected_zeros=[[round(z.real, 6), round(z.imag, 6)] for z in kin], candidate_poles_inside=len(pin),
                      candidate_poles=[[round(p.real, 6), round(p.imag, 6)] for p in pin],
                      winding_det=wi + len(kin) - len(pin), zeros_upper_bound=wi + len(kin),
                      pole_min_dist_to_edge=dmin, expected_zero_min_dist_to_edge=dz,
                      max_step=max(e["max_step"] for e in ed), max_raw_step=max(e["max_raw_step"] for e in ed),
                      max_hspace=max(e["max_hspace"] for e in ed), min_lng=min(e["min_lng"] for e in ed),
                      min_lndet=min(e["min_lndet"] for e in ed), points=sum(e["points"] for e in ed),
                      verify_mismatch=sum(e["verify_mismatch"] for e in ed))
        log(f"  box {b:4s} Re ({r0:.3f}, {r1:.3f}) Im ({i0:+.3f}, {i1:+.3f}): winding of g {w:+.6f} -> {wi}; expected zeros "
            f"inside {len(kin)}, candidate poles inside {len(pin)} -> winding of det M {wi + len(kin) - len(pin)}, zeros of "
            f"det M <= {wi + len(kin)}; max phase step {out[b]['max_step']:.3f} rad, nearest pole / expected zero to the "
            f"edge {dmin:.3f} / {dz:.3f}")
    rec = dict(info, tag=tag, aLL=aLL, aL=aL, a1=a1, a2=a2, aR=aR, y1=y1, y2=y2, h0=h0, tol=tol, trend=list(pc),
               expected_zeros=[[z.real, z.imag] for z in K], n_candidate_poles_factored=len(Q),
               real_axis=[[x, la, s] for x, la, s in real], real_sign_changes=sign_changes,
               evaluations=len(cache), new_evaluations=nev[0], sec_per_eval=(nev[1] / nev[0] if nev[0] else None),
               edges=E, boxes=out, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(res_file, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
    log(f"  done: {len(cache)} evaluations in the cache ({nev[0]} new)")
    return rec


def main():
    from modecount_common import setup, horizon_poles
    from gamma_dd import gamma_formula
    args, opts = parse(sys.argv[1:])
    path = args[0]
    threads = int(opts.get("threads", 1))
    res_file = opts.get("res", os.path.join(DATA, "modecount_results.jsonl"))
    logs = []

    def log(m):
        print(m, flush=True)
        logs.append(m)

    kf = "default" if "kf" not in opts else (None if opts["kf"] == "none" else int(opts["kf"]))
    st = setup(path, M=int(opts["M"]) if "M" in opts else None, threads=threads, kf=kf, log=log)
    D, M = st.D, st.M
    tag = opts.get("tag", f"D{D:.4f}_M{M}")
    logf = open(os.path.join(DATA, f"modecount_{tag}.log"), "a")
    for m in logs:
        logf.write(m + "\n")

    def log2(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    lam_f = float(opts["lam"]) if "lam" in opts else 1.0 / gamma_formula(D)
    log2(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {os.path.basename(path)}: D {D}, M {M}, kfreeze {st.kf}, "
         f"|F| {st.F:.1e} (in {st.F_in:.1e}); Delta {st.Delta:.12f}, omega {st.omega:.6f}; "
         f"lam_g = 1 + <c1> = {st.lam_g:.10f}; lam_phys (expected) {lam_f:.10f}")
    poles = [p for _, p in horizon_poles(st)]
    info = dict(D=D, M=M, kfreeze=st.kf, F=st.F, state=os.path.basename(path), solver="dd (hp + modecount_cx)",
                Delta=st.Delta, omega=st.omega, lam_g=st.lam_g, lam_f=lam_f)
    count(lambda re, im: st.det(re, im), info, poles, opts, log2, tag, res_file)


if __name__ == "__main__":
    main()
