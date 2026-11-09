#!/usr/bin/env python3
"""Count the zeros of the growth-mode determinant det M(lambda) in boxes of the right half plane by the argument
principle (notes/MODE_COUNT.md), for one saved double-double state.

    python3 modecount_wind.py STATE.npz [--M=N] [--kf=none|13] [--threads=N] [--tag=NAME] [--h0=0.4] [--tol=0.785]
                              [--Im2=1.5] [--aR=..] [--aL=..] [--aLL=..] [--cache=TAG] [--res=hp/data/modecount_results.jsonl]
                              [--alias=1.5708]

--fmax: the largest background residual accepted (default 1e-20, as hp/gamma_dd.py; a count needs far less).
--centre_tol=T: with --M, the centre series' tolerance in modecount_common.fit_nodes (default 1e-15). A smaller T adds
inner nodes to a layout that passes the series gate (#270), and the state is kept under its own name (_ct<T>).
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
halves are checked to add up to the whole (a hidden 2 pi turn would show as a mismatch). A half above tol sends the
edge back to the bisection, and the two steps repeat until a verification adds nothing new (a zero just inside an
edge shows only in those halves: at D = 4 a single verification pass left steps of 1.8 rad and missed a pair 0.0017
inside, notes/GHOST_ADMIXTURE.md).

Alias rule (#242). A step that looks small can hide whole turns: the edge Im = y2 of #212's kf 23 count left two
intervals at the initial spacing whose steps of +0.04 and +0.23 rad hid +19.1 rad (#230). So the bisection also holds
every interval's length h against the local scale of g, estimated by |Delta ln g| / h (the change of ln|g|, which does
not wrap, and the wrapped phase step) on the interval itself and up to 3 intervals each side. At the first and last 3
intervals of an edge the same scale of the other edges' 3 intervals at that corner is included, since |(ln g)'| does not
depend on the direction (by Cauchy-Riemann the phase rate along an edge is the normal slope of ln|g|, which a
perpendicular edge sees unaliased). An interval is bisected while h * scale > --alias (default pi/2). The edges are
sampled in sweeps that pass the corner scales round until none grows. A hidden turn then needs the rate inside an
interval to exceed about 3.5 times the largest one seen within 3 intervals of it. Planted cases (modecount_planted.py)
that the old rule miscounts: a close pair of zeros between two samples (caught through ln|g|), and a row of zeros along
the whole of Im = y2 (caught through the corners, the old rule counted U 0 for 22 zeros). --mutate=flat restores the
old rule, --mutate=nolng drops ln|g| from the scale, --mutate=nocorner the corners (the controls).

An edge is resolved when every step is at most tol, every interval meets the alias rule and no halves mismatched (the
spacing floor hmin, default 1e-4, can stop it short); a box is resolved when all its edges are, and an unresolved box's
count is not a count. --mutate=one_verify restores the single verification pass (the control). --resolve=last
counts an edge resolved when its last verification pass finds no mismatch (earlier ones refined away); the default,
--resolve=any, requires none in any pass (#169). Every evaluation is
appended to <data>/modecount_<tag>_evals.jsonl and reused on restart; --data (default hp/data) holds the cache, the log
and the record, and must be in a raw-data folder (raw_out.py; --allow_unraw writes anyway).
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


def count(det_fn, info, poles, opts, log, tag, res_file, expected=None, boxes_only=None):
    """The argument-principle count. det_fn(re, im) -> (ln|det|, arg det, pmin, pmax); info: D, M, omega, lam_g,
    lam_f (lam_phys, used only as an expected zero and for the geometry) and whatever else goes into the record;
    poles: the candidate poles of det M (complex). expected: the real parts of the expected zeros (default 0, lam_g,
    lam_f; each with its copies + 2 i k omega); boxes_only: count only these boxes (and follow only their edges)."""
    h0 = float(opts.get("h0", 0.4))
    tol = float(opts.get("tol", math.pi / 4))
    hmin = float(opts.get("hmin", 1e-4))
    im2 = float(opts.get("Im2", 1.5))
    omega, lam_g, lam_f = info["omega"], info["lam_g"], info["lam_f"]
    cache_path = os.path.join(opts.get("data", DATA), f"modecount_{opts.get('cache', tag)}_evals.jsonl")
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
    known = [complex(z, 2 * k * omega) for z in ((0.0, lam_g, lam_f) if expected is None else expected)
             for k in range(-kmax, kmax + 1) if abs(2 * k * omega) < y2 + 2 * omega]
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
    # alias rule (#242): an interval whose step of g looks small can still hold whole turns, so its length is also
    # held against the local scale |d ln g / d lam|, estimated by |Delta ln g| / h (ln|g| and the wrapped phase) on
    # itself and NB intervals each side, and at a corner on the other edges' NB intervals next to it (the modulus of
    # (ln g)' does not depend on the direction). An interval is bisected while h * scale > alias.
    alias = 0.0 if opts.get("mutate") == "flat" else float(opts.get("alias", math.pi / 2))
    NB = 3

    def scales(ts, vals, L):
        """|Delta ln g| / h per interval: ln|g| does not wrap, the phase step is the wrapped one"""
        out = []
        for a, b in zip(ts[:-1], ts[1:]):
            dl = vals[b][0] - vals[a][0] if opts.get("mutate") != "nolng" else 0.0
            out.append(abs(complex(dl, wrap(vals[b][1] - vals[a][1]))) / ((b - a) * L))
        return out

    def edge(z0, z1, name, ends=(0.0, 0.0)):
        """ends: the largest scale of the other edges' intervals at z0 and at z1 (the corners)"""
        L = abs(z1 - z0)
        n = max(2, int(math.ceil(L / h0)))
        vals = {i / n: g(z0 + (i / n) * (z1 - z0)) for i in range(n + 1)}
        checked, bad, nver, passes = set(), 0, 0, 0
        bad_last = 0          # mismatches found in the last verification pass
        nalias = 0            # bisections made by the alias rule only
        while True:
            changed = True
            while changed:
                changed = False
                ts = sorted(vals)
                sc = scales(ts, vals, L) if alias > 0 else None
                for i, (a, b) in enumerate(zip(ts[:-1], ts[1:])):
                    if (b - a) * L < hmin:
                        continue
                    split = abs(wrap(vals[b][1] - vals[a][1])) > tol
                    if not split and alias > 0:
                        s = max(sc[max(i - NB, 0):i + NB + 1])
                        if i < NB:
                            s = max(s, ends[0])
                        if i >= len(sc) - NB:
                            s = max(s, ends[1])
                        split = (b - a) * L * s > alias
                        nalias += split
                    if split:
                        m = 0.5 * (a + b)
                        vals[m] = g(z0 + m * (z1 - z0))
                        changed = True
            # verification: bisect every unchecked interval with |dphi| > tol / 2 once more; the halves must add up
            # to the whole, and a half above tol goes back to the bisection above
            ts = sorted(vals)
            added = False
            passes += 1
            bad_last = 0
            for a, b in zip(ts[:-1], ts[1:]):
                dphi = wrap(vals[b][1] - vals[a][1])
                if abs(dphi) > tol / 2 and (b - a) * L >= hmin and (a, b) not in checked:
                    checked.add((a, b))
                    m = 0.5 * (a + b)
                    vals[m] = g(z0 + m * (z1 - z0))
                    nver += 1
                    added = True
                    if abs(wrap(vals[m][1] - vals[a][1]) + wrap(vals[b][1] - vals[m][1]) - dphi) > 1e-6:
                        bad += 1
                        bad_last += 1
            if not added or opts.get("mutate") == "one_verify":
                break
        ts = sorted(vals)
        steps = [wrap(vals[b][1] - vals[a][1]) for a, b in zip(ts[:-1], ts[1:])]
        raw = [wrap(vals[b][3] - vals[a][3]) for a, b in zip(ts[:-1], ts[1:])]
        # the alias rule at the end: intervals above the spacing floor all meet it (the loop above), those below it
        # that do not are counted (a zero of g within about hmin of the edge), and make the edge unresolved
        sc = scales(ts, vals, L)
        alias_left = 0
        if alias > 0:
            for i, (a, b) in enumerate(zip(ts[:-1], ts[1:])):
                s = max([*sc[max(i - NB, 0):i + NB + 1]] + ([ends[0]] if i < NB else []) +
                        ([ends[1]] if i >= len(sc) - NB else []))
                alias_left += (b - a) * L * s > alias
        e = dict(name=name, z0=[z0.real, z0.imag], z1=[z1.real, z1.imag], points=len(ts), dphi=sum(steps),
                 max_step=max(abs(s) for s in steps), max_raw_step=max(abs(s) for s in raw),
                 max_hspace=max(b - a for a, b in zip(ts[:-1], ts[1:])) * L,
                 min_hspace=min(b - a for a, b in zip(ts[:-1], ts[1:])) * L,
                 min_lng=min(v[0] for v in vals.values()), max_lng=max(v[0] for v in vals.values()),
                 min_lndet=min(v[2] for v in vals.values()), verified=nver, verify_mismatch=bad, passes=passes,
                 verify_mismatch_last=bad_last, alias=alias, alias_bisections=nalias, alias_left=alias_left,
                 corner_scales=list(ends), max_scale_h=max(((b - a) * L * s for (a, b), s in
                                                            zip(zip(ts[:-1], ts[1:]), sc)), default=0.0),
                 # --resolve=any (default): no mismatch in any pass; --resolve=last: none in the final pass, whose
                 # bisections leave every interval within tol (earlier mismatches were refined away, #169)
                 resolved=max(abs(s) for s in steps) <= tol and (bad_last if opts.get("resolve") == "last" else bad) == 0
                 and alias_left == 0,
                 samples=[[round(t, 10), round(vals[t][0], 6), round(vals[t][1], 6), round(vals[t][2], 6),
                           round(vals[t][3], 6)] for t in ts])
        e["_log"] = (f"  edge {name}: {len(ts)} points, phase change of g {e['dphi']:+.6f}, max step {e['max_step']:.3f} rad "
                     f"(det M itself: {e['max_raw_step']:.3f}), spacing {e['min_hspace']:.2e}..{e['max_hspace']:.2e}, ln|g| "
                     f"{e['min_lng']:.2f}..{e['max_lng']:.2f}, verified {nver} (mismatch {bad}, last pass {bad_last}) in "
                     f"{passes} passes, alias bisections {nalias}" + (f" ({alias_left} below hmin)" if alias_left else "")
                     + ("" if e["resolved"] else "  UNRESOLVED"))
        return e

    def end_scale(e, at_start):
        """largest scale of the NB intervals of edge e next to its start (or end)"""
        z0, z1 = complex(*e["z0"]), complex(*e["z1"])
        S = e["samples"]
        ts = [s[0] for s in S]
        v = {s[0]: (s[1], s[2]) for s in S}
        sc = scales(ts, v, abs(z1 - z0))
        return max(sc[:NB] if at_start else sc[-NB:])

    specs = [("R1", complex(aR, 0), complex(aR, y1), "Re=aR, Im 0->y1"),
             ("V2", complex(a2, 0), complex(a2, y1), "Re=a2, Im 0->y1"),
             ("V1", complex(a1, 0), complex(a1, y1), "Re=a1, Im 0->y1"),
             ("L1", complex(aL, 0), complex(aL, y1), "Re=aL, Im 0->y1"),
             ("V0", complex(aLL, 0), complex(aLL, y1), "Re=aLL, Im 0->y1"),
             ("H1", complex(aR, y1), complex(a2, y1), "Im=y1, Re aR->a2"),
             ("H2", complex(a2, y1), complex(a1, y1), "Im=y1, Re a2->a1"),
             ("H3", complex(a1, y1), complex(aL, y1), "Im=y1, Re a1->aL"),
             ("H0", complex(aL, y1), complex(aLL, y1), "Im=y1, Re aL->aLL"),
             ("R2", complex(aR, y1), complex(aR, y2), "Re=aR, Im y1->y2"),
             ("L2", complex(aL, y1), complex(aL, y2), "Re=aL, Im y1->y2"),
             ("V0b", complex(aLL, y1), complex(aLL, y2), "Re=aLL, Im y1->y2"),
             ("T", complex(aR, y2), complex(aL, y2), "Im=y2, Re aR->aL"),
             ("T0", complex(aL, y2), complex(aLL, y2), "Im=y2, Re aL->aLL")]
    # counter-clockwise upper half paths (symmetric boxes, / pi) and closed strips (/ 2 pi): (edges with their signs,
    # divisor, rectangle)
    box_def = {
        "B0": ({"L1": 1, "H0": 1, "V0": -1}, math.pi, (aLL, aL, -y1, y1)),
        "U0": ({"H0": -1, "L2": 1, "T0": 1, "V0b": -1}, 2 * math.pi, (aLL, aL, y1, y2)),
        "B1a": ({"V1": 1, "H3": 1, "L1": -1}, math.pi, (aL, a1, -y1, y1)),
        "B1b": ({"V2": 1, "H2": 1, "V1": -1}, math.pi, (a1, a2, -y1, y1)),
        "B1c": ({"R1": 1, "H1": 1, "V2": -1}, math.pi, (a2, aR, -y1, y1)),
        "B1": ({"R1": 1, "H1": 1, "H2": 1, "H3": 1, "L1": -1}, math.pi, (aL, aR, -y1, y1)),
        "U": ({"R2": 1, "T": 1, "L2": -1, "H1": -1, "H2": -1, "H3": -1}, 2 * math.pi, (aL, aR, y1, y2)),
        "full": ({"R1": 1, "R2": 1, "T": 1, "L2": -1, "L1": -1}, math.pi, (aL, aR, -y2, y2)),
    }
    sel = list(box_def) if boxes_only is None else list(boxes_only)
    need = {e for b in sel for e in box_def[b][0]}
    # the edges are sampled in sweeps: each sweep passes every edge the largest scale that the other edges meeting at
    # its corners showed so far (kept monotone, so the samples only grow), until no corner scale grows
    corner = {}           # (corner point, edge name) -> largest scale of that edge's intervals next to the corner
    sweeps = 0
    while True:
        sweeps += 1
        E = {}
        for name, z0, z1, label in specs:
            if name in need:
                ends = tuple(max([s for (p, nm), s in corner.items() if p == z and nm != name] + [0.0])
                             for z in (z0, z1)) if alias > 0 and opts.get("mutate") != "nocorner" else (0.0, 0.0)
                E[name] = edge(z0, z1, label, ends)
        grew = False
        for name, z0, z1, label in specs:
            if name in need and alias > 0:
                for z, start in ((z0, True), (z1, False)):
                    s = end_scale(E[name], start)
                    if s > corner.get((z, name), 0.0) * (1 + 1e-12):
                        corner[(z, name)] = s
                        grew = True
        if not grew:
            break
    for name in E:
        log(E[name].pop("_log"))
    log(f"  edges sampled in {sweeps} sweeps (alias rule {'off' if alias == 0 else f'h * scale <= {alias:.4f}'})")
    boxes = {b: (sum(sg * E[e]["dphi"] for e, sg in box_def[b][0].items()) / box_def[b][1], box_def[b][2],
                 list(box_def[b][0])) for b in sel}

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
                      verify_mismatch=sum(e["verify_mismatch"] for e in ed), resolved=all(e["resolved"] for e in ed))
        log(f"  box {b:4s} Re ({r0:.3f}, {r1:.3f}) Im ({i0:+.3f}, {i1:+.3f}): winding of g {w:+.6f} -> {wi}; expected zeros "
            f"inside {len(kin)}, candidate poles inside {len(pin)} -> winding of det M {wi + len(kin) - len(pin)}, zeros of "
            f"det M <= {wi + len(kin)}; max phase step {out[b]['max_step']:.3f} rad, nearest pole / expected zero to the "
            f"edge {dmin:.3f} / {dz:.3f}" + ("" if out[b]["resolved"] else "; UNRESOLVED: not a count"))
    rec = dict(info, tag=tag, aLL=aLL, aL=aL, a1=a1, a2=a2, aR=aR, y1=y1, y2=y2, h0=h0, tol=tol, alias=alias, sweeps=sweeps,
               trend=list(pc),
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
    data = opts.get("data", DATA)
    res_file = opts.get("res", os.path.join(data, "modecount_results.jsonl"))
    logs = []

    def log(m):
        print(m, flush=True)
        logs.append(m)

    kf = "default" if "kf" not in opts else (None if opts["kf"] == "none" else int(opts["kf"]))
    st = setup(path, M=int(opts["M"]) if "M" in opts else None, threads=threads, kf=kf, log=log,
               fmax=float(opts.get("fmax", 1e-20)),
               centre_tol=float(opts["centre_tol"]) if "centre_tol" in opts else None)
    D, M = st.D, st.M
    tag = opts.get("tag", f"D{D:.4f}_M{M}")
    logf = os.path.join(data, f"modecount_{tag}.log")
    from raw_out import require_raw
    require_raw(logf, res_file)
    logf = open(logf, "a")
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
