#!/usr/bin/env python3
"""Is Newton's Delta step at a converged state determined at octo-double precision? Its spread under perturbations of
one block at a time by a relative size (#827), and that spread against D and #805's 2.2e-37/p_min (#830).

    cd hp && python3 step_spread.py RAW STATE [--threads 3] [--rel 1e-65] [--seeds 1,2,3,4] [--blocks m_h,a_h]
        [--ref D] [--repeat 1] [--controls 1] [--match OUT.json] [--json OUT] [--mutate repeat|noop]
    cd hp && python3 step_spread.py sum TABLE OUT.json [OUT.json ...] [--K 2.2e-37] [--slope 1.44] [--json SUM]
    cd hp && python3 step_spread.py power OUT.json [OUT.json ...] [--quad 1.8,2.2] [--lin 0.9,1.1] [--json POW]
        [--mutate start]
    cd hp && python3 step_spread.py --selftest [--mutate=bar|band|nosub]

The system is proj_shift.py's (octo_corrector.py planted's at --rel 0, truth_system: an eight-part state's own chain
system, #837; no projection). The plain pinned Newton step (PinnedO.newton_o) is taken at the unperturbed state, at
the seed blocks (--blocks, proj_shift.blocks' names; m_h and a_h by default) perturbed by rel with random signs,
z_i -> z_i (1 + rel s_i), s_i = +-1, for each seed, and (--controls 1) at each block other than m_h and a_h
(reseed.offsets' centre, nodes, Delta) perturbed alone by the same size with the first seed, the centre being the
control. Printed per run: the perturbation's largest move, the step's Delta part and the smallest pivot. The spread is
the largest |dDelta(seed) - dDelta(unperturbed)| over the seeds, exact from the steps' eight parts; the seeds' standard
deviation is printed beside it.

Checks (exit 1 on failure): with --ref, the unperturbed step's Delta part equals it as a double; with --repeat 1, the
unperturbed step taken a second time has the same eight parts of its Delta part and the same smallest pivot (control:
--mutate repeat perturbs the second one's centre data by rel, and the check must fail); with --controls 1, the centre's
and every node's Delta step equal the unperturbed one as a double; with --match, each seed run's dDelta and max|F|
equal, as doubles, those of the stored JSON's run with the same blocks and seed, at the same rel (#850: --blocks centre
against #840's centre control; control: --mutate noop leaves the seed runs unperturbed, and the check must fail on
max|F|). The verdict (not a check): "determined" if every seed's Delta step is within 1e-3 relative of the
unperturbed one, else "not determined".

power (#850): per (D, blocks, seed) over the run JSONs, dev(rel) = dDelta(seed) - dDelta(unperturbed), exact from the
eight parts, with |dev| / move (1 for a Newton step that returns a move of the Delta block alone), and between
neighbouring rels the power p = ln(dev1 / dev2) / ln(rel1 / rel2) (none on a sign change). The
reading: "quadratic" if every p is in --quad, "linear" if every p is in --lin, else "neither". Check (exit 1): the
JSONs of one group have the same unperturbed step, eight parts, so they come from one state and system (control:
--mutate start moves one JSON's unperturbed step by an ulp, and the check must fail).

sum (#830): per run JSON, D, the spread, p_min from #805's TABLE (I805-mac/table.json, the row at the same D to 1e-6),
E = K / p_min and the ratio spread / E in the bands registered on #808: "within a factor 3" (1/3 <= ratio <= 3),
"below 1e-2", or "between". Then the slope of log(spread) against D by least squares, against --slope (#805's 1.44)
and against the slope of log(E) at the same D. --selftest runs the verdict's, the bands' and the power's rules on
planted values; the controls --mutate=bar (the bar at 1e3), --mutate=band (the factor at 30) and --mutate=nosub (the
power read from the seed's dDelta without subtracting the unperturbed one) must fail."""
import json
import os
import sys
import time

import numpy as np

BAR = 1e-3
BAND_F, BAND_LOW = 3.0, 1e-2
QUAD, LIN = (1.8, 2.2), (0.9, 1.1)     # the power's readings (#850)


def verdict(ref, steps, bar=BAR):
    """'determined' if every step is within bar relative of ref, else 'not determined'; and the largest relative
    deviation"""
    dev = max(abs(x / ref - 1) for x in steps)
    return ("determined" if dev <= bar else "not determined"), dev


def band(ratio, f=BAND_F):
    """#808's bands for spread / (K / p_min)"""
    if 1 / f <= ratio <= f:
        return "within a factor 3"
    return "below 1e-2" if ratio < BAND_LOW else "between"


def power_read(pts, quad=QUAD, lin=LIN, mutate=""):
    """pts: [(rel, unperturbed dDelta parts, seed dDelta parts)]. (devs by rel descending, the powers between
    neighbours, the reading); dev exact at 150 digits; --mutate=nosub reads the seed's dDelta alone"""
    import mpmath as mp
    with mp.workdps(150):
        devs = []
        for rel, u, d in sorted(pts, key=lambda t: -t[0]):
            dv = mp.fsum(mp.mpf(p) for p in d) - (0 if mutate == "nosub" else mp.fsum(mp.mpf(p) for p in u))
            devs.append((rel, dv))
        ps = [float(mp.log(a / b) / mp.log(mp.mpf(r1) / r2)) if a * b > 0 else None
              for (r1, a), (r2, b) in zip(devs, devs[1:])]
    inq = [p is not None and quad[0] <= p <= quad[1] for p in ps]
    inl = [p is not None and lin[0] <= p <= lin[1] for p in ps]
    reading = ("none (one rel)" if not ps else "quadratic" if all(inq) else "linear" if all(inl) else "neither")
    return [(r, float(v)) for r, v in devs], ps, reading


def selftest(mutate):
    ok = True
    rels = (1e-65, 1e-66, 1e-67)
    # planted powers: dev = c rel^k beside an unperturbed step u, the seed's parts (u, dev) summing exactly
    for name, u, f, want in (("quadratic, u 1e-41", 1e-41, lambda r: 2.42e92 * r * r, "quadratic"),
                             ("linear, u -4.4e-58", -4.4e-58, lambda r: 6.5e15 * r, "linear"),
                             ("power 1.5", 5e-82, lambda r: 1e50 * r ** 1.5, "neither"),
                             ("a sign change", 0.0, lambda r: 1e30 * r * (1 if r != 1e-66 else -1), "neither")):
        _, ps, got = power_read([(r, [u] + [0.0] * 7, [u, f(r)] + [0.0] * 6) for r in rels], mutate=mutate)
        ok &= got == want
        print(f"  power, {name}: p {', '.join('none' if p is None else f'{p:.4f}' for p in ps)}: {got} (want {want})"
              f"{'' if got == want else '  <- wrong'}")
    bar = 1e3 if mutate == "bar" else BAR
    f = 30.0 if mutate == "band" else BAND_F
    ref = 3.3110848587422764e-8
    cases = [([ref, ref * (1 + 1e-4), ref * (1 - 9e-4), ref], "determined"),
             ([ref, ref * 332.5, ref, ref], "not determined"),
             ([ref * (1 + 2e-3)] * 4, "not determined"),
             ([-ref, ref, ref, ref], "not determined")]
    for steps, want in cases:
        got, dev = verdict(ref, steps, bar)
        ok &= got == want
        print(f"  steps / ref {', '.join(f'{x / ref:.4g}' for x in steps)}: {got}, largest deviation {dev:.3g} "
              f"(want {want}){'' if got == want else '  <- wrong'}")
    for r, want in ((0.5, "within a factor 3"), (2.9, "within a factor 3"), (5e-3, "below 1e-2"), (0.1, "between"),
                    (4.0, "between")):
        got = band(r, f)
        ok &= got == want
        print(f"  ratio {r:g}: {got} (want {want}){'' if got == want else '  <- wrong'}")
    print(f"step_spread selftest{' --mutate=' + mutate if mutate else ''}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def seed_runs(rec):
    """a run JSON's seed runs: those of its --blocks (m_h and a_h in JSONs made before --blocks), controls left out"""
    sel = rec.get("blocks", ["m_h", "a_h"])
    return [r for r in rec["runs"] if r["blocks"] == sel and not r.get("control")]


def spread_of(rec):
    """(spread, sd) of a run JSON's seeds, exact from the eight parts: the largest |dDelta(seed) -
    dDelta(unperturbed)| and the seeds' population standard deviation"""
    import mpmath as mp
    with mp.workdps(150):
        val = [mp.fsum(mp.mpf(p) for p in r["dDelta_parts"]) for r in rec["runs"]]
        u = val[0]
        ids = {id(r) for r in seed_runs(rec)}
        hz = [v for r, v in zip(rec["runs"], val) if id(r) in ids]
        m = mp.fsum(hz) / len(hz)
        return mp.mpf(max(abs(v - u) for v in hz)), mp.sqrt(mp.fsum((v - m) ** 2 for v in hz) / len(hz))


def summary(argv):
    import mpmath as mp
    opts, args, i = dict(K="2.2e-37", slope="1.44", json=""), [], 0
    while i < len(argv):
        if argv[i].startswith("--"):
            if argv[i][2:] not in opts:
                raise SystemExit(f"step_spread sum: unknown option {argv[i]}")
            opts[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            args.append(argv[i])
            i += 1
    table = json.load(open(args[0]))
    K, s805 = float(opts["K"]), float(opts["slope"])
    rows = []
    for path in args[1:]:
        rec = json.load(open(path))
        D = float(mp.mpf(rec["D_str"]))
        t = [r for r in table if abs(r["D"] - D) < 1e-6]
        if len(t) != 1:
            raise SystemExit(f"step_spread sum: {len(t)} rows of the table at D {D}")
        sp, sd = spread_of(rec)
        E = K / t[0]["min_pivot"]
        ratio = float(sp) / E
        rows.append(dict(D=D, run=path, spread=float(sp), sd=float(sd), dDelta0=rec["runs"][0]["dDelta"],
                         piv0=rec["runs"][0]["piv"][0], p_min=t[0]["min_pivot"], E=E, ratio=ratio, band=band(ratio)))
    rows.sort(key=lambda r: r["D"])
    print(f"step_spread sum: K {K:g}, #805's slope {s805:g}; per D: the spread (largest deviation over the seeds, "
          "exact), the seeds' sd, the unperturbed dDelta and smallest pivot, #805's p_min, E = K / p_min, spread / E "
          "and its band")
    for r in rows:
        print(f"  D {r['D']:.4f}: spread {r['spread']:.4e}, sd {r['sd']:.4e}, dDelta {r['dDelta0']:+.4e}, pivot "
              f"{r['piv0']:.3e}, p_min {r['p_min']:.3e}, E {r['E']:.3e}, spread / E {r['ratio']:.3e}: {r['band']}")
    fit = [r for r in rows if r["spread"] > 0]
    out = dict(K=K, slope805=s805, rows=rows)
    if len(fit) >= 2:
        Ds = np.array([r["D"] for r in fit])
        sl = float(np.polyfit(Ds, np.log([r["spread"] for r in fit]), 1)[0])
        se = float(np.polyfit(Ds, np.log([r["E"] for r in fit]), 1)[0])
        out.update(slope=sl, slope_E=se, n_fit=len(fit))
        print(f"  slope of log(spread) against D, {len(fit)} points: {sl:.4f} per unit D; #805's {s805:g} (ratio "
              f"{sl / s805:.3f}); log(E) at the same D {se:.4f} (ratio {sl / se:.3f})")
    bands = {r["band"] for r in rows}
    out["bands"] = ("within a factor 3 at every D" if bands == {"within a factor 3"} else
                    "below 1e-2 at every D" if bands == {"below 1e-2"} else "mixed or between")
    print(f"  bands: {out['bands']}")
    if opts["json"]:
        with open(opts["json"], "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    return 0


def power(argv):
    import mpmath as mp
    opts, args, i = dict(quad=",".join(map(str, QUAD)), lin=",".join(map(str, LIN)), json="", mutate=""), [], 0
    while i < len(argv):
        if argv[i].startswith("--"):
            if argv[i][2:] not in opts:
                raise SystemExit(f"step_spread power: unknown option {argv[i]}")
            opts[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            args.append(argv[i])
            i += 1
    quad, lin = (tuple(float(x) for x in opts[k].split(",")) for k in ("quad", "lin"))
    groups = {}
    for path in args:
        rec = json.load(open(path))
        for r in seed_runs(rec):
            key = (rec["D_str"], tuple(rec.get("blocks", ["m_h", "a_h"])), r["seed"])
            groups.setdefault(key, []).append((rec["rel"], rec["runs"][0], r, os.path.basename(path)))
    if opts["mutate"] == "start":
        # the same-start check's control: the last run of each group read with its unperturbed step one ulp off
        for pts in groups.values():
            u = dict(pts[-1][1])
            u["dDelta_parts"] = [float(np.nextafter(u["dDelta_parts"][0], np.inf))] + u["dDelta_parts"][1:]
            pts[-1] = (pts[-1][0], u) + pts[-1][2:]
    ok, out = True, []
    print(f"step_spread power: per D, blocks and seed, dev = dDelta(seed) - dDelta(unperturbed) at each rel, the power "
          f"p between neighbouring rels, and the reading (quadratic: p in {quad}; linear: p in {lin})")
    for (D_str, bls, seed), pts in sorted(groups.items()):
        same = all(p[1]["dDelta_parts"] == pts[0][1]["dDelta_parts"] for p in pts)
        ok &= same
        devs, ps, reading = power_read([(p[0], p[1]["dDelta_parts"], p[2]["dDelta_parts"]) for p in pts], quad, lin)
        print(f"  D {mp.nstr(mp.mpf(D_str), 10)}, {', '.join(bls)}, seed {seed}: the unperturbed step "
              f"{'the same' if same else 'NOT the same'} in the {len(pts)} runs ({'pass' if same else 'FAIL'})")
        for (rel, dv), p in zip(devs, sorted(pts, key=lambda t: -t[0])):
            print(f"    rel {rel:g} ({p[3]}): move {p[2]['move']:.4e}, dDelta {p[2]['dDelta']:+.6e}, dev {dv:+.6e}, "
                  f"|dev| / move {abs(dv) / p[2]['move'] if p[2]['move'] else float('nan'):.4e}, max|dz| "
                  f"{p[2]['dz']:.4e}, smallest pivot {p[2]['piv'][0]:.6e}, max|F| {p[2]['F']:.4e}")
        print(f"    p {', '.join('none (sign change)' if p is None else f'{p:.4f}' for p in ps) or '-'}: {reading}")
        out.append(dict(D_str=D_str, blocks=list(bls), seed=seed, devs=devs, p=ps, reading=reading, same_start=same))
    if opts["json"]:
        with open(opts["json"], "w") as fh:
            json.dump(dict(quad=quad, lin=lin, groups=out), fh, indent=1)
    print(f"step_spread power: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv):
    if argv and argv[0] == "--selftest":
        return selftest(next((a.split("=", 1)[1] for a in argv[1:] if a.startswith("--mutate=")), ""))
    if argv and argv[0] == "sum":
        return summary(argv[1:])
    if argv and argv[0] == "power":
        return power(argv[1:])
    import mpmath as mp
    from multidouble import od_axpy
    from octo_corrector import o8, o8_any, omax, plant_start, truth_system
    from proj_shift import blocks
    from reseed import offsets
    mp.mp.dps = 150
    raw, path = argv[0], argv[1]
    opt = dict(threads="3", rel="1e-65", seeds="1,2,3,4", ref="", json="", repeat="0", controls="1", mutate="",
               read="auto", held="auto", plan="auto", blocks="m_h,a_h", match="")
    for k in list(opt):
        if "--" + k in argv:
            opt[k] = argv[argv.index("--" + k) + 1]
    unknown = [a for a in argv[2:] if a.startswith("--") and a[2:] not in opt]
    if unknown:
        raise SystemExit(f"step_spread.py: unknown options {unknown}")
    rel, seeds = float(opt["rel"]), [int(x) for x in opt["seeds"].split(",") if x]
    t0 = time.time()
    # the truth and its system as planted builds them (#837): an eight-part state's chain's own system
    z4, meta, (s, sp, sq, so), info = truth_system(raw, path, threads=int(opt["threads"]), read=opt["read"],
                                                   held=opt["held"], plan=opt["plan"])
    M = int(meta["M"])
    K = (len(z4[0]) - 1) // (4 * M) - 1               # reseed.offsets: L = 4 M (K + 1) + 1
    if offsets(M, K)["L"] != len(z4[0]):
        raise SystemExit(f"step_spread.py: no layout at M {M} has the state's length {len(z4[0])}")

    class RS:           # plant_start reads the layout only
        off = offsets(M, K)
    RS.K = K
    zs, _ = plant_start(z4, RS, 0.0, 1, 0)
    z = o8_any(zs)                                   # planted --rel 0's start, as proj_shift takes it
    L = len(z[0])
    bl = {name: (lo, hi) for name, lo, hi in blocks(M, K)}
    sel = tuple(b.strip() for b in opt["blocks"].split(","))       # the seed runs' blocks
    if not sel or any(b not in bl for b in sel):
        raise SystemExit(f"step_spread.py: --blocks {opt['blocks']!r}: the blocks are {', '.join(bl)}")
    lab = ", ".join(sel)
    print(f"step_spread {path}: D {mp.nstr(mp.mpf(meta['D_str']), 15)}, M {M}, K {K}, L {L}; rel {rel:g}, seeds "
          f"{seeds}; pinned max|F| (octo) {omax(so.residual_o(z)):.4e} ({time.time() - t0:.0f} s)", flush=True)
    print(f"  the system (#837): truth read in {info['read']} of {info['file_parts']} parts, held values "
          f"{info['held']}, plan {info['plan']}", flush=True)
    zero8 = [np.zeros(L)] * 7

    def run(name, which, seed, control=False):
        """the plain Newton step at z with the blocks in which perturbed by rel (seed's signs); None: unperturbed.
        --mutate noop leaves the seed runs' state unperturbed (the --match check's control)"""
        dz = np.zeros(L)
        if which and not (opt["mutate"] == "noop" and not control):
            rng = np.random.default_rng(seed)
            sg = rng.choice([-1.0, 1.0], L)
            for b in which:
                lo, hi = bl[b]
                dz[lo:hi] = rel * sg[lo:hi] * (np.asarray(z[0][lo:hi]) + np.asarray(z[1][lo:hi]))
        zp = o8(od_axpy(1.0, np.ascontiguousarray(dz), *zero8, *z)) if which else z
        t1 = time.time()
        st = so.newton_o(zp)
        d = o8(st[0])
        dD = float(sum((mp.mpf(float(p[-1])) for p in d), mp.mpf(0)))
        r = dict(run=name, blocks=list(which or []), seed=seed if which else None, control=control,
                 move=float(np.abs(dz).max()),
                 dDelta=dD, dDelta_parts=[float(p[-1]) for p in d], dz=float(np.abs(d[0]).max()),
                 piv=[st[3], st[4]], F=omax(st[1:3]), s=round(time.time() - t1, 1))
        print(f"  {name:22s} move {r['move']:.3e}: dDelta {dD:+.10e}, max|dz| {r['dz']:.4e}, smallest pivot "
              f"{st[3]:.4e}, max|F| {r['F']:.4e} ({r['s']:.0f} s)", flush=True)
        return r

    runs = [run("unperturbed", None, None)]
    rep = None
    if opt["repeat"] == "1":
        # the same step a second time; --mutate repeat perturbs its centre data instead (the check's control)
        rep = (run("centre, the repeat's control", ("centre",), seeds[0] if seeds else 1) if opt["mutate"] == "repeat"
               else run("unperturbed again", None, None))
    for sd in seeds:
        runs.append(run(f"{lab} seed {sd}", sel, sd))
    if opt["controls"] == "1":
        for b in [n for n in bl if n not in ("m_h", "a_h")]:
            runs.append(run(f"{b} seed {seeds[0]}" + (" (control)" if b == "centre" else ""), (b,), seeds[0], True))
    ref0 = runs[0]["dDelta"]
    ok = True
    rec = dict(state=path, system=info, D_str=meta["D_str"], rel=rel, seeds=seeds, blocks=list(sel), runs=runs,
               mutate=opt["mutate"])
    if opt["ref"]:
        good = ref0 == float(opt["ref"])
        ok &= good
        rec["ref"] = dict(value=float(opt["ref"]), ok=good)
        print(f"  check: the unperturbed dDelta {ref0!r} against --ref {float(opt['ref'])!r}: "
              f"{'equal' if good else 'NOT equal'}: {'pass' if good else 'FAIL'}", flush=True)
    if rep is not None:
        good = rep["dDelta_parts"] == runs[0]["dDelta_parts"] and rep["piv"][0] == runs[0]["piv"][0]
        ok &= good
        rec["repeat"] = dict(run=rep, ok=good)
        print(f"  check: the unperturbed step taken again, its Delta part's eight parts and smallest pivot "
              f"{'the same' if good else 'NOT the same'}: {'pass' if good else 'FAIL'}", flush=True)
    ctl = [r for r in runs if r["control"] and r["blocks"][0] in ("centre",) + tuple(f"node {i}" for i in range(K))]
    if ctl:
        good = all(r["dDelta"] == ref0 for r in ctl)
        ok &= good
        rec["controls_ok"] = good
        print(f"  check: the centre's and the nodes' dDelta equal the unperturbed one as a double "
              f"({sum(r['dDelta'] == ref0 for r in ctl)} of {len(ctl)}): {'pass' if good else 'FAIL'}", flush=True)
    if opt["match"]:
        # each seed run against the run of a stored JSON with the same blocks, seed and rel: dDelta and max|F| equal
        # as doubles (max|F| catches a seed run whose state was not perturbed, --mutate noop)
        mrec = json.load(open(opt["match"]))
        res = []
        for r in seed_runs(rec):
            m = [x for x in mrec["runs"] if x["blocks"] == r["blocks"] and x.get("seed") == r["seed"]]
            good = bool(m) and mrec["rel"] == rel and m[0]["dDelta"] == r["dDelta"] and m[0]["F"] == r["F"]
            res.append(good)
            print(f"  check: {r['run']} against {os.path.basename(opt['match'])}'s "
                  + (f"{m[0]['run']!r}: dDelta {r['dDelta']!r} vs {m[0]['dDelta']!r}, max|F| {r['F']!r} vs "
                     f"{m[0]['F']!r}, rel {rel:g} vs {mrec['rel']:g}" if m else "run (none with these blocks and seed)")
                  + f": {'pass' if good else 'FAIL'}", flush=True)
        good = bool(res) and all(res)
        ok &= good
        rec["match"] = dict(file=opt["match"], ok=good)
    hz = [r["dDelta"] for r in seed_runs(rec)]
    if hz:
        v, dev = verdict(ref0, hz)
        spr, sdv = spread_of(rec)
        rec.update(verdict=v, deviation=dev, range=[min(hz), max(hz)], spread=float(spr), sd=float(sdv))
        print(f"  verdict: {v}; the {lab} seeds' dDelta {min(hz):+.6e} .. {max(hz):+.6e}, ratio to the unperturbed "
              f"{min(h / ref0 for h in hz):.4g} .. {max(h / ref0 for h in hz):.4g}, largest relative deviation "
              f"{dev:.3g} "
              f"(bar {BAR:g}); spread {mp.nstr(spr, 5)}, sd {mp.nstr(sdv, 5)}", flush=True)
    for r in runs[1 + len(seeds):]:
        print(f"  {r['run']}: dDelta / unperturbed {r['dDelta'] / ref0:.6g}", flush=True)
    rec["passed"] = bool(ok)
    if opt["json"]:
        with open(opt["json"], "w") as fh:
            json.dump(rec, fh, indent=1, default=float)
    print(f"step_spread: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
