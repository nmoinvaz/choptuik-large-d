"""The zero x1 of S1 near x = 13.5 at w = 1.09, 1.10, 1.11 and the sign change of Im x1, by lapl.py (issue #203).

    python3 second_zero.py real --dir <raw clone>/I82-win2 [--out FILE]
    python3 second_zero.py planted --wp 1.095 --seeds 1,2,3,4,5 [--out FILE]
    python3 second_zero.py planted --wp 1.15 --seeds 1,2,3,4,5 [--out FILE]

Estimator v2.  At each node every approximant of lapl.run_node is tried and admitted or not by lapl.admissible.  The
primary set at a node is its admitted qhp approximants that use 21, 22 or 23 coefficients, pooled.
  x1 at a node      = median (real and imaginary parts separately) over the primary set;
                      +- = half the range over the primary set (a spread over degree triples and numbers of
                      coefficients, not an error bound);
  the crossing      = the root in [1.09, 1.11] of the quadratic through the three medians of Im x1;
  its envelope      = the roots for the 8 shifts of the three medians by +- their half-ranges (the same spread);
  order sets        = the same from the qhp sets of 23, 22 and 21 coefficients separately (a truncation indicator);
  cross-check       = the same from the linear Pade sets (23, 22, 21 coefficients, pooled and separately).
Approximants admitted at all three nodes give their own crossings too (information only).
v1 (code 0d352f4) took the 23-coefficient qhp set alone as primary.  It failed P1 at seed 3: at w = 1.11 the 1e-12 noise
gave every 23-coefficient qhp approximant a far singularity (growth 2.7-6.3 against the A2 bound 2.3), so the primary
set was empty there (I203-mac/P1_stdout.txt).  v2 pools the orders instead; the admission rules are unchanged.
Exit status: planted mode exits 1 when its control fails (P1: |w_c - w_p| > 0.005 for any seed or no crossing;
P2: a sign change of the primary median Im x1 in 1.09-1.11 for any seed).
"""
import json
import os
import statistics
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lapl  # noqa: E402

WS = ["1.09", "1.10", "1.11"]


def quad_root(ws, ys, lo=1.09, hi=1.11):
    """roots in [lo, hi] of the quadratic through (ws, ys)"""
    import numpy as np
    w = np.array(ws, float)
    y = np.array(ys, float)
    co = np.polyfit(w - 1.10, y, 2)
    rts = [r.real + 1.10 for r in np.roots(co) if abs(r.imag) < 1e-12]
    return sorted(r for r in rts if lo - 1e-12 <= r <= hi + 1e-12)


def members(node_out):
    """{key: [rec at 1.09, 1.10, 1.11]} for approximants admitted at all three nodes"""
    keys = None
    by = []
    for recs in node_out:
        d = {(r["kind"], tuple(r["deg"]), r["nd"]): r for r in recs if r.get("ok")}
        by.append(d)
        keys = set(d) if keys is None else keys & set(d)
    return {k: [b[k] for b in by] for k in sorted(keys)}


def summarize(node_out, kind, nds, ws=WS):
    """Per node: median and half-range of x1 over the admitted approximants of kind with nd in nds at that node.
    Crossing: the root in [1.09, 1.11] of the quadratic through the three per-node medians of Im x1; its envelope: the
    roots for the 8 shifts of the medians by +-(their half-range).  Members admitted at all three nodes: their own
    crossings (information only)."""
    import itertools
    wsf = [float(w) for w in ws]
    node = {}
    for w, recs in zip(ws, node_out):
        xs = [r["x1"] for r in recs if r.get("ok") and r["kind"] == kind and r["nd"] in nds]
        if not xs:
            return {"n": 0, "empty_node": w}
        re = [x[0] for x in xs]
        im = [x[1] for x in xs]
        node[w] = {"n": len(xs), "re": statistics.median(re), "im": statistics.median(im),
                   "re_half_range": (max(re) - min(re)) / 2, "im_half_range": (max(im) - min(im)) / 2}
    med_im = [node[w]["im"] for w in ws]
    hr = [node[w]["im_half_range"] for w in ws]
    out = {"n": min(node[w]["n"] for w in ws), "node": node}
    rts = quad_root(wsf, med_im)
    if rts:
        out["cross"] = rts[0]
    env = []
    for s in itertools.product([-1, 1], repeat=3):
        r = quad_root(wsf, [m + si * h for m, si, h in zip(med_im, s, hr)])
        env.append(r[0] if r else None)
    out["cross_envelope"] = [min(e for e in env if e is not None), max(e for e in env if e is not None)] \
        if any(e is not None for e in env) else None
    out["envelope_all_cross"] = all(e is not None for e in env)
    out["median_im_sign_change"] = (min(med_im) < 0 < max(med_im))
    mem = {k: v for k, v in members(node_out).items() if k[0] == kind and k[2] in nds}
    per = []
    for k, recs in mem.items():
        r = quad_root(wsf, [x["x1"][1] for x in recs])
        per.append({"deg": list(k[1]), "x1": [x["x1"] for x in recs], "cross": r})
    out["members"] = per
    cr = [p["cross"][0] for p in per if p["cross"]]
    out["n_members"] = len(per)
    out["n_members_cross"] = len(cr)
    if cr:
        out["members_cross_range"] = [min(cr), max(cr)]
    return out


def run_three(sigs, x0=13.5, log=print):
    node_out, diag = [], []
    for w, sig in zip(WS, sigs):
        t0 = time.time()
        recs, dg = lapl.run_node(sig, x0=x0)
        nok = sum(1 for r in recs if r.get("ok"))
        log(f"  node {w}: {nok}/{len(recs)} admitted, zeta_DS {dg['zeta_ds']}, gamma_DS {dg['gamma_ds']} "
            f"[{time.time() - t0:.0f}s]")
        for r in recs:
            if not r.get("ok"):
                log(f"    rejected {r['kind']} {r['deg']} nd {r['nd']}: {r['why']}")
        node_out.append(recs)
        diag.append(dg)
    res = {"diag": diag, "nodes": node_out}
    for name, kind, nds in SETS:
        res[name] = summarize(node_out, kind, nds)
    return res


SETS = [("primary", "qhp", (21, 22, 23)), ("qhp23", "qhp", (23,)), ("qhp22", "qhp", (22,)), ("qhp21", "qhp", (21,)),
        ("pade", "pade", (21, 22, 23)), ("pade23", "pade", (23,)), ("pade22", "pade", (22,)), ("pade21", "pade", (21,))]


def fmt_summary(res, log=print):
    for name, _, _ in SETS:
        s = res[name]
        if not s["n"]:
            log(f"  {name}: no approximant admitted at node {s.get('empty_node')}")
            continue
        nodes = "  ".join(f"{w} (n {s['node'][w]['n']}): {s['node'][w]['re']:.5f} {s['node'][w]['im']:+.5f}i "
                          f"(+-{s['node'][w]['re_half_range']:.1e}, +-{s['node'][w]['im_half_range']:.1e}i)" for w in WS)
        env = s["cross_envelope"]
        cr = (f"crossing {s['cross']:.5f}" if "cross" in s else "no crossing of the medians") + \
             (f", envelope {env[0]:.5f}-{env[1]:.5f}{'' if s['envelope_all_cross'] else ' (some shifts: none)'}"
              if env else ", envelope: no crossing")
        mc = (f"; members at all nodes {s['n_members']}, {s['n_members_cross']} cross"
              + (f" in {s['members_cross_range'][0]:.5f}-{s['members_cross_range'][1]:.5f}" if s['n_members_cross'] else ""))
        log(f"  {name}: {cr}; median Im sign change {s['median_im_sign_change']}{mc}")
        log(f"     x1 {nodes}")


def main(argv):
    mode = argv[1]
    opt = {}
    i = 2
    while i < len(argv):
        opt[argv[i].lstrip("-")] = argv[i + 1]
        i += 2
    out = opt.get("out")
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    mp.mp.dps = lapl.DPS
    if mode == "real":
        D = opt["dir"]
        sigs = [lapl.load_sigma(os.path.join(D, f"sigF_w{w}_K45_J1500_p1280.json"))[0] for w in WS]
        mp.mp.dps = lapl.DPS
        log(f"real data from {D}")
        res = run_three(sigs, log=log)
        fmt_summary(res, log=log)
        if out:
            json.dump(res, open(out, "w"), indent=1, default=str)
        return 0
    if mode == "planted":
        import planted
        wp = float(opt["wp"])
        seeds = [int(s) for s in opt.get("seeds", "1").split(",")]
        eps = opt.get("eps", "0.05")
        fam = planted.Family(wp, eps=eps)
        log(f"planted family w_p {wp} eps {eps}: da {mp.nstr(fam.da, 12)}")
        truth = {}
        for w in WS + [f"{wp}"]:
            x = fam.x1(mp.mpf(w))
            truth[w] = [float(x.real), float(x.imag)]
            log(f"  true x1({w}) = {mp.nstr(x, 12)}")
        tim = [truth[w][1] for w in WS]
        true_sign_change = min(tim) < 0 < max(tim)
        log(f"  true Im x1 sign change in 1.09-1.11: {true_sign_change}")
        exact = [fam.sigma(mp.mpf(w)) for w in WS]
        ok_all = True
        allres = {"wp": wp, "truth": truth, "seeds": {}}
        for seed in seeds:
            mp.mp.dps = 50
            sigs = [planted.noisy(s, seed + 1000 * j) for j, s in enumerate(exact)]
            mp.mp.dps = lapl.DPS
            log(f" seed {seed}")
            res = run_three(sigs, log=log)
            fmt_summary(res, log=log)
            p = res["primary"]
            if abs(wp - 1.095) < 1e-9:
                ok = "cross" in p and abs(p["cross"] - wp) <= 0.005
                log(f"  P1 seed {seed}: crossing {p.get('cross')} vs {wp}: {'pass' if ok else 'FAIL'}")
            elif abs(wp - 1.15) > 1e-9:
                # a development or stress family, not a control: report the crossing error only
                ok = "cross" in p and abs(p["cross"] - wp) <= 0.005
                log(f"  stress seed {seed}: crossing {p.get('cross')} vs {wp}: "
                    f"error {p['cross'] - wp if 'cross' in p else None}")
            else:
                ok = p["n"] > 0 and not p["median_im_sign_change"]
                p1_on_p2 = "cross" in p and abs(p["cross"] - 1.095) <= 0.005
                log(f"  P2 seed {seed}: median Im x1 sign change {p.get('median_im_sign_change')}: "
                    f"{'pass' if ok else 'FAIL'}; P1's criterion on this family (must fail): "
                    f"{'pass (bad)' if p1_on_p2 else 'fails (good)'}")
            for w in WS:
                if p["n"]:
                    e = complex(p["node"][w]["re"], p["node"][w]["im"]) - complex(*truth[w])
                    log(f"   x1 error at {w}: {e.real:+.2e} {e.imag:+.2e}i "
                        f"(spread +-{p['node'][w]['re_half_range']:.1e}, +-{p['node'][w]['im_half_range']:.1e}i)")
            ok_all &= ok
            allres["seeds"][seed] = res
        log(f"control w_p {wp}: {'PASS' if ok_all else 'FAIL'}")
        if out:
            json.dump(allres, open(out, "w"), indent=1, default=str)
            open(out.replace(".json", ".log"), "w").write("\n".join(lines) + "\n")
        return 0 if ok_all else 1
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
