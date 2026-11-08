"""The zero x1 of S1 near x = 13.5 at w = 1.09, 1.10, 1.11 and the sign change of Im x1, by lapl.py (issue #203).

    python3 second_zero.py real --dir <raw clone>/I82-win2 [--out FILE]
    python3 second_zero.py planted --wp 1.095 --seeds 1,2,3,4,5 [--out FILE]
    python3 second_zero.py planted --wp 1.15 --seeds 1,2,3,4,5 [--out FILE]

Estimator (fixed before the controls and the real data were run).  At each node every approximant of lapl.run_node is
tried; a member is one approximant (kind, degrees, number of coefficients used) admitted at all three nodes.  For each
member, Im x1 at the three nodes is interpolated by the quadratic through them, and its root in [1.09, 1.11] is that
member's crossing (none if it has no root there).  The primary set is the qhp members that use all 23 coefficients.
  x1 at a node      = median (real and imaginary parts separately) over the primary set;
                      +- = half the range over the primary set (a spread over degree choices, not an error bound);
  the crossing      = median of the primary members' crossings, reported when more than half of them have one;
                      +- = half their range (spread over degree choices);
  order shift       = the same medians from the qhp members using 22 and 21 coefficients (a truncation indicator);
  cross-check       = the same from the linear Pade members (23 coefficients).
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


def summarize(node_out, kind, nd, ws=WS):
    mem = {k: v for k, v in members(node_out).items() if k[0] == kind and k[2] == nd}
    if not mem:
        return {"n": 0}
    wsf = [float(w) for w in ws]
    xs = {w: [] for w in ws}
    cross = []
    nsign = 0
    per = []
    for k, recs in mem.items():
        ys = [r["x1"][1] for r in recs]
        for w, r in zip(ws, recs):
            xs[w].append(r["x1"])
        rts = quad_root(wsf, ys)
        sign_change = (ys[0] > 0) != (ys[-1] > 0) or len(rts) > 0
        if rts:
            nsign += 1
            cross.append(rts[0])
        per.append({"deg": list(k[1]), "x1": [r["x1"] for r in recs], "cross": rts, "sign_change": sign_change})
    node = {}
    for w in ws:
        re = [x[0] for x in xs[w]]
        im = [x[1] for x in xs[w]]
        node[w] = {"re": statistics.median(re), "im": statistics.median(im),
                   "re_half_range": (max(re) - min(re)) / 2, "im_half_range": (max(im) - min(im)) / 2}
    out = {"n": len(mem), "node": node, "n_cross": nsign, "members": per}
    if nsign * 2 > len(mem):
        out["cross"] = statistics.median(cross)
        out["cross_half_range"] = (max(cross) - min(cross)) / 2
        out["cross_range"] = [min(cross), max(cross)]
    med_im = [node[w]["im"] for w in ws]
    out["median_im_sign_change"] = (min(med_im) < 0 < max(med_im))
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
    for name, kind, nd in [("primary", "qhp", 23), ("qhp22", "qhp", 22), ("qhp21", "qhp", 21),
                           ("pade23", "pade", 23), ("pade22", "pade", 22), ("pade21", "pade", 21)]:
        res[name] = summarize(node_out, kind, nd)
    return res


def fmt_summary(res, log=print):
    for name in ["primary", "qhp22", "qhp21", "pade23", "pade22", "pade21"]:
        s = res[name]
        if not s["n"]:
            log(f"  {name}: no member admitted at all three nodes")
            continue
        nodes = "  ".join(f"{w}: {s['node'][w]['re']:.5f} {s['node'][w]['im']:+.5f}i "
                          f"(+-{s['node'][w]['re_half_range']:.1e}, +-{s['node'][w]['im_half_range']:.1e}i)" for w in WS)
        cr = (f"crossing {s['cross']:.5f} +- {s['cross_half_range']:.5f} (range {s['cross_range'][0]:.5f}-"
              f"{s['cross_range'][1]:.5f})" if "cross" in s else "no majority crossing")
        log(f"  {name}: {s['n']} members, {s['n_cross']} with a root in 1.09-1.11; {cr}; median Im sign change "
            f"{s['median_im_sign_change']}")
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
        fam = planted.Family(wp)
        log(f"planted family w_p {wp}: da {mp.nstr(fam.da, 12)}")
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
