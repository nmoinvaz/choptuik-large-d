#!/usr/bin/env python3
"""Lemma 6b_1 (proof/A_certify.md section 11.4): a validated tube sweep meant to prove that y+ is analytic on
    { 0.062 <= |xi| <= 0.2, 0 <= arg xi <= 3 pi/8 }   (in u = 1/xi^2:  25 <= |u| <= 260, -3 pi/4 <= arg u <= 0),
for every x in X = x^ + [-h, h]^2. Rigorous where it finishes (ball arithmetic; validated_int.py's Lohner integrator).

STATUS: it does NOT finish at the default tube width (DFRAC = 0.045): the Lohner sets blow up within a fraction of
the first arc, because the family parameter widens every step's Jacobian at first order (section 11.4). Half-width
0.26 at u = 260 (--dfrac=0.001) stays bounded; covering the annulus at that width needs about 600 tubes, about 30 CPU
hours. Lemma 6b_1 is therefore open; this script is the tool, kept with its tests.

The system is that of lemma6_usys.py: the variables (P, V, kappa (C - delta_{nu 0})) of fixed_point.py with u = 1/xi^2 as
the independent variable, where the fast mode is e^{-u/2} with a constant rate. The region is covered by TUBES: for a
reference path u_c(t) (an arc |u| = rho_j, or the start ray arg u = -3 pi/4) and a complex square |Re d|, |Im d| <= Delta,
lemma6_usys.SysUD integrates along every translated path u_c(t) + d at once, with d a constant state of the Lohner set
(the velocity du/dt does not depend on d). A finished run, from a box containing y+(u_c(0) + d) for every d, proves that
y+ continues analytically to the tube { u_c(t) + d } (section 11, the tube argument). Start boxes:
  - for the arcs with rho_j >= 106.8, from the sector start-value lemma (fixed_point.py --sector=0.2 over X, rays
    |arg xi - 3pi/8| <= 0.2, |u| >= 100): y+ = Y_N + eta with |eta_j(z)| <= (rho/w_j) |z|^{k_j}, Y_N evaluated on the
    ball z = 1/(u_c(0) + square);
  - for the arcs with rho_j < 106.8, from a tube along the start ray, from |u| = 107 (start box from the same lemma)
    inward, with the square shrinking at every node.
Reference arcs: rho_0 = 260, rho_{j+1} = 0.925 rho_j, down to 25; Delta_j = 0.045 rho_j, so consecutive tubes overlap.

    python3 lemma6_sweep.py [--h=1e-9] [--only=i,j] [--out=f.json]
    python3 lemma6_sweep.py --dfrac=0.001 --ratio=0.9982 --ray-cache=ray.json --chunk=k/K --out=chunk_k.json
                                                  the brute-force sweep, split into K interleaved chunks (arcs with
                                                  j mod K = k) that can run in parallel; the ray tube chain is computed
                                                  once and shared through --ray-cache. Merge and check the chunks with
                                                  lemma6_sweep_merge.py. Coverage needs ratio >= (1-dfrac)/(1+dfrac):
                                                  consecutive tubes then overlap (a square of half-side D contains the
                                                  disc of radius D, so each arc tube covers the radial band rho_j +- D_j)
    python3 lemma6_sweep.py --dfrac=0.001 --ratio=0.9982 --ray-cache=ray.json --ray-only=1 --out=ray_run.json
                                                  only the shared ray tube chain (run it once, before the chunks)
    python3 lemma6_sweep.py --mutate=sign        mutation test: the fast term with the wrong sign (V_u = +V/2 + ...);
                                                  the sweep must fail
    python3 lemma6_sweep.py --rhos=11,9.5,8.3    negative control: arcs at |xi| = 0.30, 0.32, 0.35 (u = 11, 9.5, 8.3),
                                                  across the singularity of y+ found by review R6; must not all finish
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validated_int as VI  # noqa: E402
import lemma6_common  # noqa: E402
lemma6_common.install()            # a-priori inflation with a common floor (see lemma6_common.py)
import lemma6_usys as U  # noqa: E402
import lemma6_sector as L6S  # noqa: E402
from flint import arb, acb, acb_mat, ctx  # noqa: E402

AH, WH = "2.5421480475", "0.7769940136"
ORDER = 12                          # Taylor order (u-steps are about 1)
DFRAC = "0.045"                     # tube half-width Delta_j = DFRAC rho_j
RAY0 = "107"                        # the ray tube starts at |u| = RAY0 (its square then stays in |u| >= 100)


RATIO = "0.925"                     # rho_{j+1} = RATIO rho_j


def nodes_u(rmax="260", rmin="25", ratio=None):
    ratio = ratio or RATIO
    out, r = [arb(rmax)], arb(rmax)
    while True:
        r = r * arb(ratio)
        if r < arb(rmin) * arb("1.0001"):
            break
        out.append(arb(r.mid().str(6)))           # short decimals: exact enough, any value works
    out.append(arb(rmin))
    return out


def square(d):
    return acb(arb(0, d), arb(0, d))


def start_box(old, w, A, uc0, D):
    """box (u-variables) containing y+(uc0 + d) for every d in the square of half-side D, from the sector lemma"""
    P, Q, C, k, rho, wts, comps = (old[x] for x in ("P", "Q", "C", "k", "rho", "wts", "comps"))
    z = 1 / (uc0 + square(D))
    umin = abs(uc0).lower() - arb(2).sqrt() * D
    if not umin >= arb(100):
        raise ValueError("start square leaves |u| >= 100")
    zr = 1 / arb(umin)
    eb = {cp: rho / arb(wts[i]) for i, cp in enumerate(comps)}

    def disc(e):
        r = arb(0, e.upper())
        return acb(r, r)
    H = [P[mu](z) + disc(eb[("p", mu)] * zr ** k) for mu in (-1, 1)]
    for mu in (-1, 1):
        H.append(Q[mu](z) - (1 + acb(0, 1) * mu * acb(w)) * P[mu](z) + disc(eb[("v", mu)] * zr ** k))
    for nu in (-2, 0, 2):
        H.append(U.KAPPA * (C[nu](z) - (1 if nu == 0 else 0) + disc(eb[("c", nu)] * zr ** (k + 1))))
    return H + [acb(w), acb(A)]


def tube_set(s, H, D):
    """Lohner set with box H x square(D) (Q = I)"""
    Hd = list(H[:9]) + [square(D)]
    n = s.n
    c = [acb(z.mid()) for z in Hd]
    q = [Hd[i] - c[i] for i in range(n)]
    Q = acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    return VI.Lohner(s, c, Q, q, ORDER)


def run_tube(lo, leg):
    t0 = time.time()
    try:
        n = lo.run(leg, 1e-3)
        ok = True
        err = ""
    except RuntimeError as e:
        n, ok, err = -1, False, str(e)
    X = lo.box()
    rad = max(VI.rad_of(z) for z in X[:7]) if ok else float("inf")
    return ok and rad < 1e3, n, time.time() - t0, X, rad, err


def acb_s(z):
    return [z.real.str(600, radius=True, more=True), z.imag.str(600, radius=True, more=True)]


def acb_p(p):
    return acb(arb(p[0]), arb(p[1]))


def sweep(h="1e-9", only=None, mutate=None, rhos=None, ray_cache=None, out=None):
    t00 = time.time()
    ctx.prec = 320
    Ahat, wfhat = arb(AH), arb(WH)
    A, wf = Ahat + arb(0, arb(h)), wfhat + arb(0, arb(h))
    w = VI.W0 * wf
    old = L6S.sector_old(h, verbose=False)
    if not old["ok"]:
        raise RuntimeError("sector start-value lemma failed")
    s = U.SysUD(mutate=(mutate == "sign"))
    rho = [arb(x) for x in rhos] if rhos else nodes_u()
    D = [arb(DFRAC) * r for r in rho]
    E3 = (acb(0, -1) * U.PHI_U).exp()                    # e^{-3 i pi/4}
    res = {"h": h, "mutate": mutate, "order": ORDER, "kappa": U.KAPPA, "dfrac": DFRAC, "ratio": RATIO,
           "coverage_ok": bool(arb(RATIO) >= (1 - arb(DFRAC)) / (1 + arb(DFRAC))) if not rhos else None,
           "rho": [r.str(8) for r in rho], "arcs": [], "ray": []}
    print(f"tube sweep over X = x^ + [-{h}, {h}]^2: {len(rho)} reference arcs |u| = {rho[0].str(4)} ... "
          f"{rho[-1].str(4)}{', MUTATION ' + mutate if mutate else ''} [{time.time() - t00:.0f} s]", flush=True)
    starts = {}
    ray_nodes = [j for j, r in enumerate(rho) if not r >= arb("106.8")]
    for j, r in enumerate(rho):
        if j not in ray_nodes:
            ctx.prec = 320
            starts[j] = start_box(old, w, A, acb(r) * E3, D[j])
    cache = None
    if ray_cache and os.path.exists(ray_cache):
        with open(ray_cache) as fh:
            cache = json.load(fh)
        assert (cache["h"], cache["dfrac"], cache["ratio"], cache["rho"]) == (h, DFRAC, RATIO, res["rho"]), \
            "ray cache made with other parameters"
        if not cache["ok"]:
            raise RuntimeError("the cached ray tube chain did not finish")
        for j, H in cache["starts"].items():
            starts[int(j)] = [acb_p(x) for x in H]
        res["ray"] = cache["ray"]
        print(f"ray tube chain loaded from {ray_cache} ({len(cache['starts'])} start boxes)", flush=True)
    if ray_nodes and cache is None:
        ctx.prec = 320
        Dr = arb(DFRAC) * arb(RAY0)
        H = start_box(old, w, A, acb(arb(RAY0)) * E3, Dr)
        prev = arb(RAY0)
        for j in ray_nodes:
            ctx.prec = VI.PREC
            lo = tube_set(s, H, D[j])                     # the square shrinks to D_j (<= the previous one)
            ok, n, sec, X, rad, err = run_tube(lo, U.URay(prev, rho[j]))
            res["ray"].append({"to": rho[j].str(8), "ok": ok, "steps": n, "sec": sec, "rad": rad, "err": err})
            print(f"ray tube |u| {prev.str(5)} -> {rho[j].str(5)} (half-width {D[j].str(4)}): finished {ok}, {n} steps, "
                  f"{sec:.0f} s, radius {rad:.2e} {err}", flush=True)
            if not ok:
                break
            H, prev = X[:9], rho[j]
            starts[j] = list(H)
        if ray_cache:
            with open(ray_cache, "w") as fh:
                json.dump({"h": h, "dfrac": DFRAC, "ratio": RATIO, "rho": res["rho"], "ray": res["ray"],
                           "ok": all(a["ok"] for a in res["ray"]),
                           "starts": {str(j): [acb_s(z) for z in starts[j]] for j in ray_nodes if j in starts}}, fh)
    for j, r in enumerate(rho):
        if only is not None and j not in only:
            continue
        if j not in starts:
            res["arcs"].append({"j": j, "ok": False, "err": "no start box"})
            continue
        ctx.prec = VI.PREC
        lo = tube_set(s, starts[j], D[j])
        rad0 = max(VI.rad_of(z) for z in starts[j][:7])
        ok, n, sec, X, rad, err = run_tube(lo, U.UArc(r))
        rec = {"j": j, "rho": r.str(8), "xi": float((1 / r.sqrt()).mid()), "half_width": D[j].str(6), "ok": ok,
               "steps": n, "sec": sec, "start_rad": rad0, "end_rad": rad, "err": err,
               "y_end": [[z.real.str(20, radius=True), z.imag.str(20, radius=True)] for z in X[:7]] if ok else None}
        res["arcs"].append(rec)
        if out:                                   # write after every arc, so a killed run keeps what it finished
            res["ok"] = False
            res["sec"] = time.time() - t00
            with open(out, "w") as fh:
                json.dump(res, fh, indent=1)
        print(f"arc tube |u| = {r.str(5)} (|xi| = {rec['xi']:.4f}, half-width {D[j].str(4)}): finished {ok}, {n} steps, "
              f"{sec:.0f} s; start radius {rad0:.2e}, end radius {rad:.2e} {err}", flush=True)
    res["ok"] = all(a["ok"] for a in res["arcs"]) and all(a["ok"] for a in res["ray"]) and \
        (only is not None or len(res["arcs"]) == len(rho))
    res["sec"] = time.time() - t00
    print(f"tube sweep: every tube finished: {res['ok']} ({res['sec']:.0f} s)")
    return res


if __name__ == "__main__":
    opts = {a.split("=", 1)[0][2:]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--")}
    only = [int(x) for x in opts["only"].split(",")] if "only" in opts else None
    DFRAC = opts.get("dfrac", DFRAC)
    RATIO = opts.get("ratio", RATIO)
    rhos = opts["rhos"].split(",") if "rhos" in opts else None
    if "chunk" in opts:                          # k/K: the arcs with j mod K = k
        k, K = (int(x) for x in opts["chunk"].split("/"))
        nn = len(rhos) if rhos else len(nodes_u())
        only = [j for j in range(nn) if j % K == k]
    if opts.get("ray-only"):
        only = []
    res = sweep(opts.get("h", "1e-9"), only, opts.get("mutate"), rhos, opts.get("ray-cache"), opts.get("out"))
    res["chunk"] = opts.get("chunk")
    if "out" in opts:
        with open(opts["out"], "w") as fh:
            json.dump(res, fh, indent=1)
