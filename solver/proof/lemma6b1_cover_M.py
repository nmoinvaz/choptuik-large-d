#!/usr/bin/env python3
"""Lemma 6b_1 for M harmonics (proof/A_lemmas_M.md section 5): y+ is analytic on the annular sector
    A1 = { u_in <= |u| <= u_out, -3pi/4 <= arg u <= 0 }   (u = 1/xi^2; default 25 <= |u| <= 260, 0.062 <= |xi| <= 0.2)
for every x in the parameter box X, by the covering of proof/A_certify.md section 12 with validated Taylor discs.
Rigorous (ball arithmetic) where the runs and the check pass.

What carries over verbatim from M = 1 (lemma6b1_cover.py): the geometry (rows of polar tiles, the disc-in-tile and
overlap conditions C2-C8), lemma G (gluing) and the check code itself: `check` here calls lemma6b1_cover.check
unchanged, with its design and run directory pointed at this module's. What changes: the system (lemma6_usys_M.SysUM,
nd = 2(M + 1) + (M + 2) components), the disc lemma (lemma6b1_disc_M.certify, the same lemma), the start box (the
old sector lemma of fixed_point_M over X, diag splitting) and G0, whose radius comes from lemma S at M harmonics
(lemma6_sector_M.py): G0 = { |u| > g0, -3pi/4 - 0.4 < arg u < 0 } with g0 = 1/(R+ sin 0.4) (257 for R+ = 0.01), and
the disc radii and orders, set by the distance to the singularities of y+ (lemma6b1_estimate_M.py).

    python3 lemma6b1_cover_M.py design --M=3                  rows and tiles (no computation)
    python3 lemma6b1_cover_M.py probe --M=3                   NOT rigorous: discs at double-precision centre values
                                                              (inflated boxes), to choose disc radii and orders
    python3 lemma6b1_cover_M.py ray --M=3 [--h=1e-6]          start box and the Lohner run along the start ray
    python3 lemma6b1_cover_M.py row --M=3 --j=J [--h=1e-6]    arc j: Lohner run and its disc certificates
    python3 lemma6b1_cover_M.py rows --M=3 [--from=J] [--only=J1,J2]   every row not yet saved
    python3 lemma6b1_cover_M.py check --M=3 [--h=1e-6] [--json=f.json] [--drop=j,k]
    python3 lemma6b1_cover_M.py mutate --M=3 [--j=J]          the fast term with the wrong sign in every disc of row j:
                                                              no disc may be certified
Runs are saved in lemma6b1_runs_M<M><tag>/ (ray.json, row_JJ.json).
Every command takes --x=A,wf,B3,C3,.. (the box centre, decimal strings; default the stored hier.py zero, recorded in
ray.json and checked by the rows) and --tag=_name (a separate run directory), for running on the point certificate's
own box: e.g. --x=<its p^> --h=1e-9 --tag=_cert (an axis-aligned box containing its tilted box).
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validated_int as VI  # noqa: E402
import lemma6_common  # noqa: E402
lemma6_common.install()            # a-priori inflation with a common floor (see lemma6_common.py)
import lemma6b1_cover as LC  # noqa: E402
import lemma6_usys_M as UM  # noqa: E402
import lemma6b1_disc_M as LDM  # noqa: E402
import lemma6_sector_M as L6SM  # noqa: E402
import fixed_point_M as FPM  # noqa: E402
from flint import arb, acb, acb_mat, ctx  # noqa: E402

ORDER = 12
DISC_PREC = 128
GRID = 2.0 ** -30
# per M: the first row radius rho0 (> g0), the annulus, G0's radius (from lemma S), the parameter box half-width
CONF = {
    1: dict(rho0=262, u_in=25, u_out=260, g0="257"),
    3: dict(rho0=262, u_in=25, u_out=260, g0="257"),     # lemma S at R+ = 0.01 holds at M = 3 (lemma6_sector_M.py)
    5: dict(rho0=405, u_in=25, u_out=260, g0="402"),     # lemma S needs R+ <= 0.0064 at M = 5 (g0 = 401.2)
}


TAG = ""                            # --tag: a separate run directory (e.g. for the point certificate's own box)


def runs_dir(M):
    return os.path.join(HERE, f"lemma6b1_runs_M{M}{TAG}")


def disc_params(M, rho):
    """(disc radius r, Taylor order n) by row radius; M = 1 is lemma6b1_cover.disc_params. M >= 3: y+ is larger and
    its singularities nearer at the inner rows (lemma6b1_estimate_M.py: 12.7 against 15 at 25 e^{-3 i pi/4}), so the
    inner rows use smaller discs (chosen with `probe`, lemmas_M_runs/probe_M3.log)."""
    if M == 1:
        return LC.disc_params(rho)
    if rho >= 100:
        return 24, 70
    if rho >= 50:
        return 16, 60
    if rho >= 32:
        return 10, 50
    return 6, 50                     # probe: r = 8 fails at 25.96 e^{-3 i pi/4} (Lam 1.34), r = 6 passes (Lam 0.20)


def design(M):
    """lemma6b1_cover.design with this M's disc parameters and first row (the same construction, verbatim)"""
    cf = CONF[M]
    rows = []
    rho = float(cf["rho0"])
    pi34 = 3 * math.pi / 4
    while True:
        r, n = disc_params(M, rho)
        dl = 0.6 * r
        ca = 1 - ((0.92 * r) ** 2 - dl ** 2) / (2 * rho * (rho + dl))
        amax = math.acos(ca)
        K = math.ceil(pi34 / (1.6 * amax))
        dphi = pi34 / K
        alpha = 0.6 * dphi
        tks = [round(k / K / GRID) * GRID for k in range(K + 1)]
        rows.append(dict(j=len(rows), rho=rho, r=r, n=n, a=rho - dl, b=rho + dl, K=K, alpha=alpha, t=tks))
        if rho - dl < cf["u_in"] - 0.5:
            break
        rn = rho - dl
        for _ in range(50):
            rr, _n = disc_params(M, rn)
            dn = 0.6 * rr
            rn_new = (rho - dl) + 0.15 * min(dl, dn) - dn
            if abs(rn_new - rn) < 1e-9:
                break
            rn = rn_new
        rho = round(rn, 3)
    return rows


class patched_check:
    """point lemma6b1_cover.check at this M's design, run directory and G0 (its code is used unchanged)"""

    def __init__(self, M):
        self.M = M

    def __enter__(self):
        cf = CONF[self.M]
        self.saved = (LC.design, LC.RUNS, LC.U_IN, LC.U_OUT, LC.G0_RAD, LC.RHO0)
        LC.design = lambda: design(self.M)
        LC.RUNS = runs_dir(self.M)
        LC.U_IN, LC.U_OUT, LC.G0_RAD, LC.RHO0 = cf["u_in"], cf["u_out"], cf["g0"], cf["rho0"]
        return self

    def __exit__(self, *exc):
        (LC.design, LC.RUNS, LC.U_IN, LC.U_OUT, LC.G0_RAD, LC.RHO0) = self.saved


def maxrad(X, nd):
    return max(VI.rad_of(z) for z in X[:nd])


def params(M, h):
    ctx.prec = 320
    xs = FPM.ZEROS[M]
    xb = [arb(v, arb(h)) if float(h) > 0 else arb(v) for v in xs]
    return xb, FPM.W0() * xb[1]


def start_box(M, old, w, A, u0):
    """box (SysUM variables) containing y+(u0) for every x in X, from the old sector lemma over X"""
    P, Q, C, k, rho, wts, comps = (old[x] for x in ("P", "Q", "C", "k", "rho", "wts", "comps"))
    z = 1 / u0
    umin = abs(u0).lower()
    if not arb(umin) >= arb(100):
        raise ValueError("start point outside |u| >= 100 (the old sector lemma's radius 0.1)")
    zr = 1 / arb(umin)
    eb = {cp: rho / arb(wts[i]) for i, cp in enumerate(comps)}
    odd, even = UM.harmonics(M)

    def disc(e):
        r = arb(0, e.upper())
        return acb(r, r)
    H = [P[mu](z) + disc(eb[("p", mu)] * zr ** k) for mu in odd]
    for mu in odd:
        H.append(Q[mu](z) - (1 + acb(0, 1) * mu * acb(w)) * P[mu](z) + disc(eb[("v", mu)] * zr ** k))
    for nu in even:
        H.append(UM.KAPPA * (C[nu](z) - (1 if nu == 0 else 0) + disc(eb[("c", nu)] * zr ** (k + 1))))
    return H + [acb(w), acb(A)]


def lohner_from_box(M, H):
    s = UM.SysUM(M)
    n = s.n
    c = [acb(z.mid()) for z in H]
    q = [H[i] - c[i] for i in range(n)]
    Q = acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    return VI.Lohner(s, c, Q, q, ORDER)


def ray_run(M, h):
    t00 = time.time()
    rows = design(M)
    xb, w = params(M, h)
    old = L6SM.sector_old_M(M, h, d="0.2", r0="0.1", diag=True, verbose=False)
    if not old["ok"]:
        raise RuntimeError("sector start-value lemma failed")
    nd = UM.SysUM(M).nd
    E3 = (acb(0, -1) * UM.PHI_U).exp()
    ctx.prec = 320
    H = start_box(M, old, w, xb[0], acb(arb(rows[0]["rho"])) * E3)
    out = {"M": M, "h": h, "x": list(FPM.ZEROS[M]), "rho": [rw["rho"] for rw in rows],
           "starts": {"0": [LC.s_acb(z) for z in H]}, "start_rad": maxrad(H, nd), "legs": [],
           "lip_old": float(old["lip"].upper())}
    print(f"M = {M}: start box at |u| = {rows[0]['rho']} on the start ray (old sector lemma over X = x^ +- {h}, "
          f"contraction {float(old['lip'].upper()):.4f}): radius {maxrad(H, nd):.2e}", flush=True)
    ctx.prec = VI.PREC
    lo = lohner_from_box(M, H)
    os.makedirs(runs_dir(M), exist_ok=True)
    for j in range(1, len(rows)):
        t0 = time.time()
        leg = UM.URay(str(rows[j - 1]["rho"]), str(rows[j]["rho"]))
        n = lo.run(leg, 1e-2)
        X = lo.box()
        out["starts"][str(j)] = [LC.s_acb(z) for z in X]
        out["legs"].append({"to": rows[j]["rho"], "steps": n, "sec": time.time() - t0, "rad": maxrad(X, nd)})
        print(f"ray |u| {rows[j - 1]['rho']} -> {rows[j]['rho']}: {n} steps, {time.time() - t0:.0f} s, state radius "
              f"{maxrad(X, nd):.2e}", flush=True)
        out["sec"] = time.time() - t00
        with open(os.path.join(runs_dir(M), "ray_partial.json"), "w") as fh:
            json.dump(out, fh, indent=1)
    out["sec"] = time.time() - t00
    with open(os.path.join(runs_dir(M), "ray.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def row_run(M, j, h, mutate=None):
    t00 = time.time()
    rows = design(M)
    rw = rows[j]
    with open(os.path.join(runs_dir(M), "ray.json")) as fh:
        ray = json.load(fh)
    assert ray["h"] == h and ray["rho"] == [x["rho"] for x in rows], "ray.json made with another design or box"
    assert ray.get("x", list(FPM.ZEROS[M])) == list(FPM.ZEROS[M]), "ray.json made with another centre (--x)"
    nd = UM.SysUM(M).nd
    ctx.prec = VI.PREC
    H = [LC.p_acb(p) for p in ray["starts"][str(j)]]
    lo = lohner_from_box(M, H)
    leg = UM.UArc(str(rw["rho"]))
    out = {"M": M, "j": j, "h": h, "rho": rw["rho"], "r": rw["r"], "n": rw["n"], "discs": [], "mutate": mutate}
    hstep, steps = 1e-3, 0
    for k, tk in enumerate(rw["t"]):
        t0 = time.time()
        if k > 0:
            ns, hstep = LC.run_to(lo, leg, rw["t"][k - 1], tk, hstep)
            steps += ns
        X = lo.box()
        ctx.prec = DISC_PREC
        c = leg.xi(acb(arb(tk)))
        res = LDM.certify(M, c, X[:nd], X[nd], str(rw["r"]), rw["n"], mutate=mutate)
        ctx.prec = VI.PREC
        d = {"k": k, "t": tk, "ok": res["ok"], "Lam": float(res["Lam"].upper()), "F0": float(res["F0"].upper()),
             "box_rad": maxrad(X, nd), "sup": max(float(m.upper()) for m in res["Msup"]) if "Msup" in res else None,
             "low_ok": res.get("low_ok"), "steps_to": steps, "sec": time.time() - t0}
        out["discs"].append(d)
        print(f"M = {M} row {j} (|u| = {rw['rho']}, r = {rw['r']}, n = {rw['n']}) disc {k}/{rw['K']}: ok {d['ok']}, "
              f"Lam {d['Lam']:.3f}, box radius {d['box_rad']:.1e}, sup|y - y0| "
              f"{d['sup'] if d['sup'] is None else round(d['sup'], 3)}, {steps} Lohner steps [{time.time() - t00:.0f} s]"
              + (f"; defect coefficients below order n contain 0: {d['low_ok']}" if mutate else ""), flush=True)
    out["steps"] = steps
    out["ok"] = all(d["ok"] for d in out["discs"])
    out["sec"] = time.time() - t00
    if not mutate:
        with open(os.path.join(runs_dir(M), f"row_{j:02d}.json"), "w") as fh:
            json.dump(out, fh, indent=1)
    return out


def probe(M, rad="1e-7", only=None):
    """NOT rigorous: certify discs of the design (and alternatives) around double-precision values of y+ at the start
    ray and real-axis centres of chosen rows, with the boxes inflated by rad; for choosing disc radii and orders"""
    import lemma6b1_estimate_M as EST
    y0, w, N = EST.start(M, CONF[M]["rho0"])
    E3 = np.exp(-3j * np.pi / 4)
    sr = EST.integrate(lambda t: (t * E3, E3), y0, CONF[M]["rho0"], 20.0, M, w)
    ctx.prec = DISC_PREC
    wb = acb(arb(w.real, 1e-9), arb(w.imag, 1e-12))
    rows = design(M)
    for rw in rows:
        if only is not None and rw["j"] not in only:
            continue
        if only is None and rw["rho"] > 110 and rw["j"] not in (0,):
            continue
        rho = rw["rho"]
        sa = EST.integrate(lambda t, r=rho: (r * np.exp(1j * t), 1j * r * np.exp(1j * t)), EST.val(sr, rho),
                           -3 * np.pi / 4, 0.0, M, w)
        for tag, t in (("ray", -3 * np.pi / 4), ("axis", 0.0)):
            yv = EST.val(sa, t)
            Y0 = [acb(arb(v.real, float(rad)), arb(v.imag, float(rad))) for v in yv]
            c = acb(rho * np.cos(t), rho * np.sin(t))
            for (r, n) in sorted({(rw["r"], rw["n"]), (rw["r"], rw["n"] + 10), (max(rw["r"] - 2, 4), rw["n"])}):
                t0 = time.time()
                res = LDM.certify(M, c, Y0, wb, str(r), n)
                print(f"probe M = {M} row {rw['j']} |u| = {rho} ({tag}): r = {r}, n = {n}: ok {res['ok']}, Lam "
                      f"{float(res['Lam'].upper()):.3f}" + (f", sup {max(float(m.upper()) for m in res['Msup']):.3g}"
                                                             if 'Msup' in res else f" ({res.get('why')})")
                      + f"  [{time.time() - t0:.0f} s]", flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    cmd = args[0] if args else "design"
    M = int(opts.get("M", "3"))
    h = opts.get("h", "1e-6")
    global TAG
    TAG = opts.get("tag", "")
    if "x" in opts:
        # the box centre (decimal strings, M + 1 of them), e.g. the point certificate's p^; it replaces the stored
        # hier.py zero everywhere downstream (start box, w and A balls) and is recorded in ray.json
        xs = opts["x"].split(",")
        assert len(xs) == M + 1
        FPM.ZEROS[M] = xs
    if cmd == "design":
        tot = 0.0
        for rw in design(M):
            L = 3 * math.pi / 4 * rw["rho"]
            tot += L
            print(f"row {rw['j']:2d}: |u| = {rw['rho']:8.3f} (|xi| = {rw['rho'] ** -0.5:.4f}), tiles {rw['a']:.2f} < |u| < "
                  f"{rw['b']:.2f}, disc radius {rw['r']}, n = {rw['n']}, {rw['K'] + 1} discs, tile half-angle "
                  f"{rw['alpha']:.4f}, arc length {L:.0f}")
        print(f"M = {M}: total arc length {tot:.0f}; discs {sum(rw['K'] + 1 for rw in design(M))}")
    elif cmd == "probe":
        probe(M, opts.get("rad", "1e-7"), [int(v) for v in opts["rows"].split(",")] if "rows" in opts else None)
    elif cmd == "ray":
        ray_run(M, h)
    elif cmd == "row":
        row_run(M, int(opts["j"]), h)
    elif cmd == "rows":
        only = [int(v) for v in opts["only"].split(",")] if "only" in opts else None
        for rw in design(M):
            if rw["j"] < int(opts.get("from", "0")) or (only is not None and rw["j"] not in only):
                continue
            if os.path.exists(os.path.join(runs_dir(M), f"row_{rw['j']:02d}.json")):
                continue
            row_run(M, rw["j"], h)
    elif cmd == "check":
        drop = tuple(int(v) for v in opts["drop"].split(",")) if "drop" in opts else None
        with patched_check(M):
            res = LC.check(h, drop)
        if "json" in opts:
            with open(opts["json"], "w") as fh:
                json.dump(res, fh, indent=1)
        sys.exit(0 if res["ok"] else 3)
    elif cmd == "mutate":
        j = int(opts.get("j", str(len(design(M)) - 1)))
        res = row_run(M, j, h, mutate="sign")
        nok = sum(d["ok"] for d in res["discs"])
        print(f"MUTATION sign (V_u = +V/2 in the defect), M = {M}: {nok} of {len(res['discs'])} discs of row {j} "
              f"certified (must be 0)")
        sys.exit(0 if nok == 0 else 3)


if __name__ == "__main__":
    main()
