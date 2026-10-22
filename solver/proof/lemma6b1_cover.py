#!/usr/bin/env python3
"""Lemma 6b_1 (proof/A_certify.md section 12): y+ is analytic on the annular sector
    A1 = { 25 <= |u| <= 260, -3 pi/4 <= arg u <= 0 }   (u = 1/xi^2;  0.0620 <= |xi| <= 0.2, 0 <= arg xi <= 3 pi/8)
for every x in X = x^ + [-h, h]^2 (h = 1e-9), by a covering with validated Taylor discs. Rigorous (ball arithmetic).

The method (section 12.2). Rows j = 0, 1, ... of polar tiles T_jk = { a_j < |u| < b_j, th-_jk < arg u < th+_jk }:
  * the values: one QR-Lohner run (validated_int.Lohner on lemma6_usys.SysU, a single trajectory set over X) along the
    start ray arg u = -3 pi/4 from |u| = rho_0 (start box from the sector start-value lemma) inward, stopping at the row
    radii rho_j, and one run along each arc |u| = rho_j from the ray to the real axis, stopping at the tile centres
    c_jk = rho_j e^{i phi_jk};
  * the 2D part: at every c_jk, lemma6b1_disc.certify proves that every solution through the Lohner box at c_jk extends
    analytically to the disc |u - c_jk| < r_j, which contains the closed tile;
  * the gluing (`check`): tiles are added row by row from the outside, each row from the ray to the real axis. Each new
    tile meets the union of G0 = { |u| > 257, -3pi/4 - 0.4 < arg u < 0 } (where lemma 6b_0 and the sector lemma give y+)
    and the earlier tiles in a connected set (checked as a union of polar rectangles linked by overlaps), and at one
    point q of it the new disc function equals the old one (q on the arc between consecutive centres, or on the ray
    between consecutive rows, inside both discs). By the identity theorem the disc functions glue to one analytic
    function on G0 and the tiles, equal to y+ on G0; the tiles cover A1.

    python3 lemma6b1_cover.py design                      print the rows and tiles (no computation)
    python3 lemma6b1_cover.py ray [--h=1e-9]              the start box and the Lohner run along the start ray
                                                          -> lemma6b1_runs/ray.json
    python3 lemma6b1_cover.py row --j=J [--h=1e-9]        arc j: Lohner run and the disc certificates
                                                          -> lemma6b1_runs/row_JJ.json
    python3 lemma6b1_cover.py rows [--from=J]             every row not yet saved, one after the other
    python3 lemma6b1_cover.py check [--json=out.json]     the gluing and covering checks over the saved runs
    python3 lemma6b1_cover.py mutate=sign                 mutation test: every disc of row 0 with the wrong sign of
                                                          the fast term in the defect; the certificates must fail
    python3 lemma6b1_cover.py control                     negative control: discs on the start ray at |u| = 13 and 17
                                                          with radius 6, reaching the singularity of y+ (review R6);
                                                          must fail
"""
import json
import math
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
import lemma6_sweep as SW  # noqa: E402
import lemma6b1_disc as LD  # noqa: E402
from flint import arb, acb, acb_mat, ctx  # noqa: E402

AH, WH = "2.5421480475", "0.7769940136"
ORDER = 12                          # Lohner Taylor order (as lemma6_sweep)
RUNS = os.path.join(HERE, "lemma6b1_runs")
RHO0 = 262                          # outermost row radius (> 257, inside G0 on the start ray)
U_IN, U_OUT = 25, 260               # A1 in u
G0_RAD = "257"                      # G0 = { |u| > 257, -3pi/4 - 0.4 < arg u < 0 }  (|xi| < 0.06238)
G0_ANG = "0.4"
DISC_PREC = 128
GRID = 2.0 ** -30                   # Lohner step grid in t


def disc_params(rho):
    """(disc radius r, Taylor order n) by row radius"""
    if rho >= 100:
        return 24, 70
    if rho >= 50:
        return 16, 60
    return 10, 40


def design():
    """rows: rho, r, n, radial interval (a, b), K + 1 centres t_k = k/K on the 2^-30 grid, tile half-angle alpha.
    Floats are used only to choose the numbers; every condition is re-checked in arb by `check`."""
    rows = []
    rho = float(RHO0)
    pi34 = 3 * math.pi / 4
    while True:
        r, n = disc_params(rho)
        dl = 0.6 * r
        # tile corner distance at the outer radius: dl^2 + 2 (rho + dl) rho (1 - cos alpha) <= (0.92 r)^2
        ca = 1 - ((0.92 * r) ** 2 - dl ** 2) / (2 * rho * (rho + dl))
        amax = math.acos(ca)
        K = math.ceil(pi34 / (1.6 * amax))
        dphi = pi34 / K
        alpha = 0.6 * dphi
        tks = [round(k / K / GRID) * GRID for k in range(K + 1)]
        rows.append(dict(j=len(rows), rho=rho, r=r, n=n, a=rho - dl, b=rho + dl, K=K, alpha=alpha, t=tks))
        if rho - dl < U_IN - 0.5:
            break
        # next row: its outer edge overlaps this row's inner edge by 15% of the smaller half-height
        rn = rho - dl
        for _ in range(50):              # fixed point for the next row's half-height
            rr, _n = disc_params(rn)
            dn = 0.6 * rr
            rn_new = (rho - dl) + 0.15 * min(dl, dn) - dn
            if abs(rn_new - rn) < 1e-9:
                break
            rn = rn_new
        rho = round(rn, 3)
    return rows


def phi_of(t):
    """arg u at UArc parameter t (arb): -(3pi/4)(1 - t)"""
    return -(3 * arb.pi() / 4) * (1 - arb(t))


def s_acb(z):
    return [z.real.str(600, radius=True, more=True), z.imag.str(600, radius=True, more=True)]


def p_acb(p):
    return acb(arb(p[0]), arb(p[1]))


def params(h):
    ctx.prec = 320
    Ahat, wfhat = arb(AH), arb(WH)
    A, wf = Ahat + arb(0, arb(h)), wfhat + arb(0, arb(h))
    return A, wf, VI.W0 * wf


def lohner_from_box(H):
    """Lohner set c + I q containing the box H (9 components: 7 states, w, A)"""
    s = U.SysU()
    n = s.n
    c = [acb(z.mid()) for z in H]
    q = [H[i] - c[i] for i in range(n)]
    Q = acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    return VI.Lohner(s, c, Q, q, ORDER)


def run_to(lo, leg, ta, tb, h):
    """integrate t: ta -> tb (both on the 2^-30 grid), steps multiples of 2^-30 (as validated_int.Lohner.run).
    Returns (steps, next h)."""
    t, steps = ta, 0
    while t < tb:
        hs = min(max(round(h / GRID), 1) * GRID, tb - t)
        if not lo.step(leg, t, hs):
            h = hs / 2
            if h < 1e-9:
                raise RuntimeError(f"step size underflow at t = {t}")
            continue
        t = t + hs
        steps += 1
        h = min(hs * 1.25, 1.0)
    return steps, h


def maxrad(X):
    return max(VI.rad_of(z) for z in X[:7])


def ray_run(h="1e-9"):
    t00 = time.time()
    rows = design()
    A, wf, w = params(h)
    old = L6S.sector_old(h, verbose=False)
    if not old["ok"]:
        raise RuntimeError("sector start-value lemma failed")
    E3 = (acb(0, -1) * U.PHI_U).exp()
    ctx.prec = 320
    H = SW.start_box(old, w, A, acb(arb(rows[0]["rho"])) * E3, arb(0))
    out = {"h": h, "rho": [rw["rho"] for rw in rows], "starts": {"0": [s_acb(z) for z in H[:9]]},
           "start_rad": maxrad(H), "legs": []}
    print(f"start box at |u| = {rows[0]['rho']} on the start ray (sector lemma over X, contraction "
          f"{float(old['lip'].upper()):.4f}): radius {maxrad(H):.2e}", flush=True)
    ctx.prec = VI.PREC
    lo = lohner_from_box(H)
    for j in range(1, len(rows)):
        t0 = time.time()
        leg = U.URay(str(rows[j - 1]["rho"]), str(rows[j]["rho"]))
        n = lo.run(leg, 1e-2)
        X = lo.box()
        out["starts"][str(j)] = [s_acb(z) for z in X[:9]]
        out["legs"].append({"to": rows[j]["rho"], "steps": n, "sec": time.time() - t0, "rad": maxrad(X)})
        print(f"ray |u| {rows[j - 1]['rho']} -> {rows[j]['rho']}: {n} steps, {time.time() - t0:.0f} s, state radius "
              f"{maxrad(X):.2e}", flush=True)
    out["sec"] = time.time() - t00
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, "ray.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def row_run(j, h="1e-9", mutate=None):
    t00 = time.time()
    rows = design()
    rw = rows[j]
    with open(os.path.join(RUNS, "ray.json")) as fh:
        ray = json.load(fh)
    assert ray["h"] == h and ray["rho"] == [x["rho"] for x in rows], "ray.json made with another design or box"
    ctx.prec = VI.PREC
    H = [p_acb(p) for p in ray["starts"][str(j)]]
    lo = lohner_from_box(H)
    leg = U.UArc(str(rw["rho"]))
    out = {"j": j, "h": h, "rho": rw["rho"], "r": rw["r"], "n": rw["n"], "discs": [], "mutate": mutate}
    hstep, steps = 1e-3, 0
    for k, tk in enumerate(rw["t"]):
        t0 = time.time()
        if k > 0:
            ns, hstep = run_to(lo, leg, rw["t"][k - 1], tk, hstep)
            steps += ns
        X = lo.box()
        ctx.prec = DISC_PREC
        c = leg.xi(acb(arb(tk)))
        res = LD.certify(c, X[:7], X[7], str(rw["r"]), rw["n"], mutate=mutate)
        ctx.prec = VI.PREC
        d = {"k": k, "t": tk, "ok": res["ok"], "Lam": float(res["Lam"].upper()), "F0": float(res["F0"].upper()),
             "box_rad": maxrad(X), "sup": max(float(m.upper()) for m in res["Msup"]) if "Msup" in res else None,
             "low_ok": res.get("low_ok"), "steps_to": steps, "sec": time.time() - t0}
        out["discs"].append(d)
        print(f"row {j} (|u| = {rw['rho']}, r = {rw['r']}) disc {k}/{rw['K']}: ok {d['ok']}, Lam {d['Lam']:.3f}, "
              f"box radius {d['box_rad']:.1e}, sup|y - y0| {d['sup'] if d['sup'] is None else round(d['sup'], 3)}, "
              f"{steps} Lohner steps [{time.time() - t00:.0f} s]", flush=True)
        if mutate and not res["ok"]:
            continue
    out["steps"] = steps
    out["ok"] = all(d["ok"] for d in out["discs"])
    out["sec"] = time.time() - t00
    if not mutate:
        with open(os.path.join(RUNS, f"row_{j:02d}.json"), "w") as fh:
            json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    cmd = args[0] if args else "design"
    h = opts.get("h", "1e-9")
    if cmd == "design":
        tot_len = 0.0
        for rw in design():
            L = 3 * math.pi / 4 * rw["rho"]
            tot_len += L
            print(f"row {rw['j']:2d}: |u| = {rw['rho']:8.3f} (|xi| = {rw['rho'] ** -0.5:.4f}), tiles {rw['a']:.2f} < |u| < "
                  f"{rw['b']:.2f}, disc radius {rw['r']}, n = {rw['n']}, {rw['K'] + 1} discs, tile half-angle "
                  f"{rw['alpha']:.4f}, arc length {L:.0f}")
        print(f"total arc length {tot_len:.0f}; discs {sum(rw['K'] + 1 for rw in design())}")
    elif cmd == "ray":
        ray_run(h)
    elif cmd == "row":
        row_run(int(opts["j"]), h)
    elif cmd == "rows":
        for rw in design():
            if rw["j"] < int(opts.get("from", "0")):
                continue
            if os.path.exists(os.path.join(RUNS, f"row_{rw['j']:02d}.json")):
                continue
            row_run(rw["j"], h)
