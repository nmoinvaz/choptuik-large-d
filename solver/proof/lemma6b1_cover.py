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
    python3 lemma6b1_cover.py --mutate=sign [--j=11]      mutation test: every disc of row j (default the last) with
                                                          the wrong sign of the fast term in the defect; no
                                                          certificate may pass
    python3 lemma6b1_cover.py control                     negative control: discs at 14 e^{-3 i pi/4} of radius 2 ... 8;
                                                          y+ has a branch point at distance 5.12 (review R6), so the
                                                          radii above 5.12 must fail
    python3 lemma6b1_cover.py check --drop=j,k            mutation test of the gluing check: tile (j, k) left out;
                                                          the check must fail
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
              f"{steps} Lohner steps [{time.time() - t00:.0f} s]"
              + (f"; defect coefficients below order n contain 0: {d['low_ok']}" if mutate else ""), flush=True)
    out["steps"] = steps
    out["ok"] = all(d["ok"] for d in out["discs"])
    out["sec"] = time.time() - t00
    if not mutate:
        with open(os.path.join(RUNS, f"row_{j:02d}.json"), "w") as fh:
            json.dump(out, fh, indent=1)
    return out


def _bmax(x, y):
    """a ball containing max(x*, y*) for the exact reals x* in x, y* in y"""
    return x if x > y else (y if y > x else x.union(y))


def _bmin(x, y):
    return x if x < y else (y if y < x else x.union(y))


class PRect:
    """open polar rectangle { a < |u| < b, t1 < arg u < t2 }: each bound is a ball containing the exact bound (b = None
    for infinity); angles in (-pi, pi), so no wrap-around"""

    def __init__(self, a, b, t1, t2, name=""):
        self.a, self.b, self.t1, self.t2, self.name = a, b, t1, t2, name

    def meet(self, o):
        """the intersection, with bounds that are again balls containing the exact bounds"""
        b = self.b if o.b is None else (o.b if self.b is None else _bmin(self.b, o.b))
        return PRect(_bmax(self.a, o.a), b, _bmax(self.t1, o.t1), _bmin(self.t2, o.t2))

    def nonempty(self):
        """proven nonempty (True), proven empty (False) or undecided (None)"""
        rb = True if self.b is None else (self.b - self.a)
        dt = self.t2 - self.t1
        if (rb is not True and rb <= 0) or dt <= 0:
            return False
        if (rb is True or rb > 0) and dt > 0:
            return True
        return None

    def contains(self, rho, th):
        return bool(rho > self.a) and (self.b is None or bool(rho < self.b)) and bool(th > self.t1) and bool(th < self.t2)


def load_runs(h):
    rows = design()
    with open(os.path.join(RUNS, "ray.json")) as fh:
        ray = json.load(fh)
    runs = []
    for rw in rows:
        f = os.path.join(RUNS, f"row_{rw['j']:02d}.json")
        runs.append(json.load(open(f)) if os.path.exists(f) else None)
    return rows, ray, runs


def check(h="1e-9", drop=None, verbose=True):
    """the gluing and covering conditions of section 12.2, rigorous in arb, over the saved runs.
    drop=(j, k): mutation test, the disc (j, k) is treated as missing (its tile is not added); the check must fail."""
    ctx.prec = 256
    rows, ray, runs = load_runs(h)
    fails = []

    def need(cond, msg):
        if not cond:
            fails.append(msg)
        return cond
    pi = arb.pi()
    P34 = 3 * pi / 4
    need(ray["h"] == h and ray["rho"] == [rw["rho"] for rw in rows], "ray.json: other design or box")
    need(len(ray["legs"]) == len(rows) - 1 and all(str(j) in ray["starts"] for j in range(len(rows))),
         "ray run incomplete")
    stats = []
    for rw, run in zip(rows, runs):
        j = rw["j"]
        if not need(run is not None, f"row {j}: no run"):
            continue
        need(run["h"] == h and run["rho"] == rw["rho"] and run["r"] == rw["r"] and run["n"] == rw["n"],
             f"row {j}: other design")
        need([d["t"] for d in run["discs"]] == rw["t"], f"row {j}: disc centres differ from the design")
        for d in run["discs"]:
            need(d["ok"], f"disc ({j}, {d['k']}) not certified")
        stats.append(dict(j=j, rho=rw["rho"], r=rw["r"], n=rw["n"], discs=len(run["discs"]), steps=run["steps"],
                          sec=run["sec"], Lam=max(d["Lam"] for d in run["discs"]),
                          box=max(d["box_rad"] for d in run["discs"]), sup=max(d["sup"] for d in run["discs"])))
    # exact numbers of the construction
    R = [arb(str(rw["rho"])) for rw in rows]                   # row radii (the decimals the runs used)
    Ar = [arb(rw["a"]) for rw in rows]                          # tile radial bounds (binary floats, exact)
    Br = [arb(rw["b"]) for rw in rows]
    Al = [arb(rw["alpha"]) for rw in rows]
    Rd = [arb(rw["r"]) for rw in rows]
    Phi = [[phi_of(t) for t in rw["t"]] for rw in rows]
    for j, rw in enumerate(rows):
        need(rw["t"][0] == 0.0 and rw["t"][-1] == 1.0, f"row {j}: first centre not on the ray or last not on the axis")
        need(bool(Ar[j] < R[j]) and bool(R[j] < Br[j]), f"row {j}: centre radius outside the tile band")
        for k, ph in enumerate(Phi[j]):
            # closed tile inside the open disc: corners at the radial ends, |arg u - phi| <= alpha (C2)
            for rr in (Ar[j], Br[j]):
                need(bool(rr * rr + R[j] * R[j] - 2 * rr * R[j] * Al[j].cos() < Rd[j] * Rd[j]),
                     f"tile ({j}, {k}) not inside its disc")
            if k:
                # the two halves of the arc between consecutive centres, each in its own disc: every point of the arc
                # from a centre to the midpoint q is within the chord 2 rho sin(dphi/4) of that centre (C3)
                dph = ph - Phi[j][k - 1]
                need(bool(dph > 0) and bool(2 * R[j] * (dph / 4).sin() < Rd[j]), f"arc halves ({j}, {k}) not in the discs")
                need(bool(dph / 2 < Al[j]) and bool(Al[j] < dph), f"tiles ({j}, {k - 1}), ({j}, {k}): overlap rule")
        need(bool(Al[j] > 0) and bool(Al[j] < 1), f"row {j}: tile angle")
        if j:
            rq = (Ar[j - 1] + Br[j]) / 2
            need(bool(Ar[j - 1] < Br[j]), f"rows {j - 1}, {j} do not overlap")
            need(bool(R[j] < rq) and bool(rq < R[j - 1]), f"ray point q between rows {j - 1}, {j}: not between the centres")
            need(bool(R[j - 1] - rq < Rd[j - 1]) and bool(rq - R[j] < Rd[j]), f"ray segment between rows {j - 1}, {j}")
            need(bool(Ar[j] < rq) and bool(rq < Br[j]) and bool(Ar[j - 1] < rq) and bool(rq < Br[j - 1]),
                 f"ray point q between rows {j - 1}, {j}: not in both tile bands")
    # covering of A1 (C8)
    need(bool(Br[0] > U_OUT) and bool(Ar[-1] < U_IN), "rows do not cover 25 <= |u| <= 260")
    need(bool(R[0] > arb(G0_RAD)), "s_0 not in G0")
    # connectivity of each new tile's meeting set with the earlier union, and the agreement point in it (C7)
    G0 = PRect(arb(G0_RAD), None, -P34 - arb(G0_ANG), arb(0), "G0")
    done = [G0]
    nedges = npieces = 0
    for j, rw in enumerate(rows):
        for k, ph in enumerate(Phi[j]):
            if drop is not None and (j, k) == tuple(drop):
                continue
            T = PRect(Ar[j], Br[j], ph - Al[j], ph + Al[j], f"T{j},{k}")
            pieces = []
            for X in done:
                Pm = T.meet(X)
                ne = Pm.nonempty()
                if ne is False:
                    continue
                pieces.append((X.name, Pm, ne))
            npieces += len(pieces)
            # agreement point: on the arc between the centres (k > 0), on the ray between the rows (k = 0, j > 0),
            # s_0 itself (tile (0, 0), against G0)
            if k:
                qname, qr, qt = f"T{j},{k - 1}", R[j], (Phi[j][k - 1] + ph) / 2
            elif j:
                qname, qr, qt = f"T{j - 1},0", (Ar[j - 1] + Br[j]) / 2, -P34
            else:
                qname, qr, qt = "G0", R[0], -P34
            qp = [Pm for (nm, Pm, ne) in pieces if nm == qname]
            need(len(qp) == 1 and qp[0].contains(qr, qt), f"tile ({j}, {k}): agreement point not in the meeting set")
            # connected: every piece linked to the agreement piece by proven-nonempty overlaps
            if not need(all(ne is True for (_, _, ne) in pieces), f"tile ({j}, {k}): undecided piece"):
                continue
            idx = {nm: i for i, (nm, _, _) in enumerate(pieces)}
            seen, stack = set(), [idx.get(qname, 0)]
            while stack:
                i = stack.pop()
                if i in seen:
                    continue
                seen.add(i)
                for l in range(len(pieces)):
                    if l not in seen and pieces[i][1].meet(pieces[l][1]).nonempty() is True:
                        stack.append(l)
                        nedges += 1
            need(len(seen) == len(pieces), f"tile ({j}, {k}): meeting set not connected")
            done.append(T)
    # with a dropped tile, also check that A1 is still covered (it is not: the covering argument needs every tile)
    if drop is not None:
        jj, kk = drop
        ph = Phi[jj][kk]
        # a point of A1 in tile (jj, kk) only: its centre direction at the middle of its band
        pt_r, pt_t = R[jj], ph
        cov = any(X.contains(pt_r, pt_t) for X in done)
        need(cov, f"point (|u| = {rows[jj]['rho']}, arg = phi_{jj},{kk}) not covered without tile ({jj}, {kk})")
    ok = not fails
    out = dict(ok=ok, fails=fails[:200], nfails=len(fails), rows=stats, tiles=len(done) - 1, ray=ray["legs"],
               ray_sec=ray["sec"], start_rad=ray["start_rad"], drop=drop, pieces=npieces, links=nedges)
    if verbose:
        for s in stats:
            print(f"row {s['j']:2d} |u| = {s['rho']:7.2f}: {s['discs']:2d} discs (r = {s['r']}, n = {s['n']}), "
                  f"{s['steps']:4d} Lohner steps, max Lam {s['Lam']:.3f}, max box radius {s['box']:.1e}, "
                  f"max sup|y - y(c)| {s['sup']:.3g}, {s['sec']:.0f} s")
        print(f"ray: {sum(l['steps'] for l in ray['legs'])} steps, {ray['sec']:.0f} s; tiles glued: {len(done) - 1} "
              f"(meeting sets: {npieces} polar rectangles, {nedges} links)")
        print(f"LEMMA 6b_1 CHECK: {'PASS' if ok else 'FAIL'}" + ("" if ok else f" ({len(fails)} failures: {fails[:5]})"))
    return out


def control(h="1e-9"):
    """negative control: the Lohner run continued along the start ray from the last row to |u| = 14, and discs at
    14 e^{-3 i pi/4} of growing radius. y+ has a branch point near u* = 9.675 e^{-2.120 i} (xi* = 0.3215 e^{1.060 i};
    review R6; located in double precision, so not rigorous), at distance 5.12 from that centre, so no certificate may
    pass once the radius exceeds 5.12."""
    rows = design()
    with open(os.path.join(RUNS, "ray.json")) as fh:
        ray = json.load(fh)
    ctx.prec = VI.PREC
    H = [p_acb(p) for p in ray["starts"][str(len(rows) - 1)]]
    lo = lohner_from_box(H)
    n = lo.run(U.URay(str(rows[-1]["rho"]), "14"), 1e-2)
    X = lo.box()
    print(f"ray continued to |u| = 14: {n} steps, state radius {maxrad(X):.1e}")
    leg = U.URay("14", "13")
    out = []
    for r, nn in (("2", 40), ("3", 40), ("4", 50), ("4.5", 60), ("5.5", 60), ("6", 60), ("8", 60)):
        ctx.prec = DISC_PREC
        res = LD.certify(leg.xi(acb(0)), X[:7], X[7], str(r), nn)
        ctx.prec = VI.PREC
        out.append(dict(r=r, n=nn, ok=res["ok"], Lam=float(res["Lam"].upper()), why=res.get("why"),
                        sup=max(float(m.upper()) for m in res["Msup"]) if "Msup" in res else None))
        print(f"control disc at 14 e^(-3 i pi/4), radius {r}, n = {nn}: certified {res['ok']}, Lam "
              f"{float(res['Lam'].upper()):.3f}" + (f", sup|y - y(c)| {out[-1]['sup']:.3g}" if out[-1]["sup"] else
                                                    f" ({res.get('why')})"), flush=True)
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    cmd = args[0] if args else "design"
    if opts.get("mutate") == "sign" or cmd == "mutate=sign":
        cmd = "mutate"
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
    elif cmd == "check":
        drop = tuple(int(x) for x in opts["drop"].split(",")) if "drop" in opts else None
        res = check(h, drop)
        if "json" in opts:
            with open(opts["json"], "w") as fh:
                json.dump(res, fh, indent=1)
        sys.exit(0 if res["ok"] else 3)
    elif cmd == "mutate":
        j = int(opts.get("j", str(len(design()) - 1)))
        res = row_run(j, h, mutate="sign")
        nok = sum(d["ok"] for d in res["discs"])
        print(f"MUTATION sign (the fast term V_u = +V/2 in the defect): {nok} of {len(res['discs'])} discs of row {j} "
              f"certified (must be 0)")
        sys.exit(0 if nok == 0 else 3)
    elif cmd == "control":
        res = control(h)
        if "json" in opts:
            with open(opts["json"], "w") as fh:
                json.dump(res, fh, indent=1)
