"""The cost growth of the RT continuation (issue #145) from rt_run.py --solver mp records: per D and grid the Newton
steps, GMRES products, seconds and Delta; the grid differences Delta_N - Delta_N'; least-squares fits of the products per
Newton step against D, of the grid difference against D, and of the seconds per product and per preconditioner
application against N; and from them a projection of N, L, the products and the seconds to larger D. The projection is
an extrapolation of fits over the measured range, not a measurement.

    python3 solver/rt/rt_cost_fit.py A.jsonl [B.jsonl ...] [--old I105.jsonl,...] [--alt C.jsonl,...]
        [--project 60,100,215] [--table]
    python3 solver/rt/rt_cost_fit.py --planted [--mutate=slope]

Records are grouped by grid; for each grid and D the last converged record is used. --old adds records (#105's
continuations) to the products and grid-difference fits only, since they lack the per-step times. --alt names records of
second roots of the same grids from other starts (rt_run.py --start from another grid's state); their Delta against the
main records' gives the root spread per grid and D. Each fit is quoted with
its spread over three windows of D (all, first half, second half): a spread over fit choices, not an error bar.

Models (stated, then fitted):
- products per Newton step k(D) = a + b D, from the first Newton step of each D (fresh preconditioner) and from the mean.
- log10 |Delta_N - Delta_{N+20}| = c_N + d_N D for each pair of grids (model exp), or delta itself linear in D (model
  lin); r(D) = log10 delta_N - log10 delta_{N+20}, the decades gained per 20 Chebyshev points, from the two largest
  pairs under the same model. N(D) is the smallest N of the ladder whose
  delta_N(D) < --tol (1e-12), with delta_N for N above the largest pair continued at the rate r(D).
- seconds per product p(N) and per preconditioner application q(N) as powers of N + 1 through the measured grids; the
  rest of a Newton step (orthogonalization, line search) as e k^2 L.
- seconds per D step = Newton steps x (k (p + q) + e k^2 L) + 2 (f64 Jacobian + LU) (one rebuild), with the median
  Newton steps of the continuation steps, both Jacobian and LU scaled as L^3 (an upper bound for the Jacobian), and
  the memory of the f64 Jacobian 8 L^2 bytes.

--planted runs the same fits on synthetic records with known a, b, c, d and exponents, plus noise, and exits 1 unless
they come back within the stated tolerances, the projected N(100) and N(215) among them; --mutate=slope (the products'
slope planted with the wrong sign) and --mutate=rate (delta_121's slope planted 0.02 steeper than the check expects)
must then exit 1.
"""
import sys
import json
import math
import argparse
import numpy as np
import flint

flint.ctx.prec = 192
arb = flint.arb


def L_of(Nt, N):
    return Nt * (N + 1) // 2 + 3 * Nt * (N + 1) + 1


def load(paths):
    """{(Nt, N): {D: record}}, the last converged mp record of each grid and D. A restart's re-solve of its start state
    (label start) does not replace the step that made that state."""
    out = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("solver") == "mp" and r.get("ok") and not r.get("mutate"):
                    g = out.setdefault(tuple(r["grid"]), {})
                    D = round(r["D"], 9)
                    if r.get("label") == "start" and D in g and g[D].get("label") != "start":
                        continue
                    g[D] = r
    return out


def lsq(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.c_[np.ones_like(x), x]
    return np.linalg.lstsq(A, y, rcond=None)[0]           # (intercept, slope)


def windows(x):
    x = np.asarray(x, float)
    mid = 0.5 * (x.min() + x.max())
    return {"all": np.ones(len(x), bool), "first half": x <= mid, "second half": x >= mid}


def fit_spread(x, y):
    """{window: (intercept, slope)} for windows with at least 3 points."""
    out = {}
    for name, m in windows(x).items():
        if m.sum() >= 3:
            out[name] = lsq(np.asarray(x)[m], np.asarray(y)[m])
    return out


def ddiff(ra, rb):
    return float((arb(ra["Delta_str"]) - arb(rb["Delta_str"])).mid())


def analyse(G, old, tol=1e-12, project=(), table=False, say=print, delta_from=None, alt=None):
    res = {}
    grids = sorted(G)
    # ------------------------------------------------ products per Newton step against D
    say("products per Newton step, k(D) = a + b D (spread over the windows of D):")
    res["k"] = {}
    for g in grids:
        for kind in ("first", "mean"):
            R = {**old.get(g, {}), **G[g]} if kind == "first" else G[g]
            # first: continuation steps only (a fresh preconditioner at the predictor); mean: every solve, the
            # followers' re-solves from a resampled coarser state included
            Ds = sorted(D for D, r in R.items() if r["products"] and (kind == "mean" or r.get("label") != "start"))
            if len(Ds) < 3:
                continue
            y = [R[D]["products"][0] if kind == "first" else np.mean(R[D]["products"]) for D in Ds]
            fs = fit_spread(Ds, y)
            res["k"][(g, kind)] = fs
            say(f"  {g[0]}x{g[1]} {kind:5s} D {Ds[0]:g}..{Ds[-1]:g} ({len(Ds)} D): " + "; ".join(
                f"{w}: a {c[0]:.1f}, b {c[1]:.3f}" for w, c in fs.items()))
    # ------------------------------------------------ grid differences against D
    say(f"grid differences, log10 |Delta_N - Delta_N'| = c + d D (N' the next grid):")
    res["delta"] = {}
    allG = {g: {**old.get(g, {}), **G.get(g, {})} for g in sorted(set(G) | set(old))}
    ag = sorted(allG)
    for g, h in zip(ag, ag[1:]):
        if g[0] != h[0]:
            continue
        Ds = sorted(set(allG[g]) & set(allG[h]))
        pts = [(D, abs(ddiff(allG[g][D], allG[h][D]))) for D in Ds if delta_from is None or D >= delta_from]
        pts = [(D, v) for D, v in pts if v > 0]
        if len(pts) < 3:
            continue
        x, y = [p[0] for p in pts], [math.log10(p[1]) for p in pts]
        fs = fit_spread(x, y)
        res["delta"][(g[1], h[1])] = dict(fits=fs, fits_lin=fit_spread(x, [p[1] for p in pts]), pts=pts)
        cross = {w: (math.log10(tol) - c[0]) / c[1] if c[1] > 0 else float("inf") for w, c in fs.items()}
        say(f"  N {g[1]} - {h[1]} D {x[0]:g}..{x[-1]:g} ({len(x)} D), last {pts[-1][1]:.2e}: " + "; ".join(
            f"{w}: c {c[0]:.2f}, d {c[1]:.4f} decades per unit D, crosses {tol:g} at D {cross[w]:.1f}"
            for w, c in fs.items()))
    # ------------------------------------------------ seconds per product and preconditioner against N
    say("seconds per product p and per preconditioner application q (medians over the Newton steps), the rest e k^2 L:")
    res["time"] = {}
    for g in grids:
        R = [r for r in G[g].values() if "t_prod" in r and r["products"]]
        if not R:
            continue
        p = np.median([tp / k for r in R for tp, k in zip(r["t_prod"], r["products"])])
        q = np.median([tq / k for r in R for tq, k in zip(r["t_prec"], r["products"])])
        e = np.median([(s - tp - tq) / (k * k * r["L"]) for r in R
                       for s, tp, tq, k in zip(r["step_secs"], r["t_prod"], r["t_prec"], r["products"])])
        nst = np.median([len(r["products"]) for r in R if r.get("label") == "cont"]
                        or [len(r["products"]) for r in R])
        jl = np.median([r["t_jac"] + r["t_lu"] for r in R])
        mem = max(r.get("rss_peak_gb", 0) for r in R)
        res["time"][g] = dict(p=p, q=q, e=e, newton=nst, jl=jl, L=R[0]["L"], mem=mem)
        say(f"  {g[0]}x{g[1]} L {R[0]['L']}: p {p:.3f} s, q {q:.3f} s, e {e:.2e} s, Newton steps per D step {nst:g}, "
            f"Jacobian + LU {jl:.1f} s, peak memory {mem:.2f} GB")
    tg = sorted(res["time"])
    # Newton steps per D step: the continuation's (label cont), not the followers' re-solves from resampled states
    nc = [len(r["products"]) for g in grids for r in G[g].values() if r.get("label") == "cont"]
    res["newton_cont"] = float(np.median(nc)) if nc else float("nan")
    say(f"  Newton steps per continuation step: median {res['newton_cont']:g} over {len(nc)} steps"
        + (f", range {min(nc)}-{max(nc)}" if nc else ""))
    if len(tg) >= 2:
        g0, g1 = tg[0], tg[-1]
        lr = math.log((g1[1] + 1) / (g0[1] + 1))
        res["alpha_p"] = math.log(res["time"][g1]["p"] / res["time"][g0]["p"]) / lr
        res["alpha_q"] = math.log(res["time"][g1]["q"] / res["time"][g0]["q"]) / lr
        say(f"  exponents in N + 1 from {g0[1]} and {g1[1]}: p ~ (N+1)^{res['alpha_p']:.2f}, "
            f"q ~ (N+1)^{res['alpha_q']:.2f}")
    if table:
        say("\nper D and grid (main path and check points):")
        say("| D | grid | L | Newton steps | products | s per Newton step | s for the D | Delta_RT | - hp | "
            "- next grid |")
        say("|---|---|---|---|---|---|---|---|---|---|")
        for D in sorted(set().union(*[set(G[g]) for g in grids])):
            for i, g in enumerate(grids):
                if D not in G[g]:
                    continue
                r = G[g][D]
                nxt = next((h for h in grids[i + 1:] if h[0] == g[0] and D in G[h]), None)
                dn = f"{ddiff(r, G[nxt][D]):+.2e}" if nxt else ""
                hp = f"{r['diff']:+.2e}" if r["diff"] is not None else ""
                sp = r["t_newton"] / max(1, len(r["products"]))
                say(f"| {D:g} | {g[0]}x{g[1]} | {r['L']} | {len(r['products'])} | {r['products']} | {sp:.0f} | "
                    f"{r['secs']:.0f} | {arb(r['Delta_str']).mid().str(20, radius=False)} | {hp} | {dn} |")
    if alt:
        res["spread"] = root_spread(G, alt, tol, say)
    if project:
        res["project"] = projection(res, project, tol, say)
    return res


def root_spread(G, alt, tol, say):
    """|Delta_alt - Delta_main| per grid and D, where alt holds second roots of the same grids from other starts, and
    a fit of its log10 against D through the points (an extrapolation past them)."""
    say("root spread |Delta_2 - Delta_1| of two converged roots of one grid from different starts:")
    out = {}
    for g in sorted(alt):
        if g not in G:
            continue
        pts = [(D, abs(ddiff(alt[g][D], G[g][D]))) for D in sorted(set(alt[g]) & set(G[g]))]
        out[g] = pts
        line = ", ".join(f"D {D:g}: {v:.1e}" for D, v in pts)
        fit = ""
        good = [(D, v) for D, v in pts if v > 0]
        if len(good) >= 2:
            c = lsq([q[0] for q in good], [math.log10(q[1]) for q in good])
            cross = (math.log10(tol) - c[0]) / c[1] if c[1] > 0 else float("inf")
            fit = f"; log10 fit through them: {c[1]:.2f} decades per unit D, reaches {tol:g} at D {cross:.1f}"
        say(f"  {g[0]}x{g[1]}: {line}{fit}")
    return out


def log_delta(res, pair, model, w, D):
    """log10 |Delta_N - Delta_N'| of the pair at D: model exp (log10 delta linear in D) or lin (delta linear in D)."""
    if model == "exp":
        c = res["delta"][pair]["fits"][w]
        return c[0] + c[1] * D
    c = res["delta"][pair]["fits_lin"][w]
    v = c[0] + c[1] * D
    return math.log10(v) if v > 0 else -300.0


def N_of(res, pairs, model, w, D, tol):
    """The smallest N of the ladder with delta_N(D) < tol: delta of the largest pair, continued past it at r(D)
    decades per 20 points, r from the two largest pairs (window all); None when the model gives no N."""
    top = pairs[-1]
    ld = log_delta(res, top, model, w, D)
    r = None
    if len(pairs) >= 2:
        (n0, _), (n1, _) = pairs[-2], pairs[-1]
        r = (log_delta(res, pairs[-2], model, "all", D) - log_delta(res, top, model, "all", D)) * 20 / (n1 - n0)
    N = top[0]
    while ld >= math.log10(tol) and r and r > 0 and N < 2001:
        ld -= r
        N += 20
    return N if ld < math.log10(tol) else None


def projection(res, targets, tol, say):
    """An extrapolation of the fits (see the module docstring), for both models of delta(D) and every window."""
    out = {}
    tg = sorted(res["time"])
    if not tg or not res["delta"] or "alpha_p" not in res:
        say("projection: needs two timed grids and one grid pair")
        return out
    gT = tg[-1]
    T = res["time"][gT]
    Nt = gT[0]
    pairs = sorted(res["delta"])
    fk = res["k"].get((tg[0], "mean")) or res["k"].get((tg[0], "first"))
    say(f"\nprojection (an extrapolation of the fits above, not a measurement); delta_N from the pair {pairs[-1]}, "
        f"r(D) from {pairs[-2:] if len(pairs) > 1 else 'none'}; models: exp (log10 delta linear in D), lin (delta "
        f"linear in D):")
    say("| D | delta model | window | N(D) | L | k per Newton step | s per Newton step | s per D step | "
        "hours from D 30 | f64 Jacobian GB |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for D in targets:
        for model in ("exp", "lin"):
            for w in res["delta"][pairs[-1]]["fits"]:
                N = N_of(res, pairs, model, w, D, tol)
                if N is None:
                    say(f"| {D:g} | {model} | {w} | beyond the model | | | | | | |")
                    continue
                ab = fk.get(w, fk["all"])
                hours = 0.0
                for Dk in np.arange(30.0, D + 1e-9, 1.0):  # the D steps of 1 from 30, each at that D's cost
                    Nk = N_of(res, pairs, model, w, Dk, tol) or N
                    hours += step_secs(res, T, gT, Nt, Nk, ab[0] + ab[1] * Dk) / 3600
                k = ab[0] + ab[1] * D
                s_step, s_newton, L = step_secs(res, T, gT, Nt, N, k, parts=True)
                out[(D, model, w)] = dict(N=N, L=L, k=k, s_newton=s_newton, s_step=s_step, hours=hours,
                                          mem=8 * L * L / 2 ** 30)
                say(f"| {D:g} | {model} | {w} | {N} | {L} | {k:.0f} | {s_newton:.0f} | {s_step:.0f} | {hours:.1f} | "
                    f"{8 * L * L / 2 ** 30:.1f} |")
    return out


def step_secs(res, T, gT, Nt, N, k, parts=False):
    L, L0, n0 = L_of(Nt, N), T["L"], gT[1] + 1
    s = (N + 1) / n0
    p, q = T["p"] * s ** res["alpha_p"], T["q"] * s ** res["alpha_q"]
    s_newton = k * (p + q) + T["e"] * k * k * L
    s_step = res["newton_cont"] * s_newton + 2 * T["jl"] * (L / L0) ** 3
    return (s_step, s_newton, L) if parts else s_step


def planted(mutate=None):
    """Synthetic records with known parameters through load's layout and analyse's fits."""
    rng = np.random.default_rng(145)
    a, b = 10.0, 2.0                                       # k = a + b D
    c = {81: -14.0, 101: -19.5, 121: -25.0}                # log10 delta_N at D = 0
    d = {81: 0.10, 101: 0.11, 121: 0.12}                   # decades per unit D
    p0, alpha = 0.48, 2.0                                  # seconds per product at N = 101, exponent in N + 1
    G = {}
    truth = arb("1.9")
    for N in (81, 101, 121, 141):
        G[(32, N)] = {}
        for D in np.arange(30.0, 60.1, 1.0):
            k = int(round(a + (b if mutate != "slope" else -b) * D + rng.normal(0, 1)))
            # Delta_N = truth + sum of the planted deltas of the finer grids (delta_N = Delta_N - Delta_{N+20})
            off = arb(0)
            for M in (81, 101, 121):
                if M >= N:
                    dM = d[M] + (0.02 if (mutate == "rate" and M == 121) else 0.0)
                    off += arb(10.0 ** (c[M] + dM * D) * (1 + 0.05 * rng.normal()))
            p = p0 * ((N + 1) / 102) ** alpha
            q = 0.1 * ((N + 1) / 102) ** 4
            L = L_of(32, N)
            G[(32, N)][D] = dict(solver="mp", ok=True, D=D, grid=[32, N], L=L, products=[k] * 5, label="cont",
                                 Delta_str=(truth + off).mid().str(40, radius=False), diff=None,
                                 step_secs=[k * (p + q) + 1e-8 * k * k * L] * 5, t_prod=[k * p] * 5,
                                 t_prec=[k * q] * 5, t_jac=10.0, t_lu=4.0, t_newton=5 * k * (p + q), secs=0.0,
                                 rss_peak_gb=1.0)
    res = analyse(G, {}, say=lambda s: None)
    ok = []
    ka, kb = res["k"][((32, 101), "mean")]["all"]
    ok.append(("k slope", abs(kb - b) < 0.1, kb))
    for (n0, n1), v in res["delta"].items():
        if n0 in d:
            ok.append((f"delta slope {n0}", abs(v["fits"]["all"][1] - d[n0]) < 0.01, v["fits"]["all"][1]))
    ok.append(("p exponent", abs(res["alpha_p"] - alpha) < 0.05, res["alpha_p"]))
    # the projected N(D) against the planted model's: log10 delta_121 = c + d D, then r(D) decades per 20 points
    pr = projection(res, [100.0, 215.0], 1e-12, lambda s: None)
    for D in (100.0, 215.0):
        ld, r, N = c[121] + d[121] * D, (c[101] + d[101] * D) - (c[121] + d[121] * D), 121
        while ld >= -12:
            ld, N = ld - r, N + 20
        got = pr.get((D, "exp", "all"), {}).get("N", float("nan"))   # nan: beyond the model
        ok.append((f"N({D:g}) = {N}", got == N, got))
    for name, good, val in ok:
        print(f"planted {name}: {val:.4f} {'ok' if good else 'FAIL'}")
    # the root spread: second roots on 32x101 offset by 10^(-20 + 0.8 (D - 40)), so the fit crosses 1e-12 at D = 50
    alt = {(32, 101): {}}
    for D in (40.0, 45.0, 55.0):
        r = dict(G[(32, 101)][D])
        r["Delta_str"] = (arb(r["Delta_str"]) + arb(10.0 ** (-20 + 0.8 * (D - 40)))).mid().str(40, radius=False)
        alt[(32, 101)][D] = r
    sp = root_spread(G, alt, 1e-12, lambda s: None)[(32, 101)]
    c = lsq([q[0] for q in sp], [math.log10(q[1]) for q in sp])
    cross = (-12 - c[0]) / c[1]
    ok.append(("root-spread crossing at D 50", abs(cross - 50) < 0.1, cross))
    print(f"planted {ok[-1][0]}: {cross:.4f} {'ok' if ok[-1][1] else 'FAIL'}")
    passed = all(g for _, g, _ in ok)
    print("PASS" if passed else "FAIL")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--old", default="")
    ap.add_argument("--alt", default="", help="records of second roots (other starts) of the same grids, comma separated")
    ap.add_argument("--project", default="")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--delta_from", type=float, default=None,
                    help="fit the grid differences from this D on (81 - 101 changes sign near D = 13)")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--planted", action="store_true")
    ap.add_argument("--mutate", default=None, choices=["slope", "rate"])
    a = ap.parse_args()
    if a.planted:
        sys.exit(0 if planted(a.mutate) else 1)
    G = load(a.files)
    old = load([p for p in a.old.split(",") if p])
    alt = load([p for p in a.alt.split(",") if p])
    analyse(G, old, tol=a.tol, project=[float(v) for v in a.project.split(",") if v], table=a.table, alt=alt,
            delta_from=a.delta_from)


if __name__ == "__main__":
    main()
