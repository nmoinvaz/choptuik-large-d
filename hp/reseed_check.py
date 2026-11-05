#!/usr/bin/env python3
"""The basin and jump tests of the re-seed (notes/RESEED.md 3.3, 3.4).

    python3 reseed_check.py basin STATE.npz OUT.jsonl --family A|B|P --arm plain|reseed|elim [--seeds 1,2,3]
        [--pred_base BASE.npz --pred_h H] (family P)
        [--deltas 1e-12,1e-11,...,1e-2] [--threads 2] [--mutate none|skip] [--qd_it 12]
    python3 reseed_check.py summary OUT.jsonl ...
    python3 reseed_check.py rungs RAWDIR BASE.npz OUT.jsonl [--h 5]   (the cached jump series of BASE; amendment 9)
    python3 reseed_check.py floorflip RAWDIR BASE.npz OUT.jsonl [--h 0] [--trials 20]   (amendment 9)
    python3 reseed_check.py firststep STATE.npz OUT.jsonl [--deltas 1e-12,1e-14] [--groups centre,nodes,data,Delta,all]
    python3 reseed_check.py jump RAWDIR STATE.npz OUT.jsonl [--hs 2,5,10] [--arms reseed,elim,plain] [--threads 2]

basin: the truth (a converged chain state) perturbed by delta along a random direction per seed (family A: the centre
data and every node state, each entry by delta r S, r uniform in +-[0.5, 1], S the max |z| of its field block; family
B: also the horizon data, and Delta by the factor 1 + delta r; family P: along the jump test's prediction error, the
cached series of --pred_base summed at --pred_h minus the truth, max norm 1, one direction), then the chain's corrector (hpkernel.driver.newton_qd on
pinning.PinnedQ, tol 1e-28, 12 steps, 8 halvings, acceptance "dz"), after the re-seed (hp/reseed.py) in the arm
reseed. Per seed delta rises until the first failure. A run converged back to the truth when newton_qd converged,
max |z - z_truth| <= 1e-20 and |Delta - Delta_truth| <= 1e-24. One record per run in OUT.jsonl.

jump: from STATE (the chain's latest), the chain's quad-double series (anm_run.py coefficients_q, N 24, the held
harmonics on the T75 trend as anm_run.py --trend T75 holds them) summed at D + h, re-seeded (arm reseed) or not
(plain), corrected at D + h; Delta against the chain's state at that D when the chain has one, else against the
formula.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import load_q, newton_qd, qd_axpy, qmax  # noqa: E402
from raw_out import require_raw  # noqa: E402
from reseed import ElimQ, Reseeder, offsets, parse, pinned  # noqa: E402

mp.mp.dps = 50


def c4(v):
    return tuple(np.ascontiguousarray(np.asarray(p, dtype=float)) for p in v)


def qdiff(a, b):
    """a - b of four-part states as f64 values (the difference's leading parts summed)"""
    d = qd_axpy(-1.0, *c4(b), *c4(a))
    return np.asarray(d[0]) + np.asarray(d[1]) + np.asarray(d[2])


def direction(truth, M, K, family, seed):
    """the perturbation direction (an f64 vector, delta = 1): entries r S per field block (the docstring)"""
    o = offsets(M, K)
    z = np.asarray(truth[0])
    rng = np.random.default_rng(seed)
    L = len(z)
    r = rng.uniform(0.5, 1.0, L) * rng.choice([-1.0, 1.0], L)
    d = np.zeros(L)
    blocks = [(0, M), (M, 2 * M)] + [(o["node"](i) + q * M, o["node"](i) + (q + 1) * M) for i in range(K) for q in range(4)]
    if family == "B":
        blocks += [(o["mh"], o["mh"] + M), (o["ah"], o["ah"] + M)]
    for a, b in blocks:
        d[a:b] = r[a:b] * np.max(np.abs(z[a:b]))
    if family == "B":
        d[-1] = r[-1] * abs(z[-1])
    return d


def basin(argv):
    path, out = argv[0], argv[1]
    opt = parse(argv, dict(family="B", arm="reseed", seeds="1,2,3", deltas=",".join(f"1e-{k}" for k in range(12, 1, -1)),
                           threads=2, mutate="none", qd_it=12, levels=5, stop="back", pred_base="", pred_h=5.0))
    require_raw(out)
    z4, meta = load_q(path)
    D = mp.mpf(meta["D_str"])
    M, K = int(meta["M"]), len(meta["nodes"])
    s, sp, sq = pinned(meta, D, z4, threads=opt["threads"])
    truth = c4(z4)
    F_truth = qmax(sq.residual_q(*truth))
    rs = Reseeder(meta, D, threads=opt["threads"], levels=opt["levels"], mutate=opt["mutate"]) if opt["arm"] in ("reseed", "elim") else None
    deltas = [float(x) for x in opt["deltas"].split(",")]
    print(f"basin {path}: D {mp.nstr(D, 12)}, truth max|F| {F_truth:.2e}; family {opt['family']}, arm {opt['arm']}, mutate "
          f"{opt['mutate']}, seeds {opt['seeds']}, deltas {deltas[0]:.0e}..{deltas[-1]:.0e}", flush=True)
    if opt["family"] == "P":
        # family P: along the series prediction's own error, the jump test's prediction (the cached quad-double series of
        # --pred_base summed at --pred_h) minus the truth, scaled to max norm 1
        from anm_run import series_at
        zb, mb = load_q(opt["pred_base"])
        cz = np.load(os.path.join(os.path.dirname(out), "..", "jump", f"series_D{float(mb['D']):.4f}_N24.npz"))
        Zh = [np.asarray(cz[f"z{k}_0"]) for k in range(25)]
        Zl = [np.asarray(cz[f"z{k}_1"]) for k in range(25)]
        ph, pl = series_at(np.asarray(zb[0]), np.asarray(zb[1]), Zh, Zl, opt["pred_h"])
        if abs(mp.mpf(mb["D_str"]) + mp.mpf(opt["pred_h"]) - D) > mp.mpf(10) ** -40:
            raise SystemExit("family P: the prediction's D is not the truth's")
        dP = qdiff((ph, pl, np.zeros_like(ph), np.zeros_like(ph)), truth)
        dP = dP / np.max(np.abs(dP))
        print(f"  family P: the prediction from {opt['pred_base']} at h {opt['pred_h']} lies {float(np.max(np.abs(qdiff((ph, pl, np.zeros_like(ph), np.zeros_like(ph)), truth)))):.2e} "
              f"from the truth", flush=True)
    for seed in [int(x) for x in opt["seeds"].split(",")]:
        d = dP if opt["family"] == "P" else direction(truth, M, K, opt["family"], seed)
        for delta in deltas:
            t0 = time.time()
            zp = c4(qd_axpy(delta, np.ascontiguousarray(d), np.zeros_like(d), np.zeros_like(d), np.zeros_like(d), *truth))
            F_start = qmax(sq.residual_q(*zp))
            rec = dict(test="basin", state=path, D=float(D), D_str=meta["D_str"], family=opt["family"], arm=opt["arm"], mutate=opt["mutate"],
                       seed=seed, delta=delta, F_truth=F_truth, F_start=F_start, start_err=float(np.max(np.abs(qdiff(zp, truth)))))
            print(f"  seed {seed}, delta {delta:.0e}: start max|F| {F_start:.2e}, max|z - truth| {rec['start_err']:.2e}", flush=True)
            if rs is not None:
                zr, diag = rs.reseed(zp, verbose=True)
                rec["reseed"] = dict(ok=diag["ok"], dropped=diag["dropped"], rungs=diag["rungs"], seconds=diag["seconds"],
                                     plan=[dict(u=p.get("u"), steps=p["steps"], max_x_minus_c=p["max_x_minus_c"], err_est=p.get("err_est"))
                                           for p in diag["plan"]])
                zp = c4(zr)
                rec["F_reseeded"] = qmax(sq.residual_q(*zp))
                rec["reseeded_err"] = float(np.max(np.abs(qdiff(zp, truth))))
                o = offsets(M, K)
                e = qdiff(zp, truth)
                rec["reseeded_err_mh"] = float(np.max(np.abs(e[o["mh"]:o["mh"] + M])))
                rec["reseeded_err_ah"] = float(np.max(np.abs(e[o["ah"]:o["ah"] + M])))
                print(f"    re-seeded ({'ok' if diag['ok'] else 'FAILED'}, dropped {diag['dropped']}): max|F| {rec['F_reseeded']:.2e}, "
                      f"|m_h - truth| {rec['reseeded_err_mh']:.1e}, |a_h - truth| {rec['reseeded_err_ah']:.1e}, {diag['seconds']} s", flush=True)
            t1 = time.time()
            solver = ElimQ(sq, rs) if opt["arm"] == "elim" else sq
            try:
                zc, hist, conv = newton_qd(solver, zp, tol=1e-28, it_max=opt["qd_it"], verbose=True)
            except ValueError as ex:          # a singular or non-finite Jacobian at the start
                zc, hist, conv = zp, [dict(error=str(ex))], False
            if opt["arm"] == "elim":
                if not conv:
                    zc = c4(rs.project(zc)[0])     # the eliminated system's point (a converged one is the system's own)
                rec["projections"] = dict(solver.counts)
                rec["F_final"] = qmax(sq.residual_q(*zc))
            e = qdiff(zc, truth)
            rec.update(conv=bool(conv), steps=len(hist), hist=hist, t_corr=round(time.time() - t1, 1), t_total=round(time.time() - t0, 1),
                       err=float(np.max(np.abs(e))), err_delta=float(abs(e[-1])))
            rec["back"] = bool(conv and rec["err"] <= 1e-20 and rec["err_delta"] <= 1e-24)
            print(f"    corrector: {len(hist)} steps, {'converged' if conv else 'not converged'}, max|z - truth| {rec['err']:.2e}, "
                  f"|Delta - truth| {rec['err_delta']:.2e} -> {'BACK' if rec['back'] else 'NOT BACK'}; {rec['t_corr']} s", flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            # --stop back: the first run not back at the truth ends the seed (3.3); conv: the first not converged
            if not (rec["back"] if opt["stop"] == "back" else rec["conv"]):
                break
    print("done", flush=True)
    return 0


def firststep(argv):
    """the first Newton step's nonlinearity, by unknown group: the truth perturbed by delta along seed 1's family-B
    direction restricted to one group (centre data, nodes, horizon data, Delta, or all), with or without the re-seed;
    one Solver.newton_q (the pinned quad-double Newton step dz at the start) per case. |dz + e| / |e| is the step's
    departure from the exact correction -e (small where the start is in the linear regime)."""
    path, out = argv[0], argv[1]
    opt = parse(argv, dict(deltas="1e-12,1e-14", groups="centre,nodes,data,Delta,all", arms="plain,reseed", threads=2, seed=1))
    require_raw(out)
    z4, meta = load_q(path)
    D = mp.mpf(meta["D_str"])
    M, K = int(meta["M"]), len(meta["nodes"])
    o = offsets(M, K)
    s, sp, sq = pinned(meta, D, z4, threads=opt["threads"])
    truth = c4(z4)
    rs = Reseeder(meta, D, threads=opt["threads"])
    dfull = direction(truth, M, K, "B", opt["seed"])
    span = dict(centre=(0, 2 * M), nodes=(2 * M, o["mh"]), data=(o["mh"], o["mh"] + 2 * M), Delta=(o["L"] - 1, o["L"]), all=(0, o["L"]),
                mh_lo=(o["mh"], o["mh"] + M), mh_hi=(o["mh"], o["mh"] + M), ah=(o["ah"], o["ah"] + M))
    from hpkernel.driver import plow_odd
    P = plow_odd(M, 21)
    P = np.asarray(P[0]) + np.asarray(P[1])        # the odd harmonics <= 21 of a field on the half grid
    for delta in [float(x) for x in opt["deltas"].split(",")]:
        for g in opt["groups"].split(","):
            d = np.zeros_like(dfull)
            a, b = span[g]
            d[a:b] = dfull[a:b]
            if g in ("mh_lo", "mh_hi"):
                lo = P @ dfull[a:b]
                d[a:b] = lo if g == "mh_lo" else dfull[a:b] - lo
            for arm in opt["arms"].split(","):
                if arm == "reseed" and g in ("data",):
                    continue
                t0 = time.time()
                zp = c4(qd_axpy(delta, np.ascontiguousarray(d), np.zeros_like(d), np.zeros_like(d), np.zeros_like(d), *truth))
                if arm == "reseed":
                    zp = c4(rs.reseed(zp)[0])
                e = qdiff(zp, truth)
                st = sq.newton_q(*zp)
                dz = np.asarray(st[0]) + np.asarray(st[1])
                rec = dict(test="firststep", state=path, D=float(D), delta=delta, group=g, arm=arm, e=float(np.max(np.abs(e))),
                           e_mh=float(np.max(np.abs(e[o["mh"]:o["mh"] + M]))), e_ah=float(np.max(np.abs(e[o["ah"]:o["ah"] + M]))),
                           e_Delta=float(abs(e[-1])), F=qmax(st[4:6]), dz=float(np.max(np.abs(dz))), dz_Delta=float(dz[-1]),
                           dz_plus_e=float(np.max(np.abs(dz + e))), piv=[float(st[6]), float(st[7])], seconds=round(time.time() - t0, 1))
                rec["nonlin"] = rec["dz_plus_e"] / max(rec["e"], 1e-300)
                print(f"  delta {delta:.0e}, {g:6s} {arm:6s}: |e| {rec['e']:.1e} (m_h {rec['e_mh']:.1e}, a_h {rec['e_ah']:.1e}, Delta "
                      f"{rec['e_Delta']:.1e}), max|F| {rec['F']:.1e}, |dz| {rec['dz']:.1e} (Delta {rec['dz_Delta']:+.1e}), |dz + e| / |e| "
                      f"{rec['nonlin']:.1e}; {rec['seconds']} s", flush=True)
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
    return 0


def summary(argv):
    """the basins of 3.3 from basin records (OUT.jsonl files): per (D, family, arm, mutate) and seed, the largest delta
    with every run up to it back at the truth (and, separately, converged); the arm's basin is the smallest over its
    seeds. A later record of the same (seed, delta) replaces an earlier one."""
    runs = {}
    for path in argv:
        for line in open(path):
            r = json.loads(line)
            if r.get("test") != "basin":
                continue
            runs[(round(r["D"], 4), r["family"], r["arm"], r.get("mutate", "none"), r["seed"], r["delta"])] = r
    groups = {}
    for (D, fam, arm, mu, seed, delta), r in runs.items():
        groups.setdefault((D, fam, arm, mu), {}).setdefault(seed, []).append((delta, r["back"], r["conv"]))
    print("| D | family | arm | seeds | basin (back) | basin (converged) | per seed back / converged / first failure |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        per = []
        for seed, lst in sorted(groups[key].items()):
            lst.sort()
            def edge(i):
                best = None
                for d, *flags in lst:
                    if flags[i]:
                        best = d
                    else:
                        return best, d
                return best, None
            (b, fb), (c, fc) = edge(0), edge(1)
            per.append((seed, b, c, fc if fc is not None else fb))
        fmt = lambda v: "none" if v is None else f"{v:.0e}"  # noqa: E731
        bb = None if any(p[1] is None for p in per) else min(p[1] for p in per)
        cc = None if any(p[2] is None for p in per) else min(p[2] for p in per)
        print(f"| {key[0]} | {key[1]} | {key[2]}{'' if key[3] == 'none' else ' (' + key[3] + ')'} | {len(per)} | {fmt(bb)} | {fmt(cc)} | "
              + ", ".join(f"{s_}: {fmt(b_)} / {fmt(c_)} / {fmt(f_)}" for s_, b_, c_, f_ in per) + " |")
    return 0


def rungs(argv):
    """amendment 9's diagnostic: from BASE the chain's quad-double series (the jump test's, cached beside OUT) summed at h
    as the jump test sums it, then re-seeded with each floor rule (abs, the first design, and rel), every rung's Newton
    history (residual and step per iteration) printed and recorded"""
    raw, path, out = argv[0], argv[1], argv[2]
    opt = parse(argv, dict(h=5.0, threads=2, N=24))
    require_raw(out)
    z4, meta = load_q(path)
    D0 = mp.mpf(meta["D_str"])
    cache = os.path.join(os.path.dirname(out), "jump", f"series_D{float(D0):.4f}_N{opt['N']}.npz")
    if not os.path.exists(cache):
        raise SystemExit(f"{cache}: run the jump test from this base first (it caches the series)")
    from anm_run import series_at
    cz = np.load(cache)
    Zh = [np.asarray(cz[f"z{k}_0"]) for k in range(opt["N"] + 1)]
    Zl = [np.asarray(cz[f"z{k}_1"]) for k in range(opt["N"] + 1)]
    norms = [float(v) for v in cz["norms"]]
    h = opt["h"]
    terms = [norms[k] * h ** k for k in range(1, opt["N"] + 1)]
    k_opt = int(np.argmin(terms)) + 1
    ph, pl = series_at(np.asarray(z4[0]), np.asarray(z4[1]), Zh[:k_opt + 1], Zl[:k_opt + 1], h)
    Dn = D0 + mp.mpf(h)
    G = float(meta["nodes"][-1]) ** -(float(Dn) - 3.0)
    print(f"rungs: from {path} at h {h:g} -> D {mp.nstr(Dn, 12)} (G {G:.2e}), series to order {k_opt}", flush=True)
    for rule in ("abs", "rel"):
        rs = Reseeder(meta, Dn, threads=opt["threads"], floor_rule=rule)
        zr, diag = rs.reseed((ph, pl), verbose=True)
        print(f"  floor rule {rule}: {'ok' if diag['ok'] else 'FAILED'}", flush=True)
        for r in diag["rungs"]:
            print(f"    u {r['u']:.6f} ({'converged' if r['conv'] else 'NOT converged'}): "
                  + "; ".join(f"|R| {x['R']:.2e} step {x['step']:.1e}" if x.get("step") is not None else f"|R| {x['R']:.2e}" for x in r["hist"]), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(dict(test="rungs", base=path, h=h, D=float(Dn), D_str=str(Dn), G=G, rule=rule, ok=diag["ok"],
                                    rungs=diag["rungs"], dropped=diag["dropped"]), default=float) + "\n")
    return 0


def floorflip(argv):
    """amendment 9's experiment: how often the first design's floor test (abs) fails at the u0 rung, against the relative
    one (rel). The target is BASE's last node (h = 0) or the cached series' prediction at h; the ladder runs once to u0/2,
    then the u0 rung is solved TRIALS times from that rung's data moved by 1e-25 (relative, random signs, seed = trial),
    which only changes the order in which the residual's floor noise arrives. Each trial records both rules' outcome,
    steps and residuals."""
    raw, path, out = argv[0], argv[1], argv[2]
    opt = parse(argv, dict(h=0.0, trials=20, threads=2, N=24))
    require_raw(out)
    z4, meta = load_q(path)
    D0 = mp.mpf(meta["D_str"])
    M, K = int(meta["M"]), len(meta["nodes"])
    o = offsets(M, K)
    if opt["h"] > 0:
        from anm_run import series_at
        cz = np.load(os.path.join(os.path.dirname(out), "jump", f"series_D{float(D0):.4f}_N{opt['N']}.npz"))
        Zh = [np.asarray(cz[f"z{k}_0"]) for k in range(opt["N"] + 1)]
        Zl = [np.asarray(cz[f"z{k}_1"]) for k in range(opt["N"] + 1)]
        norms = [float(v) for v in cz["norms"]]
        k_opt = int(np.argmin([norms[k] * opt["h"] ** k for k in range(1, opt["N"] + 1)])) + 1
        zh, zl = series_at(np.asarray(z4[0]), np.asarray(z4[1]), Zh[:k_opt + 1], Zl[:k_opt + 1], opt["h"])
    else:
        zh, zl = np.asarray(z4[0]), np.asarray(z4[1])
    Dn = D0 + mp.mpf(opt["h"])
    G = float(meta["nodes"][-1]) ** -(float(Dn) - 3.0)
    rs = {rule: Reseeder(meta, Dn, threads=opt["threads"], floor_rule=rule) for rule in ("abs", "rel")}
    r0 = rs["abs"]
    tg, _ = r0.targets(zh, zl)
    sl_ = slice(o["mh"], o["mh"] + 2 * M)
    dh, dl = np.array(zh[sl_]), np.array(zl[sl_])
    for u in r0.us[:-1]:
        dh, dl, hist, conv = r0.invert(u, zh, zl, *tg[u], dh, dl)
        if not conv:
            raise SystemExit(f"floorflip: the ladder failed at u {u}")
    u0 = r0.us[-1]
    th, tl = tg[u0]
    print(f"floorflip {path} h {opt['h']:g}: D {mp.nstr(Dn, 12)}, G {G:.2e}, floor_tol(u0) {1e-29 * G:.1e}", flush=True)
    counts = {"abs": 0, "rel": 0}
    for t in range(opt["trials"]):
        rng = np.random.default_rng(t)
        ph_ = dh * (1 + 1e-25 * rng.uniform(0.5, 1, 2 * M) * rng.choice([-1.0, 1.0], 2 * M))
        rec = dict(test="floorflip", base=path, h=opt["h"], D=float(Dn), G=G, trial=t)
        for rule in ("abs", "rel"):
            d2h, d2l, hist, conv = rs[rule].invert(u0, zh, zl, th, tl, ph_.copy(), dl.copy())
            counts[rule] += int(conv)
            rec[rule] = dict(conv=bool(conv), its=len(hist) - 1, R=[x.get("R") for x in hist], step=[x.get("step") for x in hist],
                             lam=[x.get("lam") for x in hist])
        print(f"  trial {t}: abs {'conv' if rec['abs']['conv'] else 'FAIL'} ({rec['abs']['its']} its, |R| {rec['abs']['R'][-1]:.2e}, steps "
              + " ".join(f"{x:.0e}" for x in rec["abs"]["step"] if x is not None) + f"); rel {'conv' if rec['rel']['conv'] else 'FAIL'} "
              f"({rec['rel']['its']} its, |R| {rec['rel']['R'][-1]:.2e})", flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
    print(f"  converged: abs {counts['abs']}/{opt['trials']}, rel {counts['rel']}/{opt['trials']}", flush=True)
    return 0


def trend_slopes(raw, M, ks_held, folder="T75"):
    """anm_run.py --trend: the exponents s of |v| ~ n^s per held coefficient from the stored states of folder (every fourth),
    the same rule (constant sign, at least 6 states, fit residual in ln|v| below 0.05; else 0)"""
    import glob
    from pinning import band_coeffs
    ns, vals = [], []
    for f in sorted(glob.glob(os.path.join(raw, folder, "pin*_D*.npz")))[::4]:
        dd = np.load(f)
        mm = json.loads(str(dd["meta"]))
        if int(mm["M"]) != M:
            continue
        ns.append(float(mm["D"]) - 2.0)
        vals.append(band_coeffs(dd["zh"][:M], dd["zl"][:M], M, ks_held))
    ns = np.array(ns)
    slopes = [[0.0, 0.0] for _ in ks_held]
    for i in range(len(ks_held)):
        for j in (0, 1):
            v = np.array([c[i][j] for c in vals])
            if len(v) >= 6 and (np.all(v > 0) or np.all(v < 0)):
                sl, c0 = np.polyfit(np.log(ns), np.log(np.abs(v)), 1)
                if float(np.max(np.abs(np.log(np.abs(v)) - (c0 + sl * np.log(ns))))) < 0.05:
                    slopes[i][j] = float(sl)
    return slopes


def jump(argv):
    """notes/RESEED.md 3.4 (the docstring)"""
    raw, path, out = argv[0], argv[1], argv[2]
    opt = parse(argv, dict(hs="2,5,10", arms="reseed,elim,plain", threads=2, N=24, qd_it=12, chain="anm-mac/run_trend_pin3_qdj1"))
    require_raw(out)
    from anm_run import coefficients_q, series_at
    from hpkernel.driver import make_solver, rnorm
    from pinning import PinnedQ, PinnedSolver, band_coeffs, band_field
    from taylor_predict import formula_delta
    z4, meta = load_q(path)
    D0 = mp.mpf(meta["D_str"])
    M, K, kf = int(meta["M"]), len(meta["nodes"]), int(meta["kfreeze"])
    zq = c4(z4)
    mk = lambda Dx: make_solver(Dx, M, nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], threads=opt["threads"],  # noqa: E731
                                kfreeze=kf)
    ks_held = [k for k in range(kf + 1, 14) if k % 2 == 1]
    base_coeffs = band_coeffs(zq[0][:M], zq[1][:M], M, ks_held)
    slopes = trend_slopes(raw, M, ks_held)
    n0 = float(D0) - 2.0

    def held_at(Dx):
        r = float(Dx) - 2.0
        return [(a * (r / n0) ** sa, b * (r / n0) ** sb) for (a, b), (sa, sb) in zip(base_coeffs, slopes)]

    def binom(sx, jj):
        r = 1.0
        for t in range(jj):
            r *= (sx - t) / (t + 1)
        return r

    def held_jet(j):
        return [(a * binom(sa, j) / n0 ** j, b * binom(sb, j) / n0 ** j) for (a, b), (sa, sb) in zip(base_coeffs, slopes)]

    s = mk(D0)
    segs = s.plan(zq[0], zq[1], 1e-32, 0.05)
    s.set_segs(segs)
    Kb = K
    rows = slice(Kb * 4 * M, Kb * 4 * M + M)
    t0 = time.time()
    cache = os.path.join(os.path.dirname(out), f"series_D{float(D0):.4f}_N{opt['N']}.npz")
    if os.path.exists(cache):
        # the series of an earlier run from the same base (the quad-double jet takes 10-16 min)
        cz = np.load(cache)
        Zq = [tuple(np.ascontiguousarray(cz[f"z{k}_{q}"]) for q in range(4)) for k in range(opt["N"] + 1)]
        norms = [float(v) for v in cz["norms"]]
    else:
        s.newton_q(*zq)
        Zq, norms = coefficients_q(s, zq, opt["N"], held=(lambda j: band_field(held_jet(j), ks_held, M), rows))
        np.savez(cache, norms=np.array(norms), **{f"z{k}_{q}": np.asarray(Zq[k][q]) for k in range(opt["N"] + 1) for q in range(4)})
    Zh, Zl = [np.asarray(z[0]) for z in Zq], [np.asarray(z[1]) for z in Zq]
    print(f"jump from {path}: D {mp.nstr(D0, 12)}; series N {opt['N']} in {time.time() - t0:.0f} s, |z_k| "
          + " ".join(f"{v:.1e}" for v in norms[::3]), flush=True)
    if opt["arms"] == "none":
        return 0                     # the series only (cached for rungs and family P)
    for h in [float(x) for x in opt["hs"].split(",")]:
        terms = [norms[k] * h ** k for k in range(1, opt["N"] + 1)]
        k_opt = int(np.argmin(terms)) + 1
        Dn = D0 + mp.mpf(h)
        sn = mk(Dn)
        sn.set_segs(segs)
        hh, hl = band_field(held_at(Dn), ks_held, M)
        spn = PinnedSolver(sn, M, hh, hl)
        sqn = PinnedQ(sn, spn.ch, spn.cl)
        # the series summed to the order of its smallest term at h (all N orders when they still fall there)
        ph, pl = series_at(zq[0], zq[1], Zh[:k_opt + 1], Zl[:k_opt + 1], h)
        Fp = float(rnorm(*spn.residual(ph, pl)))
        if np.isfinite(Fp) and Fp <= 1e-6:
            sn.set_segs(sn.plan(ph, pl, 1e-32, 0.05))
        zp = c4((ph, pl, np.zeros_like(ph), np.zeros_like(ph)))
        F_pred = qmax(sqn.residual_q(*zp))
        full_h, full_l = series_at(zq[0], zq[1], Zh, Zl, h)
        F_full = qmax(sqn.residual_q(*c4((full_h, full_l, np.zeros_like(ph), np.zeros_like(ph)))))
        # the chain's own state at D + h, if it has one (its D are the base's plus integer steps, exact in mpmath)
        chain = None
        cp = os.path.join(raw, opt["chain"], f"anm_pin3_D{float(Dn):.4f}.npz")
        if os.path.exists(cp):
            cz, cm = load_q(cp)
            if abs(mp.mpf(cm["D_str"]) - Dn) < mp.mpf(10) ** -40:
                chain = (cp, c4(cz))
        print(f"  h {h:g} -> D {mp.nstr(Dn, 12)}: series to order {k_opt} (smallest term {terms[k_opt - 1]:.1e}), predicted max|F| "
              f"{F_pred:.2e} (all {opt['N']} orders: {F_full:.2e}); chain state {'at ' + chain[0] if chain else 'none'}", flush=True)
        for arm in opt["arms"].split(","):
            t1 = time.time()
            rec = dict(test="jump", base=path, D0=float(D0), D0_str=str(D0), h=h, D=float(Dn), D_str=str(Dn), arm=arm, k_opt=k_opt,
                       term_min=terms[k_opt - 1], F_pred=F_pred, F_full=F_full, chain=chain[0] if chain else None)
            zs = zp
            rs = Reseeder(meta, Dn, threads=opt["threads"])
            if arm in ("reseed", "elim"):
                zr, diag = rs.reseed(zp, verbose=True)
                zs = c4(zr)
                rec["reseed"] = dict(ok=diag["ok"], dropped=diag["dropped"], rungs=diag["rungs"], seconds=diag["seconds"])
                rec["F_reseeded"] = qmax(sqn.residual_q(*zs))
                print(f"    re-seeded ({'ok' if diag['ok'] else 'FAILED'}): max|F| {rec['F_reseeded']:.2e}, {diag['seconds']} s", flush=True)
            solver = ElimQ(sqn, rs) if arm == "elim" else sqn
            try:
                zc, hist, conv = newton_qd(solver, zs, tol=1e-28, it_max=opt["qd_it"], verbose=True)
            except ValueError as ex:
                zc, hist, conv = zs, [dict(error=str(ex))], False
            if arm == "elim":
                if not conv:
                    zc = c4(rs.project(zc)[0])
                rec["projections"] = dict(solver.counts)
                rec["F_final"] = qmax(sqn.residual_q(*zc))
            Dl = float(zc[0][-1]) + float(zc[1][-1])
            rec.update(conv=bool(conv), steps=len(hist), hist=hist, t=round(time.time() - t1, 1), Delta=Dl,
                       Delta_str=mp.nstr(mp.mpf(float(zc[0][-1])) + mp.mpf(float(zc[1][-1])) + mp.mpf(float(zc[2][-1])), 34),
                       dDelta_formula=Dl - float(formula_delta(float(Dn))),
                       moved=float(np.max(np.abs(qdiff(zc, zp)))))
            if chain:
                e = qdiff(zc, chain[1])
                rec["vs_chain"] = float(np.max(np.abs(e)))
                rec["dDelta_chain"] = float(e[-1])
                rec["pass"] = bool(conv and abs(rec["dDelta_chain"]) <= 1e-20)
            else:
                rec["pass"] = bool(conv and abs(rec["dDelta_formula"]) <= 1e-6)
            print(f"    {arm}: {len(hist)} steps, {'converged' if conv else 'not converged'}, Delta {Dl:.15f} (formula {rec['dDelta_formula']:+.2e}"
                  + (f", chain {rec['dDelta_chain']:+.2e}, max|z - chain| {rec['vs_chain']:.1e}" if chain else "")
                  + f"), moved {rec['moved']:.1e}: {'PASS' if rec['pass'] else 'FAIL'}; {rec['t']} s", flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "floorflip":
        sys.exit(floorflip(sys.argv[2:]))
    if len(sys.argv) > 2 and sys.argv[1] == "rungs":
        sys.exit(rungs(sys.argv[2:]))
    if len(sys.argv) > 2 and sys.argv[1] == "summary":
        sys.exit(summary(sys.argv[2:]))
    if len(sys.argv) > 2 and sys.argv[1] == "firststep":
        sys.exit(firststep(sys.argv[2:]))
    if len(sys.argv) > 2 and sys.argv[1] == "jump":
        sys.exit(jump(sys.argv[2:]))
    if len(sys.argv) > 2 and sys.argv[1] == "basin":
        sys.exit(basin(sys.argv[2:]))
    print(__doc__)
    sys.exit(2)
