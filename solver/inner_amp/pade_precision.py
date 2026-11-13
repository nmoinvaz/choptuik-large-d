#!/usr/bin/env python3
"""Precision audit of the real S_1 inputs to the Pade and twin scripts (issue #683; notes/PADE_FAR_END.md section 10).

sigma_load.py reads sigma_k from the files' 20-digit strings at 300 digits. Before #683, pade_geometry.load_scan and
pade_twin_count.real_sig parsed them at the precision in force, 53 bits on a script's first load. This script measures
what that did, and what the strings' own rounding leaves.

  plants   the plants' Chebyshev coefficients c_m and sigma_0..sigma_30, with the real template sigma_0..sigma_12 read
           at 53 bits, 60 and 100 digits (the precisions the callers used) against 300 digits. The c_m come from a
           double-precision least-squares fit, so a load that leaves them bit-identical leaves every plant unchanged.
  compare  every number of two JSON records of one stage (a run and its rerun): the largest absolute and relative
           differences, with their paths, and any difference of structure. --skip leaves out the paths that match a
           regular expression (rounding noise, such as chi of an interpolant).
  floor    the strings' rounding as an input box: each component of sigma_k, k >= 1, within half a unit of its 20th
           digit (sigma_load.rounding; sigma_0 = 1 by normalisation). The first-order worst case of a statistic over
           the box is sum_i |d stat/d x_i| h_i, from one finite difference per component, h_i its half unit, at the
           working precision. Statistics: the [k/k] poles, R_k and bow of pade_geometry (k = 6..11, the base's pole
           classes kept), and the DA2 members and median of pade_farend. Control: at the corner of the box that
           maximises one statistic, the actual shift must equal the first-order bound within 1% (--mutate=corner
           doubles the corner, which must make it fail).
  floor_twins  the same for the twins' residuals at M = 22 (pade_twin_size: chord, arc, smooth). The residual is linear
           in the target, r = (I - Q Q^T) W sigma with Q from mp.qr, so its shift over the box is exact to first order.
           Controls: chi from Q equals tc.lsq's within 1e-6 relative, and a refit by tc.lsq with one component moved
           moves r as Q predicts within 1e-6 (--mutate=proj drops the last column of Q, which must make both fail).
           Then the twin scans with err_k widened to max(err_k, rounding_k).

    python3 pade_precision.py plants OUTDIR [--allow_unraw]
    python3 pade_precision.py compare OLD.json NEW.json [--top=N] [--out=FILE.json] [--skip=REGEX]
    python3 pade_precision.py floor OUTDIR [--mutate=corner] [--allow_unraw]
    python3 pade_precision.py floor_twins OUTDIR [--mutate=proj] [--allow_unraw]

Exit 1 when a control fails. Numerical evidence: the first-order bound is not a rigorous enclosure.
"""
import bisect
import json
import math
import os
import re
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pade_twin_size as ts                                            # noqa: E402  (imports tc, F, pg, pp)
import sigma_load as sl                                                # noqa: E402

tc, F, pg, pp = ts.tc, ts.F, ts.pg, ts.pp
N23 = ts.N23


class Log:
    def __init__(self, outdir, name):
        self.f = open(os.path.join(outdir, name + ".log"), "a")
        self("run " + time.strftime("%Y-%m-%d %H:%M", time.gmtime()) + " UTC")

    def __call__(self, s):
        print(s)
        self.f.write(s + "\n")
        sys.stdout.flush()
        self.f.flush()


def dump(outdir, name, rec):
    json.dump(pg.jsonable(rec), open(os.path.join(outdir, name + ".json"), "w"), indent=1)


# ---------------------------------------------------------------- plants
def run_plants(outdir):
    log = Log(outdir, "plants_c")
    precs = {"53bit": 53, "60dps": mp.libmp.dps_to_prec(60), "100dps": mp.libmp.dps_to_prec(100)}
    full = sl.load_scan(pg.REAL["J1250"], pg.NODES)
    loads = {lab: sl.load_scan(pg.REAL["J1250"], pg.NODES, prec=p) for lab, p in precs.items()}
    pg.NOISE_MODE = "measured"
    rec = {}
    builders = {"FarPlant": lambda wf, kind, s: F.FarPlant(wf, kind, s),
                "Plant": lambda wf, kind, s: pg.Plant(wf, kind, s)}
    for wf in pg.NODES:
        rel = pg.noise_eps(wf)[:N23]
        for cls, kind in (("FarPlant", "chord2"), ("FarPlant", "arcjump"), ("FarPlant", "arc"),
                          ("Plant", "chord2"), ("Plant", "arcjump")):
            ref = builders[cls](wf, kind, full[wf])
            mp.mp.dps = pg.DPS_PLANT
            s_ref = ref.sigma()[:tc.NMAX]
            for lab in precs:
                pl = builders[cls](wf, kind, loads[lab][wf])
                mp.mp.dps = pg.DPS_PLANT
                s = pl.sigma()[:tc.NMAX]
                cz = [complex(z) for z in pl.c]
                cr = [complex(z) for z in ref.c]
                n_diff = sum(a != b for a, b in zip(cz, cr))
                dc = max(abs(a - b) / abs(b) for a, b in zip(cz, cr))
                ds = max(float(abs(a - b) / abs(b)) for a, b in zip(s, s_ref))
                de = max(float(abs(s[k] - s_ref[k]) / (rel[k] * abs(s_ref[k]))) for k in range(1, N23))
                r = {"M": pl.M, "c_differ": n_diff, "c_rel_max": dc, "sigma_rel_max": ds, "sigma_err_units_max": de}
                rec.setdefault(f"{cls} {kind}", {}).setdefault(wf, {})[lab] = r
                log(f"wf {wf} {cls:8s} {kind:7s} template at {lab:6s} against 300 digits: {n_diff} of {pl.M} c_m "
                    f"differ (largest rel {dc:.1e}); sigma_0..30 largest rel {ds:.1e}; sigma_1..22 largest "
                    f"{de:.1e} error units")
    dump(outdir, "plants_c", rec)
    return True


# ---------------------------------------------------------------- compare
def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out["structure"].append(f"{path}/{k}: only in {'new' if k not in a else 'old'}")
            else:
                walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["structure"].append(f"{path}: length {len(a)} -> {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or \
            isinstance(b, str):
        if a != b:
            out["structure"].append(f"{path}: {a!r} -> {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isfinite(a) and math.isfinite(b):
            d = abs(a - b)
            m = max(abs(a), abs(b))
            out["numbers"].append((d / m if m > 0 else 0.0, d, path, a, b))
        elif a != b:
            out["structure"].append(f"{path}: {a!r} -> {b!r}")
    else:
        out["structure"].append(f"{path}: type {type(a).__name__} -> {type(b).__name__}")


def run_compare(old, new, top=12, out_json=None, skip=None):
    A, B = json.load(open(old)), json.load(open(new))
    for k in ("frozen", "mutate"):
        A.pop(k, None), B.pop(k, None)
    out = {"structure": [], "numbers": []}
    walk(A, B, "", out)
    rx = re.compile(skip) if skip else None
    nums = sorted(n for n in out["numbers"] if not (rx and rx.search(n[2])))
    nums.reverse()
    print(f"{old} -> {new}: {len(nums)} numbers" + (f" (skipping /{skip}/, {len(out['numbers']) - len(nums)} left out)"
                                                    if skip else "") + f", {len(out['structure'])} structural differences")
    for s in out["structure"][:top]:
        print("   structure:", s)
    moved = [n for n in nums if n[0] > 0]
    print(f"   numbers that differ: {len(moved)}; largest relative differences:")
    for rel, d, p, a, b in nums[:top]:
        print(f"   {rel:.2e} (abs {d:.2e})  {p}: {a!r} -> {b!r}")
    res = {"old": old, "new": new, "skip": skip, "n_numbers": len(nums), "n_differ": len(moved),
           "max_rel": nums[0][0] if nums else 0.0, "structure": out["structure"],
           "largest": [{"rel": r, "abs": d, "path": p, "old": a, "new": b} for r, d, p, a, b in nums[:200]]}
    if out_json:
        json.dump(res, open(out_json, "w"), indent=1)
    return res


# ---------------------------------------------------------------- floor: the input box
def box(fn):
    """[(k, part, h)] for k = 1..22, part 0 (Re) or 1 (Im), h the half unit of the string's last digit"""
    nd, rows = sl.rounding(json.load(open(fn)))
    return [(k, part, rows[k][part]) for k in range(1, N23) for part in (0, 1) if rows[k][part] > 0], rows


def moved(sig, k, part, h):
    s = list(sig)
    z = s[k]
    s[k] = mp.mpc(z.real + h, z.imag) if part == 0 else mp.mpc(z.real, z.imag + h)
    return s


def corner(sig, comps, signs, scale=1):
    s = list(sig)
    for (k, part, h), g in zip(comps, signs):
        z = s[k]
        s[k] = mp.mpc(z.real + scale * g * h, z.imag) if part == 0 else mp.mpc(z.real, z.imag + scale * g * h)
    return s


def nearest(t, ts_):
    return min(ts_, key=lambda z: abs(z - t))


def d_arc_mp(fr, s):
    j = min(max(bisect.bisect_right(fr.prof_s, float(s)) - 1, 0), len(fr.prof_s) - 2)
    s0, s1 = mp.mpf(fr.prof_s[j]), mp.mpf(fr.prof_s[j + 1])
    d0, d1 = mp.mpf(fr.prof_d[j]), mp.mpf(fr.prof_d[j + 1])
    return d0 + (d1 - d0) * (s - s0) / (s1 - s0)


def pade_stats(sig, k, fr, base=None):
    """the [k/k] poles in mp. For the base's geometry poles (pade_geometry.geometry's classes, matched by distance):
    R and bow. 'far' is pade_geometry's far pole (the geometry pole of largest s); 'past' is the pole of largest s
    with 0.97 < s <= 1.2 and |d| <= 0.5 of any class (D_entire 7.4's pole past kappa_*, a near-doublet), with dz the
    distance to its nearest numerator zero"""
    mp.mp.dps = pg.DPS_PADE
    poles = pg.pade(sig, k)
    ts_ = [p["t"] for p in poles]
    P0, D = mp.mpc(fr.P0), mp.mpc(fr.D)

    def sd(t):
        w = (t - P0) / D
        return mp.re(w), mp.im(w) * abs(D)
    if base is None:
        g = pg.geometry(poles, fr)
        geo = [nearest(mp.mpc(*r["t"]), ts_) for r in g["poles"] if r["class"] == "geometry"]
        past = [t for t in ts_ if 0.97 < sd(t)[0] <= 1.2 and abs(sd(t)[1]) <= 0.5 and abs(t) < 3]
        base = {"geo": geo, "far": max(geo, key=lambda t: sd(t)[0]),
                "past": max(past, key=lambda t: sd(t)[0]) if past else None}
    geo = [nearest(t, ts_) for t in base["geo"]]
    num = den = mp.mpf(0)
    bow = None
    for t in geo:
        s, d = sd(t)
        bow = d if bow is None or d > bow else bow
        if 0.05 <= s <= 0.95:
            da = d_arc_mp(fr, s)
            num += d * da
            den += da * da
    out = {"geo": geo, "far": nearest(base["far"], ts_), "R": num / den if den > 0 else None, "bow": bow,
           "past": None, "past_dz": None}
    if base["past"] is not None:
        p = min(poles, key=lambda q: abs(q["t"] - base["past"]))
        out["past"], out["past_dz"] = p["t"], p["dz"]
    return base, out


def da_members(sig):
    """each DA2 member's farthest admissible zero, in mp (pade_farend.estimate's window)"""
    mp.mp.dps = F.DPS_DA
    s = [mp.mpc(z) for z in sig[:F.N_COEF]]
    out = []
    for degs, lp in F.members(2):
        try:
            zs = F.da_zeros(s, degs, lp)
        except ZeroDivisionError:
            zs = None
        ok = [t for t in zs or [] if F.WINDOW[0] < abs(t) < F.WINDOW[1] and mp.im(t) > 0]
        out.append((max(ok, key=abs) if ok else None, zs or []))
    return out


def median_mp(vals):
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def run_floor(outdir, mut=None):
    log = Log(outdir, "floor" + (f"_mutate_{mut}" if mut else ""))
    rec = {"mutate": mut, "pade": {}, "da2": {}, "controls": {}}
    ok = True
    sets = [("J1250", wf) for wf in pg.NODES] + [("J1500", 1.0)]
    for lab, wf in sets:
        fn = sl.node_file(pg.REAL[lab], wf)
        sig = sl.load_file(fn)[0][:N23]
        comps, rows = box(fn)
        fr = pg.Frame(wf)
        t0 = time.time()
        # Pade: k = 6..11
        for k in pg.KS:
            base, b0 = pade_stats(sig, k, fr)
            dgeo = [mp.mpf(0)] * len(b0["geo"])
            dfar = dR = dbow = dpast = ddz = mp.mpf(0)
            dpast_re, dR_i = [], []
            for c in comps:
                _, b1 = pade_stats(moved(sig, *c), k, fr, base)
                for j, (x, y) in enumerate(zip(b1["geo"], b0["geo"])):
                    dgeo[j] += abs(x - y)
                dfar += abs(b1["far"] - b0["far"])
                if b0["past"] is not None:
                    dpast += abs(b1["past"] - b0["past"])
                    dpast_re.append(mp.re(b1["past"] - b0["past"]))
                    ddz += abs(b1["past_dz"] - b0["past_dz"])
                if b0["R"] is not None:
                    dR_i.append(b1["R"] - b0["R"])
                    dR += abs(b1["R"] - b0["R"])
                dbow += abs(b1["bow"] - b0["bow"])
            r = {"n_geo": len(b0["geo"]), "geo_shift_max": max(dgeo) if dgeo else None, "far": b0["far"],
                 "far_shift": dfar, "past": b0["past"], "past_abs": abs(b0["past"]) if b0["past"] is not None
                 else None, "past_shift": dpast, "past_dz": b0["past_dz"], "past_dz_shift": ddz, "R": b0["R"],
                 "R_shift": dR, "bow": b0["bow"], "bow_shift": dbow}
            rec["pade"].setdefault(lab, {}).setdefault(wf, {})[k] = r
            past = "none" if b0["past"] is None else (
                f"{complex(b0['past']):.4f} (|t| {float(abs(b0['past'])):.4f}, dz {float(b0['past_dz']):.1e}) shift "
                f"{float(dpast):.1e}, dz shift {float(ddz):.1e}")
            log(f"{lab} wf {wf} [{k}/{k}]: {len(b0['geo'])} geometry poles, largest first-order shift over the box "
                f"{float(max(dgeo)):.1e}; far geometry pole shift {float(dfar):.1e}; pole past s = 0.97 {past}; "
                f"R {float(b0['R']) if b0['R'] is not None else float('nan'):+.4f} shift {float(dR):.1e}; bow "
                f"{float(b0['bow']):+.4f} shift {float(dbow):.1e}")
            if lab == "J1250" and wf == 1.0 and k == 10:      # the control: the box corners for Re t_past and R
                for name, di, val in (("Re t of the pole past kappa_*", dpast_re, lambda b: mp.re(b["past"])),
                                      ("R_10", dR_i, lambda b: b["R"])):
                    signs = [1 if x >= 0 else -1 for x in di]
                    pred = mp.fsum(abs(x) for x in di)
                    _, bc = pade_stats(corner(sig, comps, signs, 2 if mut == "corner" else 1), k, fr, base)
                    act = val(bc) - val(b0)
                    q = float(act / pred)
                    good = abs(q - 1) <= 0.01
                    ok &= good
                    rec["controls"][f"corner {name} J1250 wf 1.0 k 10"] = {"predicted": pred, "actual": act,
                                                                          "ratio": q, "pass": good}
                    log(f"  control, the box corner for {name} at k = 10: actual shift {float(act):.4e}, first-order "
                        f"bound {float(pred):.4e}, ratio {q:.5f} (pass within 1%) {'OK' if good else 'BAD'}")
        # DA2 members and median
        m0 = da_members(sig)
        dm = [mp.mpf(0)] * len(m0)
        for c in comps:
            m1 = da_members(moved(sig, *c))
            for j, ((f0, _), (_, z1)) in enumerate(zip(m0, m1)):
                if f0 is not None and z1:
                    dm[j] += abs(nearest(f0, z1) - f0)
        fs = [f for f, _ in m0 if f is not None]
        med = mp.mpc(median_mp([mp.re(f) for f in fs]), median_mp([mp.im(f) for f in fs]))
        dmed = max(d for (f, _), d in zip(m0, dm) if f is not None)
        rec["da2"].setdefault(lab, {})[wf] = {"members": [f for f, _ in m0], "member_shift": dm, "median": med,
                                              "median_abs": abs(med), "median_shift_bound": dmed}
        log(f"{lab} wf {wf} DA2: median {complex(med):.4f} (|t| {float(abs(med)):.4f}); members' farthest zeros move "
            f"by at most {float(max(dm)):.1e} over the box, so the median by at most {float(dmed):.1e} "
            f"({time.time() - t0:.0f}s)")
    log(f"floor controls pass: {ok}")
    rec["controls_pass"] = ok
    dump(outdir, "floor" + (f"_mutate_{mut}" if mut else ""), rec)
    return ok


# ---------------------------------------------------------------- floor: the twins
def real_form(basis, sig, err, n, M):
    """tc.lsq's real system: A (2n x 2M) and b (2n), rows k and n + k the real and imaginary parts"""
    A, b = mp.matrix(2 * n, 2 * M), mp.matrix(2 * n, 1)
    for k in range(n):
        for m in range(M):
            z = basis.cols[m][k] / err[k]
            A[k, m], A[k, M + m], A[n + k, m], A[n + k, M + m] = z.real, -z.imag, z.imag, z.real
        z = mp.mpc(sig[k]) / err[k]
        b[k], b[n + k] = z.real, z.imag
    return A, b


def resid(Q, v, n):
    """complex residual (I - Q Q^T) v, per order"""
    qv = Q.T * v
    r = v - Q * qv
    return [mp.mpc(r[k], r[n + k]) for k in range(n)]


def run_floor_twins(outdir, mut=None):
    log = Log(outdir, "floor_twins" + (f"_mutate_{mut}" if mut else ""))
    rec = {"mutate": mut, "scans": {}, "widened": {}, "controls": {}}
    ok = True
    real = tc.real_sig()
    n, M = N23, N23 - 1
    for wf in pg.NODES:
        fn = sl.node_file(pg.REAL["J1250"], wf)
        comps, rows = box(fn)
        fit, chk = ts.bases_part1(wf, ts.DPS_FIT), ts.bases_part1(wf, ts.DPS_CHECK)
        rel, _ = tc.err_model(wf)
        sig = real[wf][:n]
        for cls in ("chord", "arc", "smooth"):
            t0 = time.time()
            b = fit[cls]
            with mp.workdps(b.dps):
                err = tc.weights(sig, rel, n, 1.0, b.dps)
                A, v = real_form(b, sig, err, n, M)
                Q, _ = mp.qr(A, mode="skinny")
                if mut == "proj":
                    Q = Q[:, :Q.cols - 1]
                r0 = resid(Q, v, n)
                chi_q = max(abs(z) for z in r0)
                _, chi_l, _ = tc.lsq(b, sig, err, n, M)
                q1 = abs(float(chi_q) / chi_l - 1)
                # one component moved, refitted by tc.lsq, against Q's prediction
                k1, p1, h1 = next(c for c in comps if c[0] == 3)
                s1 = moved(sig, k1, p1, h1 * 10 ** 6)
                c1, _, _ = tc.lsq(b, s1, err, n, M)
                r1 = [mp.fsum(c1[m] * b.cols[m][k] for m in range(M)) / err[k] - mp.mpc(s1[k]) / err[k]
                      for k in range(n)]
                e = mp.matrix(2 * n, 1)
                e[k1 + p1 * n] = h1 * 10 ** 6 / err[k1]
                pr = resid(Q, e, n)
                dr_fit = [-(x) - y for x, y in zip(r1, r0)]       # tc's residual sign is fit - target
                q2 = float(max(abs(x - y) for x, y in zip(dr_fit, pr)) / max(abs(y) for y in pr))
                # the box: the first-order shift of every |r_k|, and of chi
                shift = [mp.mpf(0)] * n
                for k, part, h in comps:
                    e = mp.matrix(2 * n, 1)
                    e[k + part * n] = h / err[k]
                    for j, z in enumerate(resid(Q, e, n)):
                        shift[j] += abs(z)
                dchi = max(shift)
            good = q1 <= 1e-6 and q2 <= 1e-6
            ok &= good
            rec["scans"].setdefault(wf, {})[cls] = {"chi_22": chi_l, "chi_22_from_Q": chi_q, "rel_diff_chi": q1,
                                                   "one_component_rel_diff": q2, "control_pass": good,
                                                   "chi_shift_bound": dchi, "shift_per_k": shift}
            log(f"wf {wf} {cls:6s}: chi at M = {M} {chi_l:.4e} (from Q {float(chi_q):.4e}, rel diff {q1:.1e}); a refit "
                f"with Re/Im sigma_3 moved matches Q to {q2:.1e} (pass both <= 1e-6) {'OK' if good else 'BAD'}; "
                f"first-order shift of chi over the box <= {float(dchi):.2e} error units, largest at k = "
                f"{max(range(n), key=lambda j: shift[j])} ({time.time() - t0:.0f}s)")
        if mut:
            continue
        rel_w = [max(rel[k], float(rows[k][2])) if k >= 1 else rel[k] for k in range(n)] + list(rel[n:])
        for cls in ("chord", "arc", "smooth"):
            s = ts.scan(fit[cls], chk[cls], sig, rel_w, n, log, f"wf {wf} {cls} twin, err_k widened to the rounding")
            rec["widened"].setdefault(wf, {})[cls] = s
            ok &= s["C1_pass"]
    rec["controls_pass"] = ok
    log(f"floor_twins controls pass: {ok}")
    dump(outdir, "floor_twins" + (f"_mutate_{mut}" if mut else ""), rec)
    return ok


def main():
    argv = sys.argv[1:]
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "hp"))
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    top = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--top=")), 12))
    out_json = next((a.split("=", 1)[1] for a in argv if a.startswith("--out=")), None)
    skip = next((a.split("=", 1)[1] for a in argv if a.startswith("--skip=")), None)
    argv = [a for a in argv if not a.startswith("--")]
    if len(argv) == 3 and argv[0] == "compare":
        run_compare(argv[1], argv[2], top, out_json, skip)
        sys.exit(0)
    if len(argv) != 2 or argv[0] not in ("plants", "floor", "floor_twins"):
        raise SystemExit(__doc__)
    stage, outdir = argv
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    os.makedirs(outdir, exist_ok=True)
    fn = {"plants": lambda: run_plants(outdir), "floor": lambda: run_floor(outdir, mut),
          "floor_twins": lambda: run_floor_twins(outdir, mut)}[stage]
    sys.exit(0 if fn() else 1)


if __name__ == "__main__":
    main()
