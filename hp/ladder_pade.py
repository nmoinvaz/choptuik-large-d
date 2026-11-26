#!/usr/bin/env python3
"""#1098: rational fits [L/M] in 1/(D-2) on a ladder, against ladder_fit.py's polynomial rule, on the planted truths.

Fit: Delta = P_L(x) / Q_M(x), x = 1/(D-2), P in the Chebyshev basis over the fitted x range, Q = 1 + sum_j q_j (x/x_max)^j
(so Delta_inf = P(0)), weighted least squares by Gauss-Newton (Levenberg-Marquardt damping) in mpmath at 50 digits,
started from the linearized problem min sum w^2 (P - y Q)^2. s: the propagated 1-sigma error of P(0) from the inverse
of J^T J at the solution (first order in the noise). The same rule as ladder_fit.select (acceptance, nested tests
over L and over the lowest point, the candidates' median and spread), with x = 1/(D-2) only and K read as L.

    cd hp && python3 ladder_pade.py [--ladder=A] [--M=2] [--truths=T1,T2,T3] [--sigmas=1e-22,1e-23] [--seeds=21-25]
                                    [--out=FILE]
Prints per truth, sigma and seed: the polynomial rule's |err| and F E (F from --F, default 1) and the rational rule's;
and the oracle (the best |err| over L at the lowest level, knowing the truth) for both forms.
"""
import json
import math
import sys

import mpmath as mp
import numpy as np

sys.argv = [sys.argv[0], "pade"] + sys.argv[1:]          # ladder_fit reads its options from argv[2:]
import ladder_fit as LF  # noqa: E402

mp.mp.dps = 50
args = LF.args
VAR = "1/(D-2)"


def rat_fit(xs, ys, w, L, M):
    """weighted LS of P_L / Q_M; returns (P(0), s, chi2, maxres) or None when it does not converge"""
    N = len(xs)
    lo, hi = min(xs), max(xs)
    xm = hi

    def T(x):
        u = (2 * x - (lo + hi)) / (hi - lo)
        r = [mp.mpf(1), u]
        for _ in range(L - 1):
            r.append(2 * u * r[-1] - r[-2])
        return r[:L + 1]
    Ts = [T(x) for x in xs]
    us = [x / xm for x in xs]
    # linearized start: w (P - y Q) = 0 -> w (P - y sum q_j u^j) = w y
    A = mp.matrix(N, L + 1 + M)
    b = mp.matrix(N, 1)
    for i in range(N):
        for k in range(L + 1):
            A[i, k] = w[i] * Ts[i][k]
        for j in range(M):
            A[i, L + 1 + j] = -w[i] * ys[i] * us[i] ** (j + 1)
        b[i] = w[i] * ys[i]
    try:
        p = mp.lu_solve(A.T * A, A.T * b)
    except ZeroDivisionError:
        return None

    def resid(p):
        r = mp.matrix(N, 1)
        J = mp.matrix(N, L + 1 + M)
        for i in range(N):
            P = mp.fsum(p[k] * Ts[i][k] for k in range(L + 1))
            Q = 1 + mp.fsum(p[L + 1 + j] * us[i] ** (j + 1) for j in range(M))
            r[i] = w[i] * (P / Q - ys[i])
            for k in range(L + 1):
                J[i, k] = w[i] * Ts[i][k] / Q
            for j in range(M):
                J[i, L + 1 + j] = -w[i] * P * us[i] ** (j + 1) / Q ** 2
        return r, J
    lam = mp.mpf("1e-6")
    r, J = resid(p)
    c2 = mp.fsum(v ** 2 for v in r)
    for it in range(60):
        JTJ = J.T * J
        g = J.T * r
        Ad = JTJ + lam * mp.diag([JTJ[k, k] for k in range(L + 1 + M)])
        try:
            dp = mp.lu_solve(Ad, -g)
        except ZeroDivisionError:
            return None
        pn = p + dp
        rn, Jn = resid(pn)
        cn = mp.fsum(v ** 2 for v in rn)
        if cn < c2:
            rel = (c2 - cn) / max(c2, mp.mpf("1e-300"))
            p, r, J, c2 = pn, rn, Jn, cn
            lam = max(lam / 10, mp.mpf("1e-20"))
            if rel < mp.mpf("1e-12"):
                break
        else:
            lam *= 10
            if lam > 1e10:
                break
    # Q must not vanish on [0, x_max]
    for t in range(0, 41):
        u = mp.mpf(t) / 40
        if 1 + mp.fsum(p[L + 1 + j] * u ** (j + 1) for j in range(M)) <= 0:
            return None
    try:
        C = mp.inverse(J.T * J)
    except ZeroDivisionError:
        return None
    t0 = T(mp.mpf(0))
    v0 = mp.fsum(p[k] * t0[k] for k in range(L + 1))
    var = mp.fsum(t0[i] * C[i, j] * t0[j] for i in range(L + 1) for j in range(L + 1))
    return v0, math.sqrt(max(float(var), 0.0)), float(c2), float(max(abs(v) for v in r))


def recs_rat(L_, ys, sig, M, lim):
    recs = {}
    for lev in range(LF.RULE["NLEV"] + 1):
        idx = list(range(lev, len(L_)))
        if len(idx) < LF.RULE["NMIN"]:
            continue
        xs = [1 / (mp.mpf(L_[i]) - 2) for i in idx]
        for Lr in range(1, min(LF.RULE["KMAX"], len(idx) - 3 - M) + 1):
            f = rat_fit(xs, [ys[i] for i in idx], [1 / mp.mpf(sig[i]) for i in idx], Lr, M)
            if f is None:
                continue
            v0, s, c2, mr = f
            recs[(VAR, lev, Lr)] = dict(d=float(v0 - lim), s=s, chi2=c2, dof=len(idx) - Lr - 1 - M, maxres=mr)
    return recs


def recs_poly(O, TP, sg, z):
    recs = {}
    for key, (idx, g, P, gf, Pf) in O.items():
        if key[0] != VAR:
            continue
        bias, r0 = TP[key]
        zz = z[idx]
        rr = r0 / sg + Pf @ zz
        recs[key] = dict(d=bias + sg * float(gf @ zz), s=sg * float(np.linalg.norm(gf)), chi2=float(rr @ rr),
                         dof=len(idx) - key[2] - 1, maxres=float(np.max(np.abs(rr))))
    return recs


def main():
    lk = args.get("ladder", "A")
    M = int(args.get("M", "2"))
    Fc = float(args.get("F", "1"))
    truths = args.get("truths", "T1,T2,T3").split(",")
    sigmas = [float(v) for v in args.get("sigmas", "1e-22,1e-23").split(",")]
    seeds = []
    for part in args.get("seeds", "21-25").split(","):
        a_, b_ = (int(v) for v in (part.split("-") * 2)[:2])
        seeds += list(range(a_, b_ + 1))
    name, L_ = LF.LADDERS[lk]
    LF.VARS = {VAR: LF.VARS[VAR]}                       # the rule over one variable
    O = LF.ladder_ops(L_)
    Z, names = LF.patterns(L_, seeds)
    lim = LF.RLIM
    out = []
    print("ladder %s (%s), [L/%d] in %s against the polynomial rule; F %.2f" % (lk, name, M, VAR, Fc), flush=True)
    for kind in truths:
        TP = LF.truth_parts(O, L_, kind)
        tv = [LF.truth(kind, d) for d in L_]
        for sg in sigmas:
            for k, sd in enumerate(seeds):
                z = Z[:, k]
                rp = recs_poly(O, TP, sg, z)
                ep = LF.select(rp, L_)
                ys = [tv[i] + mp.mpf(float(sg * z[i])) for i in range(len(L_))]
                rr = recs_rat(L_, ys, [sg] * len(L_), M, lim)
                er = LF.select(rr, L_)
                orp = min((abs(r["d"]) for kk, r in rp.items() if kk[1] == 0), default=float("nan"))
                orr = min((abs(r["d"]) for kk, r in rr.items() if kk[1] == 0), default=float("nan"))
                row = dict(truth=kind, sigma=sg, seed=sd,
                           poly=None if ep is None else dict(err=ep["d"], E=Fc * ep["E0"], K=ep["K"], D0=ep["D0"]),
                           rat=None if er is None else dict(err=er["d"], E=Fc * er["E0"], L=er["K"], D0=er["D0"]),
                           oracle_poly=orp, oracle_rat=orr)
                out.append(row)
                fmt = lambda e: "none" if e is None else "|err| %.1e F E %.1e (%d, D0 %g)" % (  # noqa: E731
                    abs(e["err"]), e["E"], e.get("K", e.get("L")), e["D0"])
                print("  %s sigma %.0e seed %d: poly %s; [L/%d] %s; oracle poly %.1e, rat %.1e" % (
                    kind, sg, sd, fmt(row["poly"]), M, fmt(row["rat"]), orp, orr), flush=True)
    # F per form: the largest |err| / E0 over the seeds <= 20 of T1-T4 (E0 = E / Fc); coverage on the seeds > 20
    summ = {}
    for form in ("poly", "rat"):
        rs = [r for r in out if r[form] is not None and r["truth"] in LF.CAL and r["seed"] <= 20]
        F = max([1.0] + [abs(r[form]["err"]) / (r[form]["E"] / Fc) for r in rs])
        summ[form] = dict(F=F, rows={})
        for kind in truths:
            for sg in sigmas:
                q = [r for r in out if r["truth"] == kind and r["sigma"] == sg]
                ok = [r for r in q if r[form] is not None]
                val = [r for r in ok if r["seed"] > 20]
                cov = sum(abs(r[form]["err"]) <= F * r[form]["E"] / Fc for r in val)
                Es = sorted(F * r[form]["E"] / Fc for r in ok)
                errs = sorted(abs(r[form]["err"]) for r in ok)
                row = dict(quoted=len(ok), of=len(q), E_med=Es[len(Es) // 2] if Es else None,
                           E_90=Es[min(len(Es) - 1, int(math.ceil(0.9 * len(Es))) - 1)] if Es else None,
                           err_med=errs[len(errs) // 2] if errs else None, err_max=errs[-1] if errs else None,
                           cover_val="%d/%d" % (cov, len(val)))
                summ[form]["rows"]["%s %.0e" % (kind, sg)] = row
        print("%s: F %.2f (seeds <= 20 of %s)" % (form, F, "+".join(LF.CAL)))
        for key, row in summ[form]["rows"].items():
            print("  %s %s: quoted %d/%d  F E med %s 90%% %s  |err| med %s max %s  cover(seeds > 20) %s" % (
                form, key, row["quoted"], row["of"], *("%.1e" % v if v is not None else "-" for v in (
                    row["E_med"], row["E_90"], row["err_med"], row["err_max"])), row["cover_val"]))
    if "out" in args:
        json.dump(dict(cases=out, summary=summ), open(args["out"], "w"), indent=1)


if __name__ == "__main__":
    main()
