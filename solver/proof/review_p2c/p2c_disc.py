#!/usr/bin/env python3
"""P2c review (#992): lemma 6b1's M = 1 disc certificates (lemma D of A_certify.md 12.2), a second implementation,
written from A_certify.md 12.2's statement and 4.4's equations, not from lemma6b1_disc.py (not opened).

System (A_certify.md 4.4 in u = 1/z, 11.4; the state of lemma6_usys): Y = (P-1, P+1, V-1, V+1, kC-2, k(C0 - 1), kC+2),
k = 100, parameter w = omega (s_mu = i mu w), z = 1/u, with Q = V + (1 + s)P, W = V + [(C - 1)Q], E = [P^2/2]:
    P_u = -z (V + [(C-1)Q]) / 2
    V_u = -(V + [(C-1)Q] + [(C-1)W]) / 2 - z (-(2 + s)Q - (1 + s)W + [C E P]) / 2
    C_u = -z^2 [C E] / 2                       ([.] the product projected on the left side's harmonics)

Lemma D, as implemented here (own code, python-flint 0.9 acb balls at 192 bits):
  1. Taylor coefficients p_0..p_n at the centre c (a ball) from the box Y0 and the omega-ball, by Picard iteration on
     truncated series (n + 1 iterations; iteration i fixes coefficient i).
  2. The defect in cleared form, d~ = u^m (p' - F(u, p)), u = c + tau, m = 1 (P, V) or 2 (C): an exact polynomial in
     tau (acb_poly products). Its coefficients below n must contain 0; then |d(tau)| <= |tau|^n D~/(|c| - r)^m with
     D~ = sum_{k >= n} |d~_k| r^{k - n}.
  3. M0_j = sum_{1<=k<=n} |p_{j,k}| r^k; the box B = Y0 + disc(M0 + b); the z-ball 1/c + disc(r/(|c|(|c| - r))),
     which contains 1/(c + tau) for |tau| <= r; the Lipschitz
     matrix L_ij >= sup |dF_i/dY_j| over B x z-ball x omega-ball by forward-mode derivatives (series of length 2).
  4. Weights om_i = 1/v_i (v the right Perron vector of L); Lam = r/(n + 2) max_i sum_j om_i L_ij/om_j < 1, F0 = max_i om_i D_i/(n + 1),
     rho = 2 F0/(1 - Lam), and b_j >= rho r^{n+1}/om_j (b raised and L recomputed until it holds). M_j = M0_j + b_j.
  Proof (A_certify.md 12.2): R = y - p is the fixed point of T(R) = int_0^tau [d + F(p + R) - F(p)] in the norm
  max_i om_i sup |R_i|/|tau|^(n+1); ||T(0)|| <= F0 and Lip(T) <= Lam.

Inputs: the 12 discs on the start ray use the stored row-start boxes (lemma6b1_runs/ray.json 'starts'); the arc discs
use boxes reconstructed here (own high-precision integration from the row-start midpoint along the arc, NOT rigorous,
inflated to 4 times the stored box radius plus 1e-15).

    python3 p2c_disc.py --checkeq                    # the equation against lemma6_usys.SysU.rhs at random points
    python3 p2c_disc.py --checkint                   # the integrator from each s_j to s_{j+1} against the stored boxes
    python3 p2c_disc.py --discs=SPEC [--mutate=sign] [--radius=R] [--tag=T]
        SPEC: ray (the 12 k = 0 discs), row11, corners, sample (ray + row11 + corners), all, or j:k,j:k,...
--mutate=sign computes the defect with the fast term's sign flipped (V_u = +V/2 + ...), the Taylor coefficients with
the true equation: every certificate must fail. --radius=R replaces the disc radius (the too-large-disc control).
"""
import json
import math
import os
import random
import sys
import time

import flint
from flint import acb, arb, acb_series, acb_poly

flint.ctx.prec = 192
flint.ctx.cap = 512                 # series length cap (python-flint default 10)
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "lemma6b1_runs")
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
INFL = float(OPTS.get("infl", "4"))  # the reconstructed box: INFL times the stored box radius (protocol: 4)
ODD, EVEN, KAP = (-1, 1), (-2, 0, 2), 100
IU = acb(0, 1)
HALF = acb(1) / 2
IKAP = acb(1) / KAP
P34 = 3 * arb.pi() / 4


def pr(X, Y, targets, zero):
    out = {}
    for t in targets:
        s = zero
        for a in X:
            if t - a in Y:
                s = s + X[a] * Y[t - a]
        out[t] = s
    return out


def parts(Y, w, zero):
    P = {-1: Y[0], 1: Y[1]}
    V = {-1: Y[2], 1: Y[3]}
    Cm1 = {-2: Y[4] * IKAP, 0: Y[5] * IKAP, 2: Y[6] * IKAP}
    C = {-2: Cm1[-2], 0: Cm1[0] + 1, 2: Cm1[2]}
    s = {m: IU * m * w for m in ODD}
    Q = {m: V[m] + (1 + s[m]) * P[m] for m in ODD}
    CQ = pr(Cm1, Q, ODD, zero)
    W = {m: V[m] + CQ[m] for m in ODD}
    CW = pr(Cm1, W, ODD, zero)
    Eh = {k: v * HALF for k, v in pr(P, P, EVEN, zero).items()}
    CE = pr(C, Eh, EVEN, zero)
    CEP = pr(CE, P, ODD, zero)
    GP = {m: V[m] + CQ[m] for m in ODD}
    SV = {m: V[m] + CQ[m] + CW[m] for m in ODD}
    H = {m: -(2 + s[m]) * Q[m] - (1 + s[m]) * W[m] + CEP[m] for m in ODD}
    return GP, SV, H, CE, V


def rhs(Y, z, w, zero):
    GP, SV, H, CE, V = parts(Y, w, zero)
    return ([-(z * GP[m]) * HALF for m in ODD] + [-(SV[m] + z * H[m]) * HALF for m in ODD]
            + [-(z * z * CE[k]) * (HALF * KAP) for k in EVEN])


def trunc(s, N):
    co = list(s.coeffs())[:N]
    return acb_series(co + [acb(0)] * (N - len(co)), prec=N)


def taylor(c, Y0, w, n):
    """p_0..p_n of the solution through (c, Y0), by Picard iteration on series of length n + 1"""
    N = n + 1
    z = acb_series([c, 1], prec=N).inv()
    zero = acb_series([0], prec=N)
    y = [acb_series([v], prec=N) for v in Y0]
    for _ in range(n + 1):
        F = rhs(y, z, w, zero)
        y = [trunc(acb_series([Y0[i]], prec=N) + F[i].integral(), N) for i in range(7)]
    out = []
    for i in range(7):
        co = list(y[i].coeffs())
        out.append(co + [acb(0)] * (N - len(co)))
    return out


def defect(c, p, w, n):
    """d~_E = u^m (p' - F(u, p)) as exact polynomials (m = 1 for P, V; 2 for C)"""
    P = [acb_poly(p[i][:n + 1]) for i in range(7)]
    u = acb_poly([c, 1])
    zero = acb_poly([0])
    GP, SV, H, CE, V = parts(P, w, zero)
    dP = [q.derivative() for q in P]
    d = [u * dP[i] + GP[m] * HALF for i, m in enumerate(ODD)]
    for i, m in enumerate(ODD):
        e = u * dP[2 + i] + u * SV[m] * HALF + H[m] * HALF
        if MUT == "sign":
            e = e - u * P[2 + i]                  # F_V + V: the fast term V_u = +V/2 + ...
        d.append(e)
    d += [u * u * dP[4 + i] + CE[k] * (HALF * KAP) for i, k in enumerate(EVEN)]
    return d


def up_float(a):
    """a float >= the arb a's upper end (the conversion may round to nearest, so it is pushed up)"""
    v = float(a.upper().mid())
    return v * (1 + 1e-12) + 1e-300 if v >= 0 else v * (1 - 1e-12)


def inflate(x, e):
    """the box of x widened by e on each axis (contains x + disc(e))"""
    e = arb(e) if not isinstance(e, arb) else e
    return acb(arb(x.real.mid(), up_float(x.real.rad() + e)), arb(x.imag.mid(), up_float(x.imag.rad() + e)))


def lipschitz(B, zb, wb):
    L = [[arb(0)] * 7 for _ in range(7)]
    zero = acb_series([0], prec=2)
    for j in range(7):
        Ys = [acb_series([B[i], 1 if i == j else 0], prec=2) for i in range(7)]
        F = rhs(Ys, zb, wb, zero)
        for i in range(7):
            co = list(F[i].coeffs())
            L[i][j] = abs(co[1]).upper() if len(co) > 1 else arb(0)
    return L


def perron(L):
    M = [[float(L[i][j].mid()) + float(L[i][j].rad()) for j in range(7)] for i in range(7)]
    v = [1.0] * 7
    for _ in range(300):
        nv = [sum(M[i][j] * v[j] for j in range(7)) + 1e-300 for i in range(7)]
        mx = max(nv)
        v = [max(a / mx, 1e-30) for a in nv]
    return v


def lemma_d(c, Y0, wb, n, r):
    """returns dict(ok, why, Lam, Mmax, F0, ...)"""
    t0 = time.time()
    r = arb(r)
    absc = abs(c).lower()
    if not absc > r * math.sqrt(2) * 1.0001:
        return dict(ok=False, why="|c| <= r sqrt2 (the z-ball would contain 0)")
    p = taylor(c, Y0, wb, n)
    d = defect(c, p, wb, n)
    D = []
    for i, de in enumerate(d):
        co = list(de.coeffs())
        if any(not co[k].contains(acb(0)) for k in range(min(n, len(co)))):
            kk = min(k for k in range(min(n, len(co))) if not co[k].contains(acb(0)))
            return dict(ok=False, why=f"defect of component {i} does not vanish at order {kk}", sec=time.time() - t0)
        Dt = arb(0)
        for k in range(n, len(co)):
            Dt += abs(co[k]).upper() * r ** (k - n)
        m = 2 if i >= 4 else 1
        D.append(Dt / (absc - r) ** m)
    M0 = []
    for i in range(7):
        s = arb(0)
        for k in range(1, n + 1):
            s += abs(p[i][k]).upper() * r ** k
        M0.append(s)
    # z = 1/(c + tau), |tau| <= r: |z - 1/c| = |tau|/(|c||c + tau|) <= r/(|c|(|c| - r)) (an inverse of the box would
    # be far looser)
    zb = inflate(1 / c, (r / (absc * (absc - r))).upper())
    b = [arb(0)] * 7
    for it in range(12):
        B = [inflate(Y0[i], M0[i] + b[i]) for i in range(7)]
        L = lipschitz(B, zb, wb)
        om = [1.0 / x for x in perron(L)]             # om_i = 1/v_i, v the right Perron vector of L
        oma = [arb(x) for x in om]
        Lam = max(sum((oma[i] * L[i][j] / oma[j] for j in range(7)), arb(0)) for i in range(7)) * r / (n + 2)
        Lam = Lam.upper()
        if not Lam < 1:
            return dict(ok=False, why=f"Lambda {float(Lam.mid()):.4g} >= 1", Lam=float(Lam.mid()), it=it,
                        sec=time.time() - t0)
        F0 = max((oma[i] * D[i] for i in range(7)), key=lambda x: float(x.upper().mid())).upper() / (n + 1)
        rho = (2 * F0 / (1 - Lam)).upper()
        need = [(rho * r ** (n + 1) / oma[j]).upper() for j in range(7)]
        if all(need[j] <= b[j] for j in range(7)):
            Mj = [(M0[i] + b[i]).upper() for i in range(7)]
            return dict(ok=True, why="", Lam=float(Lam.mid()), F0=float(F0.mid()), Mmax=max(float(x.mid()) for x in Mj),
                        M0max=max(float(x.mid()) for x in M0), bmax=max(float(x.mid()) for x in b), it=it,
                        sec=time.time() - t0)
        b = [max(b[j], 2 * need[j] + arb("1e-60"), key=lambda x: float(x.mid())) for j in range(7)]
    return dict(ok=False, why="b did not close in 12 rounds", sec=time.time() - t0)


# ---------------------------------------------------------------------------------------------- inputs
def rows():
    out = []
    for j in range(12):
        with open(os.path.join(RUNS, f"row_{j:02d}.json")) as fh:
            d = json.load(fh)
        out.append(dict(j=j, rho=repr(d["rho"]), r=d["r"], n=d["n"],
                        discs=[(x["k"], x["t"], x["box_rad"]) for x in d["discs"]]))
    return out


def starts():
    with open(os.path.join(RUNS, "ray.json")) as fh:
        st = json.load(fh)["starts"]
    return {int(j): [acb(arb(a), arb(b)) for a, b in v] for j, v in st.items()}


def centre(rho, t):
    phi = -P34 * (1 - arb(t))
    return arb(rho) * acb(0, phi).exp()


def mids(v):
    return [acb(x.mid()) for x in v]


def step(u0, y0, w, h, order=48):
    """y(u0 + h) from the Taylor polynomial of order `order` at u0 (midpoint arithmetic, not rigorous)"""
    p = taylor(u0, y0, w, order)
    out = []
    for i in range(7):
        s = acb(0)
        for k in range(order, -1, -1):
            s = s * h + p[i][k]
        out.append(acb(s.mid()))
    return out


def integrate(u0, y0, w, u1, hmax):
    n = max(1, int(math.ceil(float(abs(u1 - u0).mid()) / hmax)))
    y, u = y0, u0
    for i in range(1, n + 1):
        un = u0 + (u1 - u0) * i / n
        y = step(u, y, w, un - u)
        u = un
    return y


def arc_values(R, st):
    """reconstructed midpoints at every disc of row R, along the arc from s_j (chords of length <= hmax)"""
    j = R["j"]
    y = mids(st[j][:7])
    w = acb(st[j][7].mid())
    hmax = min(R["r"] / 4.0, 3.0)
    vals = {0: y}
    u = centre(R["rho"], 0)
    u = acb(u.mid())
    for k, t, _ in R["discs"][1:]:
        ct = acb(centre(R["rho"], t).mid())
        y = integrate_arc(u, y, w, R["rho"], hmax, ct)
        u = ct
        vals[k] = y
    return vals


def integrate_arc(u0, y0, w, rho, hmax, u1):
    """from u0 to u1, both on |u| = rho, through arc points (each chord <= hmax)"""
    a0 = math.atan2(float(u0.imag.mid()), float(u0.real.mid()))
    a1 = math.atan2(float(u1.imag.mid()), float(u1.real.mid()))
    nseg = max(1, int(math.ceil(abs(a1 - a0) * float(rho) / hmax)))
    y, u = y0, u0
    for i in range(1, nseg + 1):
        un = u1 if i == nseg else acb((arb(rho) * acb(0, arb(a0 + (a1 - a0) * i / nseg)).exp()).mid())
        y = step(u, y, w, un - u)
        u = un
    return y


# ---------------------------------------------------------------------------------------------- checks
def checkeq():
    sys.path.insert(0, os.path.dirname(HERE))
    import lemma6_usys as U
    S = U.SysU()
    rnd = random.Random(992)
    worst = 0.0
    for _ in range(10):
        Y = [acb(rnd.uniform(-2, 2), rnd.uniform(-2, 2)) for _ in range(7)]
        w = acb(rnd.uniform(1, 4), rnd.uniform(-0.1, 0.1))
        u = acb(rnd.uniform(-200, -20), rnd.uniform(-200, 20))
        z = 1 / u
        mine = rhs(Y, z, w, acb(0))
        ref = S.rhs(Y + [w, acb(2.5)], u, z, acb(0))[0][:7]
        for a, b_ in zip(mine, ref):
            rel = float(abs(a - b_).upper().mid()) / max(1e-300, float(abs(b_).mid()))
            worst = max(worst, rel)
    ok = worst < 1e-40
    print(f"checkeq: own right-hand side against lemma6_usys.SysU.rhs at 10 random points: worst relative difference "
          f"{worst:.2e}: {'PASS' if ok else 'FAIL'}")
    return ok


def checkint():
    R_ = rows()
    st = starts()
    ok = True
    for j in range(11):
        y = mids(st[j][:7])
        w = acb(st[j][7].mid())
        u0 = acb(centre(R_[j]["rho"], 0).mid())
        u1 = acb(centre(R_[j + 1]["rho"], 0).mid())
        y1 = integrate(u0, y, w, u1, 2.0)
        ins = all(st[j + 1][i].contains(y1[i]) for i in range(7))
        dev = max(float(abs(y1[i] - acb(st[j + 1][i].mid())).mid()) for i in range(7))
        rad = max(float(max(st[j + 1][i].real.rad(), st[j + 1][i].imag.rad())) for i in range(7))
        print(f"checkint: s_{j} -> s_{j + 1} (|u| {R_[j]['rho']} -> {R_[j + 1]['rho']}): inside the stored box: {ins}; "
              f"max deviation from its midpoint {dev:.2e} (box radius {rad:.2e})", flush=True)
        ok &= ins
    print(f"checkint: {'PASS' if ok else 'FAIL'}")
    return ok


def select(spec, R_):
    allk = [(R["j"], k) for R in R_ for k, _, _ in R["discs"]]
    corners = [(0, 0), (0, len(R_[0]["discs"]) - 1), (11, 0), (11, len(R_[11]["discs"]) - 1)]
    sets = {"ray": [(j, 0) for j in range(12)], "row11": [(11, k) for k, _, _ in R_[11]["discs"]], "corners": corners,
            "all": allk}
    sets["sample"] = sorted(set(sets["ray"] + sets["row11"] + corners))
    if spec in sets:
        return sets[spec]
    return [tuple(int(v) for v in x.split(":")) for x in spec.split(",")]


def main():
    if "checkeq" in OPTS:
        return 0 if checkeq() else 1
    if "checkint" in OPTS:
        return 0 if checkint() else 1
    R_ = rows()
    st = starts()
    want = select(OPTS.get("discs", "ray"), R_)
    tag = OPTS.get("tag", OPTS.get("discs", "ray") + (f"_mut_{MUT}" if MUT else "")
                   + (f"_r{OPTS['radius']}" if "radius" in OPTS else ""))
    outp = os.path.join(HERE, f"p2c_disc_{tag}.jsonl")
    fh = open(outp, "w")
    recon = {}
    nok = nall = 0
    t0 = time.time()
    for (j, k) in want:
        R = R_[j]
        kk, t, box_rad = R["discs"][k]
        r = float(OPTS["radius"]) if "radius" in OPTS else R["r"]
        c = centre(R["rho"], t)
        if k == 0:
            Y0, wb, src = st[j][:7], st[j][7], "stored"
        else:
            if j not in recon:
                recon[j] = arc_values(R, st)
            Y0 = [inflate(v, arb(INFL * box_rad + 1e-15)) for v in recon[j][k]]
            wb, src = st[j][7], "reconstructed"
        res = lemma_d(c, Y0, wb, R["n"], r)
        res.update(j=j, k=k, t=t, n=R["n"], r=r, input=src, mutate=MUT)
        fh.write(json.dumps(res) + "\n")
        fh.flush()
        nall += 1
        nok += bool(res["ok"])
        print(f"disc ({j}, {k}) t {t:.6f} r {r} n {R['n']} input {src}: "
              + (f"ok, Lambda {res['Lam']:.4f}, max M {res['Mmax']:.4g}, F0 {res['F0']:.3g}" if res["ok"]
                 else f"NOT certified: {res['why']}") + f" [{res.get('sec', 0):.1f} s]", flush=True)
    fh.close()
    print(f"{nok} of {nall} discs certified ({time.time() - t0:.0f} s); records {outp}")
    return 0 if nok == nall else 1


if __name__ == "__main__":
    sys.exit(main())
