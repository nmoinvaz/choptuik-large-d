#!/usr/bin/env python3
"""Lemma 6 (proof/A_certify.md section 11), the parts near xi = 0: rigorous (python-flint arb/acb).

(S)  Start values on a domain that reaches the real axis (lemma S, for 6b). fixed_point.py integrates the fast
     component from z = 0 along the ray itself, which needs Re(1/z) < 0 on the ray. Here the domain is, with u = 1/z,
         D = { z : 1/z in U0 + K },   K = { w : -3 pi/4 <= arg w <= 0 },   U0 = sqrt(2)/R+ ,
     and the fast component is integrated along the path u + s e^{-3 i pi/4}, s from infinity to 0 (in the z-plane it
     ends at z and starts at 0 from the direction arg z -> 3 pi/4):
         v(z) = - int_0^inf e^{s e^{-3 i pi/4}/2} Phi_v(t(s)) dt/ds ds,   t(s) = 1/(u + s e^{-3 i pi/4}).
     The path stays in D (D - U0 is a convex cone containing the direction), |t| <= R+ on D, and D is star-shaped
     about 0, so the slow components are integrated along the segment [0, z] as before. Kernel bound (section 11):
         int |kernel| |t|^m |dt| <= |z|^{m+2} / (a - (m+2) R+/2),   a = sqrt(2)/4,
     and with |t|^q <= R+^{q-k+2} |t|^{k-2} every fast integrand K |t|^q (q >= k - 2) gives the weighted bound
     K R+^{q+2-k} / a_eff with a_eff = sqrt(2)/4 - k R+/2. Every other bound of fixed_point.py is a sum of |coefficients|
     times powers of the radius and holds on D with R = R+. D contains the real segment (0, R+/sqrt 2] and, in the
     direction arg z = theta, the radii |z| <= R+ sin(3 pi/4 - theta).
(R)  The flat estimate on the real segment (for 6a). On 0 < z <= Z (real), psi = y+ - y- (both in the ball of lemma S)
     satisfies psi = b + T0 psi exactly, with b = (0, e^{1/(2Z) - 1/(2z)} v_psi(Z), 0), the slow components integrated
     from 0 and the fast one backward from Z:  v(z) = e^{1/(2Z)-1/(2z)} v(Z) + int_Z^z e^{1/(2t)-1/(2z)} Phi_v(t) dt.
     Kernel bound: int_z^Z e^{-(1/z - 1/t)/2} t^n dt <= z^{n+2} / (1/2 - (n+2) Z), so a_eff = 1/2 - k Z.
     T0 is linear in psi with the derivative majorants of fixed_point.lipschitz_and_forcing over the ball of lemma S,
     so ||T0|| <= Lambda, and ||psi - b|| <= Lambda/(1 - Lambda) ||b||, ||b|| = Z^-k max_mu w_v |v_mu(Z)|.

The old sector lemma (fixed_point.py --sector=0.2) is rerun over the same parameter box: it covers the rays
|arg xi - 3 pi/8| <= 0.2, |xi| <= 0.1, where D does not reach.

    python3 lemma6_sector.py [--rp=0.1] [--k=30] [--box=1e-9] [--Z=0.004489] [--json=out.json]
        --rp: xi-radius with R+ = rp^2; --box: real parameter box half-width around x^ (0: the point);
        --Z: right end of the real segment of the flat estimate (Z <= R+/sqrt 2)
    python3 lemma6_sector.py --mutate=series     mutation test: one coefficient of the formal series changed by a
        relative 1e-6 (a wrong input); the residual no longer vanishes below order N, and the lemma must fail
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fixed_point as FP  # noqa: E402
from flint import arb, acb, acb_poly, ctx  # noqa: E402

A_STAR, WF_STAR = FP.A_STAR, FP.WF_STAR
RP, KW = "0.1", 30                 # lemma S: R+ = RP^2, weight exponent KW
SQ2 = None
ORIG_INTBOUND = FP.intbound


def intbound_strict(m, kind, k_i):
    """fixed_point.intbound without the slow-type alternative for the fast rows. That alternative integrates from 0
    with |kernel| <= 1 along the ray to z, which holds on the rays of fixed_point.py but not on the paths used here."""
    if m is None:
        return arb(0)
    q, K, R = m.q, m.K, FP.Maj.R
    if kind == "fast":
        if q + 2 < k_i:
            raise ValueError(f"fast integrand t^{q} not admissible for weight {k_i}")
        return K * R ** (q + 2 - k_i) / FP.A_FAST
    if q > -1 and q + 1 >= k_i:
        return K * R ** (q + 1 - k_i) / (q + 1)
    raise ValueError(f"integrand t^{q} not admissible for weight {k_i} (slow)")


class patched:
    """use the strict intbound and a given fast rate inside fixed_point; restore afterwards"""

    def __init__(self, a_fast):
        self.a = a_fast

    def __enter__(self):
        self.saved = FP.A_FAST, FP.intbound
        FP.A_FAST, FP.intbound = arb(self.a), intbound_strict
        return self

    def __exit__(self, *exc):
        FP.A_FAST, FP.intbound = self.saved


def perron_weights(L, sym_pairs=None):
    Lf = np.array([[float(x.upper()) for x in row] for row in L]) + 1e-12
    ev, evec = np.linalg.eig(Lf)
    i = int(np.argmax(ev.real))
    u = np.abs(evec[:, i].real) + 1e-6 * np.abs(evec[:, i].real).max()
    w = list(1.0 / u)
    if sym_pairs:
        w = [min(w[a], w[b]) for a, b in enumerate(sym_pairs)]
    return w


def weighted_norm(L, wts):
    """max_i sum_j w_i L_ij / w_j, an upper bound as an exact arb (rigorous max over upper bounds)"""
    vals = [sum(arb(wts[i]) * L[i][j] / arb(wts[j]) for j in range(len(L))) for i in range(len(L))]
    return arb(max(v.upper() for v in vals))


def sym_index(comps):
    idx = {cp: i for i, cp in enumerate(comps)}
    return [idx[(t, -m)] for (t, m) in comps]


def params(box):
    """(A, wf) as acb (point or real box of half-width box) and (N, kmins) from the box centre"""
    ctx.prec = 320
    if box and float(box) > 0:
        A = acb(arb(A_STAR, arb(box)))
        wf = acb(arb(WF_STAR, arb(box)))
    else:
        A, wf = acb(A_STAR), acb(WF_STAR)
    return A, wf


def lemma_S(rp=RP, k=KW, box="0", verbose=True, mutate=None):
    """the fixed point on D (domain reaching the real axis), weights symmetric under mu -> -mu.
    mutate="series": one coefficient of the formal series (D_{+1}, order z^5) is changed by a relative 1e-6 (a wrong
    input); the residual then no longer vanishes below order N and the lemma must fail."""
    ctx.prec = 320
    sq2 = arb(2).sqrt()
    Rp = arb(rp) ** 2
    a_eff = (sq2 / 4 - k * Rp / 2).lower()
    if not arb(a_eff) > 0:
        raise ValueError("a_eff <= 0")
    A, wf = params(box)
    FP.Maj.BOX = False
    centre = FP.run(acb(A_STAR), acb(WF_STAR), rp, verbose=False)          # N and vanishing orders (structural)
    N, kmins = centre["N"], centre["kmins"]
    FP.Maj.BOX = bool(box and float(box) > 0)
    W0 = 2 * arb.pi() / arb("1.64389375")
    sa, N, P, Q, C, R = FP.setup(A, acb(W0) * wf, rp, 320, N)
    if mutate == "series":
        cs = P[1].coeffs()
        cs[5] = cs[5] * (1 + acb("1e-6"))
        P[1] = acb_poly(cs)
    FP.Maj.R = R
    Np, Nv, Nc, V = FP.residual_numerators(sa, P, Q, C)
    res, lows = {}, []
    kin = iter(kmins)
    for mu in sa.odd:
        m, km, ok = FP.tail_maj(Np[mu], 1, arb(0.5), next(kin)); res[("p", mu)] = m; lows.append(ok)
        m, km, ok = FP.tail_maj(Nv[mu], 2, arb(0.5), next(kin)); res[("v", mu)] = m; lows.append(ok)
    for nu in sa.even:
        m, km, ok = FP.tail_maj(Nc[nu], 0, arb(1), next(kin)); res[("c", nu)] = m; lows.append(ok)
    kmax = min(min(res[("p", mu)].q + 1 for mu in sa.odd), min(res[("v", mu)].q + 2 for mu in sa.odd),
               min(res[("c", nu)].q for nu in sa.even))
    if k > kmax:
        raise ValueError(f"weight exponent k = {k} exceeds the forcing orders (max {kmax})")
    with patched(a_eff):
        L, comps, kw, rows = FP.lipschitz_and_forcing(sa, P, Q, C, k, arb("1e-20"), [1.0] * 7)
        sp = sym_index(comps)
        wts = perron_weights(L, sp)
        Fv = [FP.intbound(res[cp], "fast" if cp[0] == "v" else "slow", kw[cp]) for cp in comps]
        F = arb(max((arb(wts[j]) * Fv[j]).upper() for j in range(len(comps))))
        lip0 = weighted_norm(L, wts)
        rho = 4 * F / (1 - lip0) if lip0 < 1 else arb(1)
        L, comps, kw, rows = FP.lipschitz_and_forcing(sa, P, Q, C, k, rho, wts)
        lip = weighted_norm(L, wts)
    selfmap = F + lip * rho
    ok = bool(lip < 1) and bool(selfmap < rho) and all(lows)
    out = dict(N=N, k=k, kmins=kmins, lip=lip, F=F, rho=rho, selfmap=selfmap, ok=ok, wts=wts, comps=comps, kw=kw,
               sa=sa, P=P, Q=Q, C=C, R=R, a_eff=arb(a_eff), rp=rp, box=box)
    if verbose:
        Rpf = float(R.upper())
        print(f"lemma S on D (u = 1/z in U0 + K, R+ = {rp}^2 = {Rpf:.4g}, U0 = sqrt2/R+), parameters "
              f"{'x^' if not FP.Maj.BOX else 'x^ + [-' + box + ', ' + box + ']^2'}:")
        print(f"  N = {N}, vanishing orders {kmins}, weight exponent k = {k} (admissible up to {kmax}), "
              f"a_eff = sqrt2/4 - k R+/2 = {float(a_eff):.4f}")
        print(f"  weights (symmetric in mu) {np.round(wts, 4).tolist()}")
        print(f"  contraction constant <= {float(lip.upper()):.4f}; forcing {float(F.upper()):.3e}; ball rho = "
              f"{float(rho.upper()):.3e}; F + Lip rho = {float(selfmap.upper()):.3e} < rho: {bool(selfmap < rho)}")
        print(f"  => unique fixed point in the ball (y+ on D): {ok}")
        print(f"  D covers: the real segment 0 < z <= R+/sqrt2 = {Rpf / 2 ** 0.5:.5f} (xi <= {(Rpf / 2 ** 0.5) ** 0.5:.5f}),"
              f" and |z| <= R+ sin(3pi/4 - theta) in direction theta = arg z")
    FP.Maj.BOX = False
    return out


def sector_old(box="0", d="0.2", verbose=True):
    """fixed_point.py --sector=d over the parameter box (rays |arg xi - 3pi/8| <= d, |xi| <= 0.1)"""
    ctx.prec = 320
    saved = FP.A_FAST, FP.intbound
    FP.intbound = ORIG_INTBOUND
    try:
        FP.set_sector(d)
        A, wf = params(box)
        FP.Maj.BOX = False
        c0 = FP.run(acb(A_STAR), acb(WF_STAR), "0.1", verbose=False)
        FP.Maj.BOX = bool(box and float(box) > 0)
        out = FP.run(A, wf, "0.1", verbose=False, kmins_in=c0["kmins"], N_in=c0["N"])
    finally:
        FP.A_FAST, FP.intbound = saved
        FP.Maj.BOX = False
    if verbose:
        print(f"old sector lemma (rays |arg xi - 3pi/8| <= {d}, |xi| <= 0.1) over the same parameters: contraction "
              f"{float(out['lip'].upper()):.4f}, F + Lip rho < rho: {bool(out['selfmap'] < out['rho'])}, ok {out['ok']}")
    return out


def flat_estimate(S, Z="0.0045", verbose=True):
    """(R): ||T0|| on the real segment (0, Z], with the derivative majorants over the ball of lemma S"""
    ctx.prec = 320
    k = S["k"]
    Za = arb(Z)
    if not Za <= S["R"] / arb(2).sqrt():
        raise ValueError("Z beyond the real reach of D")
    a_r = (arb(0.5) - k * Za).lower()
    if not arb(a_r) > 0:
        raise ValueError("1/2 - k Z <= 0")
    FP.Maj.R = Za
    FP.Maj.BOX = S["box"] not in ("0", None) and float(S["box"]) > 0
    with patched(a_r):
        L, comps, kw, rows = FP.lipschitz_and_forcing(S["sa"], S["P"], S["Q"], S["C"], k, S["rho"], S["wts"])
    FP.Maj.BOX = False
    wts = perron_weights(L, sym_index(comps))
    lam = weighted_norm(L, wts)
    out = dict(Z=Za, k=k, L=L, wts=wts, lam=lam, comps=comps, a_r=arb(a_r), ok=bool(lam < 1))
    if verbose:
        print(f"flat estimate on the real segment 0 < z <= Z = {Z} (xi <= {float(Za.mid()) ** 0.5:.5f}), "
              f"a_eff = 1/2 - k Z = {float(a_r):.4f}:")
        print("  ||T0|| matrix (rows/cols " + ", ".join(f"{t}{m:+d}" for t, m in comps) + "):")
        for row in L:
            print("    " + " ".join(f"{float(x.upper()):9.2e}" for x in row))
        print(f"  weights {np.round(wts, 4).tolist()}; ||T0|| <= Lambda = {float(lam.upper()):.4f} < 1: {bool(lam < 1)}")
    return out


def to_json(S, Rr, O):
    return {"rp": S["rp"], "box": S["box"], "k": S["k"], "N": S["N"], "lip_S": S["lip"].str(10), "rho_S": S["rho"].str(10),
            "F_S": S["F"].str(10), "ok_S": S["ok"], "wts_S": S["wts"], "a_eff_S": S["a_eff"].str(10),
            "Z": Rr["Z"].str(10), "Lambda_R": Rr["lam"].str(10), "wts_R": Rr["wts"], "ok_R": Rr["ok"],
            "lip_old": O["lip"].str(10), "ok_old": O["ok"]}


if __name__ == "__main__":
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    rp, k, box, Z = opts.get("rp", RP), int(opts.get("k", str(KW))), opts.get("box", "1e-9"), opts.get("Z", "0.004489")
    if opts.get("mutate") == "series":
        S = lemma_S(rp, k, box, mutate="series")
        print(f"MUTATION series (one coefficient off by a relative 1e-6): lemma S passes: {S['ok']} (must be False)")
        sys.exit(0 if not S["ok"] else 3)
    S = lemma_S(rp, k, box)
    O = sector_old(box)
    Rr = flat_estimate(S, Z)
    allok = S["ok"] and O["ok"] and Rr["ok"]
    print(f"ALL CHECKS PASS: {allok}")
    if "json" in opts:
        with open(opts["json"], "w") as fh:
            json.dump(to_json(S, Rr, O), fh, indent=1)
