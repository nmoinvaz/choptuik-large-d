#!/usr/bin/env python3
"""Lemma S, the old sector lemma and the flat estimate (proof/A_certify.md section 11.3-11.5) for M harmonics.
Rigorous (python-flint arb/acb). Statements and proofs: proof/A_lemmas_M.md, sections 3-4.

(S)  Lemma S on the domain D = { z : 1/z in U0 + K }, K = { -3pi/4 <= arg w <= 0 }, U0 = sqrt(2)/R+ (lemma6_sector.py),
     with the fast components integrated along u + s e^{-3 i pi/4}. Not a ray, so the "plain" splitting of
     fixed_point_M.py (the label terms s_mu = i mu w are perturbations) and the strict kernel bound
     K R+^{q+2-k} / a_eff, a_eff = sqrt(2)/4 - k R+/2, with no slow-type alternative for fast rows. The domain, the
     path, the kernel bound and a_eff do not depend on M: the fast eigenvalue repeats across harmonics. What depends
     on M is the Lipschitz matrix, whose diagonal fast entries |3 + 2 s_mu| R+ / (2 a_eff) grow like |mu| w, so the
     contraction needs R+ roughly proportional to 1/(M w).
(O)  The old sector lemma (fixed_point_M.py --sector=d): rays |arg xi - 3pi/8| <= d, |xi| <= r0; on rays the "diag"
     splitting is valid and removes the label-dependent diagonal terms.
(R)  The flat estimate on the real segment (0, Z] (for lemma 6a): psi = y+ - y- satisfies psi = b + T0 psi, slow
     components from 0, fast ones backward from Z, kernel bound z^{n+2} / (1/2 - (n+2) Z), a_r = 1/2 - k Z ("plain"),
     or with the label terms in the principal part (the real axis is a ray: |(t/z)^{s_mu}| = 1, and (t/z)^{3/2} <=
     (Z/z)^{3/2} shifts n by 3/2), a_r = 1/2 - (k + 3/2) Z ("diag").

    python3 lemma6_sector_M.py --M=3 [--rp=0.06] [--k=30] [--box=1e-6] [--Z=...] [--json=f.json]
        lemma S, the old sector lemma and the flat estimate at the hier.py zero (or over the real box x +- box)
    python3 lemma6_sector_M.py --M=3 --scan        lemma S over a grid of (rp, k): where the contraction closes
    python3 lemma6_sector_M.py --M=3 --mutate=series   one coefficient of the top harmonic's series off by a relative
                                                        1e-6: lemma S must fail
    control: python3 lemma6_sector_M.py --M=3 --rp=0.13 --k=10     R+ too large: lemma S must fail
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fixed_point as FP  # noqa: E402
import fixed_point_M as FPM  # noqa: E402
import lemma6_sector as L6S  # noqa: E402
from flint import arb, acb, ctx  # noqa: E402


def xs_of(M, box="0"):
    ctx.prec = 320
    xs = FPM.ZEROS[M]
    if box and float(box) > 0:
        return [acb(arb(v, arb(box))) for v in xs]
    return [acb(v) for v in xs]


def lemma_S_M(M, rp, k, box="0", verbose=True, mutate=None):
    """the fixed point on D, weights symmetric under mu -> -mu. Returns run_M's dict plus a_eff."""
    ctx.prec = 320
    Rp = arb(rp) ** 2
    a_eff = arb((arb(2).sqrt() / 4 - k * Rp / 2).lower())
    if not a_eff > 0:
        raise ValueError("a_eff <= 0")
    FP.Maj.BOX = False
    c0 = FPM.run_M(xs_of(M), M, rp, verbose=False, a_in=a_eff, intbound=L6S.intbound_strict, k_in=k, sym=True)
    isbox = bool(box and float(box) > 0)
    FP.Maj.BOX = isbox
    out = FPM.run_M(xs_of(M, box), M, rp, verbose=False, kmins_in=c0["kmins"], N_in=c0["N"], k_in=k, a_in=a_eff,
                    intbound=L6S.intbound_strict, sym=True, mutate="series" if mutate == "series" else None)
    FP.Maj.BOX = False
    out["a_eff"], out["rp"], out["box"] = a_eff, rp, box
    if verbose:
        Rf = float(Rp.upper())
        print(f"lemma S (M = {M}) on D (R+ = {rp}^2 = {Rf:.4g}, U0 = sqrt2/R+ = {2 ** 0.5 / Rf:.1f}), parameters "
              f"{'x^' if not isbox else 'x^ +- ' + box}: N = {out['N']}, k = {k} (forcing allows {out['kmax']}), "
              f"a_eff = {float(a_eff):.4f}")
        FPM.report(out)
        print(f"  D covers the real segment 0 < xi <= {(Rf / 2 ** 0.5) ** 0.5:.5f}, and directions arg z <= 3pi/4 - 0.4 "
              f"up to |xi| = {(Rf * 0.389418) ** 0.5:.5f} (|u| >= {1 / (Rf * 0.389418342):.1f})")
    return out


def sector_old_M(M, box="0", d="0.2", r0="0.1", diag=True, verbose=True):
    """rays |arg xi - 3pi/8| <= d, |xi| <= r0, over the parameter box"""
    ctx.prec = 320
    FP.Maj.BOX = False
    c0 = FPM.run_M(xs_of(M), M, r0, "3pi/8", d, verbose=False, diag=diag)
    FP.Maj.BOX = bool(box and float(box) > 0)
    out = FPM.run_M(xs_of(M, box), M, r0, "3pi/8", d, verbose=False, kmins_in=c0["kmins"], N_in=c0["N"], k_in=c0["k"],
                    diag=diag)
    FP.Maj.BOX = False
    if verbose:
        print(f"old sector lemma (M = {M}, rays |arg xi - 3pi/8| <= {d}, |xi| <= {r0}, {'diag' if diag else 'plain'}): "
              f"contraction {float(out['lip'].upper()):.4f}, F + Lip rho < rho: {bool(out['selfmap'] < out['rho'])}, "
              f"ok {out['ok']}")
    return out


def flat_estimate_M(S, Z, diag=False, verbose=True):
    """(R): ||T0|| on (0, Z] with the derivative majorants over the ball of lemma S"""
    ctx.prec = 320
    k = S["k"]
    Za = arb(Z)
    if not Za <= S["R"] / arb(2).sqrt():
        raise ValueError("Z beyond the real reach of D")
    a_r = arb((arb(0.5) - (k + (arb(1.5) if diag else 0)) * Za).lower())
    if not a_r > 0:
        raise ValueError("a_r <= 0")
    FP.Maj.R = Za
    FP.Maj.BOX = S["box"] not in ("0", None) and float(S["box"]) > 0
    with FPM.rate(a_r, L6S.intbound_strict):
        L, comps, kw, rows = FPM.lipschitz_M(S["sa"], S["P"], S["Q"], S["C"], k, S["rho"], S["wts"], diag)
    FP.Maj.BOX = False
    wts = FPM.perron(L, FPM.sym_index(comps))
    lam = FPM.wnorm(L, wts)
    out = dict(Z=Za, k=k, L=L, wts=wts, lam=lam, comps=comps, a_r=a_r, ok=bool(lam < 1), diag=diag)
    if verbose:
        big = sorted(((float(L[i][j].upper()), comps[i], comps[j]) for i in range(len(L)) for j in range(len(L))),
                     reverse=True)[:4]
        print(f"flat estimate (M = {M_of(S)}) on 0 < z <= Z = {Z} (xi <= {float(Za.mid()) ** 0.5:.5f}), "
              f"{'diag' if diag else 'plain'}, a_r = {float(a_r):.4f}: ||T0|| <= Lambda = {float(lam.upper()):.4f} "
              f"< 1: {bool(lam < 1)}; largest entries " + ", ".join(f"{t[1][0]}{t[1][1]:+d}<-{t[2][0]}{t[2][1]:+d} "
                                                                   f"{t[0]:.3g}" for t in big))
    return out


def M_of(S):
    return S["M"]


def scan(M, box="0"):
    """lemma S over a grid of (rp, k)"""
    res = []
    for rp in ("0.1", "0.09", "0.08", "0.07", "0.065", "0.06", "0.055", "0.05", "0.045"):
        for k in (10, 20, 30):
            t0 = time.time()
            try:
                o = lemma_S_M(M, rp, k, box, verbose=False)
                r = dict(rp=rp, k=k, N=o["N"], a_eff=float(o["a_eff"]), lip=float(o["lip"].upper()), ok=o["ok"],
                         eta=max(float(v.upper()) for v in FPM.eta_bounds(o).values()), sec=round(time.time() - t0, 1))
            except ValueError as e:
                r = dict(rp=rp, k=k, ok=False, err=str(e))
            res.append(r)
            print(f"M {M} rp {rp:6s} k {k:2d}: " + (f"N {r['N']:3d} a_eff {r['a_eff']:.4f} contraction {r['lip']:.4f} "
                                                    f"ok {r['ok']} |eta| {r['eta']:.1e} [{r['sec']} s]"
                                                    if "lip" in r else f"refused ({r['err']})"), flush=True)
    return res


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    M = int(opts.get("M", "3"))
    box = opts.get("box", "0")
    if "scan" in opts:
        res = scan(M, box)
        if "json" in opts:
            with open(opts["json"], "w") as fh:
                json.dump(res, fh, indent=1)
        return 0
    rp = opts.get("rp", {1: "0.1", 3: "0.06", 5: "0.05"}.get(M, "0.05"))
    k = int(opts.get("k", "30"))
    mut = opts.get("mutate")
    if mut == "series":
        S = lemma_S_M(M, rp, k, box, verbose=False, mutate="series")
        print(f"MUTATION series (top harmonic's coefficient off by 1e-6): lemma S holds: {S['ok']} (must be False)")
        return 0 if not S["ok"] else 3
    S = lemma_S_M(M, rp, k, box)
    O = sector_old_M(M, box, diag=True)
    Rp = float((arb(rp) ** 2).mid())
    Z = opts.get("Z", f"{0.98 * Rp / 2 ** 0.5:.6f}")
    Rr = flat_estimate_M(S, Z, diag=False)
    Rd = flat_estimate_M(S, Z, diag=True)
    allok = S["ok"] and O["ok"] and (Rr["ok"] or Rd["ok"])
    print(f"ALL CHECKS PASS (M = {M}): {allok}")
    if "json" in opts:
        with open(opts["json"], "w") as fh:
            json.dump({"M": M, "rp": rp, "box": box, "k": k, "N": S["N"], "lip_S": S["lip"].str(10),
                       "rho_S": S["rho"].str(10), "F_S": S["F"].str(10), "ok_S": S["ok"], "a_eff": S["a_eff"].str(10),
                       "wts_S": S["wts"], "lip_old": O["lip"].str(10), "ok_old": O["ok"], "Z": Z,
                       "Lambda_plain": Rr["lam"].str(10), "Lambda_diag": Rd["lam"].str(10), "wts_R_plain": Rr["wts"],
                       "wts_R_diag": Rd["wts"], "ok": allok}, fh, indent=1)
    return 0 if allok else 2


if __name__ == "__main__":
    sys.exit(main())
