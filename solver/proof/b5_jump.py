#!/usr/bin/env python3
"""Lemma B5(2) for M = 1 with certified constants (proof/B_chain.md, section 3).

Statement certified here. Let n >= n0 = e^{73.89} (the threshold of B5(1), B8 and (H8)), p in the complex box K of
B5(1), u* = 0.0081 (a point of the Stokes line inside the matching domain |z| <= Z cos(kappa) = 0.008120, see
B_chain.md section 1), and the label map L(Y) := e^{1/(2u*)} u*^{-G} Y_Phi(u*) (b5_label.py). With Psi_out =
Y+_out - Y-_out the outer jump and k(p) the Stokes multiplier vector of b5_label.py (psi_in = Psi k),
    V^-1 L(Psi_out) = (I + R~) V^-1 k + E~,    ||R~||_inf <= eps_L,    |E~|_inf <= C_E / n,
where eps_L is lemma L's bound at u* (b5_label.py) and
    C_E = e^{1/(2u*)} u*^{1 - mu_lo} ||V^-1||_inf (x_Phi,-1 + x_Phi,+1),
with x_Phi the B5(1) super-solution (b5_bounds_k0.3_Z0.0085.json) for the (H8) data delta certified by B8
(b8m_h8_K4_L73.89.log), by superposition (super-solutions add; b5_threshold.py), and mu_lo = min Re(eig G).
Proof: E~ = V^-1 L(W+ - W-), |W+_Phi,mu(u*)| <= x_Phi,mu u*/n (B5(1) on the Stokes segment of W+_n), W-(z; p) =
sigma(W+(z; p^bar)) so |W-_Phi,mu| <= x_Phi,-mu u*/n, and |(u*^-Lambda)_ii| = u*^{-Re lambda_i} <= u*^{-mu_lo}.
All three terms are holomorphic in p on K, so for real p with |p - x^| <= 1e-6 (componentwise) Cauchy's estimate on
the polydisc of radius 1e-6 gives |d_j E~| <= C_E/(n 1e-6) and ||d_j R~|| <= eps_L/1e-6.

    python3 b5_jump.py [--label=b5_label.json] [--ustar=0.0081]
    python3 b5_jump.py --mutate=sum     drops the forcing term from the superposition: the consistency check against
                                         the directly certified super-solution for delta_ref must then FAIL
"""
import json
import math
import os
import sys

from flint import arb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
INFL = arb("1e-12")      # the JSON stores floats of arb upper bounds: inflate by a relative 1e-12


def load_h8_delta():
    with open(os.path.join(HERE, "b8m_h8_K4_L73.89.log")) as fh:
        for line in fh:
            if line.startswith("{") and '"delta"' in line:
                d = json.loads(line)
                if d.get("ok"):
                    return {"P": d["delta"]["P"], "F": d["delta"]["Phi"], "C": d["delta"]["C"], "v": d["delta"]["th"]}
    raise RuntimeError("no certified (H8) record in b8m_h8_K4_L73.89.log")


def superpose(ss, delta):
    """x = x^forcing + sum_B delta_B x^(B) per harmonic, as arb upper bounds"""
    x = {}
    for c in ss["forcing"]["x"]:
        s = arb(0) if MUT == "sum" else arb(ss["forcing"]["x"][c])
        for blk in "PFCv":
            s += arb(str(delta[blk])) * arb(ss[f"unit_{blk}"]["x"][c])
        x[c] = s * (1 + INFL)
    return x


def main():
    with open(os.path.join(HERE, "b5_bounds_k0.3_Z0.0085.json")) as fh:
        b5 = json.load(fh)[0]
    ss = b5["supersolution"]
    if not all(ss[t]["ok"] for t in ("forcing", "unit_P", "unit_F", "unit_C", "unit_v", "data")) or not b5["ok"]:
        print("B5(1) certificate not ok")
        return 1
    delta = load_h8_delta()
    # consistency: the superposition for delta_ref must reproduce the directly certified x(delta_ref) to 2 %
    xr = superpose(ss, b5["delta_ref"])
    dev = max(abs(float(xr[c]) - v) / v for c, v in ss["data"]["x"].items())
    ok_sup = dev < 0.02
    # with B8's data: (H8) needs delta <= delta_ref (then B5(1) applies); check it
    ok_ref = all(delta[b] <= b5["delta_ref"][b] for b in "PFCv")
    x = superpose(ss, delta)
    xF = {m: x[f"F{m:+d}"] for m in (-1, 1)}
    # lemma L
    lab_path = os.path.join(HERE, OPTS.get("label", "b5_label.json"))
    with open(lab_path) as fh:
        lab = json.load(fh)
    ustar = arb(OPTS.get("ustar", lab["ZL"]))
    if not arb(ustar.upper()) <= arb(arb(lab["ZL"]).upper()):
        print("u* beyond lemma L's segment")
        return 1
    Zc = arb(b5["Z"]) * arb(b5["kappa"]).cos()
    ok_dom = bool(ustar <= Zc)
    mu_lo = arb(lab["mu_lo"]) - arb("1e-12")
    Vinv = arb(lab["Vinv_norm"]) * (1 + INFL)
    epsL = arb(lab["epsL"]) * (1 + INFL)
    CE = (1 / (2 * ustar)).exp() * ustar ** (1 - mu_lo) * Vinv * (xF[-1] + xF[1])
    lnCE = CE.log()
    cauchy = arb("1e-6")
    print(f"B5(1) certificate: kappa = {b5['kappa']}, Z = {b5['Z']}, ln n0 = {b5['ln_n0_domain']:.4f}; matching radius "
          f"Z cos(kappa) = {float(Zc.lower()):.6f} >= u* = {float(ustar.mid())}: {ok_dom}")
    print(f"(H8) data certified by B8: delta = (P {delta['P']:.3g}, Phi {delta['F']:.6g}, C {delta['C']:.6g}, "
          f"vt {delta['v']:.6g}) <= delta_ref {b5['delta_ref']}: {ok_ref}" + (f"   [MUTATION {MUT}]" if MUT else ""))
    print(f"superposition check (delta_ref vs the directly certified x): worst relative difference {dev:.2e} (< 0.02): "
          f"{'PASS' if ok_sup else 'FAIL'}")
    print(f"B5(1) bound for these data: x_Phi = ({float(xF[-1]):.6g}, {float(xF[1]):.6g}) (mu = -1, +1)")
    print(f"lemma L at u*: eps_L = {float(epsL):.5f}, ||V^-1||_inf = {float(Vinv):.5f}, min Re lambda = {float(mu_lo):.6f}")
    print(f"B5(2): V^-1 L(Psi_out) = (I + R~) V^-1 k + E~, ||R~|| <= {float(epsL.upper()):.5f}, |E~| <= C_E/n with "
          f"C_E = {float(CE.upper()):.6e} = e^{float(lnCE.upper()):.4f}")
    print(f"   e^(1/(2u*)) = e^{float((1 / (2 * ustar)).upper()):.4f}; u*^(1 - mu_lo) = {float((ustar ** (1 - mu_lo)).upper()):.5g}")
    print(f"   derivatives (Cauchy, real p within 1e-6 of x^): |d E~| <= C_E/(n 1e-6) = e^{float((lnCE - cauchy.log()).upper()):.4f}/n, "
          f"||d R~|| <= {float((epsL / cauchy).upper()):.4g}")
    for L in (73.89, 80, 85, 90, 95, 100, 105):
        print(f"   ln n = {L:6.2f}: |E~| <= {float((CE * arb(-L).exp()).upper()):.3e}")
    ok = ok_sup and ok_ref and ok_dom and lab["ok"]
    out = dict(ustar=float(ustar.mid()), Zc=float(Zc.lower()), delta=delta, xPhi=[float(xF[-1].upper()), float(xF[1].upper())],
               epsL=float(epsL.upper()), Vinv=float(Vinv.upper()), mu_lo=float(mu_lo.lower()), CE=float(CE.upper()),
               lnCE=float(lnCE.upper()), cauchy=1e-6, ok=bool(ok))
    if "json" in OPTS:
        with open(os.path.join(HERE, OPTS["json"]), "w") as fh:
            json.dump(out, fh, indent=1)
    if MUT:
        print(f"MUTATION {MUT}: the check must FAIL: {'caught' if not ok else 'NOT caught'}")
        return 0 if not ok else 3
    print(f"all checks: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
