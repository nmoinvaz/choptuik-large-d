#!/usr/bin/env python3
"""Lemma 6 (proof/A_certify.md section 11): the certificate runs on the LOWER path, saving the full state.

The path is the one of `validated_int.py krawczyk --rho=0.2`: from xi0 = 0.1 e^{3 i pi/8} along the ray to |xi| = 0.2,
along the arc |xi| = 0.2 down to the real axis, then along the real axis from 0.2 to xi1 = 0.3. The integrator, the
start data and the Krawczyk assembly are validated_int.py's, unchanged (this file only calls them); what is added is
that every component of the state is saved, not only D_{+-1}, because lemma 6a needs the whole jump at xi1:

    point  run at x^: y+(0.2) on the real axis (end of the arc) and y+(xi1), all 7 components
    box    run over X = x^ + [-h, h]^2 with the variational columns: the state set {y+(xi1; x) : x in X} (mean value
           form, all 7 components) and V = dy+(xi1)/d(A, w/w_inf) over X (all 7 rows)
    krawczyk  K(X) from the saved runs with validated_int.krawczyk_assemble (the assembly of section 5.2)

    python3 lemma6_path.py point [--out=f.json]
    python3 lemma6_path.py box [--h=1e-9] [--out=f.json]
    python3 lemma6_path.py krawczyk --point=f.json --box=g.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validated_int as VI  # noqa: E402
import lemma6_common  # noqa: E402
lemma6_common.install()            # a-priori inflation with a common floor (see lemma6_common.py)
from flint import arb, acb, ctx  # noqa: E402

AH, WH = "2.5421480475", "0.7769940136"
R0, RHO, XI1 = "0.1", "0.2", "0.3"


def s_acb(z):
    return [z.real.str(600, radius=True, more=True), z.imag.str(600, radius=True, more=True)]


def legs():
    return [VI.Leg("ray", R0, RHO), VI.Leg("arc", R0, RHO), VI.Leg("real", RHO, XI1)]


def point_run(verbose=True):
    t0 = time.time()
    s = VI.Sys(gravity=True)
    ctx.prec = 320
    Ahat, wfhat = arb(AH), arb(WH)
    ctx.prec = VI.PREC
    lo, eta = VI.start_lohner_fk(s, Ahat, wfhat, Ahat, wfhat, R0)
    out = {"A": AH, "wf": WH, "path": f"ray {R0}->{RHO}, arc {RHO}, real {RHO}->{XI1}", "steps": []}
    for k, leg in enumerate(legs()):
        n = lo.run(leg, 1e-3, verbose)
        out["steps"].append(n)
        X = lo.box()
        if k == 1:
            out["y_rho"] = [s_acb(x) for x in X[:s.nd]]
    X = lo.box()
    out["y_xi1"] = [s_acb(x) for x in X[:s.nd]]
    out["Dm1"], out["Dp1"] = s_acb(X[0]), s_acb(X[1])
    out["state_rad"] = max(VI.rad_of(x) for x in X[:s.nd])
    out["sec"] = time.time() - t0
    out["eta0"] = float(eta[0][0].upper())
    return out


def box_run(h, verbose=True):
    t0 = time.time()
    s = VI.Sys(gravity=True)
    ctx.prec = 320
    Ahat, wfhat = arb(AH), arb(WH)
    A, wf = Ahat + arb(0, arb(h)), wfhat + arb(0, arb(h))
    ctx.prec = VI.PREC
    lo, eta = VI.start_lohner_fk(s, Ahat, wfhat, A, wf, R0, var=True, centre_set=True)
    out = {"A": AH, "wf": WH, "h": h, "steps": []}
    for leg in legs():
        out["steps"].append(lo.run(leg, 1e-3, verbose))
    V = lo.var_box()
    X = lo.box()
    out["V"] = [[s_acb(V[i, j]) for j in range(2)] for i in range(s.nd)]
    out["yX_xi1"] = [s_acb(x) for x in X[:s.nd]]
    out["dDm1"] = [s_acb(V[0, j]) for j in range(2)]
    out["dDp1"] = [s_acb(V[1, j]) for j in range(2)]
    out["state_rad"] = max(VI.rad_of(x) for x in X[:s.nd])
    out["V_rad"] = max(VI.rad_of(V[i, j]) for i in range(s.nd) for j in range(2))
    out["sec"] = time.time() - t0
    return out


def krawczyk(pt, bx):
    h = bx["h"]
    ok, K, margins, g_hat, JX, JD, dJD = VI.krawczyk_assemble(AH, WH, h, XI1, pt, bx)
    print(f"lower path ({pt['path']}), X = x^ + [-{h}, {h}]^2")
    print(f"   J_D(x^) = {JD.str(8, radius=True)}  (radius {VI.rad_of(JD):.1e})")
    print(f"   dJ_D/d(A, wf) over X = {dJD[0].str(6, radius=True)}, {dJD[1].str(6, radius=True)}")
    print("   J(X) = [[" + ", ".join(JX[0][j].str(6, radius=True) for j in range(2)) + "], [" +
          ", ".join(JX[1][j].str(6, radius=True) for j in range(2)) + "]]")
    print(f"   K(X) = [{K[0].str(15, radius=True)}, {K[1].str(15, radius=True)}]")
    print(f"   K(X) in int X: {ok}; margins {margins[0]:.3e}, {margins[1]:.3e}")
    # tighter boxes X' = x^ + [-h', h']^2 inside X: J(X) also encloses the Jacobian over X', so the same test applies
    import krawczyk as KR
    best = None
    for hp in ("5e-10", "3e-10", "2e-10", "1.5e-10", "1.2e-10", "1e-10", "8e-11", "6e-11"):
        okp, Kp, mp = KR.krawczyk([arb(AH), arb(WH)], g_hat, JX, [arb(hp), arb(hp)])
        if okp:
            best = (hp, Kp, mp)
    if best:
        hp, Kp, mp = best
        lo = [float(k.lower()) for k in Kp]
        hi = [float(k.upper()) for k in Kp]
        print(f"   smallest passing sub-box tried: h' = {hp} (margins {mp[0]:.2e}, {mp[1]:.2e}); the zero lies in "
              f"A in [{Kp[0].lower().str(20, radius=False)}, {Kp[0].upper().str(20, radius=False)}], "
              f"w/w_inf in [{Kp[1].lower().str(20, radius=False)}, {Kp[1].upper().str(20, radius=False)}] "
              f"(widths {hi[0] - lo[0]:.1e}, {hi[1] - lo[1]:.1e})")
    return ok, K, margins


if __name__ == "__main__":
    mode = sys.argv[1]
    opts = {a.split("=", 1)[0][2:]: a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--")}
    if mode in ("point", "box"):
        res = point_run() if mode == "point" else box_run(opts.get("h", "1e-9"))
        print(f"{mode}: steps {res['steps']}, {res['sec']:.0f} s, final state radius {res['state_rad']:.2e}")
        if "out" in opts:
            with open(opts["out"], "w") as fh:
                json.dump(res, fh, indent=1)
    elif mode == "krawczyk":
        with open(opts["point"]) as fh:
            pt = json.load(fh)
        with open(opts["box"]) as fh:
            bx = json.load(fh)
        krawczyk(pt, bx)
