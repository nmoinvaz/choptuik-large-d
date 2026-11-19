#!/usr/bin/env python3
"""The growth rate lambda* of the l-multiplet's M-harmonic inner model at D = infinity (#994; GAMMA.md 5).

A growth rate is a real lambda at which a real perturbation profile f (data 2 f, c(0) = 0) has vanishing de-mixed
multipliers: det S(lambda) = 0, S the 2N x 2N map from the profile (sin nu t, cos nu t) to the jump, computed by
first_order_run.pert_matrix on first_order_jet's JetModel with ELL = l (transport shift l on every component, E
weighted by (xi^2/2)^l, demix_matrix's weight-l fast modes). lambda = 0 (theta shift) and lambda = 1 (time
translation) are exact roots of every l, so first_order_run.find_lam's g = det S / (lambda (lambda - 1)) applies.

    python3 ell_lam.py ident          ELL = 0 against the base commit's first_order_jet.py (S and dS at the l = 0 M = 1
                                      zero, lambda 1.8): bit for bit
    python3 ell_lam.py cross L A wf [xi1]   the background jump of JetModel at ELL = L against inner_ell.InnerEll
                                      (another arrangement of the same equations) at a point off the zero (there
                                      J_1 = 0 and the two compare noise): relative 1e-9; --mutate=grav gives
                                      JetModel the other gravity weight (1 - min(L, 1)) and must fail
    python3 ell_lam.py modes L A wf [xi1] [--mutate=ellpert]   the exact modes: smallest singular value of the
                                      row-scaled S(0) and S(1) below 1e-7 (each), against S(lambda) at 0.5, 1.5, 2.5
    python3 ell_lam.py lam L A wf [--lam=2.0] [--xi1=X,...]   lambda* by find_lam from --lam at each xi1;
                                      l = 0: planted, the M = 1 model's 1.769122008 (FIRST_ORDER.md) to 1e-8;
                                      l >= 1: the same lambda* at every xi1 (relative 1e-6)
    python3 ell_lam.py scan L A wf [xi1]   g(lambda) = det S / (lambda (lambda - 1)) on lambda in [0.1, 6]: its sign
                                      changes (real roots) and local minima of |g|
    python3 ell_lam.py lam3 L A wf f3re f3im [--lam=2.0] [--xi1=X,...]   M = 3: the zero re-solved by
                                      first_order_hier.find_zero (deflated Newton, exact jets) from nested_ell's
                                      (A, w/w_inf, F3/F1 = f3re + i f3im), then lambda* by first_order_hier.find_lam;
                                      l = 0: planted, FIRST_ORDER.md's M = 3 zero (A 2.7264759, w/w_inf 1.015603095)
                                      and lambda* 2.006804459 (bars 1e-6, 1e-7, 1e-7). l >= 1: the deflated Sch is
                                      noisy at ~1e-4 relative (the harmonic-3 leakage ~2e11), which stalls the secant
                                      search, so lambda* is the root of a least-squares line through g at lambda =
                                      LAM - 0.01 .. LAM + 0.01 (5 points) with its standard error sigma; pass: the zero
                                      the same at every xi1 (relative 1e-6) and the lambda* within 3 max sigma
    --rtol=R sets first_order_run.RTOL (the jets' integration rtol, 1e-12)
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "inner_stokes"))
import first_order_jet as FJ  # noqa: E402
import first_order_run as R   # noqa: E402

BASE = "7745882e"            # wip/ell1-dinf's branch point: first_order_jet.py before the l weight
MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")
FJ.MUTATE.update(m for m in MUT.split(",") if m)
T0 = time.time()


def opt(name, default):
    v = next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), None)
    return default if v is None else v


R.RTOL = float(opt("rtol", R.RTOL))


def lam_grid(x, M, xi1, l0, half=0.01, n=5):
    """lambda* from a least-squares line through g(lambda) = det Sch / (lambda (lambda - 1)) on [l0 - half, l0 + half];
    returns lambda*, its standard error (from the line's residual scatter) and the (lambda, g) table"""
    import first_order_hier as H
    lams = l0 + np.linspace(-half, half, n)
    g = []
    for lm in lams:
        Sch, C, L, rH = H.sch_lam(x, M, xi1, lm)
        g.append(np.linalg.det(Sch) / (lm * (lm - 1)))
    g = np.array(g)
    X = np.column_stack([np.ones(n), lams - l0])
    (a, b), *_ = np.linalg.lstsq(X, g, rcond=None)
    res = g - X @ np.array([a, b])
    sg = np.sqrt(res @ res / (n - 2))
    root = l0 - a / b
    sxx = np.sum((lams - l0) ** 2)
    sl = sg / abs(b) * np.sqrt(1 / n + (root - l0 - np.mean(lams - l0)) ** 2 / sxx)
    return root, sl, list(zip(lams.tolist(), g.tolist()))


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def xi1_default(ell, wf):
    if ell == 0:
        return 0.2
    import inner_ell as E
    return E.r1_for(wf, ell, 1e-5)


def ident():
    src = subprocess.run(["git", "show", f"{BASE}:solver/inner_pert/first_order_jet.py"], cwd=HERE,
                         capture_output=True, text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), f"first_order_jet_{BASE}.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("fj_base", path)
    fb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fb)
    x = np.array([2.542148048, 0.776994014])
    FJ.ELL = 0
    S, dS, r0 = R.pert_matrix(x, 1, 0.2, 1.8)
    R.JetModel, R.demix_matrix = fb.JetModel, fb.demix_matrix       # the base commit's model in the same driver
    Sb, dSb, r0b = R.pert_matrix(x, 1, 0.2, 1.8)
    R.JetModel, R.demix_matrix = FJ.JetModel, FJ.demix_matrix
    if "ident" in MUT:
        S = S * (1 + 1e-15)
    same = np.array_equal(S, Sb) and np.array_equal(dS, dSb) and np.array_equal(r0, r0b)
    print(f"S {S.ravel()}\nS (base {BASE}) {Sb.ravel()}\nmax |dS - dS_base| {np.abs(dS - dSb).max():.1e}")
    print(f"ELL = 0 against first_order_jet.py at {BASE}, bit for bit: {'OK' if same else 'BAD'}")
    return same


def cross(ell, A, wf, xi1):
    import inner_ell as E
    FJ.ELL = ell
    if "grav" in MUT:
        FJ.ELL_GRAV = 1 - min(ell, 1)          # the mutation: JetModel with the other gravity weight
    x = np.array([A, wf])
    Fh = FJ.Fh_from(x, 1)
    jm = FJ.JetModel(1, FJ.W0 * wf, Fh, [], [0], {})
    FJ.ELL_GRAV = None
    Jj, _ = jm.jumps((xi1,))
    m = E.InnerEll(1, E.I.W0 * wf, {1: A / 2j, -1: -A / 2j}, ell=ell)
    Ji, _ = m.jumps((xi1,))
    a, b = Jj[xi1][0][1 + jm.L], Ji[xi1][1 + m.L]
    d = abs(a / b - 1)
    good = d < 1e-9
    print(f"l {ell}, A {A}, w/w_inf {wf}, xi1 {xi1}: J_1 JetModel {a:.12e}, InnerEll {b:.12e}, relative {d:.1e}  "
          f"(bar 1e-9: {'OK' if good else 'BAD'})")
    return good


def svmin(lam, x, xi1):
    S, dS, _ = R.pert_matrix(x, 1, xi1, lam)
    Ss, _ = R.scaled(S)
    sv = np.linalg.svd(Ss, compute_uv=False)
    return sv[-1] / sv[0]


def modes(ell, A, wf, xi1):
    FJ.ELL = ell
    x = np.array([A, wf])
    out = {lam: svmin(lam, x, xi1) for lam in (0.0, 1.0, 0.5, 1.5, 2.5)}
    for lam, v in out.items():
        print(f"l {ell}, xi1 {xi1}, lambda {lam}: smallest / largest singular value of the row-scaled S {v:.2e}")
    good = out[0.0] < 1e-7 and out[1.0] < 1e-7
    print(f"exact modes lambda = 0 and 1 (bar 1e-7): {'OK' if good else 'BAD'}"
          + (f"  [MUTATION {MUT}]" if MUT else ""))
    return good


def lam(ell, A, wf):
    FJ.ELL = ell
    x = np.array([A, wf])
    l0 = float(opt("lam", 2.0))
    if opt("xi1", None) is not None:
        xis = [float(t) for t in opt("xi1", "").split(",")]
    elif ell == 0:
        xis = [0.2, 0.175]
    else:
        import inner_ell as E
        xis = [E.r1_for(wf, ell, t) for t in (1e-6, 1e-5, 1e-4)]
    log(f"l {ell}, M 1, zero A {A}, w/w_inf {wf}; xi1 {xis}; start lambda {l0}" + (f"; MUTATION {MUT}" if MUT else ""))
    lams = []
    for xi1 in xis:
        lm, S, dS = R.find_lam(x, 1, xi1, l0)
        f, ll, sv = R.null_vectors(S)
        lams.append(lm)
        log(f"RESULT xi1 {xi1}: lambda* {lm:.12f}  (gamma = 1/lambda* {1 / lm:.10f}); singular values of S {sv}; "
            f"profile f = {f[0]:+.6f} sin t {f[1]:+.6f} cos t")
    lams = np.array(lams)
    spread = float(np.abs(lams - lams[0]).max() / abs(lams[0]))
    print(f"relative spread over xi1: {spread:.1e}")
    if ell == 0:
        d = abs(lams[0] - 1.769122008)
        good = d < 1e-8
        print(f"planted (l = 0 M = 1, FIRST_ORDER.md's lambda* 1.769122008, bar 1e-8): |d| {d:.1e}  "
              f"({'OK' if good else 'BAD'})")
    else:
        good = spread < 1e-6
        print(f"the same lambda* at every xi1 (relative 1e-6): {'OK' if good else 'BAD'}")
    return good


def scan(ell, A, wf, xi1):
    FJ.ELL = ell
    x = np.array([A, wf])
    lams = np.round(np.arange(0.1, 6.0001, 0.1), 10)
    g = []
    for lm in lams:
        if abs(lm - 1) < 1e-9:
            g.append(np.nan)
            continue
        S, dS, _ = R.pert_matrix(x, 1, xi1, lm)
        g.append(np.linalg.det(S) / (lm * (lm - 1)))
    g = np.array(g)
    print(f"l {ell}, A {A}, w/w_inf {wf}, xi1 {xi1}: g(lambda) = det S / (lambda (lambda - 1)), lambda 0.1..6.0")
    for lm, v in zip(lams, g):
        print(f"   lambda {lm:.1f}: g {v:+.6e}")
    roots = [(lams[i], lams[i + 1]) for i in range(len(lams) - 1)
             if np.isfinite(g[i]) and np.isfinite(g[i + 1]) and g[i] * g[i + 1] < 0]
    print(f"sign changes of g (real roots of the deflated determinant): {roots}")
    return True


def lam3(ell, A, wf, f3):
    import first_order_hier as H
    FJ.ELL = ell
    F3 = (A / 2j) * f3
    x = np.array([A, wf, -2 * F3.imag, 2 * F3.real])
    l0 = float(opt("lam", 2.0))
    if opt("xi1", None) is not None:
        xis = [float(t) for t in opt("xi1", "").split(",")]
    elif ell == 0:
        xis = [0.2, 0.175]
    else:
        import inner_ell as E
        xis = [E.r1_for(wf, ell, t) for t in (1e-6, 1e-5, 1e-4)]
    log(f"l {ell}, M 3, start A {A}, w/w_inf {wf}, F3/F1 {f3}; xi1 {xis}; start lambda {l0}")
    res = []
    for xi1 in xis:
        if ell:
            # F3 to full precision first (nested_ell's ladder): a start whose harmonic-3 jump is not ~0 is nonlinear
            # at xi1 (tau_3 ~ 1e9) and the deflated Newton diverges
            import nested_ell as N
            nz = N.Nested(ell, xi1, N.ladder(wf, ell, xi1))
            v, R1, r3, _ = nz.inner(A, wf, x[2:].copy())
            x = np.array([A, wf, v[0], v[1]])
            log(f"   F3 on the harmonic-3 ladder at xi1 {xi1}: |r_3| {r3:.1e}, Schur-corrected |r_1| {abs(R1):.1e}")
        xz, r, K, L = H.find_zero(x.copy(), 3, xi1)
        F3z = xz[2] / 2j + xz[3] / 2
        log(f"zero at xi1 {xi1}: A {xz[0]:.10f}  w/w_inf {xz[1]:.10f}  (Delta {1.64389375 / xz[1]:.7f})  F3/F1 "
            f"{F3z / (xz[0] / 2j):.8f}  |r_1| {np.linalg.norm(r[:2]):.1e}  |r_3| {np.linalg.norm(r[2:]):.1e}")
        if ell == 0:
            lm, sl = H.find_lam(xz, 3, xi1, l0), 0.0
        else:
            lm, sl, tab = lam_grid(xz, 3, xi1, l0)
            log("   g(lambda): " + "  ".join(f"{a:.4f}: {b:+.4e}" for a, b in tab))
        Sch, C, Lm, resH = H.sch_lam(xz, 3, xi1, lm)
        log(f"RESULT xi1 {xi1}: A {xz[0]:.10f}  w/w_inf {xz[1]:.10f}  lambda* {lm:.12f} +- {sl:.1e}  (gamma {1 / lm:.10f}); "
            f"Sch singular values at lambda* {np.linalg.svd(Sch, compute_uv=False)}")
        res.append((xz[0], xz[1], lm, sl))
    z = np.array(res)
    spread = np.max(np.abs(z[:, :3] - z[0, :3]), axis=0) / np.abs(z[0, :3])
    print(f"relative spread over xi1: A {spread[0]:.1e}, w/w_inf {spread[1]:.1e}, lambda* {spread[2]:.1e}")
    if ell == 0:
        d = np.abs(z[0, :3] - np.array([2.7264759, 1.015603095, 2.006804459]))
        good = d[0] < 1e-6 and d[1] < 1e-7 and d[2] < 1e-7
        print(f"planted (l = 0 M = 3, FIRST_ORDER.md: A 2.7264759, w/w_inf 1.015603095, lambda* 2.006804459; bars 1e-6, "
              f"1e-7, 1e-7): |d| {d[0]:.1e}, {d[1]:.1e}, {d[2]:.1e}  ({'OK' if good else 'BAD'})")
    else:
        dl = float(np.abs(z[:, 2] - z[0, 2]).max())
        sm = float(z[:, 3].max())
        good = spread[:2].max() < 1e-6 and dl < 3 * sm
        print(f"the same zero at every xi1 (relative 1e-6) and lambda* within 3 max sigma ({dl:.1e} against "
              f"3 x {sm:.1e}): {'OK' if good else 'BAD'}; mean lambda* {z[:, 2].mean():.6f}")
    return good


if __name__ == "__main__":
    cmd = sys.argv[1]
    rest = [a for a in sys.argv[2:] if not a.startswith("--")]
    if cmd == "ident":
        good = ident()
    elif cmd == "cross":
        ell, A, wf = int(rest[0]), float(rest[1]), float(rest[2])
        good = cross(ell, A, wf, float(rest[3]) if len(rest) > 3 else xi1_default(ell, wf))
    elif cmd == "modes":
        ell, A, wf = int(rest[0]), float(rest[1]), float(rest[2])
        good = modes(ell, A, wf, float(rest[3]) if len(rest) > 3 else xi1_default(ell, wf))
    elif cmd == "lam":
        good = lam(int(rest[0]), float(rest[1]), float(rest[2]))
    elif cmd == "lam3":
        good = lam3(int(rest[0]), float(rest[1]), float(rest[2]), complex(float(rest[3]), float(rest[4])))
    elif cmd == "scan":
        ell, A, wf = int(rest[0]), float(rest[1]), float(rest[2])
        good = scan(ell, A, wf, float(rest[3]) if len(rest) > 3 else xi1_default(ell, wf))
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
