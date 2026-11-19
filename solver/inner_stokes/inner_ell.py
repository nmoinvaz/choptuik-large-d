#!/usr/bin/env python3
"""The D = infinity inner Stokes jump of the scalar l-multiplet, as inner_np.py's M-harmonic model (#994;
matter/STRUCTURE.md 3 (d)).

The multiplet's inner problem (STRUCTURE.md, derived in solver/matter/struct_inner.py) in the variables
phi = s^(l/2) psi, Q = psi_t / c is the scalar's heat system with an l-weighted gravity term:

    psi_t = c Q,   Q_t = c psi_s,   c_s = 2 s^l c Q^2 .

psi carries the scaling weight -l (Z = 2s d_s + t d_t - l psi d_psi - (l + 1) Q d_Q), so psi = T^(-l) Psi(xi, theta)
with T = -t on a discretely self-similar solution. In inner_np's variables (y = sqrt(s) Y, sigma = sqrt(2) s S with
Y = 2 sqrt(2) Q, S = 2 sqrt(2) psi_s, both times T^l, and xi = rho / T) the equations are inner_np's with two changes:

    xi^2 y_xi + xi (l + i mu w) y = [c sigma]                          the time derivative of a T^(-l) field: d_tau ->
    xi^2 sig_xi + xi (l + i mu w) sig = (xi d_xi - 1)[c y]               l + w d_theta, as for a mode of growth rate l
    xi c_xi = (xi^2 / 2)^l [c E],   E = [y^2 / 2]                       gravity weighted by s^l = (xi^2/2)^l at T = 1

(the sigma equation's xi c_xi y is the weighted [c E] times y). l = 0 is inner_np.Inner exactly. The flat problem at
weight l is the flat scalar problem at s -> i nu w + l, so the linear jump is fast_lin(i nu w + l) with the Stokes
constant K'(i nu w + l) (STRUCTURE.md: the Stokes constant ~ 1 / Gamma(1 + sigma + l)).

The selection function is R = J_1 / (flat jump), with R -> 1 as A -> 0 and R = 0 the selection. Where the flat jump
is large against the slow solution at xi1 (tau = |flat jump| / |slow y_1| >> 1, at large w or xi1), the lateral
solution at xi1 is dominated by its fast part and R picks up terms in A^2 tau that do not vanish as A -> 0 (only at a
zero, where the jump vanishes). So for l >= 1 and at large w, xi1 is chosen per omega so that tau(xi1) is a target
(1e-6 to 1e-4); a genuine zero is the same at every xi1.

    python3 inner_ell.py checklin M L        flat jump against 2 pi i K'(i nu w + l) fast_lin(i nu w + l) 2 F_nu: R = 1
    python3 inner_ell.py sym                 the solver equations against the psi system (sympy, l symbolic)
    python3 inner_ell.py zero 0              the planted l = 0 M = 1 zero (A 2.54215, w/w_inf 0.77699) at xi1 0.2, 0.225,
                                             0.275
    python3 inner_ell.py zero 0 --mutate=grav    l = 0 with the gravity weight of l = 1 (shift 0): must fail
    python3 inner_ell.py zero 1 A wf [--tau=1e-6,1e-5,1e-4] [--expect=A,wf]   the l = 1 zero from a start, at the xi1
                                             of each tau target; pass: the same zero at every xi1 (relative 1e-6), and
                                             with --expect that zero (relative 1e-6)
    python3 inner_ell.py zero 1 A wf --mutate=grav --expect=A,wf    l = 1 with the unweighted gravity: must fail
    python3 inner_ell.py scan L [--tau=1e-5]  |R| on a grid (A 2..80, w/w_inf 1..5), xi1 per omega from tau; local minima
                                             with |R| < 0.1 refined by zero
    options --delta=D (the ray's angle past pi/2, default pi/8), --rtol=R (1e-12), --margin=M (r0's dominance, 12)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inner_np as I  # noqa: E402

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")


def opt(name, default):
    v = next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), None)
    return default if v is None else v


RTOL = float(opt("rtol", 1e-12))           # the lateral integration's rtol
MARGIN = float(opt("margin", 12.0))        # r0: the weakest fast mode dominates the series by e^MARGIN on the ray
I.DELTA = float(opt("delta", np.pi / 8))   # the ray arg xi^2 = pi/2 + DELTA (inner_np reads it at call time)


class InnerEll(I.Inner):
    def __init__(self, M, w, Fh, ell=0, gravity=True, M0=None, lam=1.0, ell_grav=None):
        super().__init__(M, w, Fh, gravity, M0, lam)
        self.ell = ell
        self.ellg = ell if ell_grav is None else ell_grav        # the gravity weight's exponent (a mutation can split it)
        self.imw = (1j * self.k * w + ell) * self.odd

    def r0_for(self, margin=None):
        return super().r0_for(MARGIN if margin is None else margin)

    def series_at(self, xi0, J):
        """inner_np.Inner.series_at with the weighted gravity: E_w = (xi^2/2)^l E enters c and the sigma equation"""
        q = xi0 * xi0
        o, e = self.odd, self.even
        lg = self.ellg
        wq = (q / 2) ** lg
        D = [2 * self.F]
        S, G = [], [e * (self.k == 0) * (1.0 + 0j)]
        Eu = [np.zeros_like(self.F)]                              # unweighted E_m (coefficient of (xi/xi0)^(2m))
        Ew, CE = [np.zeros_like(self.F)], [np.zeros_like(self.F)]
        mags = [np.abs(D[0]).max()]
        for j in range(J):
            sj = (2 * j + 1) * D[j] + self.imw * D[j]
            for i in range(1, j + 1):
                sj = sj - self.mul(G[i], S[j - i], o)
            S.append(sj * o)
            m = j + 1
            if self.grav:
                em = np.zeros_like(self.F)
                for i in range(m):
                    em = em + self.mul(D[i], D[m - 1 - i], e)
                Eu.append(q * em / 2)
                Ew.append(wq * Eu[m - lg] if m - lg >= 1 else np.zeros_like(self.F))
                cm = np.zeros_like(self.F)
                for i in range(m):
                    cm = cm + self.mul(G[i], Ew[m - i], e)
                CE.append(cm)
                G.append(cm / (2 * m))
            else:
                Eu.append(np.zeros_like(self.F)); Ew.append(np.zeros_like(self.F)); CE.append(np.zeros_like(self.F))
                G.append(np.zeros_like(self.F))
            num = q * ((2 * j + 2) * S[j] + self.imw * S[j])
            for i in range(1, j + 2):
                num = num - 2 * (j + 1 - i) * self.mul(G[i], D[j + 1 - i], o)
            if self.grav:
                for mm in range(1, j + 2):
                    num = num - self.mul(CE[mm], D[j + 1 - mm], o)
            D.append(num * o / (2 * j + 2))
            mags.append(np.abs(D[-1]).max())
            if j > 10 and mags[-1] > 10 * min(mags):
                break
        jt = int(np.argmin(mags[1:])) + 1
        Dd = xi0 * sum(D[:jt]); Sg = q * sum(S[:jt]); Cc = sum(G[:jt])
        return self.pack(Dd, Sg, Cc), jt, mags[jt] / mags[0]

    def rhs(self, xi, Y):
        o, e = self.odd, self.even
        D, S, C = self.unpack(Y)
        dD = (self.mul(C, S, o) - self.imw * xi * D) / (xi * xi)
        if self.grav:
            E = self.mul(D, D, e) / 2 * (xi * xi / 2) ** self.ellg
            cE = self.mul(C, E, e)
            r2 = self.mul(C, xi * dD, o) + self.mul(cE, D, o) - self.mul(C, D, o)
            dC = cE / xi
        else:
            r2 = xi * dD - D
            dC = np.zeros_like(C)
        dS = (r2 - self.imw * xi * S) / (xi * xi)
        return self.pack(dD, dS, dC)


def jump1(A, wf, ell, r1, ell_grav=None):
    """the M = 1 model's harmonic-1 jump at r1, scaled by e^{1/(2 r1^2)} / (2 pi |K'(i w + l)|)"""
    w = I.W0 * wf
    m = InnerEll(1, w, {1: A / 2j, -1: -A / 2j}, ell=ell, ell_grav=ell_grav)
    J, info = m.jumps((r1,))
    # divided by A: J_1 is proportional to A at small A, and A = 0 is a trivial zero
    return J[r1][1 + m.L] * np.exp(1 / (2 * r1 * r1)) / (2 * np.pi * abs(I.Kp(1j * w + ell))) / A


_FL = {}


def ratio1(A, wf, ell, r1, ell_grav=None):
    """R = J_1 / (2 pi i K'(i w + l) f_lin(r1) 2 F_1), the jump over the flat jump: smooth in (A, w), R -> 1 as A -> 0,
    and R = 0 is the selection (no trivial root at A = 0)"""
    w = I.W0 * wf
    key = (round(wf, 14), ell, r1)
    if key not in _FL:
        _FL[key] = 2j * np.pi * I.Kp(1j * w + ell) * fast_lin_ell(1, w, ell, r1)
    m = InnerEll(1, w, {1: A / 2j, -1: -A / 2j}, ell=ell, ell_grav=ell_grav)
    J, info = m.jumps((r1,), rtol=RTOL)
    ratio1.info = info                         # (r0, series terms, smallest/first term)
    return J[r1][1 + m.L] / (_FL[key] * 2 * (A / 2j))


def tau(wf, ell, r1):
    """|flat jump| / |slow y_1| at r1 (slow y_1 ~ 2 F_1 r1): how far the lateral solution at r1 is from slow-dominated"""
    w = I.W0 * wf
    return 2 * np.pi * abs(I.Kp(1j * w + ell) * fast_lin_ell(1, w, ell, r1)) / r1


def r1_for(wf, ell, target):
    """the r1 in (0.092, 0.35) with tau(r1) = target (tau grows with r1), by bisection, rounded to 1e-6"""
    lo, hi = 0.092, 0.35
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if tau(wf, ell, mid) > target else (mid, hi)
    return round(0.5 * (lo + hi), 6)


def scan_M1(ell, As, wfs, target):
    """|R| on a grid, r1 per omega from tau(r1) = target; returns the grid, the r1s and the local minima of |R| below
    0.1 (candidate zeros) as (|R|, A, wf)"""
    r1s = [r1_for(wf, ell, target) for wf in wfs]
    G = np.array([[abs(ratio1(A, wf, ell, r1)) for wf, r1 in zip(wfs, r1s)] for A in As])
    cand = []
    for i in range(len(As)):
        for k in range(len(wfs)):
            nb = G[max(i - 1, 0):i + 2, max(k - 1, 0):k + 2]
            if G[i, k] < 0.1 and G[i, k] <= nb.min():
                cand.append((G[i, k], As[i], wfs[k]))
    return G, r1s, sorted(cand)


def zero_M1(ell, A0, wf0, r1, ell_grav=None):
    from scipy.optimize import root
    def f(v):                                  # a step to w/w_inf <= 0.05 or A <= 0 (r0 undefined) is refused
        if v[1] <= 0.05 or v[0] <= 0:
            return [1e3, 1e3]
        z = ratio1(v[0], v[1], ell, r1, ell_grav)
        return [z.real, z.imag]
    sol = root(f, [A0, wf0], method="lm", options={"xtol": 1e-12, "ftol": 1e-14})
    return sol.x, np.abs(sol.fun).max(), sol.success


def fast_lin_ell(nu, w, ell, r1, xia=0.09):
    """the flat fast solution (y component) of harmonic nu at real r1, s = i nu w + l: the Borel-summed series
    x^(-2s) e^(-1/(2x^2)) sum_k c_k (-2x^2)^k at xia (as lateral_np.fast_lin), integrated outward with InnerEll's flat
    right-hand side. lateral_np.fast_lin integrates with omega = Re(s / i) only, so it serves l = 0 alone."""
    s = 1j * nu * w + ell
    a, b = (1 + s) / 2, (2 + s) / 2
    z = 2 * xia * xia
    cs, c = [], 1 + 0j
    for k in range(100):
        cs.append(c)
        c = c * (k + 1 - a) * (k + 1 - b) / (k + 1)
    kt = min(range(1, 100), key=lambda k: abs(cs[k] * z ** k))
    P = sum(cs[k] * (-z) ** k for k in range(kt))
    dP = sum(cs[k] * k * (-z) ** (k - 1) * (-4 * xia) for k in range(1, kt))
    pre = xia ** (-2 * s) * np.exp(-1 / (2 * xia * xia))
    fa, fd = pre * P, pre * (dP + P * (-2 * s / xia + 1 / xia ** 3))
    m = InnerEll(abs(nu), w, {}, ell=ell, gravity=False)
    sc = abs(fa)
    D = np.zeros(2 * m.L + 1, complex); S = np.zeros_like(D); C = np.zeros_like(D)
    D[nu + m.L] = fa / sc
    S[nu + m.L] = (xia * xia * fd + s * xia * fa) / sc
    C[m.L] = 1.0
    Y = m.integrate(m.pack(D, S, C), "ray", xia, r1, 0.0, RTOL)
    return m.unpack(Y)[0][nu + m.L] * sc


def checklin(M, ell):
    if ell == 0:                                # the new fast solution against lateral_np's at l = 0
        from lateral_np import fast_lin
        for r1 in (0.15, 0.2):
            ref, new = fast_lin(1j * I.W0, r1), fast_lin_ell(1, I.W0, 0, r1)
            print(f"  fast solution at l = 0, r1 {r1}: lateral_np {ref:.10e}, inner_ell {new:.10e}, relative "
                  f"{abs(new / ref - 1):.1e}", flush=True)
    Fh = {}
    for nu in range(1, M + 1, 2):
        Fh[nu] = (0.3 + 0.1j) / nu
        Fh[-nu] = np.conj(Fh[nu])
    m = InnerEll(M, I.W0, Fh, ell=ell, gravity=False)
    J, info = m.jumps((0.15, 0.2))
    worst = 0.0
    for r1, Jd in J.items():
        for nu in range(1, M + 1, 2):
            s = 1j * nu * I.W0 + ell
            R = Jd[nu + m.L] / (2j * np.pi * I.Kp(s) * fast_lin_ell(nu, I.W0, ell, r1) * 2 * Fh[nu])
            worst = max(worst, abs(R - 1))
            print(f"  l {ell}, r1 {r1}, nu {nu}: R = {R:.10f}", flush=True)
    print(f"checklin M {M} l {ell}: max |R - 1| = {worst:.1e} (pass < 1e-6)")
    return worst < 1e-6


def sym():
    """the solver's equations in physical form, rho d_t y = c sigma, rho d_t sigma = rho d_rho (c y) - c y and
    rho d_rho c = s^l c y^2 / 2 (s = rho^2 / 2, rho d_rho = 2 s d_s), with y = sqrt(s) 2 sqrt(2) Q,
    sigma = sqrt(2) s 2 sqrt(2) psi_s and Q = psi_t / c, against the psi system Q_t = c psi_s, c_s = 2 s^l c Q^2 (l
    symbolic). The self-similar reduction of a T^(-l) field, rho d_t (T^(-l) f) = T^(-l) (xi^2 d_xi + xi (l + d_tau)) f
    with xi = rho / T, gives the solver's shift l; at T = 1 the weight s^l is (xi^2 / 2)^l."""
    import sympy as sp
    s, t, l = sp.symbols("s t l", positive=True)
    psi = sp.Function("psi")(s, t)
    c = sp.Function("c")(s, t)
    Q = psi.diff(t) / c
    rho = sp.sqrt(2 * s)
    y = sp.sqrt(s) * 2 * sp.sqrt(2) * Q
    sig = sp.sqrt(2) * s * 2 * sp.sqrt(2) * psi.diff(s)
    rdr = lambda f: 2 * s * sp.diff(f, s)
    eqs = {"y": rho * y.diff(t) - c * sig, "sigma": rho * sig.diff(t) - (rdr(c * y) - c * y),
           "c": rdr(c) - s ** l * c * y ** 2 / 2}
    psitt = sp.solve(sp.Eq((psi.diff(t) / c).diff(t), c * psi.diff(s)), psi.diff(t, 2))[0]   # Q_t = c psi_s
    cs = 2 * s ** l * c * Q ** 2                                                               # c_s = 2 s^l c Q^2
    ok = True
    for name, ex in eqs.items():
        ex = ex.subs(psi.diff(t, 2), psitt).subs(c.diff(s), cs)
        ex = sp.simplify(ex.subs(psi.diff(t, 2), psitt).subs(c.diff(s), cs))
        ok &= ex == 0
        print(f"{name} equation on the psi system: {ex}", flush=True)
    # control: the unweighted gravity (s^0) must leave a residual in the c equation for l = 1
    ex = (rdr(c) - c * y ** 2 / 2).subs(c.diff(s), cs).subs(l, 1)
    ctrl = sp.simplify(ex)
    print(f"control, the unweighted gravity term at l = 1: {ctrl}  ({'nonzero, as required' if ctrl != 0 else 'ZERO'})")
    return ok and ctrl != 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "checklin":
        good = checklin(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "sym":
        good = sym()
    elif cmd == "scan":
        ell = int(sys.argv[2])
        target = float(opt("tau", 1e-5))
        As, wfs = np.arange(2.0, 80.01, 2.0), np.round(np.arange(1.0, 5.001, 0.1), 10)
        G, r1s, cand = scan_M1(ell, As, wfs, target)
        print(f"scan l {ell}: A {As[0]}..{As[-1]} ({len(As)}), w/w_inf {wfs[0]:.1f}..{wfs[-1]:.1f} ({len(wfs)}), xi1 per "
              f"omega from tau {target:.0e} ({min(r1s):.4f}..{max(r1s):.4f}); {len(cand)} local minima of |R| below 0.1",
              flush=True)
        for g, A, wf in cand:
            r1 = r1_for(wf, ell, target)
            x, f, ok = zero_M1(ell, A, wf, r1)
            print(f"   grid min |R| {g:.3e} at A {A:.0f}, w/w_inf {wf:.1f}: refined at xi1 {r1}: A {x[0]:.8f}, w/w_inf "
                  f"{x[1]:.8f}, |R| {f:.1e}, {ok}", flush=True)
        good = True
    elif cmd == "zero":
        ell = int(sys.argv[2])
        rest = [a for a in sys.argv[3:] if not a.startswith("--")]
        A0, wf0 = (float(rest[0]), float(rest[1])) if len(rest) >= 2 else (2.54, 0.777)
        eg = (1 if ell == 0 else 0) if "grav" in MUT else None
        if ell == 0 and opt("tau", None) is None:
            r1s = [0.2, 0.225, 0.275]
            how = "fixed"
        else:
            targets = [float(v) for v in opt("tau", "1e-6,1e-5,1e-4").split(",")]
            r1s = [r1_for(wf0, ell, tg) for tg in targets]
            how = "tau " + ", ".join(f"{t:.0e}" for t in targets) + " at the start"
        print(f"l {ell}, M 1, start A {A0}, w/w_inf {wf0}; xi1 {r1s} ({how}); delta {I.DELTA:.6f}, rtol {RTOL:.0e}, "
              f"margin {MARGIN}" + (f"; MUTATION: gravity weight l = {eg}" if eg is not None else ""), flush=True)
        zs, res = [], []
        for r1 in r1s:
            x, f, ok = zero_M1(ell, A0, wf0, r1, eg)
            ratio1(x[0], x[1], ell, r1, eg)
            r0, jt, terr = ratio1.info
            zs.append(x); res.append(f)
            print(f"   xi1 {r1}: A {x[0]:.9f}  w/w_inf {x[1]:.9f}  (Delta {1.64389375 / x[1]:.7f})  |R| {f:.1e}  {ok}; "
                  f"tau at the zero {tau(x[1], ell, r1):.1e}, r0 {r0:.4f}, series {jt} terms (smallest/first {terr:.1e})",
                  flush=True)
        zs = np.array(zs)
        spread = np.max(np.abs(zs - zs[0]), axis=0) / np.abs(zs[0])
        print(f"relative spread over xi1: A {spread[0]:.1e}, w/w_inf {spread[1]:.1e}")
        if ell == 0:
            dA, dw = abs(zs[0][0] - 2.54215), abs(zs[0][1] - 0.77699)
            good = dA < 1e-5 and dw < 1e-5
            print(f"planted (l = 0, zero_M1.py's A 2.54215, w/w_inf 0.77699 to 5 digits): |dA| {dA:.1e}, |dw| {dw:.1e}  "
                  f"({'OK' if good else 'BAD'})")
        else:
            good = spread.max() < 1e-6 and max(res) < 1e-8
            print(f"the same zero at every xi1 (relative spread < 1e-6, |R| < 1e-8): {'OK' if good else 'BAD'}")
        ex = opt("expect", None)
        if ex is not None:
            e = np.array([float(v) for v in ex.split(",")])
            d = np.abs(zs[0] - e) / np.abs(e)
            ok = d.max() < 1e-6
            good &= ok
            print(f"expected zero A {e[0]}, w/w_inf {e[1]} (relative 1e-6): relative |dA| {d[0]:.1e}, |dw| {d[1]:.1e}  "
                  f"({'OK' if ok else 'BAD'})")
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
