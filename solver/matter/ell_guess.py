"""Initial guess for the l-multiplet BVP from a near-critical evolution (ell_collapse.py), following the project's
ell = 0 route (solver/scratch/guess.py):

 1. pass 1: the subcritical run just below threshold, central history only: zero crossings T_k of chi(T, 0) in central
    proper time T give the accumulation time T* (geometric extrapolation) and Delta (crossings are Delta/2 apart);
 2. pass 2: the same run with full slices stored over the echoing window;
 3. self-similar coordinates in central time, s = T* - T, x_c = r/s, c_c = c(r)/c(0); the self-similarity horizon is
    the periodic orbit of dx_c/dtau_c = x_c - c_c, found by integrating backward in tau_c (attracting that way);
 4. SSH-normalised variables tau = -ln r_s, x = r/r_s, c = c_c/c_c(r_s); BVP data on every slice (centre q, c0;
    node states; horizon m, a, psi), least-squares Fourier fit over the chosen tau window with the parity of each field.
usage: python ell_guess.py n l slices.npz [pass]   (pass = run: do the evolutions and save slices; fit: transform)
"""
import os, sys, json
import numpy as np
from scipy.interpolate import CubicSpline
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def crossings(T, y):
    i = np.where(np.sign(y[1:]) != np.sign(y[:-1]))[0]
    return T[i] - y[i] * (T[i + 1] - T[i]) / (y[i + 1] - y[i])


def accumulation(Tk):
    """T* and the ratio rho = (T*-T_{k+1})/(T*-T_k) from consecutive triples of crossing times."""
    out = []
    for k in range(len(Tk) - 2):
        a, b, c = Tk[k:k + 3]
        den = a + c - 2 * b
        if den == 0:
            continue
        Ts = (a * c - b * b) / den
        out.append((Ts, (c - b) / (b - a)))
    return np.array(out)


def run_passes(n, l, A, out, t_end=15.0, dx=4e-4, per_unit=400, k_first=1, max_snaps=6000, rmax=30.0):
    """The evolution's time is central proper time (ell_collapse.py), so T = t. Pass 2 stores slices from the
    k_first-th crossing to the last one, spaced uniformly in ln(T* - T) (per_unit slices per unit)."""
    from ell_collapse import run
    o = run(A, n=n, l=l, dx=dx, t_end=t_end, central_dt=2e-4, rmax=rmax)
    c = o["central"]
    T, chi0 = c[:, 1], c[:, 2]
    Tk = crossings(T, chi0)
    acc = accumulation(Tk)
    print("status", o["status"], "t", o["t"])
    print("crossings (T):", np.round(Tk, 6).tolist())
    for Ts, rho in acc:
        print(f"  T* {Ts:.6f}  rho {rho:.4f}  Delta {-2 * np.log(rho):.4f}")
    Tstar = float(np.median(acc[len(acc) // 2:, 0]))
    t0, t1 = Tk[k_first], min(Tk[-1], Tstar - 1e-4)
    o2 = run(A, n=n, l=l, dx=dx, t_end=t1, central_dt=2e-4, snap_tmin=t0, snap_dt=1.0 / per_unit,
             snap_Tstar=Tstar, max_snaps=max_snaps, rmax=rmax)
    np.savez_compressed(out, r=o2["r"], snaps_t=o2["snaps_t"], slices=o2["slices"], central=c, Tk=Tk, A=A, n=n, l=l,
                        Tstar_pass1=Tstar)
    print(f"saved {len(o2['snaps_t'])} slices, T in [{o2['snaps_t'][0, 0]:.4f}, {o2['snaps_t'][-1, 0]:.4f}], "
          f"T* (pass 1) {Tstar:.6f}")


def transform(npz, Tstar, Delta, tau_a):
    """Self-similar data on the slices of one period tau_c in [tau_a, tau_a + Delta) (central-time self-similar
    coordinates tau_c = -ln(T* - T), x_c = r/(T* - T)). The SSH x_s(tau_c) is the periodic orbit of
    dx/dtau_c = x - c_c(tau_c, x), found as the fixed point of the one-period map (it is repelling forward, so the
    fixed point is well conditioned)."""
    from scipy.optimize import brentq
    d = np.load(npz)
    r = d["r"]; st = d["snaps_t"]; sl = d["slices"]
    T = st[:, 1]
    tc_all = -np.log(np.maximum(Tstar - T, 1e-300))
    keep = (T < Tstar) & (tc_all >= tau_a - 1e-12) & (tc_all <= tau_a + Delta + 1e-12)
    T, sl = T[keep], sl[keep]
    s = Tstar - T
    tauc = -np.log(s)
    chi, X, Pt, a, al = sl[:, 0], sl[:, 1], sl[:, 2], sl[:, 3], sl[:, 4]
    cc = (al / al[:, :1]) / a                                     # c / c(0)  (time = central proper time)
    ccs = [CubicSpline(r, cc[j]) for j in range(len(T))]
    J = len(T)

    def cfun(j, x):
        return ccs[j](x * s[j])

    def orbit(x0):
        xs = np.empty(J); xs[0] = x0
        for j in range(J - 1):
            h = tauc[j + 1] - tauc[j]
            x = xs[j]
            k1 = x - cfun(j, x)
            xm = x + h * k1
            k2 = xm - cfun(j + 1, xm)
            xs[j + 1] = min(max(x + 0.5 * h * (k1 + k2), 1e-6), 50.0)
        return xs
    # the period map ends one sampling step short of tau_a + Delta: correct with the last step to tau_a + Delta
    def gap(x0):
        xs = orbit(x0)
        xe = xs[-1]
        h = tau_a + Delta - tauc[-1]
        return xe + h * (xe - cfun(J - 1, xe)) - x0
    xg = np.linspace(0.2, 6, 300)
    f0 = np.array([xg_ - cfun(0, xg_) for xg_ in xg])
    i = np.where(np.sign(f0[1:]) != np.sign(f0[:-1]))[0]
    x_inst = xg[i[0]] if len(i) else 1.0
    lo, hi = 0.7 * x_inst, 1.4 * x_inst
    glo, ghi = gap(lo), gap(hi)
    while not (glo < 0 < ghi) and hi < 20:
        lo, hi = 0.8 * lo, 1.25 * hi
        glo, ghi = gap(lo), gap(hi)
    x0 = brentq(gap, lo, hi, xtol=1e-12)
    xs = orbit(x0)
    rs = xs * s
    tau = tauc - np.log(xs)
    return dict(r=r, T=T, s=s, tauc=tauc, xs=xs, rs=rs, tau=tau, chi=chi, X=X, Pt=Pt, a=a, cc=cc, x_inst=x_inst)


def bvp_samples(tr, nodes, l):
    """BVP data on every slice: q, c0, node states (p, m, a, c, psi), horizon (m, a, psi)."""
    r = tr["r"]; out = {k: [] for k in ("q", "c0", "nodes", "mS", "aS", "qS")}
    for j in range(len(tr["T"])):
        sp = {k: CubicSpline(r, tr[k][j]) for k in ("chi", "X", "Pt", "a", "cc")}
        rs = tr["rs"][j]
        ccs = sp["cc"](rs)

        def state(x):
            rr = x * rs
            chi, X, Pt = sp["chi"](rr), sp["X"](rr), sp["Pt"](rr)
            p = rr ** l * (l * chi + rr * X + rr * Pt)
            m = rr ** l * (l * chi + rr * X - rr * Pt)
            return np.array([p, m, sp["a"](rr), sp["cc"](rr) / ccs, rr ** l * chi])
        out["q"].append(rs ** l * tr["chi"][j][0])
        out["c0"].append(1.0 / ccs)
        out["nodes"].append(np.stack([state(x) for x in nodes]))
        sh = state(1.0)
        out["mS"].append(sh[1]); out["aS"].append(sh[2]); out["qS"].append(sh[4])
    return {k: np.array(v) for k, v in out.items()}


def fourier_fit(tau, y, Delta, parity, kmax):
    """Least-squares fit y(tau) = sum_k a_k cos(k w tau) + b_k sin(k w tau) over the parity's harmonics.
    Returns dict k -> complex F_k = (a_k - i b_k)/2 (k = 0: a_0)."""
    w = 2 * np.pi / Delta
    ks = list(range(1, kmax + 1, 2)) if parity == "odd" else list(range(0, kmax + 1, 2))
    cols = []
    for k in ks:
        cols.append(np.cos(k * w * tau))
        if k > 0:
            cols.append(np.sin(k * w * tau))
    A = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    F, i = {}, 0
    for k in ks:
        if k == 0:
            F[0] = coef[i] + 0j; i += 1
        else:
            F[k] = 0.5 * (coef[i] - 1j * coef[i + 1]); i += 2
    return F, float(np.sqrt(np.mean(res ** 2)))


def make_guess(S, samp, tau, Delta, kmax):
    """Solution vector for DSSEll S from the per-slice BVP samples (fit over the given tau samples)."""
    sizes, K, L = S.layout()
    fits, rms = [], {}

    def fit(y, par, name):
        F, e = fourier_fit(tau, y, Delta, par, kmax)
        rms[name] = max(rms.get(name, 0.0), e / (np.sqrt(np.mean(y ** 2)) + 1e-300))
        return F
    Fq = fit(samp["q"], "odd", "q")
    phi = -np.angle(Fq[1])                        # tau shift making Im F_1(q) = 0, Re F_1(q) > 0
    shift = lambda F: {k: v * np.exp(1j * k * phi) for k, v in F.items()}

    def to_vec(F, par):
        F = shift(F)
        ks = S.kodd if par == "odd" else S.keven
        if par == "odd":
            return np.concatenate([[F.get(k, 0j).real for k in ks], [F.get(k, 0j).imag for k in ks]])
        return np.concatenate([[F.get(0, 0j).real], [F.get(k, 0j).real for k in ks[1:]],
                               [F.get(k, 0j).imag for k in ks[1:]]])
    vec = [to_vec(Fq, "odd"), to_vec(fit(samp["c0"], "even", "c0"), "even")]
    names = ("p", "m", "a", "c", "psi")
    pars = ("odd", "odd", "even", "even", "odd")
    for i in range(K):
        for f in range(5):
            vec.append(to_vec(fit(samp["nodes"][:, i, f], pars[f], names[f]), pars[f]))
    for key, par in (("mS", "odd"), ("aS", "even"), ("qS", "odd")):
        vec.append(to_vec(fit(samp[key], par, key), par))
    return np.concatenate(vec + [[Delta]]), rms


if __name__ == "__main__":
    n, l, npz = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    dx = float(sys.argv[5]) if len(sys.argv) > 5 else 4e-4
    rmax = float(os.environ.get("ELL_RMAX", "30"))
    tag = f"n{n:g}_l{l}" + (f"_dx{dx:g}" if dx != 4e-4 else "") + (f"_r{rmax:g}" if rmax != 30 else "")
    if len(sys.argv) > 4 and sys.argv[4] == "run":
        ps = json.load(open(os.path.join(HERE, "data", f"pstar_{tag}.json")))
        run_passes(n, l, ps["A_lo"], npz, dx=dx, t_end=float(os.environ.get("ELL_TEND", "15")), rmax=rmax,
                   per_unit=int(os.environ.get("ELL_PER_UNIT", "400")))
