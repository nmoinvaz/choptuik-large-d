"""Inner fast-type Stokes multipliers order by order in the amplitude eps, all harmonics exact (version 2).

As stokes_amp.py, but each harmonic nu is normalized by its exact linear late-order form,
    N_j(eps) = 2^j Gamma(a_nu + j - delta/2) Gamma(b_nu + j - delta/2) / j!,   a, b = (1+s)/2, (2+s)/2, s = i nu w,
    delta = <F^2>(eps)  (the gravity shift of the exponent gamma = a + b - 1 - delta),
so the linear part is constant in j exactly and the large (nu w)^2/j corrections of high harmonics are removed.
R_j = [e^{-P} d_j]_nu / N_j, expanded in eps, fitted per order with 1 + Gamma ratios in 1/j and the families nu -/+ 2, 4.
F's harmonics nu >= 3 are tuned order by order so that S_nu = 0 (they are not free: the selection kills every S_nu);
S_1(eps) is what remains: S_1 = 0 is the selection condition for (eps, omega).

    python3 stokes_amp2.py wf K J        (wf = omega / omega_inf)
"""
import sys
import json
import numpy as np
import mpmath as mp
from amp_series import series, mul

W0 = 2 * np.pi / 1.64389375


def eps_mul(a, b, K):
    return np.convolve(a, b)[:K + 1]


def eps_exp(a, K):
    out = np.zeros(K + 1, complex); out[0] = 1.0
    term = out.copy()
    for n in range(1, K + 1):
        term = eps_mul(term, a, K) / n
        out = out + term
    return out


def norm_series(a, b, j, half_delta, K):
    """eps-series of (j!)^2 / (Gamma(a + j - h) Gamma(b + j - h)) (d is stored scaled by 1/(2^j j!)), h = delta/2."""
    L = np.zeros(K + 1, complex)
    hp = np.zeros(K + 1, complex); hp[0] = 1.0
    for n in range(1, K // 2 + 2):
        hp = eps_mul(hp, -half_delta, K)
        L = L - complex((mp.psi(n - 1, a + j) + mp.psi(n - 1, b + j)) / mp.factorial(n)) * hp
    base = complex(mp.factorial(j) ** 2 * mp.rgamma(a + j) * mp.rgamma(b + j))
    return base * eps_exp(L, K)


def lin_S(nu, w):
    s = 1j * nu * mp.mpf(w)
    return complex(1 / (mp.gamma((1 + s) / 2) * mp.gamma((2 + s) / 2)))


def prep(w, K, J, fextra):
    d, s, g = series(w, K, J, fextra)
    H = K
    F = d[0] / 2
    F2 = mul(F, F, K)
    mus = np.arange(-H, H + 1)
    P = np.zeros_like(F2)
    for i, mu in enumerate(mus):
        if mu != 0:
            P[:, i] = F2[:, i] / (1j * mu * w)
    E = np.zeros_like(P); E[0, H] = 1.0
    term = E.copy()
    for n in range(1, K + 1):
        term = mul(term, -P, K) / n
        E = E + term
    return d, E, F2[:, H].copy()


def fit_nu(w, K, J, d, E, delta, nu, jlo, nm=6, fam=(-4, -2, 2, 4), nfam=3, ks=None):
    H = K
    s_ = 1j * nu * mp.mpf(w)
    a, b = (1 + s_) / 2, (2 + s_) / 2
    g0 = a + b - 1
    js = list(range(jlo, J + 1))
    R = np.array([eps_mul(mul(E, d[j], K)[:, H + nu], norm_series(a, b, j, delta / 2, K), K) for j in js])
    cols = [np.ones(len(js), complex)]
    for m in range(1, nm + 1):
        cols.append(np.array([complex(mp.gamma(j + g0 - m) / mp.gamma(j + g0)) for j in js]))
    for sh in fam:
        gs = g0 + sh * 1j * mp.mpf(w)
        for m in range(1, nfam + 1):
            cols.append(np.array([complex(mp.gamma(j + gs - m) / mp.gamma(j + g0)) for j in js]))
    A = np.array(cols).T
    sc = np.abs(A).max(0)
    out, resid = {}, {}
    for k in (ks if ks is not None else range(K + 1)):
        y = R[:, k]
        if np.abs(y).max() == 0:
            out[k], resid[k] = 0j, 0.0
            continue
        c, *_ = np.linalg.lstsq(A / sc, y, rcond=None)
        out[k] = c[0] / sc[0]
        resid[k] = float(np.abs(A / sc @ c - y).max() / np.abs(y).max())
    return out, resid


def run(wf, K, J, jlo=None, verbose=True):
    w = wf * W0
    jlo = jlo or J // 3
    fextra = {}
    for order in range(3, K + 1, 2):
        nus = [nu for nu in range(-order, order + 1, 2) if abs(nu) >= 3]
        d, E, delta = prep(w, order, J, fextra)
        for nu in nus:
            S, rs = fit_nu(w, order, J, d, E, delta, nu, jlo, ks=[order])
            fextra[(order, nu)] = fextra.get((order, nu), 0) - S[order] / (2 * lin_S(nu, w))
            if verbose:
                print(f"   order {order} nu {nu:+d}: S_gen {abs(S[order]):.3e} (fit resid {rs[order]:.1e}) -> f {fextra[(order, nu)]:.6g}",
                      flush=True)
    d, E, delta = prep(w, K, J, fextra)
    S1, r1 = fit_nu(w, K, J, d, E, delta, 1, jlo)
    chk = {}
    for nu in range(3, K + 1, 2):
        Sn, _ = fit_nu(w, K, J, d, E, delta, nu, jlo)
        chk[nu] = {k: abs(Sn[k]) / max(1e-300, abs(lin_S(nu, w))) for k in Sn}
    sig = [S1[k] / S1[1] for k in range(1, K + 1, 2)]
    if verbose:
        print(f"S_1^(1) = {S1[1]:.12g}  (exact {-1j * lin_S(1, w):.12g})")
        for i, k in enumerate(range(1, K + 1, 2)):
            print(f"  sigma_{i} (eps^{k}) = {sig[i]:+.10f}   fit resid {r1[k]:.1e}")
        for nu in chk:
            print(f"  check S_{nu}/linS_{nu} by order: " + " ".join(f"{k}:{v:.1e}" for k, v in chk[nu].items() if k % 2))
    return sig, fextra


if __name__ == "__main__":
    wf = float(sys.argv[1]); K = int(sys.argv[2]); J = int(sys.argv[3])
    sig, fx = run(wf, K, J)
    json.dump({"wf": wf, "K": K, "J": J, "sigma": [[z.real, z.imag] for z in sig],
               "f": {f"{k}_{nu}": [v.real, v.imag] for (k, nu), v in fx.items()}},
              open(f"sig_w{wf:.4f}_K{K}_J{J}.json", "w"))
