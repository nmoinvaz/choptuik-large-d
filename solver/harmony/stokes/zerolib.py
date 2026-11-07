"""Shared pieces: load sigma_k, ratio-law fit, Borel-Pade exponential sums, zero search on a disc.

Borel-Pade sum: B(t) = sum_k k! sigma_k t^(-k-1); with u = 1/t, b(u) = sum_k k! sigma_k u^k ~ P/Q ([m/n], m <= n).
S_BP(x) = c0 + sum_p r_p exp(t_p x), t_p = 1/u_p (roots of Q), r_p = -P(u_p)/(u_p Q'(u_p)), c0 = p_m/q_n if m = n.
Written independently of solver/inner_amp/planted_k14.borel_pade (same definition).
"""
import json
import numpy as np
import mpmath as mp

REPO = "/Users/nathan/Source/choptuik-large-d/solver/inner_amp"
FILES = {1.0: REPO + "/scan45J1500/sigF_w1.0000_K45_J1500_p1280.json",
         0.99: REPO + "/scan43J1500/sigF_w0.9900_K43_J1500_p1280.json",
         1.01: REPO + "/scan43J1500/sigF_w1.0100_K43_J1500_p1280.json",
         "1.0K43": REPO + "/scan43J1500/sigF_w1.0000_K43_J1500_p1280.json"}


def load(key):
    d = json.load(open(FILES[key]))
    return [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in d["sigma_str"]]


def ratio_law(sig, lo=12, hi=None):
    """r_k = sigma_k/sigma_{k+1} = (k + beta)/kappa + d/k: least squares on k = lo..hi-1; C from the last k"""
    hi = hi or len(sig) - 1
    ks = list(range(lo, hi))
    A = mp.matrix([[k, 1, mp.mpf(1) / k] for k in ks])
    r = mp.matrix([sig[k] / sig[k + 1] for k in ks])
    sol = mp.qr_solve(A, r)[0]
    kappa = 1 / sol[0]
    beta = sol[1] * kappa
    k = hi
    C = sig[k] * mp.gamma(k + beta) / kappa ** k
    return kappa, beta, C


def borel_pade(sig, m, n):
    a = [sig[k] * mp.factorial(k) for k in range(m + n + 1)]
    p, q = mp.pade(a, m, n)
    us = mp.polyroots(q[::-1], maxsteps=400, extraprec=400)
    dq = [k * q[k] for k in range(1, len(q))]
    ts = [1 / u for u in us]
    rs = [-mp.polyval(p[::-1], u) / (u * mp.polyval(dq[::-1], u)) for u in us]
    c0 = p[m] / q[n] if m == n else mp.mpc(0)
    return ts, rs, c0


def bp_func(ts, rs, c0):
    return lambda x: c0 + mp.fsum(r * mp.exp(t * x) for t, r in zip(ts, rs))


def bp_numpy(ts, rs, c0):
    T = np.array([complex(t) for t in ts]); R = np.array([complex(r) for r in rs]); c = complex(c0)
    return lambda X: c + (R[None, :] * np.exp(np.outer(X.ravel(), T))).sum(1).reshape(X.shape)


def zeros_in_disc(fnp, fmp, R=40.0, h=0.25, tol=1e-25):
    """grid minima of |f| refined by Newton (mpmath), deduplicated; then the argument principle on |x| = R
    as a completeness check (returns the winding count too)"""
    xs = np.arange(-R, R + h, h)
    X, Y = np.meshgrid(xs, xs)
    Z = X + 1j * Y
    F = np.abs(fnp(Z))
    found = []
    for i in range(1, len(xs) - 1):
        for j in range(1, len(xs) - 1):
            v = F[i, j]
            if abs(Z[i, j]) > R or not np.isfinite(v):
                continue
            if v <= F[i - 1:i + 2, j - 1:j + 2].min():
                try:
                    z = mp.findroot(fmp, mp.mpc(Z[i, j].real, Z[i, j].imag), tol=tol, maxsteps=100)
                except Exception:
                    continue
                if abs(fmp(z)) > 1e-12 * max(1, float(F.max()) * 0) + 1e-15 or abs(z) > R:
                    continue
                if all(abs(z - w) > 1e-6 for w in found):
                    found.append(z)
    # winding number on |x| = R (fine sampling, numpy)
    th = np.linspace(0, 2 * np.pi, 20001)
    vals = fnp(R * np.exp(1j * th))
    wind = int(round(np.sum(np.diff(np.unwrap(np.angle(vals)))) / (2 * np.pi)))
    return sorted(found, key=lambda z: abs(z)), wind


def cnum(z, d=6):
    return f"{float(mp.re(z)):.{d}f}{float(mp.im(z)):+.{d}f}i"
