"""Rigorous route to Proposition 4: Euler integral -> Laplace transform in 1/eps -> Watson + Darboux.

y(eps) = x 2F1(a, b; kappa; z), z = x^2, kappa = (1 + 1/eps)/2, a = (1+s)/2, b = (2+s)/2.
Euler: 2F1 = Gamma(kappa)/(Gamma(b)Gamma(kappa-b)) int_0^1 t^{b-1}(1-t)^{kappa-b-1}(1-zt)^{-a} dt, and with
1 - t = e^{-2tau}:
    y = x P(kappa) * 2 int_0^inf e^{-tau/eps} G(tau) dtau,  G(tau) = e^{(2b-1)tau}(1-e^{-2tau})^{b-1}(1-z+z e^{-2tau})^{-a},
    P(kappa) = Gamma(kappa)/(Gamma(b)Gamma(kappa-b)).
Singularities of G: tau_l = Phi - i pi (l + 1/2) (zeros of 1 - z + z e^{-2tau}), Phi = ln x - ln(1-x^2)/2, and tau = i pi m
(branch points of (1 - e^{-2tau})^{b-1}); P's 1/kappa series has Borel singularities at tau = i pi m as well.
Prediction (Watson + Darboux, nearest singularities tau_0, tau_-1):
    y_k ~ x 2^{1-b}/Gamma(b) Gamma(k+b) sum_l h_l Gamma(k+a)/(Gamma(a) Gamma(k+1)) tau_l^{-k},
    h_l = lim_{tau -> tau_l} H(tau)(1 - tau/tau_l)^a,   H = tau^{1-b} G   (continued along the ray from 0).
Checks the prediction against the exact y_k (solver/largeD_linear_late.py) and the dominance switch to tau = +-i pi
for |Phi| > (sqrt 3/2) pi.
"""
import sys
import mpmath as mp
sys.argv = ["x", "122"]
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "largeD_linear_late.py")).read().split("if __name__")[0])
mp.mp.dps = 90


def logH_along(tl, z, lam_end, N=4000):
    """log H(tau) at tau = lam_end * tl, continued along the ray from 0 (unwrapping each log)."""
    def parts(t):
        return [mp.log(t), (2 * b - 1) * t, mp.log(1 - mp.exp(-2 * t)), mp.log(1 - z + z * mp.exp(-2 * t))]
    lam0 = mp.mpf("1e-6")
    prev = parts(lam0 * tl)
    acc = list(prev)
    for i in range(1, N + 1):
        lam = lam0 + (lam_end - lam0) * i / N
        cur = parts(lam * tl)
        for j in (0, 2, 3):
            d = mp.im(cur[j] - prev[j])
            k = mp.nint(d / (2 * mp.pi))
            acc[j] = acc[j] + (cur[j] - prev[j]) - 2j * mp.pi * k
        acc[1] = cur[1]
        prev = cur
    return (1 - b) * acc[0] + acc[1] + (b - 1) * acc[2] - a * acc[3]


def h_const(tl, z):
    vals = []
    for dl in ("1e-8", "1e-10"):
        lam = 1 - mp.mpf(dl)
        lh = logH_along(tl, z, lam, N=3000)
        vals.append(mp.exp(lh + a * mp.log(1 - lam)))
    return vals[-1], abs(vals[0] - vals[1]) / abs(vals[1])


for xs in ("0.5", "0.8", "0.9", "0.95"):
    x = mp.mpf(xs); z = x * x
    Phi = mp.log(x) - mp.log(1 - z) / 2
    taus = {l: Phi - 1j * mp.pi * (l + mp.mpf(1) / 2) for l in (0, -1)}
    hs = {}
    for l, tl in taus.items():
        hs[l], err = h_const(tl, z)
    out = []
    for k in (60, 90, 120):
        pred = sum(x * 2 ** (1 - b) / mp.gamma(b) * mp.gamma(k + b) * hs[l] * mp.gamma(k + a)
                   / (mp.gamma(a) * mp.gamma(k + 1)) * taus[l] ** (-k) for l in (0, -1))
        out.append(f"k {k}: y_k/pred - 1 = {mp.nstr(yk(k, x) / pred - 1, 4)}")
    w = abs(hs[-1] * taus[-1] ** (-100)) / abs(hs[0] * taus[0] ** (-100))
    print(f"x {xs}: |tau_0| {mp.nstr(abs(taus[0]), 5)}; weight(l=-1)/weight(l=0) at k=100: {mp.nstr(w, 3)}; " + "; ".join(out),
          flush=True)
