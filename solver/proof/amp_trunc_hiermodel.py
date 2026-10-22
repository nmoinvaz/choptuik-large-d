#!/usr/bin/env python3
"""The amplitude expansion of hier.py's M-harmonic model: odd harmonics |mu| <= M (y, sigma, F) and even harmonics
|nu| <= M + 1 (c, E), every product truncated to that space, harmonics 3..M tuned (S_nu = 0 order by order), late-order
fits with every neighbour family present.

Why a second truncated expansion: amp_trunc.py (B_tail.md section 4) truncates *every* function at |mu| <= M, so its
even harmonics (c, E) stop at M - 1, one below inner_np.py / hier.py, which keep the even harmonic M + 1. Its first
change is at sigma_M, exactly where the even harmonic M + 1 enters, and it then explodes at sigma_{M+1}. The model
hier.py actually solves is AmpTrunc with H = M + 1: amp_flint's truncation |mu| <= H keeps the odd harmonics <= M (M + 1
is even) and the even ones <= M + 1. Order counting (proof/B_tail_M7.md, lemma O): in this model sigma_1 .. sigma_M are
exact, and the first omitted harmonic (the profile/y harmonic M + 2) enters at sigma_{M+1}. So
Delta sigma_{M+1} = sigma_{M+1} - sigma^{[M]}_{M+1} is the lowest-order term of the truncation increment
delta_M = s_{M+2} - s_M at fixed (A, w).

One change against amp_flint.Amp.S_at: its de-mixing sum runs over mu = -min(H, k'), ..., min(H, k') in steps of 2,
which assumes H odd; with H = M + 1 even it would read the even (empty) entries of d at orders k' > H. Here mu runs over
the odd harmonics |mu| <= min(H_odd, k'), H_odd = M. The de-mixing factor e^{-P} is amp_flint's (truncated at |mu| <= H
in its products, as in amp_trunc.py).

    python3 amp_trunc_hiermodel.py M K J [wf]   writes trunc_hier_M<M>_K<K>_J<J>_w<wf>.json (sigma_k, tuned f)
"""
import sys
import os
import json
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amp_trunc import AmpTrunc  # noqa: E402
import amp_flint as AF  # noqa: E402


class AmpHierModel(AmpTrunc):
    def __init__(self, wf, K, J, M, prec=320, Mf=4):
        super().__init__(wf, K, J, M + 1, prec=prec, Mf=Mf)
        self.Hodd = M

    def S_at(self, nu, k):
        """amp_flint.Amp.S_at with the de-mixing sum over odd mu only (see the module docstring)"""
        E, delta = self.theta_series(k)
        H = self.H
        nmax = k // 2 + 1
        tab = self.psi_tab(nu, nmax)
        hser = [mp.mpc(0)] * (k + 1)
        for kk, v in delta.items():
            if kk <= k:
                hser[kk] = AF.to_mp(v) / 2

        def smul(x, y):
            out = [mp.mpc(0)] * (k + 1)
            for i, a in enumerate(x):
                if a == 0:
                    continue
                for l in range(k + 1 - i):
                    if y[l] != 0:
                        out[i + l] += a * y[l]
            return out
        hp = [[mp.mpc(1)] + [mp.mpc(0)] * k]
        for n in range(1, nmax + 1):
            hp.append(smul(hp[-1], [-x for x in hser]))
        Emp = {kk: [AF.to_mp(x) for x in E[kk]] for kk in E}
        ys = []
        for idx, j in enumerate(range(self.jlo, self.J + 1)):
            base, ps = tab[idx]
            Lser = [mp.mpc(0)] * (k + 1)
            for n in range(1, nmax + 1):
                for q in range(k + 1):
                    Lser[q] -= ps[n - 1] * hp[n][q]
            Nser = [mp.mpc(1)] + [mp.mpc(0)] * k
            term = list(Nser)
            for n in range(1, k + 1):
                term = [x / n for x in smul(term, Lser)]
                if all(t == 0 for t in term):
                    break
                Nser = [x + y for x, y in zip(Nser, term)]
            zser = [mp.mpc(0)] * (k + 1)
            for kp in range(1, k + 1, 2):
                if kp not in self.d:
                    continue
                dj = self.d[kp][j]
                m = min(self.Hodd, kp)
                for kpp in range(0, k - kp + 1, 2):
                    if kpp not in Emp:
                        continue
                    Ek = Emp[kpp]
                    acc = mp.mpc(0)
                    for mu in range(-m, m + 1, 2):
                        e_ = Ek[H + nu - mu] if 0 <= H + nu - mu <= 2 * H else 0
                        if e_ != 0:
                            acc += e_ * AF.to_mp(dj[H + mu])
                    zser[kp + kpp] += acc
            ys.append(base * sum(zser[q] * Nser[k - q] for q in range(k + 1)))
        A, sc = self.basis(nu)
        y = mp.matrix(ys)
        c, res = mp.qr_solve(A, y)
        resid = res / mp.norm(y)
        return c[0] / sc[0], resid


if __name__ == "__main__":
    M, K, J = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    wf = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    a = AmpHierModel(wf, K, J, M)
    sig, f = a.run()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"trunc_hier_M{M}_K{K}_J{J}_w{wf:.4f}.json")
    json.dump({"M": M, "H": M + 1, "wf": wf, "K": K, "J": J,
               "sigma_str": [[mp.nstr(z.real, 20), mp.nstr(z.imag, 20)] for z in sig], "f": f}, open(out, "w"))
    print("wrote", out)
