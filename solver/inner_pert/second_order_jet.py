"""Jets of the inner problem with its O(1/n) and O(1/n^2) corrections (notes/SECOND_ORDER.md, section 3): the
M-harmonic lateral integration of first_order_jet.py, extended to second order in e = 1/n.

Equations (second_order_equations.py), L = xi^2 d_xi + xi (omega d_theta + lambda), E = y^2/2,
Psi = sigma^2/2 - xi y y_xi, Phi = -xi Psi_xi - (1 - E) xi E_xi:
    L y      = c sigma + e c (xi sigma_xi - sigma + E sigma) + e^2 c Psi sigma
    L sigma  = (xi d_xi - 1)(c y)
    xi c_xi  = c (E + e Psi + e^2 Phi)

Second order by polarization. Every direction with q1 = True is a nilpotent unit e_i, and e = sum_i e_i. With two of
them, e^2 = 2 e_1 e_2, so a field Z_0 + e Z_1 + e^2 Z_2 has components Z_1 on e_1 and on e_2 and 2 Z_2 on e_1 e_2:
the mixed component holds twice the e^2 coefficient, of every field, of the forcing and of the jump. Data on the
mixed component (profile, c(0), omega, lambda) are therefore entered doubled. The O(e) terms are shifted by every
e_i; the O(e^2) terms by every pair e_i e_j (i < j), times 2.

Further features (beyond first_order_jet.JetModel):
    wextra   {mask: value}: omega jet entries on composite masks (the mixed component's 2 omega_2);
    lamj     {mask: value}: a lambda jet for the perturbation components, lambda = lam* + sum_m value_m mask_m
             (lambda_1 on e_1 and e_2, 2 lambda_2 on e_1 e_2); acts on every component that carries a perturbation
             direction (a 'pert...' direction or one with lam != 0).
The O(e^2) terms need y_xi xi, sigma_xi and y_xixi at leading order (in Phi) and the O(e) terms the derivatives to
O(e): the right-hand side solves for them by a fixed-point pass per order of e (exact for jets). Every product is the
same truncated product, in the same order, as in the series recursion, so the series is the model's formal solution.

    from second_order_jet import JetModel2
"""
import itertools

import numpy as np

from first_order_jet import JetModel, Dir, W0, Fh_from, dFh, F2h, demix_matrix, ri  # noqa: F401

MUTATE = set()   # 'eps2y': drop the O(e^2) term of the y equation; 'phi': (1 - E) -> 1 in Phi; 'pol': e^2 = e_1 e_2


class JetModel2(JetModel):
    def __init__(self, M, w, Fh, dirs, masks, data, wextra=None, lamj=None):
        assert not any(d.force is not None for d in dirs), "SymForce is not supported here"
        super().__init__(M, w, Fh, dirs, masks, data)
        self.qbits = [1 << b for b, d in enumerate(dirs) if d.q1]
        self.qmaps = [self._shiftmap(qb) for qb in self.qbits]
        self.q2maps = []
        for qa, qb in itertools.combinations(self.qbits, 2):
            both = qa | qb
            self.q2maps.append([(i, self.idx[m | both]) for i, m in enumerate(self.masks)
                                if not m & both and (m | both) in self.idx])
        self.has_q = bool(self.qbits)
        self.npass = len(self.qbits) + 1          # one pass per order of e: exact for the nilpotent jets
        for m, v in (wextra or {}).items():
            if m in self.idx:
                self.wj[self.idx[m]] = self.wj.get(self.idx[m], 0) + v
        # lambda jet on the perturbation components
        self.lamjl = []
        for m, v in (lamj or {}).items():
            for i, mi in enumerate(self.masks):
                if mi & self.pertbits and not mi & m and (mi | m) in self.idx:
                    self.lamjl.append((i, self.idx[mi | m], v))

    # ---- e and e^2 multiplication of jet fields
    def eps(self, a):
        out = np.zeros_like(a)
        for smap in self.qmaps:
            for i, j in smap:
                out[j] += a[i]
        return out

    def eps2(self, a):
        out = np.zeros_like(a)
        for smap in self.q2maps:
            for i, j in smap:
                out[j] += (1 if "pol" in MUTATE else 2) * a[i]
        return out

    def lmul(self, a):
        out = super().lmul(a)
        for i, j, v in self.lamjl:
            out[j] += v * a[i]
        return out

    # ---- formal series at xi0 (scaled as in first_order_jet: D_j = d_j xi0^(2j), ...)
    def series_at(self, xi0, J, stop=True):
        q = xi0 * xi0
        o, e = self.odd, self.even
        Z = np.zeros_like(self.d0)
        D, S, G, E, PS, PHI, OM = [self.d0.copy()], [], [self.g0.copy()], [Z], [Z], [Z], []
        PSS = []                                   # [Psi sigma]_k
        mags = [np.abs(D[0][0]).max()]
        y2 = 0.0 if "eps2y" in MUTATE else 1.0
        for j in range(J):
            R = (2 * j + 1) * D[j] + self.lmul(D[j]) + self.wmul(D[j])
            for i in range(1, j + 1):
                R = R - self.mul(G[i], S[j - i], o)
            if self.has_q:
                t = Z.copy()
                for m in range(1, j + 1):
                    t = t + self.mul(PS[m], S[j - m], o)
                PSS.append(t)
                Y2 = Z.copy()
                for i in range(0, j + 1):
                    Y2 = Y2 + self.mul(G[i], PSS[j - i], o)
                R = R - y2 * self.eps2(Y2)
            sj = self.ginv_solve(G[0], R * o, o)
            if self.has_q:
                for _ in range(self.npass):
                    Om_j = (2 * j + 1) * sj
                    for m in range(1, j + 1):
                        Om_j = Om_j + self.mul(E[m], S[j - m], o)
                    OMj = OM + [Om_j]
                    Qy = Z.copy()
                    for i in range(0, j + 1):
                        Qy = Qy + self.mul(G[i], OMj[j - i], o)
                    sj = self.ginv_solve(G[0], (R - self.eps(Qy)) * o, o)
                Om_j = (2 * j + 1) * sj
                for m in range(1, j + 1):
                    Om_j = Om_j + self.mul(E[m], S[j - m], o)
                OM.append(Om_j)
            S.append(sj * o)
            m = j + 1
            em = Z.copy()
            for i in range(m):
                em = em + self.mul(D[i], D[m - 1 - i], e)
            em = q * em / 2
            E.append(em)
            gm = Z.copy()
            for i in range(m):
                gm = gm + self.mul(G[i], E[m - i], e)
            if self.has_q:
                ss = Z.copy()
                for i in range(0, m - 1):
                    ss = ss + self.mul(S[i], S[m - 2 - i], e)
                pm = -2 * m * em + q * q * ss / 2
                PS.append(pm)
                one = 0.0 if "phi" in MUTATE else 1.0
                ph = -2 * m * pm - 2 * m * em
                for i in range(1, m):
                    ph = ph + one * self.mul(E[i], 2 * (m - i) * E[m - i], e)
                PHI.append(ph)
                cp, cf = Z.copy(), Z.copy()
                for i in range(m):
                    cp = cp + self.mul(G[i], PS[m - i], e)
                    cf = cf + self.mul(G[i], PHI[m - i], e)
                gm = gm + self.eps(cp) + self.eps2(cf)
            G.append(gm / (2 * m))
            num = q * ((2 * j + 2) * S[j] + self.lmul(S[j]) + self.wmul(S[j]))
            for i in range(1, j + 2):
                num = num - (2 * j + 2) * self.mul(G[i], D[j + 1 - i], o)
            dn = self.ginv_solve(G[0], num * o / (2 * j + 2), o)
            D.append(dn * o)
            mags.append(np.abs(D[-1][0]).max())
            if stop and j > 10 and mags[-1] > 10 * min(mags):
                break
        jt = int(np.argmin(mags[1:])) + 1
        Dd = xi0 * sum(D[:jt]); Sg = q * sum(S[:jt]); Cc = sum(G[:jt])
        self.ser = dict(D=D, S=S, G=G, E=E, PS=PS, PHI=PHI, jt=jt, mags=mags)
        return self.pack(Dd, Sg, Cc), jt, mags[jt] / mags[0]

    def rhs(self, xi, Y):
        o, e = self.odd, self.even
        D, S, C = self.unpack(Y)
        TT = lambda X: self.wmul(X) + self.lmul(X)            # omega d_theta + lambda
        trans = lambda X: xi * TT(X)
        E = self.mul(D, D, e) / 2
        cE = self.mul(C, E, e)
        lead = self.mul(C, S, o) - trans(D)
        dD = lead / (xi * xi)
        if not self.has_q:
            r2 = self.mul(C, xi * dD, o) + self.mul(cE, D, o) - self.mul(C, D, o)
            dS = (r2 - trans(S)) / (xi * xi)
            return self.pack(dD, dS * o, cE / xi)
        xdD = xi * dD
        # leading-order derivatives for Phi (it enters at e^2 only)
        dS0 = (self.mul(C, xdD, o) + self.mul(cE, D, o) - self.mul(C, D, o) - trans(S)) / (xi * xi)
        dC0 = cE / xi
        xxD = self.mul(dC0, S, o) + self.mul(C, dS0, o) - TT(D) - trans(dD) - 2 * xdD   # xi^2 y_xixi
        xdE = self.mul(D, xdD, e)                                                       # xi E_xi
        xdPsi = self.mul(S, xi * dS0, e) - self.mul(xdD, xdD, e) - self.mul(D, xdD + xxD, e)
        one = 0.0 if "phi" in MUTATE else 1.0
        Phi = -xdPsi - xdE + one * self.mul(E, xdE, e)
        cPhi = self.mul(C, Phi, e)
        y2 = 0.0 if "eps2y" in MUTATE else 1.0
        for _ in range(self.npass):
            Psi = self.mul(S, S, e) / 2 - self.mul(D, xdD, e)
            cU = cE + self.eps(self.mul(C, Psi, e)) + self.eps2(cPhi)
            dS = (self.mul(C, xdD, o) + self.mul(cU, D, o) - self.mul(C, D, o) - trans(S)) / (xi * xi)
            Om = xi * dS - S + self.mul(E, S, o)
            dD = (lead + self.eps(self.mul(C, Om, o)) + y2 * self.eps2(self.mul(C, self.mul(Psi, S, o), o))) / (xi * xi)
            xdD = xi * dD
        Psi = self.mul(S, S, e) / 2 - self.mul(D, xdD, e)
        cU = cE + self.eps(self.mul(C, Psi, e)) + self.eps2(cPhi)
        dS = (self.mul(C, xdD, o) + self.mul(cU, D, o) - self.mul(C, D, o) - trans(S)) / (xi * xi)
        return self.pack(dD, dS * o, cU / xi)
