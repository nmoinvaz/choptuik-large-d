"""Formal inner series of the M = 1 truncated system (exactly the system lateral_np.py integrates: Dd, sigma on odd
|mu| <= M, c on even |nu| <= M + 1, every product projected), in ball arithmetic (python-flint acb), and the rigorous
residual of the truncated series in the ODE. A, w may be balls (parameter boxes)."""
from flint import acb, arb, ctx


class SeriesArb:
    def __init__(self, Fh, w, M=1, prec=256):
        ctx.prec = prec
        self.w, self.M = acb(w), M
        self.odd = [mu for mu in range(-M, M + 1) if mu % 2]
        self.even = [nu for nu in range(-(M + 1), M + 2) if nu % 2 == 0]
        self.Fh = {mu: acb(Fh.get(mu, 0)) for mu in self.odd}
        self.I = acb(0, 1)

    def mul(self, A, B, keep):
        out = {k: acb(0) for k in keep}
        for i, ai in A.items():
            for j, bj in B.items():
                if i + j in out:
                    out[i + j] += ai * bj
        return out

    def series(self, J):
        w, odd, even, I = self.w, self.odd, self.even, self.I
        d = [{mu: 2 * self.Fh[mu] for mu in odd}]
        s, g = [], [{nu: acb(1 if nu == 0 else 0) for nu in even}]
        e = [{nu: acb(0) for nu in even}]
        ce = [{nu: acb(0) for nu in even}]
        for j in range(J):
            sj = {mu: (2 * j + 1 + I * mu * w) * d[j][mu] for mu in odd}
            for i in range(1, j + 1):
                t = self.mul(g[i], s[j - i], odd)
                for mu in odd:
                    sj[mu] -= t[mu]
            s.append(sj)
            m = j + 1
            em = {nu: acb(0) for nu in even}
            for i in range(m):
                t = self.mul(d[i], d[m - 1 - i], even)
                for nu in even:
                    em[nu] += t[nu] / 2
            e.append(em)
            cm = {nu: acb(0) for nu in even}
            for i in range(m):
                t = self.mul(g[i], e[m - i], even)
                for nu in even:
                    cm[nu] += t[nu]
            ce.append(cm)
            g.append({nu: cm[nu] / (2 * m) for nu in even})
            num = {mu: (2 * j + 2 + I * mu * w) * s[j][mu] for mu in odd}
            for i in range(1, j + 2):
                t = self.mul(g[i], d[j + 1 - i], odd)
                for mu in odd:
                    num[mu] -= 2 * (j + 1 - i) * t[mu]
            for mm in range(1, j + 2):
                t = self.mul(ce[mm], d[j + 1 - mm], odd)
                for mu in odd:
                    num[mu] -= t[mu]
            d.append({mu: num[mu] / (2 * j + 2) for mu in odd})
        return d, s, g

    def truncated(self, d, s, g, xi, n):
        """sums to order n-1 and their xi-derivatives, at the point xi (acb)"""
        D = {mu: sum((d[j][mu] * xi ** (2 * j + 1) for j in range(n)), acb(0)) for mu in self.odd}
        S = {mu: sum((s[j][mu] * xi ** (2 * j + 2) for j in range(n)), acb(0)) for mu in self.odd}
        C = {nu: sum((g[j][nu] * xi ** (2 * j) for j in range(n)), acb(0)) for nu in self.even}
        dD = {mu: sum(((2 * j + 1) * d[j][mu] * xi ** (2 * j) for j in range(n)), acb(0)) for mu in self.odd}
        dS = {mu: sum(((2 * j + 2) * s[j][mu] * xi ** (2 * j + 1) for j in range(n)), acb(0)) for mu in self.odd}
        dC = {nu: sum(((2 * j) * g[j][nu] * xi ** (2 * j - 1) for j in range(1, n)), acb(0)) for nu in self.even}
        return D, S, C, dD, dS, dC

    def residual(self, D, S, C, dD, dS, dC, xi):
        """ODE residuals (I1)-(I3) of the given values (rigorous balls)"""
        w, odd, even, I = self.w, self.odd, self.even, self.I
        cs = self.mul(C, S, odd)
        r1 = {mu: xi * xi * dD[mu] + I * mu * w * xi * D[mu] - cs[mu] for mu in odd}
        E = {k: v / 2 for k, v in self.mul(D, D, even).items()}
        cE = self.mul(C, E, even)
        cEd = self.mul(cE, D, odd)
        cd = self.mul(C, D, odd)
        cxdD = self.mul(C, {mu: xi * dD[mu] for mu in odd}, odd)
        r2 = {mu: xi * xi * dS[mu] + I * mu * w * xi * S[mu] - (cxdD[mu] + cEd[mu] - cd[mu]) for mu in odd}
        r3 = {nu: xi * dC[nu] - cE[nu] for nu in even}
        return r1, r2, r3


def demo(A="2.5421480475", wf="0.7769940136", r0="0.1", J=80):
    ctx.prec = 256
    W0 = 2 * arb.pi() / arb("1.64389375")
    w = W0 * arb(wf)
    Fh = {1: acb(0, -arb(A) / 2), -1: acb(0, arb(A) / 2)}
    sa = SeriesArb(Fh, w)
    d, s, g = sa.series(J)
    phi = 3 * arb.pi() / 8
    xi = acb(arb(r0) * phi.cos(), arb(r0) * phi.sin())
    x2 = abs(xi * xi)
    mags = [max(abs(d[j][mu]) for mu in sa.odd) * x2 ** j for j in range(J)]
    jt = min(range(1, J), key=lambda j: float(mags[j].mid()))
    vals = sa.truncated(d, s, g, xi, jt)
    r1, r2, r3 = sa.residual(*vals, xi)
    res = max(float(abs(v).upper()) for dct in (r1, r2, r3) for v in dct.values())
    print(f"r0 {r0}: optimal truncation at j = {jt}, smallest term {float(mags[jt].mid()):.2e}; "
          f"rigorous ODE residual of the truncated series at xi0: {res:.2e}; "
          f"max coefficient radius {max(float(d[j][1].rad()) for j in range(jt)):.1e}")
    return sa, d, s, g, jt, vals


if __name__ == "__main__":
    for r0 in ("0.1", "0.08", "0.06"):
        demo(r0=r0, J=int(1 / (2 * float(r0) ** 2)) + 30)
