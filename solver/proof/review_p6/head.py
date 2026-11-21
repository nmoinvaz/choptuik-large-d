"""The head j < j0 = max(2 nu, 10) of route 3's Darboux sum, re-done for the #1062 review (notes/P6_route3_review.md).

  --certify   H_mu(nu) = sum_{j<j0} |dR_j/K_nu| / |p| for odd nu = 3..99 from model.darboux() in mpmath.iv
              (the review's own recursion, written from amp_flint.py), and e^{-0.736 nu} H; the closed form G of B7
              at nu = 101, evaluated here from its statement. About 10 min.
  --check     B7 (from its statement) against |dR_j/K_nu| at every head term, odd nu = 3..21 (PC7); and the sum of B7
              over the head against G, odd nu = 3..41.
  --mutate=a0 M7 as fixed in the protocol, A0 / 100 in B7. It does not fire: B7's de-mixing and psi part alone stays
              above every head term (worst ratio 0.49), so --check still passes. Kept as a record.
  --mutate=b7 M7', the whole of B7 / 100; --check then fails, exit 1.

  --json=PATH writes the certified numbers, each end rounded outward.
"""
import sys
import json
import time
from mpmath import mp, iv
import model

SIGMA = mp.mpf('0.736')


def cabs(z):
    return iv.sqrt(z.real ** 2 + z.imag ** 2)


def j0_of(nu):
    return max(2 * nu, 10)


def H_iv(nu, mu, prec=200):
    C, w = model.ctx('iv', prec)
    J = j0_of(nu)
    D = model.darboux(nu, mu, J, C, w)
    tot = iv.mpf(0)
    worst_rel = mp.mpf(0)
    vals = []
    for z in D:
        a = cabs(z)
        vals.append(a)
        tot += a
        if a.a > 0:
            worst_rel = max(worst_rel, (a.b - a.a) / a.a)
    return tot, vals, worst_rel


class B7:
    """the bound of B7 from its statement (notes/P6_route3.md 11.2), in mpmath.iv"""

    def __init__(self, nu, mu, Psup=False, a0_scale=1, b7_scale=1):
        iv.prec = 200
        C, w = model.ctx('iv', 200)
        self.nu, self.mu, self.w = nu, mu, w
        w4 = w ** 2 / 4
        self.r = [iv.sqrt(((m + iv.mpf(1) / 2) ** 2 + w4) * ((m + 1) ** 2 + w4)) / (m + 1) for m in range(4 * nu + 30)]
        t1 = self.r[0]
        self.A0 = (2 + t1) ** 2 / 8 * a0_scale
        self.b7_scale = b7_scale
        k = mu - nu
        self.N, self.n = {-2: (3, 1), 0: (6, 2), 2: (3, 1)}[k]
        if k == 2:
            sh = lambda x: (iv.exp(x) - iv.exp(-x)) / 2
            self.P = iv.exp(iv.pi * w) if Psup else iv.sqrt(nu * sh(iv.pi * (nu + 2) * w) / ((nu + 2) * sh(iv.pi * nu * w)))
        else:
            self.P = iv.mpf(1)
        # de-mixing: per outer input, |E| = |sum dd/4| / (|h| w), times 2 w_r
        Dsum = iv.mpf(0)
        groups = {}
        for x1, x2, x3 in model.triples(C, nu, mu):
            h = x1[1] + x2[1]
            if h == 0:
                continue
            key = (x3[0], x3[1])
            groups[key] = groups.get(key, (iv.mpc(0, 0), h))
            groups[key] = (groups[key][0] + x1[2] * x2[2] / 4, h)
        for (name, lam), (dd4, h) in groups.items():
            E = cabs(dd4) / (abs(h) * w)
            wr = self.P if lam == nu + 2 else iv.mpf(1)
            Dsum += 2 * E * wr
        self.Ddemix = Dsum
        self.h2 = iv.mpf(1) / 4 if k == 0 else iv.mpf(0)
        self.a, self.b = model.ab(C, w, nu)

    def Sigma(self, j, lam):
        acc = iv.mpf(0)
        for i in range(1, j + 1):
            acc += iv.sqrt((2 * (j - i) + 1) ** 2 + (lam * self.w) ** 2) / i
        return acc

    def term(self, j):
        nu, w = self.nu, self.w
        Hn = sum(iv.mpf(1) / q for q in range(1, j + 2))
        v = self.N * Hn / self.r[j]
        if j >= 1:
            cj = iv.sqrt((2 * j + 2) ** 2 + (nu * w) ** 2) / (4 * (j + 1))
            v += cj * (self.n * self.Sigma(j, self.mu) + (self.N - self.n) * self.Sigma(j, 1)) / (self.r[j - 1] * self.r[j])
        Dj = self.Ddemix + self.h2 * (1 / cabs(self.a + j) + 1 / cabs(self.b + j))
        return (self.P * self.A0 * v + Dj) * self.b7_scale

    def G(self):
        """the closed form of 11.2, with P <= P_sup for mu = nu + 2 (this object built with Psup=True)"""
        nu, w = self.nu, self.w
        j0 = j0_of(nu)
        ell = 1 + iv.log(j0)
        T1 = 2 + iv.log(2 * j0 - 1)
        Sa = iv.mpf(2) / 3 * (1 + 2 * iv.log(2))
        Sb = iv.mpf(8) / 3 * (1 - iv.log(2))
        br = T1 + (nu + 2) * w + nu * w / 2 * Sa + nu * (nu + 2) * w ** 2 / 4 * Sb + 1 / self.r[0] + iv.log(2 * j0 - 1)
        Dbar = self.Ddemix + self.h2 * 4 / (nu * w)
        return self.P * self.A0 * self.N * ell * br + j0 * Dbar


def certify(out):
    t0 = time.time()
    res = {}
    for k in (-2, 0, 2):
        best, at = iv.mpf(0), None
        rows = {}
        for nu in range(3, 100, 2):
            H, vals, wr = H_iv(nu, nu + k)
            e = iv.exp(-iv.mpf('0.736') * nu) * H
            rows[nu] = (H, e, wr)
            if e.b > best.b:
                best, at = e, nu
            if nu in (3, 5, 11, 21, 41, 99):
                print(f"mu = nu{k:+d}, nu = {nu}: H in [{model.dn(H, 8)}, {model.up(H, 8)}], e^(-0.736 nu) H <= "
                      f"{model.up(e, 6)}, max relative radius {mp.nstr(wr, 2)} [{time.time() - t0:.0f}s]", flush=True)
        g = B7(101, 101 + k, Psup=True).G()
        eg = iv.exp(-iv.mpf('0.736') * 101) * g
        lam_head = max(best.b, eg.b)
        print(f"mu = nu{k:+d}: sup_nu<=99 e^(-0.736 nu) H <= {model.up(best, 8)} at nu = {at}; e^(-0.736*101) G(101) <= "
              f"{model.up(eg, 3)}; Lambda_head <= {model.up(lam_head, 8)}", flush=True)
        res[k] = {'Lambda_head': model.up(lam_head, 45), 'at': at, 'eG101': model.up(eg, 45),
                  'H': {str(n): model.ivjson(v[0]) for n, v in rows.items()}}
    if out:
        json.dump(res, open(out, 'w'), indent=1)
    return True


def check(a0_scale, b7_scale=1):
    t0 = time.time()
    ok = True
    for k in (-2, 0, 2):
        worst, wat = mp.mpf(0), None
        for nu in range(3, 22, 2):
            mu = nu + k
            C, w = model.ctx('iv', 200)
            D = model.darboux(nu, mu, j0_of(nu), C, w)
            B = B7(nu, mu, a0_scale=a0_scale, b7_scale=b7_scale)
            for jj, z in enumerate(D):
                a = cabs(z)
                b = B.term(jj)
                if not (a.b <= b.a):
                    ok = False
                rat = a.b / b.a
                if rat > worst:
                    worst, wat = rat, (nu, jj)
        print(f"B7, mu = nu{k:+d}: worst |dR_j/K|/B7 <= {model.up(worst, 4)} at (nu, j) = {wat}"
              f" [{time.time() - t0:.0f}s]", flush=True)
        gw = mp.mpf(0)
        for nu in range(3, 42, 2):
            B = B7(nu, nu + k, a0_scale=a0_scale)
            s = sum((B.term(jj) for jj in range(j0_of(nu))), iv.mpf(0))
            Bs = B7(nu, nu + k, Psup=True, a0_scale=a0_scale)
            g = Bs.G()
            if not (s.b <= g.a):
                ok = False
            gw = max(gw, s.b / g.a)
        print(f"  sum of B7 over the head / G, odd nu = 3..41: worst {model.up(gw, 4)}", flush=True)
    print(f"head.py --check {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    out = None
    for a in sys.argv[1:]:
        if a.startswith('--json='):
            out = a.split('=', 1)[1]
    a0 = mp.mpf(1) / 100 if '--mutate=a0' in sys.argv else 1
    b7 = mp.mpf(1) / 100 if '--mutate=b7' in sys.argv else 1
    ok = True
    if '--check' in sys.argv:
        ok &= check(a0, b7)
    if '--certify' in sys.argv:
        ok &= certify(out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
