"""Orders and digits a large-D discretization needs, from the asymptotic structure (#1086,
notes/LARGE_D_BASICS.md section 4).

For n = D - 2 this prints
  sigma_R     the real part of the SSH Frobenius exponent of p, (n/2 - 1 + g)/(1 - g), with g = kappa/n
              (kappa = n <(n-1)(a_h^2 - 1)> = 7.3 from the stored states near D 260; --kappa)
  ksA(eps)    the smallest horizon order whose mass-mode tail sum_{k>ks} C(n-2+k, k) u^k (1-u)^(n-1) at the
              last node u = 1 - x_K is below eps (the mass mode x^-(n-1) = (1-u)^-(n-1) is in every Jacobian column
              of a_h)
  G digits    log10 x_K^-(n-1), the mass mode's amplification from the horizon to the last node
  F digits    log10 of the fast scalar mode's decay from the first node to the last, F = x^-n (1 - x^2)^(n/2)
  kt step     the largest Taylor step h at x = x_1 for order kt with (n lam h)^(kt+1)/(kt+1)! < eps,
              n lam = n c^2 / (x (c^2 - x^2)) (c = 1), the fast mode's local rate

With --tails it prints instead the mass mode's relative tail sum_{k>ks} C(n-2+k, k) u^k (1-u)^(n-1) at the
horizon orders and D given (--ks=130,293,315 --Ds=150,200,262.01).

Usage: python3 order_laws.py [--u=0.3] [--x1=0.3] [--kt=36] [--kappa=7.0] [--mutate=binom]
       python3 order_laws.py --tails [--ks=...] [--Ds=...] [--u=0.3]
  --mutate=binom: control; the mass-mode tail with C(n-2+k, k) replaced by C(k + 10, k) (an n-independent mode),
  which must make ksA flat in n.
"""
import sys

import mpmath as mp

mp.mp.dps = 60


def tail_order(n, u, eps, mutate):
    m = 10 if mutate == "binom" else n - 2
    lw = mp.log(1 - u) * (m + 1)  # normalization (1-u)^(m+1) of the negative binomial
    total = mp.mpf(0)
    terms = []
    k = 0
    # sum backwards is not needed: accumulate the tail from a large K down
    K = int(10 * (m + 50))
    lt = [mp.loggamma(m + 1 + k) - mp.loggamma(k + 1) - mp.loggamma(m + 1) + k * mp.log(u) + lw for k in range(K)]
    tail = mp.mpf(0)
    for k in range(K - 1, -1, -1):
        tail += mp.e ** lt[k]
        if tail > eps:
            return k  # sum_{j>=k} > eps, so the order must be at least k
    return 0


def tail(n, u, ks, mutate=""):
    m = 10 if mutate == "binom" else n - 2
    u = mp.mpf(u)
    lw = (m + 1) * mp.log(1 - u)
    return mp.nsum(lambda k: mp.e ** (mp.loggamma(m + 1 + k) - mp.loggamma(k + 1) - mp.loggamma(m + 1)
                                       + k * mp.log(u) + lw), [ks + 1, mp.inf])


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if "--tails" in sys.argv:
        kss = [int(k) for k in opts.get("ks", "130,293,315,330,360,420").split(",")]
        Ds = [mp.mpf(d) for d in opts.get("Ds", "79.77,100,150,175,200,232.63,262.01,265.08,300").split(",")]
        print("# D " + " ".join("tail(ks %d)" % k for k in kss))
        for D in Ds:
            row = [mp.nstr(tail(D - 2, opts.get("u", 0.3), k, opts.get("mutate", "")), 3) for k in kss]
            print("%s " % mp.nstr(D, 6) + " ".join(row))
        return
    u = float(opts.get("u", 0.3))
    x1 = float(opts.get("x1", 0.3))
    kt = int(opts.get("kt", 36))
    kappa = float(opts.get("kappa", 7.0))
    mutate = opts.get("mutate", "")
    xK = 1 - u
    print("# n D sigma_R ksA(2^-212) ksA(2^-424) G_digits F_digits(x1->xK) kt_step(2^-106) kt_step(2^-424)")
    for n in [60, 120, 180, 240, 258, 263, 300, 400, 600, 1000]:
        g = kappa / n
        sig = (n / 2 - 1 + g) / (1 - g)
        ka = tail_order(n, u, mp.mpf(2) ** -212, mutate)
        kb = tail_order(n, u, mp.mpf(2) ** -424, mutate)
        Gd = -(n - 1) * mp.log10(xK)
        Fd = n * (mp.log10(mp.sqrt(1 - x1 ** 2) / x1) - mp.log10(mp.sqrt(1 - xK ** 2) / xK))
        lam = n / (x1 * (1 - x1 ** 2))
        steps = []
        for eps in (mp.mpf(2) ** -106, mp.mpf(2) ** -424):
            # (lam h)^(kt+1)/(kt+1)! = eps
            h = mp.e ** ((mp.log(eps) + mp.loggamma(kt + 2)) / (kt + 1)) / lam
            steps.append(h)
        print("%d %d %.2f %d %d %.1f %.1f %.2e %.2e" % (n, n + 2, sig, ka, kb, Gd, Fd, steps[0], steps[1]))


if __name__ == "__main__":
    main()
