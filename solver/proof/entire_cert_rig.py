#!/usr/bin/env python3
"""A Rouche certificate for one zero of the amplitude series, conditional on the hypotheses of proof/P5_scope.md.

S_1/S_1^(1)(x) = sum_k sigma_k x^k (x = A^2) is given through order K as acb balls (H_sigma: the fitted coefficients
with their fit errors as radii, which is a hypothesis, not an enclosure). G = e^{-kbar x} S_1 has the same zeros, and
its coefficients beyond K are assumed to obey |g_k| <= C r^k / Gamma(k + b) (H_T), with C, r, b inputs. On the circle
|x - x0| = rho, G_K (the truncation) is evaluated on arcs as balls, the tail sum_(k>K) C (r R)^k / Gamma(k + b) at
R = |x0| + rho is bounded in arb, and if min |G_K| > tail on the circle, Rouche gives G exactly as many zeros inside
as G_K has, which acb_poly.roots() isolates rigorously (acb_poly_validate_roots). One zero inside, and the zero of
S_1 in the disc is unique and simple.

What the code proves, given (H_sigma) and (H_T): at each scanned w, S_1(., w) has exactly one zero in |x - x0| < rho.
It proves nothing across w (H_omega is not implemented), so no real zero in (A, w) is certified here. Every run is
conditional on its inputs; every check can fail (the mutations below).

    python3 entire_cert_rig.py planted [--mutate=disc|rho|tail|coef]
        planted truth S(x) = e^{k1 x} + c e^{k2 x} with a zero placed at x = 7.4: the certificate must pass with the
        exact type bound, and must fail with the disc moved (no zero), rho = 7 (several zeros), the tail constant
        10^5 times larger, or the last coefficient unknown (radius |sigma_K|).
    python3 entire_cert_rig.py real DIR K C r b [--margin=2] [--rho=1e-3] [--arcs=720]
        the scans in DIR (one JSON per w/w_inf), truncated at K, with (H_T)'s constants C, r, b from
        entire_cert.py section 6 (C is multiplied by --margin), kbar from entire_cert.best_kbar.
"""
import glob
import json
import os
import sys

from flint import acb, arb, acb_poly, ctx

ctx.prec = 300
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def tail_bound(C, r, b, K, R):
    """an arb upper bound of sum_(k>K) C (r R)^k / Gamma(k + b); the terms decrease geometrically past N"""
    C, q0, b = arb(C), arb(r) * arb(R), arb(b)
    N = K + 300
    s = arb(0)
    term = None
    for k in range(K + 1, N + 1):
        term = C * q0 ** k / arb(k + b).gamma()
        s += term
    q = q0 / arb(N + 1 + b)
    if not (q < 1):
        raise ValueError("the tail terms do not decrease yet at N; raise N")
    s += term * q / (1 - q)
    return s.upper()


def gcoef(sig, kbar, K):
    """the balls g_k = sum_(m<=k) sigma_m (-kbar)^(k-m) / (k-m)!"""
    mk = -kbar
    return [sum((sig[m] * mk ** (k - m) / arb(k - m + 1).gamma() for m in range(k + 1)), acb(0)) for k in range(K + 1)]


def circle_min(p, x0, rho, arcs):
    """a lower bound of |P| on |x - x0| = rho for the exact polynomial P, from arcs of the circle as balls"""
    half = arb.pi() / arcs
    lo, wid = None, 0.0
    for j in range(arcs):
        th = arb((2 * j + 1) * half.mid(), half.upper())
        x = x0 + rho * (acb(0, 1) * acb(th)).exp()
        v = p(x)
        m = v.abs_lower()
        wid = max(wid, float(v.rad()))
        lo = m if lo is None or m < lo else lo
    return lo, wid


def roots_inside(p, x0, rho):
    """the isolated roots of the exact polynomial P with |root - x0| < rho; raises if a root ball straddles"""
    rts = p.roots()
    inside = []
    for z in rts:
        d = (z - x0).abs_upper(), (z - x0).abs_lower()
        if d[0] < rho:
            inside.append(z)
        elif not (d[1] > rho):
            raise ValueError(f"a root ball straddles the circle: {z}")
    return inside, rts


def certify(sig, K, kbar, C, r, b, x0, rho, arcs=720, label=""):
    """Rouche twice on |x - x0| = rho: P (the midpoints of the g_k, exact) against G_K (the balls), whose distance
    is at most delta = sum_k rad(g_k) R^k, and against G, whose distance is at most delta + tail. If min |P| on the
    circle exceeds delta + tail, G has as many zeros inside as P, which roots() isolates."""
    g = gcoef(sig[:K + 1], kbar, K)
    p = acb_poly([acb(z.real.mid(), z.imag.mid()) for z in g])
    R = x0.abs_upper() + rho
    delta = sum(((arb(z.real.rad()) ** 2 + arb(z.imag.rad()) ** 2).sqrt() * R ** k for k, z in enumerate(g)), arb(0)).upper()
    tail = tail_bound(C, r, b, K, R)
    lo, wid = circle_min(p, x0, rho, arcs)
    inside, rts = roots_inside(p, x0, rho)
    ok = bool(lo > delta + tail) and len(inside) == 1
    print(f"{label}disc |x - x0| < {rho} at x0 = {x0.str(8)}: min |P| on the circle >= {lo.str(3)}"
          f" (widest evaluation {wid:.1e}), coefficient radii delta <= {delta.str(2)}, tail <= {tail.str(3)},"
          f" (delta + tail) / min = {float((delta + tail) / lo) if lo > 0 else float('inf'):.2e};"
          f" roots of P inside: {len(inside)}" + (f" at {inside[0].str(10)}" if len(inside) == 1 else "")
          + f"; verdict {'CERTIFIED (given H_sigma, H_T)' if ok else 'NOT certified'}")
    return ok, inside


def planted(mutate=None):
    """S(x) = e^{k1 x} + c e^{k2 x}: sigma_k = (k1^k + c k2^k) / k!, zero at x = 7.4 by the choice of c, exact type
    max(|k1|, |k2|); in G form |g_k| <= (1 + |c|) r^k / k! with r = max |k_i - kbar|"""
    k1, k2 = acb(1.2, 1.0), acb(0.7, 0.2)
    xz = arb(7.4)
    c = -((k1 - k2) * xz).exp()
    K = 17
    sig = [(k1 ** k + c * k2 ** k) / arb(k + 1).gamma() for k in range(K + 1)]
    sig = [acb(arb(s.real.mid(), abs(s.real.mid()) * 1e-12 + 1e-40), arb(s.imag.mid(), abs(s.imag.mid()) * 1e-12 + 1e-40))
           for s in sig]                       # the fit-error model: 1e-12 relative radii
    kbar = (k1 + k2) / 2
    r = max(float((k1 - kbar).abs_upper()), float((k2 - kbar).abs_upper()))
    C, b = 1 + float(c.abs_upper()), 1.0
    x0, rho = acb(xz), 0.1
    if mutate == "disc":
        x0 = acb(xz + 3)
    elif mutate == "rho":
        rho = 7.0
    elif mutate == "tail":
        C *= 1e5
    elif mutate == "coef":
        sig[K] = acb(arb(sig[K].real.mid(), abs(sig[K])), arb(sig[K].imag.mid(), abs(sig[K])))
    print(f"planted: k1 {k1.str(3)}, k2 {k2.str(3)}, c {c.str(4)}, zero at {xz}, K {K}, kbar {kbar.str(3)},"
          f" r {r:.4f}, C {C:.2f}, b {b}" + (f", mutate {mutate}" if mutate else ""))
    ok, inside = certify(sig, K, kbar, C, r, b, x0, rho, label="   ")
    want = mutate is None
    if ok and want:
        print(f"   the disc is centred on the planted zero, so the certified zero is it; P's root {inside[0].str(12)}")
    print("PASS" if ok == want else "FAIL", f"(expected {'certified' if want else 'not certified'})")
    return ok == want


def load_balls(fn):
    rec = json.load(open(fn))
    sig = []
    for (re, im), err in zip(rec["sigma_str"], rec["sigma_err"]):
        e = float(err) if err not in (None, "inf") and float(err) < float("inf") else None
        if e is None:
            raise ValueError(f"{fn}: a coefficient with no error estimate")
        sig.append(acb(arb(re, e), arb(im, e)))
    return rec["wf"], sig


def real(d, K, C, r, b, margin=2.0, rho=1e-3, arcs=720):
    import mpmath as mp
    import entire_cert as EC
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    scans = dict(load_balls(fn) for fn in files)
    EC.SCAN = os.path.abspath(d)
    kb, last = EC.best_kbar(EC.load(1.0), K)
    kbar = acb(str(mp.re(kb)), str(mp.im(kb)))
    print(f"real: {d}, K {K}, kbar {kbar.str(5)} (entire_cert.best_kbar at w_inf), (H_T) C {C} x margin {margin},"
          f" r {r}, b {b}; coefficient radii = the fit errors (H_sigma); {arcs} arcs")
    allok = True
    for wf in sorted(scans):
        sig = scans[wf]
        g = gcoef(sig[:K + 1], kbar, K)
        rts = acb_poly([acb(z.real.mid(), z.imag.mid()) for z in g]).roots()
        x0 = min(rts, key=lambda z: float((z - 7.4).abs_lower()))
        x0 = acb(x0.real.mid(), x0.imag.mid())
        ok, _ = certify(sig, K, kbar, C * margin, r, b, x0, rho, arcs, label=f"   w/w_inf {wf}: ")
        allok &= ok
    print("all certified (given H_sigma, H_T)" if allok else "NOT all certified")
    return allok


def main():
    a = sys.argv[1:]
    opts = {k[2:].split("=")[0]: (k.split("=")[1] if "=" in k else True) for k in a if k.startswith("--")}
    a = [k for k in a if not k.startswith("--")]
    if a[0] == "planted":
        sys.exit(0 if planted(opts.get("mutate")) else 1)
    if a[0] == "real":
        d, K, C, r, b = a[1], int(a[2]), float(a[3]), float(a[4]), float(a[5])
        ok = real(d, K, C, r, b, float(opts.get("margin", 2)), float(opts.get("rho", 1e-3)), int(opts.get("arcs", 720)))
        sys.exit(0 if ok else 1)
    print(__doc__)
    sys.exit(2)


if __name__ == "__main__":
    main()
