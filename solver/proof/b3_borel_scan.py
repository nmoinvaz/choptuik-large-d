"""Borel-plane scan of the high-precision outer series written by b3_outer_m1.py (proof/B3_borel.md, Section 5).

For a series sum_{k>=1} c_k eps^k it forms the Borel transform B(zeta) = sum_k c_k zeta^(k-1)/(k-1)! and reports
  - the root-test radius |b_k|^(-1/k) of B at the last orders (the modulus of the nearest Borel singularity),
  - the poles of the diagonal Pade approximant [L/L] of B that have no Pade zero within relative distance --stab
    (default 1e-6; Froissart doublets), each with its distance to the nearest pole of [L-2/L-2] ('move': isolated
    poles are stable, poles on a branch cut move). Branch points show as the near end of a line of poles.
Everything is done in mpmath at --dps digits (default 120); the coefficients' ball radii are checked first and orders
whose relative radius exceeds 10^-(dps/3) are dropped.

    python3 b3_borel_scan.py file.json QUANTITY [--x=0.5] [--n=8] [--dps=120] [--K=...]
      QUANTITY: g0 (G_k(0)), xstar (centre gauge X*), D, S, G, B (at --x), taylorD3, taylorD5, taylorG0, taylorG2
"""
import json
import sys

import mpmath as mp

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
POS = [a for a in sys.argv[1:] if not a.startswith("--")]
FN, QTY = POS[0], POS[1]
mp.mp.dps = int(ARGS.get("dps", 120))
NSHOW = int(ARGS.get("n", 8))
STAB = float(ARGS.get("stab", 1e-6))


def load(fn, qty, x=None):
    d = json.load(open(fn))
    if qty in ("g0", "g1", "xstar"):
        raw = d[qty]
    elif qty.startswith("taylorD"):
        raw = d["taylorD"][qty[7:]]
    elif qty.startswith("taylorG"):
        raw = d["taylorG"][qty[7:]]
    else:
        key = [k for k in d["series"] if abs(float(k) - float(x)) < 1e-12][0]
        raw = d["series"][key][qty]
    cs, rads = [], []
    for re, im, r in raw:
        z = mp.mpc(mp.mpf(re), mp.mpf(im))
        cs.append(z)
        rads.append(r)
    return d, cs, rads


def clean(cs, rads):
    """drop trailing orders whose relative ball radius is too large; leading zeros are kept (they are exact)."""
    tol = mp.mpf(10) ** (-mp.mp.dps // 3)
    last = len(cs)
    for k, (c, r) in enumerate(zip(cs, rads)):
        if c != 0 and r > tol * abs(c):
            last = k
            break
    return cs[:last]


def borel_coeffs(cs):
    return [c / mp.factorial(k) for k, c in enumerate(cs)]     # c_{k+1} zeta^k / k!


def pade_poles(b, L):
    p, q = mp.pade(b[:2 * L + 1], L, L)
    poles = mp.polyroots(q[::-1], maxsteps=2000, extraprec=4 * mp.mp.dps) if len(q) > 1 else []
    zeros = mp.polyroots(p[::-1], maxsteps=2000, extraprec=4 * mp.mp.dps) if len(p) > 1 else []
    return poles, zeros


def scan(cs, label=""):
    cs = [c for c in cs]
    # strip leading exact zeros (series starting at a higher order): shift
    shift = 0
    while cs and cs[0] == 0:
        cs = cs[1:]
        shift += 1
    b = borel_coeffs(cs)
    nz = [k for k in range(len(b)) if b[k] != 0]
    kk = nz[-1]
    root = [abs(b[k]) ** (-mp.mpf(1) / k) for k in nz[-6:] if k > 0]
    sc = root[-1]
    bs = [bk * sc ** k for k, bk in enumerate(b)]
    L = (len(bs) - 1) // 2
    P1, Z1 = pade_poles(bs, L)
    P2, Z2 = pade_poles(bs, L - 2)
    good = []
    for r in P1:
        dz = min(abs(r - z) for z in Z1) if Z1 else mp.inf
        if dz < STAB * max(1, abs(r)):
            continue                                   # Froissart doublet
        d2 = min(abs(r - s) for s in P2) if P2 else mp.inf
        good.append((r * sc, d2 / max(1, abs(r)), dz / max(1, abs(r))))
    good.sort(key=lambda t: abs(t[0]))
    print(f"{label}: {len(cs)} orders (from order {shift + 1}), root-test radius at the last orders: "
          + ", ".join(mp.nstr(r, 6) for r in root[-3:]))
    print(f"   Pade [{L}/{L}] poles without a zero within {STAB:g} (rel.): {len(good)} of {len(P1)}; "
          f"'move' = distance to the nearest [{L - 2}/{L - 2}] pole")
    def show(lst):
        for r, mv, dz in lst:
            print(f"     |a| {mp.nstr(abs(r), 8):>12s}  arg/pi {mp.nstr(mp.arg(r) / mp.pi, 6):>10s}   move {mp.nstr(mv, 2):>8s}"
                  f"   zero-dist {mp.nstr(dz, 2):>8s}   a = {mp.nstr(r, 10)}")
    show(good[:NSHOW])
    sec = float(ARGS.get("sector", 0.45))
    insec = [t for t in good if abs(mp.arg(t[0])) < sec * mp.pi]
    print(f"   poles with |arg| < {sec} pi (nearest {NSHOW}; the H-a1/H-a2 question):"
          + ("" if insec else " none"))
    show(insec[:NSHOW])
    return good


PLANTED = [  # (K_i, a_i, g_i): B(zeta) = sum K_i (1 - zeta/a_i)^(-g_i)
    (1, mp.mpc("-0.2231435513", "-1.5707963268"), mp.mpc("0.5", "3.82")),
    (1, mp.mpc("-0.2231435513", "1.5707963268"), mp.mpc("0.5", "-3.82")),
    (mp.mpf("0.01"), mp.mpc(0, mp.pi), mp.mpf("1.3")),
    (mp.mpf("0.001"), mp.mpf("2.5"), mp.mpf("0.7")),
]


def planted(K):
    """c_{k+1} = k! b_k with b_k = sum K_i (g_i)_k / (k! a_i^k): the same code path as the real data from here on."""
    return [sum(Ki * mp.rf(gi, k) / ai ** k for Ki, ai, gi in PLANTED) for k in range(K)]


if __name__ == "__main__" and QTY == "planted":
    print("planted singularities: " + ", ".join(f"|a| {mp.nstr(abs(a), 6)} arg/pi {mp.nstr(mp.arg(a) / mp.pi, 4)} "
                                              f"(K {mp.nstr(Ki, 2)})" for Ki, a, g in PLANTED))
    scan(planted(int(ARGS.get("K", 120))), "planted")
    sys.exit(0)

if __name__ == "__main__":
    d, cs, rads = load(FN, QTY, ARGS.get("x"))
    if "K" in ARGS:
        cs, rads = cs[:int(ARGS["K"])], rads[:int(ARGS["K"])]
    cs = clean(cs, rads)
    scan(cs, f"{FN} {QTY}" + (f" x={ARGS['x']}" if "x" in ARGS else ""))
