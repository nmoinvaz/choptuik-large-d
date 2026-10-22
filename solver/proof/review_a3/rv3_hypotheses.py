#!/usr/bin/env python3
"""Review A3, check K3: the inclusions that Theorem A(3) (proof/A_lemmas_M.md sections 6-7) needs between the point
certificate's parallelepiped P and the boxes over which the lemmas were run, in exact rational interval arithmetic,
and an independent recomputation of the 6a input box B1 from the box run's saved Lohner sets.

  P   = p^ + T X, X = [-r_h, r_h]^2 x [-r_w, r_w]^2, with r_h, r_w the radii the run actually used (arb(0, arb("1e-10"))
        rounds the radius up to 1.0000000014e-10; bounded here by 1.00000001e-10, resp. 4.0000001e-16) and p^ within
        2^-255 relative of the decimal centre strings (certify_M.exact_p rounds them at 256 bits);
  H1: hull(P) in X6 = x^ +- 1e-6, x^ = fixed_point_M.ZEROS[3] (lemma S, 6b0, the flat estimate; A_lemmas_M 3.3);
  H2: hull(P) in X_b = x_b +- 1.1e-9, x_b = the centre recorded in lemma6b1_runs_M3_cert/ray.json, 1.1e-9 its h
      (every row file must carry the same h; row_run asserts the ray's x);
  H3: (a) hull(P) in the 6a box x_6a +- h_6a of data/A_M_6a_M3_1e-10_in.json (only its omega range enters the backward
      flow; the A, B, C ranges only have to lie in X6); (b) the 6a state box yX_xi1 contains the state set over P,
      c + Q q + V X with V = cV + QV qV (recomputed here from the box run's JSON), component by component;
  path: the certificate's start ray arg 5pi/16 lies within 0.2 of 3pi/8 (the old sector lemma) and r0 <= 0.1, the arc
      radius is 0.2, xi1 = 0.3; the 6a input is for xi1 = 0.3 and the same path.
Exit 0 iff all hold. --mutate=shift moves the 6b1 centre by 2e-9 in A (H2 must fail).
"""
import json
import math
import os
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.join(HERE, "..")


def parse_ball(s):
    s = s.strip()
    if s == "0":
        return (Fr(0), Fr(0))
    if s.startswith("["):
        s = s[1:-1].strip()
    if "+/-" in s:
        m, r = s.split("+/-")
        m = Fr(m.strip()) if m.strip() else Fr(0)
        r = Fr(r.strip())
        return (m - r, m + r)
    return (Fr(s), Fr(s))


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def mul(a, b):
    ps = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return (min(ps), max(ps))


def inside(a, lo, hi):
    return lo <= a[0] and a[1] <= hi


def main(mutate=None):
    ok = True
    bx = json.load(open(os.path.join(PROOF, "data", "A_M_box_M3_1e-10.json")))
    M = bx["M"]
    m = M + 1
    # radii actually used by the run: arb(0, arb(h)) rounds the radius up (30-bit mag); measured with python-flint 0.9
    # at 768 bits: arb(0, arb("1e-10")).rad() = 1.000000001425208...e-10 (2 ulps of mag above 1e-10). Upper bounds:
    assert bx["h"] == "1e-10" and bx["w"] == ["4e-16"]
    rh = Fr("1.00000000142520807e-10")
    rw = Fr("4.0000000013e-16")
    radii = [rh, rh, rw, rw]
    # the 6a script's parameter ball arb(x, arb("1e-10")) at 320 bits has radius 1.0000000025094102...e-10 (one more
    # ulp of mag, from adding the decimal conversion error); a lower bound:
    r6a = Fr("1.00000000250941e-10")
    eps = Fr(1, 2 ** 255)
    phat = []
    for x in bx["p"]:
        v = Fr(x)
        phat.append((v - abs(v) * eps, v + abs(v) * eps))
    T = [[(Fr(1 if i == j else 0),) * 2 for j in range(m)] for i in range(m)]
    for i, row in enumerate(bx["C"]):
        for j in range(2):
            T[2 + i][j] = parse_ball(row[j])
    hull = []
    for i in range(m):
        acc = phat[i]
        for j in range(m):
            acc = add(acc, mul(T[i][j], (-radii[j], radii[j])))
        hull.append(acc)
    names = ["A", "w/w_inf", "B3", "C3"]
    print("K3: hull of P (exact): " + "; ".join(f"{nm} centre +- {float((h[1] - h[0]) / 2):.3e}"
                                              for nm, h in zip(names, hull)))
    # H1
    sys.path.insert(0, PROOF)
    import fixed_point_M as FPM          # only for the stored centre strings ZEROS[3]
    xhat = [Fr(v) for v in FPM.ZEROS[M]]
    h1 = all(inside(hull[i], xhat[i] - Fr("1e-6"), xhat[i] + Fr("1e-6")) for i in range(m))
    dist1 = max(float(max(abs(hull[i][0] - xhat[i]), abs(hull[i][1] - xhat[i]))) for i in range(m))
    print(f"H1: hull(P) in x^ +- 1e-6: {h1} (largest |hull - x^| {dist1:.2e})")
    ok &= h1
    # H2
    ray = json.load(open(os.path.join(PROOF, "lemma6b1_runs_M3_cert", "ray.json")))
    xb = [Fr(v) for v in ray["x"]]
    hb = Fr(ray["h"])
    if mutate == "shift":
        xb[0] += Fr("2e-9")
    rows_h = set()
    for j in range(12):
        rj = json.load(open(os.path.join(PROOF, "lemma6b1_runs_M3_cert", f"row_{j:02d}.json")))
        rows_h.add(rj["h"])
        ok &= rj["ok"] and rj["M"] == M
    chk = json.load(open(os.path.join(PROOF, "lemmas_M_runs", "lemma6b1_check_M3_cert.json")))
    h2 = all(inside(hull[i], xb[i] - hb, xb[i] + hb) for i in range(m))
    dist2 = max(float(max(abs(hull[i][0] - xb[i]), abs(hull[i][1] - xb[i]))) for i in range(m))
    print(f"H2: hull(P) in x_b +- {ray['h']} (6b1 covering): {h2} (largest |hull - x_b| {dist2:.2e}); rows' h "
          f"{sorted(rows_h)}, all rows ok and the gluing check JSON ok: {chk['ok']} ({chk['tiles']} tiles)")
    ok &= h2 and rows_h == {ray["h"]} and chk["ok"]
    # H3 (a)
    d6 = json.load(open(os.path.join(PROOF, "data", "A_M_6a_M3_1e-10_in.json")))
    x6 = [Fr(v) for v in d6["x"]]
    h6 = Fr(d6["h"])
    assert d6["h"] == "1e-10"
    h3a_nom = all(inside(hull[i], x6[i] - h6, x6[i] + h6) for i in range(m))
    h3a = all(inside(hull[i], x6[i] - r6a, x6[i] + r6a) for i in range(m))
    gap = min(min(hull[i][0] - (x6[i] - r6a), (x6[i] + r6a) - hull[i][1]) for i in range(2))
    in6 = all(abs(x6[i] - xhat[i]) + r6a <= Fr("1e-6") for i in range(m))
    print(f"H3a: hull(P) in the 6a box x_6a +- {d6['h']} as written: {h3a_nom}; with the radius the code builds "
          f"({float(r6a):.13e}): {h3a}, margin in A, omega {float(gap):.2e} (from radius rounding only); that box in "
          f"X6: {in6}")
    ok &= h3a and in6
    # H3 (b): state set over P from the saved Lohner sets
    S = bx["sets"]
    n = len(S["c"])
    nd = n - 1
    c = [parse_ball(z[0]) for z in S["c"]] + [parse_ball(z[1]) for z in S["c"]]
    Q = [[Fr(float.fromhex(x)) for x in row] for row in S["Q"]]
    q = [parse_ball(x) for x in S["q"]]
    cV = [[parse_ball(S["cV"][i][k][0]) for k in range(m)] for i in range(n)] + \
         [[parse_ball(S["cV"][i][k][1]) for k in range(m)] for i in range(n)]
    QV = [[Fr(float.fromhex(x)) for x in row] for row in S["QV"]]
    qV = [[parse_ball(x) for x in row] for row in S["qV"]]
    state = []
    for i in range(2 * n):
        acc = c[i]
        for k in range(2 * n):
            if Q[i][k] != 0:
                acc = add(acc, mul((Q[i][k], Q[i][k]), q[k]))
        for j in range(m):
            v = cV[i][j]
            for k in range(2 * n):
                if QV[i][k] != 0:
                    v = add(v, mul((QV[i][k], QV[i][k]), qV[k][j]))
            acc = add(acc, mul(v, (-radii[j], radii[j])))
        state.append(acc)
    B1 = [parse_ball(z[0]) for z in d6["yX_xi1"]] + [parse_ball(z[1]) for z in d6["yX_xi1"]]
    re_ok = all(inside(state[i], B1[i][0], B1[i][1]) for i in range(nd))
    im_ok = all(inside(state[n + i], B1[nd + i][0], B1[nd + i][1]) for i in range(nd))
    rmax = max(float((state[i][1] - state[i][0]) / 2) for i in list(range(nd)) + list(range(n, n + nd)))
    print(f"H3b: the 6a input B1 contains the state set over P recomputed here: {re_ok and im_ok} (largest radius "
          f"{rmax:.2e}; B1 {max(float((b[1] - b[0]) / 2) for b in B1):.2e})")
    ok &= re_ok and im_ok
    # the 6a run's own record
    r6 = json.load(open(os.path.join(PROOF, "lemmas_M_runs", "lemma6a_M3_cert.json")))
    rho = parse_ball(r6["rho6a"])
    print(f"6a record: xi1 {r6['xi1']}, xia {r6['xia']}, box {r6['box']}, pilot {r6['pilot']}, rho_6a <= "
          f"{float(rho[1]):.4f} < 1: {rho[1] < 1}")
    ok &= r6["xi1"] == "0.3" and r6["pilot"] is None and rho[1] < 1 and r6["box"] == "1e-6"
    # path
    r0, arg, rho_arc, xi1 = bx["path"]
    th = 5 * math.pi / 16
    pth = (arg == "5pi/16" and abs(th - 3 * math.pi / 8) <= 0.2 - 1e-3 and Fr(r0) <= Fr("0.1")
           and rho_arc == "0.2" and xi1 == "0.3" and d6["xi1"] == "0.3" and d6["path"] == bx["path"])
    print(f"path: ray {arg} from {r0} (|5pi/16 - 3pi/8| = {abs(th - 3 * math.pi / 8):.4f} <= 0.2), arc {rho_arc}, "
          f"xi1 {xi1}; 6a input on the same path: {pth}")
    ok &= pth
    print("K3: " + ("ALL HOLD" if ok else "SOME FAIL") + (f" [mutation {mutate}]" if mutate else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    mut = None
    for a in sys.argv[1:]:
        if a.startswith("--mutate="):
            mut = a.split("=", 1)[1]
    sys.exit(main(mut))
