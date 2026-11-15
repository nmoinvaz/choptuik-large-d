#!/usr/bin/env python3
"""Theorem A(M)'s sets X_c and X_b, stated exactly, and the two inclusions its proof uses (#871).

Step 1 of the proof of Theorem A(M) (proof/A_lemmas_M.md 6) needs X_c in X_b (hypothesis H2): X_c the set on which
the point certificate proves one simple zero, X_b the parameter set of the lemma 6b_1 covering.

M = 5. Both sets have half-width 1e-18 in (A, w/w_inf), and they are not centred alike:
  X_b  the covering's ball. Its centre c is ray.json's "x", the certificate's p^ cut at 39 decimals. Its radius is
       that of arb(0, arb("1e-18")) (lemma6b1_cover_M.lohner_c1's dx, and start_data_u's xb): arb stores a radius as
       a 30-bit number rounded up, so it is 1e-18 + 2.19e-27. (The rows are continued from ray.json, whose printed
       dx is wider still.)
  X_c  the DECIMAL box: the chart image p(X), X = [-1e-18, 1e-18]^2 x [-1e-26, 1e-26]^2 x [-1e-27, 1e-27]^2 around
       p^ (the sub-box of the stored lemma 6a input). The certificate's own Krawczyk sub-box is the arb ball with the
       same rounded radii around p^, which contains the decimal box and passes X_b's edge by |p^ - c|.
The Krawczyk test K(X) in int X on the arb ball X gives uniqueness of the zero on every subset, so on the decimal box.
Existence in the decimal box needs K(X) inside it. So the checks are
  (0) the Krawczyk test holds on the arb ball, and that ball contains the decimal box;
  (1) K(X) lies in the decimal box: |K_i| < the decimal half-width, in the chart's coordinates;
  (2) the decimal box lies in X_b: in (A, w/w_inf), |p^_i - c_i| + 1e-18 <= X_b's radius; in the inner parameters the
      hull of p(X) lies in X_b (ball in ball).
HOLD (exit 0) when all three hold, FAIL (exit 1) with the failing part otherwise.

M = 3. X_c is the certificate's whole box and X_b = c +- 1.1e-9 with slack near 1e-9, so the rounding plays no part:
only (2) is checked, for the hull of the run's arb ball (the Krawczyk test on that box is proof/A_M.md 6.2's).

    python3 theorem_A_sets_M.py --M=5 --raw=../../../choptuik-large-d-raw     (ray.json and the 6a input are raw output)
    python3 theorem_A_sets_M.py --M=3
    python3 theorem_A_sets_M.py --M=5 --raw=.. --mutate=decimal|centre|kbox
        decimal  X_b's radius replaced by the exact decimal 1e-18: (2) must fail, by |p^ - c|
        centre   X_b's centre moved by twice its half-width in A: (2) must fail (also at M = 3)
        kbox     the decimal half-widths replaced by half of |K|'s upper bounds: (1) must fail
    python3 theorem_A_sets_M.py --M=5 --raw=.. --selftest    the clean check and the three mutations (M = 3: centre)
"""
import json
import os
import sys

import flint
from flint import arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certify_M as CM  # noqa: E402
import certify_M2 as C2M  # noqa: E402
import start_cover_M2 as SC  # noqa: E402
import start_cover_lemmas_M as SL  # noqa: E402

PREC = 320
CERT = {3: "data/A_M_box_M3_1e-10.json", 5: "data/A_M2_box_M5_1e-10.json"}
MUTATIONS = ("decimal", "centre", "kbox")


def fl(x):
    return float(arb(x).mid())


def check(M, raw, mutate=None, quiet=False):
    ctx.prec = PREC
    m = M + 1
    names = SL.pnames(M)
    say = (lambda *a: None) if quiet else print
    rn = SL.RUNS[M]
    ray = SL.load(rn["ray"], raw)
    with open(os.path.join(HERE, CERT[M])) as fh:
        bx = json.load(fh)
    fails = []
    # X_b: centre and radius as lemma6b1_cover_M builds them
    c = [arb(v) for v in ray["x"]]
    rb = arb(arb(0, arb(ray["h"])).rad())
    if mutate == "decimal":
        rb = arb(ray["h"])
    if mutate == "centre":
        c[0] = c[0] + 2 * arb(ray["h"])
    ctx.prec = PREC
    p = CM.exact_p(bx["p"])
    off = [fl(abs(p[i] - c[i])) for i in range(m)]
    io = max(range(m), key=lambda i: off[i])
    say(f"M = {M}" + (f", MUTATION {mutate}" if mutate else ""))
    say(f"X_b, the covering's ball [{rn['ray']}]: centre c = ray.json's x, radius {ray['h']} as arb rounds it up: "
        f"{ray['h']} + {fl(rb - arb(ray['h'])):.3e}")
    say(f"    c against the certificate's p^ [{CERT[M]}]: |p^ - c| = " + ", ".join(f"{names[i]} {off[i]:.2e}"
                                                                                  for i in range(m)))
    if M == 3:
        ce = SC.cert_data(bx)
        ctx.prec = PREC
        hull = ce["hull"]
        sl = [fl(rb - abs(arb(hull[i].mid()) - c[i]) - arb(hull[i].rad())) for i in range(m)]
        ok2 = all(bool(abs(arb(hull[i].mid()) - c[i]) + arb(hull[i].rad()) <= rb) for i in range(m))
        i2 = min(range(m), key=lambda i: sl[i])
        say(f"X_c, the certificate's box (h = {bx['h']}, w = {bx['w']}): hull half-widths "
            + ", ".join(f"{names[i]} {fl(hull[i].rad()):.2e}" for i in range(m)))
        say(f"(2) the hull of the certificate's ball inside X_b: {ok2} (slack {sl[i2]:+.3e} at {names[i2]})")
        if not ok2:
            fails.append("(2) X_c outside X_b")
        return not fails, fails
    # ---- M = 5: the decimal box, from the sub-box of the stored lemma 6a input
    sbox = SL.load(rn["six_a_box"], raw)
    sub = [sbox["h"]] + list(sbox["w"])
    assert all(arb(a).overlaps(arb(b)) for a, b in zip(sbox["p"], bx["p"])), "the 6a input has another centre"
    res = C2M.krawczyk2_test(bx, sub=sub, verbose=False)
    ctx.prec = PREC
    K, wb = res["K"], res["widths"]
    Xball = [arb(0, wd) for wd in wb]
    dec = [arb(sub[0]), arb(sub[0])] + [arb(sub[1 + (i // 2) % (len(sub) - 1)]) for i in range(M - 1)]
    if mutate == "kbox":
        dec = [abs(K[i]).upper() / 2 for i in range(m)]
    say(f"X_c, the decimal box: p(X), X = +-({', '.join(sub)}) in the chart around p^ (the sub-box of [{rn['six_a_box']}])")
    # (0)
    ok0 = bool(res["ok"]) and all(bool(arb(Xball[i].rad()) >= dec[i]) for i in range(m))
    say(f"(0) Krawczyk on the certificate's arb ball (radii {sub[0]} + {fl(arb(Xball[0].rad()) - arb(sub[0])):.3e}, ...): "
        f"K(X) in int X: {res['ok']}, relative margins " + ", ".join(f"{r:+.3f}" for r in res["rel_margins"])
        + f"; the ball contains the decimal box: {ok0}")
    # (1)
    ok1 = all(bool(abs(K[i]) < dec[i]) for i in range(m))
    say("(1) K(X) inside the decimal box: " + ", ".join(f"|K_{i}| <= {fl(abs(K[i]).upper()):.2e} < {fl(dec[i]):.2e}"
                                                         for i in range(m)) + f": {ok1}")
    # (2)
    T = CM.T_matrix([[arb(x) for x in r] for r in bx["C"]], M)
    G = [[arb(x) for x in r] for r in bx["Gamma"]]
    hull = C2M.p_of_x(p, T, G, Xball, M)          # a superset of the decimal box's hull, used for the inner parameters
    sl = [fl(rb - abs(p[i] - c[i]) - dec[i]) for i in range(2)]
    ok2o = all(bool(abs(p[i] - c[i]) + dec[i] <= rb) for i in range(2))
    sl += [fl(rb - abs(arb(hull[i].mid()) - c[i]) - arb(hull[i].rad())) for i in range(2, m)]
    ok2i = all(bool(abs(arb(hull[i].mid()) - c[i]) + arb(hull[i].rad()) <= rb) for i in range(2, m))
    say("(2) the decimal box inside X_b: slack " + ", ".join(f"{names[i]} {sl[i]:+.3e}" for i in range(m))
        + f": {ok2o and ok2i}")
    # for reference: the certificate's own ball against X_b, and the zero's enclosure
    slb = [fl(rb - abs(p[i] - c[i]) - arb(Xball[i].rad())) for i in range(2)]
    say(f"    for reference: the certificate's arb ball against X_b: slack A {slb[0]:+.3e}, w/w_inf {slb[1]:+.3e}")
    say("    the zero's enclosure p(K(X)): " + ", ".join(f"{names[i]} radius {fl(res['pstar'][i].rad()):.2e}"
                                                         for i in range(m)))
    fails += [t for t, okp in (("(0) Krawczyk on the ball", ok0), ("(1) K(X) outside the decimal box", ok1),
                               ("(2) the decimal box outside X_b", ok2o and ok2i)) if not okp]
    return not fails, fails


def verdict(M, raw, mutate=None, quiet=False):
    ok, fails = check(M, raw, mutate, quiet)
    print(f"THEOREM A SETS (M = {M}, mutation {mutate}): " + ("HOLD" if ok else "FAIL: " + "; ".join(fails)))
    return ok


def selftest(M, raw):
    res = []
    for mut, want in [(None, True)] + [(mu, False) for mu in (MUTATIONS if M == 5 else ("centre",))]:
        ok = verdict(M, raw, mut, quiet=True)
        res.append(ok == want)
        print(f"  mutation {mut}: holds {ok} (must be {want}): {'ok' if ok == want else 'WRONG'}")
    print(f"SELFTEST (M = {M}): {'PASS' if all(res) else 'FAIL'}")
    return all(res)


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    M = int(opts.get("M", "5"))
    raw = opts.get("raw")
    print(f"python-flint {flint.__version__}, FLINT {flint.__FLINT_VERSION__}, {PREC} bits")
    if "selftest" in opts:
        sys.exit(0 if selftest(M, raw) else 1)
    sys.exit(0 if verdict(M, raw, opts.get("mutate")) else 1)


if __name__ == "__main__":
    main()
