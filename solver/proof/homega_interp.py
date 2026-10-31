#!/usr/bin/env python3
"""(H_omega) measured: the interpolation error of sigma_k(omega) across the five nodes omega/omega_inf = 0.99, 0.995, 1.00,
1.005, 1.01 (proof/P5_scope.md, section 4.2 item 2 and its procedure of 2026-10-31). Not rigorous: a measurement of how
well a polynomial in omega through a few nodes represents sigma_k(omega), from which the number of nodes a certificate box
needs is estimated under a stated assumption.

In t = (omega/omega_inf - 1)/0.01 the nodes are t = -1, -1/2, 0, 1/2, 1. For each order k:
  m3    the miss of the cubic through the four outer nodes at t = 0, relative to |sigma_k(1.00)|: the error of a four-node
        interpolation at the centre;
  m2-, m2+  the misses of the quadratic through t = -1, 0, 1 (the three nodes every estimator of this project uses) at
        t = -1/2 and +1/2, relative: how wrong the three-node picture is halfway between its nodes;
  R     the radius of analyticity in t estimated from the degree-4 interpolant's coefficients a_j as R = |a_3/a_4|: exact
        for a simple pole at t = R (the aliasing of the higher terms onto a_3 and a_4 cancels in the ratio for these
        symmetric nodes); sqrt|a_2/a_4| is printed beside it as a consistency check of the single-pole picture: it must
        equal, to 30%, what a pure pole at t = R aliases to on these nodes (exact for the planted poles); 0.01 R is the
        radius in omega/omega_inf;
  n_box the number of nodes a box |t| <= 0.01 (radius 1e-4 in omega/omega_inf) needs for a relative interpolation error
        below TAU = 1e-6, under the assumption that sigma_k(t) is the single-pole function through the five node values,
        f(t) = p4(t) + c (1/(1 - t/R) - q4(t)) with q4 the pole's own degree-4 fit on the nodes and c = a_4/b_4 its
        de-aliased strength (f is the planted function itself on the plants): the count is found by brute force
        (Chebyshev interpolation of f on n + 1 nodes of [-1, 1], the relative error measured on 41 points of |t| <= 0.01,
        the smallest n + 1 with error <= TAU). It is given only when the consistency check passes and R > 1.05
        (a pole inside or at the node interval has no useful count); otherwise the single-pole picture is inconsistent
        and the table says so.
Planted truth first (`planted`): sigma_k(t) = sigma_k(0) (1 + eps_k / (1 - t/R)) with the real sigma_k(1.00) as scale, a
simple pole at t = R, for R = 3 and R = 1.5 (0.03 and 0.015 in omega/omega_inf): |a_3/a_4| must recover R to 5%, and n_box
must agree within one node with the brute-force minimum for the planted function itself, at every k. Control (`--mutate`):
sigma_k(0.995) scaled by 1 + 1e-6 at every k must move m2- by at least 5e-7 at every k (the real run reports the shift;
the planted run requires it).

    python3 homega_interp.py planted
    python3 homega_interp.py real [--mutate]      the five files named in FILES (the T93 pair copied into scan39J1250p960)
Exit 1 when a planted check or the control fails.
"""
import json
import math
import os
import sys

import mpmath as mp

mp.mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
AMP = os.path.join(HERE, "..", "inner_amp")
FILES = {-1.0: os.path.join(AMP, "scan41J1250", "sigF_w0.9900_K41_J1250_p1280.json"),
         -0.5: os.path.join(AMP, "scan39J1250p960", "sigF_w0.9950_K39_J1250_p960.json"),
         0.0: os.path.join(AMP, "scan39J1250p960", "sigF_w1.0000_K39_J1250_p960.json"),
         0.5: os.path.join(AMP, "scan39J1250p960", "sigF_w1.0050_K39_J1250_p960.json"),
         1.0: os.path.join(AMP, "scan41J1250", "sigF_w1.0100_K41_J1250_p1280.json")}
T = (-1.0, -0.5, 0.0, 0.5, 1.0)
KMAX = 19
TAU = 1e-6
BOX = 0.01                      # |t| <= 0.01: radius 1e-4 in omega/omega_inf
FAILED = []


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg, flush=True)
    if not ok:
        FAILED.append(msg)
    return ok


def load(fn):
    r = json.load(open(fn))
    return [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in r["sigma_str"]], r


def interp_coeffs(ts, vals):
    """coefficients a_0..a_n of the polynomial through (ts, vals), by the Vandermonde solve"""
    n = len(ts)
    A = mp.matrix(n, n)
    for i, t in enumerate(ts):
        for j in range(n):
            A[i, j] = mp.mpf(t) ** j
    b = mp.matrix(vals)
    a = mp.lu_solve(A, b)
    return [a[j] for j in range(n)]


def peval(a, t):
    return sum(c * mp.mpf(t) ** j for j, c in enumerate(a))


def analyse(sig, k):
    """sig: {t: sigma_k(t)} at the five nodes; returns the measures of the docstring for this k"""
    s = [sig[t] for t in T]
    a4 = interp_coeffs(T, s)
    p3 = interp_coeffs((-1.0, -0.5, 0.5, 1.0), [sig[-1.0], sig[-0.5], sig[0.5], sig[1.0]])
    p2 = interp_coeffs((-1.0, 0.0, 1.0), [sig[-1.0], sig[0.0], sig[1.0]])
    m3 = abs(peval(p3, 0) - sig[0.0]) / abs(sig[0.0])
    m2m = abs(peval(p2, -0.5) - sig[-0.5]) / abs(sig[-0.5])
    m2p = abs(peval(p2, 0.5) - sig[0.5]) / abs(sig[0.5])
    R34 = abs(a4[3] / a4[4]) if a4[4] != 0 else mp.inf
    R24 = mp.sqrt(abs(a4[2] / a4[4])) if a4[4] != 0 else mp.inf
    R = R34
    # the single-pole picture is consistent when sqrt|a_2/a_4| equals, to 30%, what a pure pole at t = R aliases to on
    # these five nodes (its own degree-4 fit); exact for the planted poles, a test of the picture on real data
    if mp.isfinite(R34) and R34 > 1:
        b = interp_coeffs(T, [1 / (1 - mp.mpf(t) / R) for t in T])
        R24_model = mp.sqrt(abs(b[2] / b[4]))
        consistent = mp.isfinite(R24) and abs(R24 - R24_model) <= mp.mpf("0.3") * R24_model
    else:
        R24_model, consistent = mp.nan, False
    if consistent and R > mp.mpf("1.05"):
        # the single-pole function through the five node values: p4 plus c (g - q4), g the pure pole at R, q4 its
        # degree-4 fit on the nodes (so the sum equals p4 at them), c = a_4/b_4 the de-aliased strength
        c = a4[4] / b[4]
        model = (lambda t, a=a4, b=b, c=c, R=R: peval(a, t) + c * (1 / (1 - mp.mpf(t) / R) - peval(b, t)))
        nbox = brute_nbox(model)
    else:
        nbox = None
    return {"k": k, "m3": float(m3), "m2m": float(m2m), "m2p": float(m2p), "R34": float(R34), "R24": float(R24),
            "R24_model": float(R24_model), "consistent": bool(consistent), "nbox": nbox, "a": [complex(x) for x in a4]}


def brute_nbox(f):
    """the smallest number of Chebyshev nodes of [-1, 1] whose interpolant of f has relative error <= TAU on |t| <= BOX"""
    pts = [mp.mpf(-BOX) + 2 * BOX * i / 40 for i in range(41)]
    for n in range(2, 40):
        ts = [mp.cos(mp.pi * (2 * i + 1) / (2 * (n + 1))) for i in range(n + 1)]
        a = interp_coeffs(ts, [f(t) for t in ts])
        err = max(abs(peval(a, t) - f(t)) / abs(f(t)) for t in pts)
        if err <= TAU:
            return n + 1
    return None


def table(rows, title):
    print(f"\n{title}")
    print("| k | m3 (4 nodes at centre) | m2- (3 nodes at 0.995) | m2+ (3 nodes at 1.005) | R = |a3/a4| | sqrt|a2/a4| (pole model) | nodes for the 1e-4 box |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        nb = r["nbox"] if r["nbox"] is not None else ("single-pole picture inconsistent" if not r["consistent"] else "pole at the nodes")
        print(f"| {r['k']} | {r['m3']:.2e} | {r['m2m']:.2e} | {r['m2p']:.2e} | {r['R34']:.2f} | {r['R24']:.2f} ({r['R24_model']:.2f}) | {nb} |")


def planted():
    real0, _ = load(FILES[0.0])
    for R in (3.0, 1.5):
        rows, brute = [], []
        for k in range(KMAX + 1):
            eps = mp.mpf("0.2") * (1 + mp.mpf(k) / 10)
            f = (lambda t, s0=real0[k], e=eps, R=R: s0 * (1 + e / (1 - mp.mpf(t) / R)))
            sig = {t: f(t) for t in T}
            r = analyse(sig, k)
            rows.append(r)
            brute.append(brute_nbox(f))
        table(rows, f"planted: simple pole at t = {R} (omega radius {0.01 * R}), eps_k = 0.2 (1 + k/10)")
        worst = max(abs(r["R34"] - R) / R for r in rows)
        check(worst <= 0.05, f"R = {R}: |a_3/a_4| recovers the planted radius to {worst * 100:.2f}% at worst over k = 0..{KMAX} "
                             f"(sqrt|a_2/a_4| {min(r['R24'] for r in rows):.2f}-{max(r['R24'] for r in rows):.2f}, reported)")
        diff = [abs(r["nbox"] - b) if r["nbox"] is not None and b is not None else 99 for r, b in zip(rows, brute)]
        check(max(diff) <= 1, f"R = {R}: the node count n_box agrees with the brute-force minimum within one node "
                              f"(estimates {sorted(set(r['nbox'] for r in rows), key=str)}, brute force {sorted(set(brute), key=str)}, "
                              f"largest difference {max(diff)})")
    # control: the mutation moves m2-
    rows0, rows1 = [], []
    for k in range(KMAX + 1):
        f = (lambda t, s0=real0[k]: s0 * (1 + mp.mpf("0.2") / (1 - mp.mpf(t) / 3)))
        sig = {t: f(t) for t in T}
        rows0.append(analyse(sig, k))
        sig[-0.5] = sig[-0.5] * (1 + mp.mpf("1e-6"))
        rows1.append(analyse(sig, k))
    shift = [abs(r1["m2m"] - r0["m2m"]) for r0, r1 in zip(rows0, rows1)]
    check(min(shift) >= 5e-7, f"control: scaling sigma_k(0.995) by 1 + 1e-6 moves m2- by {min(shift):.1e} .. {max(shift):.1e}")


def real(mutate=False):
    sig, meta = {}, {}
    for t, fn in FILES.items():
        s, r = load(fn)
        sig[t], meta[t] = s, (r["K"], r["J"], r["prec"], len(s))
        print(f"t = {t:+.1f} (wf {1 + 0.01 * t:.3f}): {os.path.relpath(fn, HERE)}  K {r['K']}, J {r['J']}, {r['prec']} bits, "
              f"{len(s)} coefficients")
    if mutate:
        sig[-0.5] = [x * (1 + mp.mpf("1e-6")) for x in sig[-0.5]]
        print("  MUTATION: sigma_k(0.995) scaled by 1 + 1e-6 at every k")
    rows = [analyse({t: sig[t][k] for t in T}, k) for k in range(KMAX + 1)]
    table(rows, "real data" + (" (mutated)" if mutate else ""))
    out = os.path.join(HERE, "logs", "homega_interp" + ("_mutate" if mutate else "") + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"files": {str(t): os.path.relpath(fn, HERE) for t, fn in FILES.items()},
                   "meta": {str(t): m for t, m in meta.items()}, "rows": [{k: v for k, v in r.items() if k != "a"} for r in rows],
                   "tau": TAU, "box_t": BOX, "mutate": mutate}, f, indent=1, default=str)
    print(f"written {os.path.relpath(out, HERE)}")
    if mutate:
        ref = os.path.join(HERE, "logs", "homega_interp.json")
        if os.path.exists(ref):
            r0 = json.load(open(ref))["rows"]
            shift = [abs(a["m2m"] - b["m2m"]) for a, b in zip(rows, r0)]
            check(min(shift) >= 5e-7, f"control on the real data: m2- moves by {min(shift):.1e} .. {max(shift):.1e}")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "planted"
    if stage == "planted":
        planted()
    elif stage == "real":
        real("--mutate" in sys.argv)
    else:
        raise SystemExit("stage: planted | real [--mutate]")
    if FAILED:
        print(f"\n{len(FAILED)} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
