#!/usr/bin/env python3
"""The truncation sequence c_M = (omega/omega_inf at the M-harmonic zero), M = 1, 3, 5, 7, updated with M = 7: increments,
ratio per pair of harmonics, coupling (ratio / profile decay of the added pair), limit estimates, and B_tail.md's
prediction for M = 7 (0.99997-1.00017). Arithmetic on converged zeros read from the logs named below.

Sources:
  M = 1: proof/A_certify.md (certified box), omega/omega_inf = 0.7769940136, A = 2.5421480476
  M = 3: solver/inner_pert/logs/first_order_hier_M3.log (refined hier zero; README: A = 2.7265, 1.0156)
  M = 5: solver/inner_pert/logs/first_order_hier_M5.log (refined hier --schur zero; README: 2.718579, 0.999228)
  M = 7: solver/inner_stokes/logs/hier_nb_M7_*.log, the last 'ZERO M 7' line (hier_nb.py --schur), or --c7=...

Estimators (planted-truth tested below before use, pass criteria fixed in advance):
  geometric: c_M - L = d r^((M-M0)/2) with r from the last two increments: L = c_M + d_M r / (1 - r)
  Aitken on the last three zeros (the same thing written differently; kept as a cross-check of the arithmetic)

    python3 trunc_seq_M7.py [--c7=w,A,B3,C3,B5,C5,B7,C7] [--planted [--mutate=nocorr]] [--pilot] [--amp]
"""
import glob
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

M1 = dict(c=0.7769940136, A=2.5421480476, x=[])
M3 = dict(c=1.0156030948641483, A=2.726475885636556, x=[0.1444194784693144, 0.01108742588800987])
M5 = dict(c=0.9992284031945021, A=2.718579189229888,
          x=[0.1492035749173264, 0.01471905249087869, 0.008405346906375517, 0.0014703451817031647])
BRACKET = (0.99915, 1.00011)          # untruncated, twelve orders (README)
PADE = 0.99989                        # central twelve-coefficient Pade (README)
PRED7 = (0.99997, 1.00017)            # B_tail.md section 5
DR_DW = 6.7e3                         # |d r_1/d(omega/omega_inf)| at the M = 5 point (B_tail.md, reconcile.py)
RHO2 = 0.0574                         # |F_{nu+2}/F_nu|, untruncated amplitude expansion (B_tail.md section 2)


def ratio(x, A, nu):
    """F_nu / F_1 = (B_nu + i C_nu) / A for x = [B3, C3, B5, C5, ...]"""
    i = (nu - 3)
    return complex(x[i], x[i + 1]) / A


def m7_from_log():
    """the last 'ZERO M 7' line of each logs/hier_nb_M7_full*.log; all are printed (they must agree to their noise), and
    the one named by --log=NAME, or else the last file in sorted order, is used"""
    files = sorted(glob.glob(os.path.join(ROOT, "solver/inner_stokes/logs/hier_nb_M7_full*.log")))
    pick = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--log=")]
    found = []
    for fn in files:
        txt = open(fn).read()
        m = list(re.finditer(r"ZERO M 7: A ([0-9.]+)\s+w/w_inf ([0-9.]+)", txt))
        if m:
            found.append((os.path.basename(fn), float(m[-1].group(1)), float(m[-1].group(2))))
    for name, A, c in found:
        print(f"  {name}: A {A:.9f}  w/w_inf {c:.9f}")
    if pick:
        files = [f for f in files if os.path.basename(f) == pick[0]]
    for fn in reversed(files):
        txt = open(fn).read()
        m = list(re.finditer(r"ZERO M 7: A ([0-9.]+)\s+w/w_inf ([0-9.]+).*?\n((?:\s+F_\d.*\n)+)", txt))
        if m:
            g = m[-1]
            A, c = float(g.group(1)), float(g.group(2))
            xs = []
            for line in g.group(3).strip().splitlines():
                z = re.search(r"F_(\d)/F_1 = ([+-][0-9.]+)([+-][0-9.]+)i", line)
                rr = complex(float(z.group(2)), float(z.group(3)))
                xs += [rr.real * A, rr.imag * A]
            return dict(c=c, A=A, x=xs, src=os.path.relpath(fn, ROOT))
    return None


MUT = "--mutate=nocorr" in sys.argv     # planted test with the geometric correction dropped (L = c_7): must fail


def planted():
    """planted truth: c_M = L + d0 r^k (1 + e k) for k = 0..3 (M = 1..7), with a slowly drifting ratio (e) and the
    first term pre-asymptotic (x 1.3). Pass criterion (fixed before running): the geometric estimator from the last three
    terms recovers L to within |c_7 - L| * (|e| + 0.05) / (1 - |r|) + 1e-12, for every planted case."""
    ok = True
    for L in (0.9999, 1.0, 1.0001):
        for r in (-0.045, -0.057, -0.07):
            for e in (0.0, 0.1, -0.1):
                cs = [L + 0.24 * r ** k * (1 + e * k) * (1.3 if k == 0 else 1.0) for k in range(4)]
                d5, d7 = cs[2] - cs[1], cs[3] - cs[2]
                rr = d7 / d5
                Lh = cs[3] + (0.0 if MUT else d7 * rr / (1 - rr))
                tol = abs(cs[3] - L) * (abs(e) + 0.05) / (1 - abs(r)) + 1e-12
                good = abs(Lh - L) <= tol
                ok = ok and good
                if not good or e == 0.0:
                    print(f"  planted L {L} r {r} e {e:+.1f}: estimate {Lh:.8f}, error {Lh - L:+.1e}, tolerance {tol:.1e}"
                          f" {'ok' if good else 'FAIL'}")
    print("planted-truth test:", "PASS" if ok else "FAIL")
    return ok


J5 = [[3493.5, 1968.7], [-2169.7, 6394.7]]   # M = 5 outer Jacobian of r_1 (Re/Im rows; A, w/w_inf), feasibility.py


def pilot():
    """the fixed-(A, w) M = 7 solve at the M = 5 zero: r_1^[7](p_5) = delta_5 in r_1 units, and the Newton step it gives
    with the M = 5 outer Jacobian (feasibility.py; finite differences of step 0.05 in w, so a first estimate only)"""
    fn = os.path.join(ROOT, "solver/inner_stokes/logs/hier_nb_M7_pilot_fixA_at_M5zero.log")
    txt = open(fn).read()
    m = re.search(r"FIXED-A M 7: A ([0-9.]+)\s+w/w_inf ([0-9.]+).*?r_1 ([+-][0-9.e+-]+?)([+-][0-9.]+e[+-]\d+)i", txt)
    if not m:
        print("pilot: no FIXED-A line yet")
        return
    A, w, a, b = (float(m.group(i)) for i in range(1, 5))
    det = J5[0][0] * J5[1][1] - J5[0][1] * J5[1][0]
    dA = -(J5[1][1] * a - J5[0][1] * b) / det
    dw = -(-J5[1][0] * a + J5[0][0] * b) / det
    print(f"pilot (M = 7 at the M = 5 zero, levels 3-7 tuned): r_1 = {a:+.4f}{b:+.4f}i, |r_1| = {abs(complex(a, b)):.3f}")
    print(f"   one Newton step with the M = 5 Jacobian: dA {dA:+.2e}, d(w/w_inf) {dw:+.2e} -> A {A + dA:.6f}, "
          f"w/w_inf {w + dw:.6f}")
    xs = re.findall(r"^level 3 it \d+: .*? x \[([^\]]+)\]", txt, re.M)
    xl = [float(t) for t in xs[-1].split(",")]          # last accepted level-3 iterate, 9 decimals
    rr = {nu: ratio(xl[2:], xl[0], nu) for nu in (3, 5, 7)}
    F7 = rr[7]
    print(f"   F_7/F_1 = {F7:.6f}; re-tuned lower harmonics against the M = 5 zero (same A, w):")
    for nu in (3, 5):
        d = rr[nu] - ratio(M5["x"], M5["A"], nu)
        print(f"     Delta(F_{nu}/F_1) = {d:+.3e}  |.| / |F_7/F_1| = {abs(d) / abs(F7):.3f}   relative {abs(d) / abs(rr[nu]):.2e}")


SIG_FULL = [complex(0.74358457992, 0.52447485276), complex(0.11879383784, 0.39807457035),
            complex(-0.05397185243, 0.11658742432), complex(-0.02886475753, 0.01077835879),
            complex(-0.00555413676, -0.00280595708), complex(-2.7853745790e-4, -1.0776628405e-3)]   # README, J = 750/1000


def amp():
    """lowest-order increments (lemma L): Delta sigma_{M+1} = sigma_{M+1} - sigma^{[M]}_{M+1} from the amplitude expansion
    of hier's M-model (amp_trunc_hiermodel.py), the per-pair ratio x^2 Delta sigma_{M+3} / Delta sigma_{M+1} at
    A = 2.7196, and the check that sigma_1..sigma_M are unchanged (lemma O)"""
    x = 2.7196 ** 2
    D = {}
    for M in (1, 3, 5):
        fs = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"trunc_hier_M{M}_K*_J500_w1.0000.json")))
        if not fs:
            continue
        import json
        d = json.load(open(fs[-1]))
        sig = [complex(float(a), float(b)) for a, b in d["sigma_str"]][1:]       # sigma_1, sigma_2, ...
        exact = max(abs(sig[j] - SIG_FULL[j]) for j in range(M))
        if len(sig) > M and M < len(SIG_FULL):
            D[M] = SIG_FULL[M] - sig[M]
            print(f"M = {M}: sigma_1..sigma_{M} differ from the untruncated by <= {exact:.1e};  Delta sigma_{M + 1} = "
                  f"{D[M].real:+.4e}{D[M].imag:+.4e}i  (|.| {abs(D[M]):.3e}, {abs(D[M]) / abs(SIG_FULL[M]):.2e} of |sigma_{M + 1}|)")
    for M in (1, 3):
        if M in D and M + 2 in D:
            r = x * x * D[M + 2] / D[M]
            print(f"lowest-order ratio per pair x^2 Delta sigma_{M + 3} / Delta sigma_{M + 1} = {r.real:+.4f}{r.imag:+.4f}i, "
                  f"|.| {abs(r):.4f} (x = A^2 = {x:.3f})")


def main():
    if "--amp" in sys.argv:
        amp()
        return
    if "--pilot" in sys.argv:
        pilot()
        return
    if "--planted" in sys.argv:
        sys.exit(0 if planted() else 1)
    opt = [a for a in sys.argv[1:] if a.startswith("--c7=")]
    if opt:
        v = [float(t) for t in opt[0].split("=", 1)[1].split(",")]
        M7 = dict(c=v[0], A=v[1], x=v[2:], src="--c7")
    else:
        M7 = m7_from_log()
    seq = [(1, M1), (3, M3), (5, M5)] + ([(7, M7)] if M7 else [])
    print("M   omega/omega_inf      A              |F_M/F_(M-2)|   increment d_M     d_M/d_(M-2)   coupling")
    prev_c, prev_d = None, None
    rows = []
    for M, z in seq:
        dec = (abs(ratio(z["x"], z["A"], M) / (ratio(z["x"], z["A"], M - 2) if M > 3 else 1.0)) if M >= 3 else None)
        d = z["c"] - prev_c if prev_c is not None else None
        k = d / prev_d if (d is not None and prev_d is not None) else None
        g = abs(k) / dec if (k is not None and dec) else None
        rows.append((M, z["c"], d, k, dec, g))
        print(f"{M}   {z['c']:.10f}   {z['A']:.9f}   {dec if dec else float('nan'):.5f}        "
              f"{d if d is not None else float('nan'):+.3e}       {k if k is not None else float('nan'):+.4f}       "
              f"{g if g is not None else float('nan'):.3f}")
        prev_c, prev_d = z["c"], d
    if M7:
        print(f"\nM = 7 from {M7['src']}")
        lo, hi = PRED7
        c7 = M7["c"]
        print(f"B_tail.md prediction for c_7: {lo}-{hi}; computed {c7:.6f}: {'inside' if lo <= c7 <= hi else 'OUTSIDE'}")
        d5, d7 = rows[2][2], rows[3][2]
        r = d7 / d5
        L = c7 + d7 * r / (1 - r)
        c3, c5 = M3["c"], M5["c"]
        ait = c7 - (c7 - c5) ** 2 / ((c7 - c5) - (c5 - c3))
        print(f"ratio per pair d_7/d_5 = {r:+.4f} (d_5/d_3 = {d5 / rows[1][2]:+.4f});  geometric limit L = {L:.6f}, "
              f"Aitken {ait:.6f}")
        print(f"untruncated bracket {BRACKET[0]}-{BRACKET[1]} (Pade {PADE}): L {'inside' if BRACKET[0] <= L <= BRACKET[1] else 'outside'}")
        tail = L - c7
        print(f"M > 7 tail in omega/omega_inf: {tail:+.2e}  ~ |r_1| {abs(tail) * DR_DW:.3f} (at |dr_1/dw| = {DR_DW:.1e});"
              f" needed below ~3.9 for a 1e-3 box: margin {3.9 / max(abs(tail) * DR_DW, 1e-30):.0f}")
        for kap in (0.25, abs(r)):
            b = abs(d7) * kap / (1 - kap) * DR_DW
            print(f"   with a proven |ratio| <= {kap:.4f}: |tail| <= |d_7| k/(1-k) = {abs(d7) * kap / (1 - kap):.2e} in w, "
                  f"{b:.3f} in r_1")


if __name__ == "__main__":
    main()
