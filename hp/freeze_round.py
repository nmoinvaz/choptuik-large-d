#!/usr/bin/env python3
"""How far a converged chain state is from its equations with the exact freeze projector, and how far that moves Delta
(#494).

    cd hp && python3 freeze_round.py RAWDIR STATE [--threads=2] [--dps=160] [--json=OUT] [--mutate=proj|row]

The frozen horizon rows of p (model.rs Ctx::freeze) read Pi_low d + Pi_high p1, with d the horizon block's p mismatch
and Pi_low the projector onto p's odd harmonics k <= kfreeze on the half grid. The kernel receives Pi_low rounded to
double-double (hpkernel.driver.plow_odd) and forms Pi_high = I - Pi_low in double-double (python.rs freeze_pair). Its
rows therefore differ from the exact ones by e = (Pi_ex - Pi_dd) d + ((I - Pi_ex) - Pi_high_dd) p1, of order
2^-106 |p1|. A state the chain converged solves the rounded rows. The exact rows' solution lies at dz = -J^-1 (F + e),
F the state's pinned residual with the held offset at its own coefficients (renode_o.pinned, as anm_run.py's base
polish forms it).

d is the unfrozen kernel's residual on those rows at the same plan (kfreeze off, the rows are d itself), p1 is z's
first block, and e is formed from them at --dps digits. Printed: e by harmonic, the Delta components of -J^-1 e and
-J^-1 F through the octo-double LU (Solver.newton_o, Solver.olu_solve), dz by block, and the exact rows' residual
along the Newton step z - t x at t = 1e-30 to 1. Where it grows as c2 t^2, the linear model holds only for t below
|F + e| / c2, and the first-order shift is then no estimate. The verdict then uses the second-order scale
t2 = sqrt(|F + e| / c2), where a root near z would lie along x, against the chain's error bar 3e-11.

Checks (exit 1 on failure):
  (a) The rows emulated from d, p1 and the double-double projectors equal the frozen kernel's octo-double rows to
      1e-60, and every other row of the frozen and the unfrozen residuals agrees to 1e-90. Control --mutate=proj
      emulates with the exact projector instead, and (a) must fail.
  (b) J x = e for the solved x, with J the exact octo-double Jacobian (Solver.jacobian_o, forward-mode duals) times x
      summed at 140 digits, to 1e-6 of |e|. Control --mutate=row solves for e with its largest row times 1.01, and (b)
      must fail.
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hpkernel.driver import make_solver, plow_odd  # noqa: E402
from multidouble import dd_add, od_axpy  # noqa: E402
from near_null import harmonics, layout  # noqa: E402
from octo_corrector import load_o, o8  # noqa: E402
from renode_o import pinned  # noqa: E402

DELTA_ERR = 3e-11       # Delta's error bar near D 200-215, the spread of the held fits (ALGORITHM.md 3, err_kind "fits")


def mpv(parts, idx):
    """the entries idx of a vector in parts, summed exactly at the working precision"""
    return [mp.fsum(mp.mpf(float(p[i])) for p in parts) for i in idx]


def parts8(vals):
    """mpf values as eight float64 parts, renormalized by od_axpy"""
    out = [np.zeros(len(vals)) for _ in range(8)]
    for i, v in enumerate(vals):
        r = v
        for q in range(8):
            f = float(r)
            out[q][i] = f
            r -= f
    zero = [np.zeros(len(vals))] * 8
    return o8(od_axpy(1.0, *out, *zero))


def matvec(A, x):
    return [mp.fsum(a * b for a, b in zip(row, x)) for row in A]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(args) != 2:
        sys.exit(__doc__)
    raw, st = args
    mutate = opts.get("mutate", "")
    if mutate not in ("", "proj", "row"):
        sys.exit(f"--mutate={mutate}: proj or row")
    mp.mp.dps = int(opts.get("dps", 160))
    z, m = load_o(os.path.join(raw, st))
    M, nodes = int(m["M"]), list(m["nodes"])
    K, kf = len(nodes), int(m.get("kfreeze", 3))
    D = m.get("D_str", repr(float(m["D"])))
    mk = lambda kfz: make_solver(D, M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130),  # noqa: E731
                                 threads=int(opts.get("threads", 2)), kfreeze=kfz)
    s, u = mk(kf), mk(None)
    s.set_segs(m["segs"])
    u.set_segs(m["segs"])
    L = len(z[0])
    hp = list(range(4 * M * K, 4 * M * K + M))
    P = pinned(s, z, M, kf)
    rec = dict(state=st, D=float(m["D"]), M=M, nodes=nodes, kfreeze=kf, dps=mp.mp.dps, mutate=mutate)

    # the projectors: the kernel's double-double ones, and the exact one at --dps digits
    a, b = plow_odd(M, kf)
    hh, hl = dd_add(np.eye(M).ravel(), np.zeros(M * M), -a.ravel(), -b.ravel())
    Pdd = [[mp.mpf(float(a[i, j])) + mp.mpf(float(b[i, j])) for j in range(M)] for i in range(M)]
    Phd = [[mp.mpf(float(hh[i * M + j])) + mp.mpf(float(hl[i * M + j])) for j in range(M)] for i in range(M)]
    N = 2 * M
    th = [2 * mp.pi * j / N for j in range(M)]
    Pex = [[mp.fsum(4 * mp.cos(k * (th[i] - th[j])) / N for k in range(1, kf + 1, 2)) for j in range(M)] for i in range(M)]
    Phx = [[(1 if i == j else 0) - Pex[i][j] for j in range(M)] for i in range(M)]
    rnd = max(abs(Pdd[i][j] - Pex[i][j]) for i in range(M) for j in range(M))
    rnh = max(abs(Phd[i][j] - Phx[i][j]) for i in range(M) for j in range(M))
    print(f"{st}: D {float(m['D']):.10g}, M {M}, nodes {nodes}, kfreeze {kf}, L {L}", flush=True)
    print(f"  projector rounding: max |Pi_low dd - exact| {mp.nstr(rnd, 3)}, max |Pi_high dd - exact| {mp.nstr(rnh, 3)}", flush=True)

    def rows(zz):
        """(kernel's frozen rows, emulated rounded rows, exact rows, d, p1) at zz, mpf"""
        Fs, Fu = o8(s.residual_o(list(zz))), o8(u.residual_o(list(zz)))
        d, p1 = mpv(Fu, hp), mpv(zz, range(M))
        lo, hi = (Pex, Phx) if mutate == "proj" else (Pdd, Phd)
        emu = [x + y for x, y in zip(matvec(lo, d), matvec(hi, p1))]
        ex = [x + y for x, y in zip(matvec(Pex, d), matvec(Phx, p1))]
        other = np.ones(L, bool)
        other[hp] = False
        odiff = max(float(np.abs(np.asarray(Fs[q])[other] - np.asarray(Fu[q])[other]).max()) for q in range(8))
        return Fs, mpv(Fs, hp), emu, ex, d, p1, odiff

    t0 = time.time()
    Fs, ker, emu, ex, d, p1, odiff = rows(z)
    da = max(abs(x - y) for x, y in zip(ker, emu))
    ok_a = bool(da < mp.mpf("1e-60") and odiff < 1e-90)
    print(f"  |d| {mp.nstr(max(abs(x) for x in d), 3)}, |p1| {mp.nstr(max(abs(x) for x in p1), 3)}, "
          f"kernel's rows {mp.nstr(max(abs(x) for x in ker), 3)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"  (a) emulated minus kernel's rows {mp.nstr(da, 3)} (bound 1e-60), other rows frozen minus unfrozen "
          f"{odiff:.1e} (bound 1e-90): {'pass' if ok_a else 'FAIL'}", flush=True)
    rec["check_a"] = dict(emu_minus_kernel=float(da), other_rows=odiff, ok=ok_a)
    if mutate == "proj":
        # the control of (a) needs no LU
        print("check:", "pass" if ok_a else "FAIL")
        return 0 if ok_a else 1

    # e on the horizon p rows; F the pinned residual at the state
    e_rows = [x - y for x, y in zip(ex, ker)]
    e = [mp.mpf(0)] * L
    for i, r in enumerate(hp):
        e[r] = e_rows[i]
    F8 = P.residual_o(z)
    Fmax = float(np.abs(np.asarray(F8[0]) + np.asarray(F8[1])).max())
    eh = harmonics(np.array([float(x) for x in e_rows]), 1)
    top = sorted(eh.items(), key=lambda kv: -kv[1])[:4]
    emax = float(max(abs(x) for x in e_rows))
    print(f"  e = exact minus rounded rows: max {emax:.3e}; harmonics " + ", ".join(f"k{k} {v:.2e}" for k, v in top), flush=True)
    print(f"  the state's pinned residual F: max {Fmax:.3e}", flush=True)
    rec.update(e_max=emax, e_harmonics={str(k): v for k, v in eh.items()}, F_max=Fmax)

    # the octo-double LU of J at z, then -J^-1 e and -J^-1 F
    t0 = time.time()
    out = s.newton_o(list(z))
    print(f"  octo-double LU {time.time() - t0:.0f} s, pivots {out[3]:.1e}..{out[4]:.1e}", flush=True)
    rhs = list(e)
    if mutate == "row":
        i = max(hp, key=lambda r: abs(e[r]))
        rhs[i] = rhs[i] * mp.mpf("1.01")
    xe = o8(s.olu_solve(list(parts8(rhs))))
    xF = o8(s.olu_solve(list(F8)))
    xe_mp, xF_mp = mpv(xe, range(L)), mpv(xF, range(L))
    dDe, dDF = -xe_mp[-1], -xF_mp[-1]
    blocks = layout(M, K)
    bmax = {name: float(max(abs(xe_mp[o + j] + xF_mp[o + j]) for j in range(M))) for name, o, _ in blocks}
    big = sorted(bmax.items(), key=lambda kv: -kv[1])[:5]
    print(f"  Delta's shift: -(J^-1 e)_Delta {mp.nstr(dDe, 6)}, -(J^-1 F)_Delta {mp.nstr(dDF, 6)}", flush=True)
    print("  |dz| by block, largest: " + ", ".join(f"{n} {v:.2e}" for n, v in big), flush=True)
    rec.update(dDelta_e=float(dDe), dDelta_F=float(dDF), dz_blocks=bmax)

    # (b) J x = e from the exact Jacobian at 140 digits
    t0 = time.time()
    with mp.workdps(140):
        J = [np.asarray(p) for p in s.jacobian_o(list(z))]
        nz = [c for c in range(L) if xe_mp[c] != 0]
        jx = [mp.fsum(mp.fsum(mp.mpf(float(J[q][r, c])) for q in range(8) if J[q][r, c] != 0) * xe_mp[c] for c in nz)
              for r in range(L)]
        en = mp.sqrt(mp.fsum(x * x for x in e))
        rb = mp.sqrt(mp.fsum((x - y) ** 2 for x, y in zip(jx, e))) / en
    ok_b = bool(rb < mp.mpf("1e-6"))
    print(f"  (b) ||J x - e|| / ||e|| {mp.nstr(rb, 3)} (bound 1e-6, {time.time() - t0:.0f} s): {'pass' if ok_b else 'FAIL'}", flush=True)
    rec["check_b"] = dict(rel=float(rb), ok=ok_b)

    # the exact rows' residual along the Newton step, z - t x with x = J^-1 (F + e)
    x8 = o8(od_axpy(1.0, *xe, *xF))

    def gmax(zz):
        _, k1, _, x1, _, _, _ = rows(zz)
        F1 = P.residual_o(zz)
        G = [mp.mpf(float(F1[0][r])) + mp.mpf(float(F1[1][r])) for r in range(L)]
        for i, r in enumerate(hp):
            G[r] += x1[i] - k1[i]
        return max(abs(g) for g in G)

    g0 = gmax(z)
    scan = []
    for t in (1e-30, 1e-26, 1e-22, 1e-18, 1e-14, 1e-10, 1e-6, 1e-2, 1.0):
        scan.append((t, gmax(o8(od_axpy(-t, *x8, *z)))))
    print("  exact rows along z - t x: max|F + e| " + mp.nstr(g0, 3) + " at t = 0; "
          + ", ".join(f"{t:.0e}: {mp.nstr(g, 3)}" for t, g in scan), flush=True)
    # the quadratic term c2 t^2: g / t^2 where it stays constant to 1% and g is far above the offset
    q = [g / mp.mpf(t) ** 2 for t, g in scan if g > 1e6 * g0]
    c2 = q[0] if q else mp.mpf(0)
    quad = [v for v in q if abs(v / c2 - 1) < 0.01] if q else []
    rec.update(newton_exact=[float(g0), float(scan[-1][1])], scan=[(t, float(g)) for t, g in scan])
    if len(quad) >= 2:
        # the linear model holds for t below g0 / c2, and a root near z would lie near t2 = sqrt(g0 / c2) along x
        t_lin, t2 = g0 / c2, mp.sqrt(g0 / c2)
        d2 = float(t2) * abs(float(dDe))
        print(f"  quadratic along x over {len(quad)} decades of t: c2 {mp.nstr(c2, 4)}; the linear model holds for "
              f"t < {mp.nstr(t_lin, 2)}, the second-order scale is t2 {mp.nstr(t2, 2)}", flush=True)
        # c2 is the whole residual's curvature; the root's position needs its part on the weak left directions, which
        # this script does not form, so the shift is only bracketed by the two scales
        verdict = "UNRESOLVED" if abs(float(dDe)) >= DELTA_ERR else "OK"
        print(f"  verdict: the first-order shift {abs(float(dDe)):.2e} lies outside the linear region (t = 1 > {mp.nstr(t_lin, 2)}) "
              f"and is not an estimate. At the second-order scale Delta moves by about {d2:.1e}, which holds if the "
              f"curvature on the weak left directions is of the whole residual's size, and a root may not exist. "
              f"Against the error bar {DELTA_ERR:.0e}: {verdict}", flush=True)
        rec.update(c2=float(c2), t_lin=float(t_lin), t2=float(t2), dDelta_2=d2)
    else:
        v = abs(float(dDe))
        verdict = "OK" if v < DELTA_ERR / 10 else ("WATCH" if v < DELTA_ERR else "BAD")
        print(f"  verdict: no quadratic regime found; the first-order shift {v:.2e} against {DELTA_ERR:.0e}: {verdict}", flush=True)
    rec["verdict"] = verdict
    ok = ok_a and ok_b
    print("check:", "pass" if ok else "FAIL")
    if opts.get("json"):
        with open(opts["json"], "w") as fh:
            json.dump(rec, fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
