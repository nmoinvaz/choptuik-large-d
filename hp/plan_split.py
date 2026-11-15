#!/usr/bin/env python3
"""Why the series' coefficients move by percent between two plans while D* does not (#736): the change of each
coefficient z_k between #686's two series at a base, split over the weakest right singular vectors of the chain's
pinned system at that state (near_null.py's) and over the rest.

    cd hp && python3 plan_split.py RAWDIR D0 [--threads=4] [--its=16] [--json=OUT] [--mutate=V]

RAWDIR is a clone of the raw-data repository. At D0 (252.2497 or 252.6135):
  the state   I603-win2/chain/anm_pin3_D<D0>.npz
  kept plan   I685-dh370/repro/jet_pin3_D<D0>.npz   (#603's records reproduced bit for bit, the stored plan)
  new plan    I686-dh370/newplan/jet_pin3_D<D0>.npz (--keep_plan 0, the plan made again at the state)
Each series holds z_0..z_24 in eight parts; here they are summed in f64 (the parts' two leading terms carry the
coefficients to 1e-16 relative, far below the changes split here).

The weak vectors: for each plan the pinned system at the state is factored in octo-double (Solver.newton_o) and
near_null.subspace gives the k = 4 weakest right singular vectors V and their sigma, as near_null.py does: the kept plan
is the state's stored plan, the new one Solver.plan at 1e-32, 0.05. The kept plan's v_1 is checked as near_null.py checks
it, ||J v_1|| / sigma_1 within 1e-2 of 1 with J the exact octo-double Jacobian and the product at 140 digits.

The split, k = 1..24 (5..13 the issue's range): dz_k = z_k(new) - z_k(kept), its share on span(V), ||V^T dz_k|| /
||dz_k||, each vector's component, and ||dz_k|| / ||z_k||; also the share of z_k itself on span(V). For both plans' V.

Planted (fixed before the real split was read): a copy of the kept series with P_k = |dz_k| (0.6 v_2 + 0.8 g_k)
added to z_k's leading part at k = 5..13, v_2 the kept plan's second vector and g_k a random unit vector (seed 736 + k)
orthogonal to span(V). The same loading and split, applied to the copy against the kept series, must return a share of
0.6 and V^T dz = (0, 0.6 |dz_k|, 0, 0) to 1e-8 relative at every k. Control: --mutate=V projects on V with its second
column replaced by a random unit vector, and the planted check must fail. Exit 1 when a check fails.

Reading (fixed before the real split was read): at k = 7 and 13, a share above 0.9 on both plans' span(V) at both bases
says the plan's change lies on the weak directions; below 0.5 says it does not; otherwise it is mixed.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hpkernel.driver import make_solver  # noqa: E402
from near_null import subspace  # noqa: E402
from octo_corrector import load_o  # noqa: E402

args = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
K_ISSUE = range(5, 14)


def load_series(path):
    """z_0..z_N in f64 (the parts summed) and the meta"""
    d = np.load(path)
    m = json.loads(str(d["meta"]))
    ks = sorted(k for k in d.files if k[0] == "z" and k[1:].isdigit())
    return sum(np.asarray(d[k], dtype=float) for k in ks), m, d, ks


def weak(raw, st, plan, threads, its, check, jet_segs=None):
    """(V, sigma, the LU's pivots, the check's ratio or None) for the pinned system at the state on the given plan: the
    series' own stored plan (jet_segs) when it has one, and whether the state's plan or Solver.plan reproduces it"""
    z, m = load_o(os.path.join(raw, st))
    z = [np.ascontiguousarray(np.asarray(p, dtype=float)) for p in z]
    M, nodes = int(m["M"]), list(m["nodes"])
    D = m.get("D_str", repr(float(m["D"])))
    kf = int(m.get("kfreeze", 3))
    s = make_solver(D, M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130),
                    threads=threads, kfreeze=kf)
    made = m["segs"] if plan == "kept" else s.plan(z[0], z[1], 1e-32, 0.05)
    same = None
    if jet_segs is not None:
        same = json.dumps(jet_segs) == json.dumps(made)
        print("  plan %s: the series' stored plan %s the %s" % (plan, "equals" if same else "DIFFERS FROM",
                                                                "state's" if plan == "kept" else "one Solver.plan makes"),
              flush=True)
    use = jet_segs if jet_segs is not None else made
    print("  plan %s: %d steps" % (plan, sum(len(sg) - 1 for sg in use)), flush=True)
    s.set_segs(use)
    L = len(z[0])
    t0 = time.time()
    out = s.newton_o(z)
    zero = np.zeros(L)

    def solve(b, transpose=False):
        x = s.olu_solve([np.ascontiguousarray(b)] + [zero] * 7, transpose)
        return np.asarray(x[0]) + np.asarray(x[1])

    V, hist = subspace(solve, L, 4, its)
    sig, conv = hist[-1], np.abs(hist[-1] - hist[-2]) / hist[-1]
    print("  plan %s: L %d, LU %.0f s, pivots %.1e..%.1e; sigma %s (converged to %s)" % (
        plan, L, time.time() - t0, out[3], out[4], " ".join("%.3e" % x for x in sig),
        " ".join("%.0e" % x for x in conv)), flush=True)
    ratio = None
    if check:
        # near_null.py's check: v_1 refined in octo-double, ||J v_1|| from the exact Jacobian at 140 digits
        import mpmath as mp
        from multidouble import od_axpy
        mp.mp.dps = 140
        x8 = [np.ascontiguousarray(V[:, 0])] + [zero] * 7
        for _ in range(3):
            y = s.olu_solve([np.ascontiguousarray(p) for p in x8], True)
            y = s.olu_solve([np.ascontiguousarray(p) for p in y])
            nrm = float(np.linalg.norm(np.asarray(y[0])))
            x8 = [np.ascontiguousarray(p) for p in od_axpy(1.0 / nrm, *[np.asarray(p) for p in y], *([zero] * 8))]
        J = [np.asarray(p) for p in s.jacobian_o(z)]
        xv = [mp.fsum(mp.mpf(float(x8[q][c])) for q in range(8)) for c in range(L)]
        nz = [c for c in range(L) if xv[c] != 0]
        jv = [mp.fsum(mp.fsum(mp.mpf(float(J[q][r, c])) for q in range(8) if J[q][r, c] != 0) * xv[c] for c in nz)
              for r in range(L)]
        jn = mp.sqrt(mp.fsum(a * a for a in jv)) / mp.sqrt(mp.fsum(a * a for a in xv))
        ratio = float(jn / sig[0])
        print("    ||J v_1|| / sigma_1 = %.6f (exact Jacobian, 140 digits)" % ratio, flush=True)
    return V, sig, conv, (out[3], out[4]), ratio, same


def split(dz, z, V):
    """per order: |dz|, |dz| / |z|, the share of dz on span(V), V^T dz, and the share of z on span(V)"""
    rows = []
    for k in range(len(dz)):
        n = np.linalg.norm(dz[k])
        c = V.T @ dz[k]
        nzk = np.linalg.norm(z[k])
        rows.append(dict(k=k, norm=float(n), rel=float(n / nzk) if nzk else 0.0,
                         share=float(np.linalg.norm(c) / n) if n else 0.0, vt=[float(x) for x in c],
                         z_share=float(np.linalg.norm(V.T @ z[k]) / nzk) if nzk else 0.0))
    return rows


def main():
    if len(pos) < 2:
        sys.exit(__doc__)
    raw, D0 = pos[0], pos[1]
    threads, its = int(args.get("threads", 4)), int(args.get("its", 16))
    st = "I603-win2/chain/anm_pin3_D%s.npz" % D0
    zk, mk, dk, ksk = load_series(os.path.join(raw, "I685-dh370/repro/jet_pin3_D%s.npz" % D0))
    zn, mn, _, _ = load_series(os.path.join(raw, "I686-dh370/newplan/jet_pin3_D%s.npz" % D0))
    print("D0 %s: series of %d orders, L %d; kept plan from I685-dh370/repro, new plan from I686-dh370/newplan" % (
        D0, zk.shape[0], zk.shape[1]), flush=True)
    rec = dict(D0=D0, plans={})
    Vs = {}
    for plan in ("kept", "new"):
        V, sig, conv, piv, ratio, same = weak(raw, st, plan, threads, its, check=(plan == "kept"),
                                              jet_segs=(mk if plan == "kept" else mn).get("segs"))
        Vs[plan] = V
        rows = split(zn - zk, zk, V)
        rec["plans"][plan] = dict(sigma=[float(x) for x in sig], conv=[float(x) for x in conv], pivots=list(piv),
                                  ratio=ratio, rows=rows, plan_reproduced=same)
        print("    k   |dz|/|z|   share on span(V)   V^T dz / |dz|                        z_k's share")
        for r in rows:
            if r["k"] in K_ISSUE or r["k"] in (1, 2, 24):
                print("    %2d  %.2e   %.4f             %s   %.4f" % (
                    r["k"], r["rel"], r["share"], " ".join("%+.3f" % (x / r["norm"]) for x in r["vt"]),
                    r["z_share"]), flush=True)
    rec["overlap"] = float(np.linalg.norm(Vs["kept"].T @ Vs["new"]) / 2)     # |V_k^T V_n|_F / sqrt(4)
    print("  the two plans' weak spans: |V_kept^T V_new|_F / 2 = %.6f" % rec["overlap"])
    # planted: a known change along the kept plan's v_2, through the same loading and split
    V = Vs["kept"]
    dz = zn - zk
    planted = {k: d.copy() for k, d in ((k, np.asarray(dk[k])) for k in ksk)}
    want = {}
    for k in K_ISSUE:
        g = np.random.default_rng(736 + k).standard_normal(V.shape[0])
        g -= V @ (V.T @ g)
        g /= np.linalg.norm(g)
        P = np.linalg.norm(dz[k]) * (0.6 * V[:, 1] + 0.8 * g)
        planted[ksk[0]][k] = planted[ksk[0]][k] + P
        want[k] = np.linalg.norm(dz[k])
    path = os.path.join(args.get("tmp", "."), "planted736_D%s.npz" % D0)
    np.savez(path, **planted, meta=json.dumps(mk))
    zp, _, _, _ = load_series(path)
    os.replace(path, path + ".used")
    Vp = V.copy()
    if args.get("mutate") == "V":
        r = np.random.default_rng(1).standard_normal(V.shape[0])
        Vp[:, 1] = r / np.linalg.norm(r)
    prow = split(zp - zk, zk, Vp)
    err = 0.0
    for k in K_ISSUE:
        r = prow[k]
        target = np.array([0.0, 0.6 * want[k], 0.0, 0.0])
        err = max(err, abs(r["share"] - 0.6) / 0.6, np.linalg.norm(np.array(r["vt"]) - target) / (0.6 * want[k]))
    ok_p = err <= 1e-8
    print("  planted: share 0.6 along v_2 at k 5-13, returned to %.1e relative: %s%s" % (
        err, "pass" if ok_p else "FAIL", " (--mutate=V)" if args.get("mutate") else ""))
    ratio = rec["plans"]["kept"]["ratio"]
    ok_c = ratio is not None and abs(ratio - 1) < 1e-2 and rec["plans"]["kept"]["conv"][0] < 1e-3
    print("check:", "pass" if ok_p and ok_c else "FAIL", "(planted %s, ||J v_1|| / sigma_1 %s)" % (
        "pass" if ok_p else "FAIL", "pass" if ok_c else "FAIL"))
    rec["planted"] = dict(err=float(err), ok=bool(ok_p), mutate=args.get("mutate", ""))
    if "json" in args:
        json.dump(rec, open(args["json"], "w"), indent=1)
        np.savez(args["json"].replace(".json", "_V.npz"), V_kept=Vs["kept"], V_new=Vs["new"])
    return 0 if ok_p and ok_c else 1


if __name__ == "__main__":
    sys.exit(main())
