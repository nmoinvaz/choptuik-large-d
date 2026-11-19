"""Count the deep directions of the RT Jacobian from rt_sigma.py's saved rough solutions (issue #983, P7 stage 1, T1).

    python3 solver/rt/rt_xcount.py X.npz [--root ROOT.npz --chords Y1.npz,Y2.npz] [--expect K] [--out OUT.json]
    python3 solver/rt/rt_xcount.py --selftest [--mutate=transpose]

X.npz is written by `rt_sigma.py STATE --rough --starts n --save_x X.npz`: each start's x / max|x| and its v (max|v| = 1)
in double-double parts, and its bound 1 / max|x|, so that x_i = J^-1 v_i is x_hat_i / bound_i. Write the thin SVD of
the L x n matrix of the x_i as X = P S Q^T. A direction of gain sigma_j enters every x_i with the coefficient
(u_j . v_i) / sigma_j, so X has a singular value of about |U_j^T V| / sigma_j for it, and the deep directions are the
singular values at the top:

    d = the number of j with s_j >= 1e-4 s_1,   gap = s_(d+1) / s_d (clean when <= 1e-3),

printed with every s_j / s_1. The SVD is in double precision, so singular values below about 1e-16 s_1 are rounding,
and a gap that reaches that level is printed as "at the rounding level". The gains themselves cannot be read from the
stored x: rounding x to double-double changes J x by about 1e-32 sigma_max max|x|, which is far above v for a gain of
1e-35. So each direction's gain is printed only as the estimate |V|_F / (sqrt(L) s_j), the size of U_j^T V for
random v. It is good to a factor of a few, and the exact gains are rt_sigma.py's own bounds.
With --root and --chords, each chord Y - ROOT (rt_perturb.diff, double-double) is given its 2-norm share in the span of
p_1 .. p_d. --expect K exits 1 unless d = K with a clean gap (for planted files).

--selftest plants a non-normal operator J = U diag(sigma) W^T (L = 400, U and W random orthogonal) with d = 2 and then
d = 4 deep singular values (1e-35 to 1e-34), a pair at 1e-18 below them and the rest in [1e-3, 1], solves x_i =
J^-1 v_i for 12 rough v, writes and reads them through write_x and read_x, and checks the count, the clean gap, the deep
gains' estimates (inside the planted deep interval widened by a factor 5) and the share of planted chords (vectors in the deep right singular
span, as root-set chords are) in span(p_1 .. p_d), which must be at least 0.99. --mutate=transpose solves with J^T
instead, as a solve that confused left and right singular vectors would. The count and the gains then still pass, but
the chords' share falls to the level of a random vector, and the selftest must fail.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEEP, CLEAN = 1e-4, 1e-3


def write_x(path, xh, xl, vh, vl, bounds, **meta):
    """the file format shared with rt_sigma.py --save_x: arrays of shape (n, L)"""
    np.savez(path, xh=np.asarray(xh), xl=np.asarray(xl), vh=np.asarray(vh), vl=np.asarray(vl),
             bounds=np.asarray(bounds, dtype=float), **meta)


def read_x(path):
    d = np.load(path)
    X = ((d["xh"] + d["xl"]) / d["bounds"][:, None]).T          # L x n, x_i = x_hat_i / bound_i
    V = (d["vh"] + d["vl"]).T
    meta = {k: d[k].item() if d[k].ndim == 0 else d[k] for k in d.files if k not in ("xh", "xl", "vh", "vl", "bounds")}
    return X, V, d["bounds"], meta


def count(X, V):
    P, s, Qt = np.linalg.svd(X, full_matrices=False)
    Q = Qt.T
    r = s / s[0]
    d = int(np.sum(r >= DEEP))
    gap = float(s[d] / s[d - 1]) if d < len(s) else None
    gest = np.linalg.norm(V) / np.sqrt(X.shape[0]) / s
    return {"d": d, "gap": gap, "s_over_s1": r, "gain_estimate": gest, "P": P}


def share_in_span(P, d, u):
    B = P[:, :d]
    return float(np.linalg.norm(B.T @ u) / np.linalg.norm(u))


def report(res, n, chords=None, log=print):
    d, gap, r = res["d"], res["gap"], res["s_over_s1"]
    gs = "none (every direction is deep)" if gap is None else (
        f"{gap:.1e} ({'clean' if gap <= CLEAN else 'not clean'}"
        + (", at the rounding level)" if r[d] < 1e-15 else ")"))
    log(f"  deep count d = {d} of {n} (s_j >= {DEEP:g} s_1); gap s_(d+1)/s_d = {gs}")
    log("  s_j / s_1: " + " ".join(f"{v:.1e}" for v in r))
    for j in range(min(n, d + 2)):
        log(f"  direction {j + 1}: s/s_1 {r[j]:.2e}, gain estimate {res['gain_estimate'][j]:.1e}"
            f"{'  (deep)' if j < d else ''}")
    if chords:
        for name, sh in chords.items():
            log(f"  chord {name}: share in span(p_1 .. p_d) {sh:.6f}")


def planted(d, rng, transpose=False, L=400, n=12):
    U = np.linalg.qr(rng.normal(size=(L, L)))[0]
    W = np.linalg.qr(rng.normal(size=(L, L)))[0]
    deep = np.geomspace(1e-35, 1e-34, d)
    sig = np.r_[deep, 1e-18, 2e-18, np.exp(rng.uniform(np.log(1e-3), 0, size=L - d - 2))]
    V = rng.normal(size=(L, n))
    V /= np.abs(V).max(axis=0)
    # J = U diag(sig) W^T, so J^-1 = W diag(1/sig) U^T; the mutation solves with J^T = W diag(sig) U^T instead
    X = (U @ ((W.T @ V) / sig[:, None])) if transpose else (W @ ((U.T @ V) / sig[:, None]))
    bounds = 1 / np.abs(X).max(axis=0)
    xhat = X * bounds
    chords = {f"planted {m}": W[:, :d] @ rng.normal(size=d) for m in range(2)}
    return xhat.T, V.T, bounds, chords, deep


def selftest(mutate):
    rng = np.random.default_rng(983)
    ok = True
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"rt_xcount_selftest_{os.getpid()}.npz")
    for d in (2, 4):
        xh, vh, bounds, chords, deep = planted(d, rng, transpose=(mutate == "transpose"))
        write_x(tmp, xh, np.zeros_like(xh), vh, np.zeros_like(vh), bounds, D=0.0, planted_d=d)
        X, V, b, meta = read_x(tmp)
        res = count(X, V)
        sh = {k: share_in_span(res["P"], res["d"], u) for k, u in chords.items()}
        print(f"selftest, planted d = {d} (deep sigma {deep[0]:.0e} to {deep[-1]:.0e}, a pair at 1e-18, the rest "
              f"in [1e-3, 1]):")
        report(res, X.shape[1], sh)
        g = res["gain_estimate"][:res["d"]]
        t_count = res["d"] == d and res["gap"] is not None and res["gap"] <= CLEAN
        t_gain = bool(np.all(g >= deep[0] / 5) and np.all(g <= deep[-1] * 5))
        t_span = min(sh.values()) >= 0.99
        print(f"  count {'ok' if t_count else 'FAIL'}, deep gain estimates in the planted interval (x5) "
              f"{'ok' if t_gain else 'FAIL'}, "
              f"chords in the deep span {'ok' if t_span else 'FAIL'}")
        ok &= t_count and t_gain and t_span
    os.remove(tmp)
    print(f"selftest{' --mutate=' + mutate if mutate else ''}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("x", nargs="?")
    ap.add_argument("--root", default=None)
    ap.add_argument("--chords", default=None, help="comma-separated sample states; each chord is SAMPLE - ROOT")
    ap.add_argument("--expect", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", default=None, choices=["transpose"])
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest(a.mutate) else 1
    X, V, bounds, meta = read_x(a.x)
    n = X.shape[1]
    print(f"rt_xcount: {a.x}: L {X.shape[0]}, {n} starts; bounds from {bounds.min():.3e} to {bounds.max():.3e}; "
          + ", ".join(f"{k} {v}" for k, v in meta.items()))
    res = count(X, V)
    chords = None
    if a.chords:
        from rt_perturb import diff
        r = np.load(a.root)
        chords = {}
        for p in a.chords.split(","):
            y = np.load(p)
            chords[os.path.basename(p)] = share_in_span(res["P"], res["d"], diff((y["zh"], y["zl"]), r["zh"], r["zl"]))
    report(res, n, chords)
    ok = True
    if a.expect is not None:
        ok = res["d"] == a.expect and res["gap"] is not None and res["gap"] <= CLEAN
        print(f"  expected d = {a.expect} with a clean gap: {'ok' if ok else 'FAIL'}")
    if a.out:
        out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items() if k != "P"}
        out["chords"] = chords
        out["file"] = a.x
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
    print(f"exit {0 if ok else 1}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
