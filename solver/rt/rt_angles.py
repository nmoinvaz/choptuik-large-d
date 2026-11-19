"""Principal angles between the deep spans of the RT Jacobian at several root samples (#1006's recommendation, P7).

    python3 solver/rt/rt_angles.py XA.npz XB.npz [XC.npz ...] [--d 4] [--out OUT.json]
    python3 solver/rt/rt_angles.py --selftest [--mutate=replace]

Each X.npz is rt_sigma.py --save_x's file (rt_xcount.read_x). Per sample, span(p_1 .. p_d) is the top-d left singular
span of its x's (rt_xcount.count; d is each file's own count unless --d is given). For every pair the cosines of the
principal angles are the singular values of B_a^T B_b (Bjorck and Golub, Math. Comp. 27 (1973) 579). If the samples'
deep directions are one subspace, every cosine is near 1. A direction seen at one sample only gives a cosine at the
level of a random vector, about sqrt(d/L).

--selftest plants one d = 4 span in L = 400, writes three samples whose x's are random combinations of it plus 1e-6
relative noise (the fourth direction is 1e-3 of the first, so its angle moves by about 5e-6), and requires every
cosine >= 1 - 1e-4. --mutate=replace swaps one planted direction of the third
sample for a random one: its smallest cosines with the others must fall below 0.5, so the selftest must fail.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rt_xcount as XC  # noqa: E402


def spans(paths, d=None):
    out = []
    for p in paths:
        X, V, _, _ = XC.read_x(p)
        res = XC.count(X, V)
        k = d or res["d"]
        out.append((p, k, res["P"][:, :k], res["s_over_s1"][:k + 1]))
    return out


def cosines(Ba, Bb):
    return np.linalg.svd(Ba.T @ Bb, compute_uv=False)


def selftest(mutate):
    rng = np.random.default_rng(1006)
    L, d, n = 400, 4, 12
    U = np.linalg.qr(rng.normal(size=(L, d)))[0]
    tmp = []
    for i in range(3):
        Ui = U.copy()
        if mutate and i == 2:
            Ui[:, 3] = np.linalg.qr(np.c_[U[:, :3], rng.normal(size=L)])[0][:, 3]
        X = (Ui * np.logspace(0, -3, d)) @ rng.normal(size=(d, n))       # deep directions of graded size
        X += 1e-6 * np.abs(X).max() * rng.normal(size=X.shape)
        path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"rt_angles_selftest_{os.getpid()}_{i}.npz")
        XC.write_x(path, X.T, np.zeros_like(X.T), np.zeros((n, L)), np.zeros((n, L)), np.ones(n))
        tmp.append(path)
    sp = spans(tmp, d=d)
    worst = min(float(cosines(a[2], b[2]).min()) for a, b in itertools.combinations(sp, 2))
    for p in tmp:
        os.remove(p)
    ok = worst >= 1 - 1e-4
    print(f"selftest{' --mutate=replace' if mutate else ''}: smallest principal cosine {worst:.9f}: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("x", nargs="*")
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", default="")
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest(a.mutate == "replace") else 1
    sp = spans(a.x, a.d)
    L = sp[0][2].shape[0]
    rec = {"samples": [{"x": p, "d": k, "s_over_s1": [float(v) for v in s]} for p, k, _, s in sp], "pairs": []}
    for p, k, _, s in sp:
        print(f"{os.path.basename(p)}: d {k}, s/s1 " + " ".join(f"{v:.2e}" for v in s))
    print(f"random-vector level sqrt(d/L) = {np.sqrt(sp[0][1] / L):.2e} (L {L})")
    for (pa, ka, Ba, _), (pb, kb, Bb, _) in itertools.combinations(sp, 2):
        c = cosines(Ba, Bb)
        print(f"{os.path.basename(pa)} vs {os.path.basename(pb)}: cosines " + " ".join(f"{v:.9f}" for v in c)
              + f"; largest angle {np.degrees(np.arccos(min(1.0, c.min()))):.3g} deg")
        rec["pairs"].append({"a": pa, "b": pb, "cosines": [float(v) for v in c]})
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
