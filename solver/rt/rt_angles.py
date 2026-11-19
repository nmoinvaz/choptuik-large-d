"""Principal angles between the deep spans of the RT Jacobian at several root samples (#1006's recommendation, P7).

    python3 solver/rt/rt_angles.py XA.npz XB.npz [XC.npz ...] [--d 4] [--out OUT.json]
    python3 solver/rt/rt_angles.py X.npz --complement ROOT.npz --chords A.npz,B.npz --write Q.npz [--d 4]
    python3 solver/rt/rt_angles.py --selftest [--mutate=replace]
    python3 solver/rt/rt_angles.py --selftest --complement [--mutate=noproj]

Each X.npz is rt_sigma.py --save_x's file (rt_xcount.read_x). Per sample, span(p_1 .. p_d) is the top-d left singular
span of its x's (rt_xcount.count; d is each file's own count unless --d is given). For every pair the cosines of the
principal angles are the singular values of B_a^T B_b (Bjorck and Golub, Math. Comp. 27 (1973) 579). If the samples'
deep directions are one subspace, every cosine is near 1. A direction seen at one sample only gives a cosine at the
level of a random vector, about sqrt(d/L).

--complement: the directions of X's deep span orthogonal to the plane of the chords A - ROOT and B - ROOT (formed in
double-double, rt_perturb.diff, then projected into the span): an orthonormal basis q_1 .. q_(d-2), written to Q.npz as
q (rows), for rt_perturb.py --dir. Printed: each chord's share in the span, and each q's cosine with the chords (must
be below 1e-9) and its part outside the span (must be below 1e-9). --selftest --complement plants a d = 4 span and two
chords inside it and checks both; --mutate=noproj skips removing the plane, so the q's keep the chords' directions and
the selftest must fail.

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


def complement(B, chords, noproj=False):
    """an orthonormal basis of span(B) orthogonal to the chords' projections into span(B)"""
    C = B @ (B.T @ np.column_stack(chords))
    E = np.linalg.qr(C)[0]
    Bp = B if noproj else B - E @ (E.T @ B)
    U, sv, _ = np.linalg.svd(Bp, full_matrices=False)
    k = B.shape[1] - (0 if noproj else len(chords))
    return U[:, :k]


def complement_checks(B, Q, chords):
    cos = max(abs(float(q @ u)) / (np.linalg.norm(q) * np.linalg.norm(u)) for q in Q.T for u in chords)
    out = max(float(np.linalg.norm(q - B @ (B.T @ q)) / np.linalg.norm(q)) for q in Q.T)
    return cos, out


def selftest_complement(noproj):
    rng = np.random.default_rng(1018)
    L, d = 400, 4
    B = np.linalg.qr(rng.normal(size=(L, d)))[0]
    chords = [B @ rng.normal(size=d), B @ rng.normal(size=d)]
    Q = complement(B, chords, noproj)
    cos, out = complement_checks(B, Q, chords)
    ok = cos < 1e-9 and out < 1e-9
    print(f"selftest --complement{' --mutate=noproj' if noproj else ''}: {Q.shape[1]} directions, largest cosine with "
          f"the chords {cos:.1e}, part outside the span {out:.1e}: {'PASS' if ok else 'FAIL'}")
    return ok


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
    ap.add_argument("--complement", nargs="?", const="", default=None)
    ap.add_argument("--chords", default="")
    ap.add_argument("--write", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", default="")
    a = ap.parse_args()
    if a.selftest:
        if a.complement is not None:
            return 0 if selftest_complement(a.mutate == "noproj") else 1
        return 0 if selftest(a.mutate == "replace") else 1
    if a.complement:
        import rt_perturb as RP
        (p, k, B, s), = spans(a.x[:1], a.d)
        r = np.load(a.complement)
        zh, zl = np.asarray(r["zh"], dtype=float), np.asarray(r["zl"], dtype=float)
        chords = []
        for c in a.chords.split(","):
            o = np.load(c)
            u = RP.diff((np.asarray(o["zh"], dtype=float), np.asarray(o["zl"], dtype=float)), zh, zl)
            chords.append(u)
            print(f"chord {os.path.basename(c)}: share in the d = {k} span {XC.share_in_span(B, k, u):.6f}")
        Q = complement(B, chords)
        cos, out = complement_checks(B, Q, chords)
        ok = cos < 1e-9 and out < 1e-9
        print(f"{Q.shape[1]} directions out of the chords' plane: largest cosine with the chords {cos:.1e}, part "
              f"outside the span {out:.1e}: {'ok' if ok else 'NOT OK'}")
        if a.write:
            np.savez(a.write, q=Q.T, x=p, root=a.complement, chords=a.chords)
            print(f"wrote {a.write}")
        return 0 if ok else 1
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
