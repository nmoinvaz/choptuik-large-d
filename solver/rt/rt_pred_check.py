"""The continuation predictor of rt_run.py (rt_run.predict, Lagrange extrapolation in D) on stored solutions: for a
target D with a stored solution, the max|F| of the RT residual at the predicted state and the predictor's error
max|z_pred - z| and Delta(z_pred) - Delta(z), for steps h and orders 1 (EEG's linear predictor) to 3.

    python3 solver/rt/rt_pred_check.py DIR TAG --grid 32x101 [--targets 30,29] [--steps 0.5,1,2] [--mutate=order]

DIR/TAG_D{D:.4f}_{grid}.npz are rt_run.py states (zh + zl is used). Planted check first: on a path z(D) that is a
polynomial of degree 3 in D, the order-3 prediction must be exact to 1e-12 relative and the order-2 one must not be;
exits 1 otherwise. --mutate=order passes order 2 where 3 is meant, so the check must exit 1.
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_model import RT
from rt_run import predict


def planted(mutate=None):
    rng = np.random.default_rng(145)
    c = rng.normal(size=(4, 50))
    path = lambda D: c[0] + c[1] * D + c[2] * D ** 2 + c[3] * D ** 3            # noqa: E731
    hist = [(D, path(D)) for D in (27.0, 28.0, 29.0, 30.0)]
    zt = path(31.0)
    scale = np.abs(zt).max()
    e3 = np.abs(predict(hist, 31.0, 2 if mutate == "order" else 3) - zt).max() / scale
    e2 = np.abs(predict(hist, 31.0, 2) - zt).max() / scale
    ok = e3 < 1e-12 and e2 > 1e-6
    print(f"planted cubic path: order 3 error {e3:.1e} (< 1e-12), order 2 error {e2:.1e} (> 1e-6): "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("tag")
    ap.add_argument("--grid", default="32x101")
    ap.add_argument("--targets", default="30,29")
    ap.add_argument("--steps", default="0.5,1,2")
    ap.add_argument("--mutate", default=None, choices=["order"])
    a = ap.parse_args()
    if not planted(a.mutate):
        sys.exit(1)
    Nt, N = (int(v) for v in a.grid.split("x"))

    def load(D):
        p = os.path.join(a.dir, f"{a.tag}_D{D:.4f}_{Nt}x{N}.npz")
        if not os.path.exists(p):
            return None
        d = np.load(p)
        return d["zh"] + d["zl"]

    print("| target D | step h | order | max abs F at the prediction | max abs (z_pred - z) | Delta(z_pred) - Delta(z) |")
    print("|---|---|---|---|---|---|")
    for T in [float(v) for v in a.targets.split(",")]:
        zt = load(T)
        if zt is None:
            print(f"| {T:g} | no stored solution | | | | |")
            continue
        rt = RT(T, Nt, N)
        for h in [float(v) for v in a.steps.split(",")]:
            for order in (1, 2, 3):
                pts = [round(T - k * h, 4) for k in range(order + 1, 0, -1)]
                zs = [load(D) for D in pts]
                if any(z is None for z in zs):
                    continue
                zg = predict(list(zip(pts, zs)), T, order)
                F = np.abs(rt.residual(zg)[0]).max()
                print(f"| {T:g} | {h:g} | {order} | {F:.1e} | {np.abs(zg - zt).max():.1e} | "
                      f"{rt.delta(zg)[0] - rt.delta(zt)[0]:+.1e} |")
    print("exit 0")


if __name__ == "__main__":
    main()
