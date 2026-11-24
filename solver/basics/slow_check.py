"""Stored large-D states against the leading-order slow solution (#1086, notes/LARGE_D_BASICS.md 3.2).

At leading order in 1/n, for any centre profile p1(theta) (K = 2 p1^2) and any Delta,
  v = p - m = 2 p1 x / sqrt(1 + K x^2),  c = c0 sqrt(1 + K x^2),  c0 = 1/sqrt(1 + K),
  p + m = O(1/n),  (n - 1)(a^2 - 1) = K x^2 / (1 + K x^2)
(derive.py check 3). This prints per state: the range of c0^2 (1 + 2 p1^2) (1 at leading order), the amplitude
n <p1^2>, the odd harmonics of p1 and the even harmonics of c0, and per node max|p + m|, max|p - m| against the
prediction, mean (n-1)(a^2 - 1) against the prediction, and the range of c over its prediction.

Usage: python3 slow_check.py STATE.npz [...] [--mutate=K2]
  --mutate=K2: control; uses K = 4 p1^2, which must break the c0 and v checks by O(K).
"""
import json
import sys

import numpy as np


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    fac = 4 if opts.get("mutate") == "K2" else 2
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        y = z["zh"].astype(float) + z["zl"].astype(float)
        M, nodes, D = meta["M"], meta["nodes"], meta["D"]
        n = D - 2
        p1, c0 = y[:M], y[M:2 * M]
        K = fac * p1 ** 2
        r = c0 ** 2 * (1 + K)
        print("D %.4f Delta %.12f  c0^2(1+2p1^2) %.5f..%.5f  n<p1^2> %.3f" % (D, y[-1], r.min(), r.max(),
                                                                              n * (p1 ** 2).mean()))
        full = np.concatenate([p1, -p1])
        F = np.abs(np.fft.rfft(full)) / M
        print("  |p1| harmonics 1..15: " + " ".join("%d:%.2e" % (k, F[k]) for k in range(1, 16, 2)))
        G = np.abs(np.fft.rfft(np.concatenate([c0, c0]))) / M
        print("  |c0| harmonics 0..30: " + " ".join("%d:%.2e" % (k, G[k]) for k in range(0, 31, 2)))
        for i, x in enumerate(nodes):
            o = 2 * M + 4 * M * i
            p, m, a, c = (y[o + j * M: o + (j + 1) * M] for j in range(4))
            v_pred = 2 * p1 * x / np.sqrt(1 + K * x ** 2)
            s_pred = K * x ** 2 / (1 + K * x ** 2)
            cr = c / (c0 * np.sqrt(1 + K * x ** 2))
            print("  x %.2f  max|p+m| %.2e  max|p-m| %.4f pred %.4f  <(n-1)(a^2-1)> %.5f pred %.5f  c/c_pred %.5f..%.5f"
                  % (x, np.abs(p + m).max(), np.abs(p - m).max(), np.abs(v_pred).max(),
                     ((n - 1) * (a ** 2 - 1)).mean(), s_pred.mean(), cr.min(), cr.max()))


if __name__ == "__main__":
    main()
