#!/usr/bin/env python3
"""The amplitude expansion of hier.py's M-harmonic model: odd harmonics |mu| <= M (y, sigma, F) and even harmonics
|nu| <= M + 1 (c, E), every product truncated to that space, harmonics 3..M tuned (S_nu = 0 order by order), late-order
fits with every neighbour family present.

Why a second truncated expansion: amp_trunc.py (B_tail.md section 4) truncates *every* function at |mu| <= M, so its
even harmonics (c, E, Pi) stop at M - 1, one below inner_np.py / hier.py, which keep the even harmonic M + 1. Its first
change is at sigma_M, exactly where the even harmonic M + 1 enters, and it then explodes at sigma_{M+1}. The model
hier.py actually solves is AmpTrunc with H = M + 1: amp_flint's truncation |mu| <= H keeps the odd harmonics <= M (M + 1
is even) and the even ones <= M + 1. Order counting: in this model sigma_1 .. sigma_M are exact, and the first
omitted harmonic (the profile/y harmonic M + 2) enters at sigma_{M+1}. So Delta sigma_{M+1} = sigma_{M+1} -
sigma^{[M]}_{M+1} is the lowest-order term of the truncation increment delta_M = s_{M+2} - s_M at fixed (A, w).

    python3 amp_trunc_hiermodel.py M K J [wf]   writes trunc_hier_M<M>_K<K>_J<J>_w<wf>.json (sigma_k, tuned f)
"""
import sys
import os
import json
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amp_trunc import AmpTrunc  # noqa: E402

if __name__ == "__main__":
    M, K, J = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    wf = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    a = AmpTrunc(wf, K, J, M + 1)
    sig, f = a.run()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"trunc_hier_M{M}_K{K}_J{J}_w{wf:.4f}.json")
    json.dump({"M": M, "H": M + 1, "wf": wf, "K": K, "J": J,
               "sigma_str": [[mp.nstr(z.real, 20), mp.nstr(z.imag, 20)] for z in sig], "f": f}, open(out, "w"))
    print("wrote", out)
