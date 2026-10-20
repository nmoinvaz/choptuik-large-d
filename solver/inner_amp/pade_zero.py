"""Zeros of S_1(x)/S_1^(1) = 1 + sigma_1 x + sigma_2 x^2 + ... (x = eps^2) from Pade approximants of the amplitude series.
    python3 pade_zero.py sig_*.json"""
import sys, json
import numpy as np
import mpmath as mp
for fn in sys.argv[1:]:
    D = json.load(open(fn))
    sig = [complex(a, b) for a, b in D["sigma"]]
    N = len(sig) - 1
    print(f"{fn}: wf {D['wf']}, sigma_0..{N}")
    for L in range(1, N + 1):
        M = N - L
        if M < 0: continue
        try:
            p, q = mp.pade([mp.mpc(z) for z in sig], L, M)
        except Exception as e:
            continue
        z = mp.polyroots(p[::-1]) if L > 0 else []
        zs = sorted([complex(r) for r in z], key=abs)
        print(f"   [{L}/{M}] zeros in x = eps^2: " + ", ".join(f"{r.real:.4f}{r.imag:+.4f}i (eps {np.sqrt(r):.4f})" for r in zs[:3]))
