"""Locate the selection S_1(eps^2, omega) = 0 from the amplitude series written by scan_omega.py / amp_flint.py.

For each omega: S_1/S_1^(1) = sum_k sigma_k x^k (x = eps^2). ln S_1 = sum tau_k x^k is computed exactly from the sigma_k;
its derivative L'(x) = S'/S has a simple pole of residue +1 at each zero of S_1. With few terms the ratio estimate
x0 ~ tau_k/tau_(k+1) * k/(k+1) (exact for ln(1 - x/x0)) is printed; with more terms Pade approximants [m/n] of L'(x)
give the poles, accepted when the residue is close to 1 and stable across Pade orders. A zero on the positive real
axis shows up as arg(tau_k) -> pi.
Across omega, the selection is where the accepted zero x0(omega) is real and positive: eps0 = sqrt(x0), omega0.
At the zero the tuned harmonics give F3/F1 and F5/F1.

    python3 zero_selection.py DIR_OR_JSON...     (uses "sigma_str" when present, else "sigma")
"""
import sys
import os
import glob
import json
import mpmath as mp

mp.mp.dps = 40


def load(paths):
    fs = []
    for p in paths:
        fs += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    out = []
    for fn in fs:
        D = json.load(open(fn))
        if "sigma_str" in D:
            sig = [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in D["sigma_str"]]
        else:
            sig = [mp.mpc(a, b) for a, b in D["sigma"]]
        out.append((D["wf"], sig, D.get("f", {})))
    return sorted(out, key=lambda t: t[0])


def log_series(sig):
    N = len(sig)
    L = [mp.mpc(0)] * N
    for k in range(1, N):
        L[k] = (k * sig[k] - sum(j * L[j] * sig[k - j] for j in range(1, k))) / k
    return L


def zeros(sig, verbose=True):
    L = log_series(sig)
    dL = [k * L[k] for k in range(1, len(L))]          # L'(x) coefficients
    found = []
    n = len(dL) - 1
    for m in range(0, n):
        q = n - m
        if q < 1:
            continue
        try:
            p, qq = mp.pade(dL, m, q)
        except ZeroDivisionError:
            continue
        roots = mp.polyroots(qq[::-1], maxsteps=200, extraprec=200) if q > 0 else []
        for r in roots:
            # residue of p/qq at r
            num = mp.polyval(p[::-1], r)
            dq = mp.polyval([c * (len(qq) - 1 - i) for i, c in enumerate(qq[::-1])][:-1], r)
            res = num / dq
            found.append((m, q, r, res))
    if verbose:
        for m, q, r, res in found:
            if abs(res - 1) < 0.3:
                print(f"     [{m}/{q}] zero x = {mp.nstr(r, 8)}  (eps = {mp.nstr(mp.sqrt(r), 6)})  residue {mp.nstr(res, 5)}")
    return found


if __name__ == "__main__":
    data = load(sys.argv[1:])
    W0 = 2 * mp.pi / mp.mpf("1.64389375")
    for wf, sig, f in data:
        print(f"wf {wf:.5f} (omega {mp.nstr(wf * W0, 7)}): {len(sig)} sigma terms; tau = "
              + ", ".join(mp.nstr(t, 4) for t in log_series(sig)[1:]))
        L = log_series(sig)
        for k in range(1, len(L) - 1):
            x0 = L[k] / L[k + 1] * k / (k + 1)          # ln(1 - x/x0) => tau_k = -1/(k x0^k)
            print(f"     ratio estimate (tau_{k}/tau_{k+1}): x0 = {mp.nstr(x0, 7)}  eps0 = {mp.nstr(mp.sqrt(x0), 6)}"
                  f"  arg(tau_{k+1})/pi = {mp.nstr(mp.arg(L[k + 1]) / mp.pi, 5)}")
        zeros(sig)
