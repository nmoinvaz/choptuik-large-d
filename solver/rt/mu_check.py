"""RT's echoing parameter mu from our horizon data: Delta = 4 pi mu / (1 - (n - 1) <a_h^2 - 1>) (gg_to_rt.py), against
Reiterer and Trubowitz's D = 4 enclosures of mu and K = Delta / 2 (arXiv:1203.3766v1, Eqs. (2), (3), radius 1e-80).

    python3 solver/rt/mu_check.py STATE.npz [--mutate=n]      (exit 1 when |mu - mu_RT| > --tol, default 1e-20)

STATE.npz: an hp double-double state at D = 4 (zh, zl, meta), e.g. folder T45-win/hp/data/lowD_states/ of the raw-data
repository (d4hm15_D4.0000_M192.npz). <.> is the mean over the M points of the half tau grid (exact for the
trigonometric polynomial the state carries). --mutate=n uses n in place of n - 1 and must fail.
"""
import sys
import json
import numpy as np
import mpmath as mp

mp.mp.dps = 40                     # before the constants below are parsed

MU_RT = mp.mpf("0.16830707896344996951013497904285742072100199080892966476395293134873313662587505")
K_RT = mp.mpf("1.72272620111391067498575597278819186222103458805781088634788403570540271455648855")


def main():
    mp.mp.dps = 40
    path = sys.argv[1]
    mut = "--mutate=n" in sys.argv
    tol = 1e-20
    for a in sys.argv:
        if a.startswith("--tol="):
            tol = float(a.split("=")[1])
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    M, n = meta["M"], mp.mpf(meta["D"]) - 2
    z = [mp.mpf(float(h)) + mp.mpf(float(l)) for h, l in zip(d["zh"][-1 - 2 * M:], d["zl"][-1 - 2 * M:])]
    Delta, ah = z[-1], z[M:2 * M]
    k = n if mut else n - 1
    avg = mp.fsum([a * a - 1 for a in ah]) / M
    mu = Delta * (1 - k * avg) / (4 * mp.pi)
    dmu, dK = mu - MU_RT, Delta / 2 - K_RT
    ok = abs(dmu) <= tol
    print(f"{path}: D {meta['D']} M {M}  Delta {mp.nstr(Delta, 25)}  <a_h^2 - 1> {mp.nstr(avg, 20)}")
    print(f"  mu {mp.nstr(mu, 25)}  mu - mu_RT {mp.nstr(dmu, 3)}  Delta/2 - K_RT {mp.nstr(dK, 3)}  "
          f"{'ok' if ok else 'FAIL'} (tol {tol:g}){' (mutation n)' if mut else ''}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
