"""Does the tau-analyticity strip of the l = 1 centre data close near the threshold D_c ~ 23.08 (#1103)?

usage: python i1103_strip.py KC87_DIR HI_DIR OUT_JSON [--planted] [--mutate=NAME]
  KC87_DIR: states of one fixed-k_c leg (k_c 87, D 21.28 -> 23.0); HI_DIR: k_c 127/135 states at D 22.875 - 23.125.
  File names give k_c and D. --planted runs the planted tests and stops; --mutate=flipD reverses the D axis of the
  closing-point fit, --mutate=nowin lets the fit window follow k_c (0.25 - 0.85 k_c, the window of i1103_spectrum.py);
  each must make a planted test fail.

Estimators (planted tests below):
  strip(A, k, window)  delta = beta / omega from a geometric fit ln E_k = a - beta k on the upper envelope E over a
                       fixed window of k: for an analytic function with nearest complex singularity at |Im tau| = d,
                       |c_k| ~ k^-s exp(-k omega d), so beta / omega -> d as the window moves out (Darboux); on a
                       fixed window it is a consistent proxy for d across D, biased by the power k^-s.
  close_point(D, d)    D_0 where the fitted d(D) reaches 0: d = A (D_0 - D)^nu (nu free, and nu = 1/2, 1 fixed), by
                       least squares on ln d; reports D_0 and the rms. A singularity forming at D_c needs D_0 ~ D_c.
"""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD          # noqa: E402

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")


def envelope(A, w=4):
    return np.array([A[max(0, i - w):i + w + 1].max() for i in range(len(A))])


def strip(k, A, klo, khi, omega, kc=None):
    if MUT == "nowin" and kc is not None:
        klo, khi = int(0.25 * kc), int(0.85 * kc)
    E = envelope(A)
    s = (k >= klo) & (k <= khi)
    c = np.polyfit(k[s].astype(float), np.log(E[s]), 1)
    r = np.polyval(c, k[s]) - np.log(E[s])
    return float(-c[0] / omega), float(np.sqrt(np.mean(r * r)))


def close_point(D, d, nu=None):
    """fit d = A (D0 - D)^nu; nu None: free. Grid in D0 (above max D), linear least squares in (ln A, nu) per D0."""
    D, d = np.asarray(D, float), np.asarray(d, float)
    if MUT == "flipD":
        D = D[::-1]
    best = (np.inf, None, None, None)
    for D0 in max(D) + np.geomspace(1e-3, 200.0, 4000):
        x = np.log(D0 - D)
        if nu is None:
            X = np.stack([np.ones_like(x), x], 1)
            c, *_ = np.linalg.lstsq(X, np.log(d), rcond=None)
            r = X @ c - np.log(d)
            if c[1] < 0:
                continue
        else:
            c0 = np.mean(np.log(d) - nu * x)
            c = np.array([c0, nu]); r = c0 + nu * x - np.log(d)
        rms = float(np.sqrt(np.mean(r * r)))
        if rms < best[0]:
            best = (rms, float(D0), float(c[1]), float(np.exp(c[0])))
    return dict(rms=best[0], D0=best[1] if best[1] is not None else float("nan"), nu=best[2], A=best[3])


def planted():
    ok = True
    rng = np.random.default_rng(1103)
    # (a) the strip estimator on a function with poles at tau = tau_j +- i d (d = 0.01, period 1.17): the fixed window
    # [41, 81] must give d within 15%
    Delta = 1.17; om = 2 * np.pi / Delta; N = 2048
    th = 2 * np.pi * np.arange(N) / N
    d = 0.01
    y = sum(w * np.sinh(om * d) / (np.cosh(om * d) - np.cos(th - t)) for w, t in ((1.0, 0.4), (0.6, 2.1), (0.3, 4.0)))
    y = y + 0.4 * np.cos(3 * th)
    A = np.abs(np.fft.rfft(y)) / N * 2
    k = np.arange(len(A))
    de, rms = strip(k, A, 41, 81, om, kc=N // 2)
    pa = abs(de / d - 1) < 0.15
    print(f"planted strip: d 0.01 -> {de:.5f} (rms {rms:.2f}) -> {'pass' if pa else 'FAIL'}")
    ok &= pa
    # (b) closing point: d = 0.006 ((23.08 - D) / 1.8)^0.5 with 2% noise on the k_c 87 leg's D grid: D0 within 0.05
    D = np.arange(21.28, 23.001, 1 / 32)
    dd = 0.006 * ((23.08 - D) / 1.8) ** 0.5 * (1 + 0.02 * rng.standard_normal(len(D)))
    r = close_point(D, dd)
    pb = abs(r["D0"] - 23.08) < 0.05
    print(f"planted closing point, D_c 23.08, nu 0.5: D0 {r['D0']:.3f} nu {r['nu']} -> {'pass' if pb else 'FAIL'}")
    # (c) the same with a strip that does not close near D_c: d = 0.005 + 0.0016 (23 - D), zero at 26.1; the fit must
    # put D0 above 25
    dd = (0.005 + 0.0016 * (23 - D)) * (1 + 0.02 * rng.standard_normal(len(D)))
    r = close_point(D, dd)
    pc = r["D0"] > 25
    print(f"planted closing point, none near D_c (linear, zero at 26.1): D0 {r['D0']:.3f} nu {r['nu']} -> "
          f"{'pass' if pc else 'FAIL'}")
    ok &= pb and pc
    return ok


def load(path):
    m = re.search(r"kc(\d+)[^_]*_D([0-9.]+)\.npy$", os.path.basename(path))
    kc, D = int(m.group(1)), float(m.group(2))
    z = np.load(path)
    S = DSSEll(kc=kc, nodes=NODES_STD, n=D - 2, ell=1, Delta_ref=z[-1])
    assert len(z) == S.layout()[2], path
    return S, z, kc, D


def spectra(path):
    S, z, kc, D = load(path)
    f = S.fields(z)
    om = 2 * np.pi / f["Delta"]
    out = dict(file=os.path.basename(path), D=D, kc=kc, Delta=f["Delta"], omega=om, c0min=float(f["c0"].min()),
               constraint=S.constraint(z))
    k = np.arange(S.N // 2 + 1)
    for nm, y, par in (("q", f["d"], 1), ("c0", f["c0"], 0), ("p02", f["states"][0][0], 1),
                       ("p05", f["states"][1][0], 1), ("p10", f["states"][2][0], 1)):
        A = np.abs(np.fft.rfft(y)) / len(y) * 2
        kk = k[(k % 2 == par) & (k <= kc) & (k > 0)]
        out[nm] = dict(k=[int(v) for v in kk], A=[float(v) for v in A[kk]], max=float(A[kk].max()))
    return out


def main():
    ok = planted()
    print("planted tests:", "PASS" if ok else "FAIL", flush=True)
    if "--planted" in sys.argv or not ok:
        sys.exit(0 if ok else 1)
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    leg = sorted((spectra(os.path.join(a[0], fn)) for fn in os.listdir(a[0]) if fn.endswith(".npy")),
                 key=lambda r: r["D"])
    hi = sorted((spectra(os.path.join(a[1], fn)) for fn in os.listdir(a[1]) if fn.endswith(".npy")),
                key=lambda r: (r["kc"], r["D"]))
    res = dict(leg=leg, hi=hi, fits={})
    win = {"q": (21, 61), "c0": (20, 60), "p02": (21, 61), "p05": (21, 61), "p10": (21, 61)}
    print("\nk_c 87 leg: delta = beta/omega on fixed windows (q, p: [21, 61]; c0: [20, 60])")
    print("     D     Delta    c0min  constraint |  q        c0       p(0.02)  p(0.05)  p(0.1)")
    for r in leg:
        row = []
        for nm in ("q", "c0", "p02", "p05", "p10"):
            k, A = np.array(r[nm]["k"]), np.array(r[nm]["A"])
            lo, hi_ = win[nm]
            dlt, rms = strip(k, A, lo, hi_, r["omega"], kc=r["kc"])
            r[nm]["delta"], r[nm]["rms"] = dlt, rms
            row.append(f"{dlt:.5f}")
        print(f"  {r['D']:8.5f} {r['Delta']:.6f} {r['c0min']:.4f} {r['constraint']:.2e} | " + "  ".join(row))
    D = np.array([r["D"] for r in leg])
    print("\nclosing point of delta(D) on the leg, fit d = A (D0 - D)^nu")
    for nm in ("q", "c0", "p02", "p05", "p10"):
        d = np.array([r[nm]["delta"] for r in leg])
        fr = close_point(D, d); f1 = close_point(D, d, 1.0); fh = close_point(D, d, 0.5)
        res["fits"][nm] = dict(free=fr, nu1=f1, nu05=fh, dlnd_dD=float(np.polyfit(D, np.log(d), 1)[0]))
        g = lambda v, f=".3f": "none" if v is None or v != v else format(v, f)
        print(f"  {nm:4s} nu free: D0 {g(fr['D0'])} nu {g(fr['nu'])} rms {g(fr['rms'], '.4f')}; nu = 1: D0 "
              f"{g(f1['D0'])} rms {g(f1['rms'], '.4f')}; nu = 1/2: D0 {g(fh['D0'])} rms {g(fh['rms'], '.4f')}; "
              f"d ln(delta)/dD {res['fits'][nm]['dlnd_dD']:+.4f} (a closing strip needs a negative slope)")
    print("\nhigh k_c across D_c: delta on windows [41, 81], [61, 101], and the envelope at fixed k relative to max")
    for r in hi:
        row = []
        for nm in ("q", "c0", "p02"):
            k, A = np.array(r[nm]["k"]), np.array(r[nm]["A"])
            for lo, hi_ in ((41, 81), (61, 101)):
                dlt, rms = strip(k, A, lo, hi_, r["omega"], kc=r["kc"])
                r[nm][f"delta_{lo}_{hi_}"] = dlt
                row.append(f"{nm} [{lo},{hi_}] {dlt:.5f}")
            E = envelope(A) / r[nm]["max"]
            r[nm]["E_rel"] = {int(kk): float(E[list(k).index(kk)]) for kk in k if kk in (41, 61, 81, 101, 121)}
        print(f"  kc {r['kc']} D {r['D']:.4f} Delta {r['Delta']:.8f} constraint {r['constraint']:.2e}: " + "; ".join(row))
        print("      q E/max at k 41, 61, 81, 101, 121: " + " ".join(f"{v:.2e}" for v in r["q"]["E_rel"].values()))
    with open(a[2], "w") as fh_:
        json.dump(res, fh_)
    print("\nwrote", a[2])


if __name__ == "__main__":
    main()
