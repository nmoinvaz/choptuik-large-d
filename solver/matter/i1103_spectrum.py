"""Tau-harmonic content of stored l = 1 DSSEll states (#1103): decay fits, the dip of c(tau, 0), local bandwidth, and
the k_c a mapped tau grid would need.

usage: python i1103_spectrum.py STATE_DIR OUT_JSON [--planted] [--mutate=NAME]
  STATE_DIR holds sol_l1kc<kc>*_D<D>.npy files (copies of raw-data states; the file name gives kc and D).
  --planted runs the planted tests of every estimator first and stops (exit 1) when one fails. --mutate breaks one
  estimator on purpose (swapfit, nowindow, nomap); its planted test must then fail.

Estimators (each has a planted test below):
  fit_decay   ln|c_k| against k (geometric, rate beta per harmonic) and against ln k (algebraic, power p) on the
              upper envelope of the spectrum over a window of k; the verdict is the model with the smaller rms, and
              the local rates in sub-windows show whether beta or p is the constant one.
  local_band  the largest harmonic whose Gaussian-windowed amplitude exceeds eps, per window centre: k_loc(theta).
              Any smooth map of theta can at best spread the points with density k_loc, so max k_loc / mean k_loc
              bounds the reduction of k_c by a mapped grid (an estimate, not a bound on the solver's error).
  mapped_kc   the field resampled on Boyd's arctan grid theta = g(s) (J. P. Boyd, J. Comput. Phys. 98 (1992) 181),
              g(s) = s0 + atan2(L sin(s - s0), cos(s - s0)), dense at s0 and s0 + pi for L < 1, and the smallest K
              with the s-spectrum below eps past K, against the same K on the uniform grid.
"""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD          # noqa: E402

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")


# ---------------- estimators ----------------
def envelope(A, w=4):
    """running maximum of |c_k| over k +- w (the spectra here are bumpy; the envelope carries the decay)"""
    n = len(A)
    return np.array([A[max(0, i - w):i + w + 1].max() for i in range(n)])


def fit_decay(k, A, klo, khi, w=4):
    """fits on the envelope over klo <= k <= khi: geometric ln A = a - beta k, algebraic ln A = a - p ln k.
    Returns beta, p, rms of each (natural log units), the local beta and p in four sub-windows, and the verdict."""
    E = envelope(A, w)
    sel = (k >= klo) & (k <= khi) & (E > 0)
    kk, y = k[sel].astype(float), np.log(E[sel])
    xg = kk if MUT != "swapfit" else np.log(kk)
    xa = np.log(kk) if MUT != "swapfit" else kk
    cg = np.polyfit(xg, y, 1); ca = np.polyfit(xa, y, 1)
    rg = float(np.sqrt(np.mean((np.polyval(cg, xg) - y) ** 2)))
    ra = float(np.sqrt(np.mean((np.polyval(ca, xa) - y) ** 2)))
    edges = np.linspace(klo, khi, 5)
    loc_b, loc_p = [], []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        s = (kk >= a_) & (kk <= b_)
        if s.sum() >= 3:
            loc_b.append(float(-np.polyfit(kk[s], y[s], 1)[0]))
            loc_p.append(float(-np.polyfit(np.log(kk[s]), y[s], 1)[0]))
    # constancy of the local rates: relative spread of beta against that of p
    sb = float(np.std(loc_b) / abs(np.mean(loc_b))) if loc_b else np.nan
    sp = float(np.std(loc_p) / abs(np.mean(loc_p))) if loc_p else np.nan
    verdict = "geometric" if rg < ra else "algebraic"
    # mixed: ln A = c + a ln k - beta k (a power times a geometric factor; a > 0 is a rising prefactor)
    X = np.stack([np.ones_like(kk), np.log(kk), -kk], axis=1)
    if MUT == "swapfit":
        X = np.stack([np.ones_like(kk), -kk, np.log(kk)], axis=1)
    cm, *_ = np.linalg.lstsq(X, y, rcond=None)
    rm = float(np.sqrt(np.mean((X @ cm - y) ** 2)))
    return dict(beta=float(-cg[0]), p=float(-ca[0]), rms_geo=rg, rms_alg=ra, local_beta=loc_b, local_p=loc_p,
                spread_beta=sb, spread_p=sp, verdict=verdict, klo=int(klo), khi=int(khi),
                mixed=dict(a=float(cm[1]), beta=float(cm[2]), rms=rm))


def trig_eval(y, theta):
    """the trigonometric interpolant of samples y (uniform on [0, 2 pi)) at points theta"""
    N = len(y)
    F = np.fft.rfft(y) / N
    k = np.arange(len(F))
    w = np.full(len(F), 2.0); w[0] = 1.0
    if N % 2 == 0:
        w[-1] = 1.0
    return (np.exp(1j * np.outer(theta, k)) @ (w * F)).real


def spectrum_amp(y):
    """|c_k| with the one-sided normalisation (a cosine of amplitude a gives a at its k)"""
    A = np.abs(np.fft.rfft(y)) / len(y) * 2
    A[0] /= 2
    return A


def cut_k(A, eps):
    """the smallest K with A_k < eps for every k > K (len(A) - 1 when the last value is still above)"""
    above = np.nonzero(A >= eps)[0]
    return int(above[-1]) if len(above) else 0


def local_band(y, eps, nwin=64, sig=None):
    """k_loc(theta_c) on nwin window centres; window exp(-(theta - theta_c)^2 / (2 sig^2)), periodised"""
    N = len(y)
    th = 2 * np.pi * np.arange(N) / N
    sig = sig if sig is not None else 2 * np.pi / nwin
    out = []
    for c in 2 * np.pi * np.arange(nwin) / nwin:
        d = (th - c + np.pi) % (2 * np.pi) - np.pi
        win = np.exp(-d * d / (2 * sig * sig)) if MUT != "nowindow" else np.ones(N)
        # windowed amplitude normalised to the window's mean, so that a stationary signal gives its own amplitudes
        A = spectrum_amp(y * win) / np.mean(win)
        out.append(cut_k(A, eps))
    return np.array(out)


def arctan_map(s, s0, L):
    u = s - s0
    return s0 + u + np.arctan2((L - 1) * np.sin(u) * np.cos(u), np.cos(u) ** 2 + L * np.sin(u) ** 2)


def mapped_cut(y, eps, s0, L, M=4096):
    """K needed on the arctan grid: f(g(s)) sampled on M points uniform in s, its s-spectrum's cut at eps; y is the
    field's uniform samples (evaluated by its trigonometric interpolant) or a callable of theta"""
    s = 2 * np.pi * np.arange(M) / M
    th = arctan_map(s, s0, L) if MUT != "nomap" else s
    v = y(th) if callable(y) else trig_eval(y, th)
    return cut_k(spectrum_amp(v), eps)


def best_map(y, eps, s0s, Ls, M=4096):
    y0 = y(2 * np.pi * np.arange(M) / M) if callable(y) else y
    best = (cut_k(spectrum_amp(y0), eps), None, 1.0)
    for s0 in s0s:
        for L in Ls:
            K = mapped_cut(y, eps, s0, L, M)
            if K < best[0]:
                best = (K, float(s0), float(L))
    return best


# ---------------- planted tests ----------------
def planted():
    ok = True
    rng = np.random.default_rng(1103)
    k = np.arange(1, 300)
    # (a) geometric, bumpy, noise floor 1e-12
    Ag = np.exp(-0.05 * k) * (1 + 0.6 * np.cos(0.7 * k)) ** 2 + 1e-12 * rng.random(len(k))
    r = fit_decay(k, Ag, 30, 250)
    pa = r["verdict"] == "geometric" and abs(r["beta"] - 0.05) < 0.005
    print(f"planted fit, geometric beta 0.05: verdict {r['verdict']} beta {r['beta']:.4f} rms {r['rms_geo']:.3f}/"
          f"{r['rms_alg']:.3f} -> {'pass' if pa else 'FAIL'}")
    # (b) algebraic k^-3.5
    Aa = k ** -3.5 * (1 + 0.6 * np.cos(0.7 * k)) ** 2 + 1e-14 * rng.random(len(k))
    r = fit_decay(k, Aa, 30, 250)
    pb = r["verdict"] == "algebraic" and abs(r["p"] - 3.5) < 0.35
    print(f"planted fit, algebraic p 3.5: verdict {r['verdict']} p {r['p']:.3f} rms {r['rms_geo']:.3f}/"
          f"{r['rms_alg']:.3f} -> {'pass' if pb else 'FAIL'}")
    # (b2) mixed k^6 exp(-0.1 k): the mixed fit must give a within 1 and beta within 0.01
    Am = k ** 6.0 * np.exp(-0.1 * k) * (1 + 0.6 * np.cos(0.7 * k)) ** 2 + 1e-12 * rng.random(len(k))
    r = fit_decay(k, Am, 20, 250)
    pm = abs(r["mixed"]["a"] - 6) < 1 and abs(r["mixed"]["beta"] - 0.1) < 0.01
    print(f"planted fit, mixed k^6 exp(-0.1 k): a {r['mixed']['a']:.2f} beta {r['mixed']['beta']:.4f} rms "
          f"{r['mixed']['rms']:.3f} -> {'pass' if pm else 'FAIL'}")
    ok &= pa and pb and pm
    # (c) local bandwidth: a field with a narrow pulse of width 0.02 at theta = 1 and a smooth rest; k_loc must be
    # high only near theta = 1
    N = 2048
    th = 2 * np.pi * np.arange(N) / N
    y = np.cos(th) + 0.3 / np.cosh(((th - 1.0 + np.pi) % (2 * np.pi) - np.pi) / 0.02)
    kl = local_band(y, 1e-8, nwin=64)
    i_pk = int(np.argmin(np.abs(2 * np.pi * np.arange(64) / 64 - 1.0)))
    far = np.r_[kl[:max(0, i_pk - 8)], kl[i_pk + 9:]]
    pc = kl[i_pk] > 4 * np.median(far)
    print(f"planted local band: k_loc at the pulse {kl[i_pk]}, median elsewhere {np.median(far):.0f} -> "
          f"{'pass' if pc else 'FAIL'}")
    ok &= pc
    # (d) arctan map: a Poisson kernel with poles at theta - t0 = +- i d (and the parity images at t0 + pi) plus smooth
    # harmonics. Uniform K ~ ln(1/eps)/d. The map is a Moebius map of exp(2 i theta): it moves the kernel's poles to
    # artanh(tanh(d)/L) and puts the smooth harmonics' at artanh(L), so L = sqrt(d) gives K ~ ln(1/eps)/sqrt(d), a
    # reduction of about 7 at d = 0.02. The test asks for more than 3, with the best L within a factor 2 of sqrt(d).
    # (The kernel alone would be a poor test: composed with the matched Moebius map it becomes a trig polynomial.)
    d, t0 = 0.02, 0.7
    y = lambda t: (np.sinh(2 * d) / (np.cosh(2 * d) - np.cos(2 * (t - t0))) + 0.5 * np.cos(2 * t)
                   + 0.3 * np.cos(6 * t + 1.0))
    Ku = cut_k(spectrum_amp(y(2 * np.pi * np.arange(8192) / 8192)), 1e-10)
    Km = best_map(y, 1e-10, [t0], np.geomspace(0.02, 1.0, 40), M=8192)
    pd = Km[0] * 3 < Ku and 0.5 < Km[2] / np.sqrt(d) < 2
    print(f"planted map: uniform K {Ku}, mapped K {Km[0]} at L {Km[2]:.3f} (s0 {Km[1]}) -> {'pass' if pd else 'FAIL'}")
    ok &= pd
    return ok


# ---------------- the states ----------------
def load(path):
    m = re.search(r"kc(\d+)[^_]*_D([0-9.]+)\.npy$", os.path.basename(path))
    kc, D = int(m.group(1)), float(m.group(2))
    z = np.load(path)
    S = DSSEll(kc=kc, nodes=NODES_STD, n=D - 2, ell=1, Delta_ref=z[-1])
    if len(z) != S.layout()[2]:
        raise SystemExit(f"{path}: length {len(z)} is not k_c {kc}'s {S.layout()[2]}")
    return S, z, kc, D


def dips(c0, Delta, thr=None):
    """local minima of c0 on its fine interpolant over one period: value, tau, curvature width
    w = sqrt(2 c_min / c''), the half-depth width between c_min and the period's median"""
    M = 16 * len(c0)
    th = 2 * np.pi * np.arange(M) / M
    c = trig_eval(c0, th)
    tau = th / (2 * np.pi) * Delta
    h = tau[1] - tau[0]
    med = float(np.median(c))
    out = []
    for i in range(M):
        a, b, d = c[i - 1], c[i], c[(i + 1) % M]
        if b < a and b <= d and (thr is None or b < thr):
            curv = (a - 2 * b + d) / h ** 2
            half = 0.5 * (b + med)
            j = i
            while c[j % M] < half and j - i < M // 2:
                j += 1
            l_ = i
            while c[l_ % M] < half and i - l_ < M // 2:
                l_ -= 1
            out.append(dict(c_min=float(b), tau=float(tau[i]), s=float(th[i]), w_curv=float(np.sqrt(2 * b / curv)),
                            w_half=float((j - l_) * h)))
    return out, med


def analyse(path, eps_rel=(1e-4, 1e-6)):
    S, z, kc, D = load(path)
    f = S.fields(z)
    Delta = f["Delta"]
    om = 2 * np.pi / Delta
    rec = dict(file=os.path.basename(path), D=D, kc=kc, Delta=Delta, constraint=S.constraint(z))
    dl, med = dips(f["c0"], Delta)
    deep = sorted(dl, key=lambda d: d["c_min"])
    rec["c0"] = dict(min=float(f["c0"].min()), max=float(f["c0"].max()), median=med, dips=deep[:6])
    # one fine, exact resampling of each trigonometric polynomial (the state is band-limited at kc)
    fields = {"q": f["d"], "c0": f["c0"], "1/c0": 1.0 / f["c0"]}
    for i in (0, 1, 2, 4, 7, 11):
        x = NODES_STD[i]
        p, m_, a, c, q = f["states"][i]
        fields[f"p@{x}"] = p; fields[f"m@{x}"] = m_; fields[f"c@{x}"] = c; fields[f"psi@{x}"] = q
    fields["m(1)"] = f["mS"]; fields["a(1)"] = f["aS"]; fields["psi(1)"] = f["qS"]
    k = np.arange(S.N // 2 + 1)
    spec = {}
    for name, y in fields.items():
        A = spectrum_amp(y)
        par = 0 if name in ("c0", "1/c0", "a(1)") or name.startswith("c@") else 1
        kk = k[(k % 2 == par) & (k <= kc)]
        Ak = A[kk]
        amax = float(Ak.max())
        ent = dict(max=amax, A=[float(v) for v in Ak], k=[int(v) for v in kk])
        lo, hi = int(0.25 * kc), int(0.85 * kc)
        ent["fit"] = fit_decay(kk, Ak, lo, hi)
        ent["fit_lo"] = fit_decay(kk, Ak, 10, int(0.5 * kc))
        ent["tail"] = float(envelope(Ak)[-3:].max() / amax)
        spec[name] = ent
    rec["spec"] = spec
    return rec, S, z, f


def map_estimates(rec, f, kc, names=("q", "c0", "p@0.02", "m@0.02", "p@0.05", "p@0.1"), M=4096):
    """local bandwidth and the arctan-map cut, at eps set by each field's uniform envelope at 0.4, 0.6 and 0.8 kc
    (above the spectrum's own tail); one evaluation matrix per (s0, L), shared by the fields"""
    ys = {}
    for nm in names:
        y = {"q": f["d"], "c0": f["c0"]}.get(nm)
        if y is None:
            x = float(nm.split("@")[1]); i = NODES_STD.index(x)
            y = f["states"][i][{"p": 0, "m": 1}[nm[0]]]
        ys[nm] = y
    N = len(f["c0"])
    Y = np.stack([np.fft.rfft(ys[nm]) / N for nm in names])                  # (fields, N/2 + 1)
    kq = np.arange(Y.shape[1])
    wq = np.full(len(kq), 2.0); wq[0] = 1.0; wq[-1] = 1.0
    Y = Y * wq
    eps = {}
    for nm in names:
        ent = rec["spec"][nm]
        E = envelope(np.array(ent["A"])); kk = np.array(ent["k"])
        eps[nm] = [float(E[np.searchsorted(kk, int(fr * kc))]) for fr in (0.4, 0.6, 0.8)]
    Ku = {nm: [cut_k(spectrum_amp(ys[nm]), e) for e in eps[nm]] for nm in names}
    best = {nm: [(Ku[nm][j], None, 1.0) for j in range(3)] for nm in names}
    s = 2 * np.pi * np.arange(M) / M
    s0s = sorted(set([round(d["s"] % np.pi, 4) for d in rec["c0"]["dips"][:3]] +
                     [round(v, 4) for v in np.linspace(0, np.pi, 16, endpoint=False)]))
    for s0 in s0s:
        for L in np.geomspace(0.15, 1.0, 24)[:-1]:
            th = arctan_map(s, s0, L) if MUT != "nomap" else s
            V = (np.exp(1j * np.outer(th, kq)) @ Y.T).real                   # (M, fields)
            for c, nm in enumerate(names):
                A = spectrum_amp(V[:, c])
                for j, e in enumerate(eps[nm]):
                    K = cut_k(A, e)
                    if K < best[nm][j][0]:
                        best[nm][j] = (K, float(s0), float(L))
    out = {}
    for nm in names:
        for j, fr in enumerate((0.4, 0.6, 0.8)):
            kl = local_band(ys[nm], eps[nm][j])
            K, s0, L = best[nm][j]
            out.setdefault(nm, []).append(dict(frac=fr, eps=eps[nm][j], eps_rel=eps[nm][j] / rec["spec"][nm]["max"],
                                               K_uniform=Ku[nm][j], k_loc_max=int(kl.max()),
                                               k_loc_mean=float(kl.mean()),
                                               ideal_ratio=float(kl.max() / max(1.0, kl.mean())), K_arctan=int(K),
                                               s0=s0, L=L, arctan_ratio=float(Ku[nm][j] / max(1, K))))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ok = planted()
    print("planted tests:", "PASS" if ok else "FAIL", flush=True)
    if "--planted" in sys.argv:
        sys.exit(0 if ok else 1)
    if not ok:
        sys.exit(1)
    sdir, out = args[0], args[1]
    files = sorted(fn for fn in os.listdir(sdir) if fn.startswith("sol_l1") and fn.endswith(".npy"))
    res = []
    for fn in files:
        rec, S, z, f = analyse(os.path.join(sdir, fn))
        sp = rec["spec"]
        d0 = rec["c0"]["dips"][0]
        print(f"\n== {fn}: D {rec['D']} kc {rec['kc']} Delta {rec['Delta']:.10f} constraint {rec['constraint']:.2e}")
        print(f"   c0 min {rec['c0']['min']:.4f} max {rec['c0']['max']:.4f}; deepest dip at tau {d0['tau']:.4f} "
              f"(s {d0['s']:.3f}), w_curv {d0['w_curv']:.4f}, w_half {d0['w_half']:.4f}; dips below the median: "
              + ", ".join(f"{d['c_min']:.3f}@{d['tau']:.3f}" for d in rec["c0"]["dips"]))
        for nm, ent in sp.items():
            r = ent["fit"]
            print(f"   {nm:8s} max {ent['max']:.2e} tail/max {ent['tail']:.1e}  [{r['klo']},{r['khi']}] beta {r['beta']:.4f} "
                  f"(rms {r['rms_geo']:.2f}, local {' '.join(f'{v:.3f}' for v in r['local_beta'])}) p {r['p']:.2f} "
                  f"(rms {r['rms_alg']:.2f}, local {' '.join(f'{v:.1f}' for v in r['local_p'])}) -> {r['verdict']}")
        for nm in ("q", "c0", "p@0.02", "p@0.05", "p@0.1"):
            lo20 = min(20, int(0.25 * rec["kc"]))
            mx = fit_decay(np.array(sp[nm]["k"]), np.array(sp[nm]["A"]), lo20, int(0.85 * rec["kc"]))["mixed"]
            sp[nm]["fit_mixed20"] = mx
            print(f"   mixed fit [{lo20}, {int(0.85 * rec['kc'])}] {nm:7s} a {mx['a']:6.2f} beta {mx['beta']:.4f} rms {mx['rms']:.2f}")
        if (rec["kc"] >= 87 or "--maps" in sys.argv) and "--nomaps" not in sys.argv:
            rec["maps"] = map_estimates(rec, f, rec["kc"])
            for nm, rows in rec["maps"].items():
                print(f"   map {nm:8s} " + "; ".join(
                    f"eps {r['eps_rel']:.0e}: K {r['K_uniform']} -> arctan {r['K_arctan']} (L {r['L']:.2f}, s0 "
                    f"{r['s0'] if r['s0'] is None else round(r['s0'], 3)}), k_loc max/mean {r['k_loc_max']}/{r['k_loc_mean']:.0f}"
                    for r in rows))
        res.append(rec)
        sys.stdout.flush()
    with open(out, "w") as fh:
        json.dump(res, fh)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
