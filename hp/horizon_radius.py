#!/usr/bin/env python3
"""Issue #693: the horizon series' radius of convergence at #603's chain states, against the matching distance
u0 = 1 - x_last = 0.3 where the chain sums the series (notes/HORIZON_RADIUS.md).

The series. The horizon block (model.rs horizon_series) continues p, m, a, c from the horizon data (m_h, a_h, Delta)
as a Taylor series in u = 1 - x and sums it at u0 to the order ks. Solver.series returns its coefficients in
double-double only. The quad- and octo-double coefficients come from the horizon block's residual instead: on a
two-node layout (the centre and the last node) the block's rows are S_ks(u0) - node, the series summed to order ks
minus the node, so S_k - S_(k-1) = c_k u0^k, one term per field and tau point, in the residual's own precision
(residual_q, residual_o; each difference formed with qd_axpy or od_axpy, its parts summed in f64). The kc and kt of
the two-node solver are 4 (its other rows are not used); control C1 checks that this changes no horizon row.
b[k, field, lane] = c_k u0^k is the term at u0 (the scaled coefficient times (u0 / sigma)^k for the double-double
series); fields p, m (odd in tau) and a, c (even), lanes the M points of the half grid.

Estimators, each on the window of orders [K - W, K] (W = 40, K in WINDOWS):
  E1 ratio: 13.6's late-order ratio test (pole_test.ratio_radius), R = u0 (t_(K-W) / t_K)^(1/W), t_k = max over lanes
     |b_k| (or |B_k^(j)| for harmonic j);
  E2 root: the least-squares slope s of ln t_k over the window, R = u0 exp(-s) (Cauchy-Hadamard as a fit);
  E3 Domb-Sykes, per lane: rho_k = b_k / b_(k-1) fitted as A + B / k over the window, R = u0 / A and exponent
     1 + B / A; read only in lanes whose ratios have one sign (a real singularity on the positive axis; mixed signs
     mean a complex pair and are not read). Reported: the median over the readable lanes, their count, and the
     median exponent.
Harmonics: B_k^(j), the DFT over the full grid of 2M points with the field's parity (odd fields antiperiodic), j <= M.
The spread over E1-E3 in one window is an estimator range, not an error bar; the spread over windows shows how far
the late orders are from asymptotic.

13.6's pole law (QUAD_DOUBLE 13.6, #546): a_h scaled by 1 + delta (od_axpy, so delta below 1e-16 is kept), the
series recomputed in octo-double, R(delta) by E1-E3 in the last window, R = R0 + b ln(1/delta), C = b (n - 1), and
delta_max(u0) = exp(-(u0 - R0) / b), the largest data error for which the series still converges at u0. Also recorded:
H(delta) = max over the m and a rows of |S_ks(u0) - node| at the chain's ks, the re-seed rung's residual for those data.

Planted truth (`selftest`, criteria written before any real series was read by these estimators; u0 = 0.3, M = 32,
orders to 160, relative noise 2^-106 on every term):
  P1 poles at u = 0.25 and 0.4 (real, lane-dependent residues) and a background of radius 0.8: E1, E2, E3 within 5%
     of 0.25 in the windows ending at 130 and 160 (the issue's acceptance);
  P2 the pole at 0.4 and the background only: E1, E2, E3 within 5% of 0.4 in the same windows;
  P3 a complex pair at |u| = 0.4, arguments +-60 degrees: E2 within 5% of 0.4, and E3 reads no lane (no false real pole);
  P4 the mass mode mu (1 - u)^-(n-1), n - 1 = 250, mu = 1e-54, over a decaying part of radius 0.5: E3 within 5% of 1
     in the window ending at 130 (the hump of binom(k + n - 2, k) u0^k is not a singularity near u0);
  P5 a pole at 0.15 of residue 1e-45 under a decaying part of radius 0.5: E1, E2, E3 within 5% of 0.15 in the window
     ending at 160.
`--mutate drop_term` drops P1's term with the pole at 0.25 and keeps its expectation: P1 must then fail (exit 1).

Controls on real data (`control STATE.npz`): C1, the two-node solver's horizon rows at kc = kt = 4 equal those at the
state's kc, kt bit for bit (orders 40, 130); C2a, the octo-double terms equal the double-double series' (its two
parts summed in f64) at orders 1-5 to 1e-15 relative to the largest term of each order; C2b, the quad- and
octo-double terms agree at orders 1-20 to 1e-24. `--mutate shift` pairs order k with k + 2 and must fail C2a and C2b.
Exit 1 when a control fails. C2 as first written (both against double-double at orders 1-30, to 1e-24) fired at D 226
and 253.7: the double-double series leaves the octo-double one from order 7 on and is wrong by O(1) at order 16-17,
by the same amount as its two horizon_alpha modes differ (its rounding, which the mass mode amplifies). C2a and C2b
replace it (notes/HORIZON_RADIUS.md).

    cd hp && python3 horizon_radius.py selftest [--mutate drop_term]
    cd hp && python3 horizon_radius.py control STATE.npz [--mutate shift] [--threads 2]
    cd hp && python3 horizon_radius.py run OUT.jsonl STATE.npz [...] [--kmax 160] [--precs dd,qd,od] [--deltas 0]
        [--threads 2] [--coef DIR]     (OUT in a task folder of the raw-data repository; --allow_unraw for scratch)
    cd hp && python3 horizon_radius.py csv OUT.csv IN.jsonl [...]
    cd hp && python3 horizon_radius.py summary IN.jsonl [...] [--reseed anm_run.jsonl ...]
"""
import json
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

mp.mp.dps = 60               # D_str has 50 digits; mp.mpf at the default 15 would move n by 1e-14 (12.4's dD)
FIELDS = ("p", "m", "a", "c")
PARITY = ("odd", "odd", "even", "even")
W = 40
WINDOWS = (60, 100, 130, 160)
NHARM = 4                    # harmonics reported per field: the first NHARM its parity allows
ETA = 2.0 ** -106
U0 = 0.3
TOL = 0.05


# ------------------------------------------------------------------ estimators
def harmonics(b, q):
    """B[k, j]: the DFT over the full grid (2M points) of field q's terms, with the field's parity; j = 0..M"""
    nk, _, M = b.shape
    f = b[:, q, :]
    full = np.concatenate([f, -f if PARITY[q] == "odd" else f], axis=1)
    return np.fft.rfft(full, axis=1) / (2 * M)


def allowed(q, M):
    return [j for j in range(M + 1) if (j % 2 == 1) == (PARITY[q] == "odd")]


def e1_ratio(t, K, u0):
    if K >= len(t) or not (t[K] > 0 and t[K - W] > 0):
        return None
    return float(u0 / (t[K] / t[K - W]) ** (1.0 / W))


def e2_root(t, K, u0):
    if K >= len(t):
        return None
    ks = np.arange(K - W, K + 1)
    y = t[K - W:K + 1]
    ok = y > 0
    if ok.sum() < W // 2:
        return None
    s = np.polyfit(ks[ok], np.log(y[ok]), 1)[0]
    return float(u0 * math.exp(-s))


def e3_ds(bl, K, u0):
    """Domb-Sykes on one lane's signed terms over the window: (R, exponent) or None when the ratios change sign"""
    if K >= len(bl):
        return None
    num, den = bl[K - W + 1:K + 1], bl[K - W:K]
    if np.any(den == 0) or np.any(num == 0):
        return None
    rho = num / den                                  # b_k / b_(k-1) = (c_k / c_(k-1)) u0
    if not (np.all(rho > 0) or np.all(rho < 0)):
        return None
    if np.all(rho < 0):
        return None                                  # a singularity on the negative axis: not the matching side
    ks = np.arange(K - W + 1, K + 1)
    B, A = np.polyfit(1.0 / ks, rho, 1)
    if A <= 0:
        return None
    return float(u0 / A), float(1.0 + B / A)


def estimate(b, u0, windows=WINDOWS):
    """per field: lane-max E1/E2, per-lane E3 (median, count, exponent), per harmonic E1/E2; per window K"""
    nk, _, M = b.shape
    out = {}
    for q, f in enumerate(FIELDS):
        t = np.abs(b[:, q, :]).max(axis=1)
        H = harmonics(b, q)
        js = allowed(q, M)[:NHARM]
        rec = {}
        for K in windows:
            if K >= nk:
                continue
            ds = [e3_ds(b[:, q, l], K, u0) for l in range(M)]
            ds = [d for d in ds if d is not None]
            r = dict(E1=e1_ratio(t, K, u0), E2=e2_root(t, K, u0),
                     E3=float(np.median([d[0] for d in ds])) if ds else None,
                     E3_n=len(ds), E3_alpha=float(np.median([d[1] for d in ds])) if ds else None,
                     E3_lo=float(min(d[0] for d in ds)) if ds else None, E3_hi=float(max(d[0] for d in ds)) if ds else None,
                     tK=float(t[K]), harm={})
            for j in js:
                th = np.abs(H[:, j])
                r["harm"][str(j)] = dict(E1=e1_ratio(th, K, u0), E2=e2_root(th, K, u0), tK=float(th[K]))
            vals = [v for v in (r["E1"], r["E2"], r["E3"]) if v is not None]
            r["lo"], r["hi"] = (min(vals), max(vals)) if vals else (None, None)
            rec[str(K)] = r
        out[f] = rec
    return out


def profile(b):
    """t_k = max over lanes and fields of |b_k| at a few orders (for the log)"""
    t = np.abs(b).max(axis=(1, 2))
    return {str(k): float(t[k]) for k in (1, 10, 20, 40, 60, 80, 100, 110, 120, 130, 140, 150, 160) if k < len(t)}


# ------------------------------------------------------------------ planted truth
def planted_case(name, nk, M, rng, mutate=""):
    """b[k, q, l] of a planted series (in u, terms at u0) and its expectation"""
    u0 = U0
    k = np.arange(nk)
    th = np.pi * np.arange(M) / M
    b = np.zeros((nk, 4, M))
    for q in range(4):
        for l in range(M):
            w = 1.0 + 0.4 * math.cos(th[l] + 0.3 * q)          # lane- and field-dependent residues, all positive
            bg = (u0 / 0.8) ** k * w                            # the background: a pole at 0.8
            if name == "P1":
                s = bg + 0.1 * w * (u0 / 0.4) ** k
                if mutate != "drop_term":
                    s = s + 1e-3 * w * (u0 / 0.25) ** k
            elif name == "P2":
                s = bg + 0.1 * w * (u0 / 0.4) ** k
            elif name == "P3":
                z = 0.4 * np.exp(1j * np.pi / 3)
                ph = np.exp(1j * (th[l] + q))
                s = bg + 0.1 * w * np.real(2 * ph * (u0 / z) ** k)
            elif name == "P4":
                n1 = 250
                lb = np.array([math.lgamma(kk + n1) - math.lgamma(kk + 1) - math.lgamma(n1) for kk in k])
                s = (u0 / 0.5) ** k * w + 1e-54 * w * np.exp(lb + k * math.log(u0))
            elif name == "P5":
                s = (u0 / 0.5) ** k * w + 1e-45 * w * (u0 / 0.15) ** k
            b[:, q, l] = s * (1.0 + ETA * rng.standard_normal(nk))
    exp = dict(P1=0.25, P2=0.4, P3=0.4, P4=1.0, P5=0.15)[name]
    return b, exp


def selftest(mutate):
    nk, M = 161, 32
    rng = np.random.default_rng(693)
    fails = []
    print(f"{'case':>4} {'K':>4} {'field':>5} {'E1':>7} {'E2':>7} {'E3':>7} {'n3':>3} {'alpha3':>7} {'expect':>7}  verdict")
    for name, Ks, ests in (("P1", (130, 160), ("E1", "E2", "E3")), ("P2", (130, 160), ("E1", "E2", "E3")),
                           ("P3", (130, 160), ("E2",)), ("P4", (130,), ("E3",)), ("P5", (160,), ("E1", "E2", "E3"))):
        b, exp = planted_case(name, nk, M, rng, mutate)
        est = estimate(b, U0, windows=Ks)
        for K in Ks:
            for f in FIELDS:
                r = est[f][str(K)]
                bad = [e for e in ests if r[e] is None or abs(r[e] - exp) > TOL * exp]
                if name == "P3" and r["E3_n"] > 0:
                    bad.append("E3 read a real pole")
                g = lambda x: f"{x:7.4f}" if x is not None else "      -"  # noqa: E731
                print(f"{name:>4} {K:4d} {f:>5} {g(r['E1'])} {g(r['E2'])} {g(r['E3'])} {r['E3_n']:3d} {g(r['E3_alpha'])} "
                      f"{exp:7.3f}  {'FAIL ' + ','.join(bad) if bad else 'OK'}", flush=True)
                if bad:
                    fails.append((name, K, f, bad))
            # the harmonics: E2 of the first harmonic within 5% where the case asks for E2
            for f in FIELDS:
                for j, hr in est[f][str(K)]["harm"].items():
                    if "E2" in ests and (hr["E2"] is None or abs(hr["E2"] - exp) > TOL * exp) and j == min(est[f][str(K)]["harm"], key=int):
                        fails.append((name, K, f, f"harmonic {j} E2 {hr['E2']}"))
    if fails:
        print(f"selftest{' (--mutate ' + mutate + ')' if mutate else ''}: FAILED {len(fails)} checks: {fails[:6]}")
        return 1
    print("selftest: all planted checks pass")
    return 0


# ------------------------------------------------------------------ the kernel's series
def load_state(path):
    from octo_corrector import load_o
    z, meta = load_o(path)
    return [np.asarray(p, dtype=float) for p in z], meta


class Series:
    """the horizon series of one state in dd (Solver.series), qd and od (two-node residual differences)"""

    def __init__(self, z, meta, threads=2, kc2=4, kt2=4):
        from hpkernel.driver import make_solver
        self.mk = make_solver
        self.z, self.meta = z, meta
        self.D = mp.mpf(meta["D_str"])
        self.M, self.nodes = int(meta["M"]), list(meta["nodes"])
        self.K = len(self.nodes)
        self.u0 = 1.0 - self.nodes[-1]
        self.threads, self.kc2, self.kt2 = threads, kc2, kt2
        M, K = self.M, self.K
        self.off = dict(node=lambda i: 2 * M + 4 * M * i, mh=2 * M + 4 * M * K, ah=3 * M + 4 * M * K)

    def two_node(self, z):
        """the state on the layout (x_0, x_last): centre data, node 0, the last node, m_h, a_h, Delta"""
        M, o = self.M, self.off
        L2 = 4 * M * 3 + 1
        out = []
        for p in z:
            v = np.zeros(L2)
            v[:2 * M] = p[:2 * M]
            v[2 * M:6 * M] = p[o["node"](0):o["node"](0) + 4 * M]
            v[6 * M:10 * M] = p[o["node"](self.K - 1):o["node"](self.K - 1) + 4 * M]
            v[10 * M:12 * M] = p[o["mh"]:o["mh"] + 2 * M]
            v[-1] = p[-1]
            out.append(np.ascontiguousarray(v))
        return out

    def perturbed(self, delta):
        """the state with a_h scaled by 1 + delta in octo-double (13.6's perturbation)"""
        z = [p.copy() for p in self.z]
        if delta:
            from multidouble import od_axpy
            M, a = self.M, self.off["ah"]
            parts = [np.ascontiguousarray(p[a:a + M]) for p in z]
            new = od_axpy(float(delta), *parts, *parts)
            for q in range(8):
                z[q][a:a + M] = np.asarray(new[q])
        return z

    def solver2(self, ks, kc=None, kt=None):
        s = self.mk(self.D, self.M, nodes=[self.nodes[0], self.nodes[-1]], kt=kt or self.kt2, kc=kc or self.kc2, ks=ks,
                    threads=self.threads, kfreeze=None)
        s.set_segs([[self.nodes[0], self.nodes[-1]]])
        return s

    def rows(self, s, z2, prec):
        """the horizon rows (4M) of the two-node residual in prec's parts"""
        r0 = 8 * self.M
        if prec == "od":
            F = s.residual_o(z2[:8])
            return [np.ascontiguousarray(np.asarray(F[q])[r0:r0 + 4 * self.M]) for q in range(8)]
        F = s.residual_q(*z2[:4])
        return [np.ascontiguousarray(np.asarray(F[q])[r0:r0 + 4 * self.M]) for q in range(4)]

    def terms(self, prec, kmax, delta=0.0, shift=1):
        """b[k, q, l] = c_k u0^k for k = 0..kmax in prec ('qd' or 'od') by S_k - S_(k-shift); and H at ks = meta ks.
        shift 2 is control C2's mutation."""
        from multidouble import od_axpy, qd_axpy
        z2 = self.two_node(self.perturbed(delta))
        node = [p[6 * self.M:10 * self.M] for p in z2]
        npart = 8 if prec == "od" else 4
        S = []
        for ks in range(kmax + 1):
            S.append(self.rows(self.solver2(ks), z2, prec))
        b = np.zeros((kmax + 1, 4 * self.M))
        ax = od_axpy if prec == "od" else qd_axpy
        for k in range(kmax + 1):
            if k == 0:
                d = ax(1.0, *[np.ascontiguousarray(node[q]) for q in range(npart)], *S[0])      # S_0 = c_0
            elif k - shift < 0:
                continue
            else:
                d = ax(-1.0, *S[k - shift], *S[k])
            b[k] = np.sum(np.asarray(d), axis=0)
        ks_meta = int(self.meta.get("ks", 130))
        Hr = np.abs(np.sum(np.asarray(S[min(ks_meta, kmax)]), axis=0)).reshape(4, self.M).max(axis=1)
        return b.reshape(kmax + 1, 4, self.M), [float(x) for x in Hr]

    def terms_dd(self, kmax, alpha=0, delta=0.0):
        """the double-double series (Solver.series, scaled) as terms at u0, horizon_alpha = alpha; and H at ks"""
        z = self.perturbed(delta)
        s = self.mk(self.D, self.M, nodes=self.nodes, kt=self.meta["kt"], kc=self.meta["kc"], ks=kmax,
                    threads=self.threads, kfreeze=None, horizon_alpha=alpha)
        sig = s.series_scale(1)
        hh, hl = s.series(z[0], z[1], 1, True)
        c = (np.asarray(hh) + np.asarray(hl)).reshape(-1, 4, self.M)
        b = c * ((self.u0 / sig) ** np.arange(c.shape[0]))[:, None, None]
        ks_meta = int(self.meta.get("ks", 130))
        s2 = self.mk(self.D, self.M, nodes=[self.nodes[0], self.nodes[-1]], kt=self.kt2, kc=self.kc2, ks=ks_meta,
                     threads=self.threads, kfreeze=None, horizon_alpha=alpha)
        s2.set_segs([[self.nodes[0], self.nodes[-1]]])
        z2 = self.two_node(z)
        Fh, Fl = s2.residual(z2[0], z2[1])
        F = (np.asarray(Fh) + np.asarray(Fl))[8 * self.M:12 * self.M]
        return b, [float(x) for x in np.abs(F).reshape(4, self.M).max(axis=1)]


# ------------------------------------------------------------------ controls on real data
def control(path, mutate, threads):
    z, meta = load_state(path)
    sr = Series(z, meta, threads=threads)
    fails = []
    z2 = sr.two_node(z)
    for ks in (40, int(meta["ks"])):
        a = sr.rows(sr.solver2(ks), z2, "od")
        b = sr.rows(sr.solver2(ks, kc=int(meta["kc"]), kt=int(meta["kt"])), z2, "od")
        same = all(np.array_equal(x, y) for x, y in zip(a, b))
        print(f"C1 order {ks}: horizon rows at kc = kt = 4 and at kc {meta['kc']}, kt {meta['kt']}: "
              f"{'bit for bit' if same else 'DIFFER'}", flush=True)
        if not same:
            fails.append(("C1", ks))
    shift = 2 if mutate == "shift" else 1
    bd, _ = sr.terms_dd(40)
    bo, _ = sr.terms("od", 20, shift=shift)
    bq, _ = sr.terms("qd", 20)
    tag = ", --mutate shift" if shift == 2 else ""
    for name, x, y, hi, bound in (("C2a od against dd", bo, bd, 5, 1e-15), ("C2b od against qd", bo, bq, 20, 1e-24)):
        scale = np.abs(y[1:hi + 1]).max(axis=(1, 2))
        worst = float((np.abs(x[1:hi + 1] - y[1:hi + 1]).max(axis=(1, 2)) / scale).max())
        ok = worst <= bound
        print(f"{name}, orders 1-{hi}: worst relative difference {worst:.1e} (bound {bound:.0e}{tag}): "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            fails.append((name, worst))
    if fails:
        print("control FAILED:", fails)
        return 1
    print("controls: pass")
    return 0


# ------------------------------------------------------------------ the run
def run(out, paths, kmax, precs, deltas, threads, coef_dir):
    with open(out, "a") as fo:
        for path in paths:
            z, meta = load_state(path)
            sr = Series(z, meta, threads=threads)
            name = os.path.basename(path)
            D = float(meta["D"])
            n = D - 2.0
            G = sr.nodes[-1] ** -(n - 1.0)
            for delta in deltas:
                for prec in precs:
                    t0 = time.time()
                    if prec.startswith("dd"):
                        alpha = 1 if prec == "dd1" else 0
                        b, H = sr.terms_dd(kmax, alpha=alpha, delta=delta)
                    else:
                        b, H = sr.terms(prec, kmax, delta=delta)
                    est = estimate(b, sr.u0)
                    rec = dict(state=name, folder=os.path.basename(os.path.dirname(path)), D=D, D_str=meta["D_str"],
                               n=n, u0=sr.u0, ks=int(meta["ks"]), kmax=kmax, prec=prec, delta=delta, G=G,
                               Re_s=float(n / (2.0 * (1.0 + (n - 1.0) * (1.0 - float(np.mean(z[0][sr.off["ah"]:sr.off["ah"] + sr.M]) ** 2)))) - 1.0),
                               H=dict(zip(FIELDS, H)), profile=profile(b), est=est, seconds=round(time.time() - t0, 2))
                    fo.write(json.dumps(rec) + "\n")
                    fo.flush()
                    if coef_dir:
                        os.makedirs(coef_dir, exist_ok=True)
                        np.save(os.path.join(coef_dir, f"{name[:-4]}_{prec}_d{delta:.0e}.npy"), b)
                    a130 = est["a"].get("130", {})
                    p160 = est["p"].get("160", {})
                    g = lambda x: f"{x:.3f}" if x is not None else "-"  # noqa: E731
                    print(f"{D:9.4f} {prec:>3} delta {delta:.0e}: a K130 E1 {g(a130.get('E1'))} E2 {g(a130.get('E2'))} "
                          f"E3 {g(a130.get('E3'))} (n {a130.get('E3_n')}, alpha {g(a130.get('E3_alpha'))}); p K160 E1 "
                          f"{g(p160.get('E1'))} E2 {g(p160.get('E2'))}; H(u0) m {H[1]:.1e} a {H[2]:.1e}; "
                          f"t_k " + " ".join(f"{k}:{v:.0e}" for k, v in rec["profile"].items() if k in ("40", "80", "100", "130", "160"))
                          + f"; {rec['seconds']} s", flush=True)


# ------------------------------------------------------------------ the derived table
def csv_out(out, ins):
    import csv
    rows = []
    for path in ins:
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            for f in FIELDS:
                for K, e in r["est"][f].items():
                    base = dict(state=r["state"], folder=r["folder"], D=r["D"], prec=r["prec"], delta=r["delta"], u0=r["u0"],
                                field=f, K=int(K), W=W, G=r["G"], Re_s=r["Re_s"], H_u0=r["H"][f])
                    rows.append(dict(base, harmonic="max", E1=e["E1"], E2=e["E2"], E3=e["E3"], E3_n=e["E3_n"],
                                     E3_alpha=e["E3_alpha"], lo=e["lo"], hi=e["hi"], tK=e["tK"]))
                    for j, h in e["harm"].items():
                        rows.append(dict(base, harmonic=j, E1=h["E1"], E2=h["E2"], E3=None, E3_n=None, E3_alpha=None,
                                         lo=None, hi=None, tK=h["tK"]))
    keys = ["state", "folder", "D", "prec", "delta", "u0", "field", "harmonic", "K", "W", "E1", "E2", "E3", "E3_n",
            "E3_alpha", "lo", "hi", "tK", "H_u0", "G", "Re_s"]
    fmt = lambda v: "" if v is None else (f"{v:.6g}" if isinstance(v, float) else v)  # noqa: E731
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["D"], r["prec"], r["delta"], r["field"], r["harmonic"] != "max", str(r["harmonic"]), r["K"])):
            w.writerow({k: fmt(r[k]) for k in keys})
    print(f"{out}: {len(rows)} rows")


def main():
    args = list(sys.argv[1:])
    mutate = ""
    for a in list(args):
        if a.startswith("--mutate"):
            i = args.index(a)
            if "=" in a:
                mutate = a.split("=", 1)[1]
                del args[i]
            else:
                mutate = args[i + 1]
                del args[i:i + 2]

    def opt(name, default, cast=str):
        if name in args:
            i = args.index(name)
            v = cast(args[i + 1])
            del args[i:i + 2]
            return v
        return default
    threads = opt("--threads", 2, int)
    if not args:
        print(__doc__)
        return 2
    cmd = args.pop(0)
    if cmd == "selftest":
        if mutate not in ("", "drop_term"):
            raise SystemExit("selftest: only --mutate drop_term")
        return selftest(mutate)
    if cmd == "control":
        if mutate not in ("", "shift"):
            raise SystemExit("control: only --mutate shift")
        return control(args[0], mutate, threads)
    if cmd == "run":
        from raw_out import pop_allow, require_raw
        allow = pop_allow(args)
        kmax = opt("--kmax", 160, int)
        precs = opt("--precs", "dd,qd,od").split(",")
        deltas = [float(x) for x in opt("--deltas", "0").split(",")]
        coef = opt("--coef", "")
        out, paths = args[0], args[1:]
        require_raw(out, allow=allow)
        run(out, paths, kmax, precs, deltas, threads, coef)
        return 0
    if cmd == "csv":
        csv_out(args[0], args[1:])
        return 0
    if cmd == "summary":
        reseed = []
        while "--reseed" in args:
            i = args.index("--reseed")
            reseed.append(args[i + 1])
            del args[i:i + 2]
        return summary(args, reseed)
    print(__doc__)
    return 2


def summary(ins, reseed):
    """the per-state table, 13.6's law per state and its delta_max(u0), and the re-seed records beside them"""
    recs = [json.loads(line) for p in ins for line in open(p) if line.strip()]
    g = lambda x: f"{x:.3f}" if x is not None else "    -"  # noqa: E731
    print("on the solution (delta 0): per precision, the a field's E1/E2/E3 at K 100, 130, 160 and the p field's E2 at 160;"
          " H: max |S_ks(u0) - node| over the m and a rows")
    for r in sorted((r for r in recs if r["delta"] == 0), key=lambda r: (r["D"], r["prec"])):
        e = r["est"]
        line = f"{r['D']:9.4f} {r['prec']:>3} "
        for K in ("100", "130", "160"):
            a = e["a"].get(K)
            if a:
                line += f"K{K} {g(a['E1'])}/{g(a['E2'])}/{g(a['E3'])}({a['E3_n']:2d},{a['E3_alpha'] if a['E3_alpha'] is None else round(a['E3_alpha'])}) "
        p = e["p"].get("160")
        line += f"p160 E2 {g(p['E2']) if p else '-'}  H {max(r['H']['m'], r['H']['a']):.1e}"
        print(line)
    law = {}
    for r in recs:
        if r["delta"] > 0 and r["prec"] == "od":
            law.setdefault(r["state"], []).append(r)
    if law:
        print("13.6's law in octo-double: R(delta) by E2 of the max over fields at K 160, R = R0 + b ln(1/delta), C = b (n - 1)")
        for st, rs in sorted(law.items(), key=lambda kv: kv[1][0]["D"]):
            pts = []
            for r in sorted(rs, key=lambda r: -r["delta"]):
                R = min(r["est"][f]["160"]["E2"] for f in FIELDS if r["est"][f].get("160") and r["est"][f]["160"]["E2"])
                pts.append((r["delta"], R, max(r["H"]["m"], r["H"]["a"])))
            x = np.array([math.log(1 / d) for d, _, _ in pts])
            y = np.array([R for _, R, _ in pts])
            bb, a0 = np.polyfit(x, y, 1)
            n = rs[0]["n"]
            dmax = math.exp(-(U0 - a0) / bb)
            print(f"{rs[0]['D']:9.4f}: " + " ".join(f"{d:.0e}:{R:.3f}(H {h:.0e})" for d, R, h in pts)
                  + f"  R0 {a0:+.4f} b {bb:.5f} C {bb * (n - 1):.3f}, delta_max(0.3) {dmax:.1e}")
    if reseed:
        print("re-seed records (#603): D, a, ok, the double-double u0 rung's |R|, the quad-double rung's its and |R|")
        for p in reseed:
            for line in open(p):
                r = json.loads(line)
                for at in r.get("attempts", []):
                    rs = at.get("reseed")
                    if not rs:
                        continue
                    u0r = [x for x in rs["rungs"] if abs(x["u"] - U0) < 1e-9]
                    if len(u0r) >= 2:
                        print(f"  {at['D']:9.4f} a {at['a']:.4f} {'ok    ' if rs['ok'] else 'FAILED'} dd |R| {u0r[0]['R']:.2e}  "
                              f"qd its {u0r[-1]['its']:2d} |R| {u0r[-1]['R']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
