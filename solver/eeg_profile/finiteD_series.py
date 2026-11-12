#!/usr/bin/env python3
"""Per-harmonic ratios w_nu(eps) = F_{nu+2}/F_nu of the finite-D centre profiles as series in eps = 1/(D - 1), their
D = infinity limits against the tuned profile's, and a finite-D form of the profile built from them (agent-comms#38,
notes/EEG_PROFILE.md 3.4).

Data. F = sqrt(n) p1, n = D - 2, with p1 the centre data (p ~ x p1(tau) near x = 0, the first M entries of a state's
sum of parts; hp/held_fit.py coeffs_of), so F = sqrt(n) d_x p(0) as in finiteD_ratios.py. Its odd harmonics F_nu are
gauge-fixed by a shift of theta to F_1 = -i|F_1| (EEG_PROFILE.md 3.1), and r_nu = F_{nu+2}/F_nu is the paper's
w = q e^{i delta} at nu = 1. The states:
  full    the full-system states, every harmonic solved: hp/data/states with D >= 15 (double-double; from D 35 the
          scan pins p1's harmonics above 13 to zero, kfreeze 13), T84-mac's *_full states at 68.5-71.5 and T81's
          converged states at 72.8 and 73.0 (quad-double, kfreeze 13);
  chain   the chain_fit states of T138-win2 (chain, chain_o) at D 80, 90, ..., 170, 180, 190, 200, 205, 209, 215
          (I126-mac's selection). They hold p1's harmonics 5-13 on a 1/n fit of the full states (kfreeze 3,
          notes/HELD_HARMONICS.md), so only F_1 and F_3 are solved there: only r_1 takes them as data.
STATEDIR is a directory with the raw-data repository's layout (T84-mac/, T81/, T138-win2/), the clone itself or a copy.

Fit. One fit choice (x, K, dlo, chain) is the complex least-squares polynomial r_nu = sum_{k <= K} c_k x^k (Chebyshev
basis on the data's x range, every state weight 1) over the states with D >= dlo; x is eps (the issue's variable) or
1/n; chain says whether r_1's fit takes the chain states. The D = infinity value is the fit at x = 0. The grid G is
x in {eps, 1/n} x K in KS x dlo in {15, 20, 30} (x chain in {yes, no} for r_1). Its central value is the median over
G (real and imaginary parts separately) and s = max_G |r(0) - median| is a spread over fit choices, not an error bar.
The reference choice REF (eps, K_REF, dlo 15, chain yes) gives the form.

The form. With A(eps) the reference fit of A_n = 2|F_1| over every state (free intercept) and r_nu(eps) the reference
fits, F_1 = -i A/2 and F_{nu+2} = r_nu F_nu for nu = 1..7: the harmonics through nu = 9 (EEG_PROFILE.md 8.1's
G(u) = sum_k c_k u^k, c_k = r_1 ... r_{2k-1}, truncated at k = 4). Each of A and r_nu carries the step at D 35, as the
data do.

Planted families (the truth is the tuned D = infinity ratio r_inf at A = 2.71918; the shapes are sized by the real
ratios at D 20.04, 34.05 and 60.00, the realism printed as the plant's largest distance from every real ratio):
  rat  (r - r_inf)/eps = a/(1 + b eps) through D 20.04 and 60.00, one pole in eps;
  q3   (r - r_inf)/eps = a/(1 + b1 eps + b2 eps^2) through all three, two poles.
F_1 follows EEG_PROFILE.md 3.5.1's alt12 amplitude; harmonics above 13 are zero from D 35, as the scan pins them.

Criteria (agent-comms#38, posted on the issue before the first fit of real data):
  P1  planted truth, noise-free: for each nu in 1, 3, 5, 7 and each family (rat, q3), |median - truth| <= s.
  P3  planted, realistic: the measured noise of each group of states (`noise`, D 15-35, 35-61, 68-73, the chain with
      the last) as complex Gaussian noise on r_nu, and the measured steps at D 35 (kfreeze 13) and at the chain added,
      20 seeds: |median - truth| <= s in at least 18 seeds for each nu in 1, 3, 5, 7 and each family.
  Pm  control, --mutate=offset: the noise-free planted runs compared with the truth plus OFFSET (5e-4 in r, about 1%):
      P1 must fail for every nu.
  A1  real data: for each nu in 1, 3, 5, 7, |median - r_nu(tuned, A = 2.71918)| <= s. The distances to the tuned
      ratios at A = 2.72 and at the states' A_inf 2.7192053 (EEG_PROFILE.md 3.5.1) are reported beside it.
  Am  control, --mutate=target: A1 against the tuned ratio of the next harmonic, r_{nu+2}, must fail for at least
      one nu; the nu for which it fails are the ratios the extrapolation tells apart from their neighbour.
  E1  real data: r_1's grid without the chain (D <= 73), evaluated at each chain state's D (80-215), covers the
      state's r_1: |median - r_1(state)| <= s there. A check of the spread as an extrapolation measure.
  F1  the form reproduces every full state's F_nu, nu = 1..9, and every chain state's F_1, F_3, to TOL_FORM (1e-4)
      in |F_form/F_state - 1| (complex). The chain's held F_5..F_9 against the form are reported, not tested.
  F2  the form fitted on D <= 61.5 (the double-double states) predicts the quad-double states' (68.5-73) F_nu,
      nu = 1..9, to TOL_FORM.
  Fm  control, --mutate=scale1: the state at D 40.02 with r_5 x 1.01 (F_7 and F_9 scaled): F1 must fail, and the
      move of each nu's median is reported.
  reverse  control, --mutate=reverse (theta -> -theta): every median's argument turns negative and A1 fails.

Revision 1 (posted on the issue after the first planted runs, before any fit of real data):
  - The noise estimator's window is 9 states at degree 5 (was 7 at degree 3, whose own truncation, about 2e-7 at
    D < 35, set the first table). Its floor on the noise-free plants is printed by `noise`.
  - Every fit carries a step column at D 35, where the scan starts pinning p1's harmonics above 13 (measured steps
    2.5e-7 in r_5, 6.5e-7 in r_7, 1.2e-4 in r_9). The D = infinity value is taken on the pinned side. In the realistic
    plant the step decays from D 35 like |F_15| ~ |F_13|^2/|F_11| of the real states (pin_weight).
  - The grid is K 7, 8, 9 x D >= 15, 20 x {eps, 1/n} (x chain for r_1). With the step column, K <= 6 is
    truncation-limited on q3 and K >= 10 noise-limited, and D >= 25, 30 leave too few states below 35.
  - A third family, pade, is information only. It is the real data's shape, a linearized least-squares
    (r - r_inf)/eps = P_4/Q_3 on the reference fit's states with r_inf held at the tuned value.

    python3 finiteD_series.py noise STATEDIR [--json=F]          the data's precision (local fits, no extrapolation)
    python3 finiteD_series.py planted STATEDIR [--family=rat,q3] [--noise=0|measured] [--seeds=20] [--mutate=offset]
    python3 finiteD_series.py states STATEDIR [--mutate=target|scale1|reverse] [--json=F]
    python3 finiteD_series.py controls STATEDIR                  Pm, Am, Fm and reverse together
    python3 finiteD_series.py info STATEDIR                      post hoc: the K sequence past the grid, rational fits

states --json writes the medians, the verdicts and the form's Chebyshev coefficients.

planted exits 1 when P1 (or P3, with noise) fails, and with --mutate=offset exits 0 only when Pm fires. states exits 1
when A1 or F1 fails; with a mutation it exits 0 only when that mutation's criterion fires. controls exits 0 only when
every control behaves.
"""
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
AMP = os.path.join(REPO, "solver", "inner_amp")
TUNED_JSON = [os.path.join(AMP, "scan45J1250", "sigF_w1.0000_K45_J1250_p1280.json"),
              os.path.join(AMP, "scan45J1500", "sigF_w1.0000_K45_J1500_p1280.json")]
A_TARGET = 2.71918                       # EEG's A_*/sqrt 2, the notes' tuned profile (data/tuned_profile_A2.71918.txt)
A_OTHER = (2.72, 2.7192053)              # the notes' default and the states' A_inf (EEG_PROFILE.md 3.5.1)
NUS = (1, 3, 5, 7, 9)                    # the ratios fitted
NU_ACC = (1, 3, 5, 7)                    # the acceptance's ratios: harmonics through nu = 9
NU_MAX = 13                              # harmonics solved in every full state (kfreeze 13 from D 35)
CHAIN_D = tuple(range(80, 180, 10)) + (180, 190, 200, 205, 209, 215)
DMIN = 15.0
KS = (7, 8, 9)                           # revision 1 (was 4-8): see the docstring
DLOS = (15.0, 20.0)                      # revision 1 (was 15, 20, 30)
VARS = ("eps", "inv_n")
K_REF = 8
REF = ("eps", K_REF, 15.0, True)
OFFSET = 5e-4
TOL_FORM = 1e-4
F2_DMAX = 61.5                           # F2's fits stop at the double-double states
STEP_D = 35.0                            # the scan pins p1's harmonics above 13 from here (kfreeze 13)
STEP_FIT = True                          # every fit carries a step column at STEP_D (revision 1, see the docstring)
NOISE_WINDOW = (9, 5)                    # revision 1: (7, 3) was set by the local fit's truncation at D < 35
SCALE_D, SCALE_NU, SCALE = 40.0201, 5, 1.01
FAMILIES = ("rat", "q3")
INFO_FAMILIES = ("pade",)                # revision 1: the real data's shape, information only
PADE_LM = (4, 3)
ANCHOR_D = (20.0437, 34.0521, 60.0020)   # the real ratios there size the planted families


def arg(argv, name, default=None):
    return next((a.split("=", 1)[1] for a in argv if a.startswith(f"--{name}=")), default)


# ------------------------------------------------------------------ states
def load_z(path):
    """(meta, z): the state's float64 sum of parts (hp/held_fit.py load)"""
    d = np.load(path)
    m = json.loads(str(d["meta"]))
    parts = [d["zh"], d["zl"]] + [d[k] for k in ("z2", "z3", "z4", "z5", "z6", "z7") if k in d]
    z = parts[-1].astype(float)
    for q in parts[-2::-1]:
        z = z + q
    return m, z


def odd_harmonics(f):
    """c_nu, nu = 0..M, of an antiperiodic half-grid function f, f = sum 2 Re(c_nu e^{i nu theta}), theta_j = pi j/M"""
    M = len(f)
    return np.fft.rfft(np.r_[f, -f]) / (2 * M)


def gauge(c):
    """c shifted in theta so that c_1 = -i|c_1|"""
    sh = -math.pi / 2 - np.angle(c[1])
    return c * np.exp(1j * np.arange(len(c)) * sh)


def record(D, M, F, kind, path=""):
    n = D - 2.0
    return dict(D=D, n=n, eps=1.0 / (D - 1.0), inv_n=1.0 / n, M=M, F=F, kind=kind, path=path)


def state_record(path, kind):
    m, z = load_z(path)
    D = float(m["D_str"]) if "D_str" in m else float(m["D"])
    M = int(m["M"])
    F = gauge(odd_harmonics(z[:M]) * math.sqrt(D - 2.0))
    return record(D, M, F, kind, path)


def state_paths(statedir):
    full = [p for p in sorted(glob.glob(os.path.join(REPO, "hp", "data", "states", "*.npz")))]
    full += sorted(glob.glob(os.path.join(statedir, "T84-mac", "pin5_*_full.npz")))
    full += sorted(glob.glob(os.path.join(statedir, "T84-mac", "pin7_*_full.npz")))
    full += [os.path.join(statedir, "T81", "conv_pin7_D72.8000_D72.800000.npz"),
             os.path.join(statedir, "T81", "conv_pin5_D73.0000_D73.000000.npz")]
    chain = []
    for d in CHAIN_D:
        sub = "chain" if d < 174 else "chain_o"
        chain.append(os.path.join(statedir, "T138-win2", sub, f"anm_pin3_D{d:.4f}.npz"))
    return full, chain


def load_states(statedir):
    full, chain = state_paths(statedir)
    recs, seen = [], set()
    for p in full:
        r = state_record(p, "full")
        key = round(r["D"], 6)
        if r["D"] < DMIN or key in seen:     # T84's pin5 and pin7 legs converge to the same state
            continue
        seen.add(key)
        recs.append(r)
    for p in chain:
        recs.append(state_record(p, "chain"))
    recs.sort(key=lambda r: r["D"])
    return recs


def ratios(F):
    return {nu: F[nu + 2] / F[nu] for nu in NUS}


# ------------------------------------------------------------------ the tuned D = infinity profile (tuned_profile.py)
def tuned_ratios(A, path=TUNED_JSON[1], numax=15):
    """r_nu of the tuned profile summed at amplitude A (tuned_profile.py load_f and harmonics)"""
    d = json.load(open(path))
    f = {}
    for key, (re_, im_) in d["f"].items():
        k, nu = map(int, key.split("_"))
        f[(k, nu)] = complex(re_, im_)
    F = {1: -0.5j * A}
    for nu in range(3, numax + 1, 2):
        F[nu] = sum(f.get((k, nu), 0) * A ** k for k in range(nu, d["K"] + 1, 2))
    return {nu: F[nu + 2] / F[nu] for nu in range(1, numax - 1, 2)}


# ------------------------------------------------------------------ the fits
def fit_poly(x, y, K, at=0.0, h=None):
    """complex least squares y = sum_{k <= K} c_k T_k(t) (+ s h), t the affine map of x's range onto [-1, 1], h an
    optional indicator column (the step at STEP_D); returns the fit's value at x = at (with h = 1), the residuals and a
    callable ev(x, h=1)"""
    lo, hi = float(x.min()), float(x.max())
    t = (2 * x - (lo + hi)) / (hi - lo)
    V = np.polynomial.chebyshev.chebvander(t, K)
    if h is not None:
        V = np.hstack([V, np.asarray(h, dtype=float)[:, None]])
    c, *_ = np.linalg.lstsq(V, y, rcond=None)

    def ev(xx, hh=1.0):
        tt = (2 * np.asarray(xx, dtype=float) - (lo + hi)) / (hi - lo)
        return np.polynomial.chebyshev.chebval(tt, c[:K + 1]) + (c[K + 1] * hh if h is not None else 0.0)
    ev.coef, ev.domain = c, (lo, hi)        # Chebyshev coefficients (then the step's) on the x range
    return complex(ev(at)), y - V @ c, ev


def use_for(recs, nu, dlo, chain, dmax=None):
    out = [r for r in recs if r["D"] >= dlo and (dmax is None or r["D"] <= dmax)]
    if nu != 1 or not chain:
        out = [r for r in out if r["kind"] == "full"]
    return out


def step_col(use):
    """the indicator of the pinned states (D >= STEP_D), when STEP_FIT and at least 3 states lie on each side"""
    h = np.array([1.0 if r["D"] >= STEP_D else 0.0 for r in use])
    return h if (STEP_FIT and h.sum() >= 3 and (1 - h).sum() >= 3) else None


def one_fit(recs, nu, var, K, dlo, chain, key="r", dmax=None):
    use = use_for(recs, nu, dlo, chain, dmax)
    x = np.array([r[var] for r in use])
    y = np.array([r[key][nu] for r in use])
    v0, res, ev = fit_poly(x, y, K, h=step_col(use))
    return dict(r0=v0, rms=float(np.sqrt(np.mean(np.abs(res) ** 2))), maxres=float(np.abs(res).max()), n=len(use),
                ev=ev, D=[r["D"] for r in use], res=res)


def grid(nu):
    ch = (True, False) if nu == 1 else (False,)
    return [(v, k, d, c) for v in VARS for k in KS for d in DLOS for c in ch]


def cmedian(vals):
    v = np.asarray(vals)
    return complex(np.median(v.real), np.median(v.imag))


def xval(var, D):
    return 1.0 / (D - 1.0) if var == "eps" else 1.0 / (D - 2.0)


def grid_fit(recs, nu, key="r", choices=None, at_D=()):
    """the grid's fits: D = infinity values, their median and spread s; with at_D, every fit's value at those D too"""
    rows = []
    for (v, k, d, c) in (choices or grid(nu)):
        f = one_fit(recs, nu, v, k, d, c, key)
        rows.append(dict(var=v, K=k, dlo=d, chain=c, r0=f["r0"], rms=f["rms"], n=f["n"],
                         at=[complex(f["ev"](xval(v, D))) for D in at_D]))
    vals = [r["r0"] for r in rows]
    med = cmedian(vals)
    s = max(abs(v - med) for v in vals)
    ref = next((r for r in rows if (r["var"], r["K"], r["dlo"], r["chain"] or nu != 1) == (REF[0], REF[1], REF[2], True)),
               None)
    return dict(rows=rows, median=med, s=s, ref=ref["r0"] if ref else None)


def add_ratios(recs):
    for r in recs:
        r["r"] = ratios(r["F"])
        r["A"] = 2 * abs(r["F"][1])
    return recs


def fc(z, d=7):
    return f"{z.real:+.{d}f}{z.imag:+.{d}f}i"


def print_grid(nu, g, truth=None, label=""):
    ref = truth if truth is not None else 0.0
    print(f"\n== r_{nu} {label}: D = infinity over the grid" + (" minus the truth" if truth is not None else ""))
    cols = [(v, d, c) for v in VARS for d in DLOS for c in ((True, False) if nu == 1 else (False,))]
    print("   K  " + "".join(f"{'%s D>=%g%s' % ('eps' if v == 'eps' else '1/n', d, ' ch' if (c and nu == 1) else ''):>27s}"
                           for v, d, c in cols))
    for k in KS:
        line = f"  {k:2d}  "
        for v, d, c in cols:
            x = next(r for r in g["rows"] if (r["var"], r["K"], r["dlo"], r["chain"]) == (v, k, d, c))
            z = x["r0"] - ref
            line += f"{z.real:+13.3e}{z.imag:+13.3e}i" if truth is not None else f"  {fc(x['r0'], 8):>25s}"
        print(line)
    med = g["median"] - ref
    print(f"   median {fc(g['median'], 8)} (|r| {abs(g['median']):.7f}, arg {np.angle(g['median']):+.6f}); "
          f"spread s = {g['s']:.2e}" + (f"; median - truth {fc(med, 9)}, |.| {abs(med):.2e}" if truth is not None else ""))


# ------------------------------------------------------------------ the data's precision
def noise(recs, verbose=True, window=NOISE_WINDOW[0], deg=NOISE_WINDOW[1]):
    """per nu: rms of the residuals of local fits (window consecutive full states, degree deg in eps), by group; an
    estimator of the data's noise (it includes the local fit's own truncation, whose floor on the planted families is
    printed by `noise`), not a bound"""
    full = [r for r in recs if r["kind"] == "full"]
    groups = (("D 15-35", 15.0, 35.0), ("D 35-61", 35.0, 61.5), ("D 68-73", 61.5, 74.0))
    out = {}
    for nu in NUS:
        res = np.full(len(full), np.nan)
        for i in range(len(full)):
            lo = max(0, min(i - window // 2, len(full) - window))
            w = full[lo:lo + window]
            x = np.array([r["eps"] for r in w])
            y = np.array([r["r"][nu] for r in w])
            _, rr, _ = fit_poly(x, y, deg)
            res[i] = abs(rr[i - lo]) * math.sqrt(window / (window - deg - 1))
        out[nu] = {g: float(np.sqrt(np.nanmean(res[[lo <= r["D"] < hi for r in full]] ** 2))) for g, lo, hi in groups}
        out[nu]["all"] = float(np.sqrt(np.nanmean(res ** 2)))
        out[nu]["max"] = float(np.nanmax(res))
    if verbose:
        print(f"\n== the data's precision: rms of local residuals ({window} consecutive full states, degree {deg} in eps),"
              " absolute in r")
        print("  nu " + "".join(f"{g:>11s}" for g, _, _ in groups) + "        all        max     |r| at D 60")
        r60 = next(r for r in full if abs(r["D"] - 60.002) < 0.01)
        for nu in NUS:
            print(f"  {nu:2d} " + "".join(f"{out[nu][g]:11.2e}" for g, _, _ in groups)
                  + f"{out[nu]['all']:11.2e}{out[nu]['max']:11.2e}   {abs(r60['r'][nu]):.5f}")
    return out


def step_fits(recs, verbose=True):
    """the reference fit with an extra column H(D - 35) (the kfreeze 13 change) and, for r_1, H(D - 76.5) (full states
    to chain): the steps' coefficients"""
    out = {}
    for nu in NUS:
        use = use_for(recs, nu, REF[2], REF[3])
        x = np.array([r["eps"] for r in use])
        y = np.array([r["r"][nu] for r in use])
        lo, hi = x.min(), x.max()
        t = (2 * x - (lo + hi)) / (hi - lo)
        cols = [np.polynomial.chebyshev.chebvander(t, K_REF), np.array([[1.0 if r["D"] >= 35.0 else 0.0] for r in use]),
                np.array([[1.0 if r["D"] >= 61.5 and r["kind"] == "full" else 0.0] for r in use])]
        if nu == 1:
            cols.append(np.array([[1.0 if r["kind"] == "chain" else 0.0] for r in use]))
        V = np.hstack(cols)
        c, *_ = np.linalg.lstsq(V, y, rcond=None)
        out[nu] = dict(step35=complex(c[K_REF + 1]), step_qd=complex(c[K_REF + 2]),
                       step_chain=complex(c[K_REF + 3]) if nu == 1 else None)
    if verbose:
        print("\n== steps fitted beside the reference polynomial (eps, K 8, D >= 15): at D 35 (kfreeze 13 from there), at"
              " the quad-double states (68.5-73), and for r_1 at the chain (D >= 80)")
        for nu in NUS:
            s = out[nu]
            print(f"  r_{nu}: step at 35 {s['step35'].real:+.2e}{s['step35'].imag:+.2e}i, at the quad-double states "
                  f"{s['step_qd'].real:+.2e}{s['step_qd'].imag:+.2e}i"
                  + (f", at the chain {s['step_chain'].real:+.2e}{s['step_chain'].imag:+.2e}i" if s["step_chain"] is not None else ""))
    return out


# ------------------------------------------------------------------ planted truth
def family_params(recs, tuned):
    """per nu: the planted families' parameters, fitted through the real ratios at ANCHOR_D with the tuned D = infinity
    value as intercept. rat: (r - r_inf)/eps = a/(1 + b eps), through the first and last anchors. q3:
    (r - r_inf)/eps = a/(1 + b1 eps + b2 eps^2), through all three."""
    an = [next(r for r in recs if abs(r["D"] - d) < 1e-3) for d in ANCHOR_D]
    out = {}
    for nu in NUS:
        rinf = tuned[nu]
        e = np.array([r["eps"] for r in an])
        y = np.array([(r["r"][nu] - rinf) / r["eps"] for r in an])
        b = (y[2] - y[0]) / (y[0] * e[0] - y[2] * e[2])
        a = y[0] * (1 + b * e[0])
        Mq = np.array([[1.0, -e[i] * y[i], -e[i] ** 2 * y[i]] for i in range(3)])
        aq, b1, b2 = np.linalg.solve(Mq, y)
        out[nu] = dict(rinf=rinf, rat=(a, b), q3=(aq, b1, b2), pade=pade_params(recs, nu, rinf))
    return out


def pade_params(recs, nu, rinf, L=PADE_LM[0], M=PADE_LM[1], step=None):
    """the information family pade: (r - r_inf)/eps = P_L(eps)/Q_M(eps), Q_M(0) = 1, by linearized least squares
    (rows weighted by eps, so residuals are in r) on every state the reference fit takes, after the measured step at 35
    (step, or step_fits') is taken off; r_inf is held at the tuned value, so the family has the real data's shape and a
    known intercept"""
    use = use_for(recs, nu, REF[2], REF[3])
    st = step_fits(recs, verbose=False)[nu]["step35"] if step is None else step
    e = np.array([r["eps"] for r in use])
    y = np.array([(r["r"][nu] - (st if r["D"] >= STEP_D else 0) - rinf) / r["eps"] for r in use])
    X = np.array([[ei ** l for l in range(L + 1)] + [-(ei ** m) * yi for m in range(1, M + 1)] for ei, yi in zip(e, y)])
    sol, *_ = np.linalg.lstsq(X * e[:, None], y * e, rcond=None)
    return (sol[:L + 1], np.r_[1.0, sol[L + 1:]])


def planted_r(par, nu, fam, eps):
    p = par[nu]
    if fam == "rat":
        a, b = p["rat"]
        return p["rinf"] + a * eps / (1 + b * eps)
    if fam == "pade":
        P, Q = p["pade"]
        return p["rinf"] + eps * np.polyval(P[::-1], eps) / np.polyval(Q[::-1], eps)
    a, b1, b2 = p["q3"]
    return p["rinf"] + a * eps / (1 + b1 * eps + b2 * eps * eps)


def group_of(r):
    return "D 15-35" if r["D"] < 35.0 else "D 35-61" if r["D"] < 61.5 else "D 68-73"


def pin_weight(recs, r):
    """the pinning's bias shape: 0 below STEP_D, from there the size of the first pinned harmonic, |F_15| ~
    |F_13|^2/|F_11| of the real states, relative to its value at the first pinned state (the chain takes the last
    full state's)"""
    if r["D"] < STEP_D:
        return 0.0
    full = [q for q in recs if q["kind"] == "full" and q["D"] >= STEP_D]
    prox = lambda q: abs(q["F"][13]) ** 2 / abs(q["F"][11])  # noqa: E731
    return prox(r if r["kind"] == "full" else full[-1]) / prox(full[0])


def planted_records(recs, par, fam, sig=None, rng=None, steps=None):
    """each state's F replaced by F_1 = -i A_pl/2 and F_{nu+2} = r_nu F_nu (planted r_nu, nu <= 9; r_9's family carried
    to nu = 13, zero above 13 from D 35 as the scan pins), shifted in theta by a known phase, synthesized on the state's
    half grid and passed through the same harmonics, gauge and ratios. sig: {nu: {group: rms}}, complex Gaussian noise
    added to r_nu with the rms of the state's group (the chain's is the D 68-73 group's); steps: {nu: (step at 35,
    step at the chain)} added to r_nu from D 35 and at the chain states (the measured steps, step_fits). The truth is
    the noise-free plant, steps included."""
    out = []
    for r in recs:
        n, eps, M = r["n"], r["eps"], r["M"]
        A = 2.7180 + 14.52 / n - 33.0 / n ** 2 / (1 + 12.0 / n)      # EEG_PROFILE.md 3.5.1's alt12
        rr = {nu: planted_r(par, nu, fam, eps) for nu in NUS}
        if steps is not None:
            for nu in NUS:
                rr[nu] += steps[nu][0] * pin_weight(recs, r) + (steps[nu][1] if r["kind"] == "chain" else 0)
        truth_r = dict(rr)
        if sig is not None:
            for nu in NUS:
                rr[nu] += sig[nu][group_of(r)] * (rng.normal() + 1j * rng.normal()) / math.sqrt(2)
        top = NU_MAX if (r["D"] >= 35.0 or r["kind"] == "chain") else M - 1
        F = np.zeros(M + 1, dtype=complex)
        F[1] = -0.5j * A
        for nu in range(1, top - 1, 2):
            F[nu + 2] = F[nu] * rr[min(nu, NUS[-1])]
        phi = 2 * np.pi * ((0.37 * r["D"] + 0.11) % 1.0)
        th = np.pi * np.arange(M) / M
        f = sum(2 * np.real(F[nu] * np.exp(1j * nu * (th + phi))) for nu in range(1, top + 1, 2)) / math.sqrt(n)
        q = record(r["D"], M, gauge(odd_harmonics(f) * math.sqrt(n)), r["kind"])
        q["truth"] = truth_r
        out.append(q)
    return add_ratios(out)


def cmd_planted(argv):
    statedir = [a for a in argv if not a.startswith("--")][0]
    fams = arg(argv, "family", ",".join(FAMILIES + INFO_FAMILIES)).split(",")
    nz = arg(argv, "noise", "0")
    seeds = int(arg(argv, "seeds", 20))
    mut = arg(argv, "mutate")
    recs = add_ratios(load_states(statedir))
    tuned = tuned_ratios(A_TARGET)
    par = family_params(recs, tuned)
    print("== planted families: parameters and the distance from the real ratios at every full state (realism)")
    st0 = step_fits(recs, verbose=False)
    for nu in NUS:
        a, b = par[nu]["rat"]
        aq, b1, b2 = par[nu]["q3"]
        poles = np.roots([b2, b1, 1.0])
        full = [r for r in recs if r["kind"] == "full"]
        dist = {f: max(abs(planted_r(par, nu, f, r["eps"]) + (st0[nu]["step35"] if (f == "pade" and r["D"] >= STEP_D)
                                                                else 0) - r["r"][nu]) for r in full)
                for f in FAMILIES + INFO_FAMILIES}
        pp = np.roots(par[nu]["pade"][1][::-1])
        print(f"  r_{nu}: rat a {fc(a, 4)} pole eps {fc(-1 / b, 4)};  q3 a {fc(aq, 4)} poles eps "
              f"{', '.join(fc(p, 4) for p in poles)};  pade poles eps {', '.join(fc(p, 4) for p in pp)};  "
              "max |plant - real| " + ", ".join(f"{f} {v:.1e}" for f, v in dist.items()))
    sig, steps = None, None
    if nz == "measured":
        nzv = noise(recs, verbose=False)
        sig = {nu: {g: nzv[nu][g] for g in ("D 15-35", "D 35-61", "D 68-73")} for nu in NUS}
        st = step_fits(recs, verbose=False)
        steps = {nu: (st[nu]["step35"], st[nu]["step_chain"] or 0.0) for nu in NUS}
        print("  noise added to r_nu (complex Gaussian, rms per group D 15-35, 35-61, 68-73 and chain): "
              + "; ".join(f"r_{nu} " + ", ".join(f"{v:.1e}" for v in sig[nu].values()) for nu in NUS))
        print("  steps added from D 35 (and at the chain): " + "; ".join(
            f"r_{nu} {abs(steps[nu][0]):.1e}" + (f" ({abs(steps[nu][1]):.1e})" if nu == 1 else "") for nu in NUS))
    off = OFFSET if mut == "offset" else 0.0
    summ, ok_all, oks = {}, True, {}
    nrun = seeds if sig is not None else 1
    for fam in fams:
        cover = {nu: 0 for nu in NUS}
        errs = {nu: [] for nu in NUS}
        for seed in range(nrun):
            rng = np.random.default_rng(seed)
            prec = planted_records(recs, par, fam, sig, rng, steps)
            for nu in NUS:
                g = grid_fit(prec, nu)
                truth = par[nu]["rinf"] + off
                err = abs(g["median"] - truth)
                cover[nu] += err <= g["s"]
                errs[nu].append((err, g["s"], abs(g["ref"] - truth)))
                if seed == 0 and sig is None:
                    print_grid(nu, g, truth, f"planted {fam}")
            if seed == 0:
                okf, fo = form_check(prec, verbose=False)
                print(f"  planted {fam}: F1 on the planted harmonics, worst {fo['worst'][0]:.2e} at D {fo['worst'][1]:.2f},"
                      f" nu {fo['worst'][2]} ({'pass' if okf else 'FAIL'}; information)")
                oke, eo = extrap_check(prec, verbose=False)
                print(f"  planted {fam}: E1, r_1 from D <= 73 at the chain's D, covered at {eo['covered']}/{eo['n']} D, "
                      f"worst |median - state| {eo['worst']:.2e} (information)")
        print(f"\n== planted {fam}" + (f", noise, {nrun} seeds" if sig is not None else "") + (f", offset {off}" if off else ""))
        print("  nu  covered   median error (max)   spread s (min..max)   reference choice's error (max)")
        for nu in NUS:
            e = np.array(errs[nu])
            need = nrun if sig is None else 18
            ok = cover[nu] >= need
            counted = nu in NU_ACC and fam in FAMILIES
            oks[(fam, nu)] = ok
            if counted:
                ok_all &= ok
            print(f"  {nu:2d}  {cover[nu]:2d}/{nrun:<2d}   {e[:, 0].max():.2e}            {e[:, 1].min():.2e}..{e[:, 1].max():.2e}"
                  f"      {e[:, 2].max():.2e}   {'pass' if ok else 'FAIL'}{'' if counted else ' (information)'}")
            summ[f"{fam}_{nu}"] = dict(covered=cover[nu], runs=nrun, err_max=float(e[:, 0].max()),
                                       s_min=float(e[:, 1].min()), s_max=float(e[:, 1].max()), ref_err=float(e[:, 2].max()))
    crit = "P3" if sig is not None else "P1"
    if mut == "offset":
        held = [(f, nu) for f in FAMILIES for nu in NU_ACC if (f, nu) in oks and oks[(f, nu)]]
        fired = not held
        print(f"\nPm ({crit} with the truth moved by {OFFSET:g}, must fail for every nu and family): "
              + ("fires (control behaves)" if fired else f"does NOT fire: still covered for {held}"))
        return 0 if fired else 1, summ
    print(f"\n{crit}: {'pass' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1, summ


# ------------------------------------------------------------------ the finite-D form
def form(recs, dmax=None):
    """the reference fits of A and r_1..r_9 as functions of eps; returns F_form(eps) -> harmonics nu = 1..9 (dict).
    dmax: the fits take only the full states with D <= dmax (F2)"""
    use = [r for r in recs if r["D"] >= REF[2] and (dmax is None or (r["D"] <= dmax and r["kind"] == "full"))]
    xa = np.array([r["eps"] for r in use])
    _, _, evA = fit_poly(xa, np.array([r["A"] for r in use], dtype=complex), K_REF, h=step_col(use))
    evr = {nu: one_fit(recs, nu, REF[0], REF[1], REF[2], REF[3] and dmax is None, dmax=dmax)["ev"] for nu in NUS}

    def F_of(eps):
        hh = 1.0 if 1.0 + 1.0 / eps >= STEP_D else 0.0          # the step column, as the data carry it
        F = {1: -0.5j * float(np.real(evA(eps, hh)))}
        for nu in (1, 3, 5, 7):
            F[nu + 2] = F[nu] * complex(evr[nu](eps, hh))
        return F
    coef = dict(A0=float(np.real(evA(0.0))), r0={nu: complex(evr[nu](0.0)) for nu in NUS},
                cheb={"A": (evA.coef, evA.domain), **{nu: (evr[nu].coef, evr[nu].domain) for nu in NUS}})
    return F_of, evr, coef


def form_check(recs, verbose=True):
    F_of, evr, coef = form(recs)
    rows, worst, worst_held = [], (0.0, None, None), (0.0, None, None)
    for r in recs:
        Ff = F_of(r["eps"])
        e = {nu: abs(Ff[nu] / r["F"][nu] - 1) for nu in (1, 3, 5, 7, 9)}
        tested = (1, 3, 5, 7, 9) if r["kind"] == "full" else (1, 3)
        for nu in (1, 3, 5, 7, 9):
            if nu in tested and e[nu] > worst[0]:
                worst = (e[nu], r["D"], nu)
            if nu not in tested and e[nu] > worst_held[0]:
                worst_held = (e[nu], r["D"], nu)
        rows.append(dict(D=r["D"], kind=r["kind"], err=e))
    ok = worst[0] <= TOL_FORM
    if verbose:
        print(f"\n== the finite-D form (reference fits: eps, K {K_REF}, D >= 15, r_1 with the chain): |F_form/F_state - 1|")
        print("        D  kind     nu=1      nu=3      nu=5      nu=7      nu=9")
        for x in rows:
            if x["kind"] == "chain" or any(abs(x["D"] - d) < 0.6 for d in (15, 20, 30, 40, 50, 60, 68.5, 73)):
                print(f"  {x['D']:7.3f}  {x['kind']:5s}" + "".join(f"{x['err'][nu]:10.2e}" for nu in (1, 3, 5, 7, 9))
                      + ("   (5-9 held)" if x["kind"] == "chain" else ""))
        print(f"  worst tested: {worst[0]:.2e} at D {worst[1]:.3f}, nu {worst[2]}; worst held (information): "
              f"{worst_held[0]:.2e} at D {worst_held[1]}, nu {worst_held[2]}")
        print(f"F1 (tolerance {TOL_FORM:g}): {'pass' if ok else 'FAIL'}")
    return ok, dict(worst=worst, worst_held=worst_held, rows=rows, coef=coef)


def form_ahead(recs, verbose=True):
    """F2: the form fitted on the full states with D <= F2_DMAX predicts the quad-double states' F_nu, nu = 1..9"""
    F_of, _, _ = form(recs, dmax=F2_DMAX)
    rows, worst = [], (0.0, None, None)
    for r in recs:
        if r["kind"] != "full" or r["D"] <= F2_DMAX:
            continue
        Ff = F_of(r["eps"])
        e = {nu: abs(Ff[nu] / r["F"][nu] - 1) for nu in (1, 3, 5, 7, 9)}
        for nu, v in e.items():
            if v > worst[0]:
                worst = (v, r["D"], nu)
        rows.append(dict(D=r["D"], err=e))
    ok = worst[0] <= TOL_FORM
    if verbose:
        print(f"\n== F2: the form fitted on D <= {F2_DMAX:g} against the quad-double states: |F_form/F_state - 1|")
        print("        D     nu=1      nu=3      nu=5      nu=7      nu=9")
        for x in rows:
            print(f"  {x['D']:7.3f}" + "".join(f"{x['err'][nu]:10.2e}" for nu in (1, 3, 5, 7, 9)))
        print(f"F2 (tolerance {TOL_FORM:g}): {'pass' if ok else 'FAIL'}, worst {worst[0]:.2e} at D {worst[1]}, nu {worst[2]}")
    return ok, dict(worst=worst, rows=rows)


def extrap_check(recs, verbose=True):
    """E1: r_1's grid without the chain (D <= 73), evaluated at each chain state's D: the state's r_1 within the
    grid's spread there, |median - r_1(state)| <= s"""
    chainD = [r["D"] for r in recs if r["kind"] == "chain"]
    choices = [c for c in grid(1) if not c[3]]
    g = grid_fit(recs, 1, choices=choices, at_D=chainD)
    rows, cov, worst = [], 0, 0.0
    for i, D in enumerate(chainD):
        vals = [x["at"][i] for x in g["rows"]]
        med = cmedian(vals)
        s = max(abs(v - med) for v in vals)
        st = next(r for r in recs if r["kind"] == "chain" and r["D"] == D)["r"][1]
        d = abs(med - st)
        cov += d <= s
        worst = max(worst, d)
        rows.append(dict(D=D, med=med, s=s, dist=d, state=st))
    ok = cov == len(chainD)
    if verbose:
        print("\n== E1: r_1 from the full states (D <= 73, the grid without the chain) at the chain's D")
        print("        D   state's r_1                 median - state            |.|        s       covered")
        for x in rows:
            print(f"  {x['D']:7.2f}  {fc(x['state'], 8)}  {fc(x['med'] - x['state'], 9)}  {x['dist']:.2e}  {x['s']:.2e}  "
                  f"{'yes' if x['dist'] <= x['s'] else 'NO'}")
        print(f"E1: {'pass' if ok else 'FAIL'} ({cov}/{len(chainD)})")
    return ok, dict(covered=cov, n=len(chainD), worst=worst, rows=rows)


# ------------------------------------------------------------------ real data
def mutate_recs(recs, mut):
    if mut == "scale1":
        for r in recs:
            if abs(r["D"] - SCALE_D) < 1e-3:
                F = r["F"].copy()
                F[SCALE_NU + 2:] *= SCALE
                r["F"] = F
        return add_ratios(recs)
    if mut == "reverse":
        for r in recs:
            r["F"] = gauge(np.conj(r["F"]))
        return add_ratios(recs)
    return recs


def analysis(recs, mut=None, verbose=True):
    tuned = {a: tuned_ratios(a) for a in (A_TARGET,) + A_OTHER}
    out = dict(nu={})
    ok = True
    for nu in NUS:
        g = grid_fit(recs, nu)
        if verbose:
            print_grid(nu, g, None, "real data")
        tgt = tuned[A_TARGET][nu + 2] if mut == "target" else tuned[A_TARGET][nu]
        d = abs(g["median"] - tgt)
        hit = d <= g["s"]
        if nu in NU_ACC:
            ok &= hit
        other = {a: abs(g["median"] - tuned[a][nu]) for a in A_OTHER}
        out["nu"][nu] = dict(median=g["median"], s=g["s"], ref=g["ref"], target=tgt, dist=d, hit=hit, other=other,
                             gap_next=abs(tuned[A_TARGET][nu + 2] - tuned[A_TARGET][nu]),
                             best=min(g["rows"], key=lambda x: x["rms"]))
    if verbose:
        print("\n== D = infinity against the tuned profile" + (" (--mutate=target: against r_{nu+2})" if mut == "target" else ""))
        print("  nu   median                     |median|   arg        s         tuned (A 2.71918)           |median - tuned|  "
              "verdict    |.| vs A 2.72, 2.7192053   |r_{nu+2} - r_nu| tuned")
        for nu in NUS:
            x = out["nu"][nu]
            print(f"  {nu:2d}  {fc(x['median'], 8)}  {abs(x['median']):.7f}  {np.angle(x['median']):+.6f}  {x['s']:.2e}  "
                  f"{fc(x['target'], 8)}  {x['dist']:.2e}         {'within s' if x['hit'] else 'OUTSIDE s'}"
                  f"{'' if nu in NU_ACC else ' (info)'}   {x['other'][2.72]:.1e}, {x['other'][2.7192053]:.1e}            {x['gap_next']:.2e}")
        print(f"A1{' (mutated)' if mut else ''}: {'pass' if ok else 'FAIL'}")
    return ok, out


def cmd_noise(argv):
    statedir = [a for a in argv if not a.startswith("--")][0]
    recs = add_ratios(load_states(statedir))
    print(f"{len(recs)} states: {sum(r['kind'] == 'full' for r in recs)} full (D {recs[0]['D']:.2f}-"
          f"{max(r['D'] for r in recs if r['kind'] == 'full'):.2f}), {sum(r['kind'] == 'chain' for r in recs)} chain")
    nz = noise(recs)
    par = family_params(recs, tuned_ratios(A_TARGET))
    for fam in FAMILIES:
        fl = noise(planted_records(recs, par, fam), verbose=False)
        print(f"  the estimator's floor on the noise-free {fam} plant: " + "; ".join(
            f"r_{nu} " + ", ".join(f"{fl[nu][g]:.1e}" for g in ("D 15-35", "D 35-61", "D 68-73")) for nu in NUS))
    st = step_fits(recs)
    js = arg(argv, "json")
    if js:
        json.dump(dict(noise={str(k): v for k, v in nz.items()},
                       steps={str(k): {kk: (None if vv is None else [vv.real, vv.imag]) for kk, vv in v.items()}
                              for k, v in st.items()}), open(js, "w"), indent=1)
    return 0


def cmd_states(argv):
    statedir = [a for a in argv if not a.startswith("--")][0]
    mut = arg(argv, "mutate")
    recs = add_ratios(load_states(statedir))
    print(f"{len(recs)} states: {sum(r['kind'] == 'full' for r in recs)} full, {sum(r['kind'] == 'chain' for r in recs)} chain")
    base_out = None
    if mut == "scale1":
        _, base_out = analysis(recs, verbose=False)
        recs = mutate_recs(recs, mut)
    elif mut == "reverse":
        recs = mutate_recs(recs, mut)
    ok_a, out = analysis(recs, mut if mut == "target" else None)
    ok_f, fout = form_check(recs)
    ok_f2, f2out = form_ahead(recs)
    ok_e, eout = extrap_check(recs)
    if mut == "scale1":
        print("\n  the medians' move under --mutate=scale1:")
        for nu in NUS:
            print(f"    r_{nu}: {abs(out['nu'][nu]['median'] - base_out['nu'][nu]['median']):.2e} (s {base_out['nu'][nu]['s']:.2e})")
    js = arg(argv, "json")
    if js:
        cv = lambda z: [z.real, z.imag] if isinstance(z, complex) else z  # noqa: E731
        dump = dict(nu={str(nu): {k: cv(v) for k, v in x.items() if k not in ("best", "other")} for nu, x in out["nu"].items()},
                    form=dict(worst=fout["worst"], worst_held=fout["worst_held"], A0=fout["coef"]["A0"],
                              r0={str(k): cv(v) for k, v in fout["coef"]["r0"].items()},
                              model="F_1 = -i A(eps)/2, F_{nu+2} = r_nu(eps) F_nu; each of A, r_nu = sum_k c_k T_k(t) "
                                    "+ c_{K+1} H(D - 35), t = (2 eps - lo - hi)/(hi - lo), with [c, (lo, hi)] below",
                              cheb={str(k): dict(c=[[complex(z).real, complex(z).imag] for z in c], domain=list(d))
                                    for k, (c, d) in fout["coef"]["cheb"].items()}),
                    F2=dict(worst=f2out["worst"]), E1=dict(covered=eout["covered"], n=eout["n"], worst=eout["worst"]),
                    verdicts=dict(A1=ok_a, F1=ok_f, F2=ok_f2, E1=ok_e))
        json.dump(dump, open(js, "w"), indent=1, default=str)
    if mut == "target":
        fired = [nu for nu in NU_ACC if not out["nu"][nu]["hit"]]
        print(f"\nAm: A1 against r_(nu+2) fails for nu {fired}" + ("" if len(fired) == len(NU_ACC) else
              f"; it passes for {[nu for nu in NU_ACC if nu not in fired]}, where |r_(nu+2) - r_nu| <= s or the median"
              " lies between"))
        return 0 if fired else 1
    if mut == "scale1":
        print(f"\nFm: F1 {'fails, the control fires' if not ok_f else 'PASSES: the control does not fire'}")
        return 0 if not ok_f else 1
    if mut == "reverse":
        neg = all(np.angle(out["nu"][nu]["median"]) < 0 for nu in NU_ACC)
        print(f"\nreverse: arg of every median {'negative, A1 fails as it must' if neg and not ok_a else 'NOT all negative or A1 passes'}")
        return 0 if (neg and not ok_a) else 1
    print(f"\nverdicts: A1 {'pass' if ok_a else 'FAIL'}, F1 {'pass' if ok_f else 'FAIL'}, F2 {'pass' if ok_f2 else 'FAIL'}, "
          f"E1 {'pass' if ok_e else 'FAIL'}")
    return 0 if (ok_a and ok_f and ok_f2 and ok_e) else 1


def cmd_controls(argv):
    statedir = [a for a in argv if not a.startswith("--")][0]
    me = os.path.abspath(__file__)
    import subprocess
    cases = [(["planted", statedir, "--mutate=offset"], 0), (["states", statedir, "--mutate=target"], 0),
             (["states", statedir, "--mutate=scale1"], 0), (["states", statedir, "--mutate=reverse"], 0)]
    bad = 0
    for a, want in cases:
        p = subprocess.run([sys.executable, me] + a, capture_output=True, text=True)
        good = p.returncode == want
        bad += not good
        last = [ln for ln in p.stdout.splitlines() if ln.strip()][-1] if p.stdout.strip() else p.stderr[-300:]
        print(f"{'ok  ' if good else 'BAD '} {' '.join(a[:1] + a[2:])}: exit {p.returncode}; {last}", flush=True)
    print("all controls behave" if not bad else f"{bad} control(s) misbehave")
    return 0 if not bad else 1


# ------------------------------------------------------------------ information, after the real run (post hoc)
def rational_fit(recs, nu, L, M, dlo=15.0):
    """r(eps) = P_L(eps)/Q_M(eps), Q_M(0) = 1, by linearized least squares (r Q = P) on the reference fit's states with
    D >= dlo, after the measured step at 35 is taken off: the intercept P(0), the poles and the zeros in eps"""
    use = use_for(recs, nu, dlo, REF[3])
    st = step_fits(recs, verbose=False)[nu]["step35"]
    e = np.array([r["eps"] for r in use])
    y = np.array([r["r"][nu] - (st if r["D"] >= STEP_D else 0) for r in use])
    X = np.array([[ei ** l for l in range(L + 1)] + [-(ei ** m) * yi for m in range(1, M + 1)] for ei, yi in zip(e, y)])
    sc = np.abs(X).max(axis=0)
    sol, *_ = np.linalg.lstsq(X / sc, y, rcond=None)
    sol = sol / sc
    P, Q = sol[:L + 1], np.r_[1.0, sol[L + 1:]]
    res = y - np.polyval(P[::-1], e) / np.polyval(Q[::-1], e)
    return dict(r0=complex(P[0]), poles=np.roots(Q[::-1]), zeros=np.roots(P[::-1]),
                maxres=float(np.abs(res).max()))


def k_sequence(recs, nu, var="eps", dlo=15.0, ks=range(5, 13)):
    """the polynomial fit's D = infinity value against K beyond the grid"""
    return [(k, one_fit(recs, nu, var, k, dlo, REF[3])["r0"]) for k in ks]


def shape_rms(rs, nu, c, L, M, step):
    """the rms residual of the shape (r - c)/eps = P_L/Q_M fitted with the intercept held at c, and its nearest pole
    that is not a spurious pole-zero doublet (no zero of P within 1e-4 of it)"""
    use = use_for(rs, nu, REF[2], REF[3])
    P, Q = pade_params(rs, nu, c, L, M, step)
    res = np.array([c + r["eps"] * np.polyval(P[::-1], r["eps"]) / np.polyval(Q[::-1], r["eps"])
                    + (step if r["D"] >= STEP_D else 0) - r["r"][nu] for r in use])
    zeros = np.roots(P[::-1])
    poles = [p for p in np.roots(Q[::-1]) if not len(zeros) or min(abs(zeros - p)) > 1e-4]
    return float(np.sqrt(np.mean(np.abs(res) ** 2))), complex(min(poles, key=abs)) if poles else complex("nan")


def best_intercept(rs, nu, L, M, starts):
    """c minimizing shape_rms over the complex plane (Nelder-Mead in units of 1e-3 from each start)"""
    from scipy.optimize import minimize
    step = step_fits(rs, verbose=False)[nu]["step35"]
    c0 = starts[0]
    f = lambda p: math.log(shape_rms(rs, nu, c0 + complex(p[0], p[1]) * 1e-3, L, M, step)[0])  # noqa: E731
    best = None
    for s in starts:
        x0 = ((s - c0).real * 1e3, (s - c0).imag * 1e3)
        o = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-12, maxiter=3000))
        if best is None or o.fun < best.fun:
            best = o
    c = c0 + complex(*best.x) * 1e-3
    return c, step


def cmd_info(argv):
    """post hoc, information only: the K sequence of the polynomial fits past the grid, free-intercept rational fits
    with their nearest pole, and the intercept that the shape (r - c)/eps = P_L/Q_M fits best, on the real states and
    on the noise-free planted families (known truth)"""
    statedir = [a for a in argv if not a.startswith("--")][0]
    recs = add_ratios(load_states(statedir))
    tuned = tuned_ratios(A_TARGET)
    par = family_params(recs, tuned)
    sets = [("real", recs)] + [(f"planted {f}", planted_records(recs, par, f)) for f in FAMILIES + INFO_FAMILIES]
    LM = [(L, M) for M in (1, 2, 3, 4) for L in range(2, 7) if 5 <= L + M <= 8]
    print("== the intercept the shape (r - c)/eps = P_L/Q_M fits best (rms over the reference fit's states)")
    for nu in (3, 5, 7):
        for lab, rs in sets[:3]:
            med = grid_fit(rs, nu)["median"]
            for L, M in ((3, 3), (4, 3), (5, 3), (4, 4)):
                c, step = best_intercept(rs, nu, L, M, [tuned[nu], med])
                rb, pb = shape_rms(rs, nu, c, L, M, step)
                rt, _ = shape_rms(rs, nu, tuned[nu], L, M, step)
                rm, _ = shape_rms(rs, nu, med, L, M, step)
                print(f"  r_{nu} {lab:14s} [{L}/{M}]: best c - tuned {fc(c - tuned[nu], 6)} (|.| {abs(c - tuned[nu]):.1e},"
                      f" grid median {abs(med - tuned[nu]):.1e}); rms at best {rb:.2e}, at tuned {rt:.2e}, at the grid "
                      f"median {rm:.2e}; nearest unpaired pole at best eps {fc(pb, 4)} (|.| {abs(pb):.4f})", flush=True)
    for nu in NU_ACC:
        print(f"\n== r_{nu}: tuned {fc(tuned[nu], 8)}")
        for lab, rs in sets:
            ks = k_sequence(rs, nu)
            print(f"  {lab:14s} K sequence (eps, D >= 15), |r(0) - tuned|: "
                  + "  ".join(f"K{k} {abs(v - tuned[nu]):.1e}" for k, v in ks))
        for lab, rs in sets:
            rows = []
            for L, M in LM:
                f = rational_fit(rs, nu, L, M)
                near = min(f["poles"], key=abs) if len(f["poles"]) else complex("nan")
                dz = min(abs(f["zeros"] - near)) if len(f["zeros"]) else float("nan")
                rows.append((L, M, f["r0"], near, dz, f["maxres"]))
            v = np.array([x[2] for x in rows])
            med = cmedian(v)
            print(f"  {lab:14s} rational [L/M], L + M = 5-8 ({len(rows)} fits): median |r(0) - tuned| "
                  f"{abs(med - tuned[nu]):.1e}, spread {max(abs(v - med)):.1e}")
            for L, M, r0, near, dz, mr in rows:
                print(f"      [{L}/{M}] |r(0) - tuned| {abs(r0 - tuned[nu]):.1e}  nearest pole eps {fc(near, 4)} "
                      f"(|.| {abs(near):.4f}, nearest zero {dz:.1e} from it)  max residual {mr:.1e}")
    return 0


if __name__ == "__main__":
    cmds = dict(noise=cmd_noise, planted=lambda a: cmd_planted(a)[0], states=cmd_states, controls=cmd_controls,
                info=cmd_info)
    if len(sys.argv) < 3 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    sys.exit(cmds[sys.argv[1]](sys.argv[2:]))
