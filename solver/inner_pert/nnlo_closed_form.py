"""The closed-form large-D solution: EEG's corrected NNLO (arXiv:2611.07007v1, (S35)-(S51) with the correction of
second_order_eeg.py) with the profile's own 1/n terms (agent-comms#42, notes/SECOND_ORDER.md section 1.2).

    beta~_D(theta) = sqrt(2 (1 + 1/n)) (F + K1/(2n) + K2/(2n^2)),   n = D - 2,   theta = omega tau,
    omega = 2 pi/Delta(n),   Delta(n) = Delta_inf + Delta_1/n + Delta_2/n^2,

goes into nnlo_vs_solution.model_fields (corrected, order 3), which gives f, Omega, Pi, Psi on 0 <= x <= 1. F is the
tuned D = infinity profile (EEG_PROFILE.md 3.3, J 1500), so every closed-form result is conditional on the selection
rule (Conjecture 1). K1 and K2 (Proposition 1, F_n = F + sum_k K_k/(2 n^k)) are fitted from the states' central
profiles F_n = beta~/sqrt(2 (1 + 1/n)), beta~ = Pi(tau, 0)/sqrt(eps), harmonic by harmonic with the fundamental's phase
fixed to F's (F_1 = -i A/2): F_n,nu - F_nu = sum_{k=1..kmax} a_k n^-k, K_k = 2 a_k, by least squares over the states
with D >= dlo except the test states (held out). K3..K_kmax are fitted and dropped. The closed form's time origin is
aligned on each state's fundamental (one gauge number per state).

The measure is R(D) = rms(F_sol - F_cf)/rms(F_sol) over the (tau, x) grid (f relative to rms(1 - f)), as
nnlo_vs_solution.py figs. Its exponent p is the slope of ln R = a + p ln eps, eps = 1/(D - 1), over the test D.
Criteria C1-C6 were posted on agent-comms#42 before the first run (the started comment).

    python3 nnlo_closed_form.py compare FIELDS... [--json=F] [--figs=figs.json] [--mutate=k1x0|k2x-1|k1x<s>|k2x<s>]
                                         [--kmax=6] [--dlo=20] [--test=40.020065,60.001952,100,200] [--spread=DIR,...]
                                         [--kout=K.json]          # F, K1, K2 of every harmonic, and the Delta series
    python3 nnlo_closed_form.py controls FIELDS... [--json=F]      # C3 and C4: both controls must fire
    python3 nnlo_closed_form.py planted FIELDS... [--json=F] [--plant_scale=1] [--mutate=...]

compare exits 1 when C1, C2 or C5 fails; with a mutation it exits 0 only when the control fires (k1x0: p <= 1.2,
otherwise p < 2.6). planted exits 1 when a P criterion fails; with --mutate it adds P2m, which asks the mutated closed
form's p(Omega) to differ from the true-K closed form's by more than 0.1.

--n3lo (agent-comms#53) adds the order past the NNLO (nnlo_vs_solution.model_fields order 4: Psi_3, Pi_4, Omega_4,
f_4 of psi3.closed()) as further rows: the closed form with it, and each state's exact profile with it. compare then
also asks C7 (exact profile with N3LO: p(Psi) >= 2.6) and its control C7c (exact profile with the NNLO, where Psi_3 is
absent: p(Psi) < 2.6). --n3lo_mut=M1,M2 adds the exact profile with N3LO mutations (information, as
nnlo_vs_solution's --mutate). planted --n3lo plants on N3LO with nnlo_vs_solution.PLANT4 and adds P5 (exact profile
with N3LO recovers the plant's relative orders 4, 4, 4, 3 within 0.25) and P6 (exact profile with the NNLO gives
Psi's relative order 2 within 0.25).

    python3 nnlo_closed_form.py compare FIELDS... --figs=figs.json --n3lo [--n3lo_mut=psi3x0.5,Psi3]
    python3 nnlo_closed_form.py planted FIELDS... --n3lo
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nnlo_vs_solution as NV            # noqa: E402

FIELDS = NV.FIELDS
A_TUNED = 2.71918                        # the tuned profile's fundamental amplitude, |F_1| = A/2
DELTA_SERIES = (1.643893746, 10.3753, -49.1685)    # Delta_inf (README), Delta_1, Delta_2 (FIRST_ORDER.md 6, SECOND_ORDER.md)
TEST_D = (40.020065, 60.001952, 100.0, 200.0)
NU_MAX = 31                              # odd harmonics fitted (the M = 32 grid resolves nu <= 31)
A1_M5 = 14.453173450552613               # M = 5 inner run's 1/n coefficient of A (data/first_order_hier_M5_xi0.2.json)
FIT_CHOICES = [(k, d) for k in (5, 6, 7) for d in (15.0, 20.0, 30.0)]


def arg(argv, name, default=None):
    return next((a.split("=", 1)[1] for a in argv if a.startswith(f"--{name}=")), default)


# ------------------------------------------------------------------ profiles, harmonics and the gauge
def nus():
    return np.arange(1, NU_MAX + 1, 2)


def tuned_F():
    """F_nu, nu = 1, 3, ..., NU_MAX, of the tuned D = infinity profile in the gauge F_1 = -i A/2 (zero above 25)"""
    rat = NV.dinf_ratios()
    return np.array([-1j * (A_TUNED / 2) * rat.get(int(v), 0.0) for v in nus()])


def harmonics(b):
    """coefficients c_nu (odd nu <= NU_MAX) of an antiperiodic half-grid function, b = sum 2 Re(c_nu e^{i nu theta}),
    theta_j = pi j/M"""
    M = len(b)
    c = np.fft.rfft(np.r_[b, -b]) / (2 * M)
    return c[nus()]


def synth(c, M):
    """the half-grid function of the coefficients c_nu (odd nu <= NU_MAX)"""
    th = np.pi * np.arange(M) / M
    return sum(2 * np.real(cv * np.exp(1j * v * th)) for v, cv in zip(nus(), c))


def profile(r):
    """the state's central profile: n, eps, aligned F_n coefficients (fundamental -i |c_1|) and the phase phi that
    maps them back (c_nu = chat_nu e^{i nu phi})"""
    eps, bt = NV.beta_tilde(r)
    n = r["D"] - 2.0
    c = harmonics(bt[0] / np.sqrt(2 * (1 + 1 / n)))
    phi = np.angle(c[0]) + np.pi / 2
    chat = c * np.exp(-1j * nus() * phi)
    return dict(D=r["D"], n=n, eps=eps, Delta=r["Delta"], M=len(bt[0]), chat=chat, phi=phi)


# ------------------------------------------------------------------ the fit of K_k
def fit_K(profs, F, kmax, dlo, hold=(), free=False):
    """a[k-1, nu] (k = 1..kmax), F_n,nu - F_nu = sum a_k n^-k over the states with D >= dlo not in hold; free: an
    intercept is fitted too (returned as F_fit). Columns are scaled to unit max for conditioning."""
    use = [p for p in profs if p["D"] >= dlo and not any(abs(p["D"] - h) < 1e-6 for h in hold)]
    t = np.array([1.0 / p["n"] for p in use])
    cols = [t ** k for k in range(1, kmax + 1)]
    if free:
        cols = [np.ones_like(t)] + cols
    A = np.array(cols).T
    sc = np.abs(A).max(axis=0)
    Y = np.array([p["chat"] for p in use]) - (0 if free else F[None, :])
    sol, *_ = np.linalg.lstsq(A / sc, Y, rcond=None)
    sol = sol / sc[:, None]
    res = Y - A @ sol
    out = dict(a=sol[1:] if free else sol, D=[p["D"] for p in use], kmax=kmax, dlo=dlo,
               max_res=float(np.abs(res).max()), n_states=len(use))
    if free:
        out["F_fit"] = sol[0]
    return out


# ------------------------------------------------------------------ the closed form
def cf_coeffs(n, F, a, kscale):
    """aligned coefficients of F_n = F + sum_k s_k a_k n^-k, kscale = {k: s_k} (k >= 1)"""
    c = F.astype(complex).copy()
    for k, s in kscale.items():
        if s:
            c = c + s * a[k - 1] * n ** -k
    return c


def cf_profile(r, prof, F, a, kscale, pref="full", delta="series"):
    """the closed form's beta~ and its tau derivatives at state r: beta~ = pref * F_n (pref sqrt(2 (1 + 1/n)), or sqrt2
    for #5's D = infinity run), back in the state's time origin; omega = 2 pi/Delta(n) (series) or the state's Delta"""
    n = prof["n"]
    c = cf_coeffs(n, F, a, kscale)
    fac = np.sqrt(2 * (1 + 1 / n)) if pref == "full" else np.sqrt(2.0)
    b = fac * synth(c * np.exp(1j * nus() * prof["phi"]), prof["M"])
    D0, D1, D2 = DELTA_SERIES
    Dl = D0 + D1 / n + D2 / n ** 2 if delta == "series" else r["Delta"]
    return NV.tau_derivs(b, Dl)


def cf_fields(r, prof, F, a, kscale, pref="full", delta="series", order=3):
    """the closed-form fields f, Omega, Pi, Psi at state r (corrected NNLO, or N3LO with order 4)"""
    return NV.model_fields(prof["eps"], cf_profile(r, prof, F, a, kscale, pref, delta), r["xs"], "corrected", order)


def profile_err(r, prof, F, a, kscale, pref="full", delta="series"):
    """rms(beta~_cf - beta~_state)/rms(beta~_state) on the tau grid"""
    b = cf_profile(r, prof, F, a, kscale, pref, delta)[0]
    bs = NV.beta_tilde(r)[1][0]
    return float(np.sqrt(np.mean((b - bs) ** 2) / np.mean(bs ** 2)))


def rel_rms(r, m):
    out = {}
    for k in FIELDS:
        ref = (1 - r[k]) if k == "f" else r[k]
        out[k] = float(np.sqrt(np.mean((r[k] - m[k]) ** 2)) / np.sqrt(np.mean(ref ** 2)))
    return out


def variants(mut, n3=False):
    """name -> (kscale, pref, delta[, order]); the closed form is K1K2 (with --mutate its K1 or K2 is scaled); order
    is 3 (corrected NNLO) unless given; --n3lo adds the closed form with N3LO (order 4)"""
    v = {"sqrt2 F (#5, state Delta)": ({}, "sqrt2", "state"),
         "F": ({}, "full", "series"),
         "K1 = 0 (C3)": ({2: 1}, "full", "series"),
         "K1 (C1)": ({1: 1}, "full", "series"),
         "K1K2 (C2)": ({1: 1, 2: 1}, "full", "series"),
         "K2 x -1 (C4)": ({1: 1, 2: -1}, "full", "series"),
         "K1K2, state Delta": ({1: 1, 2: 1}, "full", "state"),
         "K1K2K3": ({1: 1, 2: 1, 3: 1}, "full", "series")}
    if mut:
        v[f"K1K2, mutation {mut}"] = (mutated(mut), "full", "series")
    if n3:
        v["K1, N3LO"] = ({1: 1}, "full", "series", 4)
        v["K1K2, N3LO"] = ({1: 1, 2: 1}, "full", "series", 4)
        v["K1K2K3, N3LO"] = ({1: 1, 2: 1, 3: 1}, "full", "series", 4)
    return v


def mutated(mut):
    if not (mut.startswith("k1x") or mut.startswith("k2x")):
        raise SystemExit(f"unknown mutation {mut}")
    ks = {1: 1.0, 2: 1.0}
    ks[int(mut[1])] = float(mut[3:])
    return ks


def exponent(D, R):
    eps = [1.0 / (d - 1.0) for d in D]
    return NV.fit_exponent(eps, R)


def evaluate(recs, profs, F, a, vs, test, exact=True, n3=False, n3mut=()):
    """R per variant and field at the test states, the profile's own error (field 'beta'), and the exponents
    (p_pure is the main fit); n3: also each state's exact profile with N3LO, and with each N3LO mutation in n3mut"""
    idx = [i for i, r in enumerate(recs) if any(abs(r["D"] - t) < 1e-6 for t in test)]
    if len(idx) != len(test):
        raise SystemExit(f"test states missing: have {[recs[i]['D'] for i in idx]}, want {list(test)}")
    tab = {}
    for name, spec in vs.items():
        ks, pref, dl = spec[:3]
        order = spec[3] if len(spec) > 3 else 3
        tab[name] = [dict(rel_rms(recs[i], cf_fields(recs[i], profs[i], F, a, ks, pref, dl, order)),
                          beta=profile_err(recs[i], profs[i], F, a, ks, pref, dl)) for i in idx]
    if exact:
        tab["exact profile (floor)"] = [dict(rel_rms(recs[i], NV.model_fields(*NV.beta_tilde(recs[i]), recs[i]["xs"],
                                                                               "corrected", 3)), beta=0.0) for i in idx]
        if n3:
            for v in ("corrected",) + tuple(n3mut):
                name = "exact profile, N3LO" + ("" if v == "corrected" else f", mutation {v}")
                tab[name] = [dict(rel_rms(recs[i], NV.model_fields(*NV.beta_tilde(recs[i]), recs[i]["xs"], v, 4)),
                                  beta=0.0) for i in idx]
    Ds = [recs[i]["D"] for i in idx]
    fits = {name: {k: exponent(Ds, [row[k] for row in rows]) for k in FIELDS + ("beta",) if rows[0][k] > 0}
            for name, rows in tab.items()}
    return Ds, tab, fits


def print_table(Ds, tab, fits, label):
    print(f"\n== {label}: relative rms residual R = rms(F_sol - F_cf)/rms(F_sol) (f: rms(1 - f); beta: the "
          "profile's own error), and its exponent")
    print("   " + " " * 30 + "".join(f"{'D ' + format(d, '.4g'):>11s}" for d in Ds) + "    p_pure  p(+b eps)  local (D pairs)")
    for name, rows in tab.items():
        for k in [q for q in FIELDS + ("beta",) if q in fits[name]]:
            ft = fits[name][k]
            print(f"   {name if k == 'f' else '':26s} {k:3s}" + "".join(f"{row[k]:11.3e}" for row in rows)
                  + f"   {ft['p_pure']:7.3f}  {ft['p']:8.3f}  " + " ".join(f"{v:.2f}" for v in ft["local"]))


def k_summary(a, label, F):
    """K1, K2 for nu = 1, 3, 5 (Proposition 1's convention K_k = 2 a_k) and the fundamental's A_1 = 2i a_1,1"""
    out = {"label": label}
    for k in (1, 2):
        for j, v in enumerate((1, 3, 5)):
            out[f"K{k}_{v}"] = complex(2 * a[k - 1][j])
    out["A1"] = float(np.real(2j * a[0][0]))
    out["A2"] = float(np.real(2j * a[1][0]))
    return out


def fmt_c(z):
    return f"{z.real:+.5e}{z.imag:+.5e}i"


# ------------------------------------------------------------------ compare
def load(files):
    recs = sorted([NV.load_fields(f) for f in files], key=lambda r: r["D"])
    return recs, [profile(r) for r in recs]


def run(recs, profs, F, argv, label, a_true=None):
    """the fit, its spread over choices, and the evaluation; a_true: also the closed form with the true a"""
    mut = arg(argv, "mutate")
    kmax, dlo = int(arg(argv, "kmax", 6)), float(arg(argv, "dlo", 20))
    test = [float(v) for v in arg(argv, "test", ",".join(str(t) for t in TEST_D)).split(",")]
    fit = fit_K(profs, F, kmax, dlo, hold=test)
    n3 = "--n3lo" in argv
    n3mut = tuple(m for m in arg(argv, "n3lo_mut", "").split(",") if m)
    vs = variants(mut, n3)
    Ds, tab, fits = evaluate(recs, profs, F, fit["a"], vs, test, n3=n3, n3mut=n3mut)
    # information: F fitted as the intercept too, so that the closed form uses no D = infinity input
    ff = fit_K(profs, F, kmax, dlo, hold=test, free=True)
    _, tf, fif = evaluate(recs, profs, ff["F_fit"], ff["a"], {"K1K2, F fitted (no D = inf input)":
                                                              ({1: 1, 2: 1}, "full", "series")}, test, exact=False)
    tab.update(tf)
    fits.update(fif)
    print_table(Ds, tab, fits, f"{label}; K fit k_max {kmax}, D >= {dlo:g}, test states held out "
                               f"({fit['n_states']} states, max residual {fit['max_res']:.1e})")
    out = dict(label=label, mutation=mut, kmax=kmax, dlo=dlo, test=Ds, fit_states=fit["D"], fit_max_res=fit["max_res"],
               table=tab, fits=fits, K=k_summary(fit["a"], "main", F), n3lo=n3, n3lo_mut=list(n3mut))
    # the spread over fit choices: K1, K2 (nu = 1, 3, 5) and p(Omega) of the main variants
    sp = []
    for (k, d) in FIT_CHOICES:
        f2 = fit_K(profs, F, k, d, hold=test)
        _, t2, fi2 = evaluate(recs, profs, F, f2["a"], {n: vs[n] for n in ("K1 (C1)", "K1K2 (C2)", "K1 = 0 (C3)")},
                              test, exact=False)
        sp.append(dict(kmax=k, dlo=d, K=k_summary(f2["a"], f"{k},{d:g}", F), max_res=f2["max_res"],
                       p={n: fi2[n]["Om"]["p_pure"] for n in fi2}, R={n: [row["Om"] for row in t2[n]] for n in t2}))
    out["spread"] = sp
    print("\n== spread over fit choices (k_max, D_lo): K's fundamental A_k = 2i a_k,1 (F_n = F + sum A_k n^-k / ...),"
          " K1, K2 at nu = 3, and p(Omega)")
    for s in sp:
        K = s["K"]
        print(f"   {s['kmax']} {s['dlo']:4.0f}  A1 {K['A1']:.6f}  A2 {K['A2']:9.4f}  K1_3 {fmt_c(K['K1_3'])}  "
              f"K2_3 {fmt_c(K['K2_3'])}  p: " + "  ".join(f"{n.split(' (')[0]} {v:.3f}" for n, v in s["p"].items())
              + f"  (fit res {s['max_res']:.1e})")
    if a_true is not None:
        _, tt, ft = evaluate(recs, profs, F, a_true, vs, test, exact=False, n3=n3)
        print_table(Ds, tt, ft, f"{label}: the same with the planted (true) K")
        out["table_trueK"], out["fits_trueK"] = tt, ft
        out["fit_a"] = fit["a"]
    return out, fit


def crosschecks(profs, F, test):
    """the free-intercept fit's F against the tuned profile, and the fitted A1 against the M = 5 inner run"""
    out = []
    print("\n== cross-checks: free-intercept fit's F against the tuned D = inf profile; A1 against M = 5 (14.453)")
    for (k, d) in FIT_CHOICES:
        ff = fit_K(profs, F, k, d, hold=test, free=True)
        dF = ff["F_fit"] - F
        rel = [float(abs(dF[j]) / abs(F[j])) for j in range(3)]
        l2 = float(np.sqrt(np.sum(np.abs(dF) ** 2) / np.sum(np.abs(F) ** 2)))
        A1 = float(np.real(2j * ff["a"][0][0]))
        out.append(dict(kmax=k, dlo=d, rel_nu135=rel, l2=l2, A_fit=float(np.real(2j * ff["F_fit"][0])), A1_free=A1))
        print(f"   {k} {d:4.0f}  A_fit {out[-1]['A_fit']:.7f} (tuned {A_TUNED})  |dF_nu|/|F_nu| nu=1,3,5: "
              + " ".join(f"{v:.1e}" for v in rel) + f"  L2 {l2:.1e}  A1 (free) {A1:.4f}")
    return out


def check_c5(tab, Ds, figs_path):
    """C5: sqrt2 F reproduces #5's dinf rows of figs.json for Omega and Pi within 1%"""
    if not figs_path or not os.path.exists(figs_path):
        print(f"\nC5: figs.json not given or missing ({figs_path}): FAIL")
        return False, []
    rows = json.load(open(figs_path))["figs"]["dinf"]
    out, ok = [], True
    for i, d in enumerate(Ds):
        ref = next((r for r in rows if abs(r["D"] - d) < 1e-4), None)
        if ref is None:
            ok = False
            continue
        for k in ("Om", "Pi"):
            mine, theirs = tab["sqrt2 F (#5, state Delta)"][i][k], ref[k]["corrected"]
            dev = abs(mine / theirs - 1)
            ok &= dev <= 0.01
            out.append(dict(D=d, field=k, mine=mine, figs=theirs, rel_dev=dev))
    print("\n== C5: sqrt2 F against I126-mac/figs.json dinf (corrected): " +
          "  ".join(f"D {o['D']:.4g} {o['field']} {o['mine']:.4f}/{o['figs']:.4f}" for o in out)
          + f"  max dev {max(o['rel_dev'] for o in out):.1e}  {'PASS' if ok else 'FAIL'}")
    return bool(ok), out


def chain_spread(recs, profs, F, a, dirs, test, n3=False):
    """the closed form's R(Omega) against other chains' states at a test D (held harmonics differ there); n3: also
    R of every field with each state's exact profile and N3LO"""
    out = []
    for d in dirs:
        for f in sorted(os.listdir(d)):
            o = NV.load_fields(os.path.join(d, f))
            if not any(abs(o["D"] - t) < 1e-6 for t in test):
                continue
            po = profile(o)
            i = [j for j, r in enumerate(recs) if abs(r["D"] - o["D"]) < 1e-6][0]
            ks = {1: 1, 2: 1}
            R_o = rel_rms(o, cf_fields(o, po, F, a, ks))["Om"]
            R_m = rel_rms(recs[i], cf_fields(recs[i], profs[i], F, a, ks))["Om"]
            row = dict(D=o["D"], dir=d, R_other=R_o, R_main=R_m, Delta_diff=o["Delta"] - recs[i]["Delta"])
            print(f"   chain spread at D {o['D']:.0f}: R(Omega) {R_o:.4e} ({os.path.basename(d)}) against {R_m:.4e}")
            if n3:
                e_o = rel_rms(o, NV.model_fields(*NV.beta_tilde(o), o["xs"], "corrected", 4))
                e_m = rel_rms(recs[i], NV.model_fields(*NV.beta_tilde(recs[i]), recs[i]["xs"], "corrected", 4))
                row["exact_n3lo"] = {k: dict(other=e_o[k], main=e_m[k], rel=abs(e_o[k] / e_m[k] - 1)) for k in FIELDS}
                print("      exact profile with N3LO, |R_other/R_main - 1|: "
                      + " ".join(f"{k} {row['exact_n3lo'][k]['rel']:.1e}" for k in FIELDS))
            out.append(row)
    return out


def write_K(path, fit, F, out, profs):
    """the closed form's data: F, K1, K2 (Proposition 1's convention, F_n = F + K1/(2n) + K2/(2n^2)) as complex
    coefficients of e^{i nu theta} (b = sum 2 Re(c_nu e^{i nu theta})), with the min and max of K1, K2 over the fit
    choices, and the Delta series"""
    spread = [fit_K(profs, F, k, d, hold=out["test"])["a"] for (k, d) in FIT_CHOICES]
    rows = []
    for j, v in enumerate(nus()):
        row = dict(nu=int(v), F=[F[j].real, F[j].imag])
        for k in (1, 2):
            K = 2 * fit["a"][k - 1][j]
            Ks = [2 * s[k - 1][j] for s in spread]
            row[f"K{k}"] = [K.real, K.imag]
            row[f"K{k}_spread_abs"] = float(max(abs(q - K) for q in Ks))
        rows.append(row)
    D0, D1, D2 = DELTA_SERIES
    js = dict(what="closed-form large-D profile, agent-comms#42; beta~ = sqrt(2(1 + 1/n)) (F + K1/(2n) + K2/(2n^2)), "
                   "n = D - 2, theta = 2 pi tau/Delta(n); gauge F_1 = -i A/2 (fundamental ~ sin theta)",
              conditional="F is the tuned D = infinity profile (A = 2.71918): conditional on Conjecture 1",
              fit=dict(kmax=fit["kmax"], dlo=fit["dlo"], held_out=out["test"], states=fit["D"]),
              K_spread="max |K - K_choice| over (kmax, dlo) in " + str(FIT_CHOICES),
              Delta_series=dict(Delta_inf=D0, Delta_1=D1, Delta_2=D2), harmonics=rows)
    with open(path, "w") as fh:
        json.dump(js, fh, indent=1)
    print(f"wrote {path}")


def verdict_lines(fits, mut):
    p = lambda name: fits[name]["Om"]["p_pure"]
    v = [dict(c="C1 K1: p >= 1.8", p=p("K1 (C1)"), ok=p("K1 (C1)") >= 1.8),
         dict(c="C2 K1K2: p >= 2.6", p=p("K1K2 (C2)"), ok=p("K1K2 (C2)") >= 2.6),
         dict(c="C3 K1 = 0: p <= 1.2", p=p("K1 = 0 (C3)"), ok=p("K1 = 0 (C3)") <= 1.2),
         dict(c="C3 F alone: p <= 1.2", p=p("F"), ok=p("F") <= 1.2),
         dict(c="C4 K2 x -1: p < 2.6", p=p("K2 x -1 (C4)"), ok=p("K2 x -1 (C4)") < 2.6)]
    if mut:
        name = f"K1K2, mutation {mut}"
        thr = (lambda q: q <= 1.2) if mut == "k1x0" else (lambda q: q < 2.6)
        v.append(dict(c=f"mutation {mut} fires: p {'<= 1.2' if mut == 'k1x0' else '< 2.6'}", p=p(name),
                      ok=bool(thr(p(name)))))
    for x in v:
        x["ok"] = bool(x["ok"])
    return v


def verdict_n3(fits):
    """C7 (agent-comms#53): the exact profile with N3LO gives p(Psi) >= 2.6; C7c, its control: with the NNLO (Psi_3
    absent) p(Psi) < 2.6"""
    p = lambda name: fits[name]["Psi"]["p_pure"]
    return [dict(c="C7 exact profile, N3LO: p(Psi) >= 2.6", p=p("exact profile, N3LO"),
                 ok=bool(p("exact profile, N3LO") >= 2.6)),
            dict(c="C7c exact profile, NNLO: p(Psi) < 2.6", p=p("exact profile (floor)"),
                 ok=bool(p("exact profile (floor)") < 2.6))]


def cmd_compare(argv, controls=False):
    files = [a for a in argv if not a.startswith("--")]
    recs, profs = load(files)
    F = tuned_F()
    mut = arg(argv, "mutate")
    out, fit = run(recs, profs, F, argv, "solutions" + (f", mutation {mut}" if mut else ""))
    test = out["test"]
    out["crosschecks"] = crosschecks(profs, F, test)
    figs_path = arg(argv, "figs")
    ok5, c5 = check_c5(out["table"], test, figs_path)
    out["C5"] = dict(ok=ok5, rows=c5)
    dirs = [d for d in arg(argv, "spread", "").split(",") if d]
    n3 = "--n3lo" in argv
    if dirs:
        print("\n== the closed form against other chains at the test D")
        out["chain_spread"] = chain_spread(recs, profs, F, fit["a"], dirs, test, n3=n3)
    kout = arg(argv, "kout")
    if kout:
        write_K(kout, fit, F, out, profs)
    ver = verdict_lines(out["fits"], mut)
    ver.append(dict(c="C5 pipeline: sqrt2 F reproduces figs.json within 1%", p=float("nan"), ok=ok5))
    if n3:
        ver += verdict_n3(out["fits"])
    out["verdicts"] = ver
    print("\n== verdicts (p = slope of ln R(Omega) against ln eps over the test D)")
    for v in ver:
        print(f"   {v['c']:52s} p = {v['p']:6.3f}  {'PASS' if v['ok'] else 'FAIL'}")
    js = arg(argv, "json")
    if js:
        with open(js, "w") as fh:
            json.dump(out, fh, indent=1, default=lambda z: [z.real, z.imag] if isinstance(z, complex) else str(z))
        print(f"\nwrote {js}")
    by = {v["c"][:2]: v["ok"] for v in ver}
    if controls:
        fired = by["C3"] and all(v["ok"] for v in ver if v["c"].startswith("C3")) and by["C4"]
        print(f"\ncontrols C3 (K1 = 0, F alone) and C4 (K2 x -1): {'both fire (exit 0)' if fired else 'FAILED (exit 1)'}")
        return 0 if fired else 1
    if mut:
        fired = next(v["ok"] for v in ver if v["c"].startswith("mutation"))
        print(f"\nmutation {mut}: {'the control fires (exit 0)' if fired else 'CONTROL FAILED (exit 1)'}")
        return 0 if fired else 1
    ok = by["C1"] and by["C2"] and ok5
    if n3:
        ok7 = all(v["ok"] for v in ver if v["c"].startswith("C7"))
        ok = ok and ok7
        print(f"\n{'C1, C2, C5, C7, C7c PASS' if ok else 'FAILED'}")
        return 0 if ok else 1
    print(f"\n{'C1, C2, C5 PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


# ------------------------------------------------------------------ planted truth
RHO0, RHO_POW, SHRINK, PSI = 5.3, 1.75, 0.24, 0.02


def plant_a(F, kmax=12):
    """the planted profile's 1/n coefficients: F_n,nu = F_nu (1 + (rho/n) e^{i psi}/(1 + SHRINK rho/n)) has
    a_k = F_nu e^{i psi} rho (-SHRINK rho)^(k-1)"""
    rho = RHO0 * nus().astype(float) ** RHO_POW
    psi = PSI * (nus() - 1)
    return np.array([F * np.exp(1j * psi) * rho * (-SHRINK * rho) ** (k - 1) for k in range(1, kmax + 1)])


def plant_profile(F, n):
    rho = RHO0 * nus().astype(float) ** RHO_POW
    psi = PSI * (nus() - 1)
    return F * (1 + (rho / n) * np.exp(1j * psi) / (1 + SHRINK * rho / n))


def planted_records(recs, F, scale, order=3):
    """each state's fields replaced by the corrected NNLO (order 3; N3LO with order 4) of sqrt(2 (1 + 1/n)) F_n^pl,
    shifted in time by a known phase, plus the known remainder of nnlo_vs_solution.planted_records (PLANT, or PLANT4
    with order 4; the state's Delta in the tau derivatives)"""
    out = []
    i0 = int(np.argmin(np.abs(recs[0]["xs"])))
    for r in recs:
        n = r["D"] - 2.0
        eps = 1.0 / (r["D"] - 1.0)
        phi = 2 * np.pi * ((0.37 * r["D"] + 0.11) % 1.0)
        M = r["Pi"].shape[1]
        b = np.sqrt(2 * (1 + 1 / n)) * synth(plant_profile(F, n) * np.exp(1j * nus() * phi), M)
        q = dict(r)
        q["Pi"] = r["Pi"].copy()
        q["Pi"][i0] = np.sqrt(eps) * b
        out.append(NV.planted_records([q], NV.PLANT4 if order == 4 else NV.PLANT, scale, order=order)[0])
    return out


def cmd_planted(argv):
    files = [a for a in argv if not a.startswith("--")]
    scale = float(arg(argv, "plant_scale", 1.0))
    mut = arg(argv, "mutate")
    F = tuned_F()
    n3 = "--n3lo" in argv
    recs = planted_records(sorted([NV.load_fields(f) for f in files], key=lambda r: r["D"]), F, scale,
                           order=4 if n3 else 3)
    profs = [profile(r) for r in recs]
    a_true = plant_a(F)
    out, fit = run(recs, profs, F, argv, f"planted (plant_scale {scale})", a_true=a_true)
    ver = []
    # P1: K1 and K2 recovered for nu = 1, 3
    for k, tol in ((1, 1e-3), (2, 3e-2)):
        for j, v in enumerate((1, 3)):
            dev = float(abs(fit["a"][k - 1][j] - a_true[k - 1][j]) / abs(a_true[k - 1][j]))
            ver.append(dict(c=f"P1 K{k} nu {v} within {tol:g}", p=dev, ok=dev <= tol))
    # P2: fitted-K exponents agree with true-K exponents within 0.1 (the mutated variant must not, if mutated)
    names = ["K1 = 0 (C3)", "K1 (C1)", "K1K2 (C2)"]
    for name in names:
        d = abs(out["fits"][name]["Om"]["p_pure"] - out["fits_trueK"][name]["Om"]["p_pure"])
        ver.append(dict(c=f"P2 {name.split(' (')[0]}: |p_fit - p_true| <= 0.1", p=float(d), ok=d <= 0.1))
    # P3: the thresholds on the planted data
    p = lambda name: out["fits"][name]["Om"]["p_pure"]
    ver += [dict(c="P3 K1 = 0: p <= 1.2", p=p("K1 = 0 (C3)"), ok=p("K1 = 0 (C3)") <= 1.2),
            dict(c="P3 K1: 1.8 <= p <= 2.4", p=p("K1 (C1)"), ok=1.8 <= p("K1 (C1)") <= 2.4),
            dict(c="P3 K1K2: p >= 2.6", p=p("K1K2 (C2)"), ok=p("K1K2 (C2)") >= 2.6),
            dict(c="P3 K2 x -1: p < 2.6", p=p("K2 x -1 (C4)"), ok=p("K2 x -1 (C4)") < 2.6)]
    if mut:
        name = f"K1K2, mutation {mut}"
        d = abs(out["fits"][name]["Om"]["p_pure"] - out["fits_trueK"]["K1K2 (C2)"]["Om"]["p_pure"])
        ver.append(dict(c=f"P2m mutation {mut} moves p by more than 0.1", p=float(d), ok=d > 0.1))
    if n3:
        # P5: the exact profile with N3LO sees the plant's relative orders (plant minus each field's leading order:
        # f and Omega eps^1, Pi eps^(1/2), Psi eps^(3/2)); P6: with the NNLO it sees Psi_3 missing, relative eps^2
        rel = dict(f=NV.PLANT4["f"] - 1, Om=NV.PLANT4["Om"] - 1, Pi=NV.PLANT4["Pi"] - 0.5, Psi=NV.PLANT4["Psi"] - 1.5)
        for k in FIELDS:
            q = out["fits"]["exact profile, N3LO"][k]["p_pure"]
            ver.append(dict(c=f"P5 exact profile, N3LO, {k}: |p - {rel[k]:g}| <= 0.25", p=q, ok=abs(q - rel[k]) <= 0.25))
        q = out["fits"]["exact profile (floor)"]["Psi"]["p_pure"]
        ver.append(dict(c="P6 exact profile, NNLO, Psi: |p - 2| <= 0.25", p=q, ok=abs(q - 2) <= 0.25))
    for v in ver:
        v["ok"] = bool(v["ok"])
    print("\n== planted: K's recovered (fundamental and nu = 3), true against fitted")
    for k in (1, 2):
        for j, v in enumerate((1, 3, 5)):
            print(f"   a_{k},{v}: true {fmt_c(a_true[k - 1][j])}  fitted {fmt_c(fit['a'][k - 1][j])}")
    print("\n== planted verdicts")
    for v in ver:
        print(f"   {v['c']:48s} {v['p']:9.3e}  {'PASS' if v['ok'] else 'FAIL'}")
    out["verdicts"] = ver
    out["a_true_nu135"] = a_true[:3, :3]
    out["a_fit_nu135"] = fit["a"][:3, :3]
    js = arg(argv, "json")
    if js:
        with open(js, "w") as fh:
            json.dump(out, fh, indent=1, default=lambda z: (z.tolist() if isinstance(z, np.ndarray) else
                                                            [z.real, z.imag] if isinstance(z, complex) else str(z)))
        print(f"\nwrote {js}")
    ok = all(v["ok"] for v in ver)
    print(f"\n{'ALL PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("compare", "controls", "planted"):
        print(__doc__)
        sys.exit(2)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "planted":
        sys.exit(cmd_planted(rest))
    if cmd == "controls":
        sys.exit(cmd_compare(rest, controls=True))
    sys.exit(cmd_compare(rest))
