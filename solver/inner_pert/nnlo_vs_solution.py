"""EEG's large-D NNLO (arXiv:2611.07007v1, supplement (S35)-(S51)) against the double-double and quad-double solutions
(issue #126, notes/SECOND_ORDER.md section 1.1).

The fields of a state (hp/src/model.rs: p, m odd, a, c even, on the half tau grid of M points), with n = D - 2,
eps = 1/(D - 1) and the map of SECOND_ORDER.md section 1 (solver/supercrit/eqmap.py; b = a^2 in model.rs),
    f = c,   Omega = (n - 1)(a^2 - 1)/x^2,   Pi = (p - m)/(sqrt2 x),   Psi = (p + m)/(sqrt2 x^2),
on x in [0, 1]: from the centre series in double-double (x <= nodes[0]; the divisions by x done on its coefficients),
Taylor integration in double-double from the nearest node below (nodes[0] < x < nodes[-1]), and the horizon series in
u = 1 - x in quad-double (x >= nodes[-1]; horizon_q). beta~ = Pi(tau, 0)/sqrt(eps) is the state's exact central
profile (S34); its tau derivatives are spectral (d/dtau = omega d/dtheta, omega = 2 pi/Delta). EEG's coefficients are
those of second_order_eeg.py (printed, and with the correction of its check D).

Truncations: LO, NLO ((S40)-(S43)), NNLO printed ((S44)-(S51)), NNLO corrected, and the control NNLO corrected with
Pi_3 scaled by s (--mutate=pi3x<s>, in Pi and in Omega_3 = 2 beta~ Pi_3 + ...; f_3 unchanged). The maximal residual
max |F_sol - F_model| over the (tau, x) grid is fitted as ln R = a + p ln eps + b eps (least squares, D >= --dmin).

    python3 nnlo_vs_solution.py extract OUT_DIR STATE.npz ...       # fields of each state -> OUT_DIR/fields_D*.npz
    python3 nnlo_vs_solution.py compare FIELDS... [--json=F] [--spread=DIR,...] [--mutate=pi3x1.01] [--dmin=40]
                                         [--dmax=]
    python3 nnlo_vs_solution.py planted FIELDS... [--json=F] [--plant_pi=3.5] [--plant_scale=1] [--mutate=pi3x1.01]
    python3 nnlo_vs_solution.py figs FIELDS... [--json=F]             # Figs. 4-5 measures, exact / Eq. (16) / D = inf

extract needs the compiled kernel with Solver.residual_q (hpkernel, hp/build.sh). compare and planted exit 1 when an
acceptance criterion (raw folder I126-mac/README.md, written before the runs) fails; compare with a mutation exits 0
when the control fires. planted replaces each state's fields by the corrected NNLO of its own beta~ plus known terms,
scale (1 + 4 eps + 20 eps^2) eps^p g(tau, x) with p = 4 (f, Omega), 3.5 (Pi, Psi), so the same code must return those
exponents; --plant_pi=3 plants Pi's term at eps^3 (a second answer the fit must find).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
FIELDS = ("f", "Om", "Pi", "Psi")
XS = np.round(np.linspace(0.0, 1.0, 101), 12)
R2 = np.sqrt(2.0)


# ------------------------------------------------------------------ fields of a state
def _make(meta, nodes, kc=None):
    sys.path.insert(0, os.path.join(REPO, "hp"))
    from hpkernel.driver import make_solver
    return make_solver(meta["D"], meta["M"], nodes=nodes, kt=meta.get("kt", 36), kc=kc or meta.get("kc", 420),
                       ks=meta.get("ks", 100))


def _fields_x(parts, x, n):
    """f, Omega, Pi, Psi at x > 0 from p, m, a, c given in several parts (parts[i][q] = part i of field q);
    a - 1 is formed part by part (the first part minus 1 is exact), so its digits survive"""
    P = [np.asarray(v) for v in parts]
    am1 = (P[0][2] - 1.0) + sum(v[2] for v in P[1:])
    pmm = sum(v[0] - v[1] for v in P)
    ppm = sum(v[0] + v[1] for v in P)
    return dict(f=sum(v[3] for v in P), Om=(n - 1) * am1 * (am1 + 2.0) / x ** 2, Pi=pmm / (R2 * x),
                Psi=ppm / (R2 * x ** 2))


X_ONE = 1.0 - 2.0 ** -53          # x = 1 is evaluated here (u = 2^-53): the horizon series' scale needs u > 0


def horizon_q(z, meta, x):
    """p, m, a, c at x from the horizon series of the state's horizon data (m_h, a_h, Delta), in quad-double:
    a two-node solver whose last node is x, with both node states zero, so its horizon block (horizon series at
    u = 1 - x minus node 1) is the series itself (Solver.residual_q, every block in quad-double). The series in double-
    double (Solver.series) loses all digits near the horizon at D ~ 200 (the horizon wall); the quad-double one keeps
    the stored states' horizon residual (5e-26 at D 200). Returns (4 parts, 4 fields, M)."""
    M, K = int(meta["M"]), len(meta["nodes"])
    x0 = x - 0.01
    s = _make(meta, [x0, x], kc=8)           # its centre block is not used
    s.set_segs([[x0, x]])
    zz = []
    for part in z:
        v = np.zeros(12 * M + 1)
        v[:2 * M] = part[:2 * M]
        v[10 * M:12 * M] = part[2 * M + 4 * M * K:2 * M + 4 * M * K + 2 * M]
        v[-1] = part[-1]
        zz.append(v)
    F = np.array([np.asarray(f) for f in s.residual_q(*zz)])
    return F[:, 8 * M:12 * M].reshape(4, 4, M)


def state_fields(path):
    """dict: D, Delta, M, xs, fields (each (X, M)), and the continuity of the three representations at the nodes.
    Centre series and Taylor steps in double-double (Solver.series, Solver.integrate), horizon series in quad-double
    (horizon_q); a quad-double state's lower two parts enter the horizon series only."""
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    z = [d[k] if k in d.files else np.zeros_like(d["zh"]) for k in ("zh", "zl", "z2", "z3")]
    zh, zl = z[0], z[1]
    D, M, nodes = float(meta["D"]), int(meta["M"]), list(meta["nodes"])
    n = D - 2.0
    K = len(nodes)
    s = _make(meta, nodes)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    sig = s.series_scale(0)
    ch, cl = (np.asarray(v).reshape(-1, 4, M) for v in s.series(zh, zl, 0, True))
    C = ch + cl
    # centre data: p, m vanish at x = 0, a = 1 + O(x^2), p + m = O(x^2); checked, not assumed
    lowc = max(np.abs(C[0, 0]).max(), np.abs(C[0, 1]).max(), np.abs(C[0, 2] - 1).max(), np.abs(C[1, 2]).max(),
               np.abs(C[1, 0] + C[1, 1]).max())
    out = {k: np.zeros((len(XS), M)) for k in FIELDS}

    def node(b):
        o = 2 * M + 4 * M * b
        return [v[o:o + 4 * M].reshape(4, M) for v in z]

    def centre(x):
        t = x / sig
        w = t ** np.arange(C.shape[0])
        P, Mm, A, Cc = (C[:, q, :] for q in range(4))
        pi = np.einsum("k,km->m", w[:-1], (P - Mm)[1:]) / (R2 * sig)
        psi = np.einsum("k,km->m", w[:-2], (P + Mm)[2:]) / (R2 * sig ** 2)
        am1 = np.einsum("k,km->m", w[1:], A[1:])
        am1_x2 = np.einsum("k,km->m", w[:-2], A[2:]) / sig ** 2
        f = np.einsum("k,km->m", w, Cc)
        return dict(f=f, Om=(n - 1) * am1_x2 * (am1 + 2.0), Pi=pi, Psi=psi)

    def horizon(x):
        return _fields_x(horizon_q(z, meta, X_ONE if x == 1.0 else x), x, n)

    def taylor(x):
        b = max(i for i in range(K) if nodes[i] <= x)
        if x == nodes[b]:
            return _fields_x(node(b), x, n)
        o = 2 * M + 4 * M * b
        sh, sl = zh[o:o + 4 * M].copy(), zl[o:o + 4 * M].copy()
        pts = list(np.linspace(nodes[b], x, max(2, int(np.ceil((x - nodes[b]) / 0.02)) + 1)))
        th, tl = (np.asarray(v).reshape(4, M) for v in s.integrate(zh, zl, sh, sl, pts))
        return _fields_x([th, tl], x, n)

    for i, x in enumerate(XS):
        r = centre(x) if x <= nodes[0] else (horizon(x) if x >= nodes[-1] else taylor(x))
        for k in FIELDS:
            out[k][i] = r[k]
    # continuity at the first and last node (series against the node state), each field's max difference; at the
    # last node also the horizon block itself, part by part (the stored state's own horizon residual)
    cont = {}
    for tag, xn, fun in (("centre", nodes[0], centre), ("horizon", nodes[-1], horizon)):
        a, b = fun(xn), taylor(xn)
        cont[tag] = {k: float(np.abs(a[k] - b[k]).max()) for k in FIELDS}
    H = horizon_q(z, meta, nodes[-1])
    nd = node(K - 1)
    cont["horizon_block"] = float(np.abs(sum(H[i] - nd[i] for i in range(4))).max())
    Delta = float(zh[-1] + zl[-1])
    return dict(D=D, Delta=Delta, M=M, xs=XS, fields=out, cont=cont, low_coeffs=float(lowc), meta=meta,
                source=os.path.abspath(path))


def cmd_extract(argv):
    out_dir = argv[0]
    os.makedirs(out_dir, exist_ok=True)
    for p in argv[1:]:
        t0 = time.time()
        r = state_fields(p)
        name = os.path.join(out_dir, f"fields_D{r['D']:.4f}.npz")
        np.savez(name, D=r["D"], Delta=r["Delta"], M=r["M"], xs=r["xs"], **r["fields"],
                 info=json.dumps(dict(cont=r["cont"], low_coeffs=r["low_coeffs"], meta=r["meta"], source=r["source"])))
        print(f"{os.path.basename(p)}: D {r['D']:.4f} Delta {r['Delta']:.12f}  continuity centre "
              f"{max(r['cont']['centre'].values()):.1e} horizon {max(r['cont']['horizon'].values()):.1e} "
              f"(horizon block {r['cont']['horizon_block']:.1e})  "
              f"low coefficients {r['low_coeffs']:.1e}  -> {name} [{time.time() - t0:.1f} s]", flush=True)


# ------------------------------------------------------------------ EEG's expansion, numerically
def load_fields(path):
    z = np.load(path)
    r = dict(D=float(z["D"]), Delta=float(z["Delta"]), M=int(z["M"]), xs=z["xs"], path=path,
             info=json.loads(str(z["info"])))
    for k in FIELDS:
        r[k] = z[k]
    return r


def tau_derivs(b, Delta, nder=4):
    """b on the half grid of an antiperiodic field (b(theta + pi) = -b(theta)), its tau derivatives 0..nder"""
    M = len(b)
    full = np.r_[b, -b]
    c = np.fft.rfft(full)
    k = np.arange(len(c))
    om = 2 * np.pi / Delta
    out = []
    for r in range(nder + 1):
        out.append(np.fft.irfft(c * (1j * k * om) ** r, n=2 * M)[:M])
    return np.array(out)


_COEF = {}


def coeffs(variant):
    """numpy functions (x, b0..b4) -> f2, f3, Om2, Om3, Pi2, Pi3, Psi2 for variant printed / corrected / pi3x1.01"""
    if variant in _COEF:
        return _COEF[variant]
    import sympy as sp
    sys.path.insert(0, HERE)
    import second_order_eeg as E
    E.MUT = None                       # this script's flags are not that script's mutations
    b = sp.symbols("b0:5")
    co = E.eeg_coeffs(list(b), corrected=(variant != "printed"))
    if variant.startswith("pi3x"):                 # the control: Pi_3 scaled by s, in Pi and in Omega_3
        s = sp.Rational(variant[4:])
        co["Om3"] = sp.expand(co["Om3"] + 2 * b[0] * (s - 1) * co["Pi3"])
        co["Pi3"] = sp.expand(co["Pi3"] * s)
    fn = {k: sp.lambdify((E.x,) + tuple(b), co[k], "numpy") for k in co}
    _COEF[variant] = fn
    return fn


def model_fields(eps, bt, xs, variant, order):
    """EEG's fields on (xs, tau) from beta~ and its derivatives bt (5, T); order 1 (LO), 2 (NLO), 3 (NNLO)"""
    X = xs[:, None]
    B = [v[None, :] for v in bt]
    se = np.sqrt(eps)
    f = 1 - eps * (1 - X ** 2) / 2 * B[0] ** 2
    Om = eps * B[0] ** 2 + 0 * X
    Pi = se * (B[0] + 0 * X)
    Psi = se * eps * (B[1] + B[0]) + 0 * X
    if order >= 2:
        fn = coeffs(variant)
        ev = lambda k: fn[k](X, *B) + 0 * X * B[0]
        f = f + eps ** 2 * ev("f2")
        Om = Om + eps ** 2 * ev("Om2")
        Pi = Pi + se * eps * ev("Pi2")
        Psi = Psi + se * eps ** 2 * ev("Psi2")
        if order >= 3:
            f = f + eps ** 3 * ev("f3")
            Om = Om + eps ** 3 * ev("Om3")
            Pi = Pi + se * eps ** 2 * ev("Pi3")
    return dict(f=f, Om=Om, Pi=Pi, Psi=Psi)


def beta_tilde(r):
    eps = 1.0 / (r["D"] - 1.0)
    i0 = int(np.argmin(np.abs(r["xs"])))
    assert r["xs"][i0] == 0.0
    return eps, tau_derivs(r["Pi"][i0] / np.sqrt(eps), r["Delta"])


# expected exponents of the maximal residual (absolute, over 0 <= x <= 1): the first omitted or wrong order
MODELS = {"LO": ("corrected", 1), "NLO": ("corrected", 2), "NNLO printed": ("printed", 3),
          "NNLO corrected": ("corrected", 3)}
EXPECT = {"LO": dict(f=2, Om=2, Pi=1.5, Psi=2.5), "NLO": dict(f=3, Om=3, Pi=2.5, Psi=3.5),
          "NNLO printed": dict(f=3, Om=3, Pi=2.5, Psi=3.5), "NNLO corrected": dict(f=4, Om=4, Pi=3.5, Psi=3.5)}
TOL = 0.25          # |p_fit - p_expected| allowed (criterion C1-C3)
SEP = 0.5           # the fit must be farther than this from the competing order (one order lower)


def residuals(r, model):
    eps, bt = beta_tilde(r)
    out = {}
    for k in FIELDS:
        d = r[k] - model[k]
        ia = np.unravel_index(np.argmax(np.abs(d)), d.shape)
        out[k] = dict(max=float(np.abs(d).max()), x_at=float(r["xs"][ia[0]]), j_at=int(ia[1]),
                      rel_rms=float(np.sqrt(np.mean(d ** 2)) / np.sqrt(np.mean(r[k] ** 2))))
        if k == "f":
            out[k]["rel_rms_1mf"] = float(np.sqrt(np.mean(d ** 2)) / np.sqrt(np.mean((1 - r[k]) ** 2)))
    return out


def fit_exponent(eps, R):
    """ln R = a + p ln eps + b eps, least squares; and the local exponents between neighbours"""
    eps, R = np.asarray(eps), np.asarray(R)
    A = np.c_[np.ones_like(eps), np.log(eps), eps]
    sol, *_ = np.linalg.lstsq(A, np.log(R), rcond=None)
    res = np.log(R) - A @ sol
    A2 = np.c_[np.ones_like(eps), np.log(eps)]
    sol2, *_ = np.linalg.lstsq(A2, np.log(R), rcond=None)
    o = np.argsort(eps)
    loc = [float(np.log(R[o[i + 1]] / R[o[i]]) / np.log(eps[o[i + 1]] / eps[o[i]])) for i in range(len(o) - 1)]
    return dict(p=float(sol[1]), b=float(sol[2]), rms_log=float(np.sqrt(np.mean(res ** 2))), p_pure=float(sol2[1]),
                local=loc)


def fits_of(recs, table, dmin, drop=()):
    use = [i for i, r in enumerate(recs) if r["D"] >= dmin and i not in drop]
    eps = [1.0 / (recs[i]["D"] - 1.0) for i in use]
    return {name: {k: fit_exponent(eps, [rows[i][k]["max"] for i in use]) for k in FIELDS}
            for name, rows in table.items()}


def run_compare(recs, variants, label, verbose=True, dmin=40.0, drop=()):
    """residual table for each model on every state; exponent fits over D >= dmin (the main fit) and over D >= 20,
    leaving out the states in drop (indices into recs sorted by D)"""
    recs = sorted(recs, key=lambda r: r["D"])
    eps = [1.0 / (r["D"] - 1.0) for r in recs]
    table = {}
    for name, (variant, order) in variants.items():
        rows = []
        for r in recs:
            e, bt = beta_tilde(r)
            rows.append(residuals(r, model_fields(e, bt, r["xs"], variant, order)))
        table[name] = rows
    fits = fits_of(recs, table, dmin, drop)
    fits20 = fits_of(recs, table, 20.0, drop)
    if verbose:
        print(f"\n== {label}: maximal residual over 0 <= x <= 1 and the tau grid", flush=True)
        for name, rows in table.items():
            print(f"  {name}")
            print("    D        " + "".join(f"{k:>12s}" for k in FIELDS) + "      (x of the max)")
            for i, (r, row) in enumerate(zip(recs, rows)):
                print(f"    {r['D']:8.3f} " + "".join(f"{row[k]['max']:12.3e}" for k in FIELDS)
                      + "      " + " ".join(f"{row[k]['x_at']:.2f}" for k in FIELDS) + ("  dropped" if i in drop else ""))
            print(f"    exponent " + "".join(f"{fits[name][k]['p']:12.3f}" for k in FIELDS)
                  + f"   (a + p ln eps + b eps, D >= {dmin:g}: the main fit)")
            print("    b        " + "".join(f"{fits[name][k]['b']:12.2f}" for k in FIELDS))
            print("    pure     " + "".join(f"{fits[name][k]['p_pure']:12.3f}" for k in FIELDS) + "   (a + p ln eps)")
            print("    D >= 20  " + "".join(f"{fits20[name][k]['p']:12.3f}" for k in FIELDS) + "   (a + p ln eps + b eps)")
            print("    last two " + "".join(f"{fits[name][k]['local'][0]:12.3f}" for k in FIELDS)
                  + "   (local, two largest D)")
    return recs, eps, table, fits, fits20


def state_floor(r):
    """the state's own field error, in field units: continuity of the centre and horizon series against the nodes,
    and the horizon block's quad-double residual scaled to Omega (2 (n - 1)/x^2 at x = 0.7)"""
    c = r.get("info", {}).get("cont")
    if not c:
        return 0.0
    n = r["D"] - 2
    return max(max(c["centre"].values()), max(c["horizon"].values()), 2 * (n - 1) / 0.49 * c["horizon_block"])


def spread(recs, dirs, table):
    """the corrected NNLO's residual field from other chains' states at the same D, against these: max over the grid
    of |(F - F_E)_this - (F - F_E)_other| relative to this state's R(D), per field"""
    out = []
    byD = {round(r["D"], 4): (i, r) for i, r in enumerate(recs)}
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npz"):
                continue
            o = load_fields(os.path.join(d, f))
            key = round(o["D"], 4)
            if key not in byD:
                continue
            i, r = byD[key]
            e, bt = beta_tilde(r)
            eo, bto = beta_tilde(o)
            m = model_fields(e, bt, r["xs"], "corrected", 3)
            mo = model_fields(eo, bto, o["xs"], "corrected", 3)
            row = dict(D=r["D"], other=o["info"]["source"])
            for k in FIELDS:
                row[k] = float(np.abs((r[k] - m[k]) - (o[k] - mo[k])).max() / table["NNLO corrected"][i][k]["max"])
            row["Delta_diff"] = r["Delta"] - o["Delta"]
            out.append(row)
    return out


def verdicts(fits, expect, names):
    """C1/C2-type criteria: each field's fitted exponent within TOL of the expected, and farther than SEP from the
    exponent one order lower (eps^1 for f, Omega, Pi, Psi alike)"""
    out = []
    for name in names:
        for k in FIELDS:
            p, pe = fits[name][k]["p"], expect[name][k]
            ok = abs(p - pe) <= TOL and abs(p - (pe - 1)) > SEP
            out.append(dict(model=name, field=k, p=p, expected=pe, ok=bool(ok)))
    return out


def rms_change(recs):
    """Fig. 5's measure: the relative rms deviation rms(F_sol - F_model)/rms(F_sol) on the (tau, x) grid, printed and
    corrected, and the change printed - corrected relative to rms(F_sol), per D"""
    rows = []
    for r in recs:
        e, bt = beta_tilde(r)
        mp_ = model_fields(e, bt, r["xs"], "printed", 3)
        mc = model_fields(e, bt, r["xs"], "corrected", 3)
        row = dict(D=r["D"])
        for k in FIELDS:
            nrm = np.sqrt(np.mean(r[k] ** 2))
            row[k] = dict(printed=float(np.sqrt(np.mean((r[k] - mp_[k]) ** 2)) / nrm),
                          corrected=float(np.sqrt(np.mean((r[k] - mc[k]) ** 2)) / nrm),
                          change=float(np.sqrt(np.mean((mp_[k] - mc[k]) ** 2)) / nrm),
                          change_max=float(np.abs(mp_[k] - mc[k]).max() / np.abs(r[k]).max()))
        rows.append(row)
    return rows


def cmd_compare(argv, recs=None, label="solutions"):
    js = next((a.split("=", 1)[1] for a in argv if a.startswith("--json=")), None)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    files = [a for a in argv if not a.startswith("--")]
    dmax = float(next((a.split("=", 1)[1] for a in argv if a.startswith("--dmax=")), 1e9))
    if recs is None:
        recs = [load_fields(f) for f in files]
    recs = [r for r in recs if r["D"] <= dmax]
    variants = dict(MODELS)
    names = ["NNLO corrected"]
    expect = dict(EXPECT)
    if mut and mut.startswith("pi3x"):
        mname = f"NNLO corrected, Pi3 x {mut[4:]}"
        variants[mname] = (mut, 3)
        expect[mname] = EXPECT["NNLO corrected"]
        names = [mname]
    dmin = float(next((a.split("=", 1)[1] for a in argv if a.startswith("--dmin=")), 40))
    dirs = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--spread=")), [])
    recs, eps, table, fits, fits20 = run_compare(recs, variants, label + (f", mutation {mut}" if mut else ""),
                                                 verbose=False, dmin=dmin)
    # C6: the state's own error below 1% of the corrected NNLO's residual, else the state leaves the fit
    floors = []
    drop = []
    for i, r in enumerate(recs):
        fl = state_floor(r)
        ratio = max(fl / table["NNLO corrected"][i][k]["max"] for k in FIELDS)
        floors.append(dict(D=r["D"], floor=fl, ratio=ratio))
        if ratio > 0.01:
            drop.append(i)
    sp = spread(recs, dirs, table) if dirs else []
    for row in sp:
        i = [j for j, r in enumerate(recs) if round(r["D"], 4) == round(row["D"], 4)][0]
        if max(row[k] for k in FIELDS) > 0.01 and i not in drop:
            drop.append(i)
    recs, eps, table, fits, fits20 = run_compare(recs, variants, label + (f", mutation {mut}" if mut else ""),
                                                 dmin=dmin, drop=drop)
    print("\n== C6, the states' own error: max over fields of floor/R (corrected NNLO); > 0.01 drops the state")
    print("    " + "  ".join(f"{f['D']:.1f}:{f['ratio']:.0e}" for f in floors))
    for row in sp:
        print(f"    spread at D {row['D']:.1f} against {os.path.relpath(row['other'], os.path.dirname(row['other']) + '/../..')}: "
              + " ".join(f"{k} {row[k]:.1e}" for k in FIELDS) + f"  (of R; Delta differs by {row['Delta_diff']:.1e})")
    print(f"    dropped: {[recs[i]['D'] for i in drop] if drop else 'none'}")
    ver = verdicts(fits, expect, list(MODELS) if not mut else names)
    print("\n== verdicts (fitted exponent within %.2f of the expected, farther than %.2f from one order lower)" % (TOL, SEP))
    for v in ver:
        print(f"   {v['model']:28s} {v['field']:4s} p = {v['p']:6.3f}  expected {v['expected']:4.1f}  "
              f"{'PASS' if v['ok'] else 'FAIL'}")
    rows = rms_change(recs)
    print("\n== Fig. 5's measure, relative rms deviation rms(F_sol - F)/rms(F_sol): printed / corrected / "
          "change printed - corrected")
    print("    D       " + "".join(f"{k:>34s}" for k in FIELDS))
    for row in rows:
        print(f"    {row['D']:7.3f} " + "".join(
            f"   {row[k]['printed']:9.2e} {row[k]['corrected']:9.2e} {row[k]['change']:9.2e}" for k in FIELDS))
    if js:
        with open(js, "w") as fh:
            json.dump(dict(label=label, mutation=mut, D=[r["D"] for r in recs], eps=eps, Delta=[r["Delta"] for r in recs],
                           sources=[r.get("info", {}).get("source", r.get("path")) for r in recs],
                           continuity=[r.get("info", {}).get("cont") for r in recs],
                           table=table, fits=fits, fits_D20=fits20, dmin=dmin, floors=floors, spread=sp,
                           dropped=[recs[i]["D"] for i in drop], verdicts=ver, fig5=rows, tol=TOL, sep=SEP,
                           expect=expect), fh, indent=1)
        print(f"\nwrote {js}")
    # main run: every model's verdict must hold; mutation run: the corrected-with-mutation verdict must fail for Pi
    # and Omega (exit 0 when it does) and hold for f and Psi
    if mut and mut.startswith("pi3x"):
        broke = all(not v["ok"] for v in ver if v["field"] in ("Pi", "Om"))
        kept = all(v["ok"] for v in ver if v["field"] in ("f", "Psi"))
        print(f"\nmutation {mut}: Pi and Omega scaling broken: {broke}; f and Psi kept: {kept}  -> "
              f"{'the control fires (exit 0)' if broke and kept else 'CONTROL FAILED (exit 1)'}")
        return 0 if (broke and kept) else 1
    ok = all(v["ok"] for v in ver)
    print(f"\n{'ALL PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


# ------------------------------------------------------------------ planted truth
PLANT = dict(f=4.0, Om=4.0, Pi=3.5, Psi=3.5)


def planted_records(recs, plant, scale=1.0):
    """each state's fields replaced by the corrected NNLO of its own beta~ (same eps, Delta, grid) plus
    scale (1 + 4 eps + 20 eps^2) eps^p g with known shapes g; Pi's g vanishes at x = 0, so beta~ is unchanged"""
    out = []
    for r in recs:
        e, bt = beta_tilde(r)
        X = r["xs"][:, None]
        B0, B1 = bt[0][None, :], bt[1][None, :]
        m = model_fields(e, bt, r["xs"], "corrected", 3)
        cf = scale * (1 + 4 * e + 20 * e ** 2)
        g = dict(f=0.3 * (1 - X ** 2) * B0 ** 4, Om=0.5 * X ** 2 * B0 ** 2 * (B0 ** 2 + B1),
                 Pi=0.4 * X ** 2 * B0 * (B0 ** 2 - B1), Psi=0.4 * (B0 + B1) * (1 + X ** 2))
        q = dict(r)
        for k in FIELDS:
            q[k] = m[k] + cf * e ** plant[k] * g[k]
        q["path"] = "planted:" + os.path.basename(r.get("path", ""))
        out.append(q)
    return out


def cmd_planted(argv):
    plant = dict(PLANT)
    pp = next((a.split("=", 1)[1] for a in argv if a.startswith("--plant_pi=")), None)
    if pp is not None:
        plant["Pi"] = float(pp)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    js = next((a.split("=", 1)[1] for a in argv if a.startswith("--json=")), None)
    files = [a for a in argv if not a.startswith("--")]
    scale = float(next((a.split("=", 1)[1] for a in argv if a.startswith("--plant_scale=")), 1.0))
    recs = planted_records([load_fields(f) for f in files], plant, scale)
    variants = {"NNLO corrected": ("corrected", 3), "NNLO printed": ("printed", 3)}
    mname = f"NNLO corrected, Pi3 x {mut[4:]}" if mut and mut.startswith("pi3x") else None
    if mname:
        variants[mname] = (mut, 3)
    dmin = float(next((a.split("=", 1)[1] for a in argv if a.startswith("--dmin=")), 40))
    recs, eps, table, fits, _ = run_compare(recs, variants, f"planted {plant}" + (f", mutation {mut}" if mut else ""),
                                            dmin=dmin)
    # P1: the corrected model recovers the planted exponents (within 0.1)
    # P2: the printed model against the planted data shows its own error, 2.5 for Pi and 3 for f, Omega (within TOL),
    #     unless Pi's plant is lower than 2.5 + SEP
    ver = []
    for k in FIELDS:
        p = fits["NNLO corrected"][k]["p"]
        ver.append(dict(check="P1 corrected recovers the plant", field=k, p=p, expected=plant[k],
                        ok=bool(abs(p - plant[k]) <= 0.1)))
    for k, pe in (("f", 3.0), ("Om", 3.0), ("Pi", 2.5)):
        p = fits["NNLO printed"][k]["p"]
        ver.append(dict(check="P2 printed shows its error", field=k, p=p, expected=pe, ok=bool(abs(p - pe) <= TOL)))
    if mname:
        for k in ("Pi", "Om"):
            p = fits[mname][k]["p"]
            ver.append(dict(check="P3 the mutation breaks the plant", field=k, p=p, expected=plant[k],
                            ok=bool(abs(p - plant[k]) > TOL)))
    print("\n== planted verdicts")
    for v in ver:
        print(f"   {v['check']:36s} {v['field']:4s} p = {v['p']:6.3f}  planted/expected {v['expected']:4.2f}  "
              f"{'PASS' if v['ok'] else 'FAIL'}")
    if js:
        with open(js, "w") as fh:
            json.dump(dict(plant=plant, plant_scale=scale, mutation=mut, D=[r["D"] for r in recs], table=table, fits=fits,
                           verdicts=ver), fh, indent=1)
        print(f"wrote {js}")
    ok = all(v["ok"] for v in ver)
    print(f"\n{'ALL PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


# ------------------------------------------------------------------ Figs. 4-5: D-independent model profiles
EQ16 = dict(A=3.8455, q=0.0551, delta=0.0985)       # arXiv:2611.07007v1 Eq. (16) and (S61)-(S62), as printed


def dinf_ratios(path=None):
    """F_nu/F_1 (nu = 1, 3, ..., 25) of the tuned D = infinity profile at A = 2.71918, the J 1500 block of
    solver/eeg_profile/data/tuned_profile_A2.71918.txt (EEG_PROFILE.md 3.3; #133's [4/3] fit reproduces it to 1e-6)"""
    path = path or os.path.join(REPO, "solver", "eeg_profile", "data", "tuned_profile_A2.71918.txt")
    txt = open(path).read()
    block = txt[txt.index("J 1500"):]
    out = {}
    for line in block.splitlines()[3:]:
        w = line.split()
        if len(w) < 3 or not w[0].isdigit():
            break
        out[int(w[0])] = float(w[1]) * np.exp(1j * float(w[2]))
    return out


def model_profile(kind, r):
    """beta~ on the state's tau grid from a D-independent profile, its fundamental's phase aligned with the state's
    beta~ (EEG fix the time origin by the fundamental, EEG_PROFILE.md 1); tau derivatives with the state's Delta.
    kind eq16: Eq. (16) with the printed (A_*, q, delta), our gauge's ratios B_{nu+2}/B_nu = w = q e^{i delta}
    (EEG_PROFILE.md 3.1); kind dinf: the tuned D = infinity profile, beta~ = sqrt2 F with |F_1| = A/2, A = 2.71918."""
    eps, bt = beta_tilde(r)
    M = len(bt[0])
    phs = np.angle(np.fft.rfft(np.r_[bt[0], -bt[0]])[1])          # the state's fundamental, of e^{i theta}
    # coefficients of e^{i nu theta} in our gauge, F_1 = -i A/2 (EEG_PROFILE.md 3.1)
    if kind == "eq16":
        w = EQ16["q"] * np.exp(1j * EQ16["delta"])
        coef = {2 * k + 1: -1j * (EQ16["A"] / 2) * w ** k for k in range(12)}
    elif kind == "dinf":
        coef = {nu: -1j * R2 * (2.71918 / 2) * v for nu, v in dinf_ratios().items()}
    else:
        raise ValueError(kind)
    ts = phs - np.angle(coef[1])                                     # theta -> theta + ts aligns the fundamentals
    th = 2 * np.pi * np.arange(M) / (2 * M)
    b = sum(2 * np.real(cf * np.exp(1j * nu * (th + ts))) for nu, cf in coef.items())
    return eps, tau_derivs(b, r["Delta"])


def cmd_figs(argv):
    js = next((a.split("=", 1)[1] for a in argv if a.startswith("--json=")), None)
    files = [a for a in argv if not a.startswith("--")]
    recs = sorted([load_fields(f) for f in files], key=lambda r: r["D"])
    out = {}
    for kind in ("exact", "eq16", "dinf"):
        rows = []
        for r in recs:
            e, bt = beta_tilde(r) if kind == "exact" else model_profile(kind, r)
            mp_ = model_fields(e, bt, r["xs"], "printed", 3)
            mc = model_fields(e, bt, r["xs"], "corrected", 3)
            m2 = model_fields(e, bt, r["xs"], "corrected", 2)
            row = dict(D=r["D"], profile_rel_rms=float(np.sqrt(np.mean((bt[0] - beta_tilde(r)[1][0]) ** 2)) /
                                                         np.sqrt(np.mean(beta_tilde(r)[1][0] ** 2))))
            for k in FIELDS:
                ref = (1 - r[k]) if k == "f" else r[k]
                nrm = np.sqrt(np.mean(ref ** 2))
                row[k] = dict(printed=float(np.sqrt(np.mean((r[k] - mp_[k]) ** 2)) / nrm),
                              corrected=float(np.sqrt(np.mean((r[k] - mc[k]) ** 2)) / nrm),
                              change=float(np.sqrt(np.mean((mp_[k] - mc[k]) ** 2)) / nrm),
                              change_max=float(np.abs(mp_[k] - mc[k]).max() / np.abs(ref).max()),
                              nlo=float(np.sqrt(np.mean((r[k] - m2[k]) ** 2)) / nrm))
            # f normalised by rms(f) instead of rms(1 - f) (Fig. 5 does not say which)
            nf = np.sqrt(np.mean(r["f"] ** 2))
            row["f_by_f"] = dict(printed=float(np.sqrt(np.mean((r["f"] - mp_["f"]) ** 2)) / nf),
                                 corrected=float(np.sqrt(np.mean((r["f"] - mc["f"]) ** 2)) / nf))
            # Fig. 4: the centre profiles, x = 0. Pi_3 and Omega_3 vanish there and Psi_3 is not used, so printed and
            # corrected differ in f alone (through c_0); the others' differences are recorded as a check
            i0 = int(np.argmin(np.abs(r["xs"])))
            row["centre"] = {k: dict(change_max=float(np.abs(mp_[k][i0] - mc[k][i0]).max()),
                                     dev_printed=float(np.abs(r[k][i0] - mp_[k][i0]).max()),
                                     dev_corrected=float(np.abs(r[k][i0] - mc[k][i0]).max()),
                                     scale=float(np.abs((1 - r[k][i0]) if k == "f" else r[k][i0]).max()))
                             for k in FIELDS}
            rows.append(row)
        out[kind] = rows
        print(f"\n== profile {kind}: relative rms deviation (Fig. 5's measure; f relative to rms(1 - f)): "
              "NLO / NNLO printed / NNLO corrected / change printed - corrected", flush=True)
        print("    D       profile " + "".join(f"{k:>41s}" for k in FIELDS))
        for row in rows:
            print(f"    {row['D']:7.3f} {row['profile_rel_rms']:8.1e}" + "".join(
                f"  {row[k]['nlo']:9.2e} {row[k]['printed']:9.2e} {row[k]['corrected']:9.2e} {row[k]['change']:9.2e}"
                for k in FIELDS))
        print(f"  centre x = 0 (Fig. 4): max |printed - corrected| / max |1 - f| and max deviation of printed, "
              "corrected / max |1 - f|; other fields' max |printed - corrected| (must be 0)")
        for row in rows:
            c = row["centre"]
            sc = c["f"]["scale"]
            print(f"    {row['D']:7.3f}  f: change {c['f']['change_max'] / sc:9.2e}  deviation printed "
                  f"{c['f']['dev_printed'] / sc:9.2e} corrected {c['f']['dev_corrected'] / sc:9.2e}   "
                  + " ".join(f"{k} {c[k]['change_max']:.0e}" for k in ("Om", "Pi", "Psi")))
    if js:
        with open(js, "w") as fh:
            json.dump(dict(D=[r["D"] for r in recs], eq16=EQ16, figs=out), fh, indent=1)
        print(f"wrote {js}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("extract", "compare", "planted", "figs"):
        print(__doc__)
        sys.exit(2)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "extract":
        cmd_extract(rest)
        sys.exit(0)
    if cmd == "figs":
        sys.exit(cmd_figs(rest))
    sys.exit(cmd_compare(rest) if cmd == "compare" else cmd_planted(rest))
