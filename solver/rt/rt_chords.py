"""What the chords between the D 40.02 RT root samples carry: the read-only tests T0a to T0d of notes/P7_scope.md
section 3.4 (issue #983). Nothing is solved.

    python3 solver/rt/rt_chords.py RAW [--tests a,b,c,d] [--prec 192] [--out OUT.json] [--mutate=seed]
    python3 solver/rt/rt_chords.py --selftest [--mutate=f64|nyq|null]

RAW is a checkout of choptuik-large-d-raw with the three samples of RT_FORMULATION.md section 14 (D 40.020065,
32 x 121) at K = 0 and K = 7: I671-win/solves (a), I701-win/solves (b), I951-win/solves (c), and #163's table
I163-mac/checks/exponents.log. A chord is y - c for a sample y, formed in double-double and rounded to double
(rt_perturb.diff); every share and cosine below is in the 2-norm of the unknown vector, as #969's.

T0a  The cosine of the K = 7 chords a7 - c7 and b7 - c7. Control: the K = 0 chords must reproduce #969's cosine 0.234
     and its orthogonal share 0.97 (part of b - c orthogonal to a - c), or the run exits 1.
T0b  Each K = 0 chord's share along the phase orbit d_t z(c) (rt_model's t-derivative of every field, mu part 0) and
     the share of its Nyquist components (the top t-frequency of the periodic fields omega_1..3, whose derivative
     rt_model sets to zero). Printed with them: the chord's content by field and in its lowest t-harmonic (harmonic 0
     of omega_1..3, wavenumbers +-1/2 of omega_4).
T0c  Omega_5^+ (rt_constraint.omega5, arb at --prec) at a, b, c and at c moved by #963's random move (rt_perturb.move,
     delta 1e-20, seed 977). g(y) = max|Omega_5(y) - Omega_5(c)| / max|y - c| for each y, and rho = g(chord) / g(random).
     Omega_5 is quadratic, so each difference is its derivative at the midpoint applied to the chord, exactly. The
     chords themselves carry the double-double rounding of the samples, which is printed as each chord's rounding share
     2^-105 max|z| / max|chord|: rho cannot be read below about that share. Control: rt_sigma's 192-bit max|F| at c and
     at the moved state must print #951's 6.91e-31 and #963's 3.36e-18, or the run exits 1. --mutate=seed regenerates
     the move with seed 978, and that control must then fail (exit 1); only the control is run then.
T0d  rt_branch_cheb.py (run as a subprocess) on s^lambda4 cos(om ln s), om = k / mu for k = 1/2, 3/2, 5/2, with lambda4
     and mu of sample c (rt_horizon.exponents' formula), and the k = 1/2 branch at each row of #163's exponents table.
     Its interpolation error e on N = 121 (and N^2 e, an upper estimate of a copy's gain, Markov's factor for the
     derivative) is set against the root-level band [1.5e-35, 2.6e-33] (the K = 0 medians of section 14): below if
     N^2 e < 1.5e-35, above if e > 2.6e-33, in the band otherwise. Control: rt_branch_cheb.py --planted must exit 0.

--selftest runs the cosine, share and gain-ratio code on planted data through the same functions: chords of
double-double samples 3e-21 apart (their double parts equal), a chord along an analytic d_t z, a planted Nyquist mode,
and a quadratic map with a planted null direction. --mutate=f64 forms the chords from the double parts only, nyq
projects on the harmonic below Nyquist, null removes the planted null direction; each must make the selftest fail.
"""
import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rt_model import RT                                   # noqa: E402
from rt_perturb import diff, move, dd_add                 # noqa: E402

SAMPLES = {   # section 14's samples a, b, c at D 40.020065 on 32 x 121
    "a0": "I671-win/solves/k0_D40_mutnorows_D40.0201_32x121.npz",
    "b0": "I701-win/solves/k0b_D40_mutnorows_D40.0201_32x121.npz",
    "c0": "I951-win/solves/k0c_D40_mutnorows_D40.0201_32x121.npz",
    "a7": "I671-win/solves/k7_D40_D40.0201_32x121.npz",
    "b7": "I701-win/solves/k7b_D40_D40.0201_32x121.npz",
    "c7": "I951-win/solves/k7c_D40_D40.0201_32x121.npz",
}
EXPONENTS = "I163-mac/checks/exponents.log"
REF_COS, REF_ORTH = 0.234, 0.97          # #969: cosine of a - c and b - c, part of b - c orthogonal to a - c
REF_F_C, REF_F_MOVED = "6.91e-31", "3.36e-18"   # #951's and #963's printed max|F| (rt_sigma, 192 bits)
BAND = (1.5e-35, 2.6e-33)
CARRY, EXCLUDE = 0.5, 0.1                # T0b: share >= 0.5 carries the chord, <= 0.1 excludes it


def load(path):
    d = np.load(path)
    return d, np.asarray(d["zh"], dtype=float), np.asarray(d["zl"], dtype=float)


def chord(y, c, mutate=None):
    """y - c from double-double samples (zh, zl), rounded to double; --mutate=f64 drops the low parts"""
    if mutate == "f64":
        return y[0] - c[0]
    return diff(y, c[0], c[1])


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def orth_share(u, v):
    """the 2-norm share of v orthogonal to u"""
    w = v - (v @ u) / (u @ u) * u
    return float(np.linalg.norm(w) / np.linalg.norm(v))


# ------------------------------------------------------------------------------------------------ T0b
def dt_z(rt, z):
    """the phase orbit's tangent: rt_model's t-derivative of every field at z, mu part 0"""
    w1, w2, w3, w4, _ = rt.unpack(z)
    return rt.pack(rt.dt(w1)[0], rt.dt(w2)[0], rt.dt(w3)[0], rt.dt(w4, anti=True)[0], 0.0)


def nyquist_part(rt, v, mutate=None):
    """v's component in the Nyquist modes of the periodic fields (omega_1 on xi > 0, omega_2, omega_3); --mutate=nyq
    projects on harmonic Nt/2 - 1's cosine instead"""
    w1, w2, w3, w4, mu = rt.unpack(v)
    t = rt.t
    q = np.cos(rt.Nt // 2 * t) if mutate != "nyq" else np.cos((rt.Nt // 2 - 1) * t) * np.sqrt(2)
    proj = lambda f: np.outer(q, q @ f / rt.Nt)                                  # noqa: E731
    z0 = np.zeros((rt.Nt, rt.N + 1))
    return rt.pack(proj(w1[0]), proj(w2[0]), proj(w3[0]), z0, 0.0)


def share_along(u, v):
    return abs(cos(u, v))


def content(rt, v):
    """2-norm^2 shares of v by unknown block, and of each field's lowest harmonic"""
    w1, w2, w3, w4, mu = rt.unpack(v)
    tot = float(v @ v)
    Nt = rt.Nt
    blocks = {"w1": float(v[:rt.n1] @ v[:rt.n1]), "w2": float(np.sum(w2 ** 2)), "w3": float(np.sum(w3 ** 2)),
              "w4": float(np.sum(w4 ** 2)), "mu": float(mu[0, 0, 0] ** 2)}
    low = 0.0
    for f in (w1[0][:, rt.pos], w2[0], w3[0]):
        low += Nt * float(np.sum(f.mean(axis=0) ** 2))
    G = np.fft.fft(w4[0] * rt.shift[:, None], axis=0)
    low += float(np.sum(np.abs(G[0]) ** 2 + np.abs(G[Nt - 1]) ** 2)) / Nt
    out = {k: b / tot for k, b in blocks.items()}
    out["lowest harmonic"] = low / tot
    return out


# ------------------------------------------------------------------------------------------------ T0c
def gain(fun, fc, zc_dd, y_dd):
    """max|fun(y) - fun(c)| / max|y - c|, the differences of the samples formed in double-double"""
    return float(np.abs(fun(y_dd) - fc).max()) / float(np.abs(diff(y_dd, *zc_dd)).max())


# ------------------------------------------------------------------------------------------------ T0d
def branch_cheb(lams, oms, planted=False):
    cmd = [sys.executable, os.path.join(HERE, "rt_branch_cheb.py"), "--lam", ",".join(f"{v:.6f}" for v in lams),
           "--om", ",".join(f"{v:.6f}" for v in oms)] + (["--planted"] if planted else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    errs = [[float(x) for x in ln.split(":")[1].split()] for ln in p.stdout.splitlines() if "interpolation error" in ln]
    return p.returncode, errs, p.stdout


def place(e, N=121):
    if N * N * e < BAND[0]:
        return "below"
    if e > BAND[1]:
        return "above"
    return "in the band"


def exponents_row(path):
    import rt_horizon
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rt_horizon.exponents([path])
    row = [r for r in buf.getvalue().splitlines() if r.startswith("| ") and "grid" not in r][0]
    c = [x.strip() for x in row.strip("|").split("|")]
    return float(c[2]), float(c[4])            # mu, lambda_4


def exponents_table(path):
    rows = []
    for ln in open(path, encoding="utf-8"):
        m = re.match(r"\|\s*([\d.]+)\s*\|\s*(\S+)\s*\|\s*([\d.]+)\s*\|\s*(-?[\d.]+)\s*\|\s*(-?[\d.]+)\s*\|", ln)
        if m:
            rows.append((float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(5))))
    return rows


# ------------------------------------------------------------------------------------------------ selftest
def selftest(mutate):
    rng = np.random.default_rng(983)
    D, Nt, N = 40.020065, 8, 11
    rt = RT(D, Nt, N)
    T, X = np.meshgrid(rt.t, rt.xi, indexing="ij")
    f = lambda k, ph, j: np.cos(k * T + ph) * X ** j                                    # noqa: E731
    w1 = np.sin(1.3 * X) * np.cos(T + 0.4)
    w2, w3 = f(1, 0.3, 2) + 0.5 * f(2, 1.1, 1), f(0, 0, 1) + 0.2 * f(3, 0.7, 3)
    w4 = f(0.5, 0.2, 1) + 0.3 * f(1.5, 1.9, 2)
    zc = rt.pack(w1, w2, w3, w4, 0.13)
    c = (zc, zc * 2.0 ** -60 * rng.uniform(-1, 1, size=zc.size))
    ok = True
    # chords 3e-21 apart, carried by the low parts only
    r1, r2 = rng.normal(size=zc.size), rng.normal(size=zc.size)
    a = dd_add(*c, 3e-21 * r1, np.zeros(zc.size))
    b = dd_add(*c, 3e-21 * (0.25 * r1 + r2), np.zeros(zc.size))
    ua, ub = chord(a, c, mutate), chord(b, c, mutate)
    want = cos(r1, 0.25 * r1 + r2)
    got = cos(ua, ub) if np.linalg.norm(ua) > 0 and np.linalg.norm(ub) > 0 else float("nan")
    t1 = abs(got - want) < 1e-9 and abs(orth_share(ua, ub) - orth_share(r1, 0.25 * r1 + r2)) < 1e-9
    print(f"selftest cosine: planted {want:.9f}, got {got:.9f}: {'PASS' if t1 else 'FAIL'}")
    # the phase orbit against the analytic t-derivative
    d_an = rt.pack(-np.sin(1.3 * X) * np.sin(T + 0.4), -np.sin(T + 0.3) * X ** 2 - np.sin(2 * T + 1.1) * X,
                   -0.6 * np.sin(3 * T + 0.7) * X ** 3, -0.5 * np.sin(0.5 * T + 0.2) * X
                   - 0.45 * np.sin(1.5 * T + 1.9) * X ** 2, 0.0)
    s_on = share_along(dt_z(rt, zc), d_an)
    mu_only = np.zeros(zc.size)
    mu_only[-1] = 1.0
    s_off = share_along(dt_z(rt, zc), mu_only)
    t2 = abs(s_on - 1) < 1e-12 and s_off < 1e-12
    print(f"selftest phase share: along d_t z {s_on:.15f}, mu alone {s_off:.1e}: {'PASS' if t2 else 'FAIL'}")
    # a planted Nyquist mode in omega_2 at one xi, and a smooth chord
    nq = np.zeros((Nt, N + 1))
    nq[:, 3] = np.cos(Nt // 2 * rt.t)
    v_nq = rt.pack(np.zeros((Nt, N + 1)), nq, np.zeros((Nt, N + 1)), np.zeros((Nt, N + 1)), 0.0)
    s_nq = float(np.linalg.norm(nyquist_part(rt, v_nq, mutate)) / np.linalg.norm(v_nq))
    sm = d_an
    s_sm = float(np.linalg.norm(nyquist_part(rt, sm, mutate)) / np.linalg.norm(sm))
    t3 = abs(s_nq - 1) < 1e-12 and s_sm < 1e-12
    print(f"selftest Nyquist share: planted mode {s_nq:.15f}, smooth chord {s_sm:.1e}: {'PASS' if t3 else 'FAIL'}")
    # the gain ratio on a quadratic map with a planted null direction n: G(z) = A z + (n_perp . z)^2 e
    L = 400
    A = rng.normal(size=(L, L))
    n = rng.normal(size=L)
    n /= np.linalg.norm(n)
    if mutate != "null":
        A = A - np.outer(A @ n, n)               # A n = 0
    e1 = rng.normal(size=L)
    P = np.eye(L) - np.outer(n, n)
    G = lambda y: A @ (y[0] + y[1]) + (P @ (y[0] + y[1]) @ e1) ** 2 * e1 / L          # noqa: E731
    z0 = (rng.normal(size=L), np.zeros(L))
    g0 = G(z0)
    y_null = dd_add(*z0, 1e-6 * n, np.zeros(L))
    y_rand = dd_add(*z0, 1e-6 * rng.normal(size=L) / np.sqrt(L), np.zeros(L))
    rho = gain(G, g0, z0, y_null) / gain(G, g0, z0, y_rand)
    t4 = rho < 1e-6
    print(f"selftest gain ratio: planted null direction rho {rho:.1e} (< 1e-6): {'PASS' if t4 else 'FAIL'}")
    # the T0d parser and the band
    t5 = place(1e-40) == "below" and place(1e-30) == "above" and place(1e-36) == "in the band"
    print(f"selftest band: {'PASS' if t5 else 'FAIL'}")
    ok = t1 and t2 and t3 and t4 and t5
    print(f"selftest{' --mutate=' + mutate if mutate else ''}: {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", nargs="?")
    ap.add_argument("--tests", default="a,b,c,d")
    ap.add_argument("--prec", type=int, default=192)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", default=None, choices=["f64", "nyq", "null", "seed"])
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest(a.mutate) else 1
    tests = set(a.tests.split(","))
    if a.mutate == "seed":
        tests = {"c"}
    S = {k: load(os.path.join(a.raw, p)) for k, p in SAMPLES.items()}
    d0 = S["c0"][0]
    D, Nt, N = float(d0["D"]), int(d0["Nt"]), int(d0["N"])
    rt = RT(D, Nt, N)
    dd = {k: (v[1], v[2]) for k, v in S.items()}
    out = {"D": D, "grid": f"{Nt}x{N}", "L": rt.L, "random_line_share": 1 / np.sqrt(rt.L)}
    ok = True
    print(f"rt_chords: D {D} {Nt}x{N}, L {rt.L}; a random unit vector's share along a fixed line is about "
          f"{1 / np.sqrt(rt.L):.4f}", flush=True)
    ch = {"a-c": chord(dd["a0"], dd["c0"]), "b-c": chord(dd["b0"], dd["c0"]), "b-a": chord(dd["b0"], dd["a0"])}
    zmax = float(np.abs(dd["c0"][0] + dd["c0"][1]).max())
    if "a" in tests:
        c0 = cos(ch["a-c"], ch["b-c"])
        o0 = orth_share(ch["a-c"], ch["b-c"])
        a7, b7 = chord(dd["a7"], dd["c7"]), chord(dd["b7"], dd["c7"])
        c7, o7 = cos(a7, b7), orth_share(a7, b7)
        ctl = abs(abs(c0) - REF_COS) < 5e-4 and abs(o0 - REF_ORTH) < 5e-3
        ok &= ctl
        rd = "parallel: (H2)" if abs(c7) >= 1 - 1e-6 else ("not parallel: not (H2)" if abs(c7) <= 0.99 else
                                                           "undecided")
        print(f"T0a  K 0: cos(a-c, b-c) {c0:.6f}, part of b-c orthogonal to a-c {o0:.4f} (control #969 {REF_COS}, "
              f"{REF_ORTH}: {'ok' if ctl else 'FAILED'}); max|a-c|/max|z| {np.abs(ch['a-c']).max() / zmax:.3e}, "
              f"max|b-c|/max|z| {np.abs(ch['b-c']).max() / zmax:.3e}", flush=True)
        print(f"     K 7: cos(a7-c7, b7-c7) {c7:.9f}, orthogonal part {o7:.3e}; max|a7-c7|/max|z| "
              f"{np.abs(a7).max() / zmax:.3e}, max|b7-c7|/max|z| {np.abs(b7).max() / zmax:.3e}; reading: {rd}",
              flush=True)
        out["T0a"] = {"cos_K0": c0, "orth_K0": o0, "control": ctl, "cos_K7": c7, "orth_K7": o7, "reading": rd}
    if "b" in tests:
        tz = dt_z(rt, dd["c0"][0])
        res = {}
        for k, v in ch.items():
            sp = share_along(tz, v)
            sn = float(np.linalg.norm(nyquist_part(rt, v)) / np.linalg.norm(v))
            cn = content(rt, v)
            rp = "carries: (HP)" if sp >= CARRY else ("excluded" if sp <= EXCLUDE else "undecided")
            rn = "carries: (HN)" if sn >= CARRY else ("excluded" if sn <= EXCLUDE else "undecided")
            res[k] = {"phase": sp, "nyquist": sn, "phase_reading": rp, "nyquist_reading": rn,
                      "content": cn}
            print(f"T0b  {k}: share along d_t z(c) {sp:.3e} ({rp}), Nyquist share {sn:.3e} ({rn}); 2-norm^2 by block "
                  + ", ".join(f"{q} {w:.3f}" for q, w in cn.items()), flush=True)
        out["T0b"] = res
    if "c" in tests:
        import flint
        from rt_mp import RTmp
        import rt_mpsolve as mps
        from rt_constraint import omega5
        flint.ctx.prec = a.prec
        t0 = time.time()
        K = int(d0["horizon"]) if "horizon" in d0.files else 0
        rm = RTmp(D, Nt, N, prec=a.prec, horizon=K)
        seed = 978 if a.mutate == "seed" else 977
        mv = move(*dd["c0"], 1e-20, seed)
        zc = mps.from_dd(*dd["c0"])
        Fc = f"{mps.amax(rm.residual_m(zc)):.2e}"
        Fm = f"{mps.amax(rm.residual_m(mps.from_dd(*mv))):.2e}"
        ctl = Fc == REF_F_C and Fm == REF_F_MOVED
        ok &= ctl
        print(f"T0c  control: max|F| at c {Fc} (#951 {REF_F_C}), at the moved state {Fm} (#963 {REF_F_MOVED}), "
              f"seed {seed}: {'ok' if ctl else 'FAILED'} ({time.time() - t0:.0f} s)", flush=True)
        out["T0c_control"] = {"F_c": Fc, "F_moved": Fm, "seed": seed, "ok": ctl}
        if a.mutate != "seed":
            f5 = lambda y: np.array(omega5(rm, mps.from_dd(*y)), dtype=object).ravel()               # noqa: E731
            oc = f5(dd["c0"])

            def g5(y):
                dO = np.array([float((u - w).mid()) for u, w in zip(f5(y), oc)])
                return float(np.abs(dO).max()) / float(np.abs(diff(y, *dd["c0"])).max())
            gr = g5(mv)
            res = {"random": gr, "max_O5_c": max(abs(float(v.mid())) for v in oc)}
            print(f"T0c  max|Omega_5^+| at c {res['max_O5_c']:.3e}; g(random) {gr:.3e}", flush=True)
            for k, y in (("a-c", dd["a0"]), ("b-c", dd["b0"])):
                gk = g5(y)
                rho = gk / gr
                rnd = 2.0 ** -105 * zmax / float(np.abs(diff(y, *dd["c0"])).max())
                rd = "(HC)" if rho >= 1e-3 else ("not (HC)" if rho <= 1e-8 else "undecided")
                res[k] = {"g": gk, "rho": rho, "rounding_share": rnd, "reading": rd}
                print(f"T0c  {k}: g {gk:.3e}, rho {rho:.3e} (chord rounding share {rnd:.1e}); reading: {rd}",
                      flush=True)
            out["T0c"] = res
            print(f"T0c  {time.time() - t0:.0f} s", flush=True)
    if "e" in tests:      # outside the readings: how the K = 7 chords sit against the K = 0 plane, and per field
        a7, b7 = chord(dd["a7"], dd["c7"]), chord(dd["b7"], dd["c7"])
        o7 = b7 - (b7 @ a7) / (a7 @ a7) * a7
        Q = np.linalg.qr(np.c_[ch["a-c"], ch["b-c"]])[0]
        inplane = lambda v: float(np.linalg.norm(Q.T @ v) / np.linalg.norm(v))              # noqa: E731
        res = {"cos(a7-c7, a-c)": cos(a7, ch["a-c"]), "cos(a7-c7, b-c)": cos(a7, ch["b-c"]),
               "in K0 plane: a7-c7": inplane(a7), "in K0 plane: orth part of b7-c7": inplane(o7)}
        print("E    " + ", ".join(f"{k} {v:.4f}" for k, v in res.items()), flush=True)
        for k, v in (("a-c", ch["a-c"]), ("b-c", ch["b-c"]), ("a7-c7", a7), ("orth(b7-c7)", o7)):
            w1, w2, w3, w4, mu = rt.unpack(v)
            per = {}
            for name, f, anti in (("w1", w1[0][:, rt.pos], False), ("w2", w2[0], False), ("w3", w3[0], False),
                                  ("w4", w4[0], True)):
                tot = float(np.sum(f ** 2))
                if anti:
                    G = np.fft.fft(f * rt.shift[:, None], axis=0)
                    low = float(np.sum(np.abs(G[0]) ** 2 + np.abs(G[Nt - 1]) ** 2)) / Nt
                else:
                    low = Nt * float(np.sum(f.mean(axis=0) ** 2))
                per[name] = low / tot
            res[k + " lowest-harmonic share by field"] = per
            print(f"E    {k}: share of each field's 2-norm^2 in its lowest harmonic (0, or +-1/2 for w4): "
                  + ", ".join(f"{q} {w:.4f}" for q, w in per.items()), flush=True)
        out["extra"] = res
    if "d" in tests:
        t0 = time.time()
        rc, _, txt = branch_cheb([21.0], [0.0], planted=True)
        ctl = rc == 0
        ok &= ctl
        print(f"T0d  rt_branch_cheb.py --planted exit {rc}: {'ok' if ctl else 'FAILED'}", flush=True)
        mu, lam4 = exponents_row(os.path.join(a.raw, SAMPLES["c0"]))
        ks = [0.5, 1.5, 2.5]
        rc, errs, txt = branch_cheb([lam4] * 3, [k / mu for k in ks])
        res = {"mu": mu, "lambda4": lam4, "k": {}}
        print(f"T0d  sample c: mu {mu:.6f}, lambda_4 {lam4:.4f}", flush=True)
        for k, e in zip(ks, errs):
            e121 = e[3]
            res["k"][str(k)] = {"e_N61_81_101_121": e, "N2e": 121 ** 2 * e121, "place": place(e121)}
            print(f"T0d  k {k}: om {k / mu:.4f}, interpolation error on N = 61, 81, 101, 121 "
                  + " ".join(f"{v:.1e}" for v in e) + f"; N^2 e {121 ** 2 * e121:.1e}: {place(e121)}", flush=True)
        tab = exponents_table(os.path.join(a.raw, EXPONENTS))
        rc, errs, txt = branch_cheb([r[3] for r in tab], [0.5 / r[2] for r in tab])
        res["against_D"] = []
        for (Dd, grid, mud, l4), e in zip(tab, errs):
            res["against_D"].append({"D": Dd, "mu": mud, "lambda4": l4, "e121": e[3], "place": place(e[3])})
            print(f"T0d  D {Dd:9.6f} ({grid}): mu {mud:.6f}, lambda_4 {l4:.4f}, k 1/2: e(121) {e[3]:.1e}, "
                  f"N^2 e {121 ** 2 * e[3]:.1e}: {place(e[3])}", flush=True)
        out["T0d"] = res
        print(f"T0d  {time.time() - t0:.0f} s", flush=True)
    out["controls_ok"] = bool(ok)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    print(f"exit {0 if ok else 1}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
