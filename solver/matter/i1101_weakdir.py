#!/usr/bin/env python3
"""Weak directions of the l = 1 DSS system at k_c 55, D 16 (#1105, #1101): which directions of the solution vector do
re-solves of the same discrete system move along at an unchanged |F|, what are they, and what holds them?

The system is DSSEll as ell_fastm.load_system builds it from a stub record (kc, NODES_STD, n = D - 2, l = 1, Delta_ref =
z[-1], Ks, u0, rk; freeze_from(z, kappa) with the record's kappa). J = DSSEll.jacobian(z) (complex step), its SVD
J = U diag(s) V^T, s descending. Bands by s/s_max: null below 1e-11, weak in [1e-11, 1e-5), strong from 1e-5 up.

usage (venv-1025dd, from solver/matter):
  i1101_weakdir.py jac REC.jsonl STATE.npy D OUT.npz [--pin=STATEBLOB,RECBLOB]
      F, J and the step map of the record system, saved to OUT.npz (exit 3 if not the record system or a pin differs).
  i1101_weakdir.py svd JAC.npz OUT.npz
      the SVD of J (U, s, Vt) saved to OUT.npz.
  i1101_weakdir.py planted SVD.npz [--mutate=basis]
      planted differences dz = a v_i + b v_j + e w on the stored SVD (v_i, v_j the two weakest non-null right vectors,
      w a strong one): the coordinates come back to 1e-10 of |dz| and the weak-band fraction equals
      sqrt(a^2 + b^2)/|dz| to 1e-10. --mutate=basis projects on U instead of V: must fail. Exit 0 only if it passes.
  i1101_weakdir.py report JAC.npz SVD.npz D NAME=REC:STATE [NAME=REC:STATE ...] [--out=R.json] [--mutate=gsign]
      the first NAME is the base (J and the SVD are its). Prints and writes: the spectrum's low end and the bands; for
      each state its |F| by band of U, U_w^T F on the weak vectors with its rounding noise (z (1 + 1e-16 g), 3 seeds),
      its weak coordinates V_w^T (z - z_base) and the Gauss-Newton corrected ones (c - U_w^T F / s); for each
      difference to the base and between consecutive named states, the fraction in each band; each weak vector (right
      and left) and the null band by field, node and tau-harmonic band; the tau-translation generator g (harmonic k
      rotated by i k, Delta 0): |J_mis g| / (|J| |g|) on the mismatch rows (check: <= 1e-6; --mutate=gsign rotates the
      odd fields by -i k and must give >= 1e-3), its phase row, and its fraction in each band. A STATE written
      STATE+EPS@INDEX is planted: the state moved by EPS along v[INDEX] (counted from the smallest singular value),
      then its strong part re-solved by 8 chord steps on the stored SVD; its coordinate there must move by EPS and its Gauss-Newton corrected coordinates by at most 1% of EPS
      (--mutate=sgn adds U^T F / s instead of subtracting it: must fail). Plants at +EPS and -EPS on one vector form a
      pair: the linear part of U^T F, half their difference, must equal s EPS to 1% (--mutate=swap takes the other
      weak vector's s: must fail); added after the single-plant bar failed on the curvature of U^T F (11% at 1e-3). Exit 1 if the translation check, its
      control or the planted check reads wrong. Also the horizon relabelling t -> t (1 + eps h(tau)) for h = 1,
      cos 2 w tau, sin 2 w tau (gauge_generator): J dz outside the horizon block <= 1e-6 s_max |dz| (the control,
      without the -(h - h_tau) c term of dc, >= 1e-3; --mutate=noc applies it to all), its horizon-block residual and
      its fraction in each band. Exit 1 also if that check or its control reads wrong.
  i1101_weakdir.py valley JAC.npz SVD.npz REC STATE D [--ts=1e-4,1e-3,...] [--save=T:PATH,...] [--ncorr=8] [--refreeze=1]
      the Gauss-Newton step on the weak non-null vectors below 1e-8 (the step that would zero U_w^T F), taken by a
      fraction t, then the strong part re-solved by chord steps on the stored SVD (s/s_max >= 1e-5): per t, |F|,
      U_w^T F and Delta. --save stores the corrected state at that t (and a stub record beside it, F its |F|).
      --refreeze=1: the moved state's RK steps from freeze_from at the record's kappa before the chord steps (so that
      load_system rebuilds the same system from the saved state, which the line reports).
  i1101_weakdir.py ne REC STATE D OUTDIR [--outer=6]
      nonlinear elimination toward the discrete zero along the weak pair (fresh J and SVD per outer step, the
      Gauss-Newton step on the weak pair at fractions 1 to 1/8, re-frozen steps, chord steps on the strong band);
      stops when an outer step cuts |U_w^T F| by less than 2x; every iterate saved with a stub record.
  i1101_weakdir.py ddres SVD.npz REC STATE D
      every block's residual in double-double (ell_ddm.DDEll's blocks, values only) beside the float64 one, and both
      projected on the SVD's bands: whether the weak residual is float64 rounding.
  i1101_weakdir.py coords SVD.npz BASE.npy NAME=STATE.npy ...
      the band fractions and weak coordinates V^T (z - z_base) of saved states (any step map; no residual).
  i1101_weakdir.py detline REC_A STATE_A STATE_B D LAM [--ts=0,0.25,...] [--hybrid=T,...]
      the float64 (and, at --hybrid, the hybrid) determinant of M(LAM) along z_A + t (z_B - z_A) on A's step map.
  i1101_weakdir.py detparts SVD.npz REC_A STATE_A STATE_B D LAM [--b12=fast|ddm|rust] [--b12_threads=T]
      the hybrid determinant of M(LAM) at z_A, z_A + P_band (z_B - z_A) for the null, weak and strong bands, and z_B.
  i1101_weakdir.py floor JAC.npz SVD.npz REC STATE D
      U_w^T F at the state under one change of the discretisation at a time: every segment's RK steps x2, the horizon
      segment's x2, Ks 45, N + 56, u0 0.035 and 0.045 (horizon steps by freeze_from's rule at that u0).
  i1101_weakdir.py detsens SVD.npz REC STATE D LAM1 LAM2 NAME=EPS:INDEX ... [--out=F.jsonl]
      the hybrid determinant (d18_secant.RealEval: float64 blocks 0-11, ell_ddm's double-double block 12) at LAM1 and
      LAM2 for the state and for the state + EPS v_INDEX (INDEX counted from the smallest singular value), and the
      det-linear root of each pair: the root's shift per unit weak coordinate.
  i1101_weakdir.py locate REC STATE D LAMS
      the float64 sign and ln|det| of M(lambda) (mode_matrix_fast, slogdet) at each lambda of LAMS (comma-separated).
  i1101_weakdir.py segerr REC STATE D [--kappas=0.7,...,0.04375] [--svd=SVD.npz] [--out=F.json]
      per segment (block 0 the log segment, 1-11 node to node, 12 the horizon segment), that segment's steps alone set
      by freeze_from at each kappa' (the others at the record's): the change of its block's mismatch against the
      finest kappa', and the local order from consecutive refinements (log of the ratio over log of the step ratio);
      with --svd also U_w^T of the change at the first kappa'.
  i1101_weakdir.py chord SVD.npz REC STATE D MAP OUT.npy [--maxit=12]
      the state re-solved on a modified step map (MAP: kappa=K for every segment at K, or seg=I,J:K for those blocks'
      segments at K) by chord steps on the SVD's strong band, keeping the weak and null coordinates; saved with
      OUT.map.json (the map, |F|, Delta).
  i1101_weakdir.py root STATE LO HI OUT.jsonl [--tol=1e-7] [--maxev=20] [--dln=0.1] [--b12=ddm|fast|rust] [--b12_threads=T]
      d18_secant's Brent refinement (RealEval: float64 blocks 0-11, ell_ddm block 12) of a chord state on its own map.
  i1101_weakdir.py order KAPPA=ROOTS.jsonl ... [--out=F.json] | order selftest [--mutate=kappa]
      the root's kappa order: the least-squares fit root = r0 + A kappa^p (p on a 0.001 grid in [0.5, 8]) and the p
      of each three-point ratio, from the det-linear roots of d18_secant's root records; the bars of the README.
      selftest: planted roots (p 4, 2, 2.5, noise 3e-5) must give p within 0.5; --mutate=kappa reverses the labels.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ell_fastm import load_system                 # noqa: E402

MUTATE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")
NULL_CUT, WEAK_CUT = 1e-11, 1e-5
FIELDS = ("p", "m", "a", "c", "q")
PARF = ("odd", "odd", "even", "even", "odd")
HBANDS = (0, 1, 4, 8, 16, 32, 10 ** 6)             # harmonic bands [0], [1-3], [4-7], [8-15], [16-31], [32-]


def blob(path):
    return subprocess.run(["git", "hash-object", "--no-filters", path], capture_output=True, text=True,
                          check=True).stdout.strip()


def opts(argv):
    return dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)


def pos(argv):
    return [a for a in argv if not a.startswith("--")]


def resid(S, z):
    with np.errstate(all="ignore"):
        return S.residual(z[None, :])[0]


# ---------------------------------------------------------------- labels of unknowns and equations
def unknown_labels(S):
    """(part, node, field, k) per unknown; part is d, c0, node, mS, aS, qS or Delta"""
    sizes, K, L = S.layout()
    names = [("d", -1, "d"), ("c0", -1, "c0")] + [("node", i, f) for i in range(K) for f in FIELDS] + \
            [("mS", K, "mS"), ("aS", K, "aS"), ("qS", K, "qS")]
    out = []
    for (part, node, f), par, sz in zip(names, S.parities(), sizes):
        ks = list(S.kodd) * 2 if par == "odd" else [0] + list(S.keven[1:]) * 2
        assert len(ks) == sz
        out += [(part, node, f, int(k)) for k in ks]
    out.append(("Delta", -2, "Delta", -1))
    assert len(out) == L
    return out


def equation_labels(S):
    """(block, field, k) per residual row; the last row is the phase condition"""
    sizes, K, L = S.layout()
    out = []
    for b in range(K + 1):
        for f, par in zip(FIELDS, PARF):
            ks = list(S.kodd) * 2 if par == "odd" else [0] + list(S.keven[1:]) * 2
            out += [(b, f, int(k)) for k in ks]
    out.append((-1, "phase", 1))
    assert len(out) == L
    return out


def hband(k):
    return int(np.searchsorted(HBANDS, k, side="right") - 1)


def describe_right(e, lab):
    """energy fractions of a right vector (or an aggregate e = sum of squares) by field, node and harmonic band"""
    tot = float(e.sum())
    byf, byn, byh, byk = {}, {}, np.zeros(len(HBANDS) - 1), {}
    for w, (part, node, f, k) in zip(e, lab):
        key = f if part == "node" else part
        byf[key] = byf.get(key, 0.0) + w
        nk = "centre" if node == -1 else ("Delta" if node == -2 else ("horizon" if part in ("mS", "aS", "qS")
                                                                         else f"x{node}"))
        byn[nk] = byn.get(nk, 0.0) + w
        if k >= 0:
            byh[hband(k)] += w
            byk[k] = byk.get(k, 0.0) + w
    r = lambda d: {k: round(v / tot, 4) for k, v in d.items() if v / tot >= 5e-4}  # noqa: E731
    topk = sorted(byk.items(), key=lambda t: -t[1])[:6]
    return dict(fields=r(byf), nodes=r(byn), hbands=[round(v / tot, 4) for v in byh],
                top_k=[[int(k), round(v / tot, 4)] for k, v in topk])


def describe_left(e, lab):
    tot = float(e.sum())
    byb, byf, byk = {}, {}, {}
    for w, (b, f, k) in zip(e, lab):
        byb[b] = byb.get(b, 0.0) + w
        byf[f] = byf.get(f, 0.0) + w
        byk[k] = byk.get(k, 0.0) + w
    r = lambda d: {str(k): round(v / tot, 4) for k, v in d.items() if v / tot >= 5e-4}  # noqa: E731
    topk = sorted(byk.items(), key=lambda t: -t[1])[:6]
    return dict(blocks=r(byb), fields=r(byf), top_k=[[int(k), round(v / tot, 4)] for k, v in topk])


def bands_of(s):
    r = s / s[0]
    return np.where(r < NULL_CUT)[0], np.where((r >= NULL_CUT) & (r < WEAK_CUT))[0], np.where(r >= WEAK_CUT)[0]


# ---------------------------------------------------------------- tau translation
def tau_generator(S, z, sign_odd=1.0):
    """d z / d delta for every field shifted in its phase theta = omega tau by delta: harmonic k times i k
    (Re -> -k Im, Im -> k Re); Delta 0. sign_odd = -1 is the control (odd fields rotated the wrong way)."""
    sizes, K, L = S.layout()
    g = np.zeros_like(z)
    o = 0
    for par, sz in zip(S.parities(), sizes):
        c = z[o:o + sz]
        if par == "odd":
            m = len(S.kodd)
            k = S.kodd.astype(float)
            g[o:o + m] = sign_odd * (-k * c[m:])
            g[o + m:o + sz] = sign_odd * (k * c[:m])
        else:
            m = len(S.keven) - 1
            k = S.keven[1:].astype(float)
            g[o + 1:o + 1 + m] = -k * c[1 + m:]
            g[o + 1 + m:o + sz] = k * c[1:1 + m]
        o += sz
    return g


# ---------------------------------------------------------------- horizon relabelling (gauge)
def gauge_generator(S, z, hk, part="cos", mutate=""):
    """dz for the time relabelling t -> t (1 + eps h(tau)), h = cos or sin(hk omega tau) (hk even, so that the parities
    keep): r, a, Phi, Pi, psi are unchanged at a spacetime point and alpha dt is, so with tau' = tau - eps h and
    x' = x (1 - eps h) the scalars f = p, m, a, psi move by h (f_tau + x f_x) and c by h (c_tau + x c_x) - (h - h_tau) c.
    Centre data: q by h (q_tau + l q) (psi ~ q x^l), c0 by h c0_tau - (h - h_tau) c0; horizon data m, a, psi by
    h (f_tau - f_1) with f_1 the series' first coefficient in u = 1 - x. The SSH condition c(tau, 1) = 1 is not kept:
    there dc = h_tau - (1 - c_x(1)) h, nonzero for periodic h (its rate is gauge_lambda), so J dz is expected only in
    the horizon block. mutate=noc drops the -(h - h_tau) c term (the control: the interior blocks must then fail)."""
    f = S.fields(z)
    om = np.array([2 * np.pi / f["Delta"]])
    N = S.N
    th = 2 * np.pi * np.arange(N) / N                       # omega tau on the grid
    h = np.cos(hk * th) if part == "cos" else np.sin(hk * th)
    ht = (-hk * np.sin(hk * th) if part == "cos" else hk * np.cos(hk * th)) * om[0]
    cterm = 0.0 if mutate == "noc" else 1.0
    Dt = lambda g: S.D(g[None, :], om)[0]                   # noqa: E731
    parts = []
    cut = [0.0, 0.0]                                        # node-state energy above k_c (cut by spec) and in all
    q, c0 = f["d"], f["c0"]
    parts.append(S.spec((h * (Dt(q) + S.ell * q))[None, :], "odd")[0])
    parts.append(S.spec((h * Dt(c0) - cterm * (h - ht) * c0)[None, :], "even")[0])
    for x, st in zip(S.nodes, f["states"]):
        fx = S.rhs(x, st[:, None, :], om)[:, 0, :]          # x-derivatives of p, m, a, c, psi at the node
        for j, (fld, par) in enumerate(zip(FIELDS, PARF)):
            g = h * (Dt(st[j]) + x * fx[j])
            if fld == "c":
                g = g - cterm * (h - ht) * st[j]
            parts.append(S.spec(g[None, :], par)[0])
            G = np.abs(np.fft.rfft(g)) ** 2
            cut[0] += float(G[S.kc + 1:].sum())
            cut[1] += float(G.sum())
    P_, M_, A_, C_, Q_ = S.ssh_series(f["mS"][None, :], f["aS"][None, :], f["qS"][None, :], om)
    for g0, f1, par in ((f["mS"], M_[1][0], "odd"), (f["aS"], A_[1][0], "even"), (f["qS"], Q_[1][0], "odd")):
        parts.append(S.spec((h * (Dt(g0) - f1))[None, :], par)[0])
    gauge_generator.cut = float(np.sqrt(cut[0] / cut[1]))
    return np.concatenate(parts + [[0.0]])


# ---------------------------------------------------------------- subcommands
def jaccmd(argv):
    args, o = pos(argv), opts(argv)
    recf, st, D, out = args[0], args[1], float(args[2]), args[3]
    if "pin" in o:
        ps, pr = o["pin"].split(",")
        bs, br = blob(st), blob(recf)
        if not (bs.startswith(ps) and br.startswith(pr)):
            print(f"pin differs: state {bs[:8]} (want {ps}), record {br[:8]} (want {pr})", flush=True)
            return 3
    S, z, rec, same, msg = load_system(recf, st, D)
    print(msg, flush=True)
    if not same:
        return 3
    t0 = time.time()
    F, J = S.jacobian(z)
    sec = time.time() - t0
    keys = [str(k) for k in S.steps_map]
    vals = [int(v) for v in S.steps_map.values()]
    np.savez(out, F=F, J=J, z=z, steps_keys=np.array(keys), steps_vals=np.array(vals), D=D, kc=rec["kc"],
             kappa=rec["kappa"], u0=rec["u0"], Ks=rec["Ks"])
    print(json.dumps(dict(out=out, F=float(np.linalg.norm(F)), J_shape=list(J.shape), seconds=round(sec, 1),
                          steps=int(sum(vals)))), flush=True)
    return 0


def svdcmd(argv):
    args = pos(argv)
    d = np.load(args[0])
    t0 = time.time()
    U, s, Vt = np.linalg.svd(d["J"])
    np.savez(args[1], U=U, s=s, Vt=Vt)
    print(f"svd {d['J'].shape}: {time.time() - t0:.1f} s, s_max {s[0]:.6e}, s_min {s[-1]:.3e}", flush=True)
    return 0


def plantedcmd(argv):
    args = pos(argv)
    d = np.load(args[0])
    U, s, Vt = d["U"], d["s"], d["Vt"]
    nul, weak, strong = bands_of(s)
    basis = U.T if MUTATE == "basis" else Vt          # rows: the vectors projected on
    ok = True
    rng = np.random.default_rng(7)
    i, j = weak[-1], weak[-2]                          # the two weakest non-null right vectors
    w = strong[rng.integers(len(strong))]
    for a, b, e in ((3e-6, -1e-6, 0.0), (1e-6, 2e-6, 5e-7), (0.0, 1e-6, 1e-5)):
        dz = a * Vt[i] + b * Vt[j] + e * Vt[w]
        c = basis @ dz
        nz = np.linalg.norm(dz)
        err = max(abs(c[i] - a), abs(c[j] - b), abs(c[w] - e)) / nz
        frac = float(np.linalg.norm(c[weak]) / nz)
        want = float(np.hypot(a, b) / nz)
        good = err <= 1e-10 and abs(frac - want) <= 1e-10
        ok &= good
        print(f"  planted a {a:g} b {b:g} e {e:g}: coordinate error {err:.2e} of |dz|, weak fraction {frac:.12f} "
              f"(planted {want:.12f}) {'ok' if good else 'FAIL'}", flush=True)
    print(f"planted {'PASS' if ok else 'FAIL'}{' (mutate=' + MUTATE + ')' if MUTATE else ''}", flush=True)
    return 0 if ok else 1


def reportcmd(argv):
    args, o = pos(argv), opts(argv)
    jd, sd, D = np.load(args[0]), np.load(args[1]), float(args[2])
    named = []
    for a in args[3:]:
        name, rs = a.split("=", 1)
        recf, st = rs.split(":", 1)
        named.append((name, recf, st))
    U, s, Vt, J = sd["U"], sd["s"], sd["Vt"], jd["J"]
    n = len(s)
    nul, weak, strong = bands_of(s)
    S, zb, rec, same, msg = load_system(named[0][1], named[0][2], D)
    print(msg, flush=True)
    if not same or not np.array_equal(zb, jd["z"]):
        print("the base state is not the Jacobian's: refused", flush=True)
        return 3
    out = dict(D=D, kc=rec["kc"], kappa=rec["kappa"], u0=rec["u0"], Ks=rec["Ks"], steps=rec["steps"],
               s_max=float(s[0]), n=n, bands=dict(null=len(nul), weak=len(weak), strong=len(strong)),
               cuts=dict(null=NULL_CUT, weak=WEAK_CUT))
    low = [float(v) for v in s[::-1][:70] / s[0]]
    out["s_rel_lowest70"] = low
    print(f"s_max {s[0]:.6e}; bands: null {len(nul)} (s/s_max < {NULL_CUT:g}), weak {len(weak)}, strong {len(strong)}",
          flush=True)
    print("lowest 70 s/s_max (from the smallest):", " ".join(f"{v:.2e}" for v in low), flush=True)
    wk = list(weak[::-1])                               # weak vectors, weakest first
    widx = [int(n - 1 - i) for i in wk]                 # their index counted from the smallest
    out["weak"] = [dict(index_from_smallest=wi, s=float(s[i]), s_rel=float(s[i] / s[0])) for wi, i in zip(widx, wk)]
    print("weak band (index from the smallest, s, s/s_max):",
          " ".join(f"[{wi} {s[i]:.4e} {s[i] / s[0]:.3e}]" for wi, i in zip(widx, wk)), flush=True)
    # ---- states
    lab = unknown_labels(S)
    elab = equation_labels(S)
    rng = np.random.default_rng(1)
    zs, rows = {}, []
    for name, recf, st in named:
        plant = None
        if "+" in st:                                   # planted: STATE+EPS@INDEX, the state moved along v[INDEX]
            st, pl = st.split("+", 1)
            eps, idx = pl.split("@")
            plant = (float(eps), int(idx))
        S_, z, rec_, same_, msg_ = load_system(recf, st, D)
        if S_.steps_map != S.steps_map:
            print(f"{name}: its step map differs from the base's: refused", flush=True)
            return 3
        if plant is not None:
            z = z + plant[0] * Vt[n - 1 - plant[1]]
            for _ in range(8):                          # the strong part re-solved by chord steps on the stored SVD
                z = z - Vt[strong].T @ ((U[:, strong].T @ resid(S, z)) / s[strong])
            print(f"{name}: planted, {os.path.basename(st)} + {plant[0]:g} v[{plant[1]}], then 8 chord steps on the "
                  f"strong band", flush=True)
        zs[name] = z
        F = resid(S, z)
        a = U.T @ F
        uw = a[wk]
        noise = np.std([U[:, wk].T @ resid(S, z * (1 + 1e-16 * rng.standard_normal(z.shape))) - uw
                        for _ in range(3)], axis=0)
        c = Vt[wk] @ (z - zb)
        cs = c - uw / s[wk] if MUTATE != "sgn" else c + uw / s[wk]
        r = dict(name=name, state=os.path.basename(st), blob=blob(st)[:8], same=bool(same_), F=float(np.linalg.norm(F)),
                 plant=plant,
                 F_null=float(np.linalg.norm(a[nul])), F_weak=float(np.linalg.norm(a[weak])),
                 F_strong=float(np.linalg.norm(a[strong])), Delta=float(z[-1]), uwF=[float(v) for v in uw],
                 uwF_noise=[float(v) for v in noise], c=[float(v) for v in c], c_gn=[float(v) for v in cs],
                 constraint=float(S.constraint(z)), lam_gauge=float(S.gauge_lambda(z)))
        rows.append(r)
        print(f"{name} ({r['blob']}): |F| {r['F']:.4e} = null {r['F_null']:.3e} + weak {r['F_weak']:.3e} + strong "
              f"{r['F_strong']:.3e}; Delta {r['Delta']:.13f}", flush=True)
        print(f"   U_w^T F        {' '.join(f'{v:+.4e}' for v in uw)}", flush=True)
        print(f"   its noise      {' '.join(f'{v:.1e}' for v in noise)}", flush=True)
        print(f"   c (to base)    {' '.join(f'{v:+.4e}' for v in c)}", flush=True)
        print(f"   c - U^T F / s  {' '.join(f'{v:+.4e}' for v in cs)}", flush=True)
    out["states"] = rows
    pl_ok = True
    for r in rows:
        if r["plant"] is None:
            continue
        base = next(q for q in rows if q["state"] == r["state"] and q["plant"] is None)
        j = widx.index(r["plant"][1])
        dc = r["c"][j] - base["c"][j]
        dgn = max(abs(a - b) for a, b in zip(r["c_gn"], base["c_gn"]))
        good = abs(dc - r["plant"][0]) <= 1e-9 * abs(r["plant"][0]) + 1e-12 and dgn <= 0.01 * abs(r["plant"][0])
        pl_ok = bool(pl_ok and good)
        print(f"planted {r['name']}: coordinate moved {dc:+.6e} (planted {r['plant'][0]:+.3e}); Gauss-Newton corrected "
              f"coordinates moved by at most {dgn:.3e} (bar 1% of the plant) {'ok' if good else 'FAIL'}", flush=True)
    out["planted_check"] = bool(pl_ok)
    # antisymmetric pairs of plants (+EPS and -EPS along one vector): the linear part of U_w^T F, which must equal s EPS
    # on that vector to 1% (the quadratic part cancels); --mutate=swap compares with the other weak vector's s
    anti_ok = None
    for r in rows:
        if r["plant"] is None or r["plant"][0] <= 0:
            continue
        m = [q for q in rows if q["plant"] is not None and q["plant"][1] == r["plant"][1] and q["plant"][0] == -r["plant"][0]]
        if not m:
            continue
        q = m[0]
        j = widx.index(r["plant"][1])
        sj = s[wk[j]]
        if MUTATE == "swap":
            sj = s[wk[j + 1 if j % 2 == 0 else j - 1]]
        lin = (r["uwF"][j] - q["uwF"][j]) / 2
        quad = (r["uwF"][j] + q["uwF"][j]) / 2 - next(b for b in rows if b["state"] == r["state"] and b["plant"] is None)["uwF"][j]
        dev = abs(lin - sj * r["plant"][0]) / (sj * r["plant"][0])
        good = dev <= 0.01
        anti_ok = bool(good) if anti_ok is None else bool(anti_ok and good)
        print(f"planted pair v[{r['plant'][1]}] +-{r['plant'][0]:g}: linear part of U^T F {lin:+.4e} against s eps "
              f"{sj * r['plant'][0]:+.4e} (deviation {dev:.2e}, bar 1e-2) {'ok' if good else 'FAIL'}; quadratic part "
              f"{quad:+.3e}", flush=True)
    out["planted_pair_check"] = anti_ok
    # ---- differences
    diffs = [(f"{nm}-{named[0][0]}", zs[nm] - zb) for nm, _, _ in named[1:]]
    diffs += [(f"{named[i + 1][0]}-{named[i][0]}", zs[named[i + 1][0]] - zs[named[i][0]])
              for i in range(1, len(named) - 1)]
    out["diffs"] = []
    for name, dz in diffs:
        c = Vt @ dz
        nz = float(np.linalg.norm(dz))
        f = dict(name=name, norm=nz, null=float(np.linalg.norm(c[nul]) / nz), weak=float(np.linalg.norm(c[weak]) / nz),
                 strong=float(np.linalg.norm(c[strong]) / nz), c_weak=[float(v) for v in c[wk]],
                 J_dz=float(np.linalg.norm(J @ dz)), dDelta=float(dz[-1]))
        out["diffs"].append(f)
        print(f"{name}: |dz| {nz:.4e}, |J dz| {f['J_dz']:.3e}, dDelta {f['dDelta']:+.3e}; fraction null {f['null']:.2e} "
              f"weak {f['weak']:.6f} strong {f['strong']:.2e}; weak coords {' '.join(f'{v:+.3e}' for v in c[wk])}",
              flush=True)
    # ---- the vectors
    out["right"], out["left"] = [], []
    for wi, i in zip(widx, wk):
        dr = describe_right(Vt[i] ** 2, lab)
        dl = describe_left(U[:, i] ** 2, elab)
        out["right"].append(dict(index_from_smallest=wi, **dr))
        out["left"].append(dict(index_from_smallest=wi, **dl))
        print(f"v[{wi}] s/s_max {s[i] / s[0]:.2e}: fields {dr['fields']}; harmonic bands {dr['hbands']}; top k "
              f"{dr['top_k'][:4]}; nodes {dr['nodes']}", flush=True)
        print(f"u[{wi}]: blocks {dl['blocks']}; fields {dl['fields']}; top k {dl['top_k'][:4]}", flush=True)
    dn = describe_right((Vt[nul] ** 2).sum(axis=0), lab)
    dln = describe_left((U[:, nul] ** 2).sum(axis=1), elab)
    out["null_right"], out["null_left"] = dn, dln
    print(f"null band right (aggregate): fields {dn['fields']}; harmonic bands {dn['hbands']}; nodes {dn['nodes']}",
          flush=True)
    print(f"null band left (aggregate): blocks {dln['blocks']}; fields {dln['fields']}", flush=True)
    # ---- tau translation
    ok = True
    for sign, tag in ((1.0, "g"), (-1.0, "g_control")):
        g = tau_generator(S, zb, sign_odd=sign if MUTATE != "gsign" else -sign)
        Jg = J @ g
        rel = float(np.linalg.norm(Jg[:-1]) / (s[0] * np.linalg.norm(g)))
        c = Vt @ g
        ng = float(np.linalg.norm(g))
        rec_g = dict(rel_mismatch=rel, phase_row=float(Jg[-1]), Re_d1=float(zb[0]), norm=ng,
                     rayleigh=float(np.linalg.norm(Jg) / ng),
                     null=float(np.linalg.norm(c[nul]) / ng), weak=float(np.linalg.norm(c[weak]) / ng),
                     strong=float(np.linalg.norm(c[strong]) / ng), c_weak=[float(v / ng) for v in c[wk]])
        out[tag] = rec_g
        want = rel <= 1e-6 if tag == "g" else rel >= 1e-3
        ok = bool(ok and want)
        print(f"tau generator {tag}: |J_mis g|/(s_max |g|) {rel:.3e} ({'ok' if want else 'FAIL'}: "
              f"{'<= 1e-6' if tag == 'g' else '>= 1e-3'}); phase row {Jg[-1]:+.6e} (Re d_1 {zb[0]:+.6e}); "
              f"|J g|/|g| {rec_g['rayleigh']:.3e}; fraction null {rec_g['null']:.2e} weak {rec_g['weak']:.2e} strong "
              f"{rec_g['strong']:.6f}; on each weak vector {' '.join(f'{v:+.2e}' for v in rec_g['c_weak'])}", flush=True)
    # ---- horizon relabelling
    blk = 3 * S.n_odd + 2 * S.n_even
    gok = True
    out["gauge"] = []
    for hk, part, mut in ((0, "cos", ""), (2, "cos", ""), (2, "sin", ""), (0, "cos", "noc")):
        g = gauge_generator(S, zb, hk, part, mutate=mut if MUTATE != "noc" else "noc")
        Jg = J @ g
        ng = float(np.linalg.norm(g))
        inner = float(np.linalg.norm(Jg[:12 * blk]) / (s[0] * ng))
        hor = float(np.linalg.norm(Jg[12 * blk:-1]) / ng)
        c = Vt @ g
        rec_h = dict(h=f"{part}({hk} w tau)", control=bool(mut), norm=ng, inner_rel=inner, horizon=hor,
                     cut_above_kc=gauge_generator.cut,
                     rayleigh=float(np.linalg.norm(Jg) / ng), null=float(np.linalg.norm(c[nul]) / ng),
                     weak=float(np.linalg.norm(c[weak]) / ng), strong=float(np.linalg.norm(c[strong]) / ng),
                     c_weak=[float(v / ng) for v in c[wk]])
        out["gauge"].append(rec_h)
        want = inner <= 1e-6 if not mut else inner >= 1e-3
        gok = bool(gok and want)
        print(f"gauge generator h = {part}({hk} w tau){' (control: no -(h - h_tau) c term)' if mut else ''}: blocks 0-11 "
              f"|J g|/(s_max |g|) {inner:.3e} ({'ok' if want else 'FAIL'}: {'>= 1e-3' if mut else '<= 1e-6'}); horizon "
              f"block |J g|/|g| {hor:.3e}; |J g|/|g| {rec_h['rayleigh']:.3e}; node-state part above k_c (cut) "
              f"{rec_h['cut_above_kc']:.2e}; fraction null {rec_h['null']:.2e} weak "
              f"{rec_h['weak']:.2e} strong {rec_h['strong']:.4f}; on each weak vector "
              f"{' '.join(f'{v:+.2e}' for v in rec_h['c_weak'])}", flush=True)
    out["gauge_check"] = bool(gok)
    ok = bool(ok and gok)
    out["translation_check"] = bool(ok)
    print(f"translation and gauge checks {'PASS' if ok else 'FAIL'}, planted check {'PASS' if pl_ok else 'FAIL'}"
          f"{' (mutate=' + MUTATE + ')' if MUTATE else ''}", flush=True)
    if "out" in o:
        json.dump(out, open(o["out"], "w"), indent=1)
    if anti_ok is not None:
        print(f"planted pair check {'PASS' if anti_ok else 'FAIL'}{' (mutate=' + MUTATE + ')' if MUTATE else ''}", flush=True)
    return 0 if ok and (pl_ok or anti_ok) else 1


def stub_write(path_rec, S, z, rec, F):
    r = dict(rec)
    for k in ("lam", "gamma", "lam_guess", "n_eval", "n_signchanges"):
        r.pop(k, None)
    r.update(F=float(F), Delta=float(z[-1]), steps=int(sum(S.steps_map.values())), src=os.path.basename(path_rec))
    open(path_rec, "w").write(json.dumps(r) + "\n")


def valleycmd(argv):
    args, o = pos(argv), opts(argv)
    sd = np.load(args[1])
    U, s, Vt = sd["U"], sd["s"], sd["Vt"]
    recf, st, D = args[2], args[3], float(args[4])
    S, z0, rec, same, msg = load_system(recf, st, D)
    print(msg, flush=True)
    if not same:
        return 3
    r = s / s[0]
    wk = np.where((r >= NULL_CUT) & (r < 1e-8))[0]
    strong = np.where(r >= WEAK_CUT)[0]
    Us, ss, Vs = U[:, strong], s[strong], Vt[strong]
    Uw, sw, Vw = U[:, wk], s[wk], Vt[wk]
    F0 = resid(S, z0)
    dgn = -(Vw.T @ ((Uw.T @ F0) / sw))
    print(f"weak vectors below 1e-8: {len(wk)} (s/s_max {' '.join(f'{v:.2e}' for v in r[wk])}); Gauss-Newton step "
          f"|dz| {np.linalg.norm(dgn):.4e}, dDelta {dgn[-1]:+.4e}; U_w^T F at t = 0 "
          f"{' '.join(f'{v:+.4e}' for v in Uw.T @ F0)}", flush=True)
    ts = [float(v) for v in o.get("ts", "1e-4,1e-3,1e-2,0.03,0.1,0.3,1").split(",")]
    saves = dict((float(a.split(":", 1)[0]), a.split(":", 1)[1]) for a in o["save"].split(",")) if "save" in o else {}
    ncorr = int(o.get("ncorr", 8))
    base_map = dict(S.steps_map)
    for t in ts:
        z = z0 + t * dgn
        S.steps_map = dict(base_map)
        if "refreeze" in o:                              # the moved state's own steps (freeze_from at the record kappa)
            S.freeze_from(z, rec["kappa"])
        hist = []
        for _ in range(ncorr):
            Fz = resid(S, z)
            if not np.all(np.isfinite(Fz)):
                hist.append("nan")
                break
            hist.append(f"{np.linalg.norm(Fz):.2e}")
            z = z - Vs.T @ ((Us.T @ Fz) / ss)
        Fz = resid(S, z)
        rw = Uw.T @ Fz
        nF = float(np.linalg.norm(Fz))
        steps = int(sum(S.steps_map.values()))
        own = None
        if "refreeze" in o:
            used = dict(S.steps_map)
            own = used == S.freeze_from(z, rec["kappa"])
            S.steps_map = used
        print(f"t {t:g}: chord |F| {' '.join(hist)} -> {nF:.4e}; U_w^T F {' '.join(f'{v:+.4e}' for v in rw)}; "
              f"Delta {z[-1]:.12f} (dDelta {z[-1] - z0[-1]:+.4e}); steps {steps}"
              f"{'' if own is None else ('; the corrected state keeps them' if own else '; the corrected state re-freezes differently')}",
              flush=True)
        if t in saves:
            np.save(saves[t], z)
            stub_write(saves[t].replace(".npy", ".jsonl"), S, z, rec, nF)
            print(f"   saved {saves[t]} and its stub record (F {nF:.10e})", flush=True)
    return 0


def necmd(argv):
    """nonlinear elimination toward the discrete zero along the weak pair: at each outer step a fresh J and SVD, the
    Gauss-Newton step on the non-null weak vectors below 1e-8 (that would zero U_w^T F), tried at fractions
    1, 0.5, 0.25, 0.125, each followed by the RK steps re-frozen at the moved state and chord steps on the strong band;
    the fraction with the smallest |F| whose strong part converged is taken. Stops when an outer step cuts |U_w^T F| by
    less than 2x, or after --outer steps; each iterate is saved (OUTDIR/sol_neK_D*.npy and a stub record)."""
    args, o = pos(argv), opts(argv)
    recf, st, D, outdir = args[0], args[1], float(args[2]), args[3]
    os.makedirs(outdir, exist_ok=True)
    S, z, rec, same, msg = load_system(recf, st, D)
    print(msg, flush=True)
    if not same:
        return 3
    prev = None
    for k in range(int(o.get("outer", 6))):
        t0 = time.time()
        F, J = S.jacobian(z)
        U, s, Vt = np.linalg.svd(J)
        r = s / s[0]
        wk = np.where((r >= NULL_CUT) & (r < 1e-8))[0]
        strong = np.where(r >= WEAK_CUT)[0]
        Us, ss, Vs = U[:, strong], s[strong], Vt[strong]
        Uw = U[:, wk]
        rw0 = Uw.T @ F
        nw0 = float(np.linalg.norm(rw0))
        dgn = -(Vt[wk].T @ (rw0 / s[wk]))
        print(f"outer {k}: |F| {np.linalg.norm(F):.4e}, weak pair s/s_max {' '.join(f'{v:.3e}' for v in r[wk])}, "
              f"|U_w^T F| {nw0:.4e}, Delta {z[-1]:.12f}; Gauss-Newton |dz| {np.linalg.norm(dgn):.3e} dDelta "
              f"{dgn[-1]:+.4e} ({time.time() - t0:.0f} s for J and SVD)", flush=True)
        if prev is not None and nw0 > 0.5 * prev:
            print(f"   stop: the last outer step cut |U_w^T F| by less than 2x ({prev:.3e} -> {nw0:.3e})", flush=True)
            break
        prev = nw0
        best = None
        for frac in (1.0, 0.5, 0.25, 0.125):
            zt = z + frac * dgn
            S.freeze_from(zt, rec["kappa"])
            hist = []
            for _ in range(10):
                Ft = resid(S, zt)
                if not np.all(np.isfinite(Ft)):
                    hist.append(np.inf)
                    break
                hist.append(float(np.linalg.norm(Ft)))
                zt = zt - Vs.T @ ((Us.T @ Ft) / ss)
            Ft = resid(S, zt)
            nF = float(np.linalg.norm(Ft)) if np.all(np.isfinite(Ft)) else np.inf
            nw = float(np.linalg.norm(Uw.T @ Ft)) if np.isfinite(nF) else np.inf
            print(f"   fraction {frac}: chord |F| {' '.join(f'{v:.2e}' for v in hist)} -> {nF:.4e}, |U_w^T F| {nw:.4e}, "
                  f"Delta {zt[-1]:.12f}, steps {sum(S.steps_map.values())}", flush=True)
            if np.isfinite(nF) and (best is None or nF < best[0]):
                best = (nF, zt.copy(), frac, dict(S.steps_map))
            if np.isfinite(nF) and nF < 0.5 * float(np.linalg.norm(F)):
                break
        if best is None or best[0] >= float(np.linalg.norm(F)):
            print("   stop: no fraction lowers |F|", flush=True)
            break
        z = best[1]
        S.steps_map = best[3]
        own = dict(S.steps_map) == S.freeze_from(z, rec["kappa"])
        S.steps_map = best[3]
        path = os.path.join(outdir, f"sol_ne{k + 1}_D{D:.4f}.npy")
        np.save(path, z)
        stub_write(path.replace(".npy", ".jsonl"), S, z, rec, best[0])
        print(f"   took fraction {best[2]}: |F| {best[0]:.4e}, Delta {z[-1]:.12f}; saved {os.path.basename(path)} "
              f"({'its own step map' if own else 'its step map re-freezes differently'})", flush=True)
    return 0


def ddrescmd(argv):
    """the residual of every block in double-double (ell_ddm.DDEll's blocks, value only, no tangents) against the float64
    residual, and both projected on the weak vectors of a stored SVD: is the floor in U_w^T F float64 rounding?"""
    args, o = pos(argv), opts(argv)
    sd = np.load(args[0])
    U, s = sd["U"], sd["s"]
    recf, st, D = args[1], args[2], float(args[3])
    S, z, rec, same, msg = load_system(recf, st, D)
    print(msg, flush=True)
    if not same:
        return 3
    import ell_ddm
    import ddnp as dd
    E = ell_ddm.DDEll(S)
    E.om = 2 * np.pi / z[-1]
    E.lam = 0.0

    def inputs(X, sizes, pars):
        out, k = [], 0
        for sz in sizes:
            out.append(ell_ddm.DT(dd.R(X[None, k:k + sz].copy()), None))
            k += sz
        return out
    E._inputs = inputs
    sizes, K, L = S.layout()
    blocks, offs = S._block_list()
    blk = 3 * S.n_odd + 2 * S.n_even
    node0 = offs[2]
    F64 = resid(S, z)
    Fdd = np.zeros_like(F64)
    t0 = time.time()
    for b, (c0, nf, fun, tgt) in enumerate(blocks):
        out = E.block(b, z[c0:c0 + nf])
        h = np.concatenate([x.v.h[0] for x in out])
        l = np.concatenate([x.v.l[0] for x in out])
        t0_ = node0 + blk * tgt
        Fdd[b * blk:(b + 1) * blk] = (h - z[t0_:t0_ + blk]) + l
        d = float(np.linalg.norm(F64[b * blk:(b + 1) * blk] - Fdd[b * blk:(b + 1) * blk]))
        print(f"  block {b}: |F64| {np.linalg.norm(F64[b * blk:(b + 1) * blk]):.4e} |Fdd| "
              f"{np.linalg.norm(Fdd[b * blk:(b + 1) * blk]):.4e} |F64 - Fdd| {d:.3e}", flush=True)
    Fdd[-1] = F64[-1]
    r = s / s[0]
    wk = np.where((r >= NULL_CUT) & (r < WEAK_CUT))[0][::-1]
    nul = np.where(r < NULL_CUT)[0]
    strong = np.where(r >= WEAK_CUT)[0]
    for tag, F in (("float64", F64), ("double-double", Fdd), ("difference", F64 - Fdd)):
        a = U.T @ F
        print(f"{tag}: |F| {np.linalg.norm(F):.4e} = null {np.linalg.norm(a[nul]):.3e} + weak {np.linalg.norm(a[wk]):.3e} + "
              f"strong {np.linalg.norm(a[strong]):.3e}; U_w^T F {' '.join(f'{v:+.4e}' for v in a[wk])}", flush=True)
    print(f"({time.time() - t0:.0f} s for the double-double blocks)", flush=True)
    return 0


def coordscmd(argv):
    """weak and band coordinates V^T (z - z_base) of saved states against a stored SVD (no residual, so states on
    other step maps are allowed): BASE.npy then NAME=STATE.npy ..."""
    args = pos(argv)
    sd = np.load(args[0])
    s, Vt = sd["s"], sd["Vt"]
    zb = np.load(args[1])
    n = len(s)
    nul, weak, strong = bands_of(s)
    wk = list(weak[::-1])
    print(f"weak vectors (from the smallest): {[int(n - 1 - i) for i in wk]}", flush=True)
    for a in args[2:]:
        name, f = a.split("=", 1)
        z = np.load(f)
        dz = z - zb
        c = Vt @ dz
        nz = float(np.linalg.norm(dz))
        print(f"{name}: |dz| {nz:.4e}, dDelta {dz[-1]:+.4e}; fraction null {np.linalg.norm(c[nul]) / nz:.2e} weak "
              f"{np.linalg.norm(c[weak]) / nz:.6f} strong {np.linalg.norm(c[strong]) / nz:.2e}; weak coords "
              f"{' '.join(f'{v:+.4e}' for v in c[wk])}", flush=True)
    return 0


def detlinecmd(argv):
    """the determinant along the segment z(t) = z_A + t (z_B - z_A) between two states of one step map (A's record
    system): float64 slogdet of mode_matrix_fast at each t of --ts, and the hybrid determinant (d18_secant.RealEval,
    fast block 12) at each t of --hybrid; is ln|det| smooth in t?"""
    args, o = pos(argv), opts(argv)
    recf, sta, stb, D, lam = args[0], args[1], args[2], float(args[3]), float(args[4])
    S, za, rec, same, msg = load_system(recf, sta, D)
    print(msg, flush=True)
    if not same:
        return 3
    zb = np.load(stb)
    from ell_fastm import mode_matrix_fast
    print(f"|z_B - z_A| {np.linalg.norm(zb - za):.4e}, lambda {lam}", flush=True)
    for tt in [float(v) for v in o.get("ts", "0,0.25,0.5,0.75,1").split(",")]:
        z = za + tt * (zb - za)
        t0 = time.time()
        with np.errstate(all="ignore"):
            sg, ld = np.linalg.slogdet(mode_matrix_fast(S, z, lam))
        print(f"  t {tt:g}: float64 sign {sg:+.0f} ln|det| {ld:.6f}; |F| {np.linalg.norm(resid(S, z)):.4e} "
              f"({time.time() - t0:.0f} s)", flush=True)
    if "hybrid" in o:
        kmm = os.environ.get("D18_DIR", os.path.join(os.path.expanduser("~"), "Source/choptuik-kmm/solver/matter"))
        sys.path.insert(0, kmm)
        import d18_secant                                # noqa: E402
        for tt in [float(v) for v in o["hybrid"].split(",")]:
            z = za + tt * (zb - za)
            r = d18_secant.RealEval(S, z, rec, sta, "fast", 0.1, b12="fast")(lam)
            print(f"  t {tt:g}: hybrid sign {r['sign']:+.0f} ln|det| {r['ln']:.10f} dln {r['dln']:.2e} robust "
                  f"{r['robust']}; float64 {r['f64_sign']:+.0f} {r['f64_ln']:.6f} ({r['sec']} s)", flush=True)
    return 0


def detpartscmd(argv):
    """the hybrid determinant at LAM for z_A + P_band (z_B - z_A), band null, weak and strong of a stored SVD (the
    projection on that band's right vectors), beside z_A and z_B: which band of a state difference moves det M"""
    args, o = pos(argv), opts(argv)
    sd = np.load(args[0])
    s, Vt = sd["s"], sd["Vt"]
    recf, sta, stb, D, lam = args[1], args[2], args[3], float(args[4]), float(args[5])
    S, za, rec, same, msg = load_system(recf, sta, D)
    print(msg, flush=True)
    if not same:
        return 3
    zb = np.load(stb)
    kmm = os.environ.get("D18_DIR", os.path.join(os.path.expanduser("~"), "Source/choptuik-kmm/solver/matter"))
    sys.path.insert(0, kmm)
    import d18_secant                                    # noqa: E402
    nul, weak, strong = bands_of(s)
    dz = zb - za
    c = Vt @ dz
    cases = [("A", za)]
    for nm, idx in (("null", nul), ("weak", weak), ("strong", strong)):
        part = Vt[idx].T @ c[idx]
        print(f"  part {nm}: |P dz| {np.linalg.norm(part):.3e}", flush=True)
        cases.append((f"A + P_{nm}", za + part))
    cases.append(("B", zb))
    for nm, z in cases:
        r = d18_secant.RealEval(S, z, rec, sta, "fast", 0.1, b12=o.get("b12", "fast"),
                                b12_threads=int(o.get("b12_threads", 0)))(lam)
        print(f"  {nm}: hybrid sign {r['sign']:+.0f} ln|det| {r['ln']:.10f} dln {r['dln']:.2e} robust {r['robust']}; "
              f"|F| {np.linalg.norm(resid(S, z)):.4e} ({r['sec']} s)", flush=True)
    return 0


def floorcmd(argv):
    args = pos(argv)
    sd = np.load(args[1])
    U, s = sd["U"], sd["s"]
    recf, st, D = args[2], args[3], float(args[4])
    S, z, rec, same, msg = load_system(recf, st, D)
    print(msg, flush=True)
    if not same:
        return 3
    r = s / s[0]
    wk = np.where((r >= NULL_CUT) & (r < WEAK_CUT))[0][::-1]
    base_map = dict(S.steps_map)
    hk = (round(1 - S.u0, 9), round(S.nodes[-1], 9))

    def show(tag, S_):
        F = resid(S_, z)
        uw = U[:, wk].T @ F
        print(f"{tag}: |F| {np.linalg.norm(F):.4e}; U_w^T F {' '.join(f'{v:+.4e}' for v in uw)}", flush=True)
    show("record system", S)
    S.steps_map = {k: 2 * v for k, v in base_map.items()}
    show("every segment's steps x2", S)
    S.steps_map = dict(base_map)
    S.steps_map[hk] = 2 * base_map[hk]
    show("horizon segment's steps x2", S)
    S.steps_map = dict(base_map)
    from dss_ell import DSSEll, NODES_STD
    mk = lambda **kw: DSSEll(**dict(dict(kc=rec["kc"], nodes=NODES_STD, n=D - 2, ell=1, Delta_ref=z[-1], Ks=rec["Ks"],  # noqa
                                         u0=rec["u0"], rk=rec.get("rk", 1)), **kw))
    S2 = mk(Ks=45)
    S2.steps_map = dict(base_map)
    show("Ks 45", S2)
    S3 = mk(N=S.N + 56)
    S3.steps_map = dict(base_map)
    show(f"N {S.N + 56} (record {S.N})", S3)
    for u0 in (0.035, 0.045):
        S4 = mk(u0=u0)
        new = S4.freeze_from(z, rec["kappa"])
        hk4 = (round(1 - u0, 9), round(S4.nodes[-1], 9))
        S4.steps_map = {k: v for k, v in base_map.items() if k != hk}
        S4.steps_map[hk4] = new[hk4]
        show(f"u0 {u0} (horizon steps {new[hk4]}, record {base_map[hk]})", S4)
    return 0


def detsenscmd(argv):
    args, o = pos(argv), opts(argv)
    sd = np.load(args[0])
    s, Vt = sd["s"], sd["Vt"]
    recf, st, D, l1, l2 = args[1], args[2], float(args[3]), float(args[4]), float(args[5])
    kmm = os.environ.get("D18_DIR", os.path.join(os.path.expanduser("~"), "Source/choptuik-kmm/solver/matter"))
    sys.path.insert(0, kmm)
    import d18_secant                                    # noqa: E402
    S, z0, rec, same, msg = load_system(recf, st, D)
    print(msg, f"(d18_secant from {kmm})", flush=True)
    if not same:
        return 3
    n = len(s)
    cases = [("base", 0.0, None)]
    for a in args[6:]:
        name, v = a.split("=", 1)
        eps, idx = v.split(":")
        cases.append((name, float(eps), int(idx)))
    outf = o.get("out")
    res = []
    for name, eps, idx in cases:
        z = z0 if idx is None else z0 + eps * Vt[n - 1 - idx]
        E = d18_secant.RealEval(S, z, rec, st, "fast", 0.1)
        r1, r2 = E(l1), E(l2)
        f1, f2 = r1["sign"], r2["sign"] * np.exp(r2["ln"] - r1["ln"])
        root = l1 - f1 * (l2 - l1) / (f2 - f1) if r1["sign"] != r2["sign"] else None
        row = dict(name=name, eps=eps, index_from_smallest=idx, lam=[l1, l2], sign=[r1["sign"], r2["sign"]],
                   ln=[r1["ln"], r2["ln"]], robust=[r1["robust"], r2["robust"]], dln=[r1["dln"], r2["dln"]],
                   root_lin=root, F=float(np.linalg.norm(resid(S, z))), sec=[r1["sec"], r2["sec"]])
        res.append(row)
        print(json.dumps(row), flush=True)
        if outf:
            open(outf, "a").write(json.dumps(row) + "\n")
    b = res[0]["root_lin"]
    for r in res[1:]:
        if b is not None and r["root_lin"] is not None:
            print(f"{r['name']}: root shift {r['root_lin'] - b:+.4e} for eps {r['eps']:g} along v[{r['index_from_smallest']}]"
                  f": {(r['root_lin'] - b) / r['eps']:+.4e} per unit coordinate", flush=True)
    return 0


def locatecmd(argv):
    """float64 sign and ln|det| of M(lambda) (ell_fastm.mode_matrix_fast, slogdet) on a grid: where to bracket the hybrid
    root (the float64 root sat 1.4e-3 below the hybrid one at D 16, #1105)"""
    args = pos(argv)
    recf, st, D = args[0], args[1], float(args[2])
    lams = [float(v) for v in args[3].split(",")]
    from ell_fastm import mode_matrix_fast
    S, z, rec, same, msg = load_system(recf, st, D)
    print(msg, flush=True)
    if not same:
        return 3
    for lam in lams:
        t0 = time.time()
        with np.errstate(all="ignore"):
            sg, ld = np.linalg.slogdet(mode_matrix_fast(S, z, lam))
        print(f"  lam {lam:.6f}: float64 sign {sg:+.0f} ln|det| {ld:.6f} ({time.time() - t0:.0f} s)", flush=True)
    return 0


def seg_keys(S):
    """(key, block) per segment in the step map: 'log' is block 0, node i-1 -> i block i, the horizon segment block 12"""
    out = [("log", 0)]
    for i in range(1, len(S.nodes)):
        out.append(((round(S.nodes[i - 1], 9), round(S.nodes[i], 9)), i))
    out.append(((round(1 - S.u0, 9), round(S.nodes[-1], 9)), len(S.nodes)))
    return out


def segerrcmd(argv):
    """per segment: that segment's steps refined alone (freeze_from's count at kappa', the others at the record's), the
    change of its block's mismatch against the finest, and the order from successive refinements"""
    args, o = pos(argv), opts(argv)
    recf, st, D = args[0], args[1], float(args[2])
    kaps = [float(v) for v in o.get("kappas", "0.7,0.5,0.35,0.25,0.175,0.0875,0.04375").split(",")]
    S, z, rec, same, msg = load_system(recf, st, D)
    print(msg, flush=True)
    if not same:
        return 3
    base = dict(S.steps_map)
    maps = {k: dict(S.freeze_from(z, k)) for k in kaps}
    S.steps_map = dict(base)
    blk = 3 * S.n_odd + 2 * S.n_even
    svd = np.load(o["svd"]) if "svd" in o else None
    rows = []
    for key, b in seg_keys(S):
        Fs, ns = [], []
        for k in kaps:
            S.steps_map = dict(base)
            S.steps_map[key] = maps[k][key]
            F = resid(S, z)
            Fs.append(F)
            ns.append(maps[k][key])
        S.steps_map = dict(base)
        ref = Fs[-1]
        e = [float(np.linalg.norm((F - ref)[b * blk:(b + 1) * blk])) for F in Fs[:-1]]
        ew = None
        if svd is not None:
            s_ = svd["s"]
            r_ = s_ / s_[0]
            wk = np.where((r_ >= NULL_CUT) & (r_ < WEAK_CUT))[0][::-1]
            ew = [[float(v) for v in svd["U"][:, wk].T @ (F - ref)] for F in Fs[:-1]]
        ps = [float(np.log(e[i] / e[i + 1]) / np.log(ns[i + 1] / ns[i])) if e[i + 1] > 0 and e[i] > 0 else None
              for i in range(len(e) - 1)]
        row = dict(segment=str(key), block=b, steps=ns, err=e, order=ps, weak=ew)
        rows.append(row)
        print(f"segment {key} (block {b}): steps {ns}; |dF_block| vs finest {' '.join(f'{v:.3e}' for v in e)}; "
              f"local order {' '.join('-' if v is None else f'{v:.2f}' for v in ps)}", flush=True)
        if ew is not None:
            print(f"   U_w^T dF at kappa {kaps[0]}: {' '.join(f'{v:+.3e}' for v in ew[0])}", flush=True)
    if "out" in o:
        json.dump(dict(kappas=kaps, rows=rows), open(o["out"], "w"), indent=1)
    return 0


def chordcmd(argv):
    """the state re-solved on a modified step map by chord steps on a stored SVD's strong band (the weak and null
    coordinates kept, as LM keeps them): MAP is kappa=K (every segment at kappa K) or seg=I,J,..:K (those segments'
    steps at kappa K, block numbers as seg_keys, the others at the record's)"""
    args, o = pos(argv), opts(argv)
    sd = np.load(args[0])
    U, s, Vt = sd["U"], sd["s"], sd["Vt"]
    recf, st, D, mp, outp = args[1], args[2], float(args[3]), args[4], args[5]
    S, z, rec, same, msg = load_system(recf, st, D)
    print(msg, flush=True)
    if not same:
        return 3
    base = dict(S.steps_map)
    kind, val = mp.split("=", 1)
    if kind == "kappa":
        new = dict(S.freeze_from(z, float(val)))
    else:
        segs, k = val.split(":")
        segs = [int(v) for v in segs.split(",")]
        full = dict(S.freeze_from(z, float(k)))
        new = dict(base)
        for key, b in seg_keys(S):
            if b in segs:
                new[key] = full[key]
    S.steps_map = new
    strong = np.where(s / s[0] >= WEAK_CUT)[0]
    Us, ss, Vs = U[:, strong], s[strong], Vt[strong]
    hist = []
    best = None
    for it in range(int(o.get("maxit", 12))):
        F = resid(S, z)
        nF = float(np.linalg.norm(F))
        hist.append(nF)
        if best is None or nF < best[0]:
            best = (nF, z.copy())
        if it >= 2 and nF > 0.5 * hist[-2]:
            break
        z = z - Vs.T @ ((Us.T @ F) / ss)
    nF, z = best
    np.save(outp, z)
    meta = dict(D=D, kc=rec["kc"], kappa=rec["kappa"], u0=rec["u0"], Ks=rec["Ks"], rk=rec.get("rk", 1), map=mp,
                steps_map={str(k): int(v) for k, v in new.items()}, steps=int(sum(new.values())), F=nF,
                Delta=float(z[-1]), src=os.path.basename(st), chord_F=hist)
    json.dump(meta, open(outp.replace(".npy", ".map.json"), "w"), indent=1)
    print(f"{mp}: steps {meta['steps']} (record {sum(base.values())}); chord |F| {' '.join(f'{v:.3e}' for v in hist)};"
          f" kept {nF:.4e}; Delta {z[-1]:.13f} (from {np.load(st)[-1]:.13f}); saved {outp}", flush=True)
    return 0


def rootcmd(argv):
    """d18_secant's refinement (RealEval, Brent on the signed hybrid determinant) on a state saved by chord with its own
    step map (STATE.map.json); refused (exit 3) unless the state's |F| on that map equals the saved one to 1e-2"""
    args, o = pos(argv), opts(argv)
    st, lo, hi, outf = args[0], float(args[1]), float(args[2]), args[3]
    meta = json.load(open(st.replace(".npy", ".map.json")))
    kmm = os.environ.get("D18_DIR", os.path.join(os.path.expanduser("~"), "Source/choptuik-kmm/solver/matter"))
    sys.path.insert(0, kmm)
    import d18_secant                                    # noqa: E402
    from dss_ell import DSSEll, NODES_STD
    z = np.load(st)
    D = meta["D"]
    S = DSSEll(kc=meta["kc"], nodes=NODES_STD, n=D - 2, ell=1, Delta_ref=z[-1], Ks=meta["Ks"], u0=meta["u0"],
               rk=meta["rk"])
    S.freeze_from(z, meta["kappa"])
    keys = {str(k): k for k in S.steps_map}
    S.steps_map = {keys[k]: v for k, v in meta["steps_map"].items()}
    F = float(np.linalg.norm(resid(S, z)))
    same = abs(F - meta["F"]) <= 1e-2 * meta["F"]
    print(f"{os.path.basename(st)}: map {meta['map']}, {sum(S.steps_map.values())} steps, |F| {F:.10e} (saved "
          f"{meta['F']:.10e}): {'the saved system' if same else 'NOT the saved system'} (d18_secant from {kmm})",
          flush=True)
    if not same:
        return 3
    rec = dict(D=D, kc=meta["kc"], kappa=meta["kappa"], steps=meta["steps"], F=F)
    E = d18_secant.RealEval(S, z, rec, st, "fast", float(o.get("dln", 0.1)), b12=o.get("b12", "ddm"),
                            b12_threads=int(o.get("b12_threads", 0)))
    print(f"block 12 {E.b12}, tol {o.get('tol', 1e-7)}, bracket ({lo!r}, {hi!r})", flush=True)

    def out(r):
        print(f"  lam {r['lam']!r} ({r['role']}): hybrid {r['sign']:+.0f} ln|det| {r['ln']:.10f}, noise {r['noise']} "
              f"dln {r['dln'] if r['dln'] is None else format(r['dln'], '.2e')}, robust {r['robust']}; float64 "
              f"{r['f64_sign']:+.0f} ({r.get('sec')} s)", flush=True)
        open(outf, "a").write(json.dumps(r) + "\n")
    fin, recs = d18_secant.refine(E, lo, hi, float(o.get("tol", 1e-7)), int(o.get("maxev", 20)), out)
    fin.update(D=D, kc=meta["kc"], state=os.path.basename(st), map=meta["map"])
    open(outf, "a").write(json.dumps(fin) + "\n")
    print(f"root: status {fin['status']} ({fin['why'] or '-'}), bracket [{fin['bracket'][0]!r}, {fin['bracket'][1]!r}] "
          f"width {fin['width']:.3e}, signs {fin['signs']}, det-linear {fin['lam_lin']!r}, {fin['n_eval']} evaluations, "
          f"{fin['sec']} s", flush=True)
    return {"tol": 0, "maxev": 4, "nonrobust": 4}.get(fin["status"], 2)


def order_fit(ks, rs):
    """least-squares fit r = r0 + A k^p over p in [0.5, 8] (r0, A linear at each p): (p, r0, A, rms); and the p of
    each three-point ratio (r_i - r_i+1) / (r_i+1 - r_i+2) = (k_i^p - k_i+1^p) / (k_i+1^p - k_i+2^p), by bisection"""
    ks, rs = np.asarray(ks, float), np.asarray(rs, float)
    best = None
    for p in np.arange(0.5, 8.0001, 0.001):
        A = np.stack([np.ones_like(ks), ks ** p], axis=1)
        sol, *_ = np.linalg.lstsq(A, rs, rcond=None)
        rms = float(np.sqrt(np.mean((A @ sol - rs) ** 2)))
        if best is None or rms < best[3]:
            best = (float(p), float(sol[0]), float(sol[1]), rms)
    ratios = []
    for i in range(len(ks) - 2):
        a, b, c = ks[i:i + 3]
        rat = (rs[i] - rs[i + 1]) / (rs[i + 1] - rs[i + 2]) if rs[i + 1] != rs[i + 2] else np.inf
        g = lambda q: (a ** q - b ** q) / (b ** q - c ** q) - rat  # noqa: E731
        lo, hi = 0.05, 12.0
        pq = None
        if np.isfinite(rat) and g(lo) * g(hi) < 0:
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if g(lo) * g(mid) > 0 else (lo, mid)
            pq = 0.5 * (lo + hi)
        ratios.append(dict(kappas=[float(a), float(b), float(c)], ratio=float(rat), p=pq))
    return best, ratios


def ordercmd(argv):
    """the order of the root's kappa error from root records (d18_secant run output, kind root, lam_lin) at several
    kappa: KAPPA=FILE.jsonl ...; selftest: planted roots with p = 4 and p = 2 (noise 3e-5) must give p within 0.5 of
    the planted one and the right reading of the bars; --mutate=kappa reverses the kappa labels (must fail)"""
    args, o = pos(argv), opts(argv)
    if args and args[0] == "selftest":
        ok = True
        rng = np.random.default_rng(3)
        ks = [0.7, 0.5, 0.35, 0.25]
        for p0, A0 in ((4.0, 0.02), (2.0, 0.02), (2.5, -0.05)):
            rs = [2.25 + A0 * k ** p0 + 3e-5 * rng.standard_normal() for k in ks]
            kk = ks[::-1] if MUTATE == "kappa" else ks
            (p, r0, A, rms), rat = order_fit(kk, rs)
            good = abs(p - p0) <= 0.5
            ok &= good
            print(f"  planted p {p0} A {A0}: fit p {p:.3f} r0 {r0:.6f} A {A:+.5f} rms {rms:.1e}; ratios "
                  f"{[None if r['p'] is None else round(r['p'], 2) for r in rat]} {'ok' if good else 'FAIL'}", flush=True)
        print(f"order selftest {'PASS' if ok else 'FAIL'}{' (mutate=' + MUTATE + ')' if MUTATE else ''}", flush=True)
        return 0 if ok else 1
    ks, rs, bs = [], [], []
    for a in args:
        k, f = a.split("=", 1)
        rows = [json.loads(l) for l in open(f) if l.strip()]
        fin = [r for r in rows if r.get("kind") == "root"][-1]
        ks.append(float(k))
        rs.append(float(fin["lam_lin"]))
        bs.append(fin["bracket"])
        print(f"kappa {k}: root {fin['lam_lin']:.10f} status {fin['status']} bracket {fin['bracket']} width "
              f"{fin['width']:.2e} ({os.path.basename(f)})", flush=True)
    (p, r0, A, rms), rat = order_fit(ks, rs)
    print(f"fit r = r0 + A kappa^p: p {p:.3f}, r0 {r0:.8f}, A {A:+.6e}, rms {rms:.2e}; error at kappa {ks[0]}: "
          f"{A * ks[0] ** p:+.4e}", flush=True)
    for r in rat:
        print(f"  three-point ratio over kappa {r['kappas']}: {r['ratio']:.4f} -> p "
              f"{'none' if r['p'] is None else format(r['p'], '.3f')}", flush=True)
    span = abs(rs[-1] - rs[0])
    print(f"|root({ks[-1]}) - root({ks[0]})| = {span:.3e}: {'readable' if span >= 3e-3 else 'NOT readable'} "
          f"(bar 3e-3); reading: {'p >= 3.5: RK4 order' if p >= 3.5 else ('p <= 2.8: reduced order' if p <= 2.8 else 'mixed')}",
          flush=True)
    if "out" in o:
        json.dump(dict(kappas=ks, roots=rs, brackets=bs, p=p, r0=r0, A=A, rms=rms, ratios=rat, span=span),
                  open(o["out"], "w"), indent=1)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    f = dict(jac=jaccmd, svd=svdcmd, planted=plantedcmd, report=reportcmd, valley=valleycmd, floor=floorcmd,
             detsens=detsenscmd, locate=locatecmd, segerr=segerrcmd, chord=chordcmd, root=rootcmd,
             order=ordercmd, ne=necmd, ddres=ddrescmd, coords=coordscmd, detline=detlinecmd,
             detparts=detpartscmd).get(cmd)
    if f is None:
        print(__doc__)
        sys.exit(2)
    sys.exit(f(sys.argv[2:]))
