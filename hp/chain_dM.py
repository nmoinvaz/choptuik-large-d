#!/usr/bin/env python3
"""The chain's truncation in tau (#636): a chain state at M = 32, resampled to M = 48 and its pinned system solved
again at its own D, gives Delta(M 48) - Delta(M 32) and lam_g(M 48) - lam_g(M 32).

The system is the chain's, unchanged except for M: the same nodes, Taylor orders and kfreeze, p1's harmonics in
(kfreeze, 13] held at the state's own values (the chain holds them on the fit at that D, hp/anm_run.py --held_fit, and
a converged state reproduces those values to its residual; band_coeffs exact, from all its parts), its harmonics above
13 at zero, and the Taylor plan made at the start at 1e-32 with steps at most 0.05, as the chain plans a prediction. A
state that stores its plan (I603) keeps it at M = 32, as the chain does. The corrector is the chain's quad-double one
(anm_run.py --corrector qd --reseed 2 --reseed_q 1): the horizon data re-seeded from the last node (hp/reseed.py), then
hpkernel.driver.newton_qd on the system with the horizon data eliminated (reseed.ElimQ), the final iterate projected.
The quad-double kernel is built for M = 32 and 48 only and the octo-double one for M = 32 only, so M = 48 is solved in
quad-double at every D, also where the chain used octo-double (D 207.9 on).

The resampling is trigonometric interpolation (resample_z's, hpkernel.driver) in mpmath on the exact sum of the state's
parts, with p1's harmonics above 13 removed, and split into four float64 parts. Without the re-seed the start's
horizon data carry the truncation times the mass mode's gain, 2e15 in |F| at D 200 (notes/BIAS.md, Resolution).

Each state is solved at M = 32 by the same path (the same code, the stored state as the start), so the difference is
taken between two solutions of one corrector; the stored state's own Delta is recorded beside it. A solution is
refused, with no difference reported and exit status 1, when its pinned quad-double max |F| exceeds --refuse (1e-20,
the acceptance of #636). The corrector stops at --tol (1e-30), and plain Newton on the pinned system then polishes the
solution to the floor (--floor_tol 1e-45, at most --polish_it 2 steps, 4 in the runs at D 80-150, whose steps past the
second moved Delta by 1e-50 or less; kept unless it raises max |F| tenfold): at D 80
a stop at |F| 5.7e-31 left one more step of 3.3e-25 in Delta. The solution is converged when max |F| <= --tol; in
between it is accepted at the quad-double floor. Either way the record carries est, the Delta component of one more
Newton step at the solution (gamma_dd.py's floor check), as that solution's error estimate. The record holds both solutions' spectra in tau (largest |c_k| per field kind), from
which the decay with k, and so with M, can be read.

    cd hp && python3 chain_dM.py run RAW OUTDIR STATE [STATE ...] [--Mhi=48] [--threads=4] [--it=16] [--tol=1e-30]
        [--refuse=1e-20] [--reseed=2] [--kfreeze=none|K] [--m32=1] [--save=1] [--mutate=unconverged|D|held]
        [--floor_tol=1e-45] [--polish_it=2] [--halvings=8]
    cd hp && python3 chain_dM.py planted RAW OUTDIR [--threads=4] [--only=NAME,...] [--mutate=D|held] [--records=JSONL]
    cd hp && python3 chain_dM.py refit OUTDIR RECORDS.jsonl [...]     lam_gauge_fit.py on the table with M = 48 values
    cd hp && python3 chain_dM.py table RECORDS.jsonl [...] [--out=data/chain_dM.csv] [--raw=COMMIT]   the CSV
    cd hp && python3 chain_dM.py ramp [--out=data/chain_dM_ramp.csv]   the free c0's response to a smooth chain error

RAW is a checkout of the raw-data repository; STATE a path in it, or in this repository with the prefix repo:. Records
go to OUTDIR/chain_dM.jsonl and the M = 48 solutions to OUTDIR/m48_<state>.npz (save_q; meta M 48 and the plan).

Controls (each must exit 1):
  --mutate=unconverged  the M = 48 corrector takes no step, so the resample stays unconverged and must be refused.
  --mutate=D            the M = 48 system at D (1 + 1e-14), a different system: planted must fail.
  --mutate=held         the M = 48 system with harmonic 5's cosine held at (1 + 1e-10) times its value: planted must
                        fail on the pinned state (the D 80 plant).

planted (criteria fixed before the runs, 2026-11-13): the same path on states whose M = 48 solution is known.
  dd20, dd40, dd58: the double-double states up2_D20.0437 (no cut), up7_D40.0201 and up12_D58.0060 (kfreeze 13), whose
    M = 48 solutions gamma_dd.py --M48 recorded (raw I271-macm3/gamma_dd.jsonl): our Delta(M 48) must lie within
    3 |est| + 1e-25 of the record's (est the record's floor estimate at its M = 48 solution).
  chain80: the chain_fit state T138-win2/chain/anm_pin3_D80.0000, whose Delta(M 48) - Delta(stored M 32) is -2.376e-25
    by resolution_check.py (no re-seed, junk-mode Newton, float held values, raw resolution-mac/scan_chainfit_D80.log,
    printed to 4 digits): ours must lie within 1e-27 of it. This plant runs the re-seed and the elimination (the dd
    plants ran with --reseed=0: at D 40 one re-seed took 431 s). Amended after its run: that value is the kernel of
    2026-11-07's. Rerun on this build, the same script gives -2.309e-25 (raw I636-mac/planted/
    resolution_check_D80_thisbuild.log), which is the plant's value now, at the same tolerance (PLANTED below).
  --records=FILE grades the records a run already wrote instead of solving again.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import make_solver, newton_qd, qmax, save_q  # noqa: E402
from pinning import PinnedQ, PinnedSolver, band_coeffs, band_field  # noqa: E402
from raw_out import require_raw  # noqa: E402
import lam_gauge  # noqa: E402

mp.mp.dps = 50            # the comparisons' precision (reseed.py sets the same on import; a bare grade did not import it)
PARTS = ("zh", "zl", "z2", "z3", "z4", "z5", "z6", "z7")
DPS = 90                 # digits of the resampling: the four quad-double parts span ~64 digits
P1_CUT = 13               # p1's largest held harmonic; the chain's p1 is zero above it
REPO = os.path.dirname(HERE)

# the plants (docstring): name, state, kfreeze, and the est of the record's M = 48 solution or the known difference
PLANTED = [
    dict(name="dd20", state="repo:hp/data/states/up2_D20.0437.npz", kfreeze=None, est=-4.944299673096901e-27,
         record="I271-macm3/gamma_dd.jsonl D 20.043653"),
    dict(name="dd40", state="repo:hp/data/states/up7_D40.0201.npz", kfreeze=13, est=-3.816369468847703e-22,
         record="I271-macm3/gamma_dd.jsonl D 40.020065"),
    dict(name="dd58", state="repo:hp/data/states/up12_D58.0060.npz", kfreeze=13, est=2.037251959259797e-18,
         record="I271-macm3/gamma_dd.jsonl D 58.00599"),
    # the known value on this build: resolution_check.py rerun (I636-mac/planted/resolution_check_D80_thisbuild.log).
    # Its first run, on the kernel of 2026-11-07 (resolution-mac/scan_chainfit_D80.log), gave -2.376e-25, and the same
    # script finds the stored M 32 state at |F| 1.0e-35 there and 2.9e-34 here: the kernel's system moved by 3e-27 in
    # Delta at M 32 between the builds (the exact freeze projector, #523), so that value is not this build's
    dict(name="chain80", state="T138-win2/chain/anm_pin3_D80.0000.npz", kfreeze=3, dM=-2.309e-25, tol=1e-27,
         record="I636-mac/planted/resolution_check_D80_thisbuild.log"),
]


def opts_of(argv):
    return dict((a[2:] + "=1").split("=")[:2] if "=" not in a else a[2:].split("=", 1) for a in argv if a.startswith("--"))


def resolve(raw, st):
    return os.path.join(REPO, st[5:]) if st.startswith("repo:") else os.path.join(raw, st)


def load_parts(path):
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    return [np.asarray(d[k], dtype=float) for k in PARTS if k in d.files], meta


def mp_sum(parts):
    """the parts summed entry by entry, exactly at DPS digits (eight parts span at most ~130)"""
    with mp.workdps(max(DPS, 140)):
        v = [mp.mpf(0)] * len(parts[0])
        for p in parts:
            v = [a + mp.mpf(float(x)) for a, x in zip(v, p)]
    return v


def split4(v):
    """an mpmath vector as four float64 parts, each the rounding of what is left"""
    out, r = [], list(v)
    with mp.workdps(max(DPS, 140)):
        for _ in range(4):
            p = np.array([float(x) for x in r])
            out.append(np.ascontiguousarray(p))
            r = [x - mp.mpf(float(y)) for x, y in zip(r, p)]
    return tuple(out)


def parities(K):
    """odd/even about theta -> theta + pi of the 4(K + 1) field blocks (resample_z's list)"""
    return ["odd", "even"] + ["odd", "odd", "even", "even"] * K + ["odd", "even"]


def kinds(K):
    return ["p1", "c0"] + ["p", "m", "a", "c"] * K + ["m_h", "a_h"]


_TAB = {}


def table(M, kmax):
    """cos and sin of k theta_j, theta_j = pi j / M, k = 0..kmax, at DPS digits"""
    key = (M, kmax)
    if key not in _TAB:
        with mp.workdps(DPS):
            _TAB[key] = ([[mp.cos(mp.pi * k * j / M) for j in range(M)] for k in range(kmax + 1)],
                         [[mp.sin(mp.pi * k * j / M) for j in range(M)] for k in range(kmax + 1)])
    return _TAB[key]


def coeffs(f, M, odd):
    """(a_k, b_k), k = 0..M, of a field of the half grid (f(theta + pi) = -f(theta) if odd, +f if even):
    f = a_0 + sum a_k cos k theta + b_k sin k theta, the Nyquist term a_M cos M theta (b_M undetermined, 0)"""
    C, S = table(M, M)
    out = []
    with mp.workdps(DPS):
        for k in range(M + 1):
            if (k % 2 == 1) != odd:
                out.append((mp.mpf(0), mp.mpf(0)))
                continue
            w = mp.mpf(1) / M if k in (0, M) else mp.mpf(2) / M
            a = w * mp.fsum(x * c for x, c in zip(f, C[k]))
            b = mp.mpf(0) if k in (0, M) else w * mp.fsum(x * s for x, s in zip(f, S[k]))
            out.append((a, b))
    return out


def synth(cf, M):
    """the samples on the half grid of M points of sum_k a_k cos k theta + b_k sin k theta"""
    C, S = table(M, len(cf) - 1)
    with mp.workdps(DPS):
        return [mp.fsum(a * C[k][j] + b * S[k][j] for k, (a, b) in enumerate(cf)) for j in range(M)]


def resample(v, M0, M1, K, cut=None):
    """v (mpmath, the hp layout at M0) on M1 points: each block's coefficients up to min(M0, M1), p1's above cut zeroed;
    Delta kept"""
    out = []
    for j, par in enumerate(parities(K)):
        cf = coeffs(v[j * M0:(j + 1) * M0], M0, par == "odd")[:min(M0, M1) + 1]
        if j == 0 and cut is not None:
            cf = [c if k <= cut else (mp.mpf(0), mp.mpf(0)) for k, c in enumerate(cf)]
        out += synth(cf, M1)
    return out + list(v[4 * (K + 1) * M0:])


def spectrum(v, M, K):
    """per field kind, the largest |c_k| = sqrt(a_k^2 + b_k^2) over its blocks, k = 0..M (floats)"""
    sp = {}
    for j, (kind, par) in enumerate(zip(kinds(K), parities(K))):
        cf = coeffs(v[j * M:(j + 1) * M], M, par == "odd")
        mags = [float(mp.sqrt(a * a + b * b)) for a, b in cf]
        sp[kind] = [max(x, y) for x, y in zip(sp[kind], mags)] if kind in sp else mags
    return sp


def tail_of(sp, k0):
    """the largest |c_k| with k >= k0 over every field kind"""
    return max(max(m[k0:]) if len(m) > k0 else 0.0 for m in sp.values())


def held_of(parts, M, ks):
    """the held harmonics' (a_k, b_k) of the state's p1, exact (mpf) from all its parts"""
    if not ks:
        return []
    return band_coeffs(parts[0][:M], parts[1][:M], M, ks, exact=True, rest=[p[:M] for p in parts[2:]])


def system(D, M, meta, kf, threads, z4, held, ks, segs=None, mutate=""):
    """the chain's solver at D, M (its plan, made at z4 at 1e-32 unless given) and its pinned quad-double wrapper"""
    Dx = D * (1 + mp.mpf("1e-14")) if mutate == "D" else D
    s = make_solver(Dx, M, nodes=list(meta["nodes"]), kt=meta.get("kt", 36), kc=meta.get("kc", 146),
                    ks=meta.get("ks", 130), threads=threads, kfreeze=kf)
    if segs is None:
        segs = s.plan(np.asarray(z4[0]), np.asarray(z4[1]), 1e-32, 0.05)
    s.set_segs(segs)
    if ks:
        h = [list(c) for c in held]
        if mutate == "held":
            h[0][0] = h[0][0] * (1 + mp.mpf("1e-10"))          # harmonic 5's cosine
        hh, hl = band_field(h, ks, M)
    else:
        hh, hl = np.zeros(M), np.zeros(M)
    sp = PinnedSolver(s, M, hh, hl)
    return s, PinnedQ(s, sp.ch, sp.cl), [[float(x) for x in sg] for sg in segs]


def converge(D, M, meta, kf, threads, z4, held, ks, segs, opt, mutate=""):
    """the chain's corrector at D, M from z4: re-seed, newton_qd on ElimQ, the projection. Returns (z, rec)"""
    from reseed import ElimQ, Reseeder, StartNotProjected
    t0 = time.time()
    s, sq, segs = system(D, M, meta, kf, threads, z4, held, ks, segs, mutate)
    rec = dict(M=M, F_start=qmax(sq.residual_q(*z4)), plan_steps=sum(len(sg) - 1 for sg in segs), segs=segs)
    print(f"  M {M}: start max|F| {rec['F_start']:.2e} (plan {rec['plan_steps']} steps)", flush=True)
    rs = None
    reseed = int(opt.get("reseed", 2))
    if reseed:
        rs = Reseeder(dict(M=M, nodes=list(meta["nodes"]), kt=meta.get("kt", 36), kc=meta.get("kc", 146),
                           ks=meta.get("ks", 130)), D * (1 + mp.mpf("1e-14")) if mutate == "D" else D,
                      threads=threads, qd=True)
        zr, diag = rs.reseed(z4)
        rec["reseed"] = dict(ok=diag["ok"], seconds=diag.get("seconds"), dropped=diag["dropped"],
                             rungs=[dict(u=r.get("u"), qd=r.get("qd", False), conv=r.get("conv"), R=r.get("R"))
                                    for r in diag["rungs"]])
        if diag["ok"]:
            z4 = zr
        rec["F_reseeded"] = qmax(sq.residual_q(*z4))
        print(f"  M {M}: re-seed {'ok' if diag['ok'] else 'FAILED (start kept)'}, max|F| {rec['F_reseeded']:.2e}, "
              f"{diag.get('seconds')} s", flush=True)
    corr = ElimQ(sq, rs) if (reseed == 2 and rs is not None) else sq
    it = 0 if mutate == "unconverged" else int(opt.get("it", 16))
    try:
        z, hist, conv = newton_qd(corr, tuple(np.ascontiguousarray(p) for p in z4), tol=float(opt.get("tol", 1e-30)),
                                  it_max=it, verbose=True, max_halvings=int(opt.get("halvings", 8)))
    except StartNotProjected as e:
        rec.update(start_not_projected=e.args[0], F=rec.get("F_reseeded", rec["F_start"]), conv=False)
        return z4, rec, s, sq
    if isinstance(corr, ElimQ):
        rec["elim"] = dict(corr.counts)
        rec["F_unprojected"] = qmax(sq.residual_q(*z))
        z = rs.project(z)[0]                    # the point whose residual converged (anm_run.py does the same)
    z = tuple(np.ascontiguousarray(np.asarray(p)) for p in z)
    Fz = qmax(sq.residual_q(*z))
    # the polish to the floor: plain Newton on the pinned system to --floor_tol (1e-45), at most --polish_it steps, kept
    # unless it raises max|F| more than tenfold. Stopping at --tol left one more step of 3.3e-25 in Delta at D 80, M 48
    # (|F| 5.7e-31), which the plant chain80 caught
    if it and Fz <= float(opt.get("refuse", 1e-20)) and int(opt.get("polish_it", 2)):
        zp, hp_, _ = newton_qd(sq, z, tol=float(opt.get("floor_tol", 1e-45)), it_max=int(opt.get("polish_it", 2)), verbose=True,
                               max_halvings=int(opt.get("halvings", 8)))
        zp = tuple(np.ascontiguousarray(np.asarray(p)) for p in zp)
        Fp = qmax(sq.residual_q(*zp))
        rec["polish"] = dict(F_before=Fz, F_after=Fp, steps=len(hp_), kept=bool(Fp <= 10 * Fz))
        print(f"  M {M}: polish to the floor, max|F| {Fz:.2e} -> {Fp:.2e} in {len(hp_)} steps, "
              f"{'kept' if Fp <= 10 * Fz else 'not kept'}", flush=True)
        if Fp <= 10 * Fz:
            z, Fz, hist = zp, Fp, hist + hp_
    conv = Fz <= float(opt.get("tol", 1e-30))
    rec.update(steps=len(hist), conv=bool(conv), F=Fz,
               hist=[dict(F=h.get("F"), lam=h.get("lam"), dDelta=h.get("dDelta"), mode=h.get("mode")) for h in hist])
    st = sq.newton_q(*z)                        # one more Newton step at the solution: its Delta component
    rec["est"] = float(sum(mp.mpf(float(np.asarray(p)[-1])) for p in st[:4]))
    rec["pivots"] = [float(st[6]), float(st[7])]
    # the plan's dependence: max |F| at the solution with a plan made there
    s2 = make_solver(D, M, nodes=list(meta["nodes"]), kt=meta.get("kt", 36), kc=meta.get("kc", 146), ks=meta.get("ks", 130),
                     threads=threads, kfreeze=kf)
    s2.set_segs(s2.plan(np.asarray(z[0]), np.asarray(z[1]), 1e-32, 0.05))
    rec["F_fresh_plan"] = qmax(PinnedQ(s2, sq.ch, sq.cl).residual_q(*z))
    rec["seconds"] = round(time.time() - t0, 1)
    print(f"  M {M}: {'converged' if conv else 'not converged'} in {len(hist)} steps, max|F| {rec['F']:.2e}, est {rec['est']:+.1e}, "
          f"pivots {rec['pivots'][0]:.1e}..{rec['pivots'][1]:.1e}, F with a fresh plan {rec['F_fresh_plan']:.1e}, {rec['seconds']} s",
          flush=True)
    return z, rec, s, sq


def values(parts, meta):
    """Delta and lam_g of a state (every part), at lam_gauge's precision"""
    with mp.workprec(lam_gauge.PREC):
        return lam_gauge.delta(parts), lam_gauge.lam_g(parts, meta)


def measure(raw, out, st, opt, mutate="", kf_override="unset"):
    """one state: M 32 by the path, M Mhi by the path; returns the record and the exit status"""
    path = resolve(raw, st)
    parts, meta = load_parts(path)
    D = mp.mpf(meta["D_str"]) if "D_str" in meta else mp.mpf(float(meta["D"]))
    M0, K = int(meta["M"]), len(meta["nodes"])
    kf = meta.get("kfreeze") if kf_override == "unset" else kf_override
    ks = [k for k in range(kf + 1, P1_CUT + 1) if k % 2 == 1] if kf else []
    cut = P1_CUT if kf else None
    Mhi = int(opt.get("Mhi", 48))
    threads = int(opt.get("threads", 4))
    refuse = float(opt.get("refuse", 1e-20))
    tol = float(opt.get("tol", 1e-30))
    print(f"{st}: D {mp.nstr(D, 15)}, M {M0}, K {K}, {len(parts)} parts, kfreeze {kf}, held {ks}"
          + (f", mutation {mutate}" if mutate else ""), flush=True)
    t0 = time.time()
    held = held_of(parts, M0, ks)
    v0 = mp_sum(parts)
    D32s, L32s = values(parts, meta)
    rec = dict(state=st, D=float(D), D_str=str(D), M0=M0, K=K, parts=len(parts), kfreeze=kf, held_ks=ks,
               held=[[mp.nstr(a, 40), mp.nstr(b, 40)] for a, b in held], Mhi=Mhi, tol=tol, refuse=refuse,
               mutate=mutate or None, reseed=int(opt.get("reseed", 2)),
               Delta_stored=mp.nstr(D32s, 40), lam_stored=mp.nstr(L32s, 40))
    rec["spectrum_32"] = spectrum(v0, M0, K)
    # the resampling's check: M0 -> Mhi -> M0 without the cut returns the state to 1e-60 relative
    with mp.workdps(DPS):
        back = resample(resample(v0, M0, Mhi, K), Mhi, M0, K)
        rt = float(max(abs(a - b) for a, b in zip(back, v0)) / max(abs(x) for x in v0))
    rec["roundtrip"] = rt
    print(f"  round trip {M0} -> {Mhi} -> {M0}: {rt:.1e} relative ({'pass' if rt <= 1e-60 else 'FAIL'})", flush=True)
    if rt > 1e-60:
        return rec, 1
    sols = {}
    status = 0
    for M in ([M0] if int(opt.get("m32", 1)) else []) + [Mhi]:
        if M == M0:
            z4 = split4(v0)
            segs = meta.get("segs")              # a stored plan is kept (anm_run.py --keep_plan 1)
        else:
            z4 = split4(resample(v0, M0, M, K, cut))
            segs = None
        z, r, s, sq = converge(D, M, meta, kf, threads, z4, held, ks, segs, opt, mutate if M != M0 else "")
        mM = dict(meta, M=M, segs=r["segs"])
        r["refused"] = not (r["F"] <= refuse)
        r["at_floor"] = (not r["refused"]) and r["F"] > tol
        if not r["refused"]:
            Dz, Lz = values(list(z), mM)
            r.update(Delta=mp.nstr(Dz, 40), lam=mp.nstr(Lz, 40))
            sols[M] = (Dz, Lz, z)
            if M != M0:
                r["spectrum"] = spectrum(mp_sum(list(z)), M, K)
                if int(opt.get("save", 1)):
                    p = os.path.join(out, f"m{M}_{os.path.basename(path)}")
                    save_q(p, z, dict(mM, source=st, M_from=M0, held=rec["held"], F=r["F"], est=r["est"]))
                    r["saved"] = os.path.basename(p)
        else:
            status = 1
            print(f"  M {M}: REFUSED, max|F| {r['F']:.2e} > {refuse:.0e}: no difference is reported", flush=True)
        r.pop("segs")
        rec[f"M{M}"] = r
    if Mhi in sols:
        D48, L48, _ = sols[Mhi]
        rec["dDelta_stored"] = float(D48 - D32s)
        rec["dlam_stored"] = float(L48 - L32s)
        if M0 in sols:
            D32, L32, _ = sols[M0]
            rec.update(dDelta=float(D48 - D32), dlam=float(L48 - L32), M32_moved=float(D32 - D32s),
                       M32_lam_moved=float(L32 - L32s))
        rec["tail_32_from_28"] = tail_of(rec["spectrum_32"], 28)
        rec[f"tail_{Mhi}_from_32"] = tail_of(rec[f"M{Mhi}"]["spectrum"], 32)
        print(f"  Delta(M {Mhi}) - Delta(M {M0}) {rec.get('dDelta', float('nan')):+.3e} (against the stored state "
              f"{rec['dDelta_stored']:+.3e}); lam_g {rec.get('dlam', float('nan')):+.3e} (stored {rec['dlam_stored']:+.3e}); "
              f"est at M {Mhi} {rec[f'M{Mhi}']['est']:+.1e}; tau tails: M {M0} |c_k>=28| {rec['tail_32_from_28']:.1e}, "
              f"M {Mhi} |c_k>=32| {rec[f'tail_{Mhi}_from_32']:.1e}", flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "chain_dM.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec, status


def cmd_run(argv):
    args = [a for a in argv if not a.startswith("--")]
    opt = opts_of(argv)
    raw, out, states = args[0], args[1], args[2:]
    os.makedirs(out, exist_ok=True)
    require_raw(os.path.join(out, "chain_dM.jsonl"))
    kf = opt.get("kfreeze", "unset")
    kf = "unset" if kf == "unset" else (None if kf == "none" else int(kf))
    status = 0
    for st in states:
        _, s_ = measure(raw, out, st, opt, opt.get("mutate", ""), kf)
        status |= s_
    return status


def cmd_planted(argv):
    args = [a for a in argv if not a.startswith("--")]
    opt = opts_of(argv)
    raw, out = args[0], args[1]
    os.makedirs(out, exist_ok=True)
    require_raw(os.path.join(out, "chain_dM.jsonl"))
    mutate = opt.get("mutate", "")
    only = opt["only"].split(",") if "only" in opt else None
    # --records=FILE grades the last record of each plant's state in FILE (a chain_dM.jsonl this command wrote, with
    # the polish to the floor, unmutated unless --mutate names its mutation) instead of solving again
    stored = {}
    if "records" in opt:
        for line in open(opt["records"]):
            r = json.loads(line)
            hi = r.get(f"M{r['Mhi']}", {})
            if (r.get("mutate") or "") == mutate and ("polish" in hi or hi.get("F", 1.0) <= 1e-45):
                stored[r["state"]] = r
    ok_all = True
    for pl in PLANTED:
        if only and pl["name"] not in only:
            continue
        if "records" in opt:
            if pl["state"] not in stored:
                print(f"planted {pl['name']}: no record in {opt['records']}", flush=True)
                ok_all = False
                continue
            rec, status = stored[pl["state"]], 0
            print(f"planted {pl['name']}: graded from the record of {opt['records']}", flush=True)
        else:
            rec, status = measure(raw, out, pl["state"], opt, mutate, pl["kfreeze"])
        m48 = rec.get(f"M{rec['Mhi']}", {})
        if status or "Delta" not in m48:
            ok = False
            why = "no accepted M 48 solution"
        elif "dM" in pl:
            got = rec["dDelta_stored"]
            ok = abs(got - pl["dM"]) <= pl["tol"]
            why = f"dDelta against the stored state {got:+.4e}, known {pl['dM']:+.4e}, tolerance {pl['tol']:.0e}"
        else:
            kn = known_D48(raw, pl)
            got = mp.mpf(m48["Delta"])
            tol = 3 * abs(pl["est"]) + 1e-25
            ok = abs(float(got - kn)) <= tol
            why = f"Delta(M 48) - the record's {float(got - kn):+.3e}, tolerance {tol:.1e}"
        print(f"planted {pl['name']}: {'pass' if ok else 'FAIL'} ({why}; record {pl['record']})", flush=True)
        ok_all &= ok
    print(f"planted: {'PASS' if ok_all else 'FAIL'}" + (f" (mutation {mutate})" if mutate else ""), flush=True)
    return 0 if ok_all else 1


def known_D48(raw, pl):
    """the M = 48 Delta of a gamma_dd.py --M48 record (raw I271-macm3/gamma_dd.jsonl), at the plant's D"""
    f, Dk = pl["record"].split(" D ")
    for line in open(os.path.join(raw, f)):
        r = json.loads(line)
        if abs(r["D"] - float(Dk)) < 1e-9:
            hi = r["M48"]
            return mp.mpf(r["Delta"]) + mp.mpf(r["Delta_lo"]) + mp.mpf(hi["Delta_diff"])
    raise SystemExit(f"no record at D {Dk} in {f}")


def measured(paths):
    """{D_str: record} of the accepted, unmutated records from chain_dM.jsonl files, with dlam set: M 48 - M 32 by this
    path, or against the stored state where M 32 was not solved (--m32=0)"""
    out = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r.get("mutate") or "dlam_stored" not in r:
                continue
            r.setdefault("dlam", r["dlam_stored"])
            out[r["D_str"]] = r
    return out


def cmd_refit(argv):
    """lam_gauge_fit.py fit and planted --mutate=limit on hp/data/lam_gauge.csv with the measured M = 48 values: 'swap'
    puts lam_g(M 48) in place of the backbone's value at each measured D; 'shift' adds dlam (M 48 - M 32, both by
    this path) to every chain row from D 73 on, linear in D between the measured D and constant past the ends. The
    tables are written to OUTDIR (lam_gauge_swap.csv, lam_gauge_shift.csv), the fits' output beside them."""
    import csv
    import subprocess
    args = [a for a in argv if not a.startswith("--")]
    out, recs = args[0], measured(args[1:])
    if not recs:
        raise SystemExit("refit: no accepted record")
    pts = sorted((float(r["D"]), r) for r in recs.values())
    src = os.path.join(HERE, "data", "lam_gauge.csv")
    lines = open(src).read().splitlines()
    head = [x for x in lines if x.startswith("#")]
    rows = list(csv.DictReader(x for x in lines if not x.startswith("#")))
    cols = list(rows[0].keys())

    def dlam_at(D):
        if D <= pts[0][0]:
            return mp.mpf(pts[0][1]["dlam"])
        if D >= pts[-1][0]:
            return mp.mpf(pts[-1][1]["dlam"])
        for (d0, r0), (d1, r1) in zip(pts, pts[1:]):
            if d0 <= D <= d1:
                w = (D - d0) / (d1 - d0)
                return (1 - w) * mp.mpf(r0["dlam"]) + w * mp.mpf(r1["dlam"])

    status = 0
    for mode in ("swap", "shift"):
        mp.mp.dps = 50
        n = 0
        new = []
        for r in rows:
            r = dict(r)
            key = r["D_str"] or r["D"]
            if r["role"] == "backbone":
                if mode == "swap" and key in recs:
                    r["lam_g"] = recs[key][f"M{recs[key]['Mhi']}"]["lam"]
                    n += 1
                elif mode == "shift" and r["kind"] == "chain" and float(r["D"]) >= 73:
                    r["lam_g"] = mp.nstr(mp.mpf(r["lam_g"]) + dlam_at(float(r["D"])), 40)
                    n += 1
            new.append(r)
        p = os.path.join(out, f"lam_gauge_{mode}.csv")
        with open(p, "w", newline="") as fh:
            fh.write("\n".join(head + [f"# {mode}: chain_dM.py refit, {n} rows changed"]) + "\n")
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(new)
        print(f"{mode}: {n} backbone rows changed, {p}", flush=True)
        for sub, extra, want in (("fit", [], 0), ("planted", ["--mutate=limit"], 1)):
            r = subprocess.run([sys.executable, os.path.join(HERE, "lam_gauge_fit.py"), sub, f"--csv={p}"] + extra,
                               capture_output=True, text=True)
            with open(os.path.join(out, f"lam_gauge_fit_{mode}_{sub}{'_limit' if extra else ''}.log"), "w") as fh:
                fh.write(r.stdout + r.stderr + f"exit {r.returncode}\n")
            last = [x for x in r.stdout.splitlines() if x.startswith(("c0 = 1", "planted:", "  free c0", "  c0=1 c1"))]
            print(f"  {sub} {' '.join(extra)}: exit {r.returncode} (wanted {want}); " + " | ".join(last), flush=True)
            if r.returncode != want:
                status = 1
    return status


def cmd_ramp(argv):
    """the scale of #622's discrepancy: lam_gauge_fit.py's free fits of the backbone (hp/data/lam_gauge.csv) with
    lam_g + eps (D - 73) / (245 - 73) on the chain (D >= 73), eps = 0 and +-1e-12. Writes hp/data/chain_dM_ramp.csv
    (eps, the free c0's median, lowest and highest)."""
    import csv
    import lam_gauge_fit as LF
    opt = opts_of(argv)
    out = opt.get("out", os.path.join(HERE, "data", "chain_dM_ramp.csv"))
    with mp.workdps(40):
        pts = LF.backbone()
        rows = []
        for eps in ("0", "1e-12", "-1e-12"):
            e = mp.mpf(eps)
            p = [(D, y + (e * (D - 73) / (245 - 73) if D >= 73 else 0)) for D, y in pts]
            med, lo, hi = LF.summary(LF.fits(p), 0)
            rows.append([eps, mp.nstr(med, 15), mp.nstr(lo, 15), mp.nstr(hi, 15)])
            print(f"ramp {eps}: free c0 {mp.nstr(med, 13)} [{mp.nstr(lo, 13)}, {mp.nstr(hi, 13)}], 1 - highest "
                  f"{mp.nstr(1 - hi, 3)}", flush=True)
    with open(out, "w", newline="") as fh:
        fh.write("# hp/chain_dM.py ramp: lam_gauge_fit.py's free fits with lam_g + eps (D - 73)/(245 - 73) on the chain (#636)\n")
        w = csv.writer(fh)
        w.writerow(["eps", "c0_median", "c0_lowest", "c0_highest"])
        w.writerows(rows)
    return 0


TABLE_COLS = ["role", "rule", "mutate", "D", "D_str", "state", "kfreeze", "parts", "F32", "F48", "conv48", "est32", "est48", "dDelta", "dlam",
              "dDelta_stored", "dlam_stored", "M32_moved", "F48_fresh_plan", "F48_start", "F48_reseeded", "tail32_k28",
              "tail48_k32", "steps48", "seconds", "Delta32", "Delta48", "lam32", "lam48"]


def cmd_table(argv):
    """hp/data/chain_dM.csv from chain_dM.jsonl records: every unmutated record with M 48 solved, in file order (where M 32
    was not solved, its cells are empty and Delta32, lam32 are the stored state's). role
    is the record's folder (planted or chain), rule how the M 48 solve ended (below)"""
    import csv
    args = [a for a in argv if not a.startswith("--")]
    opt = opts_of(argv)
    out = opt.get("out", os.path.join(HERE, "data", "chain_dM.csv"))
    recs = []
    for p in args:
        for line in open(p):
            r = json.loads(line)
            if "dlam_stored" in r:              # a mutated record (a control) is kept, its mutate cell set
                r["role"] = os.path.basename(os.path.dirname(os.path.abspath(p)))
                recs.append(r)
    with open(out, "w", newline="") as fh:
        fh.write(f"# hp/chain_dM.py table: M 32 and M {recs[0]['Mhi']} solutions of the chain's pinned system by one path "
                 f"(#636); d* = M 48 - M 32, *_stored against the stored state; raw-data repository {opt.get('raw', '')}\n")
        w = csv.writer(fh)
        w.writerow(TABLE_COLS)
        for r in recs:
            b = r[f"M{r['Mhi']}"]
            # without the M 32 solve (--m32=0) its cells are empty and the differences are against the stored state
            a = r.get("M32", {})
            g = lambda d, k, f: format(d[k], f) if k in d else ""  # noqa: E731
            # rule: 'polish' (polished to the floor), 'tol' (stopped at --tol, the first D 80 plant) or 'floor' (stopped
            # at the quad-double floor without the polish, D 240.12)
            rule = "polish" if ("polish" in b or b["F"] <= 1e-45) else ("tol" if b["conv"] else "floor")
            w.writerow([r["role"], rule, r.get("mutate") or "",
                        r["D"], r["D_str"], r["state"], r["kfreeze"], r["parts"], g(a, "F", ".3e"), f"{b['F']:.3e}",
                        int(b["conv"]), g(a, "est", ".3e"), f"{b['est']:.3e}", g(r, "dDelta", ".4e"), g(r, "dlam", ".4e"),
                        f"{r['dDelta_stored']:.4e}", f"{r['dlam_stored']:.4e}", g(r, "M32_moved", ".3e"),
                        f"{b['F_fresh_plan']:.3e}", f"{b['F_start']:.3e}", f"{b.get('F_reseeded', float('nan')):.3e}",
                        f"{r['tail_32_from_28']:.2e}", f"{r[f'tail_{r['Mhi']}_from_32']:.2e}", b["steps"], r["seconds"],
                        a.get("Delta", r["Delta_stored"]), b["Delta"], a.get("lam", r["lam_stored"]), b["lam"]])
    print(f"{len(recs)} rows, {out}")
    return 0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "table":
        return cmd_table(sys.argv[2:])
    if cmd == "ramp":
        return cmd_ramp(sys.argv[2:])
    if cmd == "run":
        return cmd_run(sys.argv[2:])
    if cmd == "planted":
        return cmd_planted(sys.argv[2:])
    if cmd == "refit":
        return cmd_refit(sys.argv[2:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
