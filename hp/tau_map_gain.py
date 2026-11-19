#!/usr/bin/env python3
"""The tau-mapped grid's gain at low D, measured (#1033; notes/LOW_D_SPEEDUP.md, step 5). For each arctan map L of
tau_map.py: the modes M' its grid needs for the uniform grid's Delta to 1e-12, the steps of its Taylor plan, and the
projected cost of a Newton call against the uniform grid's. A call is 99.5% Jacobian, which costs the plan's steps times
c(M), c(M) ~ M^3 (notes/LOW_D_SPEEDUP.md appendix A, 1.2-1.3).

    cd hp && python3 tau_map_gain.py selftest [--mutate lo|parity|dps40|inverse|gp|phase_gp|lenient|unbound]
    cd hp && python3 tau_map_gain.py check --out F.jsonl [--Ls 0.9,0.7,0.5] [--mutate gp] [--threads 4]
    cd hp && python3 tau_map_gain.py plan STATE (--Ls 1,0.9,... --Ms 128,... | --pairs L:M',...) --out F.jsonl [--threads 8]
    cd hp && python3 tau_map_gain.py solve STATE (--L 0.5 --M 192 | --pairs L:M',...) --states DIR --out F.jsonl [--threads 8]
    cd hp && python3 tau_map_gain.py cost [STATE ...] [--Ms 320,384] [--reps 2] [--mutate plan_m] --out F.jsonl [--threads 8]
    cd hp && python3 tau_map_gain.py report F.jsonl [F.jsonl ...] --ref STATE [--ref2 STATE2] [--uniform STATE] [--cost F.jsonl]
        [--json F.json] [--mutate delta]
    cd hp && python3 tau_map_gain.py stored STATE [STATE ...]

selftest (no kernel, under a minute; exit 1 on failure). (1) tau_map.interp_dd: the identity map returns a state bit for
bit; a planted band-limited field of each parity goes from the uniform grid to three mapped grids (one with a target
1e-17 from a source point), from a mapped grid (the field band-limited in that grid's s) to the uniform grid, and from
one map to another, to within 1e-28 of its exact values, which come from the map's closed form and a Newton inverse
(mpmath), not from tau_map's own map. (2) tau_map.mapped_inputs: the mapped d/dtheta of a planted odd field equals the
exact derivative at the mapped points to 1e-26, and the mapped phase row gives the uniform grid's sum of sin(theta) p1
to 1e-26. (3) report on planted solve records: a failed identity solve fails it, and a failed larger M' certifies no
smaller one; a converged retry supersedes the failure. (4) the resume rules: a checkpoint of another configuration is
refused, a converged result not yet published is recovered from its checkpoint, and only a converged and saved result is
reused. Controls, each of which must make it fail: --mutate lo (interp_dd drops the lo parts, as map_state does), parity
(the odd kernel for even fields), dps40 (the interpolation matrix at 40 digits), inverse (the inverse map skipped), gp
(the derivative rows without 1/g'), phase_gp (the phase row without g'), lenient (report drops failed records) and
unbound (any checkpoint restored, any record with Delta reused).

check (the first check, on the resolved ladder at D 3.7 in data/lowD_states, highm15_D3.7000_M{96,128,256}; minutes):
the mapped grid's Delta at M 128 equals the uniform grid's at M 128 within the uniform grid's own resolution error
there, |Delta(128) - Delta(96)| (3.4e-11), for each L; each mapped M 128 solve starts from the uniform M 128 state
interpolated onto it. Exit 1 when an L fails. Control: --mutate gp (the derivative rows without 1/g') must fail. The
phase row's g' is no control: the phase row fixes only the tau origin, so dropping it moves Delta by no more than the
resolution error (--mutate phase_gp shows it).

plan (no Jacobian): for each (L, M') the state interpolated onto the mapped grid in double-double (interp_dd), its own
Taylor plan (steps per segment), its residual on that plan and its top harmonics. Planning from a state whose residual
is nan does not return (lowD_highM.py), so the start is first read on gate plans (gate_plan: the state's stored plan,
split 2, 4 and 8 times, then near x = c 4 and 16 times more); without a finite read there is no plan. L 1 at the state's
own M is the state's own plan.

solve: the two passes of the low-D legs (lowD_highM.py) on the mapped grid from the interpolated state, Newton with the
f64 Jacobian in block storage (Solver.newton_mixed(block=True); a step without decrease is retried once with the dd4
Jacobian): on the start's own plan to |F| < --tol1 (1e-14), then on the plan of that state to --tol (1e-28); then the
plan of the converged state, the lean plan a leg reuses. Records Delta, the three plans' steps and each call's wall
seconds. A configuration is the source state, L, M' and the map's centre s0, all in the file names
(tmap_<state>_L<L>_M<M'>_s<s0>). Every call is checkpointed in --states, and a checkpoint restores only into its own
configuration; a converged result stays in its checkpoint until the state is saved and the record written. A rerun skips
a configuration only when it converged and its state was saved; a failed or unconverged one resumes from its checkpoint
(a larger --it2 continues it). The converged state is saved there with meta["map"] = [s0, L]. L 1 at the state's own M
times one uniform call on the same machine (the reference of the cost ratio).

cost: c(M), the wall seconds per plan step of one f64 Newton call on this machine, at each given converged state's own M
on its own stored plan (default the D 3.7 ladder at M 128 to 256; raw T82-win/states for 256 to 384). A plan made at
another M is refused (require_plan_at): its steps are too long for the tangents of a larger M (on win2 the D 3.7 state
resampled to M 320 on the M 256 plan gave a nan pivot). Control: --mutate plan_m (the first state's plan for every M)
must exit 1. step_timing's t_jac is not used: in block storage it times only the staircase LU's fill.

stored (no kernel): the stored states' Delta, their plans' steps (the busiest segment and its steps) and top harmonics,
and the Delta differences between consecutive M at each D.

report: per L and M' the Delta difference from the uniform reference state (--ref) and whether it is below 1e-12 and
below the uniform grid's own resolution error |Delta(ref) - Delta(ref2)|; the steps of the three plans; the projected
cost per Newton call against the uniform grid at M_u (--uniform, default --ref) on its own plan, by the M^3 law, by c(M)
of a cost file, and measured (pass-2 Newton calls' wall seconds against the L 1 record's at M_u). Of several records of
one configuration it reads the last. Per L, the modes needed: the smallest M' whose Delta, and that of every larger M'
tried for that L, converged within 1e-12 of the reference; a failed size counts as a miss. Its checks (exit 1 on
failure): an L 1 solve from a stored state converges to the stored Delta of its M (--ref, --ref2 or --uniform) within
1e-20, at the state's own M with its own plan within 10% of the stored plan's steps, and an L 1 solve of those M that
did not converge fails; and, with --ref2, each L at its largest M' converged with Delta within the uniform grid's own
resolution error |Delta(ref) - Delta(ref2)|. Control: --mutate delta moves every stored reference Delta by 1e-10, and
the checks must then fail.

The map's centre s0 is chosen per L as in tau_map_study.py: the centre of 24 in [0, pi) at which the slowest field block
needs the fewest modes at eps 1e-4 (--s0 sets it)."""
import argparse
import functools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, HERE)
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

import tau_map  # noqa: E402
from tau_map import dd_matmul, interp_dd, mapped_inputs  # noqa: E402

STATES37 = os.path.join(HERE, "data", "lowD_states")
TOL_DELTA = 1e-12
UNBOUND = False      # selftest --mutate unbound: checkpoints restored for any configuration, any record with Delta reused
PLAN_SPREAD = 0.10   # the identity's plan bound: plans of bit-different copies of one solution differ by a few percent
MIXED = dict(block=True)


# ------------------------------------------------------------------ helpers
def load_state(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    return np.asarray(d["zh"], float), np.asarray(d["zl"], float), meta


def delta_of(zh, zl):
    return float(zh[-1]), float(zl[-1])


def ddiff(a, b):
    """(hi, lo) - (hi, lo) as a float (the hi parts first: exact for close values)"""
    return (a[0] - b[0]) + (a[1] - b[1])


def nsteps(segs):
    return sum(len(g) - 1 for g in segs)


def fmax(s, zh, zl):
    fh, fl = s.residual(zh, zl)
    return float(np.abs(np.asarray(fh) + np.asarray(fl)).max())


def tail_ratio(zh, zl, M, K):
    """the largest ratio, over field blocks, of the top 12 harmonics of the block's parity to its largest harmonic
    (the state's resolution on its own grid), and that block"""
    import tau_map_study as tms
    worst = (0.0, "")
    for name, par, f in tms.blocks(zh, zl, M, K):
        c = np.abs(tms.coeffs(par, f))
        k = np.arange(len(c))
        a = c[k % 2 == (1 if par == "odd" else 0)]          # the block's parity only, so a[-12:] is its top 12
        if a.max() > 0:
            worst = max(worst, (float(a[-12:].max() / a.max()), name))
    return worst


def pick_s0(zh, zl, M, K, L, eps=1e-4, fine=8, n=24):
    """the study's centre (tau_map_study.py): among n centres in [0, pi), the first at which the slowest block of the
    state, mapped with the arctan map L, needs the fewest modes at eps"""
    import tau_map_study as tms
    if L == 1:
        return 0.0
    B = tms.blocks(zh, zl, M, K)
    pars = [b[1] for b in B]
    C = np.array([tms.coeffs(par, f) for _, par, f in B])
    Nf = fine * 2 * M
    sf = 2 * np.pi * np.arange(Nf) / Nf
    best = None
    for s0 in np.linspace(0, np.pi, n, endpoint=False):
        u = sf - s0
        g = s0 + np.arctan(L * np.tan(u)) + np.pi * np.round(u / np.pi)
        Q = tms.interpolant(C, g)
        Cm = np.array([np.fft.rfft(Q[:, i]) / Nf for i in range(len(B))])
        score = max(tms.k_needed(Cm[i], pars[i], eps) for i in range(len(B)))
        if best is None or score < best[0]:
            best = (score, float(s0))
    return best[1]


def read_done(path):
    out = []
    if path and os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def log(msg):
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


@functools.lru_cache(maxsize=1)
def kernel_info():
    """the hpkernel in use (path and crate versions in its binary), for the records"""
    import glob
    import re
    import hpkernel
    d = os.path.dirname(hpkernel.__file__)
    vers = set()
    for so in glob.glob(os.path.join(d, "_hpkernel*")):
        data = open(so, "rb").read()
        vers |= {m.decode() for m in re.findall(rb"(multi(?:double|shoot|solver)-\d+\.\d+\.\d+)", data)}
    return dict(path=d, crates=sorted(vers))


# ------------------------------------------------------------------ selftest (no kernel)
def _planted(par, K, rng):
    """a band-limited field of the parity: odd harmonics 1..K (odd), or even 0..K (even), random coefficients"""
    ks = [k for k in range(K + 1) if (k % 2 == 1) == (par == "odd")]
    return [(k, mp.mpf(float(rng.normal())), mp.mpf(float(rng.normal()))) for k in ks]


def _eval(terms, t, deriv=False):
    if deriv:
        return sum(k * (-a * mp.sin(k * t) + b * mp.cos(k * t)) for k, a, b in terms)
    return sum(a * mp.cos(k * t) + b * (mp.sin(k * t) if k else 0) for k, a, b in terms)


def _g_ref(s, s0, L):
    """the arctan map in its closed form, g = s0 + atan2(L sin u, cos u) on the branch continuous with s, and g'
    (tau_map.py's docstring; independent of tau_map.g_and_gp's deviation form and of its inverse)"""
    u = s - s0
    v = mp.atan2(L * mp.sin(u), mp.cos(u))
    return s0 + v + 2 * mp.pi * mp.nint((u - v) / (2 * mp.pi)), L / (mp.cos(u) ** 2 + L ** 2 * mp.sin(u) ** 2)


def _g_ref_inverse(t, s0, L):
    """s with _g_ref(s) = t, by Newton's method from s = t"""
    s = t
    for _ in range(60):
        g, gp = _g_ref(s, s0, L)
        ds = (g - t) / gp
        s -= ds
        if abs(ds) < mp.mpf(10) ** (-mp.mp.dps + 5):
            break
    return s


def selftest(a):
    from multidouble import dd_from_mp
    mp.mp.dps = 40
    rng = np.random.default_rng(1033)
    mut = a.mutate or None
    ok = True
    res = {}

    # (1) interp_dd
    M = 32
    K1 = 1                                                    # one node: blocks p1, c0, p, m, a, c, m_h, a_h
    par = tau_map.parities(K1)
    zh = rng.normal(size=len(par) * M + 1)
    zl = zh * 2.0 ** -60 * rng.normal(size=zh.size)
    oh, ol = interp_dd(zh, zl, M, M, K1, None, None, mutate=mut if mut in ("lo", "parity") else None)
    bits = int(np.sum(oh != zh) + np.sum(ol != zl))
    res["identity_differing_words"] = bits
    ok &= bits == 0
    print(f"(1a) identity map: {bits} differing words of {2 * zh.size} ({'PASS' if bits == 0 else 'FAIL'})")
    # the last case: the map's centre is the float nearest a source point, so the target there lies 1e-17 from it
    s0n = float(mp.pi * 8 / M)
    grids = [("uniform -> arctan(1.9, 0.6), M 32 -> 40", None, (1.9, 0.6), 40),
             ("uniform -> arctan(0.4, 0.3), M 32 -> 24", None, (0.4, 0.3), 24),
             ("arctan(1.9, 0.6) -> uniform, M 32 -> 48", (1.9, 0.6), None, 48),
             ("arctan(1.9, 0.6) -> arctan(1.2, 0.8), M 32 -> 32", (1.9, 0.6), (1.2, 0.8), 32),
             ("uniform -> arctan(fl(pi/4), 0.5), M 32 -> 32, a target 1e-17 from a source point", None, (s0n, 0.5), 32)]
    for label, frm, to, Mt in grids:
        # the planted field band-limited in the source's s (harmonics < M): exact values on the source points and at
        # the target points' source coordinates
        fields = [_planted(p, M - 1 if p == "odd" else M - 2, rng) for p in par]
        s_src = [mp.pi * j / M for j in range(M)]
        # the exact values at the target points from the closed-form map and its Newton inverse, not tau_map's
        with mp.workdps(60):
            th_to = [mp.pi * i / Mt if to is None else _g_ref(mp.pi * i / Mt, mp.mpf(to[0]), mp.mpf(to[1]))[0]
                     for i in range(Mt)]
            s_at = [t if frm is None else _g_ref_inverse(t, mp.mpf(frm[0]), mp.mpf(frm[1])) for t in th_to]
        src = [v for f in fields for v in (_eval(f, s) for s in s_src)]
        exact = [v for f in fields for v in (_eval(f, s) for s in s_at)]
        h, lo = dd_from_mp(src + [mp.mpf(2.5)])
        oh, ol = interp_dd(h, lo, M, Mt, K1, frm, to, mutate=mut if mut in ("lo", "parity", "dps40", "inverse") else None)
        err = max(abs(mp.mpf(float(oh[i])) + mp.mpf(float(ol[i])) - exact[i]) for i in range(len(exact)))
        scale = max(abs(v) for v in exact)
        rel = float(err / scale)
        res[label] = rel
        ok &= rel < 1e-28
        print(f"(1b) {label}: max error {rel:.1e} of the largest value ({'PASS' if rel < 1e-28 else 'FAIL'}; < 1e-28)")

    # (2) mapped_inputs: the derivative and the phase row
    M2, s0, L = 128, 1.9, 0.7
    terms = _planted("odd", 7, rng)
    dh, dl, ph, pl = mapped_inputs(M2, s0, L, mutate=mut if mut in ("gp", "phase_gp") else None)
    th = tau_map._grid_theta(M2, (s0, L), 40)
    fh, fl = dd_from_mp([_eval(terms, t) for t in th])
    Dh, Dl = dd_matmul(dh, dl, fh[:, None], fl[:, None])
    exact = [_eval(terms, t, deriv=True) for t in th]
    err = max(abs(mp.mpf(float(Dh[i, 0])) + mp.mpf(float(Dl[i, 0])) - exact[i]) for i in range(M2))
    rel = float(err / max(abs(v) for v in exact))
    res["derivative"] = rel
    ok &= rel < 1e-26
    print(f"(2a) mapped d/dtheta, arctan({s0}, {L}), M {M2}: max error {rel:.1e} of the largest derivative "
          f"({'PASS' if rel < 1e-26 else 'FAIL'}; < 1e-26)")
    uni = sum(mp.sin(mp.pi * j / M2) * _eval(terms, mp.pi * j / M2) for j in range(M2))
    Ph, Pl = dd_matmul(ph[None, :], pl[None, :], fh[:, None], fl[:, None])
    scale = sum(abs(mp.mpf(float(ph[j])) * mp.mpf(float(fh[j]))) for j in range(M2))
    rel = float(abs(mp.mpf(float(Ph[0, 0])) + mp.mpf(float(Pl[0, 0])) - uni) / scale)
    res["phase"] = rel
    ok &= rel < 1e-26
    print(f"(2b) mapped phase row: its sum differs from the uniform grid's by {rel:.1e} of sum |w p1| "
          f"({'PASS' if rel < 1e-26 else 'FAIL'}; < 1e-26)")
    ok &= _selftest_report(mut, res)
    ok &= _selftest_workflow(mut, res)
    print(f"selftest{' --mutate ' + mut if mut else ''}: {'PASS' if ok else 'FAIL'}")
    if a.out:
        append(a.out, dict(cmd="selftest", mutate=mut, ok=bool(ok), **{k: v for k, v in res.items()}))
    return 0 if ok else 1


def _selftest_report(mut, res):
    """(3) report on planted solve records against the stored D 3.7 states (no kernel): (A) a converged identity and
    L 0.5 at M' 80 and 96 within 1e-12 give exit 0 and M' 80 as the modes needed; (B) the identity solve failed in
    pass 1 gives exit 1; (C) L 0.5 at M' 96 failed certifies no M' for L 0.5; (D) that failure followed by a
    converged retry gives (A)'s answer. --mutate lenient drops the records without Delta, as report did before, and
    must make (B) and (C) fail."""
    import contextlib
    import io
    import tempfile
    path128, path256 = (os.path.join(STATES37, f"highm15_D3.7000_M{M}.npz") for M in (128, 256))
    zu, zul, mu = load_state(path128)
    zr, zrl, _ = load_state(path256)
    hist = [dict(F=1e-29, t=10.5, t_step=10.0)]

    def solved(L, Mp, Delta, steps):
        return dict(cmd="solve", state="planted/highm15_D3.7000_M128.npz", D=3.7, M=128, L=L, Mp=Mp, s0=0.0 if L == 1 else 1.7,
                    converged=True, Delta=Delta[0], Delta_lo=Delta[1], steps_start=2 * steps, steps_p1=steps,
                    steps_own=steps, hist2=hist)

    def failed(L, Mp):
        return dict(cmd="solve", state="planted/highm15_D3.7000_M128.npz", D=3.7, M=128, L=L, Mp=Mp,
                    s0=0.0 if L == 1 else 1.7, failed="pass 1: planted", hist1=[])

    ident = solved(1.0, 128, delta_of(zu, zul), nsteps(mu["segs"]))
    m80 = solved(0.5, 80, (float(zr[-1]) + 1e-14, float(zrl[-1])), 110)
    m96 = solved(0.5, 96, delta_of(zr, zrl), 130)
    cases = dict(A=[ident, m80, m96], B=[failed(1.0, 128), m80, m96], C=[ident, m80, failed(0.5, 96)],
                 D=[ident, m80, failed(0.5, 96), m96])
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for name, recs in cases.items():
            f, j = os.path.join(d, f"{name}.jsonl"), os.path.join(d, f"{name}.json")
            for r in recs:
                append(f, r)
            ns = argparse.Namespace(files=[f], ref=path256, ref2="", uniform=path128, cost=None, json=j,
                                    mutate="lenient" if mut == "lenient" else "")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = report(ns)
            need = json.load(open(j))["need"]
            out[name] = (rc, need.get("0.5", {}).get("Mp"))
    want = dict(A=(0, 80), B=(1, 80), C=(0, None), D=(0, 80))
    ok = True
    for name in "ABCD":
        good = out[name] == want[name]
        ok &= good
        print(f"(3{name}) report on planted records: exit {out[name][0]}, modes needed for L 0.5 {out[name][1]} "
              f"(want {want[name][0]}, {want[name][1]}): {'PASS' if good else 'FAIL'}")
    res["report"] = {k: list(v) for k, v in out.items()}
    return ok


def _selftest_workflow(mut, res):
    """(4) the resume rules, no kernel: (a) a checkpoint written for another centre is refused, and two_pass returns a
    failed record before building a solver; (b) a pass-3 checkpoint (converged, not yet published) is returned as the
    result; (c) only a converged record with its saved state is reused, not an unconverged one, nor one whose state is
    missing. --mutate unbound (any checkpoint restored, any record with Delta reused, the rules before) must fail."""
    import tempfile
    global UNBOUND
    UNBOUND = mut == "unbound"
    ok = True
    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "t_ckpt.npz")
        ident = dict(state="x.npz", L=0.5, Mp=96, s0=1.7)
        other = dict(ident, s0=1.8)
        rec3 = dict(converged=True, Delta=2.5, Delta_lo=0.0, steps_own=3, _segs_own=[[0.1, 0.2, 0.3, 0.4]])
        np.savez(ck, zh=np.ones(3), zl=np.zeros(3), meta=json.dumps({"pass": 3, "hist1": [], "hist2": [], "rec": rec3,
                                                                    "ident": other}))
        ns = argparse.Namespace(threads=1, tol1=1e-14, tol=1e-28, it1=1, it2=1, tol_step=1e-30, hmax=0.05)
        r, fin = two_pass({}, (np.zeros(3), np.zeros(3)), 96, None, ns, "t", ck=ck, ident=ident)
        good_a = fin is None and "belongs to" in r.get("failed", "")
        np.savez(ck, zh=np.ones(3), zl=np.zeros(3), meta=json.dumps({"pass": 3, "hist1": [], "hist2": [], "rec": rec3,
                                                                    "ident": ident}))
        r, fin = two_pass({}, (np.zeros(3), np.zeros(3)), 96, None, ns, "t", ck=ck, ident=ident)
        good_b = fin is not None and float(fin[0][0]) == 1.0 and fin[2] == rec3["_segs_own"] and r.get("converged")
        saved = os.path.join(d, "saved.npz")
        open(saved, "w").close()
        base = dict(cmd="solve", state="x.npz", L=0.5, Mp=96, s0=1.7, Delta=2.5, Delta_lo=0.0)
        key = solve_key(base)
        good_c = (reusable([dict(base, converged=True, saved=saved)], key)
                  and not reusable([dict(base, converged=False)], key)
                  and not reusable([dict(base, converged=True, saved=os.path.join(d, "gone.npz"))], key))
    UNBOUND = False
    for name, good in (("a", good_a), ("b", good_b), ("c", good_c)):
        ok &= bool(good)
    print(f"(4) resume rules: other centre refused {'PASS' if good_a else 'FAIL'}; pass-3 result recovered "
          f"{'PASS' if good_b else 'FAIL'}; only converged and saved results reused {'PASS' if good_c else 'FAIL'}")
    res["workflow"] = [bool(good_a), bool(good_b), bool(good_c)]
    return ok


# ------------------------------------------------------------------ Newton on a mapped grid
def newton_f64(s, zh, zl, tol, it_max, hist, save=None, stop_below=False):
    """Newton with the f64 Jacobian in block storage and backtracking on |F|_2 (driver.newton_solve's rules, without
    its dense exact retry, which M <= 128 would take): a step without decrease is retried once with the dd4
    Jacobian; once |F| < tol the computed step is still taken if |F| stays below it. stop_below (pass 1): stop at an
    accepted point with max |F| < tol, without the Jacobian there; without it, an accepted point below tol on the last
    allowed call also ends the pass converged (no call is left for the polish). A Jacobian or LU that the kernel
    refuses, in the f64 call or in the dd4 retry, ends the pass as a recorded failure. One record per call in hist
    (resumable: hist holds the calls made before); its t_step and t_retry are the wall seconds of the f64 call and of
    the dd4 retry. save(zh, zl, hist) after each accepted step. Returns (zh, zl, ok)."""
    from hpkernel.driver import dd_add, dd_norm, newton_step, rnorm

    def search(dh, dl, n0):
        lam = 1.0
        for _ in range(11):
            th, tl = dd_add(zh, zl, dh * lam, dl * lam)
            Th, Tl = s.residual(th, tl)
            n1 = rnorm(Th, Tl)
            if np.isfinite(n1) and n1 <= (1 - 1e-4 * lam) * n0:
                return True, lam, th, tl, dd_norm(Th, Tl)
            lam *= 0.5
        return False, lam, None, None, None

    while len(hist) < it_max:
        t0 = time.time()
        try:
            dh, dl, Fh, Fl, pmin, pmax, info = newton_step(s, zh, zl, "f64", MIXED)
        except ValueError as e:                               # a non-finite Jacobian: the LU refuses it
            hist.append(dict(it=len(hist), failed=f"jacobian: {e}"))
            return zh, zl, False
        # t_step: the step's wall seconds (residual, Jacobian, LU, solve); step_timing's t_jac is the block fill only
        t_step = round(time.time() - t0, 2)
        n0, nF = rnorm(Fh, Fl), dd_norm(Fh, Fl)
        rec = dict(it=len(hist), F=nF, dz=dd_norm(dh, dl), piv=[pmin, pmax], t_step=t_step,
                   **{k: round(v, 2) for k, v in info.items() if k.startswith("t_")})
        if nF < tol:
            th, tl = dd_add(zh, zl, dh, dl)
            n1 = dd_norm(*s.residual(th, tl))
            if n1 < tol:
                zh, zl = th, tl
            rec.update(converged=True, F_polished=n1, t=round(time.time() - t0, 1))
            hist.append(rec)
            log(f"    call {rec['it']}: |F| {nF:.2e} converged; step {t_step:.1f} s")
            return zh, zl, True
        acc, lam, th, tl, F1 = search(dh, dl, n0)
        if not acc:
            t1 = time.time()
            try:
                dh, dl, Fh, Fl, pmin, pmax = s.newton_mixed(zh, zl, tangents=4, dd_base=True, block=True)
            except ValueError as e:                           # the dd4 Jacobian refused too: a failed call
                rec.update(retry_dd4=True, t_retry=round(time.time() - t1, 2), failed=f"dd4 retry: {e}",
                           t=round(time.time() - t0, 1))
                hist.append(rec)
                log(f"    call {rec['it']}: |F| {nF:.2e}: no decrease, and the dd4 retry failed: {e}")
                return zh, zl, False
            rec.update(retry_dd4=True, t_retry=round(time.time() - t1, 2))
            acc, lam, th, tl, F1 = search(dh, dl, n0)
        if not acc:
            rec.update(failed="line search", t=round(time.time() - t0, 1))
            hist.append(rec)
            log(f"    call {rec['it']}: |F| {nF:.2e}: no decrease along the step")
            return zh, zl, False
        zh, zl = th, tl
        rec.update(lam=lam, dDelta=float(dh[-1] + dl[-1]) * lam, Delta=float(zh[-1]), F_after=F1,
                   t=round(time.time() - t0, 1))
        hist.append(rec)
        log(f"    call {rec['it']}: |F| {nF:.2e} -> {F1:.2e} |dz| {rec['dz']:.1e} lam {lam:g} Delta {zh[-1]:.15f} "
            f"step {t_step:.1f} s")
        if save:
            save(zh, zl, hist)
        if F1 < tol and (stop_below or len(hist) >= it_max):
            rec["converged"] = True
            if not stop_below:
                rec["unpolished"] = True                     # the budget's last call: below tol, no polish step
            return zh, zl, True
        if rec["dz"] * lam < 1e-31:
            break
    return zh, zl, False


def call_seconds(hist):
    """the wall seconds of each Newton call of a pass: t_step plus the dd4 retry's t_retry, or for records made before
    t_step the call's t (the step, its retry and its line-search residuals, about 1% more); never step_timing's t_jac,
    the block fill only"""
    return [h["t_step"] + h.get("t_retry", 0.0) if "t_step" in h else h["t"] for h in hist if "F" in h and "t" in h]


def gate_plan(s, zh, zl, segs_ref, M, nodes, tol_step, hmax):
    """the start's own plan, planned only from a start whose residual is finite on a gate plan: segs_ref, then segs_ref
    with every step split 2, 4 and 8 times (a mapped grid's top harmonics are stiffer by up to 1/L at the same M, and
    an L 0.7 solve on the uniform plan overflowed, notes/LOW_D_SPEEDUP.md step 5), then that with the steps near
    x = c split 4 and 16 times more (sonic_valley.refine_near). Returns (plan, gate label) or (None, reason)."""
    import sonic_valley
    z = zh + zl
    levels = [("stored plan", segs_ref)]
    levels += [(f"stored plan split {r}", [sonic_valley.refine(g, r) for g in segs_ref]) for r in (2, 4, 8)]
    levels += [(f"stored plan split 8, near x = c {8 * r}", sonic_valley.refine_near(levels[-1][1], z, M, nodes, r, 1e-2))
               for r in (4, 16)]
    for gate, sg in levels:
        s.set_segs(sg)
        if np.isfinite(fmax(s, zh, zl)):
            segs = s.plan(zh, zl, tol_step, hmax)
            s.set_segs(segs)
            return segs, gate
    return None, "nan on every gate plan (the stored plan split up to 8 times, near x = c up to 128 times)"


def parse_pairs(a):
    """[(L, M')] from --pairs L:M,... or the product of --Ls and --Ms"""
    if getattr(a, "pairs", ""):
        return [(float(p.split(":")[0]), int(p.split(":")[1])) for p in a.pairs.split(",")]
    return [(float(L), int(M)) for L in a.Ls.split(",") for M in a.Ms.split(",")]


def resolve_s0(path, L, s0_opt):
    """the map's centre for L: --s0 when given, else pick_s0 on the state at path (0 for L 1)"""
    if s0_opt not in (None, "auto"):
        return float(s0_opt)
    zh, zl, meta = load_state(path)
    return pick_s0(zh, zl, int(meta["M"]), len(meta["nodes"]), L)


def map_start(path, L, Mp, s0_opt):
    """the state at path, its map's centre for L, and the state interpolated onto the mapped grid (s0, L) at M'"""
    zh, zl, meta = load_state(path)
    M, K = int(meta["M"]), len(meta["nodes"])
    frm = tuple(meta["map"]) if meta.get("map") else None
    s0 = resolve_s0(path, L, s0_opt)
    to = None if L == 1 else (s0, L)
    if frm is None and to is None and Mp == M:
        z0 = (zh, zl)
    else:
        z0 = interp_dd(zh, zl, M, Mp, K, frm, to)
    return meta, (zh, zl), s0, to, z0


def solver_for(meta, Mp, to, threads, mutate=None, segs=None):
    from hpkernel.driver import make_solver
    kw = dict(nodes=list(meta["nodes"]), kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], threads=threads)
    if to is None and not mutate:
        s = make_solver(meta["D"], Mp, **kw)
    else:
        s0, L = to if to is not None else (0.0, 1.0)
        s = tau_map.make_solver_mapped(meta["D"], Mp, s0, L, mutate=mutate, **kw)
    if segs is not None:
        s.set_segs(segs)
    return s


def two_pass(meta, z0, Mp, to, a, tag, mutate=None, ck=None, ident=None):
    """lowD_highM's two passes on the grid `to` at M' from z0; returns a record and the final state (or None).
    ck: the checkpoint file (None: none), written after every accepted call and bound to ident (the source state, L,
    M', s0, mutation): a checkpoint of another configuration is refused. Its pass 3 holds a converged result until
    the caller has published it (retire_ckpt), so an interruption between the two loses nothing."""
    rec = dict(t_start=time.strftime("%Y-%m-%d %H:%M", time.gmtime()))
    hist1, hist2, p = [], [], 1
    zh, zl = z0
    if ck and os.path.exists(ck):
        d = np.load(ck, allow_pickle=False)
        cm = json.loads(str(d["meta"]))
        if not ckpt_matches(cm, ident):
            rec.update(failed=f"checkpoint {ck} belongs to {cm.get('ident')}, not {ident}")
            return rec, None
        zh, zl, p, hist1, hist2 = np.asarray(d["zh"]), np.asarray(d["zl"]), cm["pass"], cm["hist1"], cm["hist2"]
        rec.update(cm["rec"])
        log(f"  resuming {tag} from its checkpoint (pass {p}, {len(hist1)} + {len(hist2)} calls)")
        if p == 3:                                           # converged, not yet published
            segs_own = rec.pop("_segs_own")
            return rec, (zh, zl, segs_own)
    s = solver_for(meta, Mp, to, a.threads, mutate)

    def save(p_, zh_, zl_):
        if ck:
            m = {"pass": p_, "hist1": hist1, "hist2": hist2, "rec": rec, "ident": ident}
            np.savez(ck + ".tmp.npz", zh=zh_, zl=zl_, meta=json.dumps(m))
            os.replace(ck + ".tmp.npz", ck)

    t0 = time.time()
    if p == 1 and hist1 and hist1[-1].get("F_after", np.inf) < a.tol1:
        p = 2                                                # pass 1 ended below tol1 before its phase was saved
    if p == 1:
        if "steps_start" not in rec:
            segs, gate = gate_plan(s, zh, zl, meta["segs"], Mp, meta["nodes"], a.tol_step, a.hmax)
            if segs is None:
                rec.update(failed=gate)
                return rec, None
            rec.update(gate=gate, steps_start=nsteps(segs), segs_start=[len(g) - 1 for g in segs],
                       F_start=fmax(s, zh, zl), t_plan_start=round(time.time() - t0, 1))
            log(f"  {tag}: start |F| {rec['F_start']:.2e} on its own plan, {rec['steps_start']} steps ({gate})")
            rec["_segs1"] = segs
        s.set_segs(rec["_segs1"])
        if rec["F_start"] < a.tol1 and not hist1:
            rec["pass1"] = "skipped: the start is below tol1 on its own plan"
            rec["_segs2"] = rec["_segs1"]                    # the same state: its plan is the start's
            rec.update(steps_p1=rec["steps_start"], segs_p1=rec["segs_start"], t_plan_p1=0.0)
        else:
            zh, zl, ok1 = newton_f64(s, zh, zl, a.tol1, a.it1, hist1, save=lambda h, l, _: save(1, h, l), stop_below=True)
            if not ok1:
                last = hist1[-1] if hist1 else {}
                rec.update(failed=f"pass 1: {last.get('failed', 'no |F| below tol1 in it1 calls')}", hist1=hist1,
                           Delta_last=float(zh[-1]))
                return rec, None
        p = 2
        if "pass1" not in rec:
            rec.pop("_segs2", None)
        save(2, zh, zl)
    if "_segs2" not in rec:
        t1 = time.time()
        segs2 = s.plan(zh, zl, a.tol_step, a.hmax)
        rec.update(steps_p1=nsteps(segs2), segs_p1=[len(g) - 1 for g in segs2], t_plan_p1=round(time.time() - t1, 1))
        rec["_segs2"] = segs2
        log(f"  {tag}: pass 2 on the pass-1 state's plan, {rec['steps_p1']} steps")
    s.set_segs(rec["_segs2"])
    zh, zl, ok2 = newton_f64(s, zh, zl, a.tol, a.it2, hist2, save=lambda h, l, _: save(2, h, l))
    t2 = time.time()
    segs_own = s.plan(zh, zl, a.tol_step, a.hmax) if ok2 else rec["_segs2"]
    rec.update(converged=bool(ok2), Delta=float(zh[-1]), Delta_lo=float(zl[-1]), hist1=hist1, hist2=hist2,
               steps_own=nsteps(segs_own), segs_own=[len(g) - 1 for g in segs_own], t_plan_own=round(time.time() - t2, 1),
               t_step_p2=call_seconds(hist2), seconds=round(time.time() - t0, 1))
    final = (zh, zl, segs_own) if ok2 else None
    rec.pop("_segs1", None)
    rec.pop("_segs2", None)
    if ok2:
        rec["_segs_own"] = segs_own
        save(3, zh, zl)                                      # the result, kept until it is published
        rec.pop("_segs_own")
    return rec, final


def ckpt_matches(cm, ident):
    """a checkpoint restores only into the configuration it was written for (the source state, L, M', s0, mutation)"""
    return UNBOUND or cm.get("ident") == ident


def reusable(recs, key):
    """a solve is skipped only when a record of its configuration converged and its saved state exists"""
    return any(r.get("cmd") == "solve" and solve_key(r) == key
               and (("Delta" in r) if UNBOUND else (r.get("converged") and os.path.exists(r.get("saved", ""))))
               for r in recs)


def retire_ckpt(ck):
    """the checkpoint of a published result, kept under _ckpt_done"""
    if ck and os.path.exists(ck):
        os.replace(ck, ck.replace("_ckpt.npz", "_ckpt_done.npz"))


def save_state(path, zh, zl, meta, Mp, to, segs):
    m = dict(meta, M=Mp, segs=segs, map=list(to) if to else None)
    np.savez(path, zh=zh, zl=zl, meta=json.dumps(m))


# ------------------------------------------------------------------ check (D 3.7)
def check(a):
    if a.out:
        from raw_out import require_raw
        require_raw(a.out)
    st = {M: load_state(os.path.join(STATES37, f"highm15_D3.7000_M{M}.npz")) for M in (96, 128, 256)}
    D96, D128, D256 = (delta_of(*st[M][:2]) for M in (96, 128, 256))
    e_u = abs(ddiff(D128, D96))
    print(f"D 3.7, uniform: Delta(128) - Delta(256) {ddiff(D128, D256):+.2e}; the uniform grid's own resolution error "
          f"at M 128, |Delta(128) - Delta(96)|, {e_u:.2e}")
    path = os.path.join(STATES37, "highm15_D3.7000_M128.npz")
    ok = True
    info = kernel_info()
    for L in [float(v) for v in a.Ls.split(",")]:
        s0 = resolve_s0(path, L, a.s0)
        # a result is reused only for the same L, mutation and centre, and only when it converged
        prev = [r for r in read_done(a.out) if r.get("cmd") == "check" and r["L"] == L and r.get("mutate") == a.mutate
                and round(r.get("s0", -1), 12) == round(s0, 12) and r.get("converged")]
        if prev:
            r = prev[-1]
        else:
            meta, _, s0, to, z0 = map_start(path, L, 128, s0)
            log(f"check L {L}: s0 {s0:.3f}{', mutate ' + a.mutate if a.mutate else ''}")
            tag = f"check_L{L}_s{s0:.4f}" + (f"_{a.mutate}" if a.mutate else "")
            ck = os.path.join(a.states, f"{tag}_ckpt.npz") if a.states else None
            r, _ = two_pass(meta, z0, 128, to, a, tag, a.mutate, ck=ck, ident=dict(L=L, s0=s0, mutate=a.mutate))
            r.update(cmd="check", L=L, s0=s0, M=128, mutate=a.mutate, kernel=info)
            if a.out:
                append(a.out, r)
            if r.get("converged"):
                retire_ckpt(ck)
        if "Delta" not in r:
            print(f"L {L}: failed ({r.get('failed')})")
            ok = False
            continue
        d = ddiff((r["Delta"], r["Delta_lo"]), D128)
        good = r["converged"] and abs(d) <= e_u
        ok &= good
        print(f"L {L}: Delta(mapped, 128) - Delta(uniform, 128) {d:+.2e} (bound {e_u:.2e}): {'PASS' if good else 'FAIL'}; "
              f"- Delta(uniform, 256) {ddiff((r['Delta'], r['Delta_lo']), D256):+.2e}; plans {r.get('steps_start')} / "
              f"{r.get('steps_p1')} / {r.get('steps_own')} steps (uniform M 128 state's: {nsteps(st[128][2]['segs'])})")
    print(f"check{' --mutate ' + a.mutate if a.mutate else ''}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ------------------------------------------------------------------ plan, solve, cost
def plan_cmd(a):
    from raw_out import require_raw
    require_raw(a.out)
    zh, zl, meta = load_state(a.state)
    M, K = int(meta["M"]), len(meta["nodes"])
    info = kernel_info()
    done = {(r["L"], r["Mp"], round(r["s0"], 12)) for r in read_done(a.out)
            if r.get("cmd") == "plan" and r.get("state") == a.state and "steps" in r}
    print(f"{a.state}: D {meta['D']}, M {M}, its stored plan {nsteps(meta['segs'])} steps; kernel {info['crates']}")
    for L, Mp in parse_pairs(a):
        if True:
            s0 = resolve_s0(a.state, L, a.s0)
            if (L, Mp, round(s0, 12)) in done:              # a failed gate is tried again
                continue
            t0 = time.time()
            meta_, _, s0, to, z0 = map_start(a.state, L, Mp, s0)
            t_int = time.time() - t0
            s = solver_for(meta, Mp, to, a.threads)
            t1 = time.time()
            segs, gate = gate_plan(s, *z0, meta["segs"], Mp, meta["nodes"], a.tol_step, a.hmax)
            rec = dict(cmd="plan", state=a.state, D=meta["D"], M=M, L=L, Mp=Mp, s0=s0, t_interp=round(t_int, 1), kernel=info)
            if segs is None:
                rec.update(failed=gate)
            else:
                tp = time.time() - t1
                t2 = time.time()
                F = fmax(s, *z0)
                tail, blk = tail_ratio(*z0, Mp, K)
                rec.update(gate=gate, steps=nsteps(segs), segs=[len(g) - 1 for g in segs], F=F, tail=tail,
                           tail_block=blk, t_plan=round(tp, 1), t_res=round(time.time() - t2, 1))
            append(a.out, rec)
            log(f"L {L} M' {Mp} s0 {s0:.3f}: " + (f"plan {rec['steps']} steps, |F| {rec['F']:.2e}, tail {rec['tail']:.1e} "
                                                f"({rec['tail_block']}), plan {rec['t_plan']} s" if segs else rec["failed"]))
    return 0


def solve_cmd(a):
    from raw_out import require_raw
    require_raw(a.out, os.path.join(a.states, "x.npz"))
    pairs = parse_pairs(a) if a.pairs else [(a.L, a.M)]
    bad = 0
    for L, Mp in pairs:
        bad += solve_one(a, L, Mp)
    return 1 if bad else 0


def solve_key(r):
    """a solve's configuration: the source state's file name, L, M' and the map's centre (old records too)"""
    return (os.path.basename(r["state"]), float(r["L"]), int(r["Mp"]), round(float(r["s0"]), 12))


def solve_one(a, L, Mp):
    s0 = resolve_s0(a.state, L, a.s0)
    base = os.path.splitext(os.path.basename(a.state))[0]
    tag = f"tmap_{base}_L{L:g}_M{Mp}_s{s0:.4f}"
    final = os.path.join(a.states, f"{tag}.npz")
    ck = os.path.join(a.states, f"{tag}_ckpt.npz")
    key = solve_key(dict(state=a.state, L=L, Mp=Mp, s0=s0))
    # only a converged result whose state was saved is reused; a failed or unconverged one is solved again,
    # from its checkpoint (a larger --it2 continues it)
    if reusable(read_done(a.out), key):
        log(f"{tag}: already converged and saved in {a.out}")
        return 0
    meta, _, s0, to, z0 = map_start(a.state, L, Mp, s0)
    log(f"{tag}: D {meta['D']}, from M {meta['M']} to the arctan map L {L} at s0 {s0:.3f}, M' {Mp}")
    ident = dict(state=os.path.basename(a.state), L=L, Mp=Mp, s0=s0)
    rec, fin = two_pass(meta, z0, Mp, to, a, tag, ck=ck, ident=ident)
    rec.update(cmd="solve", tag=tag, state=a.state, D=meta["D"], M=int(meta["M"]), L=L, Mp=Mp, s0=s0,
               threads=a.threads, kernel=kernel_info())
    if fin:
        save_state(final, fin[0], fin[1], meta, Mp, to, fin[2])
        rec["saved"] = final
    append(a.out, rec)
    if fin:
        retire_ckpt(ck)
    log(f"{tag}: " + (f"Delta {rec['Delta']:.16f} {rec['Delta_lo']:+.3e}, converged {rec['converged']}, plans "
                      f"{rec.get('steps_start')} / {rec.get('steps_p1')} / {rec.get('steps_own')} steps, {rec['seconds']} s"
                      if "Delta" in rec else f"failed: {rec.get('failed')}"))
    return 0 if rec.get("converged") else 1


COST_DEFAULT = [os.path.join(STATES37, f"highm15_D3.7000_M{M}.npz") for M in (128, 160, 192, 256)]


def require_plan_at(plan_M, state_M, what):
    """refuse a measurement whose plan was made at another M than its state's: a plan carries the stiffness of its M's
    top harmonics, and the tangents of a larger M overflow on it (run 2 on win2: the D 3.7 state resampled to M 320 on
    the M 256 state's plan, a nan pivot)"""
    if plan_M != state_M:
        raise SystemExit(f"{what}: the plan was made at M {plan_M}, the state is at M {state_M}; a measurement needs the "
                         f"plan of its own M")


def cost_cmd(a):
    """c(M) from one f64 Newton call per repetition at each given state's own M, on that state's stored plan (the plan
    of its pass-1 state, the plan the legs' last calls ran on), as wall seconds per plan step. step_timing's second
    entry is kept but not used: in block storage it is the staircase LU's fill, a copy, since multishoot computes
    every block's columns before the factorization (multishoot 3.3.0 fast.rs jacobian_staircase)."""
    from raw_out import require_raw
    require_raw(a.out)
    paths = a.states or COST_DEFAULT
    want = {int(v) for v in a.Ms.split(",")} if a.Ms else None
    jobs = []
    for path in paths:
        zh, zl, meta = load_state(path)
        M = int(meta["M"])
        if want is not None and M not in want:
            continue
        jobs.append((path, zh, zl, meta, M))
    if not jobs:
        raise SystemExit("cost: no state at the requested M")
    # every plan checked before any kernel work (--mutate plan_m, the control: the first state's plan for all)
    for path, zh, zl, meta, M in jobs:
        plan_M = jobs[0][4] if a.mutate == "plan_m" else M
        require_plan_at(plan_M, M, os.path.basename(path))
    from hpkernel.driver import newton_step
    info = kernel_info()
    done = {r.get("state") for r in read_done(a.out) if r.get("cmd") == "cost" and r.get("c_source") == "wall"}
    for path, zh, zl, meta, M in jobs:
        if path in done:
            continue
        segs = meta["segs"]
        n = nsteps(segs)
        to = tuple(meta["map"]) if meta.get("map") else None   # a state saved by solve: its own mapped grid
        s = solver_for(meta, M, to, a.threads, segs=segs)
        reps = []
        for _ in range(a.reps):
            t0 = time.time()
            newton_step(s, zh, zl, "f64", MIXED)
            tw = time.time() - t0
            tr, tf, tfac, ts = s.step_timing()
            reps.append(dict(t_call=round(tw, 3), t_res=tr, t_fill=tf, t_factor=tfac, t_solve=ts))
        tw = sorted(r["t_call"] for r in reps)
        rec = dict(cmd="cost", state=path, D=meta["D"], M=M, map=list(to) if to else None, steps=n, threads=a.threads,
                   reps=reps, c_source="wall",
                   c=tw[len(tw) // 2] / n, c_spread=[tw[0] / n, tw[-1] / n], kernel=info)
        append(a.out, rec)
        log(f"{os.path.basename(path)}: M {M}, one f64 Newton call {tw[len(tw) // 2]:.1f} s on its {n}-step plan, "
            f"c(M) {rec['c']:.3e} s per step (range {rec['c_spread'][0]:.3e}..{rec['c_spread'][1]:.3e} over {a.reps})")
    return 0


# ------------------------------------------------------------------ stored (no kernel)
def stored(a):
    """the stored states' Delta, plan steps (per segment; the segment of the most steps and its share) and top
    harmonics, and the Delta differences between consecutive M at each D (the uniform grid's resolution error)"""
    rows = []
    for path in a.states:
        zh, zl, meta = load_state(path)
        M, K = int(meta["M"]), len(meta["nodes"])
        segs = meta.get("segs") or []
        per = [len(g) - 1 for g in segs]
        b = int(np.argmax(per)) if per else -1
        tail, blk = tail_ratio(zh, zl, M, K)
        rows.append(dict(state=os.path.basename(path), D=meta["D"], M=M, Delta=delta_of(zh, zl), steps=sum(per),
                         seg_max=b, seg_x=[meta["nodes"][b], meta["nodes"][b + 1]] if b >= 0 else [0.0, 0.0],
                         seg_steps=per[b] if per else 0, outer=sum(n for n, x in zip(per, meta["nodes"]) if x >= 0.6),
                         tail=tail, tail_block=blk))
    rows.sort(key=lambda r: (-r["D"], r["M"]))
    print("| D | M | Delta | Delta - Delta(previous M) | plan steps | busiest segment (x) | its steps | steps from x 0.6 "
          "| top harmonics |")
    print("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows):
        prev = rows[i - 1] if i and rows[i - 1]["D"] == r["D"] else None
        dd = f"{ddiff(r['Delta'], prev['Delta']):+.2e}" if prev else ""
        print(f"| {r['D']} | {r['M']} | {r['Delta'][0]!r} | {dd} | {r['steps']} | {r['seg_x'][0]:g}-{r['seg_x'][1]:g} | "
              f"{r['seg_steps']} | {r['outer']} | {r['tail']:.1e} ({r['tail_block']}) |")
    # the geometric estimate of the top M's error from three equally spaced M: differences d1, d2, ratio q = d2 / d1,
    # Delta(M3) - Delta(inf) ~ d2 q / (1 - q) (an estimate that assumes geometric convergence, not a bound)
    for i in range(2, len(rows)):
        r1, r2, r3 = rows[i - 2], rows[i - 1], rows[i]
        if r1["D"] == r2["D"] == r3["D"] and r2["M"] - r1["M"] == r3["M"] - r2["M"]:
            d1, d2 = ddiff(r2["Delta"], r1["Delta"]), ddiff(r3["Delta"], r2["Delta"])
            q = d2 / d1 if d1 else float("inf")
            if not abs(q) < 1:                               # equal Deltas or no decay: no geometric estimate
                print(f"D {r3['D']}: from M {r1['M']}, {r2['M']}, {r3['M']}: the differences do not decay "
                      f"({d1:+.1e}, {d2:+.1e}); no geometric estimate")
                continue
            print(f"D {r3['D']}: from M {r1['M']}, {r2['M']}, {r3['M']}: ratio of differences {q:.3g}, geometric estimate "
                  f"of Delta({r3['M']}) - Delta(inf) {d2 * q / (1 - q):+.1e}")
    return 0


# ------------------------------------------------------------------ report
def c_model(M, Mu, cost):
    """c(M) / c(Mu): from the cost records when they have both M, else the M^3 law"""
    if cost and M in cost and Mu in cost:
        return cost[M] / cost[Mu]
    return None


def report(a):
    recs = [r for f in a.files for r in read_done(f)]
    zr, zrl, mr = load_state(a.ref)
    # --mutate delta (the report's control): every stored reference Delta moved by 1e-10, more than the low-D
    # resolution errors (4e-11, 8e-11); the identity and resolution checks must then fail
    shift = 1e-10 if a.mutate == "delta" else 0.0
    Dref = (delta_of(zr, zrl)[0] + shift, delta_of(zr, zrl)[1])
    e_u = None
    if a.ref2:
        z2, z2l, m2 = load_state(a.ref2)
        e_u = abs(ddiff(delta_of(zr, zrl), delta_of(z2, z2l)))     # unshifted: the bound stays the uniform grid's
    # c(M) from the cost records of wall seconds (earlier records timed the block fill); at an M measured on several
    # states, the one on the longest plan, where the per-call series cost weighs least
    cost, csteps = {}, {}
    for f in a.cost or []:
        for r in read_done(f):
            if r.get("cmd") == "cost" and r.get("c_source") == "wall" and r["steps"] > csteps.get(r["M"], 0):
                cost[r["M"]], csteps[r["M"]] = r["c"], r["steps"]
    # the latest attempt of each configuration (a retry supersedes an earlier failure), in the files' order
    latest = {}
    for r in recs:
        if r.get("cmd") == "solve" and r.get("D") == mr["D"]:
            latest[solve_key(r)] = r
    sol = list(latest.values())
    if a.mutate == "lenient":                                 # the selftest's control: failed records dropped
        sol = [r for r in sol if "Delta" in r]
    # the cost comparator: the uniform grid at --uniform's M (default the reference's), on that state's own plan (an
    # L 1 solve record at that M, else the state's stored plan); a call's measured seconds from that record
    mu = load_state(a.uniform)[2] if a.uniform else mr
    Mu = int(mu["M"])
    ref_rec = [r for r in sol if r["L"] == 1 and r["Mp"] == Mu and r.get("converged")]
    S_u = ref_rec[-1]["steps_own"] if ref_rec else nsteps(mu["segs"])
    tj_u = (np.median(call_seconds(ref_rec[-1]["hist2"])) if ref_rec and call_seconds(ref_rec[-1]["hist2"]) else None)
    print(f"reference: {a.ref}: D {mr['D']}, uniform M {int(mr['M'])}, Delta {Dref[0]!r} {Dref[1]:+.3e}"
          + (f", |Delta - Delta(ref2)| {e_u:.2e}" if e_u is not None else ""))
    print(f"cost against the uniform grid at M {Mu} on its own plan, {S_u} steps"
          + (f"; one Newton call there {tj_u:.1f} s" if tj_u else ""))
    plans = {(r["L"], r["Mp"]): r for r in recs if r.get("cmd") == "plan" and r.get("D") == mr["D"] and "steps" in r}
    rows = []
    hdr = ("L", "M'", "Delta - ref", "<1e-12", "<e_u", "plan start", "pass 2", "own", "own/S_u", "cost M^3", "cost c(M)",
           "cost measured")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in sorted(sol, key=lambda r: (-r["L"], r["Mp"])):
        if "Delta" not in r:
            # a failed size stays in the decision as unsuccessful: no smaller M' is certified past it
            rows.append(dict(L=r["L"], Mp=r["Mp"], s0=r["s0"], dDelta=None, ok12=False, ok_eu=False, converged=False,
                             failed=r.get("failed", "no Delta")))
            print(f"| {r['L']:g} | {r['Mp']} | failed: {r.get('failed')} |")
            continue
        d = ddiff((r["Delta"], r["Delta_lo"]), Dref)
        so = r["steps_own"]
        r3 = so / S_u * (r["Mp"] / Mu) ** 3
        cm = c_model(r["Mp"], Mu, cost)
        rc = so / S_u * cm if cm else None
        tj = np.median(call_seconds(r["hist2"])) if call_seconds(r.get("hist2", [])) else None
        rm = tj / tj_u if (tj and tj_u) else None
        row = dict(L=r["L"], Mp=r["Mp"], s0=r["s0"], dDelta=d, ok12=bool(r["converged"]) and abs(d) <= TOL_DELTA,
                   ok_eu=(abs(d) <= e_u) if e_u else None,
                   steps_start=r.get("steps_start"), steps_p1=r.get("steps_p1"), steps_own=so, ratio_steps=so / S_u,
                   cost_M3=r3, cost_c=rc, cost_measured=rm, converged=r["converged"])
        rows.append(row)
        print(f"| {r['L']:g} | {r['Mp']} | {d:+.2e} | {'yes' if row['ok12'] else 'no'} | "
              f"{'yes' if row['ok_eu'] else 'no' if e_u else '-'} | {row['steps_start']} | {row['steps_p1']} | {so} | "
              f"{so / S_u:.2f} | {r3:.3f} | {f'{rc:.3f}' if rc else '-'} | {f'{rm:.3f}' if rm else '-'} |")
    need = {}
    for L in sorted({r["L"] for r in rows}, reverse=True):
        rs = sorted([r for r in rows if r["L"] == L], key=lambda r: r["Mp"])
        for i, r in enumerate(rs):
            if all(x["ok12"] for x in rs[i:]):
                need[L] = r
                break
    print("modes needed for Delta to 1e-12 of the reference, per L (the smallest M' from which every larger M' tried "
          "converged and agrees):")
    for L, r in need.items():
        print(f"  L {L:g}: M' {r['Mp']} ({Mu / r['Mp']:.2f}x fewer than M {Mu}); own plan {r['steps_own']} steps "
              f"({r['ratio_steps']:.2f}x); projected cost per call {r['cost_M3']:.3f} (M^3)"
              + (f", {r['cost_c']:.3f} (c(M))" if r["cost_c"] else "") + (f", {r['cost_measured']:.3f} measured" if r["cost_measured"] else ""))
    # the identity checks (fixed before the runs): an L 1 solve from a stored uniform state, at the state's own M or
    # at the M of --ref / --ref2 / --uniform, converges to that stored state's Delta within 1e-20 (two solutions of
    # the same discrete equations at |F| 1e-28), and at the own M its own plan has the stored plan's steps within 10%.
    # The plan is not a function of the solution alone: its tolerance (1e-30) sits near double-double's resolution,
    # so copies of one converged state that differ below 1e-25 get plans a few percent apart (the stored plan is the
    # pass-1 state's; a first bound of 1% failed on the Mac's D 3.7 pilot for this reason, notes/LOW_D_SPEEDUP.md)
    known = {int(mr["M"]): (Dref, nsteps(mr["segs"]))}
    if a.ref2:
        known[int(m2["M"])] = ((delta_of(z2, z2l)[0] + shift, delta_of(z2, z2l)[1]), nsteps(m2["segs"]))
    if a.uniform:
        zu, zul, _ = load_state(a.uniform)
        known[Mu] = ((delta_of(zu, zul)[0] + shift, delta_of(zu, zul)[1]), nsteps(mu["segs"]))
    idok = True
    for r in sol:
        if r["L"] == 1 and r["Mp"] in known and ("Delta" not in r or not r.get("converged")):
            idok = False
            print(f"identity: L 1 at M {r['Mp']} from M {r['M']}: not converged ({r.get('failed', 'it_max')}): FAIL")
        elif r["L"] == 1 and r["Mp"] in known:
            dD = ddiff((r["Delta"], r["Delta_lo"]), known[r["Mp"]][0])
            good = r["converged"] and abs(dD) <= 1e-20
            line = f"identity: L 1 at M {r['Mp']} from M {r['M']}: Delta - stored Delta(M {r['Mp']}) {dD:+.1e} (bound 1e-20)"
            if r["Mp"] == r["M"]:
                rs = r["steps_own"] / known[r["Mp"]][1]
                good &= abs(rs - 1) <= PLAN_SPREAD
                line += f", own plan {r['steps_own']} against the stored {known[r['Mp']][1]} steps ({rs:.3f}; bound {PLAN_SPREAD:.0%})"
            idok &= good
            print(line + f": {'PASS' if good else 'FAIL'}")
    if plans:
        print("start plans (plan records): " + ", ".join(f"L {L:g} M' {Mp}: {r['steps']}" for (L, Mp), r in sorted(plans.items(), key=lambda kv: (-kv[0][0], kv[0][1]))))
    if e_u is not None:
        for L in sorted({r["L"] for r in sol if r["L"] != 1}, reverse=True):
            top = max((r for r in sol if r["L"] == L), key=lambda r: r["Mp"])
            if "Delta" not in top or not top.get("converged"):
                idok = False
                print(f"check: L {L:g} at its largest M' {top['Mp']}: not converged ({top.get('failed', 'it_max')}): FAIL")
                continue
            dd = ddiff((top["Delta"], top["Delta_lo"]), Dref)
            good = abs(dd) <= e_u
            idok &= good
            print(f"check: L {L:g} at its largest M' {top['Mp']}: |Delta - ref| {abs(dd):.1e} against the uniform "
                  f"grid's own resolution error {e_u:.1e}: {'PASS' if good else 'FAIL'}")
    if a.json:
        json.dump(dict(ref=a.ref, ref2=a.ref2, D=mr["D"], Mu=Mu, S_u=S_u, e_u=e_u, rows=rows,
                       need={str(L): r for L, r in need.items()}, checks_pass=bool(idok)), open(a.json, "w"), indent=1)
    return 0 if idok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--threads", type=int, default=4)
    common.add_argument("--tol_step", type=float, default=1e-30)
    common.add_argument("--hmax", type=float, default=0.05)
    common.add_argument("--s0", default="auto")
    common.add_argument("--allow_unraw", action="store_true", help="write outside the raw-data repository (tests)")
    p = sub.add_parser("selftest")
    p.add_argument("--mutate", default="", choices=["", "lo", "parity", "dps40", "inverse", "gp", "phase_gp", "lenient",
                                                    "unbound"])
    p.add_argument("--out", default="")
    p = sub.add_parser("check", parents=[common])
    p.add_argument("--Ls", default="0.9,0.7,0.5")
    p.add_argument("--mutate", default=None, choices=[None, "gp", "phase_gp"])
    p.add_argument("--out", default="")
    p.add_argument("--states", default="")
    p.add_argument("--tol1", type=float, default=1e-14)
    p.add_argument("--tol", type=float, default=1e-28)
    p.add_argument("--it1", type=int, default=4)
    p.add_argument("--it2", type=int, default=5)
    p = sub.add_parser("plan", parents=[common])
    p.add_argument("state")
    p.add_argument("--Ls", default="1")
    p.add_argument("--Ms", default="")
    p.add_argument("--pairs", default="", help="L:M',L:M',... in place of the product of --Ls and --Ms")
    p.add_argument("--out", required=True)
    p = sub.add_parser("solve", parents=[common])
    p.add_argument("state")
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--M", type=int, default=0)
    p.add_argument("--pairs", default="", help="L:M',L:M',... solved one after another")
    p.add_argument("--states", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tol1", type=float, default=1e-14)
    p.add_argument("--tol", type=float, default=1e-28)
    p.add_argument("--it1", type=int, default=4)
    p.add_argument("--it2", type=int, default=5)
    p = sub.add_parser("cost", parents=[common])
    p.add_argument("states", nargs="*", help="converged states, each measured at its own M on its own plan "
                   "(default: the D 3.7 ladder at M 128, 160, 192, 256)")
    p.add_argument("--Ms", default="", help="only the states at these M")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--mutate", default="", choices=["", "plan_m"], help="plan_m: the first state's plan for every M "
                   "(the control: the plan check must refuse it)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("stored")
    p.add_argument("states", nargs="+")
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    p.add_argument("--ref", required=True)
    p.add_argument("--ref2", default="")
    p.add_argument("--uniform", default="", help="the uniform state the cost is compared with (default --ref)")
    p.add_argument("--mutate", default="", choices=["", "delta", "lenient"],
                   help="delta: the control of the report's checks; lenient: the selftest's (failed records dropped)")
    p.add_argument("--cost", nargs="*")
    p.add_argument("--json", default="")
    a = ap.parse_args()
    # the planner caps every step at hmax and aims each at tol_step: a step limit of 0 never ends a plan
    for name in ("hmax", "tol_step"):
        v = getattr(a, name, None)
        if v is not None and not (np.isfinite(v) and v > 0):
            ap.error(f"--{name} must be a positive finite number, not {v}")
    return dict(selftest=selftest, check=check, plan=plan_cmd, solve=solve_cmd, cost=cost_cmd, stored=stored,
                report=report)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
