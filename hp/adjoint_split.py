#!/usr/bin/env python3
"""The adjoint split of the chain's departure past D 245 (#905). At a stored state z of the pinned M = 32 system
(F(z) = 0, Jacobian J), a change dF of the residual at fixed z moves the solution's Delta by -psi^T dF to first order,
with psi = J^-T e_Delta. psi is formed once per state, by one transposed solve through the octo-double LU, and each
setting's dF is the residual of the system with that one setting changed, at the same z, minus the base's. No Newton
step is taken, so no cross term of an offset start enters (#852, #857).

Reference: M. B. Giles, E. Suli, Adjoint methods for PDEs: a posteriori error analysis and postprocessing by duality,
Acta Numerica 11 (2002) 145-236.

    cd hp && python3 adjoint_split.py run RAW STATE --out FILE.json [--threads 4] [--settings S1,S2,...]
    cd hp && python3 adjoint_split.py planted RAW STATE [--threads 4] [--mutate notrans|shift]

The base is the state's own system: its ks, kc and kt, its stored plan, and its own held values (eight parts, as
anm_run.py --held_o 1 holds them at the state's D). Settings, each changed alone:
  ks<K>        the horizon series to order K, on the stored plan
  ks<K>p       ks K on a plan made again (anm_run.py --ks K, which plans again); its dF is against plan (below), so
               that it holds ks alone, as #696's r<K>k - r130k does
  plan         the plan made again at 1e-32 (anm_run.py --keep_plan 0)
  plan<T>      the plan made again at tolerance T
  kc<K>, kt<K> the centre and Taylor orders, on the stored plan
  held<d>      the held values of data/held_fit_d<d>.json in place of data/held_fit.json's, both carried from
               D0 = 73.0, the end of the fits' full-system states (v(D) = v(D0) + fit(D) - fit(D0) along the chains)
Each line gives psi^T dF (the setting's share of chain - truth, if the changed setting is the truer one) and
-psi^T dF (the predicted Delta shift when the setting changes).

Checks, each of which can fail (exit 1): (1) the rebuilt base reproduces the state, max|F| <= 1e-50; (2) the Newton
step's Delta at the state equals -psi^T F(z) to 1e-8; (3) the transposed and the plain solve agree, psi^T r = (J^-1
r)_Delta for a random r, to 1e-20. `planted`: dF = (F(z + eps x) - F(z - eps x)) / 2 for x = e_Delta and two random
unit x must give psi^T dF = eps x_Delta to 1e-6. The difference is central because psi carries the weak direction's
1/sigma_min: a one-sided difference's second order, times |psi|, outgrows the signal before the residual's octo-double
noise falls below it; at eps 1e-45 the third order and the noise are both far below. Its mutations must fail: notrans
(psi = J^-1 e_Delta), shift (psi moved by one row).
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DPS = 200
D0_HELD = "73.0"


def say(s):
    print(s, flush=True)


def tomp(z):
    """eight float64 parts as mpf entries, each its parts' exact sum"""
    P = [np.asarray(p, dtype=float) for p in z]
    return [mp.fsum(mp.mpf(float(P[q][i])) for q in range(len(P))) for i in range(len(P[0]))]


def unit8(L, i):
    e = np.zeros(L)
    e[i] = 1.0
    return [e] + [np.zeros(L) for _ in range(7)]


class State:
    """a stored eight-part state of #603's chain and the pieces anm_run.py builds at its D"""

    def __init__(self, raw, rel, threads):
        from anm_run import load_state
        from octo_corrector import load_o, o8
        from pinning import band_coeffs
        self.path = os.path.join(raw, rel)
        self.rel = rel
        self.zh, self.zl, self.m = load_state(self.path)
        self.z8 = o8(load_o(self.path)[0])
        m = self.m
        self.M, self.nodes, self.kf = int(m["M"]), list(m["nodes"]), int(m["kfreeze"])
        self.kt, self.kc, self.ks = m.get("kt", 36), m.get("kc", 146), m.get("ks", 130)
        self.D = mp.mpf(m["D_str"])
        self.segs = m["segs"]
        self.ks_held = [k for k in range(self.kf + 1, 14) if k % 2 == 1]
        self.held = band_coeffs(self.zh[:self.M], self.zl[:self.M], self.M, self.ks_held, exact=True,
                                rest=[np.asarray(p)[:self.M] for p in self.z8[2:]])
        self.threads = threads
        self.L = len(self.z8[0])

    def system(self, ks=None, kc=None, kt=None, plan=None, held=None):
        """(solver, PinnedO, segs): the state's system with the given settings changed; plan None keeps the stored
        plan, a float plans again at that tolerance"""
        from hpkernel.driver import make_solver
        from octo_corrector import PinnedO
        from pinning import PinnedSolver, band_field
        sn = make_solver(self.D, self.M, nodes=self.nodes, kt=kt or self.kt, kc=kc or self.kc, ks=ks or self.ks,
                         threads=self.threads, kfreeze=self.kf, horizon_alpha=0)
        segs = self.segs if plan is None else sn.plan(self.zh, self.zl, float(plan), 0.05)
        sn.set_segs(segs)
        h = band_field(held if held is not None else self.held, self.ks_held, self.M, parts=8)
        spn = PinnedSolver(sn, self.M, h[0], h[1], h_rest=h[2:])
        return sn, PinnedO(sn, spn.ch, spn.cl, c8=spn.c8), segs

    def held_from(self, deg):
        """the held values with data/held_fit_d<deg>.json's increment from D0 in place of data/held_fit.json's"""
        from pinning import HeldFit
        f0 = HeldFit(os.path.join(HERE, "data", "held_fit.json"))
        f1 = HeldFit(os.path.join(HERE, "data", f"held_fit_d{deg}.json"))
        i0, i1 = f0.increment(mp.mpf(D0_HELD), self.D, self.ks_held), f1.increment(mp.mpf(D0_HELD), self.D, self.ks_held)
        out, dmax = [], mp.mpf(0)
        for (a, b), (a0, b0), (a1, b1) in zip(self.held, i0, i1):
            da, db = a1 - a0, b1 - b0
            dmax = max(dmax, abs(da), abs(db))
            out.append((mp.mpf(a) + da, mp.mpf(b) + db))
        return out, dmax


def resid(so, z8):
    from octo_corrector import o8
    return o8(so.residual_o(z8))


def adjoint(st, sn, so, mutate=""):
    """psi = J^-T e_Delta through the LU at the state (PinnedO.newton_o factors it), with the Newton step's record"""
    from octo_corrector import o8
    t0 = time.time()
    d, Fh, Fl, pmin, pmax = so.newton_o(st.z8)
    t1 = time.time()
    e = unit8(st.L, st.L - 1)
    psi8 = o8(sn.olu_solve([np.ascontiguousarray(p) for p in e], mutate != "notrans"))
    psi = tomp(psi8)
    if mutate == "shift":
        psi = psi[1:] + psi[:1]
    return dict(psi=psi, step=tomp(o8(d)), pmin=float(pmin), pmax=float(pmax), t_lu=round(t1 - t0, 1),
                t_solve=round(time.time() - t1, 1))


def checks(st, sn, so, A, F0):
    """the three consistency checks of the docstring; returns (ok, record)"""
    from octo_corrector import o8
    rec, ok = {}, True
    fmax = float(max(abs(x) for x in F0))
    rec["F_base"] = fmax
    ok &= fmax <= 1e-50
    pF = mp.fdot(A["psi"], F0)
    dD = A["step"][-1]
    rec["step_Delta"], rec["minus_psiF"] = float(dD), float(-pF)
    e2 = abs(dD + pF) / max(abs(dD), mp.mpf(10) ** -75)
    rec["step_rel"] = float(e2)
    ok &= e2 <= 1e-8
    rng = np.random.default_rng(905)
    r = rng.standard_normal(st.L)
    y = tomp(o8(sn.olu_solve([np.ascontiguousarray(r)] + [np.zeros(st.L)] * 7, False)))
    pr = mp.fdot(A["psi"], [mp.mpf(float(v)) for v in r])
    e3 = abs(pr - y[-1]) / abs(y[-1])
    rec["transpose_rel"] = float(e3)
    ok &= e3 <= 1e-20
    say(f"checks: base max|F| {fmax:.1e} (<= 1e-50), Newton step Delta {float(dD):+.3e} against -psi^T F {float(-pF):+.3e} "
        f"(rel {float(e2):.1e}, <= 1e-8), psi^T r against (J^-1 r)_Delta rel {float(e3):.1e} (<= 1e-20) -> {'pass' if ok else 'FAIL'}")
    return ok, rec


def parse(argv):
    pos = [a for a in argv if not a.startswith("--")]
    opt = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in argv if a.startswith("--"))
    return pos, opt


def cmd_run(argv):
    pos, opt = parse(argv)
    raw, rel = pos[0], pos[1]
    mp.mp.dps = DPS
    st = State(raw, rel, int(opt.get("threads", 4)))
    say(f"state {rel}: D {mp.nstr(st.D, 20)}, M {st.M}, kt {st.kt}, kc {st.kc}, ks {st.ks}, kfreeze {st.kf}, "
        f"plan {sum(len(s) - 1 for s in st.segs)} steps, L {st.L}")
    t0 = time.time()
    sn, so, _ = st.system()
    F0 = tomp(resid(so, st.z8))
    say(f"base system built ({round(time.time() - t0, 1)} s)")
    A = adjoint(st, sn, so)
    pn = float(mp.sqrt(mp.fdot(A["psi"], A["psi"])))
    say(f"psi = J^-T e_Delta: |psi| {pn:.3e}, pivots {A['pmin']:.1e}..{A['pmax']:.1e} (LU {A['t_lu']} s, solve {A['t_solve']} s)")
    ok, crec = checks(st, sn, so, A, F0)
    out = dict(state=rel, D=mp.nstr(st.D, 30), checks=crec, checks_pass=bool(ok), psi_norm=pn, pivots=[A["pmin"], A["pmax"]],
               settings={})
    cache = {}
    sets = [s for s in opt.get("settings", "ks131,ks131p,plan").split(",") if s]
    for s in sets:
        t0 = time.time()
        kw, ref, note = {}, None, ""
        if s.startswith("ks") and s.endswith("p"):
            kw = dict(ks=int(s[2:-1]), plan=1e-32)
            ref = "plan"
        elif s.startswith("ks"):
            kw = dict(ks=int(s[2:]))
        elif s.startswith("plan"):
            kw = dict(plan=float(s[4:]) if len(s) > 4 else 1e-32)
        elif s.startswith("kc"):
            kw = dict(kc=int(s[2:]))
        elif s.startswith("kt"):
            kw = dict(kt=int(s[2:]))
        elif s.startswith("held"):
            hv, dmax = st.held_from(int(s[4:]))
            kw = dict(held=hv)
            note = f", held values moved by at most {mp.nstr(dmax, 3)}"
        else:
            raise SystemExit(f"unknown setting {s}")
        _, so1, segs1 = st.system(**kw)
        F1 = tomp(resid(so1, st.z8))
        cache[s] = F1
        if ref is not None:
            if ref not in cache:
                _, so0, _ = st.system(plan=1e-32)
                cache[ref] = tomp(resid(so0, st.z8))
            Fr = cache[ref]
        else:
            Fr = F0
        dF = [a - b for a, b in zip(F1, Fr)]
        p = mp.fdot(A["psi"], dF)
        dmax = float(max(abs(x) for x in dF))
        rec = dict(psi_dF=float(p), dDelta_pred=float(-p), dF_max=dmax, against=ref or "base",
                   plan_steps=sum(len(g) - 1 for g in segs1), seconds=round(time.time() - t0, 1))
        out["settings"][s] = rec
        say(f"  {s:>9} (against {ref or 'base'}{note}): max|dF| {dmax:.3e}, psi^T dF {float(p):+.4e}, "
            f"predicted Delta shift {float(-p):+.4e}, plan {rec['plan_steps']} steps, {rec['seconds']} s")
        if "out" in opt:
            json.dump(out, open(opt["out"], "w"), indent=1)
    if "out" in opt:
        json.dump(out, open(opt["out"], "w"), indent=1)
    return 0 if ok else 1


def cmd_planted(argv):
    from multidouble import od_axpy
    from octo_corrector import o8
    pos, opt = parse(argv)
    raw, rel = pos[0], pos[1]
    mp.mp.dps = DPS
    mut = opt.get("mutate", "")
    st = State(raw, rel, int(opt.get("threads", 4)))
    sn, so, _ = st.system()
    F0 = tomp(resid(so, st.z8))
    A = adjoint(st, sn, so, mutate=mut)
    eps = 1e-45
    rng = np.random.default_rng(7)
    xs = [("e_Delta", unit8(st.L, st.L - 1)[0])]
    for j in range(2):
        x = rng.standard_normal(st.L)
        xs.append((f"random{j}", x / np.linalg.norm(x)))
    ok = True
    for name, x in xs:
        x8 = [np.ascontiguousarray(x)] + [np.zeros(st.L)] * 7
        zp, zm = o8(od_axpy(eps, *x8, *st.z8)), o8(od_axpy(-eps, *x8, *st.z8))
        Fp, Fm = tomp(resid(so, zp)), tomp(resid(so, zm))
        dz = [(a - b) / 2 for a, b in zip(tomp(zp), tomp(zm))]
        dF = [(a - b) / 2 for a, b in zip(Fp, Fm)]
        p = mp.fdot(A["psi"], dF)
        rel_ = abs(p - dz[-1]) / abs(dz[-1]) if dz[-1] != 0 else abs(p)
        good = rel_ <= 1e-6
        ok &= bool(good)
        say(f"planted {name}: eps x_Delta {float(dz[-1]):+.6e}, psi^T dF {float(p):+.6e}, rel {float(rel_):.1e} -> "
            f"{'pass' if good else 'FAIL'}")
    say(f"planted{' (mutation ' + mut + ')' if mut else ''}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        sys.exit(cmd_run(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "planted":
        sys.exit(cmd_planted(sys.argv[2:]))
    raise SystemExit(__doc__)
