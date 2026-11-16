#!/usr/bin/env python3
"""The adjoint split of the chain's departure past D 245 (#905). At a stored state z of the pinned M = 32 system
(F(z) = 0, Jacobian J), a change dF of the residual at fixed z moves the solution's Delta by -psi^T dF to first order,
with psi = J^-T e_Delta. psi is formed once per state, by one transposed solve through the octo-double LU, and each
setting's dF is the residual of the system with that one setting changed, at the same z, minus the base's. No Newton
step is taken, so no cross term of an offset start enters (#852, #857).

Reference: M. B. Giles, E. Suli, Adjoint methods for PDEs: a posteriori error analysis and postprocessing by duality,
Acta Numerica 11 (2002) 145-236.

    cd hp && python3 adjoint_split.py run RAW STATE --out FILE.json [--threads 4] [--settings S1,S2,...] [--project 1]
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
  heldstate:REL  the held values another stored state at the same D carries (its own chain's), against which its
               Delta difference is the measured response (the held control)
Each line gives psi^T dF (the setting's share of chain - truth, if the changed setting is the truer one) and
-psi^T dF (the predicted Delta shift when the setting changes).

--project 1 eliminates the horizon data nonlinearly first, as the chain's corrector does (ElimO): z_s = P_s(z) re-solves
only the horizon data with the setting's re-seeder, and the prediction is (z_s - z_r)_Delta - psi^T (F_s(z_s) -
F_r(z_r)), z_r = P_r(z) for the reference. To first order this is the adjoint of the reduced system (the Schur
complement on the horizon block); without it a change of the horizon rows falls outside the full system's linear
range (#905: the order-131 shift at D 245.14 came out -4.9e-11 against Newton's +3.8e-13). The zero control is
psi^T F_base(P_base(z)), the projection's own error seen through psi.

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


def to_parts(v, n=8):
    """an mpf vector as n float64 arrays, each entry split exactly"""
    out = [np.zeros(len(v)) for _ in range(n)]
    for i, x in enumerate(v):
        r = mp.mpf(x)
        for q in range(n):
            f = float(r)
            out[q][i] = f
            r -= f
    return out


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
        self._rs = {}

    def project(self, z8, ks=None, kc=None, kt=None):
        """z8 with only its horizon data re-solved by the chain's re-seeder of the given orders (reseed.Reseeder with
        anm_run.py's --reseed_q 1 --reseed_o 3 --reseed_start pred, as ElimO.project): (z', how)"""
        from octo_corrector import o8, q4_of_o, with_horizon
        from reseed import Reseeder
        key = (ks or self.ks, kc or self.kc, kt or self.kt)
        if key not in self._rs:
            self._rs[key] = Reseeder(dict(M=self.M, nodes=self.nodes, kt=key[2], kc=key[1], ks=key[0]), self.D,
                                     threads=self.threads, horizon_alpha=0, qd=True, od=3, qd_start="pred")
        rs = self._rs[key]
        zp, how = rs.project(q4_of_o(z8))
        return (z8 if how == "failed" else o8(with_horizon(z8, zp, rs))), how

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
    proj = opt.get("project", "0") == "1"
    if proj:
        zb, howb = st.project(st.z8)
        Fb = tomp(resid(so, zb))
        p0 = mp.fdot(A["psi"], Fb)
        dzb = float(max(abs(a - b) for a, b in zip(tomp(zb), tomp(st.z8))))
        out["zero_control"] = dict(projection=howb, dz_max=dzb, psi_F=float(p0), F_max=float(max(abs(x) for x in Fb)))
        say(f"zero control: the base projected by its own re-seeder ({howb}), max|P z - z| {dzb:.2e}, max|F| "
            f"{float(max(abs(x) for x in Fb)):.2e}, psi^T F {float(p0):+.3e} (|.| <= 1e-14)")
    else:
        zb, Fb = st.z8, F0
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
        elif s.startswith("heldstate:"):
            other = State(raw, s[10:], st.threads)
            hv = [(mp.mpf(a), mp.mpf(b)) for a, b in other.held]
            dmax = max(max(abs(mp.mpf(a1) - mp.mpf(a0)), abs(mp.mpf(b1) - mp.mpf(b0))) for (a1, b1), (a0, b0) in zip(hv, st.held))
            kw = dict(held=hv)
            dD_states = tomp(other.z8)[-1] - tomp(st.z8)[-1]
            note = f", that state's held values, moved by at most {mp.nstr(dmax, 3)}; its Delta - this one's {mp.nstr(dD_states, 6)}"
        elif s.startswith("held"):
            hv, dmax = st.held_from(int(s[4:]))
            kw = dict(held=hv)
            note = f", held values moved by at most {mp.nstr(dmax, 3)}"
        else:
            raise SystemExit(f"unknown setting {s}")
        _, so1, segs1 = st.system(**kw)
        okw = {k: v for k, v in kw.items() if k in ("ks", "kc", "kt")}
        z1, how1 = st.project(st.z8, **okw) if proj else (st.z8, "-")
        F1 = tomp(resid(so1, z1))
        if ref is not None:
            if ref not in cache:
                _, so0, _ = st.system(plan=1e-32)
                cache[ref] = tomp(resid(so0, zb))
            Fr, zr = cache[ref], zb
        else:
            Fr, zr = Fb, zb
        dF = [a - b for a, b in zip(F1, Fr)]
        p = mp.fdot(A["psi"], dF)
        dzD = tomp(z1)[-1] - tomp(zr)[-1]
        dh = float(max(abs(a - b) for a, b in zip(tomp(z1), tomp(zr))))
        dmax = float(max(abs(x) for x in dF))
        pred = dzD - p
        rec = dict(psi_dF=float(p), dDelta_pred=float(pred), dF_max=dmax, against=ref or "base", projection=how1,
                   dz_projection_max=dh, dz_projection_Delta=float(dzD), plan_steps=sum(len(g) - 1 for g in segs1),
                   seconds=round(time.time() - t0, 1))
        out["settings"][s] = rec
        say(f"  {s:>9} (against {ref or 'base'}{note}{', projected ' + how1 + f', max|z_s - z_r| {dh:.2e}' if proj else ''}): "
            f"max|dF| {dmax:.3e}, psi^T dF {float(p):+.4e}, predicted Delta shift {float(pred):+.4e}, "
            f"plan {rec['plan_steps']} steps, {rec['seconds']} s")
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
    if opt.get("diag"):
        checks(st, sn, so, A, F0)
        ps = [abs(v) for v in A["psi"]]
        i = max(range(st.L), key=lambda j: ps[j])
        say(f"|psi| {float(mp.sqrt(mp.fdot(A['psi'], A['psi']))):.3e}, max |psi_i| {float(ps[i]):.3e} at row {i} of {st.L}")
    epss = [float(e) for e in opt.get("eps", "1e-45").split(",")]
    rng = np.random.default_rng(7)
    xs = [("e_Delta", unit8(st.L, st.L - 1)[0])]
    for j in range(2):
        x = rng.standard_normal(st.L)
        xs.append((f"random{j}", x / np.linalg.norm(x)))
    ok = True
    for name, x, eps in [(nm + (f" eps {e:g}" if len(epss) > 1 else ""), x, e) for nm, x in xs for e in epss]:
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
        if opt.get("diag"):
            y = tomp(o8(sn.olu_solve([np.ascontiguousarray(q) for q in to_parts(dF)], False)))
            ey = max(abs(a - b) for a, b in zip(y, dz))
            say(f"    max|dF| {float(max(abs(v) for v in dF)):.3e}; plain solve (J^-1 dF)_Delta {float(y[-1]):+.6e}, "
                f"max|J^-1 dF - eps x| {float(ey):.3e}")
    say(f"planted{' (mutation ' + mut + ')' if mut else ''}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        sys.exit(cmd_run(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "planted":
        sys.exit(cmd_planted(sys.argv[2:]))
    raise SystemExit(__doc__)
