#!/usr/bin/env python3
"""The factored raise of the horizon order (notes/FACTORED_HORIZON.md 5.3, #1035 stage 2).

A chain state converged at one horizon order is taken to a higher order ks by Newton's method on the pinned system
(octo-double, the kernel's exact Jacobian and LU, as anm_run.py's base polish), with each trial point's a_h projected
by method A: the a rows of the horizon block solved for a_h, with the last node's a held and the rest of the trial
(m_h included) the step's. That is section 3's change of variables, a_K in place of a_h, under which the horizon rows'
curvature loses its G-sized part (5.2: by 1e-60 to 1e-64 at D 242.36), so the trial carries no G-sized second-order
term. The corrector's own projection (ElimO, --rung ma here) solves the m and a rows for m_h and a_h from the trial's
own data, the linear step's, which miss by that term, about 1e30 at D 262.0099 (#1064); its Newton also moves m_h
linearly, which meets the same curvature. Here the projection starts from a predictor of second order in the step:
a_h + lam da_h + lam^2 w, with w = -J_a^-1 q2, q2 the a rows' second difference along the step (at tau and tau/4,
which must agree to 10%) and J_a their block in a_h. When the projection from there fails, the step is split into 2,
4, ... stages, each started from the last stage's projected data plus the predictor's increment. A trial is accepted by
Deuflhard's natural test (the simplified correction through the LU at the iterate, at most (1 - lam/4) times the step),
or when its max |F| <= tol.

    python3 factored_raise.py RAW OUTDIR --state STATE.npz --ks 315 [--threads 10] [--tol 1e-80] [--it_max 12]
        [--halvings 10] [--stages 16] [--plan stored|new] [--rung a|ma] [--proj_bar 1e-60] [--ref REF.npz --bar 1e-60]
        [--mutate=old|linear|order] [--allow_unraw]
    python3 factored_raise.py selftest [STATE.npz] [--mutate=nopred|old|linear|gate]
    python3 factored_raise.py polishgate POLISH.log POLISH_OUTDIR [--bar 1e-60]
    python3 factored_raise.py stepgate STEP.log STEP_OUTDIR

STATE and REF are paths inside RAW (the raw-data repository); OUTDIR must be a task folder of it (hp/raw_out.py).
The run prints the kernel it uses and probes the methods it needs. --plan stored (the default) keeps the state's own
plan at the new order, so that a later anm_run.py polish with --keep_plan 1 solves the same system; new plans again at
1e-32 as anm_run.py does for a changed --ks. The raised state is written as OUTDIR/raise_pin<k>_D<D>.npz in the
chain's eight-part layout (octo_corrector.save_o), its meta the state's with ks, segs, source and a record of the
raise; a run that does not converge writes OUTDIR/lastiter_pin<k>_D<D>.npz instead. --ref compares the result with a
reference state of the same plan and order, entry by entry (exact sums of the eight parts): it passes when every
difference is at most --bar. Exit 1 when the raise does not converge to --tol, or when the comparison fails.

--proj_bar is the projection's acceptance: its a rows must end at or below it (they end at their floor).

Mutations (each a control): old, no projection (the old map, the corrector without ElimO); linear, the projection
from the trial's own data, with no stages (with --rung ma, the corrector's ElimO); order, the raise to ks - 1 while
--ref is of ks, which the comparison must refuse.

selftest needs no kernel: a three-unknown model of the chain with a mass-mode gain G = 1e30 in its rung row and in its
c row, a weak direction (sigma 1e-6) and a pole in the rung at a miss of 1e3. Raising it moves the weak unknown by
1e-10, so the linear step misses the rung by 1e10. The factored raise must converge and match the model's solution to
1e-80, and the old map, the linear projection, the projection with the second-order term left out, and with its sign
flipped must each fail. --mutate=nopred runs the main case without the second-order term, and the selftest must then
fail. It also checks the state writer on a stored state (read back exactly), the comparison on planted copies, and
the two gates on planted logs (--mutate=gate adds a case that must pass and cannot).

polishgate is stage 3's pass rule for anm_run.py's polish of a state at its order (5.3), by the chain's own rules:
(a) its start line says the polish was skipped, or (b) its last octo-double Newton line says converged and the base is
saved. --bar 1e-60 adds part 4's bound to both (gate903d and gate973). stepgate is the chain step's: anm_run.py
--steps 1 wrote its jet_pin file and saved an accepted step's state, and did not stop. Each exits 1 when it fails.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJ_MAX = 12                                      # project_a's corrections at most
TAUS = ((2.0 ** -10, 2.0 ** -12), (2.0 ** -20, 2.0 ** -22), (2.0 ** -30, 2.0 ** -32))   # the second difference's pairs
T2_BAR = 0.1                                       # the predictor's t^2 check between tau and tau/4


# ------------------------------------------------------------------ the corrector (any system)
def second_order(sy, z, dz, log=print):
    """w = -J_r^-1 q2, q2 = [r(z + tau dz) + r(z - tau dz) - 2 r(z)] / (2 tau^2) the rung rows' second difference along
    the step, from the first pair of TAUS whose two values agree to T2_BAR. When no pair agrees, w is zero (the step
    goes on without the t^2 term, its trials staged as any failed projection is) and the check returned is the last
    pair's. Returns (w, check, tau)."""
    r0 = sy.rung(z)
    last = None
    for ta, tb in TAUS:
        ws = []
        for tau in (ta, tb):
            rp = sy.rung(sy.axpy(tau, dz, z))
            rm = sy.rung(sy.axpy(-tau, dz, z))
            q2 = sy.rscale(sy.radd(sy.radd(rp, rm), sy.rscale(r0, -2.0)), 0.5 / (tau * tau))
            ws.append(sy.hscale(sy.rung_solve(z, q2), -1.0))
        nd, nw = sy.hnorm(sy.hadd(ws[0], sy.hscale(ws[1], -1.0))), sy.hnorm(ws[0])
        chk = 0.0 if nd == 0 else (float("inf") if nw == 0 else nd / nw)     # two zero estimates agree
        last = (ws[0], chk, ta)
        if chk <= T2_BAR:
            return last
        log(f"    predictor: t^2 check {chk:.2e} at tau {ta:g} above {T2_BAR}; a smaller tau", flush=True)
    log(f"    predictor: no pair of TAUS passed the t^2 check (last {last[1]:.2e}); the step goes on without the t^2 "
        "term", flush=True)
    return sy.hscale(last[0], 0.0), last[1], None


def staged_trial(sy, z, dz, lam, w, n, log=print):
    """the trial z + lam dz with its horizon data projected in n stages t_j = lam j / n: stage 1 starts from the
    predictor h + t dh + t^2 w, stage j > 1 from stage j-1's projected data plus the predictor's increment. Returns
    (z', record) or (None, record) when a projection fails."""
    h0, dh = sy.get_h(z), sy.get_h(dz)

    def pred(t):
        return sy.hadd(sy.hadd(h0, sy.hscale(dh, t)), sy.hscale(w, t * t))
    hp = pp = None
    rec = dict(stages=n, proj=[])
    zp = None
    for j in range(1, n + 1):
        t = lam * j / n
        pt = pred(t)
        start = pt if j == 1 else sy.hadd(hp, sy.hadd(pt, sy.hscale(pp, -1.0)))
        zp, ok, info = sy.project(sy.set_h(sy.axpy(t, dz, z), start))
        rec["proj"].append(info)
        if not ok:
            rec["failed_at"] = j
            return None, rec
        hp, pp = sy.get_h(zp), pt
    return zp, rec


def raise_newton(sy, z, tol, it_max=12, halvings=10, stages=16, mode="factored", log=print):
    """Newton on sy from z. mode 'factored': trials projected from the second-order predictor, staged on failure;
    'linear': projected from the trial's own data, unstaged; 'old': not projected; 'nopred' and 'sign' (the
    selftest's controls): the second-order term left out, or with its sign flipped, staged as 'factored'. Returns
    (z, history, converged, why)."""
    hist = []
    if mode != "old":
        z, ok, info = sy.project(z)
        log(f"  start projection: {info.get('line', '')}{'' if ok else ' FAILED'}", flush=True)
        if not ok:
            return z, hist, False, "the start projection failed"
    for it in range(it_max):
        t0 = time.time()
        dz, nF, extra = sy.newton(z)
        ndz = sy.norm(dz)
        rec = dict(it=it, F=nF, dz=ndz, dDelta=sy.delta_of(dz), trials=[], **extra)
        hist.append(rec)
        if nF <= tol:
            log(f"  on {it}: max|F| {nF:.3e} converged ({time.time() - t0:.1f} s)", flush=True)
            return z, hist, True, "converged"
        w = None
        if mode in ("factored", "nopred", "sign"):
            w, chk, tau = second_order(sy, z, dz, log)
            rec.update(t2_check=chk, tau=tau, w=sy.hnorm(w))
            if tau is None:
                rec["predictor"] = "none: no pair of TAUS agreed"
            if mode == "nopred":
                w = sy.hscale(w, 0.0)
            elif mode == "sign":
                w = sy.hscale(w, -1.0)
        log(f"  on {it}: max|F| {nF:.3e}, |dz| {ndz:.3e}, dDelta {rec['dDelta']:+.3e}"
            + (f", predictor |w| {rec['w']:.2e} (t^2 check {rec['t2_check']:.1e} at tau {rec['tau']})" if "w" in rec
               else "") + (f", {extra['line']}" if extra.get("line") else "") + f", {time.time() - t0:.1f} s",
            flush=True)
        lam, took = 1.0, None
        for _ in range(halvings + 1):
            t1 = time.time()
            if mode == "old":
                zt, trec = sy.axpy(lam, dz, z), dict(stages=0)
            elif mode == "linear":
                zt, ok, info = sy.project(sy.axpy(lam, dz, z))
                trec = dict(stages=1, proj=[info])
                if not ok:
                    zt = None
            else:
                zt, trec, n = None, None, 1
                while n <= stages and zt is None:
                    zt, trec = staged_trial(sy, z, dz, lam, w, n, log)
                    n *= 2
            trec.update(lam=lam)
            if zt is None:
                trec["accepted"] = False
                rec["trials"].append(trec)
                log(f"    lam {lam:g}: projection failed at every staging up to {trec.get('stages')} "
                    f"({trec['proj'][-1].get('line', '') if trec.get('proj') else ''}), {time.time() - t1:.1f} s",
                    flush=True)
                lam /= 2
                continue
            Ft = sy.residual(zt)
            nFt = sy.norm(Ft)
            theta = sy.norm(sy.simplified(Ft)) / max(ndz, 1e-300) if np.isfinite(nFt) else float("inf")
            acc = bool(np.isfinite(nFt) and (nFt <= tol or theta <= 1.0 - lam / 4))
            trec.update(F=nFt, theta=theta, accepted=acc)
            rec["trials"].append(trec)
            log(f"    lam {lam:g}: {trec.get('stages')} stage(s), max|F| {nFt:.3e}, theta {theta:.3e}: "
                f"{'accepted' if acc else 'refused'}, {time.time() - t1:.1f} s", flush=True)
            if acc:
                took = zt
                break
            lam /= 2
        if took is None:
            return z, hist, False, "no trial accepted"
        z = took
        if hist[-1]["trials"][-1]["F"] <= tol:
            hist.append(dict(it=it + 1, F=hist[-1]["trials"][-1]["F"], trials=[]))
            log(f"  on {it + 1}: max|F| {hist[-1]['F']:.3e} converged (the accepted trial)", flush=True)
            return z, hist, True, "converged"
    return z, hist, False, "it_max"


# ------------------------------------------------------------------ the chain (hpkernel; Linux)
def kernel_report():
    """hpkernel's path, ISA and crate versions, the worktree its wheel was built in (if pip recorded the wheel's path)
    and that worktree's HEAD now, which is the build's commit only if the worktree has not moved since; also this
    script's checkout. Returns a dict for the record."""
    import hpkernel
    d = os.path.dirname(hpkernel.__file__)
    crates = set()
    for so in glob.glob(os.path.join(d, "_hpkernel*")):
        with open(so, "rb") as f:
            crates |= {m.decode() for m in re.findall(rb"(multi(?:double|shoot|solver)-\d+\.\d+\.\d+)", f.read())}
    info = dict(path=hpkernel.__file__, isa=getattr(hpkernel, "isa", lambda: "?")(), crates=sorted(crates))
    try:
        from importlib.metadata import distribution
        du = json.loads(distribution("hpkernel").read_text("direct_url.json") or "{}")
        wheel = du.get("url", "").replace("file://", "")
        info["wheel"] = wheel
        repo = os.path.dirname(os.path.dirname(os.path.dirname(wheel))) if wheel else ""
        if repo and os.path.isdir(repo):
            r = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True)
            c = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", "hp/src", "hp/Cargo.toml"],
                               capture_output=True, text=True)
            info["build_worktree"] = repo
            info["build_worktree_head"] = r.stdout.strip()
            info["build_worktree_src_clean"] = c.returncode == 0 and not c.stdout.strip()
    except Exception as e:          # the record is information; a missing metadata file is not an error
        info["build_lookup"] = repr(e)
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True)
    info["script_head"] = r.stdout.strip()
    return info


NEEDED = ("residual_o", "newton_o", "olu_solve", "set_segs", "plan")
OPTIONAL = ("residual_sub", "jacobian_sub", "newton_o_residual")


def probe_kernel():
    """exit clearly when the kernel lacks a method the raise needs; print a notice for each optional one missing (the
    callers fall back to the full residual or Jacobian)"""
    from hpkernel.driver import Solver
    missing = [n for n in NEEDED if not hasattr(Solver, n)]
    if missing:
        raise SystemExit(f"factored_raise.py: this hpkernel has no Solver.{', Solver.'.join(missing)}; build hp at "
                         "main with hp/build.sh")
    for n in OPTIONAL:
        if not hasattr(Solver, n):
            print(f"  notice: this hpkernel has no Solver.{n}; the fallback (full residual or Jacobian) is used",
                  flush=True)
    try:
        from multidouble import od_axpy  # noqa: F401
    except ImportError:
        raise SystemExit("factored_raise.py: multidouble's Python package has no od_axpy (f092018 on); reinstall it "
                         "with hp/build.sh")
    import inspect
    from reseed import Reseeder
    if "stop" not in inspect.signature(Reseeder.refine_o).parameters:
        raise SystemExit("factored_raise.py: reseed.Reseeder.refine_o has no stop (#1032); run from a checkout of main")


class Chain:
    """the pinned chain at order ks on the kernel: octo-double residual, Newton step and LU (octo_corrector.PinnedO,
    held values the state's own from all eight parts, anm_run.py --held_o 1), and the rung of the projection on the
    two-node layout of reseed.Reseeder (built as anm_run.py --reseed 2 --reseed_q 1 --reseed_o 3 --reseed_start pred
    --proj_start pred --reseed_o_stop 1 --reseed_qd_bar 1e-13 builds it). rung 'a' (the default) is method A: the a
    rows of the horizon block for a_h, with the node's a held and m_h the step's; its Newton is project_a. rung 'ma' is
    the chain's ElimO rung, the m and a rows for m_h and a_h (octo_corrector.project_o): its own Newton moves m_h
    linearly, so it meets the curvature it is meant to remove (5.3)."""

    def __init__(self, meta, z8, ks, threads, plan, proj_bar, rung="a"):
        from octo_corrector import D_of, pinned_o
        from reseed import Reseeder, offsets
        self.meta = dict(meta, ks=int(ks))
        if plan == "stored":
            segs = meta.get("segs")
            if not segs:
                raise SystemExit("--plan stored: the state stores no plan (segs); give --plan new")
        else:
            # anm_run.py's plan for a changed --ks: at the state, at 1e-32 with steps at most 0.05
            from hpkernel.driver import make_solver
            s0 = make_solver(D_of(meta), int(meta["M"]), nodes=meta["nodes"], kt=meta.get("kt", 36),
                             kc=meta.get("kc", 146), ks=int(ks), threads=threads, kfreeze=meta["kfreeze"])
            segs = s0.plan(np.asarray(z8[0]), np.asarray(z8[1]), 1e-32, 0.05)
        self.segs = [[float(x) for x in g] for g in segs]
        self.s, self.sp, _, self.so = pinned_o(self.meta, z8, threads=threads, plan_tol=1e-32, held_exact=True,
                                               segs=self.segs)
        self.M, self.K = int(meta["M"]), len(meta["nodes"])
        self.off = offsets(self.M, self.K)
        self.rung_mode = rung
        M, a = self.M, offsets(self.M, 2)
        if rung == "a":
            self.hs = slice(self.off["ah"], self.off["ah"] + M)
            self.rrows, self.rcols = np.r_[8 * M + 2 * M:8 * M + 3 * M], np.r_[a["ah"]:a["ah"] + M]
        else:
            self.hs = slice(self.off["mh"], self.off["mh"] + 2 * M)
            self.rrows, self.rcols = np.r_[8 * M + M:8 * M + 3 * M], np.r_[a["mh"]:a["mh"] + 2 * M]
        self.rs = Reseeder(dict(M=self.M, nodes=list(meta["nodes"]), kt=meta.get("kt", 36), kc=meta.get("kc", 146),
                                ks=int(ks)), D_of(meta), threads=threads, horizon_alpha=0, qd=True, od=3,
                           qd_start="pred", qd_bar=1e-13, proj_start="pred", od_stop=True, exact_recip=0)
        self.proj_bar = proj_bar
        self._lu = None

    # vectors: eight-part tuples of float64 arrays; horizon data: eight parts of the rung's columns (M or 2M)
    def axpy(self, a, x, y):
        from multidouble import od_axpy
        from octo_corrector import o8
        return o8(od_axpy(float(a), *o8(x), *o8(y)))

    def norm(self, x):
        return float(np.abs(np.asarray(x[0])).max())

    def delta_of(self, x):
        return float(np.asarray(x[0])[-1])

    def get_h(self, z):
        return tuple(np.ascontiguousarray(np.asarray(p)[self.hs]) for p in z)

    def set_h(self, z, h):
        out = [np.array(p, dtype=float) for p in z]
        for q in range(8):
            out[q][self.hs] = h[q]
        return tuple(np.ascontiguousarray(p) for p in out)

    def hadd(self, x, y):
        return self.axpy(1.0, x, y)

    radd = hadd

    def hscale(self, x, a):
        zero = tuple(np.zeros(len(x[0])) for _ in range(8))
        return self.axpy(a, x, zero)

    rscale = hscale
    hnorm = norm

    def residual(self, z):
        return self.so.residual_o(z)

    def newton(self, z):
        d, Fh, Fl, pmin, pmax = self.so.newton_o(z)
        nF = float(np.abs(np.asarray(Fh) + np.asarray(Fl)).max())
        return d, nF, dict(piv=[float(pmin), float(pmax)], line=f"piv {float(pmin):.1e}..{float(pmax):.1e}")

    def simplified(self, F):
        """-J(z)^-1 F through the LU the last newton_o factored (the natural test's simplified correction)"""
        from octo_corrector import o8
        return tuple(-p for p in o8(self.s.olu_solve(list(o8(F)))))

    def _two_node(self, z8, parts=8):
        """the rung's two-node point (refine_o's layout): centre data, node 0, the last node as node 1, Delta and z8's
        horizon data"""
        M, a, o = self.M, self.rs.aoff, self.off
        n0, n1, nk = o["node"](0), o["node"](1), o["node"](self.K - 1)
        out = []
        for q in range(parts):
            v = np.zeros(a["L"])
            p = np.asarray(z8[q])
            v[:2 * M] = p[:2 * M]
            v[a["node"](0):a["node"](1)] = p[n0:n1]
            v[a["node"](1):a["node"](1) + 4 * M] = p[nk:nk + 4 * M]
            v[a["mh"]:a["mh"] + 2 * M] = p[o["mh"]:o["mh"] + 2 * M]
            v[-1] = p[-1]
            out.append(np.ascontiguousarray(v))
        return out

    def rung(self, z8):
        """the rung rows at z8 (eight parts), from the blocks that hold them (reseed.res_rows)"""
        from reseed import res_rows
        s = self.rs.rung_solver(self.rs.us[-1])
        return tuple(np.ascontiguousarray(np.asarray(p)) for p in res_rows(s, self._two_node(z8), self.rrows))

    def _block_lu(self, v4):
        from multidouble import QDLu
        from reseed import jac_block
        s = self.rs.rung_solver(self.rs.us[-1])
        blk = [np.ascontiguousarray(np.array(b)) for b in jac_block(s, v4, self.rrows, self.rcols)]
        return QDLu(*blk)

    def rung_solve(self, z8, r):
        """J_r^-1 r: the rung block of the quad-double Jacobian at z8's leading four parts, factored once per point"""
        import hashlib
        key = hashlib.sha1(b"".join(np.ascontiguousarray(np.asarray(p, dtype=float)).tobytes()
                                    for p in z8[:4])).digest()
        if self._lu is None or self._lu[0] != key:
            self._lu = (key, self._block_lu(self._two_node(z8, 4)))
        x = self._lu[1].solve(*[np.ascontiguousarray(p) for p in r[:4]])
        zero = np.zeros(len(self.rrows))
        return tuple(np.ascontiguousarray(np.asarray(p)) for p in x) + (zero, zero, zero, zero)

    def project(self, z8):
        if self.rung_mode == "ma":
            from octo_corrector import project_o
            zo, ho, end, ok = project_o(self.rs, z8, self.proj_bar)
        else:
            zo, ho, end, ok = self.project_a(z8)
        line = f"rung max|R| {' -> '.join(f'{x:.2e}' for x in ho)} ({end})"
        return zo, bool(ok), dict(hist=[float(x) for x in ho], end=end, line=line)

    def project_a(self, z8):
        """method A in octo-double: Newton on a_h for the a rows of the horizon block, the rest of z8 held, as
        Reseeder.refine_o does for its rung: the residual from all eight parts, each correction through a QDLu of the
        block at the point's leading four parts, factored again when a correction gains less than OD_REFACTOR, and
        ended at the floor (|R| <= OD_FLOOR, or a correction from a fresh LU that does not halve |R|), when not finite,
        or after PROJ_MAX corrections. Returns (z', hist, end, ok): ok when |R| ends finite and at most proj_bar."""
        from multidouble import od_axpy
        from reseed import OD_FLOOR, OD_REFACTOR, OD_STALL
        v = [np.array(p) for p in self._two_node(z8)]
        cols = self.rcols
        d = [np.ascontiguousarray(p[cols]) for p in v]
        zero = np.zeros(len(cols))
        hist, lu, fresh, end = [], None, False, "cap"
        for k in range(PROJ_MAX + 1):
            for q in range(8):
                v[q][cols] = d[q]
            R = self.rung(self.set_h(z8, d))
            hist.append(float(np.max(np.abs(np.asarray(R[0]) + np.asarray(R[1])))))
            if not np.isfinite(hist[-1]):
                end = "nonfinite"
                break
            if hist[-1] <= OD_FLOOR or (fresh and hist[-1] > OD_STALL * hist[-2]):
                end = "floor"
                break
            if k == PROJ_MAX:
                break
            try:
                if lu is None or hist[-1] > OD_REFACTOR * hist[-2]:
                    lu, fresh = self._block_lu([np.ascontiguousarray(p) for p in v[:4]]), True
                else:
                    fresh = False
                x = [np.ascontiguousarray(np.asarray(p)) for p in lu.solve(*[-np.asarray(r) for r in R[:4]])]
            except ValueError:
                end = "error"
                break
            d = [np.ascontiguousarray(np.asarray(p)) for p in od_axpy(1.0, *x, *([zero] * 4), *d)]
        zo = self.set_h(z8, d) if end != "nonfinite" else tuple(z8)
        ok = bool(np.isfinite(hist[-1]) and hist[-1] <= self.proj_bar and end in ("floor", "cap"))
        return zo, hist, end, ok


def save_state(path, z8, meta):
    """octo_corrector.save_o's layout: zh, zl, z2, z3 (load_q reads them), z4..z7 and the meta as JSON"""
    z = [np.ascontiguousarray(np.asarray(p, dtype=float)) for p in z8]
    np.savez(path, zh=z[0], zl=z[1], z2=z[2], z3=z[3], z4=z[4], z5=z[5], z6=z[6], z7=z[7], meta=json.dumps(meta))


def load_parts(path):
    """(eight parts, meta) with numpy alone: the parts a state stores, zero for those it does not"""
    d = np.load(path, allow_pickle=False)
    names = ["zh", "zl"] + [f"z{i}" for i in range(2, 8)]
    n = len(d["zh"])
    return [np.asarray(d[k], dtype=float) if k in d.files else np.zeros(n) for k in names], json.loads(str(d["meta"]))


def exact_entries(parts, dps=200):
    """each entry's exact sum over the parts, as mpmath numbers at dps digits"""
    with mp.workdps(dps):
        return [mp.fsum(mp.mpf(float(p[i])) for p in parts) for i in range(len(parts[0]))]


def compare(parts_a, parts_b, bar):
    """the largest entry difference |a - b| (exact sums) and Delta's (the last entry), and whether the first is at most
    bar"""
    if len(parts_a[0]) != len(parts_b[0]) or not len(parts_a[0]):
        return dict(max=float("inf"), at=-1, delta=float("nan"), ok=False,
                    why=f"lengths {len(parts_a[0])} and {len(parts_b[0])}")
    with mp.workdps(200):
        a, b = exact_entries(parts_a), exact_entries(parts_b)
        d = [abs(x - y) for x, y in zip(a, b)]
        i = max(range(len(d)), key=lambda k: d[k])
        return dict(max=float(d[i]), at=i, delta=float(a[-1] - b[-1]), ok=bool(d[i] <= bar))


def run(argv):
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    args = [a for a in argv if not a.startswith("--")]
    opt = dict(state=None, ks=None, threads="10", tol="1e-80", it_max="12", halvings="10", stages="16", plan="stored",
               ref=None, bar="1e-60", mutate="", proj_bar="1e-60", rung="a")
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            if k not in opt:
                raise SystemExit(f"factored_raise.py: unknown option --{k}; known: {sorted(opt)}")
            if v:
                opt[k] = v
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and "=" not in a and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            opt[a[2:]] = argv[i + 1]
            args.remove(argv[i + 1])
            i += 1
        i += 1
    if len(args) != 2 or not opt["state"] or not opt["ks"]:
        print(__doc__)
        return 2
    raw, out = args
    if opt["mutate"] not in ("", "old", "linear", "order"):
        raise SystemExit(f"--mutate={opt['mutate']}: old, linear or order")
    if opt["plan"] not in ("stored", "new"):
        raise SystemExit("--plan stored or new")
    if opt["rung"] not in ("a", "ma"):
        raise SystemExit("--rung a or ma")
    os.makedirs(out, exist_ok=True)
    require_raw(os.path.join(out, "raise.json"), allow=allow)
    t_start = time.time()
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC factored_raise.py {' '.join(argv)}", flush=True)
    kinfo = kernel_report()
    print(f"hpkernel {kinfo['path']} ({kinfo['isa']}; {', '.join(kinfo['crates'])}); its wheel was built in "
          f"{kinfo.get('build_worktree', '?')}, whose HEAD is now {kinfo.get('build_worktree_head', '?')} (hp/src "
          f"clean now: {kinfo.get('build_worktree_src_clean', '?')}; the build's own commit is not recorded); this "
          f"script at {kinfo['script_head']}", flush=True)
    probe_kernel()
    sp = os.path.join(raw, opt["state"])
    z8, meta = load_parts(sp)
    z8 = tuple(np.ascontiguousarray(p) for p in z8)
    ks = int(opt["ks"]) - (1 if opt["mutate"] == "order" else 0)
    tol = float(opt["tol"])
    proj_bar = float(opt["proj_bar"])
    mode = {"": "factored", "old": "old", "linear": "linear", "order": "factored"}[opt["mutate"]]
    print(f"state {opt['state']}: D {meta.get('D_str', meta['D'])}, ks {meta.get('ks')} -> {ks}"
          f"{' (--mutate=order: ks - 1)' if opt['mutate'] == 'order' else ''}, M {meta['M']}, plan {opt['plan']}; "
          f"mode {mode}, rung {opt['rung']}, tol {tol:g}, projection bar {proj_bar:g}, threads {opt['threads']}",
          flush=True)
    sy = Chain(meta, z8, ks, int(opt["threads"]), opt["plan"], proj_bar, opt["rung"])
    if int(meta.get("ks", 130)) != ks:
        # information: the state's residual at its own order on this system (stored plan, its own held values), which
        # #1064 read as 1.5e-41 through near_null.py and 3.88e-25 as anm_run.py's polish start at D 262.0099
        own = Chain(meta, z8, int(meta.get("ks", 130)), int(opt["threads"]), opt["plan"], proj_bar, opt["rung"])
        print(f"the state at its own order {meta.get('ks')} on this system: pinned octo-double max|F| "
              f"{own.norm(own.residual(z8)):.3e}; at order {ks}: {sy.norm(sy.residual(z8)):.3e}", flush=True)
        del own
    z, hist, conv, why = raise_newton(sy, z8, tol, it_max=int(opt["it_max"]), halvings=int(opt["halvings"]),
                                      stages=int(opt["stages"]), mode=mode)
    kf = meta.get("kfreeze")
    name = f"{'raise' if conv else 'lastiter'}_pin{kf}_D{float(meta['D']):.4f}.npz"
    with mp.workdps(60):
        d0, d1 = exact_entries(z8, 60)[-1], exact_entries(z, 60)[-1]
        dDelta = float(d1 - d0)
    rec = dict(cmd="factored_raise", argv=argv, state=opt["state"], ks_from=meta.get("ks"), ks=ks, mode=mode,
               rung=opt["rung"], tol=tol,
               converged=conv, why=why, hist=hist, dDelta=dDelta, kernel=kinfo, seconds=round(time.time() - t_start))
    meta_out = dict(meta, ks=ks, segs=sy.segs, source=opt["state"], parts=8,
                    factored_raise=dict(converged=conv, max_F=[h["F"] for h in hist], tol=tol, mode=mode,
                                        rung=opt["rung"], why=why,
                                        script=kinfo["script_head"]))
    save_state(os.path.join(out, name), z, meta_out)
    print(f"raise {'converged' if conv else 'NOT converged'} ({why}) after {len(hist)} line(s), last max|F| "
          f"{hist[-1]['F'] if hist else float('nan'):.3e}; Delta moved by {dDelta:+.6e}; written {name}; "
          f"{rec['seconds']} s", flush=True)
    ok = conv
    if opt["ref"]:
        zr, mr = load_parts(os.path.join(raw, opt["ref"]))
        same_plan = mr.get("segs") == meta_out.get("segs")
        c = compare(list(z), zr, float(opt["bar"]))
        with mp.workdps(60):
            dref = float(exact_entries(zr, 60)[-1] - d0)
        same_ks = int(mr.get("ks", -1)) == ks
        rec["compare"] = dict(ref=opt["ref"], ref_ks=mr.get("ks"), same_ks=same_ks, same_plan=same_plan,
                              bar=float(opt["bar"]), dDelta_ref=dref, **c)
        good = c["ok"] and same_plan and same_ks
        print(f"against {opt['ref']} (ks {mr.get('ks')}{'' if same_ks else f', NOT the raise order {ks}'}, plan "
              f"{'the same' if same_plan else 'DIFFERENT'}): largest entry difference {c['max']:.3e} at entry "
              f"{c['at']}{' (' + c['why'] + ')' if 'why' in c else ''}, Delta {c['delta']:+.3e}; the raise moved Delta "
              f"by {dDelta:+.6e}, the reference's by {dref:+.6e}: {'PASS' if good else 'FAIL'} (bar "
              f"{float(opt['bar']):g}; the entries {'within' if c['ok'] else 'NOT within'} it)", flush=True)
        ok = ok and good
    with open(os.path.join(out, "raise.json"), "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    return 0 if ok else 1


# ------------------------------------------------------------------ the model (selftest, no kernel)
class Model:
    """three unknowns z = (d, b, h): d the weak one (Delta's part), b a node value, h the horizon datum. With
    x = G (h - phi(d, b)), the rung row r = x / (1 - (x/P)^2) (poles at a miss of P), the c row sigma d + c3 d^3 - eps
    + mu r, the node row b - b0 - d. phi = phi0 + 0.3 d + kappa d^2 + kappa3 d^3 + 0.5 (b - b0). The raise sets eps;
    its solution has r = 0, b = b0 + d and sigma d + c3 d^3 = eps."""
    G, P, sigma, mu, c3 = mp.mpf("1e30"), mp.mpf(1000), mp.mpf("1e-6"), mp.mpf(1), mp.mpf(1)
    phi0, kappa, kappa3, b0 = mp.mpf("0.7"), mp.mpf(1), mp.mpf(1), mp.mpf("0.2")

    def __init__(self, eps):
        self.eps = mp.mpf(eps)
        self.proj_tol = mp.mpf(10) ** (-mp.mp.dps + 15) * self.G

    def phi(self, d, b):
        return self.phi0 + mp.mpf("0.3") * d + self.kappa * d * d + self.kappa3 * d ** 3 + (b - self.b0) / 2

    def x(self, z):
        return self.G * (z[2] - self.phi(z[0], z[1]))

    def rr(self, x):
        return x / (1 - (x / self.P) ** 2)

    def residual(self, z):
        r = self.rr(self.x(z))
        return [self.sigma * z[0] + self.c3 * z[0] ** 3 - self.eps + self.mu * r, r, z[1] - self.b0 - z[0]]

    def jac(self, z):
        x = self.x(z)
        dr = (1 + (x / self.P) ** 2) / (1 - (x / self.P) ** 2) ** 2
        d = z[0]
        xd = -self.G * (mp.mpf("0.3") + 2 * self.kappa * d + 3 * self.kappa3 * d * d)
        xb, xh = -self.G / 2, self.G
        return mp.matrix([[self.sigma + 3 * self.c3 * d * d + self.mu * dr * xd, self.mu * dr * xb, self.mu * dr * xh],
                          [dr * xd, dr * xb, dr * xh], [-1, 1, 0]])

    def truth(self):
        d = self.eps / self.sigma
        for _ in range(60):
            d -= (self.sigma * d + self.c3 * d ** 3 - self.eps) / (self.sigma + 3 * self.c3 * d * d)
        b = self.b0 + d
        return [d, b, self.phi(d, b)]

    # the corrector's interface
    def axpy(self, a, x, y):
        return [mp.mpf(a) * u + v for u, v in zip(x, y)]

    def norm(self, x):
        return float(max(abs(v) for v in x))

    def delta_of(self, x):
        return float(x[0])

    def get_h(self, z):
        return [z[2]]

    def set_h(self, z, h):
        return [z[0], z[1], h[0]]

    def hadd(self, x, y):
        return [u + v for u, v in zip(x, y)]

    radd = hadd

    def hscale(self, x, a):
        return [mp.mpf(a) * u for u in x]

    rscale = hscale
    hnorm = norm

    def newton(self, z):
        self._J = self.jac(z)
        F = self.residual(z)
        st = mp.lu_solve(self._J, mp.matrix(F))
        return [-st[i] for i in range(3)], float(max(abs(v) for v in F)), {}

    def simplified(self, F):
        st = mp.lu_solve(self._J, mp.matrix(F))
        return [-st[i] for i in range(3)]

    def rung(self, z):
        return [self.rr(self.x(z))]

    def rung_solve(self, z, r):
        x = self.x(z)
        dr = (1 + (x / self.P) ** 2) / (1 - (x / self.P) ** 2) ** 2
        return [r[0] / (dr * self.G)]

    def project(self, z):
        """Newton on h alone for the rung row; ok when it ends finite at proj_tol"""
        z = list(z)
        hist = []
        for _ in range(40):
            r = self.rung(z)[0]
            hist.append(float(abs(r)))
            if not mp.isfinite(r) or hist[-1] <= self.proj_tol or (len(hist) > 3 and hist[-1] > hist[-4]):
                break
            z[2] -= self.rung_solve(z, [r])[0]
        ok = bool(np.isfinite(hist[-1]) and hist[-1] <= self.proj_tol)
        return z, ok, dict(hist=hist, line=f"rung |r| {hist[0]:.2e} -> {hist[-1]:.2e} in {len(hist) - 1}")


# ------------------------------------------------------------------ the gates of stages 3 and 3b (5.3)
GATE_F = 1e-60                                     # part 4's tolerance (#1064)


def polishgate(log, outdir, bar=None):
    """the pass rule for anm_run.py's polish of a state at its order (5.3). The chain's rule (bar None, stage 3 since
    the coordinator's change of 2026-11-21): (a) the start line says the polish was skipped (the state already within
    the run's own --tol_od), or (b) the last octo-double Newton line says converged, by the run's own tolerance or its
    floor rule, and a base was saved in outdir. With bar (part 4's 1e-60): the same, each with max|F| <= bar (gate903d
    and gate973). Returns (ok, reason)."""
    t = open(log, errors="replace").read()
    m = re.search(r"octo-double max\|F\| ([0-9.eE+-]+|nan|inf) <= --tol_od \S+: base polish skipped", t)
    if m and (bar is None or float(m.group(1)) <= bar):
        return True, f"(a) the start line: octo-double max|F| {m.group(1)}, polish skipped"
    lines = re.findall(r"octo-double Newton: (\d+) steps, max\|F\| ([0-9.eE+-]+|nan|inf), (not converged|converged)", t)
    base = glob.glob(os.path.join(outdir, "base_pin*_D*.npz"))
    if not lines:
        tail = "" if bar is None else f" at or below {bar:g}"
        return False, "no octo-double Newton line and no skipped polish" + tail
    n, F, c = lines[-1]
    ok = c == "converged" and (bar is None or float(F) <= bar) and bool(base)
    return ok, f"(b) the last octo-double Newton line: {n} steps, max|F| {F}, {c}; base saved {bool(base)}"


def stepgate(log, outdir):
    """stage 3's chain step (5.3): anm_run.py --steps 1 from the polished base wrote its jet_pin file, accepted a step
    by the chain's own rules (an anm_pin state saved, its Delta line printed) and did not stop. Returns (ok, reason)."""
    t = open(log, errors="replace").read()
    jl = re.search(r"step 0: the series saved as (.+?jet_pin\d+_D[0-9.]+\.npz) \(", t)
    jet = bool(jl) and os.path.exists(os.path.join(outdir, os.path.basename(jl.group(1))))
    saved = "Delta of the saved state, its eight parts summed exactly" in t
    st = glob.glob(os.path.join(outdir, "anm_pin*_D*.npz"))
    stopped = bool(re.search(r"^stopped:", t, re.M))
    ok = jet and saved and bool(st) and not stopped
    return ok, (f"jet {'written' if jet else 'MISSING'}, step {'accepted and saved' if saved and st else 'NOT saved'}"
                f"{', stopped' if stopped else ''}")


def gate_selftest():
    """the gates on planted logs and folders: each broken case must fail"""
    import tempfile
    out = []
    good_b = "      octo-double Newton: 1 steps, max|F| 2.7e-86, converged; double-double |F| 1e+08\n"
    skip = "base: double-double |F| 1e-21 <= --tol 1e-20, octo-double max|F| {} <= --tol_od {}: base polish skipped\n"
    cases = [("part 4 (a) 1e-62", "part4", skip.format("1.0e-62", "1e-60"), [], True),
             ("part 4 (a) 5e-60", "part4", skip.format("5.0e-60", "1e-59"), [], False),
             ("part 4 (b)", "part4", good_b, ["base_pin3_D262.0099.npz"], True),
             ("part 4 (b) no base", "part4", good_b, [], False),
             ("part 4 (b) not converged", "part4", good_b.replace(", converged", ", not converged"),
              ["base_pin3_D262.0099.npz"], False),
             ("part 4 (b) 2e-34", "part4", good_b.replace("2.7e-86", "2.0e-34"), ["base_pin3_D262.0099.npz"], False),
             ("part 4 empty", "part4", "", ["base_pin3_D262.0099.npz"], False),
             ("chain (a) 1e-35", "chain", skip.format("1.0e-35", "2e-34"), [], True),
             ("chain (b) 2.5e-34", "chain", good_b.replace("2.7e-86", "2.5e-34"), ["base_pin3_D262.0099.npz"], True),
             ("chain (b) no base", "chain", good_b, [], False),
             ("chain (b) not converged", "chain", good_b.replace(", converged", ", not converged"),
              ["base_pin3_D262.0099.npz"], False),
             ("chain empty", "chain", "", ["base_pin3_D262.0099.npz"], False)]
    step_ok = ("step 0: the series saved as /x/jet_pin3_D262.0099.npz (8 parts, 400 s)\n"
               "    a 0.092 -> D 262.10: predicted |F| 1e-20, od 3 steps -> 1e-35 (tol 2e-34), Delta 1.6 ...\n"
               "      Delta of the saved state, its eight parts summed exactly: 1.6830910901545985\ndone\n")
    cases += [("step", "step", step_ok, ["jet_pin3_D262.0099.npz", "anm_pin3_D262.1021.npz"], True),
              ("step no jet", "step", step_ok, ["anm_pin3_D262.1021.npz"], False),
              ("step stopped", "step", step_ok.replace("done", "stopped: no convergence after eight attempts"),
               ["jet_pin3_D262.0099.npz", "anm_pin3_D262.1021.npz"], False),
              ("step not saved", "step", step_ok.replace("      Delta of the saved state", "      Delta of a"),
               ["jet_pin3_D262.0099.npz"], False),
              ("step, a path with a space", "step", step_ok.replace("/x/jet_pin3", "/x y/jet_pin3"),
               ["jet_pin3_D262.0099.npz", "anm_pin3_D262.1021.npz"], True)]
    for name, kind, text, files, want in cases:
        with tempfile.TemporaryDirectory() as d:
            lp = os.path.join(d, "run.log")
            with open(lp, "w") as fh:
                fh.write(text)
            for f in files:
                open(os.path.join(d, f), "w").close()
            ok, why = (polishgate(lp, d, GATE_F) if kind == "part4" else polishgate(lp, d) if kind == "chain"
                       else stepgate(lp, d))
        out.append((f"gate {name}", ok == want))
        print(f"  gate, {name}: {'pass' if ok else 'fail'} ({why}); expected {'pass' if want else 'fail'}: "
              f"{'ok' if ok == want else 'WRONG'}", flush=True)
    return out


def selftest(argv):
    mutate = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), "")
    mp.mp.dps = 120
    res = []
    quiet = lambda *a, **k: None  # noqa: E731
    m = Model("1e-16")
    start = [mp.mpf(0), m.b0, m.phi(0, m.b0)]
    truth = m.truth()
    tol = 1e-85
    main_mode = mutate if mutate in ("nopred", "old", "linear") else "factored"
    cases = [("factored", main_mode, True),
             ("old", "old", False), ("linear", "linear", False), ("nopred", "nopred", False), ("sign", "sign", False)]
    for name, mode, want in cases:
        z, hist, conv, why = raise_newton(m, start, tol, mode=mode, log=quiet)
        err = max(abs(a - b) for a, b in zip(z, truth))
        good = bool(conv and err <= mp.mpf("1e-80"))
        res.append((f"model {name}{' (--mutate=' + mutate + ')' if name == 'factored' and mutate else ''}",
                    good == want))
        print(f"  model, {name}: {'converged' if conv else 'not converged'} ({why}) in {len(hist)} line(s), "
              f"|z - truth| {mp.nstr(err, 3)}; expected {'to converge' if want else 'to fail'}: "
              f"{'ok' if good == want else 'WRONG'}", flush=True)
    # the state writer: a stored state written again and read back exactly, with the raise's meta
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    tmpdir = tempfile.TemporaryDirectory()
    tmp = os.path.join(tmpdir.name, "state.npz")
    st = next(iter(sorted(glob.glob(os.path.join(here, "data", "**", "*.npz"), recursive=True))), None)
    st = next((a for a in argv if a.endswith(".npz")), st)
    if st:
        parts, meta = load_parts(st)
        meta_out = dict(meta, ks=int(meta.get("ks", 130)) + 22, parts=8, factored_raise=dict(converged=True))
        save_state(tmp, parts, meta_out)
        back, mb = load_parts(tmp)
        same = all(np.array_equal(a, b) for a, b in zip(parts, back)) and mb == meta_out
        keys = set(np.load(tmp).files) == {"zh", "zl", "z2", "z3", "z4", "z5", "z6", "z7", "meta"}
        res.append(("state writer", same and keys))
        print(f"  state writer on {os.path.basename(st)}: parts and meta read back "
              f"{'exactly' if same else 'DIFFERENT'}, keys {'as save_o' if keys else 'WRONG'}", flush=True)
        # the comparison: a copy with one entry moved by a known amount (in its lowest part)
        short = [p[:-1].copy() for p in parts]
        c = compare(parts, short, 1e-60)
        res.append(("comparison of unequal lengths", not c["ok"]))
        print(f"  comparison of unequal lengths: {'refused' if not c['ok'] else 'ACCEPTED'} ({c.get('why')}): "
              f"{'ok' if not c['ok'] else 'WRONG'}", flush=True)
        for amt, want in ((1e-65, True), (1e-55, False)):
            cp = [p.copy() for p in parts]
            cp[7][5] = cp[7][5] + amt
            c = compare(parts, cp, 1e-60)
            got = c["ok"] == want and abs(c["max"] - amt) <= 1e-3 * amt and c["at"] == 5
            res.append((f"comparison {amt:g}", got))
            print(f"  comparison, one entry moved by {amt:g}: largest difference {c['max']:.3e} at {c['at']}, "
                  f"{'within' if c['ok'] else 'above'} the bar 1e-60: {'ok' if got else 'WRONG'}", flush=True)
    else:
        print("  no stored state found for the writer's check (give one: selftest STATE.npz)", flush=True)
        res.append(("state writer", False))
    tmpdir.cleanup()
    res += gate_selftest()
    if mutate == "gate":                    # the control: a broken expectation must fail the selftest
        res.append(("gate control (--mutate=gate)", polishgate(os.devnull, os.path.dirname(os.devnull))[0]))
    fails = [n for n, ok in res if not ok]
    print(f"selftest{' --mutate=' + mutate if mutate else ''}: {len(res) - len(fails)} of {len(res)} pass"
          + (f"; FAIL: {', '.join(fails)}" if fails else ""), flush=True)
    return 0 if not fails else 1


def main(argv):
    if argv and argv[0] == "selftest":
        return selftest(argv[1:])
    if argv and argv[0] in ("polishgate", "stepgate") and len(argv) in (3, 5):
        bar = float(argv[4]) if len(argv) == 5 and argv[3] == "--bar" else None
        ok, why = polishgate(argv[1], argv[2], bar) if argv[0] == "polishgate" else stepgate(argv[1], argv[2])
        print(f"{argv[0]} {argv[1]}: {'PASS' if ok else 'FAIL'}: {why}", flush=True)
        return 0 if ok else 1
    return run(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
