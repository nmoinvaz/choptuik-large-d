#!/usr/bin/env python3
"""#1083: the matched mode determinant D_m(lam), det M(lam) on the layout whose segments past the sonic valley are
integrated inward (notes/MODE_HORIZON.md).

    python3 mode_matched.py valley STATE.npz [--threads=6] [--hmax=0.05] [--sub=2]
    python3 mode_matched.py det OUT.jsonl STATE.npz LAM [LAM ...] [--match=valley|out|N] [--both]
    python3 mode_matched.py root OUT.jsonl STATE.npz [--lam0=L] [--xtol=1e-11] [--match=...] [--both]
    python3 mode_matched.py four OUT.jsonl STATE.npz                          (pass criterion 1)
    python3 mode_matched.py d315 OUT.jsonl STATE.npz [--lam0=4.397684106690]  (pass criterion 2)
    python3 mode_matched.py planted OUT.jsonl STATE.npz [--lam_star=5.0] [--mutate=shift|outward]  (criterion 3)
    python3 mode_matched.py identity OUT.jsonl STATE.npz LAM1 LAM2 [--sub=8]  (pass criterion 4)
    python3 mode_matched.py blocks OUT.jsonl STATE.npz LAM1 LAM2 [--hmax=0.05] [--match=valley]  (per segment)
    python3 mode_matched.py toy [--gap=2e-3] [--mutate=shift]                 (the planted toy, no kernel)
    common options: [--threads=6] [--hmax=0.05] [--allow_unraw]

M(lam) is the multiple-shooting mode matrix (Solver.mode_matrix): the centre data, the K node states and the horizon
data, with one block row per segment. With every segment integrated outward (the default layout, `--match=out`), its
determinant D_out(lam) equals det [Phi_(K-1) ... Phi_1 C_0 | H], the Evans matrix at the last node. Near D 3 the
segment over the sonic valley (min over tau of c - x near x 0.0144) carries the p perturbations' outward growth
exp((lam - kappa) T) with T of order 10, so its transfer matrix Phi_v is formed with absolute errors of eps_dd times
that growth, which swamp the parts the determinant's sign depends on. The matched layout (`--match=valley`, the
kernel's match_node = the valley segment's inner node m) integrates every segment past m inward, so the valley's
transfer runs in the direction in which those perturbations decay. Its determinant D_m(lam) satisfies

    D_out(lam) = D_m(lam) * prod_(b > m) det Phi_b(lam),    det Phi_b(lam) = exp(int tr A(x; lam) dx) > 0,

exactly when each inward map is the inverse of the outward one (they agree to the integration's error), so D_m and
D_out have the same zeros and the same signs (notes/MODE_HORIZON.md section 4). The kernel needs no change: the layout
is the Solver's match_node.

Subcommands:
  valley   per segment, from the background alone (Taylor integration along the plan, Simpson on `--sub` points per
           step): Tp = sum_j int dx / (c_j - x), its largest term over the tau points j, Tm = sum_j int dx / (x + c_j);
           the valley segment is the one of largest max_j Tp, and m its inner node
  det      sign, ln|det| and the double-double LU's pivot range at each lam (`--both`: D_out beside D_m)
  root     lowD_gamma.find_lam on D_m (`--both`: and on D_out)
  four     criterion 1: D_m at the four lam of #1078 comment 7120204793 (D 3.044375); passes when the signed values
           f_i = sign_i exp(ln_i - ln_1) lie on one line in lam to 1e-6 of max |f_i| and the pivot ratio is above 1e-20
  d315     criterion 2: the roots of D_m and D_out near lam0 (find_lam, xtol 1e-11) agree to 1e-8
  planted  criterion 3: on the matched layout, the a_h column at the valley's tau point replaced by its value at lam
           minus its value at lam* (a zero column at lam*, so an exact root); the root search returns lam* within
           1e-9. Controls: --mutate=shift subtracts the column at lam* + 1e-6 (the search must then miss lam* by more
           than 1e-9); --mutate=outward plants on the outward layout and reads the signs at lam* -+ 1e-3 ... 1e-6 (one
           clean change there is the control's failure). `--lu=dd|qd|f64`: multidouble's double-double LU (dd_logdet,
           as the kernel's mode_det; the default where the package has it), its quad-double LU after power-of-two
           equilibration (hp/mode_qd.py), or the equilibrated float64 LU (hp/mode_scale.py, the fallback).
  identity criterion 4: at two lam where both are resolved, sign(D_out) = sign(D_m), and the change of
           ln|D_out| - ln|D_m| between them equals (lam1 - lam2) sum_(b > m) sum_j int [1/(c_j - x) - 1/(x + c_j)] dx,
           Liouville's formula for the lam part of ln det Phi_b, to 1e-6 of that change

Each subcommand writes one JSON row per evaluation and a verdict row to OUT; the verdict subcommands exit 1 on FAIL.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FOUR = [5.1408197887857, 5.140819788786, 5.1408197887863025, 5.1408197887866]
D315_LAM0 = 4.397684106690


def say(msg):
    print(msg, flush=True)


def opts_args(argv):
    args = [a for a in argv if not a.startswith("--")]
    o = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in argv if a.startswith("--"))
    return args, o


def make(meta, threads, match_node=None):
    """the state's solver (uniform or tau-mapped, as lowD_gamma.build_solver) with the given match_node"""
    import lowD_gamma as LG
    kw = LG.solver_kw(meta, threads)
    grid = meta.get("map")
    if grid is None:
        from hpkernel.driver import make_solver
        return make_solver(meta["D"], meta["M"], match_node=match_node, **kw)
    import tau_map
    return tau_map.make_solver_mapped(meta["D"], meta["M"], float(grid[0]), float(grid[1]), match_node=match_node, **kw)


def planned(s, zh, zl, hmax):
    segs = s.plan(zh, zl, 1e-32, hmax)
    s.set_segs(segs)
    return segs


def residual_max(s, zh, zl):
    fh, fl = s.residual(zh, zl)
    return float(np.abs(np.asarray(fh) + np.asarray(fl)).max())


def trace_integrals(s, zh, zl, meta, segs, sub=2, log=None):
    """per outward segment b (1..K-1): (b, x0, x1, sum_j Tp, max_j Tp, argmax, sum_j Tm, max_j Tm), with
    Tp_j = int dx / (c_j - x) and Tm_j = int dx / (x + c_j) along the background, by Simpson's rule on `sub` (even)
    equal parts of each step of the plan; each point integrated from the step's start (one Taylor step)"""
    log = log or say
    M, K = meta["M"], len(meta["nodes"])
    sub = max(2, sub + (sub % 2))
    w = np.ones(sub + 1)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    rows = []
    for b in range(1, K):
        xs = segs[b - 1]
        o = 2 * M + 4 * M * (b - 1)
        sh, sl = np.array(zh[o:o + 4 * M]), np.array(zl[o:o + 4 * M])
        tp, tm = np.zeros(M), np.zeros(M)
        c0 = (sh + sl)[3 * M:]
        for j in range(1, len(xs)):
            pts = np.linspace(xs[j - 1], xs[j], sub + 1)
            cs = [c0]
            for q in range(1, sub + 1):
                out = s.integrate(zh, zl, sh, sl, [xs[j - 1], float(pts[q])])
                cs.append((np.asarray(out[0]) + np.asarray(out[1]))[3 * M:])
                if q == sub:
                    nh, nl = np.asarray(out[0]), np.asarray(out[1])
            cs = np.array(cs)
            h = (pts[-1] - pts[0]) / sub
            tp += h / 3 * (w[:, None] / (cs - pts[:, None])).sum(0)
            tm += h / 3 * (w[:, None] / (cs + pts[:, None])).sum(0)
            sh, sl, c0 = nh, nl, cs[-1]
        rows.append(dict(b=b, x0=xs[0], x1=xs[-1], steps=len(xs) - 1, Tp_sum=float(tp.sum()), Tp_max=float(tp.max()),
                         Tp_arg=int(tp.argmax()), Tm_sum=float(tm.sum()), Tm_max=float(tm.max())))
        log(f"  seg {b:2d} [{xs[0]:.6f}, {xs[-1]:.6f}] {len(xs) - 1:4d} steps: sum_j Tp {tp.sum():.6f}, max_j Tp "
            f"{tp.max():.4f} (j {tp.argmax()}), sum_j Tm {tm.sum():.6f}, max_j Tm {tm.max():.4f}")
    return rows


def valley_node(rows):
    """(m, row): the valley segment is the one of largest max_j Tp; m is its inner node"""
    r = max(rows, key=lambda t: t["Tp_max"])
    return r["b"] - 1, r


class Case:
    """a state with its outward and matched solvers, each on its own plan"""

    def __init__(self, path, o, log=None, need=("out", "m")):
        from hpkernel.driver import load
        log = log or say
        self.path, self.log = path, log
        self.zh, self.zl, self.meta = load(path)
        self.threads = int(o.get("threads", 6))
        self.hmax = float(o.get("hmax", 0.05))
        self.M, self.K = self.meta["M"], len(self.meta["nodes"])
        self.out = make(self.meta, self.threads)
        self.segs_out = planned(self.out, self.zh, self.zl, self.hmax)
        self.F_out = residual_max(self.out, self.zh, self.zl)
        match = o.get("match", "valley")
        self.rows = None
        if match == "out":
            self.m = self.K - 1
        elif match == "valley":
            self.rows = trace_integrals(self.out, self.zh, self.zl, self.meta, self.segs_out, int(o.get("sub", 2)), log)
            self.m, _ = valley_node(self.rows)
        else:
            self.m = int(match)
        self.mat = make(self.meta, self.threads, match_node=self.m) if "m" in need else None
        self.segs_m = planned(self.mat, self.zh, self.zl, self.hmax) if self.mat else None
        self.F_m = residual_max(self.mat, self.zh, self.zl) if self.mat else None
        log(f"{os.path.basename(path)}: D {self.meta['D']} M {self.M} K {self.K}; outward plan "
            f"{sum(len(g) - 1 for g in self.segs_out)} steps, |F| {self.F_out:.2e}; match node {self.m} "
            f"(x {self.meta['nodes'][self.m]})"
            + (f", matched plan {sum(len(g) - 1 for g in self.segs_m)} steps, |F| {self.F_m:.2e}" if self.mat else ""))

    def info(self):
        return dict(state=os.path.basename(self.path), D=self.meta["D"], M=self.M, K=self.K, m=self.m,
                    x_m=self.meta["nodes"][self.m], F_out=self.F_out, F_m=self.F_m, hmax=self.hmax,
                    steps_out=sum(len(g) - 1 for g in self.segs_out),
                    steps_m=sum(len(g) - 1 for g in self.segs_m) if self.segs_m else None)

    def det(self, which, lam):
        s = self.out if which == "out" else self.mat
        t = time.time()
        sg, la, pmin, pmax = s.mode_det(self.zh, self.zl, float(lam), 0.0)
        r = dict(which=which, lam=float(lam), sign=float(sg), ln=float(la), pmin=float(pmin), pmax=float(pmax),
                 ratio=float(pmin) / float(pmax), sec=round(time.time() - t, 1))
        self.log(f"    {which:3s} lam {lam!r}: sign {sg:+.0f} ln|det| {la:.9f} pivots {pmin:.2e}..{pmax:.2e} "
                 f"({r['sec']} s)")
        return r


def writer(out):
    def w(row):
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return w


def line_residual(lams, f):
    """largest residual of the least-squares line through (lams, f), relative to max |f|"""
    x = np.asarray(lams, float) - lams[0]
    A = np.vstack([np.ones_like(x), x * 1e12]).T
    c, *_ = np.linalg.lstsq(A, np.asarray(f, float), rcond=None)
    return float(np.abs(A @ c - f).max() / np.abs(f).max())


def cmd_valley(args, o):
    from hpkernel.driver import load
    zh, zl, meta = load(args[0])
    s = make(meta, int(o.get("threads", 6)))
    segs = planned(s, zh, zl, float(o.get("hmax", 0.05)))
    rows = trace_integrals(s, zh, zl, meta, segs, int(o.get("sub", 2)))
    m, r = valley_node(rows)
    print(f"valley: segment {r['b']} [{r['x0']}, {r['x1']}], max_j Tp {r['Tp_max']:.4f}; match node m = {m} "
          f"(x {meta['nodes'][m]}); outward part left: max_j Tp over segments b <= m "
          f"{max(t['Tp_max'] for t in rows if t['b'] <= m):.4f}")
    return 0


def cmd_det(args, o):
    out, path, lams = args[0], args[1], [float(v) for v in args[2:]]
    w = writer(out)
    c = Case(path, o, need=("out", "m"))
    for lam in lams:
        for which in (("out", "m") if "both" in o else ("m",)):
            w(dict(kind="det", **c.info(), **c.det(which, lam)))
    return 0


def cmd_root(args, o):
    import lowD_gamma as LG
    out, path = args[0], args[1]
    w = writer(out)
    c = Case(path, o)
    lam0 = float(o.get("lam0", LG.lam_guess(c.meta["D"])))
    xtol = float(o.get("xtol", 1e-11))
    res = {}
    for which in (("m", "out") if "both" in o else ("m",)):
        s = c.mat if which == "m" else c.out
        t = time.time()
        lam, nch, nev = LG.find_lam(s, c.zh, c.zl, lam0, c.log, xtol=xtol)
        res[which] = (lam, nch)
        w(dict(kind="root", which=which, **c.info(), lam0=lam0, xtol=xtol, lam=lam, gamma=(1 / lam if lam else None),
               n_signchanges=nch, n_det=nev, sec=round(time.time() - t, 1)))
        c.log(f"  root of D_{which}: lam {lam!r} ({nch} sign changes in the first bracket, {nev} dets)")
    return res


def cmd_four(args, o):
    out, path = args[0], args[1]
    w = writer(out)
    c = Case(path, o, need=("out", "m"))
    rs = [c.det("m", lam) for lam in FOUR]
    for r in rs:
        w(dict(kind="four", **c.info(), **r))
    f = [r["sign"] * math.exp(r["ln"] - rs[0]["ln"]) for r in rs]
    lr = line_residual(FOUR, f)
    ratio = min(r["ratio"] for r in rs)
    ok = lr <= 1e-6 and ratio > 1e-20
    v = dict(kind="verdict", test="four", **c.info(), f=f, line_residual=lr, min_pivot_ratio=ratio,
             signs=[r["sign"] for r in rs], ln=[r["ln"] for r in rs], passed=ok)
    w(v)
    c.log(f"criterion 1 (four lam): f {['%.12g' % x for x in f]}, line residual {lr:.1e} (<= 1e-6), smallest pivot "
          f"ratio {ratio:.1e} (> 1e-20): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_d315(args, o):
    out, path = args[0], args[1]
    o = dict(o)
    o.setdefault("lam0", str(D315_LAM0))
    o["both"] = "1"
    res = cmd_root([out, path], o)
    (lm, nm), (lo, no) = res["m"], res["out"]
    ok = lm is not None and lo is not None and abs(lm - lo) <= 1e-8 and nm == no
    writer(out)(dict(kind="verdict", test="d315", state=os.path.basename(path), lam_m=lm, lam_out=lo,
                     diff=(abs(lm - lo) if lm and lo else None), n_m=nm, n_out=no, passed=ok))
    print(f"criterion 2 (D 3.15): root D_m {lm!r}, D_out {lo!r}, |diff| {abs(lm - lo) if lm and lo else None} "
          f"(<= 1e-8), sign changes {nm} and {no}: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def logdet_matrix(Mh, Ml, lu):
    """sign, ln|det|, min and max |u_kk| of a double-double matrix (hi, lo): multidouble's double-double LU with
    partial pivoting (dd_logdet, the kernel mode_det's DDLu), its quad-double LU after power-of-two equilibration
    (mode_qd.qd_det), or the equilibrated float64 LU (mode_scale.scaled_det)"""
    if lu == "dd":
        import multidouble
        sg, la, pmin, pmax = multidouble.dd_logdet(np.ascontiguousarray(Mh), np.ascontiguousarray(Ml), par=True)
        return float(sg), float(la), float(pmin), float(pmax)
    if lu == "qd":
        import mode_qd
        d = mode_qd.qd_det(Mh, Ml)
    else:
        import mode_scale
        d = mode_scale.scaled_det(Mh, Ml)
    return d["sign"], d["ln"], d["pmin"], d["pmax"]


def default_lu():
    """dd when this multidouble has dd_logdet (5.1.0, the commit hp/pyproject.toml pins), else f64"""
    import multidouble
    return "dd" if hasattr(multidouble, "dd_logdet") else "f64"


def cmd_planted(args, o):
    from multidouble import dd_add
    from scipy.optimize import brentq
    out, path = args[0], args[1]
    w = writer(out)
    mut = o.get("mutate")
    lam_star = float(o.get("lam_star", 5.0))
    lu = o.get("lu", default_lu())
    if lu == "f64":
        print("planted: no dd_logdet in this multidouble (older than 5.1.0); the equilibrated float64 LU is used",
              flush=True)
    c = Case(path, o, need=("out", "m"))
    s = c.out if mut == "outward" else c.mat
    M, K = c.M, c.K
    # the a_h column at the valley's tau point: mode-matrix columns are z's before Delta, a_h at 3M + 4MK
    jv = 0
    if c.rows:
        jv = max(c.rows, key=lambda t: t["Tp_max"])["Tp_arg"]
    jcol = 3 * M + 4 * M * K + jv
    lam_p = lam_star + (1e-6 if mut == "shift" else 0.0)
    t = time.time()
    Ph, Pl = s.mode_matrix(c.zh, c.zl, lam_p, 0.0)
    vh, vl = np.array(np.asarray(Ph)[:, jcol]), np.array(np.asarray(Pl)[:, jcol])
    del Ph, Pl
    c.log(f"  planted column {jcol} (a_h at tau point {jv}) at lam {lam_p!r} ({time.time() - t:.1f} s); LU {lu}")
    cache = {}

    def ev(lam):
        if lam not in cache:
            t = time.time()
            Mh, Ml = s.mode_matrix(c.zh, c.zl, float(lam), 0.0)
            Mh, Ml = np.asarray(Mh), np.asarray(Ml)                 # the kernel's own arrays, changed in place
            ch, cl = dd_add(np.ascontiguousarray(Mh[:, jcol]), np.ascontiguousarray(Ml[:, jcol]), -vh, -vl)
            Mh[:, jcol], Ml[:, jcol] = ch, cl
            sg, la, pmin, pmax = logdet_matrix(Mh, Ml, lu)
            cache[lam] = (sg, la)
            row = dict(kind="planted_det", **c.info(), layout=("out" if mut == "outward" else "m"), mutate=mut, lam=lam,
                       lam_star=lam_star, jcol=jcol, lu=lu, sign=sg, ln=la, pmin=pmin, pmax=pmax,
                       sec=round(time.time() - t, 1))
            w(row)
            c.log(f"    planted lam {lam!r}: sign {sg:+.0f} ln|det| {la:.9f} pivots {pmin:.2e}..{pmax:.2e} ({row['sec']} s)")
        return cache[lam]

    if mut == "outward":
        ds = [1e-3, 1e-4, 1e-5, 1e-6]
        left = [ev(lam_star - d)[0] for d in ds]
        right = [ev(lam_star + d)[0] for d in ds]
        clean = len(set(left)) == 1 and len(set(right)) == 1 and left[0] == -right[0]
        ok = not clean
        w(dict(kind="verdict", test="planted_outward_control", **c.info(), lam_star=lam_star, left=left, right=right,
               clean=clean, control_fired=ok))
        c.log(f"control (outward layout): signs left {left}, right {right}: "
              f"{'one clean change (the control did NOT fire)' if clean else 'no clean change (the control fired)'}")
        return 0 if ok else 1
    root = None
    for wd in (1e-4, 1e-3, 1e-2):
        a, b = lam_star - wd, lam_star + wd
        sa, sb = ev(a)[0], ev(b)[0]
        if sa != sb:
            ref = 0.5 * (ev(a)[1] + ev(b)[1])
            f = lambda l: ev(l)[0] * math.exp(ev(l)[1] - ref)
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-15, maxiter=60)
            break
    err = abs(root - lam_star) if root is not None else None
    ok = err is not None and err <= 1e-9
    w(dict(kind="verdict", test="planted", **c.info(), mutate=mut, lam_star=lam_star, lam_plant=lam_p, root=root,
           err=err, n_det=len(cache), passed=ok))
    c.log(f"criterion 3 (planted{', control ' + mut if mut else ''}): root {root!r}, |root - lam*| {err} (<= 1e-9): "
          f"{'PASS' if ok else 'FAIL'}" + (" (the control must FAIL)" if mut else ""))
    return 0 if ok else 1


def cmd_identity(args, o):
    out, path, l1, l2 = args[0], args[1], float(args[2]), float(args[3])
    w = writer(out)
    o = dict(o)
    o.setdefault("sub", "8")
    c = Case(path, o)
    rows = c.rows or trace_integrals(c.out, c.zh, c.zl, c.meta, c.segs_out, int(o["sub"]), c.log)
    I = sum(r["Tp_sum"] - r["Tm_sum"] for r in rows if r["b"] > c.m)
    d = {}
    for lam in (l1, l2):
        d[lam] = (c.det("out", lam), c.det("m", lam))
    gap = [d[l][0]["ln"] - d[l][1]["ln"] for l in (l1, l2)]
    change, want = gap[0] - gap[1], (l1 - l2) * I
    signs = all(d[l][0]["sign"] == d[l][1]["sign"] for l in (l1, l2))
    rel = abs(change - want) / abs(want)
    ok = signs and rel <= 1e-6
    w(dict(kind="verdict", test="identity", **c.info(), lam=[l1, l2], gap=gap, change=change, liouville=want,
           rel=rel, signs_agree=signs, dets=[d[l1], d[l2]], passed=ok))
    c.log(f"criterion 4 (identity): ln|D_out| - ln|D_m| = {gap[0]:.9f} and {gap[1]:.9f}; change {change:.9f}, "
          f"Liouville (lam1 - lam2) * {I:.9f} = {want:.9f}, relative difference {rel:.1e} (<= 1e-6); signs agree "
          f"{signs}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_blocks(args, o):
    """per outward segment, the change of ln|det Phi_b| between two lam (float64 slogdet of the mode matrix's blocks)
    against Liouville's lam part of the trace integral, on the plan with --hmax; with --match also the inward blocks"""
    out, path, l1, l2 = args[0], args[1], float(args[2]), float(args[3])
    w = writer(out)
    o = dict(o)
    o.setdefault("sub", "8")
    o.setdefault("match", "out")
    c = Case(path, o, need=("out", "m") if o["match"] != "out" else ("out",))
    rows = c.rows or trace_integrals(c.out, c.zh, c.zl, c.meta, c.segs_out, int(o["sub"]), c.log)
    M, K, n4 = c.M, c.K, 4 * c.M
    col = lambda i: 2 * M + n4 * i
    lds = {}
    for lam in (l1, l2):
        for tag, s in (("out", c.out), ("m", c.mat)):
            if s is None:
                continue
            t = time.time()
            Mh, Ml = s.mode_matrix(c.zh, c.zl, lam, 0.0)
            A = np.asarray(Mh) + np.asarray(Ml)
            del Mh, Ml
            per = []
            for b in range(1, K):
                j0 = col(b) if (tag == "m" and b > c.m) else col(b - 1)   # an inward row reads node b
                T = A[b * n4:(b + 1) * n4, j0:j0 + n4]
                sv = np.linalg.svd(T, compute_uv=False)
                per.append((float(np.sum(np.log(sv))), float(sv.max()), float(sv.min())))
            sg, ld = np.linalg.slogdet(A)
            lds[(tag, lam)] = per
            c.log(f"    {tag} lam {lam}: full float64 sign {sg:+.0f} ln|det| {ld:.9f} ({time.time() - t:.0f} s)")
            del A
    for b in range(1, K):
        I = rows[b - 1]["Tp_sum"] - rows[b - 1]["Tm_sum"]
        d = lds[("out", l1)][b - 1][0] - lds[("out", l2)][b - 1][0]
        row = dict(kind="block", **c.info(), b=b, steps=len(c.segs_out[b - 1]) - 1, lam=[l1, l2],
                   liouville=I * (l1 - l2), dln_out=d, diff=d - I * (l1 - l2), sv_out=lds[("out", l1)][b - 1][1:])
        if c.mat is not None and b > c.m:
            row["dln_in"] = -(lds[("m", l1)][b - 1][0] - lds[("m", l2)][b - 1][0])
        w(row)
        c.log(f"  seg {b:2d} {row['steps']:4d} steps: Liouville {row['liouville']:.9f}, ln det Phi_b {d:.9f}, "
              f"difference {row['diff']:+.2e}; sv {row['sv_out'][0]:.2e}..{row['sv_out'][1]:.2e}")
    return 0


def toy_A(x, lam, P):
    """the toy's coefficient matrix at x: field 0 has the valley's rate (lam - kappa) / (gap + beta (x - xv)^2), field 1
    decays at lam / 2, fields 2 and 3 are lam-independent, and a fixed coupling C ties them"""
    q = 1.0 / (P["gap"] + P["beta"] * (x - P["xv"]) ** 2)
    return np.diag([(lam - P["kappa"]) * q, -0.5 * lam, 0.3, -0.2]) + P["C"]


def toy_transfers(lam, P, prec):
    """per segment b = 1..K-1, the transfer over it (outward) and its inverse integrated inward, as products of
    exponentials of A at the steps' midpoints: float64 (prec None, scipy's expm) or mpmath at prec digits (the truth)"""
    import mpmath as mp
    from scipy.linalg import expm
    out, inw = [], []
    for b in range(1, len(P["nodes"])):
        xs = np.linspace(P["nodes"][b - 1], P["nodes"][b], P["steps"][b - 1] + 1)
        if prec is None:
            F, G = np.eye(4), np.eye(4)
            for k in range(len(xs) - 1):
                h, xm = xs[k + 1] - xs[k], 0.5 * (xs[k] + xs[k + 1])
                A = toy_A(xm, lam, P)
                F = expm(A * h) @ F
                G = G @ expm(-A * h)
            out.append(F)
            inw.append(G)
        else:
            with mp.workdps(prec):
                F = mp.eye(4)
                for k in range(len(xs) - 1):
                    h, xm = mp.mpf(xs[k + 1]) - mp.mpf(xs[k]), (mp.mpf(xs[k]) + mp.mpf(xs[k + 1])) / 2
                    A = mp.matrix(toy_A(float(xm), 0.0, P).tolist())
                    q = 1 / (mp.mpf(P["gap"]) + mp.mpf(P["beta"]) * (xm - mp.mpf(P["xv"])) ** 2)
                    A[0, 0] = (mp.mpf(lam) - mp.mpf(P["kappa"])) * q + mp.mpf(P["C"][0, 0])
                    A[1, 1] = -mp.mpf(lam) / 2 + mp.mpf(P["C"][1, 1])
                    F = mp.expm(A * h) * F
                out.append(F)
                inw.append(None)
    return out, inw


def toy_matrix(lam, P, m, H, prec=None):
    """the toy's multiple-shooting matrix with segments past node m inward (m = K - 1: all outward), centre block C0
    (4 x 2) and horizon block H (4 x 2), float64"""
    K = len(P["nodes"])
    n = 4 * (K + 1)
    out, inw = toy_transfers(lam, P, None)
    A = np.zeros((n, n))
    c0 = lambda i: 2 + 4 * i
    A[0:4, 0:2] = P["C0"]
    A[0:4, c0(0):c0(0) + 4] = -np.eye(4)
    for b in range(1, K):
        r = slice(4 * b, 4 * b + 4)
        if b <= m:
            A[r, c0(b - 1):c0(b - 1) + 4] = out[b - 1]
            A[r, c0(b):c0(b) + 4] = -np.eye(4)
        else:
            A[r, c0(b):c0(b) + 4] = inw[b - 1]
            A[r, c0(b - 1):c0(b - 1) + 4] = -np.eye(4)
    A[4 * K:4 * K + 4, c0(K - 1):c0(K - 1) + 4] = -np.eye(4)
    A[4 * K:4 * K + 4, c0(K):c0(K) + 2] = H
    return A


def toy_T(P):
    """the valley segment's int dx / (gap + beta (x - xv)^2)"""
    s = math.sqrt(P["gap"] * P["beta"])
    a, b = P["nodes"][2] - P["xv"], P["nodes"][3] - P["xv"]
    k = math.sqrt(P["beta"] / P["gap"])
    return (math.atan(b * k) - math.atan(a * k)) / s


def toy_case(lam_star, mutate=None, gap=2e-3):
    """the toy's parameters and its horizon block H(lam) = H0 + (lam - lam*) H1, with H0's first column the centre
    solutions' non-growing combination at lam* computed at 60 digits, so that the exact Evans matrix is singular at
    lam* along a direction the outward transfer forms only by cancellation"""
    import mpmath as mp
    rng = np.random.default_rng(1083)
    P = dict(gap=gap, beta=40.0, xv=0.45, kappa=0.6, C=0.1 * rng.standard_normal((4, 4)),
             nodes=[0.0, 0.2, 0.4, 0.5, 0.7, 1.0], steps=[40, 40, 400, 40, 60], C0=rng.standard_normal((4, 2)))
    lp = lam_star + (1e-6 if mutate == "shift" else 0.0)
    out, _ = toy_transfers(lp, P, 60)
    with mp.workdps(60):
        Y = mp.matrix(P["C0"].tolist())
        for F in out:
            Y = F * Y
        # the centre solutions' combination that does not grow across the valley (Y's smallest right singular
        # vector), as a mode's eigenfunction does: regular at both ends, so free of the outward-growing direction
        _, _, V = mp.svd_r(Y)
        w = V[V.rows - 1, :].T
        v = Y * w
        v = v / mp.norm(v)
    H0 = np.column_stack([np.array([float(t) for t in v]), rng.standard_normal(4)])
    H1 = rng.standard_normal((4, 2))
    return P, (lambda lam: H0 + (lam - lam_star) * H1)


def cmd_toy(args, o):
    """the planted toy (no kernel): a four-field linear system with a valley and a root planted at lam* = 5;
    outward and matched determinants in float64, roots by brentq. The valley stretches field 0 by exp((lam - 0.6) T),
    T = toy_T (10.1 at gap 2e-3: e^44 at lam*, past float64's 1/eps; 1.35 at gap 0.05)"""
    from scipy.optimize import brentq
    lam_star = 5.0
    mut = o.get("mutate")
    P, H = toy_case(lam_star, mutate=mut, gap=float(o.get("gap", 2e-3)))
    K = len(P["nodes"])
    m = 2                                          # the valley segment 3 (0.4 -> 0.5) runs inward from node 3 to 2

    def det(lam, mm):
        s, l = np.linalg.slogdet(toy_matrix(lam, P, mm, H(lam)))
        return s, l

    def root(mm):
        for w in (1e-4, 1e-3, 1e-2):
            a, b = lam_star - w, lam_star + w
            (sa, la), (sb, lb) = det(a, mm), det(b, mm)
            if sa != sb:
                ref = 0.5 * (la + lb)
                return brentq(lambda l: det(l, mm)[0] * math.exp(det(l, mm)[1] - ref), a, b, xtol=1e-14, rtol=1e-15)
        return None

    rm, ro = root(m), root(K - 1)
    em = abs(rm - lam_star) if rm is not None else None
    eo = abs(ro - lam_star) if ro is not None else None
    # the identity at a lam where both are resolved: ln|D_out| - ln|D_m| = sum_(b>m) ln det Phi_b, Liouville
    lam1 = 1.0
    out, _ = toy_transfers(lam1, P, None)
    liou = sum(np.linalg.slogdet(out[b - 1])[1] for b in range(m + 1, K))
    so, lo = det(lam1, K - 1)
    sm, lm = det(lam1, m)
    ident = abs((lo - lm) - liou)
    ok = em is not None and em <= 1e-9 and ident <= 1e-8 and so == sm
    print(f"toy (gap {P['gap']}, T {toy_T(P):.2f}{', mutate ' + mut if mut else ''}): "
          f"matched root {rm!r} (|err| {em}), outward root {ro!r} (|err| {eo}); identity at lam 1: signs "
          f"{so:+.0f} {sm:+.0f}, ln|D_out| - ln|D_m| - sum ln det Phi_b = {ident:.1e}: {'PASS' if ok else 'FAIL'}",
          flush=True)
    return 0 if ok else 1


def main():
    from raw_out import pop_allow, require_raw
    argv = sys.argv[1:]
    allow = pop_allow(argv)
    args, o = opts_args(argv)
    if not args:
        print(__doc__)
        return 2
    cmd, rest = args[0], args[1:]
    if cmd not in ("valley", "toy"):
        require_raw(rest[0], allow=allow)
    fn = dict(valley=cmd_valley, det=cmd_det, root=cmd_root, four=cmd_four, d315=cmd_d315, planted=cmd_planted,
              identity=cmd_identity, toy=cmd_toy, blocks=cmd_blocks).get(cmd)
    if fn is None:
        print(__doc__)
        return 2
    r = fn(rest, o)
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
