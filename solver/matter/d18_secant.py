#!/usr/bin/env python3
"""#1025: refine a bracketed root of det M(lambda) (the l = 1 mode determinant) on d18_hybsign.py's hybrid sign, by Brent's
method on the signed hybrid determinant.

Each evaluation is d18_hybsign.py's: M(lambda) in float64 (blocks 0-11; ell_fastm.mode_matrix_fast by default, or
DSSEll.mode_matrix with --mm=cs), block 12 replaced by its double-double version (ell_ddm.DDEll's by default, or
ell_fastdd's with --b12=fast), the sign and
ln|det| of multidouble's dd_logdet, and the noise test at 1e-15 on blocks 0-11 with the seeds 100 and 101; the plain
float64 slogdet beside it. Each record adds the noise members' ln|det| and dln, the largest |ln|det_noise| - ln|det||.
An evaluation is robust when both noise signs equal its sign (d18_hybsign's test) and dln <= --dln (default 0.1, a
change of |det| by 10% under the 1e-15 perturbation; --dln=inf is d18_hybsign's test alone).

The refinement: f(lambda) = sign exp(ln|det| - ref), ref the larger end's ln|det|, and Brent's zero finder on it
(inverse quadratic or secant steps, bisection when they do not shrink the bracket fast enough; R. P. Brent, Algorithms
for Minimization without Derivatives (Prentice-Hall, 1973), ch. 4 *(unverified)*, in the form of scipy's brentq). The bracket ends always carry opposite robust signs: the two input ends are evaluated first and the
run is refused (exit 2) unless both are robust with opposite signs; an interior evaluation that is not robust stops the
run with the bracket it had (status nonrobust). Otherwise the run stops when the bracket is at most --tol wide (status
tol) or after --maxev evaluations, the two ends included (status maxev).
Output (OUT.jsonl, appended): one record per evaluation (kind "eval"), then one record of kind "root": status, bracket
and its width, the ends' signs, the det-linear root inside the final bracket (lam_lin), Brent's last point (lam_brent),
the evaluation count and seconds.

usage (venv-1025dd):
  d18_secant.py run STUB.jsonl STATE.npy D LO HI OUT.jsonl [--tol=1e-6] [--maxev=12] [--mm=fast|cs] [--dln=0.1]
                [--basis=jac|coef|grid] [--b12=ddm|fast] [--eps=1e-15] [--pin=STATEBLOB,STUBBLOB]
                [--ends=SCAN.jsonl]
      --mm=cs is d18_hybsign.py's evaluation exactly (DSSEll.mode_matrix, ell_ddm's block 12); the default float64
      matrix is ell_fastm.mode_matrix_fast (basis jac). --b12=fast takes ell_fastdd's block 12 (the same double-double
      block to 4e-25), faster at k_c 55 (1.4 to 1.7x) but not at k_c 103 (0.8x);
      --eps: the noise test's relative amplitude (d18_hybsign.py's 1e-15); the root's float64 floor from blocks 0-11
      is larger than that test sees (SPEED_ELL1.md), and 1e-12 makes the robustness test conservative;
      --pin: refuse (exit 3) unless the state's and the stub's git blobs (hash-object --no-filters) start with these;
      --ends: take the two ends' records from a d18_hybsign scan file of the same state instead of evaluating them
      (their dln is unknown, so their robustness is d18_hybsign's test).
      exit 0: tol reached; 4: stopped by maxev or a non-robust interior point (the bracket is still valid); 2: refused;
      3: not the record system or a pin differs.
  d18_secant.py scan STUB.jsonl STATE.npy D LAMS OUT.jsonl [--mm=fast|cs] [--basis=..] [--b12=..] [--eps=1e-15]
                [--dln=0.1] [--pin=STATEBLOB,STUBBLOB]
      d18_hybsign.py's scan (LAMS comma-separated) through the same evaluation as run, the fast float64 matrix by
      default: one record per lambda with d18_hybsign.py's fields (D, kc, lam, sign, ln, noise, noise_robust,
      f64_sign, f64_ln, state, sec) and run's (noise_ln, dln, robust, mm, b12, sec_parts), role "scan". With --mm=cs a
      record equals d18_hybsign.py's bit for bit (raw I1025-mac/speed_ell1/runs/scan55). exit 3: not the record
      system or a pin differs.
  d18_secant.py selftest [--mutate=half|norobust|nosigncheck]
      the planted cases below through the same refinement and the same dd_logdet sign and noise code; exit 0 only if every
      case reads as written.

Planted cases (written before the first run): a 24 x 24 matrix with det = s(lambda) det R, R a fixed well-conditioned
random block, s the determinant of the 2 x 2 block [[1, 1], [1, 1 + s]] (1 + s carried exactly in double-double), so the
noise test's perturbation moves det by about 2e-15 det R; s_true = k1 (lam - r) exp(300 (lam - r)), r = 0.48731234,
bracket (0.485, 0.490), tol 1e-6.
  C1 clean (k1 = 1): status tol; r in the final bracket; width <= 1e-6; |lam_lin - r| <= 1e-6; at most 10 evaluations;
     every evaluation robust with the planted sign.
  C2 noisy (k1 = 1e-10, and inside |lam - r| < 5e-5 the computed s is s_true - 1e-15, a spurious root at r + 1e-5 with
     wrong signs between r and it): status tol or nonrobust; r in the final bracket; both ends robust with the planted
     sign; every evaluation marked robust has the planted sign.
  C3 wrongly signed bracket (r + 0.001, r + 0.006), clean: refused after the 2 end evaluations.
  C4 an end inside the noise (C2's matrix, bracket (r - 0.003, r + 2e-6)): refused after the 2 end evaluations, because
     an end is not robust (this reason was added to C4 after the first run: under norobust the end at r + 2e-6, which
     carries the spurious sign, made C4 refuse for same-signed ends instead).
  Controls (each must make the selftest exit 1): half (the bracket keeps the wrong end: C1 fails), norobust (every
  evaluation counts as robust: C2 ends at the spurious root, C4 is refused for another reason), nosigncheck (same-signed ends accepted: C3
  runs)."""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WTM = os.environ.get("P4_WT_MATTER", HERE)
sys.path.insert(0, WTM)
import multidouble as md                          # noqa: E402

MUTATE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")
SEEDS = (100, 101)


# ---------------------------------------------------------------- one hybrid evaluation (matrix -> record)
def hybrid_eval(H, Hl, mask, dln_max, eps=1e-15):
    """d18_hybsign.py's sign, ln|det| and noise test (relative amplitude eps, 1e-15 there) of the hybrid matrix (H, Hl),
    with the noise members' ln|det|"""
    s, ld = md.dd_logdet(H, Hl, True)[:2]
    nz = [md.dd_logdet(H + eps * np.random.default_rng(sd).standard_normal(H.shape) * mask * H, Hl, True)[:2]
          for sd in SEEDS]
    nsg = [float(v[0]) for v in nz]
    nln = [float(v[1]) for v in nz]
    dln = max(abs(v - float(ld)) for v in nln)
    nrob = all(x == s for x in nsg)
    rob = nrob and dln <= dln_max
    if MUTATE == "norobust":
        rob = True
    return dict(sign=float(s), ln=float(ld), noise=nsg, noise_ln=nln, dln=dln, noise_robust=nrob, robust=rob)


class RealEval:
    """d18_hybsign.py's hybrid matrix of a stub record's state (the record system checked by the caller)"""

    def __init__(self, S, z, rec, st, mm, dln_max, basis="jac", b12="ddm", eps=1e-15):
        import ell_ddm
        self.eps = eps
        self.S, self.z, self.rec, self.st, self.mm, self.dln_max, self.basis = S, z, rec, st, mm, dln_max, basis
        self.E = ell_ddm.DDEll(S)
        self.b12 = b12
        if b12 == "fast":
            from ell_fastdd import DDJac
            self.J = DDJac(self.E)
        blocks, offs = S._block_list()
        blk = 3 * S.n_odd + 2 * S.n_even
        c0, nf, fun, tgt = blocks[12]
        r0 = 12 * blk
        self.sl = (slice(r0, r0 + blk), slice(c0, c0 + nf))
        self.mask = np.zeros((S.layout()[2] - 1,) * 2, bool)
        for b, (cc, nn, ff, tt) in enumerate(blocks[:12]):
            self.mask[b * blk:(b + 1) * blk, cc:cc + nn] = True

    def __call__(self, lam):
        t0 = time.time()
        with np.errstate(all="ignore"):
            if self.mm == "fast":
                from ell_fastm import mode_matrix_fast
                Mf = mode_matrix_fast(self.S, self.z, lam, basis=self.basis)
            else:
                Mf = self.S.mode_matrix(self.z, lam)
        t1 = time.time()
        sf, lf = np.linalg.slogdet(Mf)
        H, Hl = Mf.copy(), np.zeros_like(Mf)
        Md = self.J.mode_matrix(self.z, lam) if self.b12 == "fast" else self.E.mode_matrix(self.z, lam, only=[12])
        H[self.sl] = Md.h[self.sl]
        Hl[self.sl] = Md.l[self.sl]
        t2 = time.time()
        r = hybrid_eval(H, Hl, self.mask, self.dln_max, self.eps)
        r = dict(D=self.rec["D"], kc=self.rec["kc"], lam=lam, **r, f64_sign=float(sf), f64_ln=float(lf),
                 state=os.path.basename(self.st), mm=self.mm if self.mm == "cs" else f"fast-{self.basis}", b12=self.b12,
                 sec=round(time.time() - t0),
                 sec_parts=[round(t1 - t0), round(t2 - t1), round(time.time() - t2)])
        return r


class PlantedEval:
    """det = s(lam) det R through the same hybrid_eval (see the module docstring)"""

    def __init__(self, root, k1, bias=0.0, zone=0.0, dln_max=0.1, n=24, seed=5):
        rng = np.random.default_rng(seed)
        q1, _ = np.linalg.qr(rng.standard_normal((n - 2, n - 2)))
        q2, _ = np.linalg.qr(rng.standard_normal((n - 2, n - 2)))
        self.R = q1 @ np.diag(1.0 + 2.0 * rng.random(n - 2)) @ q2
        self.sR = float(np.linalg.slogdet(self.R)[0])
        self.Bc = 0.3 * rng.standard_normal((2, n - 2))           # coupling above the diagonal: det unchanged
        self.root, self.k1, self.bias, self.zone, self.dln_max, self.n = root, k1, bias, zone, dln_max, n
        self.mask = np.zeros((n, n), bool)
        self.mask[:n - 6] = True                                 # the last 6 rows play block 12 (not perturbed)

    def s_true(self, lam):
        d = lam - self.root
        return self.k1 * d * math.exp(300.0 * d)

    def true_sign(self, lam):
        return float(np.sign(self.s_true(lam)) * self.sR)

    def __call__(self, lam):
        t0 = time.time()
        s = self.s_true(lam) + (self.bias if abs(lam - self.root) < self.zone else 0.0)
        n = self.n
        H = np.zeros((n, n))
        Hl = np.zeros((n, n))
        H[0, 0] = H[0, 1] = H[1, 0] = 1.0
        hi = 1.0 + s
        H[1, 1] = hi
        Hl[1, 1] = (1.0 - hi) + s                               # two-sum: hi + lo = 1 + s exactly (|s| < 1)
        H[:2, 2:] = self.Bc
        H[2:, 2:] = self.R
        r = hybrid_eval(H, Hl, self.mask, self.dln_max)
        return dict(lam=lam, **r, true_sign=self.true_sign(lam), sec=round(time.time() - t0, 3))


# ---------------------------------------------------------------- Brent's method on the signed determinant
def refine(evalf, lo, hi, tol=1e-6, maxev=12, out=None, ends=None):
    """Returns (final record, evaluation records). ends: optional (record at lo, record at hi) to use instead of
    evaluating them."""
    recs = []
    t_start = time.time()

    def ev(lam, role):
        r = dict(evalf(lam))
        r.update(kind="eval", it=len(recs), role=role)
        recs.append(r)
        if out:
            out(r)
        return r

    if ends is not None:
        ra, rb = (dict(e, kind="eval", it=i, role=("lo", "hi")[i], reused=True) for i, e in enumerate(ends))
        recs += [ra, rb]
        if out:
            out(ra)
            out(rb)
    else:
        ra = ev(lo, "lo")
        rb = ev(hi, "hi")

    def final(status, ra_, rb_, xbest, why=""):
        a, b = sorted([ra_, rb_], key=lambda r: r["lam"])
        fa, fb = g(a), g(b)
        lin = a["lam"] - fa * (b["lam"] - a["lam"]) / (fb - fa) if fa != fb else None
        return dict(kind="root", status=status, why=why, bracket=[a["lam"], b["lam"]], width=b["lam"] - a["lam"],
                    signs=[a["sign"], b["sign"]], robust=[a["robust"], b["robust"]], lam_lin=lin, lam_brent=xbest,
                    n_eval=len(recs), tol=tol, sec=round(time.time() - t_start)), recs

    ref = max(ra["ln"], rb["ln"])

    def g(r):
        return r["sign"] * math.exp(min(r["ln"] - ref, 700.0))
    if not (ra["robust"] and rb["robust"]):
        return final("refused", ra, rb, None, "an end is not robust")
    if ra["sign"] == rb["sign"] and MUTATE != "nosigncheck":
        return final("refused", ra, rb, None, "the ends have the same hybrid sign")
    xpre, xcur, fpre, fcur, rpre, rcur = lo, hi, g(ra), g(rb), ra, rb
    xblk, fblk, rblk = 0.0, 0.0, None
    spre = scur = 0.0
    delta = tol / 2
    while True:
        opposite = (fpre < 0) != (fcur < 0)
        if MUTATE == "half" and len(recs) > 2:                     # the control: interior points replace the wrong end
            opposite = not opposite
        if fpre != 0 and fcur != 0 and opposite:
            xblk, fblk, rblk = xpre, fpre, rpre
            spre = scur = xcur - xpre
        if rblk is None:                                            # (only under the mutations)
            return final("maxev", rcur, rpre, xcur, "no bracket")
        if abs(fblk) < abs(fcur):
            xpre, xcur, xblk = xcur, xblk, xcur
            fpre, fcur, fblk = fcur, fblk, fcur
            rpre, rcur, rblk = rcur, rblk, rcur
        sbis = (xblk - xcur) / 2
        if fcur == 0 or abs(sbis) <= delta:
            return final("tol", rcur, rblk, xcur)
        if len(recs) >= maxev:
            return final("maxev", rcur, rblk, xcur)
        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)            # secant
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))   # inverse quadratic
            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
                spre, scur = scur, stry
            else:
                spre = scur = sbis
        else:
            spre = scur = sbis
        xnew = xcur + (scur if abs(scur) > delta else (delta if sbis > 0 else -delta))
        r = ev(xnew, "interior")
        if not r["robust"]:
            return final("nonrobust", rcur, rblk, xcur, f"not robust at {xnew!r}")
        xpre, fpre, rpre = xcur, fcur, rcur
        xcur, fcur, rcur = xnew, g(r), r


# ---------------------------------------------------------------- planted selftest
def selftest():
    r, lo, hi, tol = 0.48731234, 0.485, 0.490, 1e-6
    ok = True

    def show(name, fin, recs):
        print(f"{name}: status {fin['status']} ({fin['why'] or '-'}), bracket [{fin['bracket'][0]!r}, {fin['bracket'][1]!r}] "
              f"width {fin['width']:.2e}, lam_lin {fin['lam_lin']!r}, {fin['n_eval']} evaluations", flush=True)
        for e in recs:
            print(f"    lam {e['lam']!r}: sign {e['sign']:+.0f} (planted {e['true_sign']:+.0f}), noise {e['noise']}, "
                  f"dln {e['dln']:.2e}, robust {e['robust']}", flush=True)

    def inside(fin):
        return fin["bracket"][0] <= r <= fin["bracket"][1]

    # C1 clean
    P = PlantedEval(r, 1.0)
    fin, recs = refine(P, lo, hi, tol)
    show("C1 clean", fin, recs)
    c = (fin["status"] == "tol" and inside(fin) and fin["width"] <= tol and fin["lam_lin"] is not None
         and abs(fin["lam_lin"] - r) <= 1e-6 and fin["n_eval"] <= 10
         and all(e["robust"] and e["sign"] == e["true_sign"] for e in recs))
    print(f"  C1 {'as written' if c else 'NOT as written'}", flush=True)
    ok &= c
    # C2 noisy
    P2 = PlantedEval(r, 1e-10, bias=-1e-15, zone=5e-5)
    fin, recs = refine(P2, lo, hi, tol)
    show("C2 noisy", fin, recs)
    ends = [e for e in recs if e["lam"] in fin["bracket"]]
    c = (fin["status"] in ("tol", "nonrobust") and inside(fin) and all(fin["robust"])
         and all(e["sign"] == e["true_sign"] for e in ends) and all(e["sign"] == e["true_sign"] for e in recs if e["robust"]))
    print(f"  C2 {'as written' if c else 'NOT as written'}", flush=True)
    ok &= c
    # beside C2 (information, not a criterion): d18_hybsign's sign test alone
    P2b = PlantedEval(r, 1e-10, bias=-1e-15, zone=5e-5, dln_max=math.inf)
    fin, recs = refine(P2b, lo, hi, tol)
    print(f"  (information: C2 with --dln=inf, the sign test alone: status {fin['status']}, bracket {fin['bracket']}, "
          f"contains r: {inside(fin)}, {fin['n_eval']} evaluations)", flush=True)
    # C3 wrongly signed bracket
    fin, recs = refine(P, r + 0.001, r + 0.006, tol)
    show("C3 same-signed ends", fin, recs)
    c = fin["status"] == "refused" and fin["n_eval"] == 2
    print(f"  C3 {'as written' if c else 'NOT as written'}", flush=True)
    ok &= c
    # C4 an end in the noise
    fin, recs = refine(P2, r - 0.003, r + 2e-6, tol)
    show("C4 an end in the noise", fin, recs)
    c = fin["status"] == "refused" and fin["n_eval"] == 2 and fin["why"] == "an end is not robust"
    print(f"  C4 {'as written' if c else 'NOT as written'}", flush=True)
    ok &= c
    print(f"selftest (mutate {MUTATE or 'none'}): {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------- real runs
def blob(path):
    return subprocess.run(["git", "hash-object", "--no-filters", path], capture_output=True, text=True,
                          check=True).stdout.strip()


def setup(argv, nargs):
    """the positional arguments, the options, the pins and the record system of a run or a scan"""
    args = [a for a in argv if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    if len(args) != nargs:
        print(__doc__)
        sys.exit(2)
    recf, st = os.path.abspath(args[0]), os.path.abspath(args[1])
    mm = opts.get("mm", "fast")
    if mm not in ("fast", "cs"):
        print(f"--mm={mm}: fast or cs")
        sys.exit(2)
    if "pin" in opts:
        pst, prec = opts["pin"].split(",")
        bst, brec = blob(st), blob(recf)
        print(f"pins: state {bst} (wanted {pst}), stub {brec} (wanted {prec})", flush=True)
        if not (bst.startswith(pst) and brec.startswith(prec)):
            print("a pin differs: refused", flush=True)
            sys.exit(3)
    cwd0 = os.getcwd()
    os.chdir(WTM)
    from ell_fastm import load_system
    S, z, rec, same, msg = load_system(recf, st, float(args[2]))
    print(msg, flush=True)
    if not same:
        sys.exit(3)
    E = RealEval(S, z, rec, st, mm, float(opts.get("dln", 0.1)), opts.get("basis", "jac"), opts.get("b12", "ddm"),
                 float(opts.get("eps", 1e-15)))
    return args, opts, cwd0, rec, st, E


def scancmd(argv):
    """d18_hybsign.py's scan through RealEval: one record per lambda, appended as each lands"""
    args, opts, cwd0, rec, st, E = setup(argv, 5)
    D, outf = float(args[2]), os.path.abspath(os.path.join(cwd0, args[4]))
    lams = [float(v) for v in args[3].split(",")]
    print(f"scan of {len(lams)} lambda, mode matrix {E.mm} ({E.basis}), block 12 {E.b12}, noise {E.eps:g}, "
          f"{os.path.basename(st)}", flush=True)
    for lam in lams:
        r = dict(E(lam), role="scan")
        print(f"  lam {lam!r}: hybrid {r['sign']:+.0f} ln|det| {r['ln']:.10f} (noise {E.eps:g} on blocks 0-11: "
              f"{r['noise']}); float64 {r['f64_sign']:+.0f} {r['f64_ln']:.10f} ({r['sec']} s {r['sec_parts']}); "
              f"dln {r['dln']:.2e}, mode matrix {r['mm']}", flush=True)
        open(outf, "a").write(json.dumps(r) + "\n")


def runcmd(argv):
    args = [a for a in argv if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    recf, st, outf = (os.path.abspath(a) for a in (args[0], args[1], args[5]))
    D, lo, hi = float(args[2]), float(args[3]), float(args[4])
    tol = float(opts.get("tol", 1e-6))
    maxev = int(opts.get("maxev", 12))
    mm = opts.get("mm", "fast")
    dln_max = float(opts.get("dln", 0.1))
    basis = opts.get("basis", "jac")
    b12 = opts.get("b12", "ddm")
    eps = float(opts.get("eps", 1e-15))
    if mm not in ("fast", "cs"):
        print(f"--mm={mm}: fast or cs")
        sys.exit(2)
    if "pin" in opts:
        pst, prec = opts["pin"].split(",")
        bst, brec = blob(st), blob(recf)
        print(f"pins: state {bst} (wanted {pst}), stub {brec} (wanted {prec})", flush=True)
        if not (bst.startswith(pst) and brec.startswith(prec)):
            print("a pin differs: refused", flush=True)
            sys.exit(3)
    cwd0 = os.getcwd()
    os.chdir(WTM)
    from ell_fastm import load_system
    S, z, rec, same, msg = load_system(recf, st, D)
    print(msg, flush=True)
    if not same:
        sys.exit(3)
    print(f"bracket ({lo!r}, {hi!r}), tol {tol}, maxev {maxev}, mode matrix {mm} ({basis}), block 12 {b12}, "
          f"noise {eps:g}, dln <= {dln_max}, "
          f"{os.path.basename(st)}", flush=True)
    ends = None
    if "ends" in opts:
        rows = [json.loads(l) for l in open(os.path.abspath(os.path.join(cwd0, opts["ends"]))) if l.strip()]

        def pick(lam):
            m = [w for w in rows if w.get("D") == D and w.get("kc") == rec["kc"] and abs(w["lam"] - lam) < 1e-12
                 and w.get("state") == os.path.basename(st)]
            if not m:
                print(f"--ends: no record at lambda {lam!r} of this state: refused", flush=True)
                sys.exit(2)
            w = dict(m[-1])
            w.update(dln=None, robust=bool(w["noise_robust"]))
            return w
        ends = (pick(lo), pick(hi))
    E = RealEval(S, z, rec, st, mm, dln_max, basis, b12, eps)

    def out(r):
        print(f"  lam {r['lam']!r} ({r['role']}): hybrid {r['sign']:+.0f} ln|det| {r['ln']:.10f}, noise {r['noise']} "
              f"dln {r['dln'] if r['dln'] is None else format(r['dln'], '.2e')}, robust {r['robust']}; float64 "
              f"{r['f64_sign']:+.0f} {r['f64_ln']:.10f} ({r.get('sec')} s {r.get('sec_parts', '')})", flush=True)
        open(outf, "a").write(json.dumps(r) + "\n")
    fin, recs = refine(E, lo, hi, tol, maxev, out, ends)
    fin.update(D=D, kc=rec["kc"], state=os.path.basename(st), mm=mm, b12=b12, dln_max=dln_max, eps=eps)
    open(outf, "a").write(json.dumps(fin) + "\n")
    print(f"root: status {fin['status']} ({fin['why'] or '-'}), bracket [{fin['bracket'][0]!r}, {fin['bracket'][1]!r}] "
          f"width {fin['width']:.3e}, signs {fin['signs']}, det-linear {fin['lam_lin']!r}, Brent {fin['lam_brent']!r}, "
          f"{fin['n_eval']} evaluations, {fin['sec']} s", flush=True)
    sys.exit({"tol": 0, "maxev": 4, "nonrobust": 4}.get(fin["status"], 2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        runcmd(sys.argv[2:])
    elif cmd == "scan":
        scancmd(sys.argv[2:])
    elif cmd == "selftest":
        selftest()
    else:
        print(__doc__)
        sys.exit(2)
