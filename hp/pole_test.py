#!/usr/bin/env python3
"""Route 1, test T5 (notes/QUAD_DOUBLE.md 13.4): where the horizon series' nearest singularity sits for horizon data
off the solution, from the series alone. At each stored state (a double-double solution of the series block) the
horizon datum a_h is scaled by 1 + delta (the mass direction), the state otherwise held, and the horizon series is
recomputed. From its coefficients c_k (fields p, m, a, c at the M tau points):
  1. the radius R by the late-order ratio test of horizon_radius.py (T84-mac): t_k = max |c_k| u0^k over fields and
     tau points, r = (t_K / t_{K-40})^(1/40), R = u0 / r;
  2. a Domb-Sykes test on the tau-mean of the a field's coefficients a_k: rho_k = a_k / a_{k-1} fitted as A + B/k over
     the last 60 orders. The nearest singularity is then at u = 1/A (A > 0: on the positive real axis, as H1's pole;
     A < 0: on the negative axis), with exponent alpha = 1 + B/A (1 for a simple pole). Ratios whose signs are mixed
     mean a complex pair, and the fit is not read.
H1 predicts R = R0 + C ln(1/delta) / (n - 1) with one C for all D; the fit per state is printed. Control: the
perturbation of horizon_radius.py (D moved by 1e-4 with the state held) is repeated, and its R must reproduce
T84-mac/horizon_radius.jsonl (passed with --control; a mismatch above 2% is reported and stops the reading).

    cd hp && python3 pole_test.py OUT.jsonl STATE.npz [STATE.npz ...] [--threads N] [--control horizon_radius.jsonl]
"""
import json
import math
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from hpkernel.driver import load, make_solver  # noqa: E402

DELTAS = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)
NDS = 60                      # orders in the Domb-Sykes fit
NR = 40                       # orders in the ratio test


def ratio_radius(c, u0, sig):
    """horizon_radius.py's estimator from the scaled coefficients c (orders, 4, M)"""
    nk = c.shape[0]
    t = np.abs(c).max(axis=(1, 2)) * (u0 / sig) ** np.arange(nk)
    fin = np.isfinite(t) & (t > 0)
    if fin.sum() < NR + 1:
        return None, None
    K = np.flatnonzero(fin)[-1]
    r = (t[K] / t[K - NR]) ** (1.0 / NR)
    return float(u0 / r), float(t[K])


def domb_sykes(c, sig):
    """(A, B, alpha, fraction of negative ratios, orders used): rho_k = a_k/a_{k-1} = A + B/k on the tau-mean of a"""
    a = c[:, 2, :].mean(axis=1)                       # scaled: a_k sigma^k
    nk = len(a)
    ks = np.arange(1, nk)
    rho = (a[1:] / a[:-1]) / sig                      # a_k / a_{k-1}
    ok = np.isfinite(rho) & (np.abs(a[1:]) > 0) & (np.abs(a[:-1]) > 0)
    ks, rho = ks[ok], rho[ok]
    if len(ks) < 10:
        return None
    ks, rho = ks[-NDS:], rho[-NDS:]
    neg = float(np.mean(rho < 0))
    B, A = np.polyfit(1.0 / ks, rho, 1)              # rho = A + B (1/k)
    alpha = 1.0 + B / A if A != 0 else None
    return dict(A=float(A), B=float(B), alpha=alpha, R_ds=float(1.0 / A) if A else None, neg=neg, orders=[int(ks[0]), int(ks[-1])])


def main():
    args = list(sys.argv[1:])
    threads, control = 2, None
    if "--threads" in args:
        i = args.index("--threads"); threads = int(args[i + 1]); del args[i:i + 2]
    if "--control" in args:
        i = args.index("--control"); control = args[i + 1]; del args[i:i + 2]
    out, paths = args[0], args[1:]
    ctl = {}
    if control:
        for line in open(control):
            if line.strip():
                r = json.loads(line)
                ctl[r["state"]] = r
    print(f"{'D':>8} {'delta':>7} {'R':>7} {'R_ds':>8} {'alpha':>7} {'neg':>5} {'t_K':>9}   (R: ratio test; R_ds, alpha: Domb-Sykes on the a field's tau-mean)", flush=True)
    fails = []
    lines = []
    with open(out, "a") as f:
        for path in paths:
            zh, zl, m = load(path)
            D, M, nodes = m["D"], m["M"], list(m["nodes"])
            kc, ks, kt = m.get("kc", 146), m.get("ks", 130), m.get("kt", 36)
            K = len(nodes)
            u0 = 1.0 - nodes[-1]
            n = D - 2
            kf = 13 if D >= 35 else None
            s = make_solver(D, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=threads, kfreeze=kf)
            s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
            segs = s.get_segs()
            sig = s.series_scale(1)
            oa = 3 * M + 4 * M * K                      # a_h: the second horizon field, after the K nodes
            rows = []
            for delta in DELTAS:
                zh2, zl2 = zh.copy(), zl.copy()
                zh2[oa:oa + M] *= 1.0 + delta
                zl2[oa:oa + M] *= 1.0 + delta
                hh, hl = s.series(zh2, zl2, 1, True)
                c = (np.asarray(hh) + np.asarray(hl)).reshape(-1, 4, M)
                R, tK = ratio_radius(c, u0, sig)
                ds = domb_sykes(c, sig)
                rec = dict(state=os.path.basename(path), D=D, n=n, u0=u0, ks=ks, delta=delta, R=R, tK=tK, ds=ds)
                rows.append(rec)
                f.write(json.dumps(rec) + "\n"); f.flush()
                a = ds["alpha"] if ds and ds["alpha"] is not None else float("nan")
                print(f"{D:8.4f} {delta:7.0e} {R if R else float('nan'):7.3f} {ds['R_ds'] if ds and ds['R_ds'] else float('nan'):8.3f} "
                      f"{a:7.2f} {ds['neg'] if ds else float('nan'):5.2f} {tK if tK else float('nan'):9.1e}", flush=True)
            # the control: horizon_radius.py's perturbation, D moved by 1e-4 with the state held
            s2 = make_solver(D + 1e-4, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=threads, kfreeze=kf)
            s2.set_segs(segs)
            hh, hl = s2.series(zh, zl, 1, True)
            c = (np.asarray(hh) + np.asarray(hl)).reshape(-1, 4, M)
            Rc, _ = ratio_radius(c, u0, s2.series_scale(1))
            rc = ctl.get(os.path.basename(path), {}).get("R", {}) if ctl else {}
            ref = next((v for k, v in rc.items() if abs(float(k) - 1e-4) < 1e-12), None)
            line = f"{D:8.4f} control dD 1e-4: R {Rc if Rc else float('nan'):.3f}"
            if ref is not None:
                ok = Rc is not None and abs(Rc - ref) <= 0.02 * ref
                line += f" against horizon_radius.jsonl {ref:.3f}: {'PASS' if ok else 'FAIL'}"
                if not ok:
                    fails.append((D, Rc, ref))
            print(line, flush=True)
            # H1's line: R against ln(1/delta) over delta > 0
            pts = [(math.log(1.0 / r["delta"]), r["R"]) for r in rows if r["delta"] > 0 and r["R"] is not None]
            if len(pts) >= 3:
                x, y = zip(*pts)
                b, a0 = np.polyfit(x, y, 1)
                rec = dict(state=os.path.basename(path), D=D, n=n, fit="R = a0 + b ln(1/delta)", a0=float(a0), b=float(b),
                           C=float(b * (n - 1)), resid=float(np.max(np.abs(np.polyval([b, a0], x) - np.array(y)))))
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"{D:8.4f} H1 line: R = {a0:.3f} + {b:.4f} ln(1/delta), C = b (n - 1) = {rec['C']:.2f}, worst residual {rec['resid']:.3f}", flush=True)
                # the series' basin on H1's line: it converges at the node (R > u0) for delta below delta_max(u0)
                dmax = {u: math.exp(-(u - a0) / b) for u in (0.3, 0.2, 0.15)}
                rec2 = dict(state=os.path.basename(path), D=D, n=n, delta_max=dmax)
                f.write(json.dumps(rec2) + "\n"); f.flush()
                print(f"{D:8.4f} delta_max for u0 = 0.30, 0.20, 0.15: " + ", ".join(f"{dmax[u]:.1e}" for u in (0.3, 0.2, 0.15)), flush=True)
                lines.append((D, n, a0, b))
        if lines:
            # extrapolation to D = 200 from the last state's line, with C fixed and R0 scaled as 1/(n - 1)
            D1, n1, a1, b1 = lines[-1]
            C = b1 * (n1 - 1)
            n2 = 198.0
            a2, b2 = a1 * (n1 - 1) / (n2 - 1), C / (n2 - 1)
            dmax = {u: math.exp(-(u - a2) / b2) for u in (0.3, 0.2, 0.15)}
            rec = dict(extrapolation="D = 200 from the last state's line", from_D=D1, C=C, a0=a2, b=b2, delta_max=dmax)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"extrapolated to D = 200 (C = {C:.2f} from D {D1:.1f}, R0 scaled as 1/(n - 1)): delta_max for u0 = 0.30, 0.20, 0.15: "
                  + ", ".join(f"{dmax[u]:.1e}" for u in (0.3, 0.2, 0.15)), flush=True)
    if fails:
        print("control FAILED at", fails)
        sys.exit(1)


if __name__ == "__main__":
    main()
