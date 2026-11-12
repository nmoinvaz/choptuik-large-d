#!/usr/bin/env python3
"""Issue #592: the complex zeros of x - c = (1 - u) - c(u) nearest the horizon, from the horizon series of c, against
the series' radius R of notes/QUAD_DOUBLE.md 13.6.

At a state with a_h scaled by 1 + delta (hp/pole_test.py's perturbation), the kernel's horizon series of c in the
scaled variable t = u / sigma is taken lane by lane in tau. x - c = 1 - sigma t - c(t) vanishes at the horizon (c = 1
there, the record keeps 1 - c_0 per lane), so g(t) = (x - c) / t is formed and its [L/L] Pade
approximant computed (mpmath at 60 digits from the double-double coefficients) at two orders L. The zeros of the
numerator, less spurious ones, are the zeros of g: a zero is spurious when a pole of the approximant lies within
FROISSART of it relative (a Froissart doublet), or when the two orders do not both have it (to AGREE relative). The
nearest kept zero over lanes gives |u| = sigma |t| and its argument, and |u| / R is printed beside R (13.6's
estimator, pole_test.ratio_radius).

Verdict per (state, delta), fixed before the run (#592): "sonic" when |u| lies within 3% of R; otherwise where it
lies is reported. A case where the two orders disagree on the nearest zero by more than 1% is not read.

Planted (`planted`, run first): g(t) = (1 - t/t0)(1 - t/conj t0) (1 - t/3)^(-1/2), t0 = 0.8 e^(i phi), so the
nearest zeros sit at |t| = 0.8 (|u| = 0.2 at sigma = 0.25); its series through the same Pade path must give |u| to 1%,
and --mutate=move (the check against |u| 0.21) must fail the 3% test.

    cd hp && python3 pole_sonic.py planted [--mutate=move]
    cd hp && python3 pole_sonic.py OUT.jsonl STATE.npz [STATE.npz ...] [--deltas=1e-8,1e-6,1e-4,-1e-8,-1e-6,-1e-4]
        [--orders=24,32] [--lanes=all|N] [--threads N]
"""
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

mp.mp.dps = 60
FROISSART = 1e-3     # a zero with a pole this close (relative) is a doublet
AGREE = 1e-2         # the two orders must give the nearest zero to this (relative)


def pade_zeros(cs, L):
    """zeros and poles of the [L/L] Pade approximant of the series cs (mpf list, at least 2L + 1 terms)"""
    for LL in (L, L - 1, L - 2):
        try:
            p, q = mp.pade(cs[:2 * LL + 1], LL, LL)
            break
        except ZeroDivisionError:
            # an exactly rational series of lower type: its system is singular at this order
            continue
    else:
        return [], []
    zp = mp.polyroots(p[::-1], maxsteps=200, extraprec=200) if len(p) > 1 else []
    zq = mp.polyroots(q[::-1], maxsteps=200, extraprec=200) if len(q) > 1 else []
    return [complex(z) for z in zp], [complex(z) for z in zq]


def kept(zeros, poles):
    """the zeros that are not one half of a zero-pole doublet"""
    return [z for z in zeros if not any(abs(z - w) <= FROISSART * max(abs(z), 1e-30) for w in poles)]


def nearest(cs, orders):
    """the nearest genuine zero of the series cs: present at both orders to AGREE, not a doublet. Returns (t, agree)
    or (None, why)"""
    sets = []
    for L in orders:
        z, w = pade_zeros(cs, L)
        sets.append(sorted(kept(z, w), key=abs))
    if not all(sets):
        return None, "no zero kept at one order"
    a = sets[0][0]
    b = min(sets[1], key=lambda z: abs(z - a))
    d = abs(a - b) / abs(a)
    if d > AGREE:
        return None, f"orders disagree by {d:.1e}"
    return b, d


def planted(mutate=""):
    """the nearest zeros at |t| = 0.8 for three phases, through nearest()"""
    sig, ok = 0.25, True
    for phi in (0.2, 0.9, 2.0):
        t0 = 0.8 * mp.expj(phi)
        # g(t) = (1 - t/t0)(1 - t/conj t0) (1 - t/3)^(-1/2): its Taylor coefficients (a branch point at 3, so not
        # rational)
        a1, a2 = -(1 / t0 + 1 / mp.conj(t0)), 1 / (t0 * mp.conj(t0))
        num = [mp.mpf(1), a1.real, a2.real]
        b = [mp.binomial(-0.5, m) * (-mp.mpf(1) / 3) ** m for m in range(80)]
        cs = [sum(num[j] * b[k - j] for j in range(3) if k - j >= 0) for k in range(80)]
        z, d = nearest(cs, (24, 32))
        target = 0.21 if mutate == "move" else 0.2
        u = sig * abs(z) if z is not None else None
        good = u is not None and abs(u - target) <= 0.01 * target
        print(f"  planted phase {phi}: nearest |u| {u}, arg {np.angle(z) if z is not None else float('nan'):.4f} (want "
              f"{phi:.4f}), orders agree to {d}; against {target}: {'pass' if good else 'FAIL'}", flush=True)
        ok = ok and good
    return ok


def run(out, paths, deltas, orders, lanes, threads):
    from hpkernel.driver import load, make_solver
    from pole_test import ratio_radius
    with open(out, "a") as fo:
        for path in paths:
            zh, zl, meta = load(path)
            D, M, nodes = meta["D"], meta["M"], list(meta["nodes"])
            K = len(nodes)
            u_last = 1.0 - nodes[-1]
            s = make_solver(D, M, nodes=nodes, kt=meta.get("kt", 36), kc=meta.get("kc", 146), ks=meta.get("ks", 130),
                            threads=threads, kfreeze=meta.get("kfreeze", 13 if D >= 35 else None))
            s.set_segs(meta["segs"] if "segs" in meta else s.plan(zh, zl, 1e-32, 0.05))
            sig = s.series_scale(1)
            oa = 3 * M + 4 * M * K
            js = range(M) if lanes == "all" else np.linspace(0, M - 1, int(lanes)).astype(int)
            for delta in deltas:
                t0 = time.time()
                z2h, z2l = zh.copy(), zl.copy()
                z2h[oa:oa + M] *= 1.0 + delta
                z2l[oa:oa + M] *= 1.0 + delta
                hh, hl = s.series(z2h, z2l, 1, True)
                hh, hl = np.asarray(hh).reshape(-1, 4, M), np.asarray(hl).reshape(-1, 4, M)
                R, _ = ratio_radius(hh + hl, u_last, sig)
                best, per_lane = None, []
                for j in js:
                    c = [mp.mpf(float(hh[k, 3, j])) + mp.mpf(float(hl[k, 3, j])) for k in range(hh.shape[0])]
                    # x - c vanishes at the horizon itself (c = 1 at x = 1): g = t g1, and the zeros sought are g1's
                    g0 = float(1 - c[0])
                    g = [-mp.mpf(sig) - c[1]] + [-x for x in c[2:]]
                    z, d = nearest(g, orders)
                    if z is None:
                        per_lane.append(dict(lane=int(j), why=d, g0=g0))
                        continue
                    u = sig * z
                    per_lane.append(dict(lane=int(j), u=[u.real, u.imag], agree=d, g0=g0))
                    if best is None or abs(u) < abs(best[0]):
                        best = (u, int(j), d)
                if best is None:
                    v, rec = "not read: no lane gives a zero at both orders", dict()
                else:
                    u, j, d = best
                    ratio = abs(u) / R
                    v = "sonic" if abs(ratio - 1) <= 0.03 else f"not sonic: |u| / R = {ratio:.3f}"
                    rec = dict(u=[u.real, u.imag], abs_u=abs(u), arg=float(np.angle(u)), lane=j, agree=d, ratio=ratio)
                rec.update(state=os.path.basename(path), D=D, delta=delta, R=R, orders=list(orders), verdict=v,
                           lanes=per_lane)
                print(f"D {D:.4f} delta {delta:+g}: R {R:.4f}; nearest zero of x - c "
                      f"{'none' if best is None else f'|u| {abs(best[0]):.4f} arg {np.angle(best[0]):+.3f} (lane {best[1]}, orders agree {best[2]:.1e}), |u| / R {abs(best[0]) / R:.3f}'}"
                      f"; verdict {v} ({time.time() - t0:.0f} s)", flush=True)
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
    return 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    if args and args[0] == "planted":
        ok = planted(opts.get("mutate", ""))
        print("planted:", "pass" if ok else "FAIL")
        return 0 if ok else 1
    if len(args) < 2:
        sys.exit(__doc__)
    if not planted():
        print("planted: FAIL, the states are not read")
        return 1
    deltas = tuple(float(x) for x in opts.get("deltas", "1e-8,1e-6,1e-4,-1e-8,-1e-6,-1e-4").split(","))
    orders = tuple(int(x) for x in opts.get("orders", "24,32").split(","))
    return run(args[0], args[1:], deltas, orders, opts.get("lanes", "all"), int(opts.get("threads", 1)))


if __name__ == "__main__":
    sys.exit(main())
