#!/usr/bin/env python3
"""The horizon series in alpha = a - 1 (Solver.set_horizon_alpha, Ctx::horizon_alpha in src/model.rs): does it lower
the large-D residual floor, and by how much?

At large D the horizon series' rounding excites the mass mode (b - 1 ~ x^-(n-1)), which grows by G = x_m^-(n-1) between
the horizon and the last node x_m; Newton's residual floor is then about 2.5 eps_dd G (eps_dd = 2^-106). In a, the
series' a^2 - 1 and a - a^3 cancel down to about alpha_h = a_h - 1 (about 1e-3 at D = 60), which loses that many digits
of every term; in alpha they do not cancel.

For each saved state (hp/data/states/up*_D<D>.npz nearest each --D; p1 harmonics above 13 frozen from D = 35, as the
scan ran; the Taylor plan of the state, tol_step 1e-32, hmax 0.05) and each mode (0 the default; 1 alpha; 2 the
control: alpha carried, the cancelling forms kept, which must reproduce mode 0 bit for bit), --its full Newton steps
with the double-double Jacobian from the saved state, without a line search, recording max|F| before each step and
after the last. The floor is the smallest |F| after the first step. Also: Delta at the end; the Delta component of
each step after the first (how far Newton moves Delta at the floor: the noise in Delta); the f64 Jacobian against the
double-double one's hi part at the saved state (max |J64 - J_hi| / max |J_hi|).

Pass criteria (written before the first run):
  - mode 2 (control): the |F| history, the final state and the f64 Jacobian are mode 0's, bit for bit;
  - mode 1: the floor at least 30 times below mode 0's at D >= 50, and at least 3 times below it at D < 50;
  - Delta: |Delta(mode 1) - Delta(mode 0)| at most 10 times the largest Delta step of mode 0 at its floor (+ 1e-30).
Exit 1 if any fails. --modes=0 runs mode 0 only (on a build without set_horizon_alpha, for the bit-identity of the
default against main: compare the two --json outputs' "hist" and "Delta" of mode 0).

    cd hp && python3 alpha_floor.py [--D=40,50,60] [--its=5] [--modes=0,1,2] [--threads=2] [--json=out]
"""
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import dd_add, load, make_solver  # noqa: E402

EPS_DD = 2.0 ** -106


def states():
    out = {}
    for p in glob.glob(os.path.join(HERE, "data", "states", "up*_D*.npz")):
        out[float(re.search(r"_D([0-9.]+)\.npz$", p).group(1))] = p
    return out


def fmax(Fh, Fl):
    return float(np.max(np.abs(np.asarray(Fh) + np.asarray(Fl))))


def run(path, mode, its, threads):
    zh, zl, m = load(path)
    kf = 13 if m["D"] >= 35 else None
    s = make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads, kfreeze=kf)
    if mode:
        s.set_horizon_alpha(mode)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    Jh, _ = s.jacobian(zh, zl)
    Jh = np.asarray(Jh)
    J64 = np.asarray(s.jacobian_f64(zh, zl))
    jdiff = float(np.max(np.abs(J64 - Jh)) / np.max(np.abs(Jh)))
    j64_sha = hashlib.sha256(np.ascontiguousarray(J64).tobytes()).hexdigest()
    zh, zl = np.array(zh), np.array(zl)
    hist, ddel = [], []
    t = time.time()
    for it in range(its):
        dh, dl, Fh, Fl, _, _ = s.newton(zh, zl)
        hist.append(fmax(Fh, Fl))
        if it >= 1:
            ddel.append(abs(float(dh[-1]) + float(dl[-1])))
        zh, zl = dd_add(zh, zl, np.asarray(dh), np.asarray(dl))
        zh, zl = np.asarray(zh), np.asarray(zl)
    Fh, Fl = s.residual(zh, zl)
    hist.append(fmax(Fh, Fl))
    n = m["D"] - 2
    xm = m["nodes"][-1]
    return dict(D=m["D"], M=m["M"], mode=mode, xm=xm, G=xm ** -(n - 1), predicted_floor=2.5 * EPS_DD * xm ** -(n - 1),
                hist=hist, floor=min(hist[1:]), Delta=[float(zh[-1]), float(zl[-1])], dDelta_steps=ddel,
                jac_f64_vs_hi=jdiff, jac_f64_sha=j64_sha, z_sha=hashlib.sha256(zh.tobytes() + zl.tobytes()).hexdigest(),
                secs=round(time.time() - t, 1))


def main():
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    Ds = [float(x) for x in opts.get("D", "40,50,60").split(",")]
    modes = [int(x) for x in opts.get("modes", "0,1,2").split(",")]
    its, threads = int(opts.get("its", 5)), int(opts.get("threads", 2))
    avail = states()
    out, bad = [], []
    for D0 in Ds:
        path = avail[min(avail, key=lambda d: abs(d - D0))]
        res = {}
        for mode in modes:
            r = run(path, mode, its, threads)
            res[mode] = r
            out.append(r)
            print(f"D {r['D']:.4f} mode {mode}: |F| " + " ".join(f"{v:.1e}" for v in r["hist"]) +
                  f"; floor {r['floor']:.2e} (2.5 eps_dd G = {r['predicted_floor']:.1e}); Delta "
                  f"{r['Delta'][0]!r} + {r['Delta'][1]:.3e}; Delta steps at the floor "
                  + " ".join(f"{v:.0e}" for v in r["dDelta_steps"]) + f"; J64 vs J_hi {r['jac_f64_vs_hi']:.2e}; "
                  f"{r['secs']} s", flush=True)
        if 0 in res and 2 in res:
            a, c = res[0], res[2]
            if not (a["hist"] == c["hist"] and a["z_sha"] == c["z_sha"] and a["jac_f64_sha"] == c["jac_f64_sha"]):
                bad.append(f"D {a['D']}: the control (mode 2) differs from mode 0")
        if 0 in res and 1 in res:
            a, b = res[0], res[1]
            gain = a["floor"] / b["floor"]
            need = 30 if a["D"] >= 50 else 3
            print(f"  floor gain {gain:.0f}x (need >= {need}x)")
            if gain < need:
                bad.append(f"D {a['D']}: floor gain {gain:.1f}x < {need}x")
            dD = abs((b["Delta"][0] - a["Delta"][0]) + (b["Delta"][1] - a["Delta"][1]))
            lim = 10 * max(a["dDelta_steps"] or [0.0]) + 1e-30
            print(f"  Delta(1) - Delta(0) = {dD:.1e} (limit {lim:.1e})")
            if dD > lim:
                bad.append(f"D {a['D']}: Delta moved by {dD:.1e} > {lim:.1e}")
    if "json" in opts:
        json.dump(out, open(opts["json"], "w"), indent=1)
    print("PASS" if not bad else "FAIL:\n  " + "\n  ".join(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
