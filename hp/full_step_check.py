#!/usr/bin/env python3
"""The pin's correction at a converged pinned state, by the one full-system Newton step that scan_hp --pin takes:
with the double-double LU (QdhSolver, the horizon block in quad-double, as scan_hp --qdh 1), and by GCR on the
residual with every block in quad-double (QdSolver), preconditioned by that LU. At the same D, two pin choices'
states must give the same full-system Delta; if they agree after the quad-double step and not after the
double-double one, the double-double step's error along the weakest direction is what limited Delta_full1 (README,
"Why the solver stalls").

    python3 full_step_check.py OUT.jsonl STATE.npz ... [--kfreeze 13] [--max_products 40] [--step_tol 1e-14]
        [--threads N]

For each state: Delta, Delta after the double-double step and after the quad-double step, the GCR's products and its
last changes, and the quad-double residual's max |F| before and after each step. OUT must be in a raw/<task> worktree
(raw_out.py; --allow_unraw writes anyway).
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from hpkernel.driver import QdhSolver, QdSolver, load, make_solver  # noqa: E402
from raw_out import FLAG, HELP, require_raw  # noqa: E402


def fmax(F):
    return float(np.abs(np.asarray(F[0]) + np.asarray(F[1])).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("states", nargs="+")
    ap.add_argument("--kfreeze", type=int, default=13)
    ap.add_argument("--max_products", type=int, default=40)
    ap.add_argument("--step_tol", type=float, default=1e-14)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    require_raw(a.out, allow=a.allow_unraw)
    for path in a.states:
        zh, zl, meta = load(path)
        D, M = meta["D"], meta["M"]
        kw = dict(nodes=list(meta["nodes"]), kt=meta.get("kt", 36), kc=meta.get("kc", 146), ks=meta.get("ks", 130),
                  threads=a.threads, kfreeze=a.kfreeze)
        rec = dict(state=path, D=D, Delta=float(zh[-1]) + float(zl[-1]), Delta_hl=[float(zh[-1]), float(zl[-1])])
        for name, wrap in (("dd", lambda s: QdhSolver(s)),
                           ("qd", lambda s: QdSolver(s, gcr=dict(max_products=a.max_products, step_tol=a.step_tol)))):
            s = make_solver(D, M, **kw)
            s.set_segs(s.plan(zh, zl, a.tol_step, a.hmax))
            q = wrap(s)
            t0 = time.time()
            F0 = fmax(QdSolver(s).residual(zh, zl))
            dh, dl, _, _, _, _ = q.newton(zh, zl)
            th, tl = q.zadd(zh, zl, np.asarray(dh), np.asarray(dl))
            R = QdSolver(s)                      # the stepped state's |F| in full quad-double, with its four parts
            R._parts[R._key(th, tl)] = q.parts(th, tl)
            F1 = fmax(R.residual(th, tl))
            rec[name] = dict(Delta=float(th[-1]) + float(tl[-1]), Delta_hl=[float(th[-1]), float(tl[-1])],
                             dDelta=float(dh[-1]) + float(dl[-1]), F_before=F0,
                             F_after=F1, seconds=round(time.time() - t0, 1))
            if name == "qd" and q.last_gcr:
                g = q.last_gcr
                rec[name].update(products=g.get("products"), converged=g.get("converged"),
                                 changes=[float(c) for c in g.get("changes", [])][-4:])
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"D {D:.4f} {os.path.basename(path)}: Delta {rec['Delta']:.17f}; after the double-double step "
              f"{rec['dd']['Delta']:.17f} (|F| {rec['dd']['F_before']:.1e} -> {rec['dd']['F_after']:.1e}); after the "
              f"quad-double step {rec['qd']['Delta']:.17f} (|F| -> {rec['qd']['F_after']:.1e}, {rec['qd'].get('products')} "
              f"products, converged {rec['qd'].get('converged')})", flush=True)


if __name__ == "__main__":
    main()
