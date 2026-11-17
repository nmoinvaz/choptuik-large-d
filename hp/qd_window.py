#!/usr/bin/env python3
"""Quad-double Delta between the double-double frontier and the chain (D 67.77-73.27), from T73's and T75's pinned
double-double states: the full system (kfreeze 13) where its basin is open, the pinned system elsewhere.

    cd hp && python3 qd_window.py OUT.jsonl STATE.npz [STATE.npz ...] --mode full|pinned [--threads 2] [--states DIR]
        [--mutate dshift]

full: newton_qd on the full system from the state, on its own plan (T84-mac/qd_fullleg.py's method, which converged
from double-double starts up to D 70.5; notes/QUAD_DOUBLE.md 12.4-12.5). pinned: newton_qd on the state's pinned
system, p1's harmonics above its pin held at the state's own values (reseed.pinned, the PinnedQ of hp/pinning.py,
T84-mac/qd_pinned.py's system). One record per state in OUT.jsonl, in qd_fullleg's fields plus mode, pin and source.
A converged state is saved in --states (default: OUT's folder) as <name>_<mode>.npz, never beside its input.
--mutate dshift solves at D + --dshift (1e-10, small enough that the full system's narrow basin still holds the
start) and must converge to a Delta moved by about dDelta/dD times the shift, far above the starts' spread (the
control).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from hpkernel.driver import load_q, make_solver, newton_qd, save_q  # noqa: E402

from raw_out import FLAG, HELP, require_raw  # noqa: E402


def solve(path, mode, threads, dshift=0.0, plan_tol=1e-32):
    z, m = load_q(path)
    D, M = float(m["D"]) + dshift, int(m["M"])
    pin = int(m.get("kfreeze", 0))
    if mode == "full":
        s = make_solver(D, M, nodes=list(m["nodes"]), kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads, kfreeze=13)
        s.set_segs(s.plan(z[0], z[1], plan_tol, 0.05))
        sq = s
        plan_steps = sum(len(g) - 1 for g in s.get_segs())
    else:
        from reseed import pinned
        _, _, sq = pinned(m, D, z, threads=threads)
        plan_steps = None
    t0 = time.time()
    zq, hist, conv = newton_qd(sq, z, tol=1e-50, it_max=10, verbose=False)
    f = hist[-1].get("F_new", hist[-1]["F"]) if hist else float("nan")
    return zq, m, dict(state=os.path.basename(path), source=path, D=D, mode=mode, pin=pin, plan_tol=plan_tol if mode == "full" else None, conv=bool(conv), F=f,
                       steps=len(hist), plan_steps=plan_steps, Delta_q=[float(p[-1]) for p in zq], seconds=round(time.time() - t0), hist=hist)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("states_in", nargs="+")
    ap.add_argument("--mode", choices=["full", "pinned"], required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--states", default=None)
    ap.add_argument("--mutate", choices=["dshift"], default=None)
    ap.add_argument("--dshift", type=float, default=1e-10)
    ap.add_argument("--plan_tol", type=float, default=1e-32, help="full mode: the Taylor plan's step tolerance; a finer "
                    "one (1e-34) measures the plan's share of the error")
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    sdir = a.states or os.path.dirname(os.path.abspath(a.out))
    require_raw(a.out, os.path.join(sdir, "x"), allow=a.allow_unraw)
    os.makedirs(sdir, exist_ok=True)
    for path in a.states_in:
        zq, m, rec = solve(path, a.mode, a.threads, a.dshift if a.mutate == "dshift" else 0.0, a.plan_tol)
        rec["mutate"] = a.mutate
        print(f"{rec['state']}: D {rec['D']}: {a.mode}, {'converged' if rec['conv'] else 'NOT converged'}, |F| "
              f"{rec['F']:.2e}, {rec['steps']} steps, {rec['seconds']} s, Delta {zq[0][-1]:.17f} {zq[1][-1]:+.6e} "
              f"{zq[2][-1]:+.3e}", flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if rec["conv"] and not a.mutate:
            tag = "" if a.plan_tol == 1e-32 else f"_p{a.plan_tol:g}"
            save_q(os.path.join(sdir, rec["state"][:-4] + f"_{a.mode}{tag}.npz"), zq,
                   dict(m, solver="qd", kfreeze=13 if a.mode == "full" else m.get("kfreeze")))


if __name__ == "__main__":
    sys.exit(main())
