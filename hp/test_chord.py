#!/usr/bin/env python3
"""The chord phase of lowD_dd.solve_point (notes/LOW_D_SPEEDUP.md, step 3) on a small case: the converged M = 48
state at D 4 (hp/data/lowD_states/lowD_D4.0000_M48.npz) perturbed as highm_check.py's converge check does (Delta +
1e-6, every unknown + 1e-8 N(0, 1), seed 1) and solved at D 4 again, three ways through solve_point:
(1) Newton only (the former rule); (2) with chord steps from max |F| < 1e-8; (3) the same with the chord step's sign
flipped (chord_mutate 1: the guard must reject every trial, and Newton finishes).
Pass: all three converge to the same Delta within 1e-26; (2) builds fewer Jacobians than (1), counted on the solver
(the final refresh for the sensitivities included); (3) rejected its trials.

    python3 test_chord.py [--threads 1]          exit 0 when every case passes
    python3 test_chord.py --mutate               the guard removed (chord_mutate 2): exit 1 is the expected result

Seconds at one thread."""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from hpkernel.driver import load  # noqa: E402
import lowD_dd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class Counting:
    """the solver, with its Jacobian builds counted (newton_mixed and newton; chord and residual are free)"""
    def __init__(self, s):
        self.s, self.jacobians = s, 0

    def __getattr__(self, name):
        return getattr(self.s, name)

    def newton_mixed(self, *a, **k):
        self.jacobians += 1
        return self.s.newton_mixed(*a, **k)

    def newton(self, *a, **k):
        self.jacobians += 1
        return self.s.newton(*a, **k)


MADE = []
_make = lowD_dd.make_solver


def _counting_make(*a, **k):
    MADE.append(Counting(_make(*a, **k)))
    return MADE[-1]


lowD_dd.make_solver = _counting_make


def run(zh, zl, meta, D, outdir, threads, chord_from, chord_mutate):
    a = argparse.Namespace(out=os.path.join(outdir, "points.jsonl"), tag=f"t{chord_mutate}_{chord_from:g}", resume=False,
                           kt=meta["kt"], kcs=meta["kc"], ks=meta["ks"], tol_step=1e-32, hmax=0.05, solver="newton",
                           it_max=20, tol=1e-28, accept=1e-27, lm=0, jac="f64", threads=threads, svd=0, dn_its=2, w=24,
                           stop_lam=0.0, floor_ratio=0, chord_from=chord_from, chord_mutate=chord_mutate)
    lines = []
    n0 = len(MADE)
    ok, h, l = lowD_dd.solve_point(D, meta["M"], zh.copy(), zl.copy(), "test", a, meta["nodes"], outdir, lines.append)
    chords = [ln for ln in lines if ln.lstrip().startswith("chord")]
    return ok, float(h[-1]) + float(l[-1]), sum(c.jacobians for c in MADE[n0:]), chords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args()
    zh, zl, meta = load(os.path.join(HERE, "data", "lowD_states", "lowD_D4.0000_M48.npz"))
    zh = zh + 1e-8 * np.random.default_rng(1).standard_normal(len(zh))
    zh[-1] += 1e-6
    fails = []
    with tempfile.TemporaryDirectory() as d:
        ok1, D1, n1, _ = run(zh, zl, meta, 4.0, d, a.threads, 0.0, 0)
        ok2, D2, n2, c2 = run(zh, zl, meta, 4.0, d, a.threads, 1e-8, 2 if a.mutate else 0)
        ok3, D3, n3, c3 = run(zh, zl, meta, 4.0, d, a.threads, 1e-8, 1)
    print(f"(1) Newton only: converged {ok1}, Delta {D1:.17f}, {n1} Jacobians")
    print(f"(2) chord from 1e-8{' (guard removed)' if a.mutate else ''}: converged {ok2}, Delta {D2:.17f}, {n2} Jacobians, "
          f"{len(c2)} chord lines")
    print(f"(3) sign-flipped chord: converged {ok3}, Delta {D3:.17f}, {n3} Jacobians, {sum('rejected' in c for c in c3)} rejected")
    if not (ok1 and ok2 and ok3):
        fails.append("a case did not converge")
    if ok1 and ok2 and not abs(D1 - D2) <= 1e-26 * abs(D1):
        fails.append(f"(2) differs from (1) by {abs(D1 - D2):.1e}")
    if ok1 and ok3 and not abs(D1 - D3) <= 1e-26 * abs(D1):
        fails.append(f"(3) differs from (1) by {abs(D1 - D3):.1e}")
    if not n2 < n1:
        fails.append(f"(2) built {n2} Jacobians, not fewer than (1)'s {n1}")
    if not any("rejected" in c for c in c3):
        fails.append("(3) rejected no trial")
    print("; ".join(fails) if fails else "all cases pass")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
