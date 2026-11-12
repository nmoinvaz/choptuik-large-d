#!/usr/bin/env python3
"""Planted check of near_null_eq.curtis_reid's floor (#580).

    cd hp && python3 cr_floor_check.py

A sparse matrix with row and column scales spread over 40 orders (10^u, u uniform in [-20, 20]) gets entries
negligible in the scaled problem, 1e-45 of the equilibrated matrix's scale, at a fifth of its empty places (the
small entries of the 0.8 start's Jacobian, #504). Curtis-Reid's D_c, up to a constant, is compared with and without
those entries. (Before #613 the entries were 1e-45 below their raw row max. With column scales over 40 orders some of
those lie only about 1e-5 below the equilibrated scale, so a floor covariant under scaling must keep them; that case
is printed as information.) Pass: with --cr_floor's 1e-12 the column scales move by
less than 0.5 in log10. Control: without the floor they move by more than 2, so the check can tell the two apart.
Then the invariance (#613): a planted row and column scaling of the same matrix must leave the floored fit's
equilibrated matrix the same to 1e-8 in log10; the floor on the raw max-norms (the rule before #613) must not.
Exit 1 on failure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from near_null_eq import curtis_reid  # noqa: E402


def move(a, b):
    """the largest change of log10 a against log10 b, up to a constant"""
    d = np.log10(a) - np.log10(b)
    return float(np.abs(d - np.median(d)).max())


def main():
    rng = np.random.default_rng(1)
    n = 40
    B = rng.uniform(0.5, 2.0, (n, n)) * (rng.uniform(size=(n, n)) < 0.3)
    np.fill_diagonal(B, 1.5)
    r, c = 10.0 ** rng.uniform(-20, 20, n), 10.0 ** rng.uniform(-20, 20, n)
    A = B * r[:, None] * c[None, :]
    tiny = (B == 0) & (rng.uniform(size=(n, n)) < 0.2)
    _, dc0 = curtis_reid(A)
    # negligible in the scaled problem: 1e-45 of the equilibrated matrix's typical entry, r_i c_j 1e-45 in A
    T = A.copy()
    T[tiny] = (r[:, None] * c[None, :])[tiny] * 1e-45
    _, dc_plain = curtis_reid(T)
    _, dc_floor = curtis_reid(T, floor=1e-12)
    m_floor, m_plain = move(dc_floor, dc0), move(dc_plain, dc0)
    ok1 = m_floor < 0.5 and m_plain > 2.0
    print(f"{int(tiny.sum())} entries 1e-45 below the equilibrated scale added: D_c moves {m_plain:.2f} in log10 without "
          f"the floor (control, must exceed 2), {m_floor:.3f} with floor 1e-12 (must stay below 0.5): "
          f"{'pass' if ok1 else 'FAIL'}")
    # information, not a check: entries 1e-45 below their raw row max (the rule before #613 dropped them), which
    # in the equilibrated matrix lie between 1e-45 and about 1e-5 of the scale, so a covariant floor keeps some
    Traw = A.copy()
    Traw[tiny] = np.broadcast_to(A.max(axis=1)[:, None], A.shape)[tiny] * 1e-45
    print(f"  (information) entries 1e-45 below their raw row max instead: D_c moves "
          f"{move(curtis_reid(Traw, floor=1e-12)[1], dc0):.3f} with the covariant floor, "
          f"{move(curtis_reid(Traw, floor=1e-12, legacy_floor=True)[1], dc0):.3f} with the raw-max floor")
    # invariance (#613): a planted row and column scaling of T leaves the equilibrated matrix of the floored fit the
    # same; the rule before #613 (the floor on T's own max-norms) does not, the control
    t, sc = 10.0 ** rng.uniform(-10, 10, n), 10.0 ** rng.uniform(-10, 10, n)
    TS = T * t[:, None] * sc[None, :]

    def eq(X, legacy):
        dr, dc = curtis_reid(X, floor=1e-12, legacy_floor=legacy)
        return X * dr[:, None] * dc[None, :]
    rel = lambda X, Y: float(np.max(np.abs(np.log10(np.where(Y > 0, X / np.where(Y > 0, Y, 1), 1)))))
    d_new, d_old = rel(eq(TS, False), eq(T, False)), rel(eq(TS, True), eq(T, True))
    ok2 = d_new < 1e-8 and d_old > 1e-3
    print(f"a planted row and column scaling (10^±10): the equilibrated matrix moves {d_new:.1e} in log10 with the floor "
          f"on the equilibrated matrix (must stay below 1e-8), {d_old:.1e} with the floor on the raw max-norms "
          f"(control, must exceed 1e-3): {'pass' if ok2 else 'FAIL'}")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
