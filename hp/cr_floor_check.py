#!/usr/bin/env python3
"""Planted check of near_null_eq.curtis_reid's floor (#580).

    cd hp && python3 cr_floor_check.py

A sparse matrix with row and column scales spread over 40 orders (10^u, u uniform in [-20, 20]) gets entries 1e-45
below their row's max-norm at a fifth of its empty places, as the 0.8 start's Jacobian has (#504). Curtis-Reid's D_c,
up to a constant, is compared with and without those entries. Pass: with --cr_floor's 1e-12 the column scales move by
less than 0.5 in log10. Control: without the floor they move by more than 2, so the check can tell the two apart.
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
    T = A.copy()
    tiny = (B == 0) & (rng.uniform(size=(n, n)) < 0.2)
    T[tiny] = np.broadcast_to(A.max(axis=1)[:, None], A.shape)[tiny] * 1e-45
    _, dc0 = curtis_reid(A)
    _, dc_plain = curtis_reid(T)
    _, dc_floor = curtis_reid(T, floor=1e-12)
    m_floor, m_plain = move(dc_floor, dc0), move(dc_plain, dc0)
    ok = m_floor < 0.5 and m_plain > 2.0
    print(f"{int(tiny.sum())} entries 1e-45 below their row max added: D_c moves {m_plain:.2f} in log10 without the floor "
          f"(control, must exceed 2), {m_floor:.3f} with floor 1e-12 (must stay below 0.5): {'pass' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
