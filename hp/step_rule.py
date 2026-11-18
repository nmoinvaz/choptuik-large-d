#!/usr/bin/env python3
"""The ANM step tied to the re-seed (anm_run.py --a_rule reseed). Pure Python, no kernel.

With --reseed_start pred the re-seed polishes the prediction's own horizon data in quad-double (reseed.polish_q) and
accepts when that Newton converges below 1e-24. Its start is off by the prediction's error in the a_h rows, so the
re-seed accepts while that error lies inside the horizon series' basin. 13.6's pole law (notes/QUAD_DOUBLE.md 13.6,
checked at the chain's front in notes/HORIZON_RADIUS.md 3) puts the pole of data off by delta at
R = (C ln(1/delta) - R0) / (n - 1), so the basin at the matching distance u0 = 1 - x_last is

    delta_pole(D, u0) = exp(-(u0 (n - 1) + R0) / C),     n = D - 2, C = 0.88, R0 = 2.6.

The series' truncation error in those rows at the step a is about e_(N+1) a^(N+1), with e_k = max |z_k| over the a_h
rows relative to max |z_0| there, and e_(N+1) = e_N / rho from the geometric-mean ratio rho of the last four orders.
The ratios grow with the order on the chain (about k / 3 at orders 12-24, D 212-233), so this estimate is high by
about (N + 1) / (N - 1.5). The step is the a with

    e_(N+1) a^(N+1) = s delta_pole(D + a, u0),

capped by h_max, a_max and the room to D_max. The level s is learned from the re-seed's outcomes: a re-seed that fails
at a sets s to the level of a divided by drop (the next attempt is the a this gives), and a first attempt accepted at
the rule's own step raises s by rise. On #903 the re-seed's edge lies at s = 1.2-1.9 on leg 6 (ks 230) and 0.067-0.076
on leg 7 (ks 240), so the law's D-dependence is used and its level is learned (step_rule_check.py offline).
"""
import math

import numpy as np

# 13.6's line (notes/QUAD_DOUBLE.md 13.6): R = (POLE_C ln(1/delta) - POLE_R0) / (n - 1)
POLE_C, POLE_R0 = 0.88, 2.6
# the orders the tail's ratio is read over
TAIL_W = 4


def pole_delta(D, u0):
    """the a_h error (relative) whose series pole lies at u0, by 13.6's line; n - 1 = D - 3"""
    return math.exp(-(u0 * (float(D) - 3.0) + POLE_R0) / POLE_C)


def ah_norms(Z, M, K):
    """e_k = max |z_k| over the a_h rows / max |z_0| there, from the leading parts Z[k] of the series (K nodes)"""
    lo = 3 * M + 4 * M * K
    sl = slice(lo, lo + M)
    e0 = float(np.max(np.abs(np.asarray(Z[0])[sl])))
    return [float(np.max(np.abs(np.asarray(z)[sl]))) / e0 for z in Z]


def tail_next(e, w=TAIL_W):
    """(e_next, rho): rho = (e_(N-w) / e_N)^(1/w), e_next = e_N / rho, the window w shortened to N below it; (0, inf)
    for a zero tail or N = 0"""
    N = len(e) - 1
    w = min(w, N)
    if w < 1 or not (e[N] > 0 and e[N - w] > 0):
        return 0.0, math.inf
    rho = (e[N - w] / e[N]) ** (1.0 / w)
    return e[N] / rho, rho


class ReseedRule:
    """--a_rule reseed. u0: the re-seed's matching distance; scale: the first level s; drop, rise: its changes."""

    def __init__(self, u0, scale=1.0, drop=10.0, rise=10.0 ** 0.5):
        self.u0, self.scale, self.drop, self.rise = float(u0), float(scale), float(drop), float(rise)
        self.e_next, self.rho, self.N, self.D0, self.capped = 0.0, math.inf, 0, 0.0, True

    def err(self, a):
        """the estimated a_h error of the prediction at a"""
        return self.e_next * a ** (self.N + 1)

    def level(self, a):
        """err(a) in units of delta_pole at D0 + a"""
        return self.err(a) / pole_delta(self.D0 + a, self.u0)

    def solve(self, s, cap):
        """the a <= cap with err(a) = s delta_pole(D0 + a): a fixed point, contracting by about u0 / (C (N + 1))"""
        if not self.e_next > 0:
            return cap
        a = cap
        for _ in range(8):
            a = min(cap, (s * pole_delta(self.D0 + a, self.u0) / self.e_next) ** (1.0 / (self.N + 1)))
        return a

    def start(self, e, D0, cap):
        """the first attempt's a at a base: e the series' a_h norms e_0..e_N, D0 the base's D, cap the other limits"""
        self.N, self.D0 = len(e) - 1, float(D0)
        self.e_next, self.rho = tail_next(e)
        a = self.solve(self.scale, cap)
        self.capped = not self.e_next > 0 or (self.scale * pole_delta(self.D0 + cap, self.u0) / self.e_next) ** (
            1.0 / (self.N + 1)) >= cap
        return a

    def refused(self, a):
        """a re-seed failure at a: the level drops below a's by drop; the next attempt's a, at most a. With a zero tail
        the estimate says nothing: the level is kept and the step halves."""
        self.capped = False
        if not self.e_next > 0:
            return 0.5 * a
        self.scale = min(self.scale, self.level(a) / self.drop)
        return self.solve(self.scale, a)

    def accepted(self, first):
        """an accepted attempt; first: at the base's first attempt, at the rule's step unscreened and re-seeded. The
        level rises after such an attempt at the rule's own (uncapped) step."""
        if first and not self.capped:
            self.scale *= self.rise

    def record(self):
        return dict(e_next=self.e_next, rho=self.rho, scale=self.scale, u0=self.u0, drop=self.drop, rise=self.rise,
                    capped=bool(self.capped))
