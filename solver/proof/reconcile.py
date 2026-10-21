#!/usr/bin/env python3
"""Reconcile the truncated (M harmonics, hier.py) and untruncated (amplitude expansion) selections of omega/omega_inf
(gap (b), harmonic tail). Pure arithmetic on the converged numbers quoted in the README.

The truncated zeros c_M alternate about the limit L with a ratio r per added pair of harmonics: c_M - L = d_M,
d_{M+2} = r d_M. Given c_3 and c_5, each r fixes L = c_5 + (c_5 - c_3) r / (1 - r) and predicts c_7 = L + r (c_5 - L).
"""
c1, c3, c5 = 0.7768, 1.0156, 0.999228          # M = 1, 3, 5 zeros (omega/omega_inf); M = 1: 2.969777 / 3.82213589
c1 = 2.969777 / 3.82213589
Lb = (0.99915, 1.00011)                         # untruncated bracket at order 12 (README)
Lpade = 0.99989                                 # central 12-coefficient Pade
dr_dw = 6.7e3                                   # |d r_1 / d(omega/omega_inf)| at the M = 5 point (fixed-A runs at 0.95, 1.05)
print(f"c_1 = {c1:.6f}, c_3 = {c3}, c_5 = {c5}")
print(f"observed (c5 - c3)/(c3 - c1) = {(c5 - c3) / (c3 - c1):+.4f}   (M = 1 is pre-asymptotic)")
for L in (Lb[0], Lpade, 1.0, Lb[1]):
    r = (c5 - L) / (c3 - L)
    print(f"L = {L:.5f}: d_3 = {c3 - L:+.6f}, d_5 = {c5 - L:+.6f}, ratio d_5/d_3 = {r:+.4f};  M > 5 tail in omega {L - c5:+.6f}"
          f" ~ |r_1| {abs(L - c5) * dr_dw:.2f}")
print("\nlimit and M = 7 prediction for a per-pair ratio r (alternating):")
for r in (-0.07, -0.0574, -0.0495, -0.045, -0.035):
    L = c5 + (c5 - c3) * r / (1 - r)
    c7 = L + r * (c5 - L)
    print(f"  r = {r:+.4f}: L = {L:.6f}, predicted c_7 = {c7:.6f}, M > 7 tail in omega {L - c7:+.2e} ~ |r_1| {abs(L - c7) * dr_dw:.3f}")
