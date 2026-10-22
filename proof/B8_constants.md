# Lemma B8 with constants: the global continuation of the outer solution

*Working notes, in progress. Scripts: `solver/proof/b8_*.py`.*

## Design (to be written out in full)

- Variables (M = 1): V = (P_M[cΔ] − xS)/x (slow), S = P + M (fast), β = (n−1)(b−1) (fast, b-mode), c (slow).
  Exact identity x V' = ω ∂_θ S for every M (b8_system.py).
- Remainder R = Z − Z^[K]; bootstrap (continuity) argument instead of a contraction: the solution exists by ODE theory
  as long as the a priori bounds hold, and the bounds improve.
- S: integration by parts against the fast kernel for the terms without R_β (bounded kernel, gain from the algebraic
  slaving); the R_β terms by kernel decay (gain γ_S ~ √ρ at large |x|).
- β: decoupled variable R̂_β = R_β − q₊R_S₊ − q₋R_S₋ removes the O(1) fast forcing by S; then integration by parts with
  the kernel (s/x)^{n−1}, |·| ≤ 1 on paths with |s| ≤ |x|.
- Slow integrals on the low path Q_x (radial, then an arc of angle ≤ φ_δ = arcsin(δ/2) behind the light-cone disc).
- Regions: N (|x| ≤ x_F) and F (|x| ≥ x_F); in F, ρ ≤ ρ_max(n) = (1 + R_n)²/n, ℓ = X²/|x² − 1| ≤ X_F/(x_F − 1).
