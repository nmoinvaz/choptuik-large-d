import Mathlib

/-!
# The inner linear constant (PROOF.md §2; proof/FORMAL.md, row 1)

With α = (1 + s)/2 and β = α + 1/2 = (2 + s)/2,

  2^(-s-1/2) / (Γ(α) Γ(β)) = 1 / (√(2π) Γ(1 + s))

for every s : ℂ. Under Mathlib's conventions Γ vanishes at its poles and z / 0 = 0, so both sides are 0 when
1 + s is a pole of Γ. The proof is Legendre's duplication formula.
-/

open Complex

namespace DSS

/-- `(2 : ℂ) ^ (-1/2) = 1 / √2`, with the square root taken in ℝ. -/
lemma two_cpow_neg_half : (2 : ℂ) ^ (-(1 / 2 : ℂ)) = 1 / ((Real.sqrt 2 : ℝ) : ℂ) := by
  have h2 : (0 : ℝ) ≤ 2 := by norm_num
  have e : (2 : ℂ) ^ (-(1 / 2 : ℂ)) = (((2 : ℝ) ^ (-(1 / 2 : ℝ)) : ℝ) : ℂ) := by
    rw [Complex.ofReal_cpow h2]
    push_cast
    ring_nf
  rw [e, Real.rpow_neg h2, ← Real.sqrt_eq_rpow]
  push_cast
  ring

theorem inner_weber_constant (s : ℂ) :
    (2 : ℂ) ^ (-s - 1 / 2) / (Gamma ((1 + s) / 2) * Gamma ((2 + s) / 2))
      = 1 / ((Real.sqrt (2 * Real.pi) : ℝ) * Gamma (1 + s)) := by
  -- Legendre: Γ(α) Γ(α + 1/2) = Γ(2α) 2^(1-2α) √π, with 2α = 1 + s and 1 - 2α = -s.
  have h := Complex.Gamma_mul_Gamma_add_half ((1 + s) / 2)
  rw [show (1 + s) / 2 + 1 / 2 = (2 + s) / 2 by ring, show 2 * ((1 + s) / 2) = 1 + s by ring,
    show (1 : ℂ) - (1 + s) = -s by ring] at h
  rw [h]
  have h2 : (2 : ℂ) ≠ 0 := two_ne_zero
  have hsplit : (2 : ℂ) ^ (-s - 1 / 2) = (2 : ℂ) ^ (-s) * (2 : ℂ) ^ (-(1 / 2 : ℂ)) := by
    rw [← Complex.cpow_add _ _ h2]
    ring_nf
  have hne : (2 : ℂ) ^ (-s) ≠ 0 := by simp [Complex.cpow_eq_zero_iff, h2]
  rw [hsplit, two_cpow_neg_half, Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2)]
  push_cast
  -- at a pole of Γ both sides are 0 (z / 0 = 0); elsewhere it is field arithmetic
  by_cases hg : Gamma (1 + s) = 0
  · simp [hg]
  have hs2 : ((Real.sqrt 2 : ℝ) : ℂ) ≠ 0 := by
    exact_mod_cast (Real.sqrt_pos.mpr (by norm_num : (0 : ℝ) < 2)).ne'
  have hsp : ((Real.sqrt Real.pi : ℝ) : ℂ) ≠ 0 := by
    exact_mod_cast (Real.sqrt_pos.mpr Real.pi_pos).ne'
  field_simp

end DSS
