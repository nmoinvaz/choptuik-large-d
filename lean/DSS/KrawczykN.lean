import DSS.Krawczyk

/-!
# The Krawczyk test in n unknowns, on enclosures of the preconditioned products (proof/FORMAL.md, rows 17 and 24)

The M-harmonic certificates (proof/A_lemmas_M.md, `certify_M.py`) enclose Y g(x̂) and Y Dg over the box directly, from
Lohner sets, which is tighter than a product of interval matrices. So the hypotheses here are enclosures of those
products. On the unit box B = {u : ‖u‖∞ ≤ 1} the exact test

  max |(Y f(0))_i| + Σ_j max |δ_ij - (Y Df)_ij| < 1   for every i

gives exactly one zero of f in B, with Df invertible there. A box with half-widths w_i reduces to it by u = W⁻¹(x - x̂)
(`Box` below), and the reduction only rescales the enclosures by the exact ratios w_j / w_i.
-/

open Metric Set Matrix

namespace DSS.Krawczyk

variable {n : ℕ}

/-- ℝⁿ with Mathlib's sup norm. -/
abbrev Vn (n : ℕ) := Fin n → ℝ
abbrev Mn (n : ℕ) := Matrix (Fin n) (Fin n) ℝ

/-- One test on the unit box: enclosures of Y f(0) and of Y Df over the box, and the matrix Y. -/
structure DataN (n : ℕ) where
  Yg : Fin n → I
  YJ : Fin n → Fin n → I
  Y : Fin n → Fin n → ℚ

namespace DataN

variable (d : DataN n)

def Yr : Mn n := Matrix.of fun i j => (d.Y i j : ℝ)

/-- M_ij, the largest modulus of δ_ij - (Y Df)_ij. -/
def Mij (i j : Fin n) : ℚ := ((I.const (if i = j then 1 else 0)).sub (d.YJ i j)).mag

/-- The row sum Σ_j M_ij. -/
def C (i : Fin n) : ℚ := ∑ j, d.Mij i j

/-- The test in exact rational arithmetic. -/
def check : Bool := decide (∀ i, (d.Yg i).mag + d.C i < 1)

/-- The hypothesis certified outside Lean, on the unit box. -/
def EnclOK (f : Vn n → Vn n) (Df : Vn n → Mn n) : Prop :=
  (∀ i, (d.Yg i).mem ((d.Yr *ᵥ f 0) i)) ∧ ∀ u ∈ closedBall (0 : Vn n) 1, ∀ i j, (d.YJ i j).mem ((d.Yr * Df u) i j)

lemma C_nonneg (i : Fin n) : (0 : ℝ) ≤ (d.C i : ℝ) := by
  simp only [C, Rat.cast_sum]
  exact Finset.sum_nonneg fun j _ => I.mag_nonneg _

lemma entry_mem {A : Mn n} (hA : ∀ i j, (d.YJ i j).mem ((d.Yr * A) i j)) (i j : Fin n) :
    ((I.const (if i = j then 1 else 0)).sub (d.YJ i j)).mem ((1 - d.Yr * A) i j) := by
  have e : (1 - d.Yr * A) i j = ((if i = j then 1 else 0 : ℚ) : ℝ) - (d.Yr * A) i j := by
    simp only [Matrix.sub_apply, Matrix.one_apply]
    split_ifs <;> simp
  rw [e]
  exact I.mem_sub (I.mem_const _) (hA i j)

/-- The row bound: |((I - Y A) v)_i| ≤ C_i ‖v‖ for every A whose product with Y lies in the enclosure. -/
lemma row_bound {A : Mn n} (hA : ∀ i j, (d.YJ i j).mem ((d.Yr * A) i j)) (v : Vn n) (i : Fin n) :
    |((1 - d.Yr * A) *ᵥ v) i| ≤ (d.C i : ℝ) * ‖v‖ := by
  have hn : 0 ≤ ‖v‖ := norm_nonneg v
  calc |((1 - d.Yr * A) *ᵥ v) i| = |∑ j, (1 - d.Yr * A) i j * v j| := by
        simp [Matrix.mulVec, dotProduct]
    _ ≤ ∑ j, |(1 - d.Yr * A) i j * v j| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j, (d.Mij i j : ℝ) * ‖v‖ := by
        refine Finset.sum_le_sum fun j _ => ?_
        rw [abs_mul]
        have hv : |v j| ≤ ‖v‖ := by simpa [Real.norm_eq_abs] using norm_le_pi_norm v j
        exact mul_le_mul (I.abs_le_mag (d.entry_mem hA i j)) hv (abs_nonneg _) (I.mag_nonneg _)
    _ = (d.C i : ℝ) * ‖v‖ := by simp only [C, Rat.cast_sum, Finset.sum_mul]

lemma check_spec (hc : d.check = true) : ∀ i, ((d.Yg i).mag : ℝ) + d.C i < 1 := by
  simp only [check, decide_eq_true_eq] at hc
  intro i
  exact_mod_cast hc i

/-- With every row sum below 1, Y A is invertible. -/
lemma det_ne_zero [NeZero n] {A : Mn n} (hA : ∀ i j, (d.YJ i j).mem ((d.Yr * A) i j)) (hC : ∀ i, (d.C i : ℝ) < 1) :
    (d.Yr * A).det ≠ 0 := by
  intro h0
  obtain ⟨v, hv, hYA⟩ := Matrix.exists_mulVec_eq_zero_iff.mpr h0
  set q : ℝ := Finset.univ.sup' Finset.univ_nonempty fun i => (d.C i : ℝ)
  have hCq : ∀ i, (d.C i : ℝ) ≤ q := fun i => Finset.le_sup' (fun i => (d.C i : ℝ)) (Finset.mem_univ i)
  have hq : q < 1 := (Finset.sup'_lt_iff _).mpr fun i _ => hC i
  have hq0 : 0 ≤ q := (d.C_nonneg ⟨0, Nat.pos_of_ne_zero (NeZero.ne n)⟩).trans (hCq _)
  have hfix : (1 - d.Yr * A) *ᵥ v = v := by
    rw [Matrix.sub_mulVec, Matrix.one_mulVec, hYA, sub_zero]
  have hle : ‖v‖ ≤ q * ‖v‖ := by
    refine (pi_norm_le_iff_of_nonneg (mul_nonneg hq0 (norm_nonneg v))).mpr fun i => ?_
    have := d.row_bound hA v i
    rw [hfix] at this
    rw [Real.norm_eq_abs]
    exact this.trans (mul_le_mul_of_nonneg_right (hCq i) (norm_nonneg v))
  have hn : ‖v‖ = 0 := by
    have := norm_nonneg v
    nlinarith
  exact hv (norm_eq_zero.mp hn)

end DataN

/-- Krawczyk's test on the unit box in n unknowns, in contraction form. -/
theorem krawczykN_of_check [NeZero n] (d : DataN n) (hc : d.check = true) (f : Vn n → Vn n) (Df : Vn n → Mn n)
    (hf : ∀ u ∈ closedBall (0 : Vn n) 1,
      HasFDerivWithinAt f (LinearMap.toContinuousLinearMap (Matrix.toLin' (Df u))) (closedBall 0 1) u)
    (hE : d.EnclOK f Df) :
    (∃! u, u ∈ closedBall (0 : Vn n) 1 ∧ f u = 0) ∧
      ∀ u ∈ closedBall (0 : Vn n) 1, f u = 0 → (Df u).det ≠ 0 := by
  have hcheck := d.check_spec hc
  obtain ⟨hgx, hJ⟩ := hE
  set B := closedBall (0 : Vn n) 1 with hBdef
  have hC : ∀ i, (d.C i : ℝ) < 1 := fun i => by
    have := hcheck i
    have := I.mag_nonneg (d.Yg i)
    linarith
  set q : ℝ := Finset.univ.sup' Finset.univ_nonempty fun i => (d.C i : ℝ)
  have hCq : ∀ i, (d.C i : ℝ) ≤ q := fun i => Finset.le_sup' (fun i => (d.C i : ℝ)) (Finset.mem_univ i)
  have hq1 : q < 1 := (Finset.sup'_lt_iff _).mpr fun i _ => hC i
  have hq0 : 0 ≤ q := (d.C_nonneg ⟨0, Nat.pos_of_ne_zero (NeZero.ne n)⟩).trans (hCq _)
  let Ycl : Vn n →L[ℝ] Vn n := LinearMap.toContinuousLinearMap (Matrix.toLin' d.Yr)
  let G : Vn n → Vn n := fun u => u - d.Yr *ᵥ f u
  have hG : ∀ u ∈ B, HasFDerivWithinAt G
      (ContinuousLinearMap.id ℝ (Vn n) - Ycl.comp (LinearMap.toContinuousLinearMap (Matrix.toLin' (Df u)))) B u := by
    intro u hu
    have h1 := Ycl.hasFDerivAt.comp_hasFDerivWithinAt u (hf u hu)
    have h2 := (hasFDerivWithinAt_id u B).sub h1
    convert h2 using 1
    funext z
    simp [G, Ycl, Matrix.toLin'_apply]
  have hconv : Convex ℝ B := convex_closedBall _ _
  have hmvt : ∀ i, ∀ x ∈ B, ∀ y ∈ B, |G y i - G x i| ≤ (d.C i : ℝ) * ‖y - x‖ := by
    intro i x hx y hy
    have hGi : ∀ z ∈ B, HasFDerivWithinAt (fun w => G w i)
        ((ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin n => ℝ) i).comp (ContinuousLinearMap.id ℝ (Vn n) -
          Ycl.comp (LinearMap.toContinuousLinearMap (Matrix.toLin' (Df z))))) B z := fun z hz =>
      (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin n => ℝ) i).hasFDerivAt.comp_hasFDerivWithinAt z (hG z hz)
    have hbound : ∀ z ∈ B, ‖(ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin n => ℝ) i).comp
        (ContinuousLinearMap.id ℝ (Vn n) - Ycl.comp (LinearMap.toContinuousLinearMap (Matrix.toLin' (Df z))))‖
          ≤ (d.C i : ℝ) := by
      intro z hz
      refine ContinuousLinearMap.opNorm_le_bound _ (d.C_nonneg i) fun v => ?_
      have e : ((ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin n => ℝ) i).comp
          (ContinuousLinearMap.id ℝ (Vn n) - Ycl.comp (LinearMap.toContinuousLinearMap (Matrix.toLin' (Df z))))) v
            = ((1 - d.Yr * Df z) *ᵥ v) i := by
        show (v - d.Yr *ᵥ (Df z *ᵥ v)) i = ((1 - d.Yr * Df z) *ᵥ v) i
        rw [Matrix.sub_mulVec, Matrix.one_mulVec, Matrix.mulVec_mulVec]
      rw [e, Real.norm_eq_abs]
      exact d.row_bound (hJ z hz) v i
    have := hconv.norm_image_sub_le_of_norm_hasFDerivWithin_le hGi hbound hx hy
    simpa [Real.norm_eq_abs] using this
  have h0B : (0 : Vn n) ∈ B := mem_closedBall_self zero_le_one
  have hmaps : MapsTo G B B := by
    intro x hx
    have hxd : ‖x‖ ≤ 1 := by simpa [B, dist_zero_right] using hx
    show dist (G x) 0 ≤ 1
    rw [dist_zero_right]
    refine (pi_norm_le_iff_of_nonneg zero_le_one).mpr fun i => ?_
    have h1 := hmvt i 0 h0B x hx
    have h2 : G 0 i = -(d.Yr *ᵥ f 0) i := by simp [G]
    have h3 := I.abs_le_mag (hgx i)
    have h4 := hcheck i
    have h5 := d.C_nonneg i
    rw [Real.norm_eq_abs]
    have h6 : |G x i| ≤ |G x i - G 0 i| + |G 0 i| := by
      have := abs_add_le (G x i - G 0 i) (G 0 i)
      simpa using this
    have h3' : |G 0 i| ≤ ((d.Yg i).mag : ℝ) := by rw [h2, abs_neg]; exact h3
    rw [sub_zero] at h1
    have h7 : (d.C i : ℝ) * ‖x‖ ≤ (d.C i : ℝ) := by nlinarith
    linarith
  have hcontr : ∀ x ∈ B, ∀ y ∈ B, ‖G y - G x‖ ≤ q * ‖y - x‖ := by
    intro x hx y hy
    refine (pi_norm_le_iff_of_nonneg (mul_nonneg hq0 (norm_nonneg _))).mpr fun i => ?_
    rw [Real.norm_eq_abs, Pi.sub_apply]
    exact (hmvt i x hx y hy).trans (mul_le_mul_of_nonneg_right (hCq i) (norm_nonneg _))
  have hdet : ∀ u ∈ B, d.Yr.det ≠ 0 ∧ (Df u).det ≠ 0 := by
    intro u hu
    have := d.det_ne_zero (hJ u hu) hC
    rw [Matrix.det_mul] at this
    exact ⟨left_ne_zero_of_mul this, right_ne_zero_of_mul this⟩
  have hY : ∀ w : Vn n, d.Yr *ᵥ w = 0 → w = 0 := by
    intro w hw
    by_contra hne
    exact (hdet _ h0B).1 (Matrix.exists_mulVec_eq_zero_iff.mp ⟨w, hne, hw⟩)
  have hfix_iff : ∀ x, G x = x ↔ f x = 0 := by
    intro x
    constructor
    · intro h
      exact hY _ (sub_eq_self.mp h)
    · intro h
      show x - d.Yr *ᵥ f x = x
      rw [h, Matrix.mulVec_zero, sub_zero]
  refine ⟨?_, fun x hx _ => (hdet x hx).2⟩
  have hcomplete : IsComplete B := isClosed_closedBall.isComplete
  obtain ⟨y, hyB, hyfix, -⟩ := by
    refine ContractingWith.exists_fixedPoint' (K := Real.toNNReal q) hcomplete hmaps ?_ h0B
      (edist_ne_top _ _)
    refine ⟨by rw [← NNReal.coe_lt_coe, Real.coe_toNNReal _ hq0]; simpa using hq1, ?_⟩
    refine LipschitzWith.of_dist_le_mul fun a b => ?_
    rw [Real.coe_toNNReal _ hq0, Subtype.dist_eq, Subtype.dist_eq, dist_eq_norm, dist_eq_norm]
    exact hcontr b b.2 a a.2
  refine ⟨y, ⟨hyB, (hfix_iff y).mp hyfix⟩, ?_⟩
  rintro z ⟨hzB, hz⟩
  have hzf := (hfix_iff z).mpr hz
  have hyf : G y = y := hyfix
  have := hcontr y hyB z hzB
  rw [hzf, hyf] at this
  have hn : ‖z - y‖ = 0 := by
    have := norm_nonneg (z - y)
    nlinarith
  exact sub_eq_zero.mp (norm_eq_zero.mp hn)

end DSS.Krawczyk
