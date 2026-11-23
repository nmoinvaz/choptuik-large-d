import Mathlib

/-!
# A Krawczyk test in contraction form (proof/FORMAL.md §4, row 17)

For a C¹ map g on a box X = x̂ + [-h, h]² whose value at x̂ and whose Jacobian over X lie in given rational intervals,
the exact test

  |Y g(x̂)|_i + h (M_i0 + M_i1) < h,   M_ij = max |(I - Y J)_ij| over the enclosure J,   i = 0, 1,

implies that g has exactly one zero in X and that Dg is invertible there. The map G(x) = x - Y g(x) sends X into X and
is a contraction (the mean value inequality, row by row); Banach's theorem gives its fixed point, and the row bound
makes Y and every Dg(x), x ∈ X, invertible, so the fixed points of G are the zeros of g. No Brouwer theorem is used.
The enclosures themselves are a hypothesis (`EnclOK`): Lean checks the logic and the arithmetic, not the integration
that produced them.
-/

open Metric Set Matrix

namespace DSS.Krawczyk

/-- ℝ² with Mathlib's sup norm, so a box of half-width h is a closed ball. -/
abbrev V := Fin 2 → ℝ
abbrev M2 := Matrix (Fin 2) (Fin 2) ℝ

/-! ## Rational intervals -/

/-- A closed interval with rational ends. -/
structure I where
  lo : ℚ
  hi : ℚ

namespace I

def mem (a : I) (x : ℝ) : Prop := (a.lo : ℝ) ≤ x ∧ x ≤ a.hi

def const (c : ℚ) : I := ⟨c, c⟩
def add (a b : I) : I := ⟨a.lo + b.lo, a.hi + b.hi⟩
def sub (a b : I) : I := ⟨a.lo - b.hi, a.hi - b.lo⟩
def smul (y : ℚ) (a : I) : I := ⟨min (y * a.lo) (y * a.hi), max (y * a.lo) (y * a.hi)⟩
/-- The largest modulus in the interval. -/
def mag (a : I) : ℚ := max |a.lo| |a.hi|

lemma mem_const (c : ℚ) : (const c).mem (c : ℝ) := ⟨le_rfl, le_rfl⟩

lemma mem_add {a b : I} {x y : ℝ} (ha : a.mem x) (hb : b.mem y) : (a.add b).mem (x + y) := by
  obtain ⟨h1, h2⟩ := ha
  obtain ⟨h3, h4⟩ := hb
  simp only [mem, add, Rat.cast_add]
  constructor <;> linarith

lemma mem_sub {a b : I} {x y : ℝ} (ha : a.mem x) (hb : b.mem y) : (a.sub b).mem (x - y) := by
  obtain ⟨h1, h2⟩ := ha
  obtain ⟨h3, h4⟩ := hb
  simp only [mem, sub, Rat.cast_sub]
  constructor <;> linarith

lemma mem_smul (y : ℚ) {a : I} {x : ℝ} (h : a.mem x) : (a.smul y).mem ((y : ℝ) * x) := by
  obtain ⟨h1, h2⟩ := h
  simp only [mem, smul, Rat.cast_min, Rat.cast_max, Rat.cast_mul]
  rcases le_total 0 (y : ℝ) with hy | hy
  · have := mul_le_mul_of_nonneg_left h1 hy
    have := mul_le_mul_of_nonneg_left h2 hy
    exact ⟨min_le_of_left_le (by linarith), le_max_of_le_right (by linarith)⟩
  · have := mul_le_mul_of_nonpos_left h1 hy
    have := mul_le_mul_of_nonpos_left h2 hy
    exact ⟨min_le_of_right_le (by linarith), le_max_of_le_left (by linarith)⟩

lemma abs_le_mag {a : I} {x : ℝ} (h : a.mem x) : |x| ≤ (a.mag : ℝ) := by
  simp only [mag, Rat.cast_max, Rat.cast_abs]
  exact abs_le_max_abs_abs h.1 h.2

lemma mag_nonneg (a : I) : (0 : ℝ) ≤ (a.mag : ℝ) := by
  simp only [mag, Rat.cast_max, Rat.cast_abs]
  exact le_max_of_le_left (abs_nonneg _)

end I

/-! ## The data of one test and the test itself -/

/-- One Krawczyk test. Every entry is an exact rational (arb's midpoints and radii are dyadic). -/
structure Data where
  /-- the box centre -/
  xhat : Fin 2 → ℚ
  /-- the half-width, the same in both coordinates -/
  h : ℚ
  /-- an enclosure of g(x̂) -/
  g : Fin 2 → I
  /-- an entrywise enclosure of Dg over the box -/
  J : Fin 2 → Fin 2 → I
  /-- the preconditioner: any matrix -/
  Y : Fin 2 → Fin 2 → ℚ

namespace Data

variable (d : Data)

def center : V := fun i => (d.xhat i : ℝ)
def box : Set V := closedBall d.center (d.h : ℝ)
def Yr : M2 := Matrix.of fun i j => (d.Y i j : ℝ)

/-- An enclosure of (Y v)_i for v_k ∈ a k. -/
def rowY (i : Fin 2) (a : Fin 2 → I) : I := ((a 0).smul (d.Y i 0)).add ((a 1).smul (d.Y i 1))

/-- An enclosure of (I - Y J)_ij over the Jacobian's enclosure. -/
def entry (i j : Fin 2) : I := (I.const (if i = j then 1 else 0)).sub (d.rowY i fun k => d.J k j)

/-- M_ij, the largest modulus of (I - Y J)_ij. -/
def Mij (i j : Fin 2) : ℚ := (d.entry i j).mag

/-- The row sum M_i0 + M_i1. -/
def C (i : Fin 2) : ℚ := d.Mij i 0 + d.Mij i 1

/-- The largest modulus of (Y g(x̂))_i. -/
def gmag (i : Fin 2) : ℚ := (d.rowY i d.g).mag

/-- The test in exact rational arithmetic. -/
def check : Bool :=
  decide (0 < d.h) && decide (d.gmag 0 + d.h * d.C 0 < d.h) && decide (d.gmag 1 + d.h * d.C 1 < d.h)

end Data

/-- The hypothesis that the integration certifies outside Lean: g(x̂) and Dg over the box lie in the enclosures. -/
def EnclOK (d : Data) (g : V → V) (Dg : V → M2) : Prop :=
  (∀ i, (d.g i).mem (g d.center i)) ∧ ∀ x ∈ d.box, ∀ i j, (d.J i j).mem (Dg x i j)

/-! ## Soundness of the interval layer -/

namespace Data

variable (d : Data)

lemma rowY_mem (i : Fin 2) {a : Fin 2 → I} {v : V} (hv : ∀ k, (a k).mem (v k)) :
    (d.rowY i a).mem ((d.Yr *ᵥ v) i) := by
  have e : (d.Yr *ᵥ v) i = (d.Y i 0 : ℝ) * v 0 + (d.Y i 1 : ℝ) * v 1 := by
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two, Yr]
  rw [e]
  exact I.mem_add (I.mem_smul _ (hv 0)) (I.mem_smul _ (hv 1))

lemma entry_mem {A : M2} (hA : ∀ i j, (d.J i j).mem (A i j)) (i j : Fin 2) :
    (d.entry i j).mem ((1 - d.Yr * A) i j) := by
  have e : (1 - d.Yr * A) i j = ((if i = j then 1 else 0 : ℚ) : ℝ) - (d.Yr *ᵥ fun k => A k j) i := by
    simp [Matrix.sub_apply, Matrix.one_apply, Matrix.mul_apply, Matrix.mulVec, dotProduct]
    split_ifs <;> simp
  rw [e]
  exact I.mem_sub (I.mem_const _) (d.rowY_mem i fun k => hA k j)

lemma C_nonneg (i : Fin 2) : (0 : ℝ) ≤ (d.C i : ℝ) := by
  simp only [C, Rat.cast_add]
  exact add_nonneg (I.mag_nonneg _) (I.mag_nonneg _)

/-- The row bound: |((I - Y A) v)_i| ≤ C_i ‖v‖ for every A inside the enclosure. -/
lemma row_bound {A : M2} (hA : ∀ i j, (d.J i j).mem (A i j)) (v : V) (i : Fin 2) :
    |((1 - d.Yr * A) *ᵥ v) i| ≤ (d.C i : ℝ) * ‖v‖ := by
  have e : ((1 - d.Yr * A) *ᵥ v) i = (1 - d.Yr * A) i 0 * v 0 + (1 - d.Yr * A) i 1 * v 1 := by
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two]
  have b0 : |(1 - d.Yr * A) i 0| ≤ (d.Mij i 0 : ℝ) := I.abs_le_mag (d.entry_mem hA i 0)
  have b1 : |(1 - d.Yr * A) i 1| ≤ (d.Mij i 1 : ℝ) := I.abs_le_mag (d.entry_mem hA i 1)
  have v0 : |v 0| ≤ ‖v‖ := by simpa [Real.norm_eq_abs] using norm_le_pi_norm v 0
  have v1 : |v 1| ≤ ‖v‖ := by simpa [Real.norm_eq_abs] using norm_le_pi_norm v 1
  have hn : 0 ≤ ‖v‖ := norm_nonneg v
  rw [e]
  calc |(1 - d.Yr * A) i 0 * v 0 + (1 - d.Yr * A) i 1 * v 1|
      ≤ |(1 - d.Yr * A) i 0| * |v 0| + |(1 - d.Yr * A) i 1| * |v 1| := by
        rw [← abs_mul, ← abs_mul]; exact abs_add_le _ _
    _ ≤ (d.Mij i 0 : ℝ) * ‖v‖ + (d.Mij i 1 : ℝ) * ‖v‖ :=
        add_le_add (mul_le_mul b0 v0 (abs_nonneg _) (I.mag_nonneg _))
          (mul_le_mul b1 v1 (abs_nonneg _) (I.mag_nonneg _))
    _ = (d.C i : ℝ) * ‖v‖ := by simp only [C, Rat.cast_add]; ring

/-- What `check` gives, in ℝ. -/
lemma check_spec (hc : d.check = true) :
    (0 : ℝ) < d.h ∧ ∀ i, (d.gmag i : ℝ) + d.h * d.C i < d.h := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨h0, h1⟩, h2⟩ := hc
  refine ⟨by exact_mod_cast h0, fun i => ?_⟩
  fin_cases i
  · exact_mod_cast h1
  · exact_mod_cast h2

/-- Inside the enclosure, Y A is invertible when every row sum is below 1. -/
lemma det_ne_zero {A : M2} (hA : ∀ i j, (d.J i j).mem (A i j)) (hC : ∀ i, (d.C i : ℝ) < 1) :
    (d.Yr * A).det ≠ 0 := by
  intro h0
  obtain ⟨v, hv, hYA⟩ := Matrix.exists_mulVec_eq_zero_iff.mpr h0
  have hfix : (1 - d.Yr * A) *ᵥ v = v := by
    rw [Matrix.sub_mulVec, Matrix.one_mulVec, hYA, sub_zero]
  have hle : ‖v‖ ≤ (max (d.C 0 : ℝ) (d.C 1)) * ‖v‖ := by
    refine (pi_norm_le_iff_of_nonneg
      (mul_nonneg (le_max_of_le_left (d.C_nonneg 0)) (norm_nonneg v))).mpr fun i => ?_
    have := d.row_bound hA v i
    rw [hfix] at this
    rw [Real.norm_eq_abs]
    refine this.trans (mul_le_mul_of_nonneg_right ?_ (norm_nonneg v))
    fin_cases i
    · exact le_max_left _ _
    · exact le_max_right _ _
  have hq : max (d.C 0 : ℝ) (d.C 1) < 1 := max_lt (hC 0) (hC 1)
  have hn : ‖v‖ = 0 := by
    have := norm_nonneg v
    nlinarith
  exact hv (norm_eq_zero.mp hn)

end Data

/-! ## The theorem -/

/-- Krawczyk's test in contraction form: Banach's fixed point theorem, no Brouwer. -/
theorem krawczyk_of_check (d : Data) (hc : d.check = true) (g : V → V) (Dg : V → M2)
    (hg : ∀ x ∈ d.box,
      HasFDerivWithinAt g (LinearMap.toContinuousLinearMap (Matrix.toLin' (Dg x))) d.box x)
    (hE : EnclOK d g Dg) :
    (∃! x, x ∈ d.box ∧ g x = 0) ∧ ∀ x ∈ d.box, g x = 0 → (Dg x).det ≠ 0 := by
  obtain ⟨hh, hcheck⟩ := d.check_spec hc
  obtain ⟨hgx, hJ⟩ := hE
  set B := d.box with hBdef
  have hC : ∀ i, (d.C i : ℝ) < 1 := by
    intro i
    have h1 := hcheck i
    have h2 : (0 : ℝ) ≤ (d.gmag i : ℝ) := I.mag_nonneg _
    by_contra hcon
    have hcon := not_lt.mp hcon
    nlinarith
  -- G(x) = x - Y g(x) and its derivative, component by component
  let Ycl : V →L[ℝ] V := LinearMap.toContinuousLinearMap (Matrix.toLin' d.Yr)
  let G : V → V := fun x => x - d.Yr *ᵥ g x
  have hG : ∀ x ∈ B, HasFDerivWithinAt G
      (ContinuousLinearMap.id ℝ V - Ycl.comp (LinearMap.toContinuousLinearMap (Matrix.toLin' (Dg x)))) B x := by
    intro x hx
    have h1 := Ycl.hasFDerivAt.comp_hasFDerivWithinAt x (hg x hx)
    have h2 := (hasFDerivWithinAt_id x B).sub h1
    convert h2 using 1
    funext z
    simp [G, Ycl, Matrix.toLin'_apply]
  have hconv : Convex ℝ B := convex_closedBall _ _
  -- the mean value inequality for each component of G
  have hmvt : ∀ i, ∀ x ∈ B, ∀ y ∈ B, |G y i - G x i| ≤ (d.C i : ℝ) * ‖y - x‖ := by
    intro i x hx y hy
    have hGi : ∀ z ∈ B, HasFDerivWithinAt (fun w => G w i)
        ((ContinuousLinearMap.proj i).comp (ContinuousLinearMap.id ℝ V -
          Ycl.comp (LinearMap.toContinuousLinearMap (Matrix.toLin' (Dg z))))) B z := fun z hz =>
      (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) i).hasFDerivAt.comp_hasFDerivWithinAt z (hG z hz)
    have hbound : ∀ z ∈ B, ‖(ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) i).comp
        (ContinuousLinearMap.id ℝ V - Ycl.comp (LinearMap.toContinuousLinearMap (Matrix.toLin' (Dg z))))‖
          ≤ (d.C i : ℝ) := by
      intro z hz
      refine ContinuousLinearMap.opNorm_le_bound _ (d.C_nonneg i) fun v => ?_
      have e : ((ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) i).comp
          (ContinuousLinearMap.id ℝ V - Ycl.comp (LinearMap.toContinuousLinearMap (Matrix.toLin' (Dg z))))) v
            = ((1 - d.Yr * Dg z) *ᵥ v) i := by
        fin_cases i <;>
          simp [Ycl, Matrix.toLin'_apply, Matrix.mulVec, dotProduct,
            Matrix.mul_apply, Fin.sum_univ_two] <;> ring
      rw [e, Real.norm_eq_abs]
      exact d.row_bound (hJ z hz) v i
    have := hconv.norm_image_sub_le_of_norm_hasFDerivWithin_le hGi hbound hx hy
    simpa [Real.norm_eq_abs] using this
  have hxc : d.center ∈ B := mem_closedBall_self hh.le
  -- G maps the box into itself
  have hmaps : MapsTo G B B := by
    intro x hx
    have hxd : ‖x - d.center‖ ≤ d.h := by rw [← dist_eq_norm]; exact hx
    show dist (G x) d.center ≤ d.h
    rw [dist_eq_norm]
    refine (pi_norm_le_iff_of_nonneg hh.le).mpr fun i => ?_
    have h1 := hmvt i d.center hxc x hx
    have h2 : G d.center i - d.center i = -(d.Yr *ᵥ g d.center) i := by
      show (d.center - d.Yr *ᵥ g d.center) i - d.center i = _
      rw [Pi.sub_apply]; ring
    have h3 := I.abs_le_mag (d.rowY_mem i hgx)
    have h4 := hcheck i
    have h5 := d.C_nonneg i
    rw [Real.norm_eq_abs, Pi.sub_apply]
    have : |G x i - d.center i| ≤ |G x i - G d.center i| + |G d.center i - d.center i| := by
      have := abs_add_le (G x i - G d.center i) (G d.center i - d.center i)
      simpa using this
    rw [h2, abs_neg] at this
    have h6 : (d.C i : ℝ) * ‖x - d.center‖ ≤ (d.C i : ℝ) * d.h := mul_le_mul_of_nonneg_left hxd h5
    have h7 : |G x i - G d.center i| ≤ (d.C i : ℝ) * ‖x - d.center‖ := h1
    simp only [Data.gmag] at h4
    linarith
  -- G is a contraction on the box
  set q : ℝ := max (d.C 0 : ℝ) (d.C 1) with hqdef
  have hq0 : 0 ≤ q := le_max_of_le_left (d.C_nonneg 0)
  have hq1 : q < 1 := max_lt (hC 0) (hC 1)
  have hcontr : ∀ x ∈ B, ∀ y ∈ B, ‖G y - G x‖ ≤ q * ‖y - x‖ := by
    intro x hx y hy
    refine (pi_norm_le_iff_of_nonneg (by positivity)).mpr fun i => ?_
    rw [Real.norm_eq_abs, Pi.sub_apply]
    refine (hmvt i x hx y hy).trans (mul_le_mul_of_nonneg_right ?_ (norm_nonneg _))
    fin_cases i
    · exact le_max_left _ _
    · exact le_max_right _ _
  -- Y and every Dg(x), x ∈ B, are invertible
  have hdet : ∀ x ∈ B, d.Yr.det ≠ 0 ∧ (Dg x).det ≠ 0 := by
    intro x hx
    have := d.det_ne_zero (hJ x hx) hC
    rw [Matrix.det_mul] at this
    exact ⟨left_ne_zero_of_mul this, right_ne_zero_of_mul this⟩
  have hY : ∀ w : V, d.Yr *ᵥ w = 0 → w = 0 := by
    intro w hw
    by_contra hne
    exact (hdet _ hxc).1 (Matrix.exists_mulVec_eq_zero_iff.mp ⟨w, hne, hw⟩)
  have hfix_iff : ∀ x, G x = x ↔ g x = 0 := by
    intro x
    constructor
    · intro h
      exact hY _ (sub_eq_self.mp h)
    · intro h
      show x - d.Yr *ᵥ g x = x
      rw [h, Matrix.mulVec_zero, sub_zero]
  refine ⟨?_, fun x hx _ => (hdet x hx).2⟩
  -- uniqueness by the contraction, existence by Banach's theorem on the complete box
  have hcomplete : IsComplete B := isClosed_closedBall.isComplete
  obtain ⟨y, hyB, hyfix, -⟩ := by
    refine ContractingWith.exists_fixedPoint' (K := Real.toNNReal q) hcomplete hmaps ?_ hxc
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
