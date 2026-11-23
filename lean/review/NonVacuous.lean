import DSS

/-!
Review of #1077 (notes/LEAN_REVIEW_1077.md): the hypotheses of the four instances can be met. Run from `lean/`:

    lake env lean review/NonVacuous.lean

When no interval is empty, the affine map g(x) = g_lo + J_lo (x - x̂) is differentiable on the box and satisfies
`EnclOK`. For n unknowns the hypotheses are on Y g and Y Dg, and the same holds when Y is invertible, with
g(x) = Y⁻¹ (Yg_lo + YJ_lo (x - x̂)). The kernel decides that no interval is empty and that det Y ≠ 0. So none of the
four `*_unique_simple_zero` theorems is vacuous.
-/

open Metric Set Matrix

namespace DSS.Krawczyk.Review

theorem exists_map (d : Data) (hg : ∀ i, (d.g i).lo ≤ (d.g i).hi) (hJ : ∀ i j, (d.J i j).lo ≤ (d.J i j).hi) :
    ∃ (g : V → V) (Dg : V → M2),
      (∀ x ∈ d.box, HasFDerivWithinAt g (LinearMap.toContinuousLinearMap (Matrix.toLin' (Dg x))) d.box x) ∧
        EnclOK d g Dg := by
  let A : M2 := Matrix.of fun i j => ((d.J i j).lo : ℝ)
  let c : V := fun i => ((d.g i).lo : ℝ)
  let Acl : V →L[ℝ] V := LinearMap.toContinuousLinearMap (Matrix.toLin' A)
  refine ⟨fun x => c + Acl (x - d.center), fun _ => A, ?_, ?_, ?_⟩
  · intro x _
    have h1 : HasFDerivAt (fun x : V => x - d.center) (ContinuousLinearMap.id ℝ V) x :=
      (hasFDerivAt_id x).sub_const d.center
    have h2 := (Acl.hasFDerivAt.comp x h1).const_add c
    simpa using h2.hasFDerivWithinAt
  · intro i
    have e : (c + Acl (d.center - d.center)) i = ((d.g i).lo : ℝ) := by simp [c]
    show (d.g i).mem ((c + Acl (d.center - d.center)) i)
    rw [e]
    exact ⟨le_rfl, by exact_mod_cast hg i⟩
  · intro x _ i j
    show ((d.J i j).lo : ℝ) ≤ ((d.J i j).lo : ℝ) ∧ ((d.J i j).lo : ℝ) ≤ ((d.J i j).hi : ℝ)
    exact ⟨le_rfl, by exact_mod_cast hJ i j⟩

theorem M1capd_nonvacuous : ∃ (g : V → V) (Dg : V → M2),
    (∀ x ∈ M1capd.box, HasFDerivWithinAt g (LinearMap.toContinuousLinearMap (Matrix.toLin' (Dg x))) M1capd.box x) ∧
      EnclOK M1capd g Dg :=
  exists_map M1capd (by decide +kernel) (by decide +kernel)

theorem M1arb_nonvacuous : ∃ (g : V → V) (Dg : V → M2),
    (∀ x ∈ M1arb.box, HasFDerivWithinAt g (LinearMap.toContinuousLinearMap (Matrix.toLin' (Dg x))) M1arb.box x) ∧
      EnclOK M1arb g Dg :=
  exists_map M1arb (by decide +kernel) (by decide +kernel)

/-- The same for n unknowns, when Y is invertible: g(x) = Y⁻¹ (Yg_lo + YJ_lo (x - x̂)). -/
theorem exists_mapN {n : ℕ} (b : BoxData n) (hY : (Matrix.of b.Y).det ≠ 0)
    (hg : ∀ i, (b.Yg i).lo ≤ (b.Yg i).hi) (hJ : ∀ i j, (b.YJ i j).lo ≤ (b.YJ i j).hi) :
    ∃ (g : Vn n → Vn n) (Dg : Vn n → Mn n),
      (∀ x ∈ b.box, HasFDerivWithinAt g (LinearMap.toContinuousLinearMap (Matrix.toLin' (Dg x))) b.box x) ∧
        b.EnclOK g Dg := by
  have hdet : IsUnit b.Yr.det := by
    have e : b.Yr = (Rat.castHom ℝ).mapMatrix (Matrix.of b.Y) := by
      ext i j; simp [BoxData.Yr]
    rw [e, ← RingHom.map_det]
    exact isUnit_iff_ne_zero.mpr (by simpa using hY)
  let A : Mn n := Matrix.of fun i j => ((b.YJ i j).lo : ℝ)
  let c : Vn n := fun i => ((b.Yg i).lo : ℝ)
  let B : Mn n := b.Yr⁻¹ * A
  let Bcl : Vn n →L[ℝ] Vn n := LinearMap.toContinuousLinearMap (Matrix.toLin' B)
  refine ⟨fun x => b.Yr⁻¹ *ᵥ c + Bcl (x - b.center), fun _ => B, ?_, ?_, ?_⟩
  · intro x _
    have h1 : HasFDerivAt (fun x : Vn n => x - b.center) (ContinuousLinearMap.id ℝ (Vn n)) x :=
      (hasFDerivAt_id x).sub_const b.center
    have h2 := (Bcl.hasFDerivAt.comp x h1).const_add (b.Yr⁻¹ *ᵥ c)
    simpa using h2.hasFDerivWithinAt
  · intro i
    have e : b.Yr *ᵥ (b.Yr⁻¹ *ᵥ c + Bcl (b.center - b.center)) = c := by
      simp [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hdet]
    show (b.Yg i).mem ((b.Yr *ᵥ (b.Yr⁻¹ *ᵥ c + Bcl (b.center - b.center))) i)
    rw [e]
    show ((b.Yg i).lo : ℝ) ≤ ((b.Yg i).lo : ℝ) ∧ ((b.Yg i).lo : ℝ) ≤ ((b.Yg i).hi : ℝ)
    exact ⟨le_rfl, by exact_mod_cast hg i⟩
  · intro x _ i j
    have e : b.Yr * B = A := Matrix.mul_nonsing_inv_cancel_left _ _ hdet
    show (b.YJ i j).mem ((b.Yr * B) i j)
    rw [e]
    show ((b.YJ i j).lo : ℝ) ≤ ((b.YJ i j).lo : ℝ) ∧ ((b.YJ i j).lo : ℝ) ≤ ((b.YJ i j).hi : ℝ)
    exact ⟨le_rfl, by exact_mod_cast hJ i j⟩

theorem M3cert_detY : (Matrix.of M3cert.Y).det ≠ 0 := by decide +kernel
theorem M5cert_detY : (Matrix.of M5cert.Y).det ≠ 0 := by decide +kernel
/-- The control of the determinant check: a matrix of rank one is seen. -/
theorem detY_fires : ¬ (Matrix.of fun i j : Fin 4 => M3cert.Y i 0 * M3cert.Y 0 j).det ≠ 0 := by decide +kernel

theorem M3cert_nonvacuous : ∃ (g : Vn 4 → Vn 4) (Dg : Vn 4 → Mn 4),
    (∀ x ∈ M3cert.box, HasFDerivWithinAt g (LinearMap.toContinuousLinearMap (Matrix.toLin' (Dg x))) M3cert.box x) ∧
      M3cert.EnclOK g Dg :=
  exists_mapN M3cert M3cert_detY (by decide +kernel) (by decide +kernel)

theorem M5cert_nonvacuous : ∃ (g : Vn 6 → Vn 6) (Dg : Vn 6 → Mn 6),
    (∀ x ∈ M5cert.box, HasFDerivWithinAt g (LinearMap.toContinuousLinearMap (Matrix.toLin' (Dg x))) M5cert.box x) ∧
      M5cert.EnclOK g Dg :=
  exists_mapN M5cert M5cert_detY (by decide +kernel) (by decide +kernel)

#print axioms M1capd_nonvacuous
#print axioms M1arb_nonvacuous
#print axioms M3cert_nonvacuous
#print axioms M5cert_nonvacuous

end DSS.Krawczyk.Review
