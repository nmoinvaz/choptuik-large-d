import DSS

/-!
Review of #1077 (notes/LEAN_REVIEW_1077.md). Run from `lean/` after `lake build`:

    lake env lean review/Axioms.lean

It prints the axioms of every main theorem and instance, and decides that no interval of an instance is empty
(an empty interval would make `EnclOK` false and the instance's theorem vacuous; `check` does not test this).
-/

open DSS DSS.Krawczyk

#print axioms DSS.inner_weber_constant
#print axioms DSS.Krawczyk.krawczyk_of_check
#print axioms DSS.Krawczyk.krawczykN_of_check
#print axioms DSS.Krawczyk.krawczyk_box
#print axioms DSS.Krawczyk.M1capd_check
#print axioms DSS.Krawczyk.M1capd_unique_simple_zero
#print axioms DSS.Krawczyk.M1arb_check
#print axioms DSS.Krawczyk.M1arb_unique_simple_zero
#print axioms DSS.Krawczyk.M3cert_check
#print axioms DSS.Krawczyk.M3cert_unique_simple_zero
#print axioms DSS.Krawczyk.M5cert_check
#print axioms DSS.Krawczyk.M5cert_unique_simple_zero
#print axioms DSS.Krawczyk.M1capdWide_check_fails
#print axioms DSS.Krawczyk.M1arbWide_check_fails
#print axioms DSS.Krawczyk.M3certWide_check_fails
#print axioms DSS.Krawczyk.M5certWide_check_fails

namespace DSS.Krawczyk.Review

/-- Every interval of a 2-unknown instance is non-empty. -/
def wf2 (d : Data) : Bool :=
  decide (∀ i, (d.g i).lo ≤ (d.g i).hi) && decide (∀ i j, (d.J i j).lo ≤ (d.J i j).hi)

/-- Every interval of an n-unknown instance is non-empty. -/
def wfN {n : ℕ} (b : BoxData n) : Bool :=
  decide (∀ i, (b.Yg i).lo ≤ (b.Yg i).hi) && decide (∀ i j, (b.YJ i j).lo ≤ (b.YJ i j).hi)

theorem M1capd_wf : wf2 M1capd = true := by decide +kernel
theorem M1arb_wf : wf2 M1arb = true := by decide +kernel
theorem M3cert_wf : wfN M3cert = true := by decide +kernel
theorem M5cert_wf : wfN M5cert = true := by decide +kernel

/-- The control of this check: an inverted interval is seen. -/
theorem wf_fires : wf2 { M1capd with g := ![⟨1, 0⟩, ⟨0, 1⟩] } = false := by decide +kernel

/-- `check` itself does not see an inverted interval: with g's first interval inverted to ⟨hi, lo⟩ the test still
    passes, and `EnclOK` is then false for every map. -/
theorem check_blind_to_empty :
    Data.check { M1capd with g := ![⟨(M1capd.g 0).hi, (M1capd.g 0).lo⟩, M1capd.g 1] } = true := by decide +kernel

/-- For n unknowns `check` does not read Y at all: with Y = 0 the test is unchanged, while `EnclOK` is then false
    for every map (Y Dg = 0 does not lie in M3cert's YJ, whose diagonal is near 1). -/
theorem checkN_blind_to_Y : BoxData.check { M3cert with Y := fun _ _ => 0 } = true := by decide +kernel

end DSS.Krawczyk.Review
