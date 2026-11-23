import DSS

/-!
Review of #1079: the Lean instances of the two excited M = 3 zeros (branch `feat/lean-excited`). Copy this file to
`lean/review/` of a checkout of that branch and run, from `lean/` after `lake build`:

    lake env lean review/Exc1079.lean

It prints the axioms of the new theorems (expected: propext, Classical.choice, Quot.sound), decides that no interval
of the two instances is empty, and decides mutated copies. A `= false` theorem is a mutation the test must catch: the
file compiles only if the kernel decides each one as stated. Each verdict was predicted first by
`solver/proof/review_1079/lean_predict.py`.
-/

open DSS DSS.Krawczyk

#print axioms DSS.Krawczyk.M3exc1_check
#print axioms DSS.Krawczyk.M3exc1_unique_simple_zero
#print axioms DSS.Krawczyk.M3exc1Wide_check_fails
#print axioms DSS.Krawczyk.M3exc1_detY
#print axioms DSS.Krawczyk.M3exc1_nonvacuous
#print axioms DSS.Krawczyk.M3exc2_check
#print axioms DSS.Krawczyk.M3exc2_unique_simple_zero
#print axioms DSS.Krawczyk.M3exc2Wide_check_fails
#print axioms DSS.Krawczyk.M3exc2_detY
#print axioms DSS.Krawczyk.M3exc2_nonvacuous

namespace DSS.Krawczyk.Review1079

/-- Every interval of an n-unknown instance is non-empty (as review/Axioms.lean's wfN). -/
def wfN {n : ℕ} (b : BoxData n) : Bool :=
  decide (∀ i, (b.Yg i).lo ≤ (b.Yg i).hi) && decide (∀ i j, (b.YJ i j).lo ≤ (b.YJ i j).hi)

theorem M3exc1_wf : wfN M3exc1 = true := by decide +kernel
theorem M3exc2_wf : wfN M3exc2 = true := by decide +kernel

/-! The half-widths are the boxes of the runs: (1e-12, 1e-12, 1e-13, 1e-13) and (1e-12, 1e-12, 1e-12, 1e-12). -/
theorem M3exc1_w : M3exc1.w = ![1 / 10 ^ 12, 1 / 10 ^ 12, 1 / 10 ^ 13, 1 / 10 ^ 13] := by decide +kernel
theorem M3exc2_w : M3exc2.w = ![1 / 10 ^ 12, 1 / 10 ^ 12, 1 / 10 ^ 12, 1 / 10 ^ 12] := by decide +kernel

/-! Mutations the test must catch. -/
theorem exc1_YJ_transposed : BoxData.check { M3exc1 with YJ := fun i j => M3exc1.YJ j i } = false := by decide +kernel
/-- Y g(0) moved by 2 w₀ in the first unknown (a zero planted outside the box), and by 2 w₂ in the third. -/
theorem exc1_planted_outside_0 : BoxData.check { M3exc1 with
    Yg := fun i => if i = 0 then (M3exc1.Yg 0).add (I.const (2 * M3exc1.w 0)) else M3exc1.Yg i } = false := by
  decide +kernel
theorem exc1_planted_outside_2 : BoxData.check { M3exc1 with
    Yg := fun i => if i = 2 then (M3exc1.Yg 2).add (I.const (2 * M3exc1.w 2)) else M3exc1.Yg i } = false := by
  decide +kernel
theorem exc2_planted_outside_0 : BoxData.check { M3exc2 with
    Yg := fun i => if i = 0 then (M3exc2.Yg 0).add (I.const (2 * M3exc2.w 0)) else M3exc2.Yg i } = false := by
  decide +kernel
theorem exc2_planted_outside_2 : BoxData.check { M3exc2 with
    Yg := fun i => if i = 2 then (M3exc2.Yg 2).add (I.const (2 * M3exc2.w 2)) else M3exc2.Yg i } = false := by
  decide +kernel
/-! Controls that must pass: a zero planted inside (moved by a tenth of the half-width). -/
theorem exc1_planted_inside_0 : BoxData.check { M3exc1 with
    Yg := fun i => if i = 0 then (M3exc1.Yg 0).add (I.const (M3exc1.w 0 / 10)) else M3exc1.Yg i } = true := by
  decide +kernel
theorem exc2_planted_inside_0 : BoxData.check { M3exc2 with
    Yg := fun i => if i = 0 then (M3exc2.Yg 0).add (I.const (M3exc2.w 0 / 10)) else M3exc2.Yg i } = true := by
  decide +kernel

/-! Not seen by `check` (as #1077 found for Y at M3cert): at the second zero all four half-widths are equal and a
    transposed YJ still passes, so only a comparison with the source catches a transposition there. -/
theorem exc2_YJ_transposed_not_seen : BoxData.check { M3exc2 with YJ := fun i j => M3exc2.YJ j i } = true := by
  decide +kernel

end DSS.Krawczyk.Review1079
