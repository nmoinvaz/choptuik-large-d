import DSS

/-!
Review of #1077 (notes/LEAN_REVIEW_1077.md): three false verdicts. This file must NOT compile. From `lean/`:

    lake env lean review/MustFail.lean      -- exit 1, with three errors "Tactic `decide` failed"

It shows that a verdict stated in `review/Mutations.lean` or in an instance file can fail.
-/

namespace DSS.Krawczyk.Review

theorem wrong1 : Data.check { M1capd with Y := fun i j => M1capd.Y j i } = true := by decide +kernel
theorem wrong2 : BoxData.check { M3cert with YJ := fun i j => M3cert.YJ j i } = true := by decide +kernel
theorem wrong3 : M5cert.check = false := by decide +kernel

end DSS.Krawczyk.Review
