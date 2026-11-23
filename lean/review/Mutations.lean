import DSS

/-!
Review of #1077 (notes/LEAN_REVIEW_1077.md): what `check` sees and what it does not, decided by the kernel on
mutated copies of the instances. Run from `lean/` after `lake build`:

    lake env lean review/Mutations.lean

Each verdict was predicted first by `review/reproduce.py`'s exact mirror of `check` (its `mirror` command agrees with
Lean on the 8 instance files). A `= false` theorem is a mutation the test catches; a `= true` theorem under
"not seen" is a wrong transcription that would pass, so only a comparison with the source (`reproduce.py all`)
can catch it.
-/

namespace DSS.Krawczyk.Review

/-! ## Two unknowns: seen by `check` -/

theorem capd_Y_transposed : Data.check { M1capd with Y := fun i j => M1capd.Y j i } = false := by decide +kernel
theorem capd_Y_negated : Data.check { M1capd with Y := fun i j => -M1capd.Y i j } = false := by decide +kernel
theorem capd_J_rows_swapped : Data.check { M1capd with J := fun i j => M1capd.J (i + 1) j } = false := by
  decide +kernel
theorem capd_J_transposed : Data.check { M1capd with J := fun i j => M1capd.J j i } = false := by decide +kernel

/-- A zero planted 2h from the centre along the first unknown, g(x̂) → g(x̂) + 2h ∂g/∂x₀ over the enclosure. -/
theorem capd_planted_outside : Data.check { M1capd with
    g := fun i => (M1capd.g i).add ((M1capd.J i 0).smul (2 * M1capd.h)) } = false := by decide +kernel
theorem arb_planted_outside : Data.check { M1arb with
    g := fun i => (M1arb.g i).add ((M1arb.J i 0).smul (2 * M1arb.h)) } = false := by decide +kernel

/-- The same zero planted h/2 from the centre: inside the box, the test passes. -/
theorem capd_planted_inside : Data.check { M1capd with
    g := fun i => (M1capd.g i).add ((M1capd.J i 0).smul (M1capd.h / 2)) } = true := by decide +kernel
theorem arb_planted_inside : Data.check { M1arb with
    g := fun i => (M1arb.g i).add ((M1arb.J i 0).smul (M1arb.h / 2)) } = true := by decide +kernel

/-! ## Two unknowns: not seen by `check` -/

/-- J and Y transposed together: the test passes on the transposed Jacobian. The exporter computes Y from the J it
    writes, so a transposed J in the exporter would give exactly this instance. -/
theorem capd_J_and_Y_transposed :
    Data.check { M1capd with J := fun i j => M1capd.J j i, Y := fun i j => M1capd.Y j i } = true := by decide +kernel
theorem arb_J_and_Y_transposed :
    Data.check { M1arb with J := fun i j => M1arb.J j i, Y := fun i j => M1arb.Y j i } = true := by decide +kernel

/-- The two components of g exchanged (rows of J swapped, columns of Y swapped): passes, and is harmless, since
    it is the same map with its components renamed. -/
def capdSwapped : Data where
  xhat := M1capd.xhat
  h := M1capd.h
  g := fun i => M1capd.g (i + 1)
  J := fun i j => M1capd.J (i + 1) j
  Y := fun i j => M1capd.Y i (j + 1)

theorem capd_components_swapped : capdSwapped.check = true := by decide +kernel

/-! ## n unknowns: seen by `check` -/

theorem m3_YJ_rows_swapped : BoxData.check { M3cert with
    YJ := fun i j => M3cert.YJ (if i = 2 then 3 else if i = 3 then 2 else i) j } = false := by decide +kernel
theorem m3_YJ_transposed : BoxData.check { M3cert with YJ := fun i j => M3cert.YJ j i } = false := by decide +kernel
theorem m3_w_swapped : BoxData.check { M3cert with w := ![M3cert.w 0, M3cert.w 2, M3cert.w 1, M3cert.w 3] } = false := by
  decide +kernel
/-- Y g(0) moved by 2 w₀ in the first unknown (a zero planted outside the box), and by 2 w₂ in the third. -/
theorem m3_planted_outside_0 : BoxData.check { M3cert with
    Yg := fun i => if i = 0 then (M3cert.Yg 0).add (I.const (2 * M3cert.w 0)) else M3cert.Yg i } = false := by
  decide +kernel
theorem m3_planted_outside_2 : BoxData.check { M3cert with
    Yg := fun i => if i = 2 then (M3cert.Yg 2).add (I.const (2 * M3cert.w 2)) else M3cert.Yg i } = false := by
  decide +kernel
theorem m3_planted_inside_2 : BoxData.check { M3cert with
    Yg := fun i => if i = 2 then (M3cert.Yg 2).add (I.const (M3cert.w 2 / 2)) else M3cert.Yg i } = true := by
  decide +kernel
theorem m5_YJ_transposed : BoxData.check { M5cert with YJ := fun i j => M5cert.YJ j i } = false := by decide +kernel
theorem m5_planted_outside_2 : BoxData.check { M5cert with
    Yg := fun i => if i = 2 then (M5cert.Yg 2).add (I.const (2 * M5cert.w 2)) else M5cert.Yg i } = false := by
  decide +kernel
theorem m5_planted_inside_2 : BoxData.check { M5cert with
    Yg := fun i => if i = 2 then (M5cert.Yg 2).add (I.const (M5cert.w 2 / 2)) else M5cert.Yg i } = true := by
  decide +kernel

/-! ## n unknowns: not seen by `check` (it does not read Y) -/

theorem m3_Y_transposed : BoxData.check { M3cert with Y := fun i j => M3cert.Y j i } = true := by decide +kernel
theorem m5_Y_zero : BoxData.check { M5cert with Y := fun _ _ => 0 } = true := by decide +kernel

end DSS.Krawczyk.Review
