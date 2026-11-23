import DSS

/-!
Review of #1079: the control of `Exc1079.lean`. Every statement here is false, so `lake env lean` must reject the file
(exit status 1, one "decide" error per theorem). Copy to `lean/review/` of a checkout of `feat/lean-excited` and run,
from `lean/`:

    lake env lean review/Exc1079MustFail.lean      # must fail
-/

open DSS DSS.Krawczyk

namespace DSS.Krawczyk.Review1079

/-- a zero planted outside the box must not pass -/
theorem wrong1 : BoxData.check { M3exc1 with
    Yg := fun i => if i = 0 then (M3exc1.Yg 0).add (I.const (2 * M3exc1.w 0)) else M3exc1.Yg i } = true := by
  decide +kernel
/-- the widened control must not pass -/
theorem wrong2 : M3exc2Wide.check = true := by decide +kernel
/-- the clean instance must not be decided false -/
theorem wrong3 : M3exc2.check = false := by decide +kernel

end DSS.Krawczyk.Review1079
