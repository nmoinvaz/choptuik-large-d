# Mode count: how many unstable modes does the DSS solution have?

The referee objection (notes/REFEREE_SIM.md, row 2): the growth rate λ is a local root of det M(λ) near its predicted
value (`hp/gamma_dd.py`, `find_lam`), so a second unstable mode would have gone unnoticed, and then 1/λ would not be
the critical exponent. Here the zeros of det M(λ) are counted by the argument principle in boxes of the complex λ plane
that cover −0.1 < Re λ < 6.2–6.6 and both symmetry sectors of the perturbations.

**Verdict.** At D = 30.07, 45.05 and 61.03 (double-double states, production resolution M = 32), det M has exactly
three zeros in −0.1 < Re λ < 6.2–6.6, |Im λ| < 3ω/2, a region that holds every growth rate of both symmetry sectors:

- λ = 0, the translation in τ (marginal);
- λ_g = 1 + ⟨c₁⟩ (0.83, 0.87, 0.90), a residual gauge mode of this formulation, a change of the time coordinate
  that changes nothing physical (section 1);
- λ_phys (2.0797, 2.0567, 2.0435), the known growth rate.

So the solution has **one physical unstable mode** at each of these D, and γ = 1/λ_phys. The deliberate checks behave
as they should: the box holding λ_phys alone counts 1, the box next to it counts 0, the gauge box counts 1, and the box
holding seven poles of det M winds −7. RESOLUTION_SUMMARY At D = 4 (f64 solver, real λ only) det M changes sign only
at λ_g = 0.6139 and λ_phys = 2.6741 on 0.02 < λ < 6.47. What is not covered (Re λ > 6.6, lower D with complex λ, other
resolutions) is in sections 5 and 6.

DETAILS_SECTIONS
