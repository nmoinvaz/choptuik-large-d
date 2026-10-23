# A solver beyond the light cone: feasibility study

*Work in progress (branch `wip/beyond-lightcone`). Sections are filled in as milestones are reached.*

## 0. Pre-registered criteria for the toy test (written before running it)

**Toy model.** The linear, gravity-free, one-harmonic problem of Proposition 2 in z = x², y = x w(z), is Gauss's
equation z(1−z)w″ + [κ − (α+β+1)z]w′ − αβw = 0 with κ = (n+1)/2. In Proposition 2, α = (1+s)/2, β = (2+s)/2 and the
light-cone coefficient C(n,s) ∝ 1/(Γ(α)Γ(β)) never vanishes for imaginary s. The toy shifts α by a constant so that it
does vanish at a chosen s*:

- α = (s − s*)/2 − 1, β = 1 + s/2, κ = (n+1)/2, unknown complex s (Re s plays the growth rate, Im s the frequency).
- Exact answer: s = s* for every n (α = −1, C(n, s*) = 0 by Gauss's connection formula; the centre-regular solution is
  the polynomial 1 − βz/κ).
- Same structure as the full problem at leading order: the 1/n expansion exists for every s and is analytic at the light
  cone order by order; the selection is the vanishing of an n-independent Stokes constant 1/(Γ(α)Γ(β)); the obstruction
  is the fast mode x^{2−n}(1−x²)^{κ−α−β}.
- Plants: s* = iω* with ω* = 2π/1.6438937, and s* = 0.3 + 2.5i.

**Methods compared**, each a residual R(u) solved by (Gauss–)Newton in mpmath:

- **S(x_m)**, the standard layout of `hp/`: centre series to x = 0.3, Taylor steps along the real axis to x_m, horizon
  series at the light cone evaluated at x_m, match the state (w, (1−z)zw′/κ); unknowns s and the horizon amplitude. x_m =
  0.7 is the `hp/` solver's last node; x_m = 0.5, 0.8, 0.9 are variations.
- **C(ξ₀)**, candidate (c): the centre-regular solution continued from z = ½ along two paths in Re z ≥ ½ (where the fast
  mode is never larger than the slow one), one above and one below the light cone, to the real point z₀ = ξ₀²n (x₀ =
  ξ₀√n); residual Y⁺(z₀) − Y⁻(z₀); unknown s. ξ₀ = 1.
- **A**, candidate (a) on the real axis: the light-cone-analytic solution continued outward along real z from z = 1.9 (the
  horizon series converges for z < 2) to z₀ = n and matched there against Y⁺(z₀) of C.

**Measures**, at n = 20, 60, 100, 200: the condition number κ_s of the selected s (absolute error in s per unit
relative perturbation of the computed states entering R, from the exact Jacobian), and the working precision P_min (decimal
digits) at which the full pipeline returns s* to 10⁻⁸.

**Pass criterion (from the task, made operational).** Fit ln κ_s = a n + p ln n + c over the four n. "Polynomial" means
a < 0.05; "like e^{n/2}" means a > 0.3. The candidate passes if a_C < 0.05 and a_S(0.7) > 0.3, and its control fails.

**Controls that must fail.** (i) C with both paths through the upper half plane: the residual vanishes identically, so no
s is selected (Jacobian zero). (ii) C started from the centre-singular solution (the x^{2−n} branch) instead of the regular
one: it must not return s*. (iii) `sens_fit.py --mutate=exp`: a planted e^{n/2} law must be classified as exponential.

**Prediction, written before running** (from §2 below): the fast mode is exactly x^{2−n}(1−x²)^{n/2}·(…), whose modulus is
1 on the curve Re x² = ½, which passes through x = 1/√2 ≈ 0.707. So S(x_m) should have κ_s ∝ e^{nΦ(x_m)} × poly(n), with
Φ(x) = ln x − ½ln(1−x²): a ≈ −0.02 at x_m = 0.7 (polynomial), a ≈ +0.72 at x_m = 0.9; C polynomial; A exponential
(a ≈ 0.37). If this holds, the second half of the pass criterion fails: the standard layout at the `hp/` matching point is
not e^{n/2}-ill-conditioned in the toy, and the candidate has nothing exponential to remove.
