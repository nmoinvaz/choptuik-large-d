# Other matter models at large D: literature survey

This page surveys critical collapse for matter models other than the massless real scalar. It asks one question:
do these models share the large-D structure found in this repository for the scalar? That structure is a finite
Δ∞ selected beyond all orders by an inner Stokes condition, and γ∞ = 1/2 from the s-translation symmetry of the inner
heat equation ([README, "Large D"](../README.md#large-d-what-the-1d-expansion-can-and-cannot-see), [GAMMA.md](../GAMMA.md),
[PROOF.md](../PROOF.md)). Compiled in October 2026.

**How to read it.**
- *References.* Every reference was checked by web search for its title, authors, journal reference and arXiv
  number. Where a journal reference could not be confirmed, the entry says so. "Not found" means a targeted search
  found nothing; it does not prove that nothing exists.
- *Numbers.* Values are quoted as published, with the paper's convention for γ (mass or radius exponent).
- *Remarks.* Statements marked *(remark)* are not from the literature. They are short derivations made while writing
  this page, and the symbolic ones were checked with sympy
  ([`solver/matter/lit_remarks_check.py`](../solver/matter/lit_remarks_check.py)). Treat them as leads, not results.

**Conventions** (as in the README):
- D is the number of spacetime dimensions and n = D − 2, so the round sphere is $`S^n`$. The PDE literature counts
  d = D − 1 space dimensions.
- x = r/(−t) and τ = −ln(−t). The inner variables are $`s = r^2/(2n)`$ and the physical time t.
- γ is the *radius* exponent, $`r_{\rm BH} \propto (p-p_*)^\gamma`$. The mass exponent is (D − 3)γ, and the two
  coincide in D = 4.
- Δ is the echoing period in τ.

Contents

- [1. The structural question: which models keep the s-translation](#1-the-structural-question-which-models-keep-the-s-translation)
- [2. Summary table](#2-summary-table)
- [3. Scalar fields with angular momentum](#3-scalar-fields-with-angular-momentum)
- [4. SU(2) Yang–Mills](#4-su2-yangmills)
- [5. σ-models and wave maps](#5-σ-models-and-wave-maps)
- [6. Vacuum collapse](#6-vacuum-collapse)
- [7. Perfect fluid, complex-scalar CSS, axion–dilaton](#7-perfect-fluid-complex-scalar-css-axiondilaton)
- [8. Charged and massive scalar](#8-charged-and-massive-scalar)
- [9. Large-D work on collapse](#9-large-d-work-on-collapse)
- [10. Recommendations](#10-recommendations)
- [11. References](#11-references)

## 1. The structural question: which models keep the s-translation

At large D the scalar's period and growth rate are selected in the inner region $`x \sim \sqrt n`$. There, in s and t,
the equations become an autonomous nonlinear heat equation, and its s-translation (a shift of r²) gives the exact
unstable mode λ = 2 (GAMMA.md). The radial operator becomes, at fixed t,

```math
\partial_r^2 + \frac{n}{r}\,\partial_r - \frac{V}{r^2} \;=\; \partial_s - \frac{V}{2n\,s} + O(1/n), \qquad s = \frac{r^2}{2n},
```

since $`(n/r)\partial_r = \partial_s`$ exactly and $`\partial_r^2 = n^{-1}(\partial_s + 2s\,\partial_s^2)`$. A potential with
V = O(1) drops out at D = ∞. A potential with **V = O(n) leaves an O(1) term ∝ 1/s**, and that term is not invariant
under s → s + const. Centrifugal barriers on $`S^n`$, the potentials of equivariant σ-models and Yang–Mills fields, and
the squashing modes of vacuum gravity are all of this kind. For such models the argument behind γ∞ = 1/2 fails, and
γ∞ is then an open number (1/2 is not excluded).

A pure centrifugal term can be removed from the linear operator by a power of s (§3.1). The stress tensor is quadratic
in the field, so the power then reappears in the metric equations. Whether those factors break the symmetry has not
been worked out for any model.

| Model | O(n)/r² term in the radial equation | At D = ∞ in the inner variables *(remark)* |
|---|---|---|
| Real or complex scalar; Hirschmann–Eardley κ-family; axion–dilaton; charged and massive scalar in their small-scale limit | none | none: s-translation not obviously broken |
| Scalar ℓ-multiplet | $`\ell(\ell+n-1)\,\psi/r^2`$ | $`\ell\,\psi/(2s)`$ |
| Hedgehog wave map or σ-model (target $`S^{D-1}`$) | $`n\sin(2u)/(2r^2)`$ | $`\sin(2u)/(4s)`$ |
| Magnetic Yang–Mills | $`(n-1)\,w(1-w^2)/r^2`$ | $`w(1-w^2)/(2s)`$ |
| Yang–Mills linearized about vacuum (the EYM small-scale system, §4.1) | $`2(n-1)\,\delta/r^2`$ | $`\delta/s`$ |
| Vacuum squashing mode (Bianchi IX type), linearized | $`2(n+1)\,B/r^2`$ (D = 5 and 9) | $`B/s`$ |
| Perfect fluid | none (a first-order system with (D − 2)/r geometric terms) | the large-D structure differs (§7.1) |

## 2. Summary table

All values are for D = 4 unless stated. "DSS" means discretely and "CSS" continuously self-similar.

| Model | Critical solution | Δ | γ | Other D (published) | Flat-space self-similar blowup | Large-D work |
|---|---|---|---|---|---|---|
| Real scalar (reference) | DSS | 3.445452402(3) | 0.374 | many; see README | none (linear) | Rozali–Way; Ecker et al.; this repository |
| Scalar ℓ-multiplet [OVCU07] | DSS for ℓ ≤ 9.5 | 0.460 (ℓ = 1) … 0.0015 (ℓ = 6) | 0.119 (ℓ = 1) … 0.0013 (ℓ = 6) | not found | none (linear) | not found |
| Complex scalar, m = 1, axisymmetric [CHLP04] | DSS, non-spherical | ≈ 0.42 | ≈ 0.11 | not found (2+1 with Λ below 0: [JG17]) | none | not found |
| SU(2) Einstein–Yang–Mills, magnetic [CCB96, G97] | type I (Bartnik–McKinnon) and type II, asymptotically DSS | 0.73784 | 0.1964 | not found with gravity | none in D = 4; D = 5 non-self-similar; D ≥ 6 CSS | not found |
| Self-gravitating S³ σ-model, coupling η [HLPTA00, LTHA02] | CSS for η from about 0.07 to 0.17, DSS above | diverges as η → 0.17⁺ | ≈ 0.18 (CSS), ≈ 0.11 (DSS), 0.1185 as η → ∞ | not found | D ≥ 4 (CSS) | not found |
| Hirschmann–Eardley κ-family [HE97, LC96] | DSS (Choptuik) for small κ; phase-rotating CSS for κ ≳ 0.075 | — | — | via axion–dilaton | — | not found |
| Axion–dilaton (κ = 1) [EHH95, HHS96] | phase-rotating CSS | — | 0.264 | γ for D = 4–10 [AGH12] | — | not found |
| Vacuum, biaxial Bianchi IX [BCS05, BCRST05] | DSS in D = 5 and 9 | 0.47 (D = 5), 0.78 (D = 9) | 0.164 (D = 5), 0.273 (D = 9) | D = 5, 9 only | none | not found |
| Vacuum, 4D axisymmetric [AE93, B+23] | contested | ≈ 0.6 (1993) | ≈ 0.36 (1993) | — | — | — |
| Perfect fluid p = kρ [EC94, M96, NC00] | CSS | — | 0.3558 (k = 1/3); depends on k | D = 5–7 [AG+09] | not surveyed | not found |
| Charged scalar [GMG96, MGM26] | the real DSS plus a charge mode | 3.445 | 0.374; charge exponent ≈ 0.88 | D = 5 (2026) | — | not found |
| Massive scalar [BCG97] | type I, or type II = the massless DSS | — | — | — | — | the mass drops out at leading order in 1/D [RW18] |

## 3. Scalar fields with angular momentum

### 3.1 The ℓ-multiplet in spherical symmetry

**Model** [OVCU07]. Take 2ℓ + 1 real massless scalars $`\Psi_\ell^m = \psi^{(\ell)}(t,r)\,Q_{\ell m}(\theta,\phi)`$, where
the $`Q_{\ell m}`$ are real spherical harmonics. The summed stress tensor is spherically symmetric, so the metric
$`ds^2 = -\alpha^2dt^2 + a^2dr^2 + r^2d\Omega^2`$ stays spherical. With Φ = ψ′ and Π = (a/α)ψ̇, the matter equations
(their Eqs. (15)–(16)) are the massless wave equation plus the centrifugal term $`-\ell(\ell+1)\,a\alpha\,\psi/r^2`$ in
the equation for $`\partial_t\Pi`$; this term is the only place ℓ enters the wave equation. The Hamiltonian
constraint, their Eq. (17) in the paper's units, is

```math
\frac{a_r}{a} = \frac{2\ell+1}{2}\,r\Big(\Pi^2 + \Phi^2 + \ell(\ell+1)\frac{a^2}{r^2}\psi^2\Big) - \frac{a^2-1}{2r} .
```

*In D dimensions (remark).* The harmonics on $`S^n`$ have eigenvalue ℓ(ℓ + n − 1), so the matter equation becomes

```math
\partial_t\Phi = \partial_r\Big(\frac{\alpha}{a}\Pi\Big), \qquad
\partial_t\Pi = \frac{1}{r^{n}}\,\partial_r\Big(r^{n}\frac{\alpha}{a}\Phi\Big) - \ell(\ell+n-1)\,\frac{a\alpha}{r^2}\,\psi .
```

The constraints acquire the angular gradient energy $`\ell(\ell+n-1)\psi^2/r^2`$, with a multiplicity factor that can
be absorbed into the amplitude. No D-dimensional study of the ℓ-multiplet was found.

**Results in D = 4.** The critical solutions are DSS for every ℓ studied up to 9.5 (non-integer ℓ is allowed in the
radial equations). At ℓ = 10 the authors could not tell DSS with very small Δ from a periodic solution. Selected
values from their Table III:

| ℓ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Δ_ℓ | 3.43 ± 0.05 | 0.460 ± 0.002 | 0.119 ± 0.003 | 0.039 ± 0.001 | 0.0132 ± 0.0008 | 0.0044 ± 0.0007 | 0.0015 ± 0.0005 |
| γ_ℓ | 0.376 ± 0.003 | 0.119 ± 0.001 | 0.0453 ± 0.0002 | 0.020 ± 0.001 | 0.0082 ± 0.0008 | 0.0033 ± 0.0005 | 0.0013 ± 0.0005 |

*Our reading of that table:* for 2 ≤ ℓ ≤ 6, ln Δ_ℓ falls by about 1.1 per unit of ℓ and ln γ_ℓ by about 0.9, so
both fall roughly exponentially. [OVCU07] note that their ℓ = 1 solution "is apparently the same solution" as the
self-gravitating SO(3) σ-model in the hedgehog ansatz (§5.1). They also note that Δ₁ and γ₁ agree well with the m = 1
complex scalar of [CHLP04] (§3.2), although that solution is not spherical.

**Flat space (remark).** With $`\psi = r^\ell g`$,

```math
\Big(\partial_r^2 + \frac{n}{r}\partial_r - \frac{\ell(\ell+n-1)}{r^2}\Big)(r^\ell g) = r^\ell\Big(\partial_r^2 + \frac{n+2\ell}{r}\partial_r\Big)g ,
```

so without gravity the ℓ-multiplet in D dimensions is the radial field in D + 2ℓ dimensions. This contrasts with the
gravitating data. For the radial scalar, Δ(D) falls to the finite Δ∞ = 1.644, while in D = 4 Δ_ℓ appears to fall to 0
as ℓ grows. Gravity therefore treats the angular gradient energy differently from extra dimensions, and the
ℓ-multiplet is not the radial scalar in disguise. (The "Δ → 0" conjecture of Ecker et al. for D looks more like what
happens in ℓ.)

**Large D (remark).** In the inner variables the flat operator gives $`\partial_t^2\psi = \partial_s\psi - \ell\,\psi/(2s)`$.
The substitution $`\psi = s^{\ell/2} g`$ removes the 1/s term from this operator ($`\partial_t^2 g = \partial_s g`$), but then the
metric equations carry explicit factors of $`s^\ell`$. Whether the s-translation, and with it the exact λ = 2 mode,
survives is open. No flat-space self-similar blowup exists, because the equation is linear.

### 3.2 Complex scalar with angular momentum

**Model** [CHLP04]. The model is 3+1 axisymmetric with $`\Psi = \Phi(\rho,z,t)\,e^{im\varphi}`$, where m is an integer
and regularity on the axis requires $`\Phi \to \rho^m f(z,t)`$. The stress tensor is φ-independent but carries net
angular momentum. Only m = 1 was studied.

**Results.** The critical solution is a new, non-spherical DSS solution with Δ ≈ 0.42 (±4%) and γ ≈ 0.11 (±10%).
It appears universal up to a family-dependent constant phase. The authors mention possible evidence for a
non-spherical unstable mode, which would change the picture. [JG17] study a rotating scalar in 2+1 dimensions
($`\Psi = e^{im\theta}\Phi`$, with Λ below 0 needed to form black holes). For m > 0 they find only weak evidence for a
self-similar critical solution, and no universality. No version in D ≥ 5 (for example in odd D, with equal angular
momenta) was found.

### 3.3 Angular-momentum perturbations of the spherical solution

[GGM99] compute the black-hole angular momentum from the ℓ = 1 axial perturbation of the spherical D = 4 DSS solution
at second order. They find $`J \propto (p-p_*)^\mu`$ with μ ≃ 0.76, using a complex eigenvalue from an earlier
first-order analysis. [CHLP04] quote Re λ′ ≈ −0.017 for this mode, so it decays slowly. These perturbative results are
D = 4 only, and no D-dimensional version was found.

## 4. SU(2) Yang–Mills

### 4.1 Einstein–Yang–Mills in D = 4

**Equations** [CCB96, G97]. The magnetic ansatz uses one potential W(t, r) with vacua W = ±1. With Φ = W′ and
Π = (a/α)Ẇ, in units that set the coupling to one,

```math
r\,\partial_t\Phi = r\,\partial_r\Big(\frac{\alpha}{a}\Pi\Big), \qquad
r\,\partial_t\Pi = r\,\partial_r\Big(\frac{\alpha}{a}\Phi\Big) + \frac{a\alpha}{r}\,W(1-W^2),
```
```math
r\frac{a_r}{a} = \frac{1-a^2}{2} + \Phi^2 + \Pi^2 + \frac{a^2}{2r^2}(1-W^2)^2, \qquad
r\frac{\alpha_r}{\alpha} = \frac{a^2-1}{2} + \Phi^2 + \Pi^2 - \frac{a^2}{2r^2}(1-W^2)^2 .
```

**Results.**
- [CCB96] found two critical solutions. Type I is the n = 1 Bartnik–McKinnon soliton, with a mass gap. Type II is
  DSS, with Δ ≈ 0.74 and γ ≈ 0.20.
- [G97] constructed the type II solution and its single unstable mode: Δ = 0.73784 ± 0.00002 and
  γ = 0.1964 ± 0.0007.
- [CHM99] found a third kind of threshold between the two black-hole outcomes, controlled by a colored black hole with
  one unstable mode.
- [MR18] studied the general (non-magnetic) spherical SU(2) system. There type I disappears and the threshold is always
  type II, but the critical solution differs from the magnetic DSS one, and universality may be lost. In the magnetic
  sector they measure γ = 0.20018 ± 0.00017.

**Why the type II solution is only asymptotically self-similar.** The coupling G/g² is a length squared, so EYM is not
scale invariant. *(Remark: this holds in every D, since G/g² has dimension length² in all D.)* [G97] states that "the
presence of a scale in the field equations excludes the existence of an exactly self-similar solution". He writes
W = 1 − rS, after which

```math
\frac{W(1-W^2)}{r} = (1-rS)(2-rS)\,S, \qquad \frac{(1-W^2)^2}{r^2} = (2-rS)^2S^2 .
```

The terms with explicit positive powers of r "become negligible on small spacetime scales", and the critical
solution is self-similar for the r-free remainder.

*Our reading:* that remainder keeps 2S and 4S², so the leading-order system is Einstein gravity coupled to Yang–Mills
*linearized about the vacuum* W = 1, with a quadratic stress tensor. The Yang–Mills nonlinearity is gone and only the
coupling to gravity is nonlinear. In D = 4 Yang–Mills is energy-subcritical and globally regular in flat space (the
2025 review [GHMG25]), so this DSS solution has no flat-space counterpart.

### 4.2 Yang–Mills in higher D, without gravity

**Equation** [BT01]. The SO(d)-equivariant ansatz in d space dimensions is
$`A^{ij}_\alpha = (\delta^i_\alpha x^j - \delta^j_\alpha x^i)\,(1-w)/r^2`$, which gives

```math
\partial_t^2 w = \partial_r^2 w + \frac{d-3}{r}\,\partial_r w + \frac{d-2}{r^2}\,w(1-w^2),
```

that is, (D − 4)/r and (D − 3)/r² = (n − 1)/r².

**Blowup.**
- *d = 3 (D = 4):* energy-subcritical; no blowup.
- *d = 4 (D = 5):* energy-critical. Blowup proceeds by adiabatic shrinking of the static instanton
  $`W_S = (1-\eta^2)/(1+\eta^2)`$ and is not self-similar [BT01].
- *d ≥ 5 (D ≥ 6):* energy-supercritical, with exact self-similar (CSS) blowup.
  - [CST98] proved existence for some d; [BT01] cite d = 5, 7, 9.
  - In d = 5 the generic blowup profile is $`W_0 = (1-\eta^2)/(1+\tfrac35\eta^2)`$, η = r/(T − t). The threshold
    between blowup and dispersal is the excited solution W₁, with one unstable mode of eigenvalue 5 [BT01].

**The ground state in every d.** [BB15] give it in closed form (their Yang–Mills equation uses u = 1 − w, and their d
is the present d − 2). In the form of [Gl23]:

```math
w = 1 - \frac{\alpha(d)\,\rho^2}{\rho^2+\beta(d)}, \qquad \rho = \frac{r}{T-t}, \qquad
\alpha(d) = 2\Big(1+\sqrt{\tfrac{d-4}{3(d-2)}}\Big), \qquad \beta(d) = \tfrac13\big(2d-8+\sqrt{3(d-2)(d-4)}\big).
```

This reduces to the d = 5 profile above.
- *Stability:* in d = 5, nonlinear stability given mode stability, which was checked numerically [D14]; stability in
  the whole space for odd supercritical d [DO23]; global-in-space stability for all d ≥ 5, new for even d [Gl23].
- *Heat flow:* with the same nonlinearity, [BB15] note that self-similar blowup exists only for 3 ≤ d ≤ 7, in their
  convention.

**Large d (remark, checked symbolically).** β(d) ≈ 1.244 d, so the profile varies on $`\rho \sim \sqrt d`$, the same
inner scale as the DSS scalar. With s = r²/(2(d − 3)) and t unscaled, the equation becomes

```math
\partial_t^2 w = \partial_s w + \frac{w(1-w^2)}{2s} + O(1/d), \qquad
w_\infty = 1 - \frac{2\alpha_\infty\,s}{2s + c_\infty\,t^2}, \qquad \alpha_\infty = 2\Big(1+\frac{1}{\sqrt3}\Big),\ c_\infty = \frac{2+\sqrt3}{3},
```

and $`w_\infty`$ solves the leading-order equation exactly. The 1/s potential is nonlinear and cannot be removed.
Linearized about the vacuum (w = 1 − δ), the equation has the potential $`2(n-1)\delta/r^2`$. In flat space, δ = r²g
turns it into the radial wave equation in D + 2 dimensions, as for an ℓ = 1 field.

### 4.3 Gravity and Yang–Mills in D ≥ 5

No dynamical study of Einstein–Yang–Mills collapse in D ≥ 5 was found. The 2025 review [GHMG25] notes the exact CSS
solutions of pure Yang–Mills in 5+1 and higher.

*Remark (dimensional argument).* Because G/g² is a length², gravity grows stronger relative to the Yang–Mills field
as a configuration shrinks. A flat-space self-similar blowup with O(1) amplitude can therefore not persist to
arbitrarily small scales once gravity is on: either a black hole forms first, or the field approaches the vacuum (the
mechanism of [G97]). So in D ≥ 6 the type II threshold should again be controlled by the linearized-about-vacuum
system, which belongs to the 1/s class of §1.

## 5. σ-models and wave maps

### 5.1 The self-gravitating S³ σ-model: a coupling that switches DSS and CSS

**Model** [BW00, HLPTA00]. Corotational wave maps into S³, coupled to gravity through the dimensionless coupling
$`\eta = \alpha = 4\pi G f_\pi^2`$. [HLPTA00] note that η can be read as the inverse curvature of the target. In flat
d-dimensional space the reduced equation is

```math
\partial_t^2 u = \partial_r^2 u + \frac{d-1}{r}\,\partial_r u - \frac{d-1}{2r^2}\,\sin 2u .
```

*(Remark.)* With gravity, in D dimensions with target $`S^{D-1}`$, the covariant form is
$`\Box_g u = (D-2)\sin(2u)/(2r^2)`$, with r the areal radius. Since G f² is dimensionless in every D, the model is
scale invariant in every D.

**Results in D = 4.**
- [BW00]: for 0 ≤ α below 1/2 there is a countable family of CSS solutions. The n-th one is regular for α below α_n,
  with α₀ ≈ 0.0688 and α₁ ≈ 0.1518.
  - For α below α₀ the n = 0 solution is a generic naked singularity, inherited from flat space.
  - Type II CSS criticality via n = 1 is expected only for α between α₀ and α₁.
- [HLPTA00]: DSS critical collapse for η from about 0.18–0.2 up to 100, with Δ rising sharply toward the lower end. γ
  is "almost independent of η, asymptoting to 0.1185 ± 0.0005 at large η".
- [LTHA02]: the DSS and CSS solutions merge at η_c ≈ 0.17, with
  $`\Delta = -\tfrac{2}{\lambda}\ln(\eta-\eta_c) + {\rm const}`$, where λ ≈ 5.14 is the CSS unstable eigenvalue. They
  describe it as a heteroclinic-loop bifurcation; the 2025 review calls it Shil'nikov-type.
  - γ_CSS ≈ 0.18 for 0.1 ≤ η ≲ 0.14, and γ_DSS ≈ 0.11 for η ≳ 0.2.
- [BSW04] show, by mixed numerical and analytical methods, that an unstable periodic (DSS) self-similar solution of
  this model exists.

*Remark.* As η → ∞ the target flattens and the hedgehog becomes a triplet of free scalars $`\phi^i = f(r)\,x^i/r`$,
which is exactly the ℓ = 1 multiplet of §3.1. This is consistent with γ → 0.1185 ± 0.0005 [HLPTA00] against
γ₁ = 0.119 ± 0.001 [OVCU07], and with [OVCU07]'s statement that the two solutions are the same. In D dimensions the
η → ∞ limit is likewise the ℓ = 1 multiplet of D − 1 scalars.

**How flat-space blowup interacts with the black-hole threshold.** In D = 4:
- At small η the flat-space generic blowup (the n = 0 CSS solution) survives as a naked singularity, and the threshold
  is the flat-space threshold solution n = 1, turned CSS-critical.
- At larger η the CSS solutions acquire horizons or disappear (all are gone at η = 1/2), and a DSS solution takes over.

How η_c, α₀ and α₁ move with D: not found.

### 5.2 Two-component σ-models: the Hirschmann–Eardley κ-family and Brans–Dicke

**Model** [HE97]: $`S = \int d^4x\sqrt{-g}\,\big(R - 2\lvert\nabla F\rvert^2/(1-\kappa\lvert F\rvert^2)^2\big)`$, with F complex
and a target of constant curvature ∝ −κ. Special cases: κ = 0 is the free complex scalar and κ = 1 is the axion–dilaton
system. For κ ≥ 0 the model is equivalent to Brans–Dicke gravity with $`\omega_{\rm BD} = -\tfrac32 + 1/(8\kappa)`$.

**Results.**
- [HE97] find a family of CSS solutions whose stability changes at κ ≈ 0.0754: the free complex scalar lies on the
  unstable side and the axion–dilaton on the stable side.
- The 2025 review [GHMG25] summarizes the picture as CSS criticality for κ > 0.0754, with the Choptuik DSS solution
  taking over below.
- [LC96] observed this switch in evolutions of the Brans–Dicke model, at λ = 2/(2ω_BD + 3) ≈ 2/3, which is κ ≈ 0.083
  under the map above *(remark: λ = 8κ)*. Their Table 1 lists Δ = 3.447 (at λ = 0.57 and 0.67) and γ between 0.364
  and 0.379 for λ = 0.001–1.0.

Because the fields are spherically symmetric scalars, there is no 1/r² term (§1). D-dimensional versions exist only
for κ = 1 (§7.3).

### 5.3 Flat-space wave maps in d ≥ 3

- **Existence of blowup.** In d = 3 there is self-similar blowup [Sh88] and a countable family of CSS solutions [Bi00].
  - Generic blowup goes to the ground state (the Turok–Spergel solution), and the threshold of blowup is the n = 1
    excited solution [BCT00, LHI00].
  - [LHI00] give its single unstable eigenvalue as about 6.33 (λ = 1 is the gauge mode).

**All d ≥ 3** [BB15]. The ground state is explicit in every d, and it is the generic blowup profile:

```math
u_0 = 2\arctan\frac{r}{\sqrt{d-2}\,(T-t)} .
```

- **The threshold** [BBM17]. For 3 ≤ d ≤ 6 there is an infinite sequence of CSS solutions, and the threshold is the
  stable manifold of the one with one instability. For d ≥ 7 they conjecture that only u₀ exists, and say that "the
  threshold dynamics is qualitatively different and will be described elsewhere" (not found).
- **Rigorous stability.** Mode stability of the ground state in higher d [CDG17]. Stable self-similar blowup is also
  known for specially constructed negatively curved targets in d ≥ 8, proved for d = 9 [DG19].
- **A flat-space DSS example** [GHW25]. Wave maps into S¹ with a null-form nonlinearity, $`\Box\theta + (\partial\theta)^2 = 0`$,
  have a countable family of stable DSS blowup solutions in every d, with explicit polynomial profiles and period 2π/k
  for the k-th member.
  - *(Remark.)* With $`v = e^\theta`$ the equation is □v = 0, so the profiles are homogeneous polynomial solutions of
    the free wave equation. The period is fixed by the polynomial (analytic) termination at the light cone and does
    not depend on d. This is a toy version of period selection by light-cone analyticity, but it has no gravity and no
    D-dependence.

**Large d (remark, checked symbolically).** The ground state varies on $`\rho \sim \sqrt d`$. With s = r²/(2(d − 1)),

```math
\partial_t^2 u = \partial_s u - \frac{\sin 2u}{4s} + O(1/d), \qquad u_\infty = 2\arctan\frac{\sqrt{2s}}{T-t}\ \ \text{(exact at leading order)} .
```

So the flat-space blowup of wave maps has an n-free inner problem at x ~ √n, like the gravitating scalar, but with a
nonlinear 1/s potential that cannot be removed. No large-d study of wave-map or Yang–Mills blowup was found.

## 6. Vacuum collapse

### 6.1 Cohomogeneity-two Bianchi IX models in odd D

**Five dimensions** [BCS05]. The metric is

```math
ds^2 = -Ae^{-2\delta}dt^2 + A^{-1}dr^2 + \tfrac14 r^2\big[e^{2B}(\sigma_1^2+\sigma_2^2) + e^{-4B}\sigma_3^2\big],
```

with σ_k the invariant one-forms on S³ and A = 1 − m/r². The field equations, as written in [BS06], are

```math
m' = 2r^3\big(e^{2\delta}A^{-1}\dot B^2 + AB'^2\big) + \tfrac23 r\big(3 + e^{-8B} - 4e^{-2B}\big), \qquad
\delta' = -2r\big(e^{2\delta}A^{-2}\dot B^2 + B'^2\big),
```
```math
\big(e^{\delta}A^{-1}r^3\dot B\big)^{\cdot} - \big(e^{-\delta}Ar^3B'\big)' + \tfrac43\,e^{-\delta}\,r\,\big(e^{-2B}-e^{-8B}\big) = 0 .
```

**Results in D = 5.**
- Type II DSS critical behaviour with Δ ≈ 0.47 (the 2025 review quotes 0.49) and mass exponent 0.3289, that is a radius
  exponent of 0.164 [BCS05]. [PVG22] give Δ_B ≈ 0.47 and γ_B ≈ 0.164.
- There are no regular CSS solutions [BW10].
- A codimension-two DSS solution with Δ₂ ≈ 0.395 [BCS06].
- [PVG22] add a scalar field. They note that the wave field obeys an equation with the principal part of a spherical
  wave equation in 9+1 dimensions (as for an ℓ = 2 mode, §3.1), and find a bi-critical solution between the two fields.

**Nine dimensions** [BCRST05]. A squashed S⁷ with SO(8) broken to SO(5) × SU(2). The squashing equation is
$`(e^\delta A^{-1}r^7\dot B)^{\cdot} - (e^{-\delta}Ar^7B')' + \tfrac47 e^{-\delta}r^5(6e^{-3B} - e^{4B} - 5e^{-10B}) = 0`$.
- DSS with Δ ≈ 0.78 and mass exponent ≈ 1.64; since mass ∝ r⁶, the radius exponent is ≈ 0.273.
- The product Δλ ≈ 2.85, with λ the growth rate, is "approximately the same" as in five dimensions. (Check: 0.47/0.164
  = 2.87 and 0.78/0.273 = 2.86. For the scalar, Δ/γ is 9.21 at D = 4 and 3.29 at D = ∞, so this near-constancy is
  specific to the vacuum model.)
- The mass function is not monotonic in 9D.

**Other odd D.** [BS06] point out that the construction extends to all odd D, for example with SU(k + 1) acting on
$`S^{2k+1}`$. No collapse study in D = 7, 11, … was found.

*Remark.* Linearized about B = 0, the potential is 8B/r² in D = 5 and 16B/r² in D = 9, that is 2(D − 1)/r² =
ℓ(ℓ + n − 1)/r² with ℓ = 2 (the 9D paper identifies the ℓ = 2 tensor mode of the Ishibashi–Kodama master equation).
This is 1/s at D = ∞, so the vacuum DSS solutions are in the 1/s class. There is no flat-space analogue. In both
studied dimensions Δ and γ rise from D = 5 to D = 9, the opposite of the scalar's Δ(D).

### 6.2 Four-dimensional axisymmetric vacuum collapse

- [AE93] reported γ ≈ 0.36 and tentative echoing with Δ ≈ 0.6 for polar gravitational waves. This was never
  reproduced in its original setting [R25].
- [LK21] report universal, irregular approximate echoes in curvature invariants.
- A three-code comparison [B+23] finds no universality, with approximate self-similarity for some families of initial
  data only.
- Electromagnetic waves in axisymmetry [BGH19]: approximate DSS with Δ ≃ 0.55 and γ ≃ 0.145, without exact DSS or
  universality.

None of this has a D-family.

## 7. Perfect fluid, complex-scalar CSS, axion–dilaton

### 7.1 Perfect fluid p = kρ

**Equations.** $`\nabla_a T^{ab} = 0`$ with $`T^{ab} = (\rho + p)u^a u^b + p\,g^{ab}`$. In spherical symmetry in D dimensions the
divergence adds geometric terms ∝ (D − 2)/r times the fluxes, and there is no 1/r² potential. [AG+09] (their
Eq. (2.22)) write the self-similar ODEs in d dimensions.

**Results in D = 4.** The critical solution is CSS: radiation fluid with γ ≈ 0.36 [EC94], and γ = 0.3558 from
perturbation theory [KHA95]. γ depends on k [M96]:

| k | 0.01 | 0.10 | 0.30 | 1/3 | 0.50 | 0.80 | 0.888 |
|---|---|---|---|---|---|---|---|
| γ | 0.1143 | 0.1875 | 0.3322 | 0.3558 | 0.4774 | 0.7294 | 0.8157 |

[NC00] extend the range to stiff fluids (Γ = 1 + k up to 2) and find CSS throughout, with evidence that γ → 1 as
Γ → 2. The general-k perturbation analysis is [KHA99].

**Higher D** [AG+09]. γ(d, k) for d = 4–7 and k ≤ 0.25 (±0.001; their table):

| k | d = 4 | d = 5 | d = 6 | d = 7 |
|---|---|---|---|---|
| 0.01 | 0.114 | 0.225 | 0.290 | 0.330 |
| 0.05 | 0.148 | 0.256 | 0.316 | 0.353 |
| 0.10 | 0.187 | 0.291 | 0.346 | 0.380 |
| 0.15 | 0.225 | 0.324 | 0.375 | 0.406 |
| 0.20 | 0.261 | 0.356 | 0.403 | 0.431 |
| 0.25 | 0.297 | 0.387 | 0.430 | 0.454 |

They give γ = 0.388 for the conformal fluid in d = 5 (k = 1/(d − 1) = 1/4). At fixed k, γ rises with d. γ for the
conformal fluid is 0.356, 0.388 and 0.403 at d = 4, 5, 6, and about 0.414 at d = 7 (interpolated). It is not known
whether either sequence tends to 1/2. [AG+09] do not discuss large d.

*Remark.* The fluid's CSS solution is selected by regularity at the sonic point, not by a Stokes condition, and the
conformal fluid's k = 1/(D − 1) tends to 0. The large-D structure is therefore not the scalar's. Flat-space
self-similar blowup of the Euler equations was not surveyed.

### 7.2 The Hirschmann–Eardley complex CSS solution

[HE95a] find a CSS solution of the free complex scalar in D = 4 with the phase rotating in log-time:

```math
\phi(t,r) = (-t)^{i\omega} f\big(r/(-t)\big), \qquad \omega = 1.9154446 \pm 10^{-7}, \qquad e^{\Delta} = e^{2\pi/\omega} = 26.583086,
```

that is, Δ = 3.2803. The metric functions depend on r/(−t) only.

[HE95b] find γ ≈ 0.387111 from its real growing mode, but also a growing oscillatory mode, so for κ = 0 it is not the
critical solution (Choptuik's DSS is). For κ > 0.0754 the same kind of solution is critical (§5.2).

*Remark.* The field is exactly one harmonic in θ = ωτ and the metric is exactly θ-independent. So, unlike the
real-field DSS, this is an ODE eigenvalue problem for ω, and its D = ∞ inner problem should close exactly with one
harmonic.

### 7.3 Axion–dilaton (κ = 1)

**D = 4.** CSS (phase-rotating) critical solution with γ = 0.264 [EHH95, HHS96]; [HE97] quote 0.2641066.

**Higher D.** [AGH12] compute γ for the "elliptic" (phase-rotating) class in d = 4–10. They warn that the third digit
is not reliable, especially in d = 9 and 10.

| d | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|
| γ | 0.261 | 0.278 | 0.287 | 0.311 | 0.352 | 0.373 | 0.347 |

- [AH20a] give a numerical method for arbitrary D and new families in D = 4, 5.
- [HV20] find several new families in D = 6 and 7: four distinct elliptic-class solutions in 6D and three in 7D.
- [AH20b, HK21] conclude that the exponent depends on D, on the conjugacy class (elliptic, parabolic, hyperbolic) and
  on the solution within a class.

No large-D work was found. The rise from 0.261 to about 0.37 over d = 4–9 is compatible with, but far from showing, a
limit of 1/2.

## 8. Charged and massive scalar

**Charged scalar.** With $`D_a = \nabla_a - iqA_a`$ and the Maxwell field, the coupling q is irrelevant on small scales
([MGM26]: in similarity coordinates q always appears as $`e^{-\tau}q`$). The critical solution is the real DSS solution,
real up to a constant phase.
- D = 4: [HP97]; charge exponent δ ≈ 0.88 [GMG96].
- D = 5 [MGM26]: γ ≃ 0.41325, Δ ≃ 3.224, imaginary-perturbation rate μ = −0.448, and charge exponent δ_Q ≃ 1.425.
  They state the general-D relation $`\delta_Q = (D-2-\mu)\gamma`$. Universality of the black-hole charge breaks down
  at large q.

*Remark.* μ(D) is a second growth rate of the *same* DSS background: the rate of a test (imaginary) field on it, with
no metric perturbation at linear order. Like λ, it is selected beyond all orders at large D, but no symmetry is
apparent that would fix μ∞. It can be computed with the existing solver, without a new background.

**Massive scalar.** [BCG97] find type I (oscillating soliton stars) and type II critical behaviour. In type II the mass
is irrelevant at small scales and the critical solution is the massless DSS solution. At large D, [RW18] note that the
mass drops out of the leading-order equations.

## 9. Large-D work on collapse

- **[RW18]** Rozali and Way, arXiv title "Gravitating scalars and critical collapse in the large D limit" (published as
  "Gravitating scalar stars in the large D limit").
  - Massless real scalar with rescaled time √D t. At leading order the probe equation is $`\partial_\tau^2\phi = r^{-1}\partial_r\phi`$
    (parabolic, with the roles of space and time swapped; this is the s-heat equation of GAMMA.md).
  - An exact "oscillaton" family whose metric function diverges where $`\varepsilon^2e^{-r^2} = 1`$ gives γ = 1/2.
  - The mass drops out at leading order. No other matter and no angular momentum.
- **[CP25]** Clark and Pimentel. CSS solutions of the massless scalar at large D by matched asymptotics, in three
  regions. They obtain γ_q = (D − 3)/√(2(D − 2)) for CSS-preserving perturbations and 2(D − 3)/(D − 2) for CSS-breaking
  ones, with critical dimensions D = 6 and 10 in the spectrum. They mention perfect fluids as a future extension. See
  GAMMA.md for the relation to the DSS problem.
- **[EEG26], [EEGJ26]** Ecker, Ecker, Grumiller (and Jechtl). Real scalar only: large-D DSS at leading order, and
  "critical spacetime crystals" in continuous 3.05 ≤ D ≤ 5.5, with the conjecture Δ → 0 (README).
- **[EST13], [EH20]** Emparan, Suzuki and Tanabe introduced the large-D limit of GR; Emparan and Herzog review it.
  These concern black-hole dynamics (effective membranes). No paper by Emparan and collaborators on critical collapse
  was found.
- **[K05]** Kol: an analytic picture of critical collapse and the black-string merger. The double-cone exponents become
  real for D ≥ 10, which is tangential here.
- **[SH96]** Soda and Hirata: CSS (Roberts-type) scalar collapse in higher D (journal reference not verified).
- **[GHMG25]** The 2025 review covers D-dependence for the scalar, Einstein–Gauss–Bonnet, Kaluza–Klein, Bianchi IX in
  4+1 and 8+1, and σ-models. The parts we could read do not discuss large D.

**Not found:** any large-D study of the ℓ-multiplet, the complex scalar, Yang–Mills (with or without gravity),
σ-models or wave maps (with or without gravity), axion–dilaton, perfect fluids, or vacuum collapse.

## 10. Recommendations

Two things are to be tested: **(a)** a finite Δ∞ (or, for a phase-rotating CSS solution, a finite frequency ω∞)
selected by an inner Stokes condition, and **(b)** γ∞ = 1/2 against a different limit.

The models split into two classes (§1):
- **No O(n)/r² term:** spherically symmetric scalars with any target geometry, and the charged and massive scalar in
  their small-scale limit. The s-translation is not obviously broken. If their inner problems are autonomous like the
  scalar's (not derived), the mechanism of GAMMA.md predicts γ∞ = 1/2 for any self-similar critical solution selected
  in the inner region.
- **An O(n)/r² term:** ℓ-multiplets, hedgehog σ-models, Yang–Mills, and vacuum squashing modes. These give an O(1)
  1/s term at D = ∞, and γ∞ = 1/2 is not expected from that argument.

The best tests:

1. **The phase-rotating CSS solution of a complex scalar: axion–dilaton, or the Hirschmann–Eardley κ-family, in D
   dimensions.**
   - *For (a):* ω(D) is selected by an ODE eigenvalue problem. At D = ∞ the inner problem should close exactly with
     one harmonic, so the truncation problem of the real field (README: "harmonic truncation is not a controlled
     approximation") disappears. A finite ω∞ from a single Stokes multiplier would be the cleanest test of the selection
     hypothesis, and it is cheap: ODEs, not PDEs.
   - *For (b):* there is no 1/r² term, so γ∞ = 1/2 is predicted. For the axion–dilaton (and κ > 0.075 in D = 4) this
     is the actual critical solution, and published γ(d) exists up to d = 10 [AGH12] for comparison.
   - *Caveats:* for κ = 0 the solution has an extra unstable mode. Whether the target curvature κ survives at D = ∞
     depends on how κ is scaled with n, which has not been worked out. The inner equations for the complex field
     (expected: the README's inner equations with $`y = e^{i\theta}Y(\xi)`$ and y² replaced by $`\lvert y\rvert^2`$ in the gravitational terms)
     have not been derived.

2. **The scalar ℓ-multiplet in D dimensions.**
   - *Cost:* the smallest change to the existing solver: one term in the wave equation (§3.1), the angular energy in
     the constraints, and $`\psi \propto r^\ell`$ at the centre. The critical solution is DSS for every ℓ studied, and ℓ can be varied
     continuously.
   - *For (b):* the inner term ℓψ/(2s) is a direct test of whether the 1/s class gives γ∞ ≠ 1/2. A finite Δ∞(ℓ) that
     differs from 1.644 would show that the inner Stokes selection is generic and not tied to the radial scalar.
   - *A sharp question:* in D = 4, Δ_ℓ → 0 as ℓ grows, while Δ(D) → 1.644 for ℓ = 0. Which limit wins in Δ∞(ℓ)?
   - ℓ = 1 is also the η → ∞ limit of the σ-model, which connects this test to candidate 3.

3. **The hedgehog σ-model (target $`S^{D-1}`$, dimensionless coupling η) in D dimensions.**
   - It is the only candidate with a flat-space self-similar blowup in every D ≥ 4. Its large-D inner profile is
     explicit, $`u_\infty = 2\arctan(\sqrt{2s}/(T-t))`$ (§5.3). It has a coupling that switches DSS and CSS in D = 4,
     and it reduces to candidate 2 as η → ∞.
   - It tests (a) and (b) together with a question no other model can answer: how the flat-space blowup and the
     black-hole threshold trade places as D grows (η_c(D), α₀(D)).
   - It costs a new code, so it comes third.

   *Alternative third candidate:* Bianchi-IX-type vacuum collapse in odd D. It is pure gravity with one function, and
   Δ and γ both rise from D = 5 to 9 with Δ/γ ≈ 2.86 in both. But the general-odd-D equations (for example the SU(k + 1)
   squashing of $`S^{2k+1}`$) are not in the literature and would have to be derived first.

**Cheap side test:** the charge exponent's μ(D) on the existing DSS background (§8). This is a second beyond-all-orders
growth rate, with no symmetry apparent behind it.

**Not recommended:**
- Perfect fluid: CSS selected at a sonic point, with a different large-D structure.
- Einstein–Yang–Mills: its small-scale system is linearized Yang–Mills, another 1/s-class field like candidate 2, with
  a dimensionful coupling on top.
- 4D axisymmetric vacuum: no D-family, and the threshold is contested.
- Massive and charged scalar: the same background as the real scalar.

## 11. References

Journal reference and arXiv number checked unless noted.

*Scalar fields with angular momentum*
- [OVCU07] I. Olabarrieta, J. F. Ventrella, M. W. Choptuik, W. G. Unruh, PRD 76, 124014 (2007), [arXiv:0708.0513](https://arxiv.org/abs/0708.0513).
- [CHLP04] M. W. Choptuik, E. W. Hirschmann, S. L. Liebling, F. Pretorius, PRL 93, 131101 (2004), [gr-qc/0405101](https://arxiv.org/abs/gr-qc/0405101).
- [GGM99] D. Garfinkle, C. Gundlach, J. M. Martín-García, PRD 59, 104012 (1999), [gr-qc/9811004](https://arxiv.org/abs/gr-qc/9811004).
- [JG17] J. Jałmużna, C. Gundlach, PRD 95, 084001 (2017), [arXiv:1702.04601](https://arxiv.org/abs/1702.04601).

*Yang–Mills*
- [CCB96] M. W. Choptuik, T. Chmaj, P. Bizoń, PRL 77, 424 (1996), [gr-qc/9603051](https://arxiv.org/abs/gr-qc/9603051).
- [G97] C. Gundlach, PRD 55, 6002 (1997), [gr-qc/9610069](https://arxiv.org/abs/gr-qc/9610069).
- [CHM99] M. W. Choptuik, E. W. Hirschmann, R. L. Marsa, PRD 60, 124011 (1999), [gr-qc/9903081](https://arxiv.org/abs/gr-qc/9903081).
- [MR18] M. Maliborski, O. Rinne, PRD 97, 044053 (2018), [arXiv:1712.04458](https://arxiv.org/abs/1712.04458).
- [BT01] P. Bizoń, Z. Tabor, PRD 64, 121701 (2001), [math-ph/0105016](https://arxiv.org/abs/math-ph/0105016).
- [CST98] T. Cazenave, J. Shatah, A. S. Tahvildar-Zadeh, Ann. Inst. H. Poincaré Phys. Théor. 68, 315 (1998) (no arXiv version found).
- [BB15] P. Biernat, P. Bizoń, Comm. Math. Phys. 338, 1443 (2015), [arXiv:1409.8214](https://arxiv.org/abs/1409.8214).
- [D14] R. Donninger, Math. Z. 278, 1005 (2014), [arXiv:1202.1389](https://arxiv.org/abs/1202.1389).
- [DO23] R. Donninger, M. Ostermann, Comm. PDE 48, 1148 (2023), [arXiv:2108.13668](https://arxiv.org/abs/2108.13668).
- [Gl23] I. Glogić, [arXiv:2305.10312](https://arxiv.org/abs/2305.10312) (journal reference not checked).

*σ-models and wave maps*
- [BW00] P. Bizoń, A. Wasserman, PRD 62, 084031 (2000), [gr-qc/0006034](https://arxiv.org/abs/gr-qc/0006034).
- [BSW04] P. Bizoń, S. Szybka, A. Wasserman, PRD 69, 064014 (2004), [gr-qc/0310038](https://arxiv.org/abs/gr-qc/0310038).
- [HLPTA00] S. Husa, C. Lechner, M. Pürrer, J. Thornburg, P. C. Aichelburg, PRD 62, 104007 (2000), [gr-qc/0002067](https://arxiv.org/abs/gr-qc/0002067).
- [LTHA02] C. Lechner, J. Thornburg, S. Husa, P. C. Aichelburg, PRD 65, 081501 (2002), [gr-qc/0112008](https://arxiv.org/abs/gr-qc/0112008).
- [HE97] E. W. Hirschmann, D. M. Eardley, PRD 56, 4696 (1997), [gr-qc/9511052](https://arxiv.org/abs/gr-qc/9511052).
- [LC96] S. L. Liebling, M. W. Choptuik, PRL 77, 1424 (1996), [gr-qc/9606057](https://arxiv.org/abs/gr-qc/9606057).
- [Sh88] J. Shatah, Comm. Pure Appl. Math. 41, 459 (1988), doi:10.1002/cpa.3160410405 (no arXiv version).
- [Bi00] P. Bizoń, Comm. Math. Phys. 215, 45 (2000), [math-ph/9910026](https://arxiv.org/abs/math-ph/9910026).
- [BCT00] P. Bizoń, T. Chmaj, Z. Tabor, Nonlinearity 13, 1411 (2000), [math-ph/9912009](https://arxiv.org/abs/math-ph/9912009).
- [LHI00] S. L. Liebling, E. W. Hirschmann, J. Isenberg, J. Math. Phys. 41, 5691 (2000), [math-ph/9911020](https://arxiv.org/abs/math-ph/9911020).
- [BBM17] P. Biernat, P. Bizoń, M. Maliborski, Nonlinearity 30, no. 4 (2017) (page number not checked), doi:10.1088/1361-6544/aa61ab, [arXiv:1608.07707](https://arxiv.org/abs/1608.07707).
- [CDG17] O. Costin, R. Donninger, I. Glogić, Comm. Math. Phys. 351, 959 (2017) (arXiv number not looked up).
- [DG19] R. Donninger, I. Glogić, Anal. PDE 12, 389 (2019), [arXiv:1705.06352](https://arxiv.org/abs/1705.06352).
- [GHW25] I. Glogić, D. Hilditch, D. Wallauch, [arXiv:2512.16623](https://arxiv.org/abs/2512.16623) (2025).

*Vacuum*
- [BCS05] P. Bizoń, T. Chmaj, B. G. Schmidt, PRL 95, 071102 (2005), [gr-qc/0506074](https://arxiv.org/abs/gr-qc/0506074).
- [BCRST05] P. Bizoń, T. Chmaj, A. Rostworowski, B. G. Schmidt, Z. Tabor, PRD 72, 121502 (2005), [gr-qc/0511064](https://arxiv.org/abs/gr-qc/0511064).
- [BS06] P. Bizoń, B. G. Schmidt, [gr-qc/0605114](https://arxiv.org/abs/gr-qc/0605114) (journal reference not checked).
- [BCS06] P. Bizoń, T. Chmaj, B. G. Schmidt, [gr-qc/0608102](https://arxiv.org/abs/gr-qc/0608102) (journal reference not checked).
- [BW10] P. Bizoń, A. Wasserman, Class. Quantum Grav. 27, no. 12 (2010) (article number not checked), [arXiv:1003.3306](https://arxiv.org/abs/1003.3306).
- [PVG22] B. Porto Veronese, C. Gundlach, PRD 106, 104044 (2022), [arXiv:2209.08404](https://arxiv.org/abs/2209.08404).
- [AE93] A. M. Abrahams, C. R. Evans, PRL 70, 2980 (1993) (no arXiv version).
- [LK21] T. Ledvinka, A. Khirnov, PRL 127, 011104 (2021), [arXiv:2102.09579](https://arxiv.org/abs/2102.09579).
- [B+23] T. W. Baumgarte, B. Brügmann, D. Cors, C. Gundlach, D. Hilditch, A. Khirnov, T. Ledvinka, S. Renkhoff, I. Suárez Fernández, PRL 131, 181401 (2023), [arXiv:2305.17171](https://arxiv.org/abs/2305.17171).
- [R25] A. Rostworowski, [arXiv:2503.17549](https://arxiv.org/abs/2503.17549) (2025).
- [BGH19] T. W. Baumgarte, C. Gundlach, D. Hilditch, PRL 123, 171103 (2019), [arXiv:1909.00850](https://arxiv.org/abs/1909.00850).

*Fluid, complex-scalar CSS, axion–dilaton*
- [EC94] C. R. Evans, J. S. Coleman, PRL 72, 1782 (1994), [gr-qc/9402041](https://arxiv.org/abs/gr-qc/9402041).
- [KHA95] T. Koike, T. Hara, S. Adachi, PRL 74, 5170 (1995), [gr-qc/9503007](https://arxiv.org/abs/gr-qc/9503007).
- [KHA99] T. Koike, T. Hara, S. Adachi, PRD 59, 104008 (1999), [gr-qc/9607010](https://arxiv.org/abs/gr-qc/9607010).
- [M96] D. Maison, Phys. Lett. B 366, 82 (1996), doi:10.1016/0370-2693(95)01381-4, [gr-qc/9504008](https://arxiv.org/abs/gr-qc/9504008).
- [NC00] D. W. Neilsen, M. W. Choptuik, Class. Quantum Grav. 17, 761 (2000), doi:10.1088/0264-9381/17/4/303, [gr-qc/9812053](https://arxiv.org/abs/gr-qc/9812053).
- [AG+09] L. Álvarez-Gaumé, C. Gómez, A. Sabio Vera, A. Tavanfar, M. A. Vázquez-Mozo, Nucl. Phys. B 806, 327 (2009), [arXiv:0804.1464](https://arxiv.org/abs/0804.1464).
- [HE95a] E. W. Hirschmann, D. M. Eardley, PRD 51, 4198 (1995), [gr-qc/9412066](https://arxiv.org/abs/gr-qc/9412066).
- [HE95b] E. W. Hirschmann, D. M. Eardley, PRD 52, 5850 (1995), [gr-qc/9506078](https://arxiv.org/abs/gr-qc/9506078).
- [EHH95] D. M. Eardley, E. W. Hirschmann, J. H. Horne, PRD 52, R5397 (1995), [gr-qc/9505041](https://arxiv.org/abs/gr-qc/9505041).
- [HHS96] R. S. Hamadé, J. H. Horne, J. M. Stewart, Class. Quantum Grav. 13, 2241 (1996), doi:10.1088/0264-9381/13/8/017, [gr-qc/9511024](https://arxiv.org/abs/gr-qc/9511024).
- [AGH12] L. Álvarez-Gaumé, E. Hatefi, Class. Quantum Grav. 29, 025006 (2012), [arXiv:1108.0078](https://arxiv.org/abs/1108.0078).
- [AH20a] R. Antonelli, E. Hatefi, JHEP 03 (2020) 074, [arXiv:1912.00078](https://arxiv.org/abs/1912.00078).
- [AH20b] R. Antonelli, E. Hatefi, JHEP 03 (2020) 180, [arXiv:1912.06103](https://arxiv.org/abs/1912.06103).
- [HV20] E. Hatefi, E. Vanzan, Eur. Phys. J. C 80, 952 (2020), [arXiv:2005.11646](https://arxiv.org/abs/2005.11646).
- [HK21] E. Hatefi, A. Kuntz, Eur. Phys. J. C 81, no. 1 (2021), doi:10.1140/epjc/s10052-020-08788-w, [arXiv:2010.11603](https://arxiv.org/abs/2010.11603).

*Charged and massive scalar*
- [HP97] S. Hod, T. Piran, PRD 55, 3485 (1997), [gr-qc/9606093](https://arxiv.org/abs/gr-qc/9606093).
- [GMG96] C. Gundlach, J. M. Martín-García, PRD 54, 7353 (1996), [gr-qc/9606072](https://arxiv.org/abs/gr-qc/9606072).
- [MGM26] S. Mittal, C. Gundlach, L. Martel, [arXiv:2609.17207](https://arxiv.org/abs/2609.17207) (2026).
- [BCG97] P. R. Brady, C. M. Chambers, S. M. C. V. Gonçalves, PRD 56, R6057 (1997), [gr-qc/9709014](https://arxiv.org/abs/gr-qc/9709014).

*Large D and reviews*
- [RW18] M. Rozali, B. Way, JHEP 11 (2018) 106, [arXiv:1807.10283](https://arxiv.org/abs/1807.10283).
- [CP25] C. R. Clark, G. L. Pimentel, JHEP 11 (2025) 017, [arXiv:2504.10590](https://arxiv.org/abs/2504.10590).
- [EEG26] C. Ecker, F. Ecker, D. Grumiller, PRL 136, 191401 (2026), [arXiv:2601.14358](https://arxiv.org/abs/2601.14358).
- [EEGJ26] C. Ecker, F. Ecker, D. Grumiller, T. Jechtl, SciPost Phys. 20, 176 (2026), [arXiv:2602.10185](https://arxiv.org/abs/2602.10185).
- [EST13] R. Emparan, R. Suzuki, K. Tanabe, [arXiv:1302.6382](https://arxiv.org/abs/1302.6382) (JHEP 2013; volume and page not checked).
- [EH20] R. Emparan, C. P. Herzog, Rev. Mod. Phys. 92, 045005 (2020), [arXiv:2003.11394](https://arxiv.org/abs/2003.11394).
- [K05] B. Kol, [hep-th/0502033](https://arxiv.org/abs/hep-th/0502033) (journal reference not checked).
- [SH96] J. Soda, K. Hirata, [gr-qc/9605023](https://arxiv.org/abs/gr-qc/9605023) (journal reference not checked).
- [GHMG25] C. Gundlach, D. Hilditch, J. M. Martín-García, review, [arXiv:2507.07636](https://arxiv.org/abs/2507.07636) (2025; journal version pending).
