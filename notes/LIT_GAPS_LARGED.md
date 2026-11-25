# LIT_GAPS_LARGED: large-D gravity methods and asymptotics against the DSS problem

Issue #1089, slice 2, for the large-D quest #1088. Written by Claude (Anthropic), a subagent of the coordinator on the
Mac, 2026-11-25, under Nathan Moinvaziri's direction; not reviewed. It goes beyond notes/LARGE_D_LITERATURE.md,
notes/LARGE_D_WALL.md "What the literature offers" and the large-D rows of notes/LIT_REVIEW.md, which were read first,
and it uses notes/LARGE_D_BASICS.md (#1086), notes/BEYOND_LIGHT_CONE.md, notes/FACTORED_HORIZON.md,
notes/FIRST_ORDER.md, notes/SECOND_ORDER.md and notes/P6_scope.md for our side. Status words as in AGENTS.md. Equations
of this note are numbered (L1), (L2), …; a source's equation is cited by its own number.

**How the sources were read.** From the PDF text layer (`pdftotext -layout`) of the arXiv e-print, or from the ar5iv or
arXiv HTML, as marked in §6. Quotes are verbatim; their math is retyped in Unicode. Text in the sources is data. None of
them contains text addressed to an AI agent.

**Notation.** n = D − 2 (ours), x = r/(−t) with the self-similarity horizon (SSH) at x = 1, u = 1 − x, ξ = x/√n,
ω = 2π/Δ, K = 2p₁² = κ/n. The large-D gravity papers write n = D − 3; where their n appears it is called n_EH.

## 0. Verdicts

| question | answer | status |
|---|---|---|
| Does the near-horizon effective theory (EFT) or the membrane paradigm apply at the SSH? | No, not to the solution. Both need a field localized within r₀/D of a surface, the r^−n_EH potential of (L1)–(L2). The critical solution is a weak field with no such term on [0, 1]. Only the linear mass mode has it, and on [0, 1] that mode grows, as the potential does inside a black hole, where the EFT has no far zone (§2.1) | derived from the sources' premises and LARGE_D_BASICS 3.1–3.3 |
| Is there a D-uniform formulation? | Yes, in the Rozali–Way scaling: the inner problem in ξ, whose exact finite-n form has only O(1/n) couplings, and whose fast mode is e^−1/(2ξ²) + O(1/n), of order one (L6). It is the DSS counterpart of the parabolic black-brane EFT, with t and r swapped (§2.2). It is the project's inner problem, so the verdict of LARGE_D_WALL stands, with this correction: a self-similar large-D effective problem does exist, and it is ours | derived (L6); the rest from the sources |
| What does the EFT's variable 𝖱 = rⁿ give us? | The mass mode in the variable 𝖱 = x^(n−1), δB = −μ/𝖱 (L4), which is FACTORED_HORIZON §3's exact factoring. It removes the 0.155(n − 1) digits and the mass-mode series tail, nothing else (§2.1, lead 4) | checked by hand (L3) |
| Δ₂, Δ₃ cheaply? | Δ₂ exists already (SECOND_ORDER, M = 5, 1.8% off). The bottleneck is the harmonic truncation, not the order in 1/n. A Melnikov form of the Stokes constant's derivative, (11.13) of arXiv:2107.14566, gives every order's forcing from one adjoint solution. The amplitude expansion at O(1/n) would give Δ₁ exact in the harmonics (§3, lead 5) | literature; the transfer is a proposal |
| Uniformity in the number of harmonics (P6)? | The closest rigorous precedent, arXiv:2107.14566 (Klein–Gordon breathers, all harmonics, in ℓ¹), works because harmonic k's exponential rate grows with k, so one harmonic dominates. In ours every harmonic shares one fast singularity and the weights e^(π\|ν\|ω/2) grow. That reduction fails, so (★) stays open (§3.3) | literature against P6_scope |
| Clark–Pimentel's CSS against our DSS | A different solution (the D-dimensional Roberts family), not the attractor. Their spectrum's upper edge κ = 1/(2ε) = n/2 equals our SSH exponent at leading order. Their sentence that the CSS mass exponent's finite limit agrees with the DSS mixes conventions: the DSS mass exponent is (D − 3)γ = 28.4 at D 61.03, against their 1.97 (§4.1) | read; the edge identification inferred; numbers from hp/data/gamma_dd.jsonl |
| Published claims our numbers contradict | Δ → 0 as D → ∞ (arXiv:2601.14358, arXiv:2602.10185, and Bland's thesis's Δ∞ = 5/2, recorded before); Emparan and Herzog's γ ≈ ½(1 − 1/D) (their (10.3)), 2.46×10⁻³ above our γ at D 61.03, with a slope 0.66 of ours; Clark–Pimentel's agreement sentence (§4.3) | numerical evidence |
| Non-perturbative terms in Δ(1/n) | For black holes they are O(n_EH^−n_EH) ((3.24) of arXiv:2003.11394) or n_EH^(−n_EH/2) (§3.4 of arXiv:1802.08191). For the DSS free field the analogous term does not decay with n: 2e^(−2πjω) sin²(πn/2), 1.3×10⁻¹⁰ for j = 1 (LARGE_D_BASICS §8). That is a floor near 10⁻¹⁰ for a series route if it is inherited, harmless at 10⁻⁸ (§4.4) | literature; inheritance not known |
| Δ at D 1000, 2000, 10000 | Track C's fits give 1.654240854245 ± 7.8×10⁻¹⁰, 1.649074313886 ± 1.2×10⁻⁹ and 1.644930989934 ± 1.7×10⁻⁹ (lead 1). The ± is a spread over fit choices, conditional on Δ being analytic in 1/n | numerical evidence (conditional) |

**The decision it informs** (#1088). For Δ to 10⁻⁸ at D 400–10000, the series is already enough, under its condition.
For the solution itself, nothing in the large-D gravity literature replaces our formulation. A direct solve past
D ≈ 700 is gated by one measurement, whether the grip on Δ falls algebraically or exponentially in D (lead 3). The
matched asymptotics read here predict algebraically.

## 1. What each source does and does not contain

| source | read | what it does (with equation numbers) | what it does not do |
|---|---|---|---|
| R. Emparan, C. P. Herzog, *Large D limit of Einstein's equations*, RMP 92, 045005 (2020), arXiv:2003.11394 | PDF text: Secs. II, III.F, VII (the black-brane equations near (7.37)–(7.41)), VII's Bjorken paragraph (7.31)–(7.32), X.C | The scale r₀/D, (2.3)–(2.6); 𝖱 = (r/r₀)^n_EH, (2.8); the near-zone metric, (2.10); the ingoing condition Ψ ∝ (𝖱 − 1)^(−iωr₀/n_EH), (3.4). III.F: "The breakdown is non-perturbative in 1/n", "non-perturbative corrections = O(n^−n)" (3.24), from the far-zone Hankel functions (3.22)–(3.23). VII: "The corrections indicate that the large D expansion is only asymptotic in nature". X.C: γ ≈ ½(1 − 1/D), (10.3); the Rozali–Way scaling t = τ/√D, (10.5), and the parabolic ∂²_τφ = (1/r)∂_rφ, (10.6), "the same scaling (with t and r swapped)" as the black hole/black string transition | No self-similar solution, no Δ, no treatment of a null surface other than an event or apparent horizon |
| S. Bhattacharyya, A. De, S. Minwalla, R. Mohan, A. Saha, *A membrane paradigm at large D*, JHEP 04 (2016) 076, arXiv:1504.06613 | ar5iv, first 100k characters | ds² = ds²_flat + O_μO_ν dx^μdx^ν/ψ^(d+p) (2.13) with ψ from (2.11); the membrane at ψ = 1; the field lives within 1/d of it, and outside the metric "approaches the metric of flat space exponentially rapidly", as e^(−(ψ−1)d). Leading-order membrane equations (1.7)–(1.8) | Vacuum only; the membrane is the event horizon; no matter, no other null surface |
| R. Emparan, R. Suzuki, K. Tanabe, *Evolution and endpoint of the black string instability: Large D solution*, PRL 115, 091102 (2015), arXiv:1506.06772v2 | PDF text, pp. 1–4 | Effective equations ∂_t m − ∂_b∂^b m = −∂_b p^b (11), ∂_t p_a − ∂_b∂^b p_a = ∂_a m − ∂_b(p_a p^b/m) (12); for strings (19)–(20); static strings m″ + m − (m′)²/m = const (17), with m = e^(2P). "The Mathematica function NDSolve handles them without difficulty" | No self-similarity, no matter |
| R. Emparan, R. Luna, M. Martínez, R. Suzuki, K. Tanabe, *Phases and stability of non-uniform black strings*, JHEP 05 (2018) 104, arXiv:1802.08191v4 | PDF text: abstract, Sec. 3.4 | Expansion "including corrections up to 1/D⁴", solutions "from D ≈ 9 all the way up to D = ∞" (abstract). Sec. 3.4: non-perturbative effects of size "∼ n^(−n/2)", "comparable to perturbative corrections 1/n^k at order k ∼ n/2"; n* = 8 / 9.48 / 9.65 / 9.93 order by order (3.23), "which signals a loss of convergence in the last correction" | No Borel or Padé resummation of the series |
| M. Rozali, B. Way, arXiv:1807.10283 (JHEP 11 (2018) 106) | PDF text, Secs. 2.2–2.3 (the arXiv numbering; LIT_REVIEW.md cites the journal's (9)–(12)) | Schwarzschild gauge (2.5)–(2.6); τ = √D t and A = 1 − A₀/D (2.7); leading order (2.8a)–(2.8c), ∂_rφ₀ = (r/√(1−δ₀)) ∂_τ(∂_τφ₀/√(1−δ₀)); "Analogy with the heat equation suggests that 'initial data' φ₀(τ, r = 0) should be given, and then integrated to larger values of r"; the invariance δ₀ → δ₀ + c(τ); "linear equations in the higher-order variables that are sourced nonlinearly by the lower order solutions"; the oscillaton (2.9)–(2.10); r_div = √(2(ε − 1)) (2.11), γ = 1/2; Bland et al.'s fit γ = ½ − m₃/(D − m₂)^m₁ (Fig. 2) | No self-similar solution: "Our solutions, however, do not exhibit self symmetry of any sort" (Sec. 2.3) |
| C. R. Clark, G. L. Pimentel, JHEP 11 (2025) 017, arXiv:2504.10590v2 | PDF text: Secs. 1, 2, 3 (summary), 4 (to (4.24) and (4.54)), 5 | ε = 1/(D − 2) (2.8); the CSS ODEs (2.24a)–(2.24c) and first integral (2.32); q*(ε) = ε^ε(1 − ε)^(1−ε) (3.8); regions A (flat), B (f″ + ¼f′² − f = 0, timescale O(1/√D)), C (strong gravity); region B's (3.38) maps to the large-D non-uniform black-string equation under x → iz, f → 4(P − P₀) (Sec. 1); perturbations e^(ks) (4.18); M ∼ δp^((D−3)/k) (4.23), γ_p = (1 − ε)/(κε) (4.24), κ = sup Re k; a continuous spectrum (4.54); γ_p = 2(D − 3)/(D − 2) (1.5), γ_q = (D − 3)/√(2(D − 2)) (1.4). Footnote 1: "it has not yet been proven that there are no further restrictions from the 'matching' of these asymptotics"; footnote 23: "a full solution could impose additional constraints on the Lyapunov exponent spectrum through matching". Sec. 5: DSS "would likely involve introducing a Boltzmann-like hierarchy of equations, one for each partial wave" | No DSS computation, no analyticity condition at the self-similar horizon in the perturbation problem |
| O. M. L. Gomide, M. Guardia, T. M. Seara, C. Zeng, *On small breathers of nonlinear Klein-Gordon equations via exponentially small homoclinic splitting*, arXiv:2107.14566v3 (24 Feb 2025) | PDF text: abstract, Secs. 1–3, 11 | Spatial dynamics for time-periodic solutions, a Bogdanov–Takens bifurcation "coupled to infinitely many rapidly oscillatory directions"; the inner equation ∂²_zφ⁰ − ∂²_τφ⁰ − φ⁰ + (φ⁰)³/3 + f(φ⁰) = 0 (3.9), with all temporal harmonics, on the inner domains (3.10); the jump φ^(0,u) − φ^(0,s) = e^(−2√2 r)(C_in sin 3τ + O_ℓ¹(1/r)) (2.11); "The splitting ∆φ⁰ is dominated by the weakest exponential decay rate" (Sec. 3); the derivative of C_in along a perturbation as a Melnikov integral against an exact variational solution, (11.13) with (11.10)–(11.12); C_in ≠ 0 on an open dense set (Theorem 1.4) | Nothing on self-similarity or gravity |
| V. Gelfreich, C. Simó, *High-precision computations of divergent asymptotic series and homoclinic phenomena*, DCDS-B 10 (2008) 511, doi:10.3934/dcdsb.2008.10.511 | zbMATH review only | Values of an exponentially small invariant at high precision, then the coefficients of its asymptotic expansion extracted from them; Gevrey-1 tests and the Borel singularities located | Full text not read |
| T. Aoki, M. Tanda, *Parametric Stokes phenomena of the Gauss hypergeometric differential equation with a large parameter*, J. Math. Soc. Japan 68 (2016) 1099–1132; T. Koike, *On the exact WKB analysis of second order linear ordinary differential equations with simple poles*, Publ. RIMS 36 (2000) 297–319 | abstracts only | Voros coefficients of the Gauss equation with a large parameter, their explicit forms and Borel sums, and the Stokes jumps in the parameters; Koike: the connection formula at a simple pole, which "also acts as a turning point" | Full texts not read |

The rows of LIT_REVIEW.md for Emparan–Suzuki–Tanabe 1302.6382, Emparan et al. 1504.06489, Emparan–Rafecas-Ventosa–Way
2512.14186, Emparan–Suzuki 1905.01062 and the Ecker–Ecker–Grumiller papers were not repeated.

## 2. Question 1: a D-uniform formulation, and whether the EFT applies at the SSH

### 2.1 The EFT's premise, against our fields

The near-horizon EFT starts from the Tangherlini potential and its two zones (arXiv:2003.11394, (2.3)–(2.10)):

```math
\nabla\Phi \sim \frac{D}{r_0}, \qquad \mathsf R = \Big(\frac{r}{r_0}\Big)^{n_{EH}}, \qquad
ds^2_{nh} \to \frac{4r_0^2}{n_{EH}^2}\Big(-\big(1 - \mathsf R^{-1}\big)d\bar t^2 + \frac{d\mathsf R^2}{\mathsf R(\mathsf R - 1)}\Big) + r_0^2\,\mathsf R^{2/n_{EH}}\,d\Omega^2_{n_{EH}+1}
\tag{L1}
```

The membrane paradigm's ansatz has the same structure (arXiv:1504.06613, (2.13)):

```math
ds^2 = ds^2_{\rm flat} + \frac{O_\mu O_\nu\,dx^\mu dx^\nu}{\psi^{\,d+p}}, \qquad \text{deviation} \sim e^{-(\psi-1)d}\ \text{outside } \psi = 1 .
\tag{L2}
```

Both need a term of the form (surface variable)^−D in the background. Ours, from LARGE_D_BASICS §1's a-equation with
B = 1/a² (checked by hand; FACTORED_HORIZON §3.1 uses the same variable):

```math
x\,B_x = -(n-1)(B - 1) - B\,(p^2 + m^2) .
\tag{L3}
```

- **The solution has no such term.** On the slow manifold (n − 1)(a² − 1) = Kx²/(1 + Kx²) (LARGE_D_BASICS 3.2), so
  B − 1 = O(1/n) and smooth on [0, 1]. The stored states have n⟨p₁²⟩ = 3.86–3.89 (3.1). There is no near zone.
- **The mass mode has it, in the EFT's own variable.** The homogeneous part of (L3)'s linearization is
  ```math
  \delta B = -\mu\,x^{-(n-1)} = -\frac{\mu}{\mathsf R_{\rm SSH}}, \qquad \mathsf R_{\rm SSH} = x^{\,n-1},
  \tag{L4}
  ```
  the Tangherlini term with the SSH in the role of r₀. Its "near zone" is the layer u = O(1/n) at the SSH.
- **On [0, 1] it grows.** The EFT's far zone, where (r₀/r)^n_EH → 0 and corrections are "exponentially small in 1/n",
  is r > r₀. Our domain 0 ≤ x ≤ 1 is the analogue of r < r₀, the black-hole interior, where (L4) grows by
  0.7^−(n−1) between the SSH and the last node, 0.155(n − 1) digits (LARGE_D_BASICS §4). There is no far zone to match
  to, and nothing exponentially small to drop.
- **Where a membrane does form.** A datum μ ≠ 0 at the SSH is a Tangherlini mass whose apparent horizon sits at
  x_AH = μ^(1/(n−1)) (LARGE_D_LITERATURE (f)). The EFT would describe that black hole, i.e. supercritical evolution and
  γ, which is Rozali and Way's reading of their divergence (their Sec. 2.2), not the critical solution.

So the EFT's tool, the variable 𝖱, applies to one mode, and FACTORED_HORIZON §3 already factors that mode exactly (its
gain becomes 1 for every n and x_K). The fast scalar mode has no near zone at the SSH at all:

```math
F = e^{-n\Phi}, \qquad \Phi = \ln x - \tfrac{1+K}{2}\ln(1 - x^2)
\tag{L5}
```

(LARGE_D_BASICS 3.3) vanishes there faster than any power, as u^σ with σ ≈ n/2 + 2.6. It is O(1) only on the curve
Re x² = ½ (BEYOND_LIGHT_CONE 2.6).

### 2.2 Where the large-D structure is uniform: the Rozali–Way region

For x > 1 and K → 0, |F| = e^(−nΦ) with Φ = −½ ln(1 − 1/x²). At x = ξ√n (derived here):

```math
n\,\Phi(\xi\sqrt n) = \frac{1}{2\xi^2} + \frac{1}{4n\,\xi^4} + O(n^{-2}) .
\tag{L6}
```

In the inner variable the fast mode is O(1) and smooth, with corrections in powers of 1/n. That is the inner region of
the project's README, at r = O(1), t = O(1/√D), Rozali and Way's scaling (arXiv:2003.11394 (10.5); arXiv:1807.10283
(2.7)). BEYOND_LIGHT_CONE §2.5 writes the exact finite-n equations there: every n enters as an O(1/n) coupling, except
the stiff b-mode ξ^−(n−1), which is (L4) again.

Against the large-D gravity literature this is the EFT pattern with the roles reversed:
- the EFT integrates out the near zone, whose dynamics is slaved, and keeps the long-wavelength dynamics. Here the outer
  region x = O(1) is slaved (the slow manifold, explicit order by order, LARGE_D_BASICS 3.2), and the inner solution
  regular at ξ = 0 resums it (notes/INNER_STOKES.md, first item);
- the effective dynamics is parabolic on scales O(√n) in x, as the black-brane EFT's is on horizon scales O(1/√D)
  (arXiv:1506.06772 (11)–(12); arXiv:2512.14186 per LIT_REVIEW.md), "with t and r swapped" (arXiv:2003.11394 X.C).
  Rozali and Way's heat-equation remark, centre data integrated outward in r, is the same statement;
- the selection is the inner problem's Stokes data, the counterpart of the EFT's decoupled sector.

**A D-uniform formulation therefore exists only in inner variables, with the selection imposed where F is O(1).** Two
placements are known, both in BEYOND_LIGHT_CONE:
- (E1), on Re x² = ½ inside the light cone. This is hp's node 0.7. Its costs grow with n: the mass mode (L4), unless
  factored; the horizon order ks ≈ σ_R + 7 ≈ n/2 + 10; and the grip, max|∂Δ/∂F| ∝ n^21.7 at D 40–61 (§5 there).
- (E2), reality on the Stokes line x₀ = ξ₀√n after continuation around x = 1. Its modulus is flat in n (§3(c) there),
  but its weights e^(πkω) ≈ 10^(5.2k) cost 70–85 digits at every D, and its basin is δ ≲ e^(−πkω) (§7 there).

Neither the near-horizon EFT nor the membrane paradigm adds a third. The 1/n treatment of a layer at the SSH, which
LARGE_D_WALL considered and dismissed, has no layer to treat: only (L4) lives there.

## 3. Question 2: Δ₂, Δ₃ from n-independent problems, and the harmonics

### 3.1 What we have

FIRST_ORDER and SECOND_ORDER derive Δ₁ and Δ₂ from the O(1/n) and O(1/n²) inner problems, in the M-harmonic
truncation:

| | M = 1 | M = 3 | M = 5 | fit (double-double) |
|---|---|---|---|---|
| Δ₁ | 15.00 | 9.886 | 10.402 | 10.3753 |
| Δ₂ | −45.86 | −44.42 | −50.06 ± 0.18 | −49.1685 |

The truncation error at M = 5, 0.3% and 1.8%, is the whole error. Higher orders in 1/n are mechanical (jets in 1/n,
SECOND_ORDER §3). Uniformity in M is not.

### 3.2 What the asymptotics literature offers for the orders

- **The derivative of a Stokes constant as a Melnikov integral.** arXiv:2107.14566 (11.13):
  ```math
  \frac{d}{d\mu}\,C_{in}(\mu\Delta)\Big|_{\mu=0} = \frac{1}{2\pi i\mu_3}\int_{-\infty}^{\infty}\!\!\int_0^{2\pi}
  \Delta\big(\phi^0_b(z+s,\tau)\big)\,\xi_3^{+}(z+s,\tau)\,d\tau\,ds ,
  \tag{L7}
  ```
  with ξ₃⁺ an exact solution of the variational equation (11.10). Transferred (a proposal, not checked): ω₁ =
  −S_{1/n}/∂_ωS (FIRST_ORDER §3) with S_{1/n} = ⟨ξ_adj, N₁[Y∞]⟩, where N₁ is the O(1/n) part of the inner operator
  (FIRST_ORDER §1) and ξ_adj the adjoint solution with the fast mode's asymptotics. One adjoint solve gives the
  forcing of every perturbation term and of the gauge data. At second order the first-order solution is needed anyway,
  so this saves work but does not change what limits Δ₂.
- **Coefficients from high-precision values.** Gelfreich and Simó (DCDS-B 10, 2008) extract the coefficients of the
  divergent expansion of an exponentially small quantity from its values at high precision, then test Gevrey-1 and
  locate the Borel singularities. Track C (LARGE_D_WALL, #801) is this method applied to Δ(D) at D ≤ 245. Its 19 fits
  give no Gevrey test, since their high coefficients are spreads over fit choices.
- **How far large-D series go in gravity.** arXiv:1802.08191 carries a black-string observable to 1/D⁴. The series
  breaks down where n^(−n/2) competes, at k ∼ n/2 ((3.23)). Track C fits Δ with K = 12–20 from D 10 at χ²/dof ≤ 2.7
  and shows no such breakdown at the data's σ (LARGE_D_WALL, Track C).
- **Exact WKB.** The free field's regular solution is ₂F₁(λ/2, (λ+1)/2; (n+1)/2; x²) (LARGE_D_BASICS 3.4), a Gauss
  equation with a large parameter. Aoki and Tanda give its Voros coefficients and their Borel sums, and the jumps of
  its connection formulas as parameters cross Stokes curves. Those jumps are the non-analytic-in-n terms (§4.4). Koike
  treats simple poles, which the centre and the SSH are in the eikonal (λ = c²/(x(x² − c²)), LARGE_D_BASICS 3.3).
  So the free field's dependence on n is known in closed, Borel-summed form. Nothing comparable exists for the
  nonlinear inner problem.

### 3.3 Uniformity in the harmonics (P6)

arXiv:2107.14566 proves an exponentially small obstruction for an inner equation with all temporal harmonics, without
truncation and without Borel summation: invariant manifolds of an ill-posed PDE, a fixed point in ℓ¹, inner domains
(3.10) and complex matching. Its reduction to one constant needs the harmonics' rates to grow, µ_k = √(k² − 1): the
jump (2.11) is carried by the third harmonic, and the others are exponentially smaller.

In our inner problem every harmonic shares the same fast singularity, at ζ = ½ in the Borel plane (INNER_STOKES.md),
and the weights e^(π|ν|ω/2) grow faster than the profile's harmonics decay (P6_scope (★)). So that reduction is not
available, and the ℓ¹ framework would need a weight that beats e^(π|ν|ω/2), which is (★) itself. This is not one of
P6_scope's five routes. It is recorded as a sixth candidate that fails at the same inequality.

## 4. Question 3: gaps and connections

### 4.1 Clark and Pimentel's CSS against our DSS

- **Different solutions.** Theirs is the D-dimensional Roberts family, with the critical member q*(ε) (3.8) known in
  closed form. Its CSS-breaking perturbations form a continuum, (4.54), so "this critical exponent is not universal"
  (Sec. 4). Ours has one unstable mode, λ → 2 (Proposition 5).
- **The continuum's edge is our SSH exponent at leading order (inferred, not checked).** (4.24) with (1.5) gives κ = sup
  Re k = 1/(2ε) = n/2. For a perturbation e^(λτ) of our setup, LARGE_D_BASICS §2's exponent at the SSH, with ω∂_θ
  replaced by λ + ω∂_θ (derived here), is σ = (n/2 − 1 + ḡ − λ − ijω)/(1 − ḡ), positive for Re λ below n/2 − 1 + ḡ. They
  impose decay at one end and boundedness at late x, (4.19). A spectrum built from boundedness admits the non-analytic
  branch u^σ, and its edge would then sit at Re σ = 0, which is n/2 at leading order, as found. Their background is not
  ours, so ḡ and the O(1) shift differ. Their footnote 23 allows that matching may restrict it. Analyticity at the SSH
  is the condition that makes such spectra discrete (Gundlach, gr-qc/9604019, as recorded in LARGE_D_LITERATURE (e)).
  Test: the CSS-breaking problem of their (4.18) with analyticity imposed at the SSH, at one D. Off the quest's path.
- **Conventions.** After (1.5): "the critical exponent for breaking the self-similarity asymptotes to a finite value in
  infinite dimensions, in alignment with the numerical evidence for the discretely self-similar solution". Their γ is a
  mass exponent, (2.1) and (4.23). The DSS evidence is for the length exponent γ → 1/2. In the mass convention the DSS
  exponent is (D − 3)γ, which grows like (D − 3)/2. From hp/data/gamma_dd.jsonl: 3.189 at D 10, 5.122 at D 14 and
  28.395 at D 61.03, against their 2(D − 3)/(D − 2) = 1.750, 1.833 and 1.966. LIT_REVIEW.md flagged the convention
  (C5 item); this gives the numbers.
- **Their region B and our inner region.** Region B lives on a timescale O(1/√D) in their self-similar time, and its
  equation (3.38) maps to the large-D static black-string equation. Our inner region is at x = O(√n), also Rozali and
  Way's scaling. The sources read do not connect region B with Rozali–Way or with any DSS problem. Their DSS outlook (a
  hierarchy over partial waves) is what the M-harmonic inner problem is.

### 4.2 Rozali and Way's heat-type equations against our inner problem

Already credited (LIT_REVIEW.md C4, C5). This reading adds:
- (2.8c) is the inner problem's scalar equation. Their stated method, centre data integrated outward in r, is a Cauchy
  problem in r, i.e. spatial dynamics. The DSS selection is then a condition on that evolution's analytic
  continuation, the Stokes data, which their oscillaton does not have, since it is a single harmonic.
- Their invariance δ₀ → δ₀ + c(τ) is our time-reparametrization gauge (LARGE_D_BASICS §1, DINF_ANALYTIC (A)).
- "One will then obtain linear equations in the higher-order variables that are sourced nonlinearly" is FIRST_ORDER's
  and SECOND_ORDER's structure. They did not carry it out.

### 4.3 Published claims that our numbers contradict

| claim | where | ours | status |
|---|---|---|---|
| Δ → 0 as D → ∞ | arXiv:2601.14358 ("Existing data … suggest Δ→0", per LIT_REVIEW.md); arXiv:2602.10185 Sec. 5.2 ("compatible with the conjectures Δ → 0") | Δ(10000) = 1.644931 (lead 1), Δ∞ = 1.643893746 ± 1.5×10⁻⁸ (fitted); arXiv:2611.07007 also has a finite limit | contradicted; known |
| Δ∞ = 5/2 | Bland's thesis (2007), per LIT_REVIEW.md | as above | contradicted; known |
| γ ≈ ½(1 − 1/D) | arXiv:2003.11394 (10.3), "the data are well approximated by" | γ(61.03) = 0.489349 against 0.491807, a difference of −2.46×10⁻³; at D 10, +5.59×10⁻³. Our slope: λ = 2 + λ₁/n with λ₁ = 3.0324 (FIRST_ORDER, fit), so γ = ½(1 − 1.516/n + …) against their ½(1 − 1/D) | contradicted at large D; new here |
| γ → 0.466 | Bland et al. 2005, per LIT_REVIEW.md | γ(61.03) = 0.489349 | contradicted; known |
| Δ minimum and γ maximum near D 11–13 | Sorkin and Oren 2005 | Δ falls and γ rises monotonically to D 61 (LIT_REVIEW.md) | contradicted; known |
| CSS mass exponent's limit "in alignment with" the DSS | arXiv:2504.10590, after (1.5) | §4.1 | contradicted in the mass convention; new here |

The γ numbers come from `python3 -c` over `hp/data/gamma_dd.jsonl` (§7). Emparan and Herzog do not claim (10.3) at
large D; they cite Rozali and Way for "more accurate fits". The row records how far the formula is from ours, not an
error in their review.

### 4.4 Non-perturbative terms

- **Black holes.** (3.24) of arXiv:2003.11394, O(n_EH^−n_EH), and Sec. 3.4 of arXiv:1802.08191, n_EH^(−n_EH/2). Both
  vanish faster than any power and are negligible at D ≥ 100.
- **Our free field.** The connection coefficient carries 1/sin(π(ijω − n/2)), whose modulus has a period-2 modulation
  2e^(−2πjω) sin²(πn/2), 1.3×10⁻¹⁰ for j = 1, which does not decay with n (LARGE_D_BASICS §8). In exact-WKB terms it is
  a parametric Stokes jump (§3.2). If the nonlinear selection inherited it with an O(1) coupling, Δ(D) would carry a
  period-2 or period-4 term near 10⁻¹⁰. The D ≤ 61 fits' residual of 4×10⁻¹⁵ rules that out there (LARGE_D_BASICS §8).
  At large D it is a possible floor for any series route near 10⁻¹⁰, below the 10⁻⁸ that #1088 asks for.
- **arXiv:2611.07007's "instanton corrections of the form e^{−1/ε}".** SECOND_ORDER §6 reads them as the inner jump
  e^(−nΦ(x)) in outer variables, entering Δ order by order, not as an additive term.

## 5. Leads, ranked

Each lead has a test that can fail and a cost. Ranked by value per cost for #1088.

**Lead 1. Δ at the targets from Track C's series (done here, 1 CPU-minute).** Track C's own fits (hp/track_c.py:
`load_real`, `all_fits`, `estimate`; command in §7), evaluated beyond its table:

| D | Δ | E (spread over fit choices) |
|---|---|---|
| 300 | 1.678171015417 | 1.25×10⁻¹⁰ |
| 400 | 1.669658064111 | 2.31×10⁻¹⁰ |
| 1000 | 1.654240854245 | 7.84×10⁻¹⁰ |
| 2000 | 1.649074313886 | 1.20×10⁻⁹ |
| 10000 | 1.644930989934 | 1.70×10⁻⁹ |
| ∞ | 1.643893745321 | 1.85×10⁻⁹ |

The rows at 300 and 400 reproduce LARGE_D_WALL's table, so the call is the script's. Conditional on Δ analytic in 1/n
below 245 up to exponentially small terms (Track C's premise). The large-D gravity precedent supports such series at
large n (arXiv:1802.08191). §4.4 names the one term that could break it, near 10⁻¹⁰. **Test:** Track A's direct solves
at D 300 and 400 (#1088) must land within E. A miss by more than about 3E rejects the premise at that D.

**Lead 2. A D-uniform seed from n-independent problems (an hour of code, seconds of CPU).** The corrected NNLO outer
expansion (SECOND_ORDER §1), whose largest error against 63 solutions from D 15 to 215 falls like ε⁴ (f, Ω) and
ε^(7/2) (Π), with the profile p₁√n extrapolated in 1/n from the stored states. Its error falls with D: with
SECOND_ORDER's ε = 1/(n + 1), ε^(7/2) is 7.9×10⁻¹⁰ at D 400, 2.8×10⁻¹² at D 2000 and 1.0×10⁻¹⁴ at D 10⁴, times
SECOND_ORDER's constant. A rescaled D-263 state's error is fixed by
its extrapolation. **Test:** against the stored chain states at D 245–263, held out of the profile fit, the seed's
error must follow SECOND_ORDER's ε^(7/2) law with its constant. It fails if the error stalls, which would point at the
held harmonics (LARGE_D_BASICS D1). Then the seed's residual in hp at D 300 and 400.

**Lead 3. The gate: does the grip fall algebraically or exponentially? (LARGE_D_BASICS T1, hours of octo-double).** The
matched asymptotics read here predict algebraically. The grip on Δ is the inner multiplier's O(1) derivative ∂_ωS
times matching factors that are Γ-ratios and powers of n (LARGE_D_BASICS 3.4; the n^⟨F²⟩ of C_selection Lemma B5.1,
BEYOND_LIGHT_CONE 2.6). That is consistent in kind with the measured max|∂Δ/∂F| ∝ n^21.7 at D 40–61
(BEYOND_LIGHT_CONE §5); the exponent is not derived. Take the weak directions' gain of about 3.4×10²¹ at D 259.6
(#1008, LARGE_D_BASICS D2) and carry it with each law:

| D | n^21.7 from D 259.6 | e^(0.41 per unit D) from D 259.6 (#941) |
|---|---|---|
| 400 | 10^25.6 | 10^46.5 |
| 2000 | 10^40.8 | 10^331 |
| 10000 | 10^56.0 | 10^1756 |

Octo-double's 2⁻⁴²⁴ is 10^−127.6. With 30 working digits, the power law fits at every target, and the exponential law
stops near D 687, the range #1088 already expects for octo-double. **Test:** T1 as written (every harmonic free at
D 262.57). Pass for "algebraic": the Δ-mode's σ is not smaller than the pinned system's, and Δ moves toward the series
by about 1.07×10⁻⁶. Every direct solve past D 700 depends on this result.

**Lead 4. The mass mode out by the EFT's variable, with every harmonic free, at D ≥ 400 (days; FACTORED_HORIZON's
stage 1 measured 21–32 inner sweeps).** (L4) is FACTORED_HORIZON §3's factoring. It removes the 0.155(n − 1) digits:
61.8 at n 400 (LARGE_D_BASICS §4), about 310 at D 2000 and 1550 at D 10⁴ by the same law. It also removes the
mass-mode tail that sets ks: 732 at n 400 for a 2⁻⁴²⁴ tail, against σ_R + 7 ≈ 210 for the SSH exponent alone (same
table). **Test:** at D 300, the factored formulation's Δ against lead 1's 1.678171015417 ± 1.25×10⁻¹⁰. It fails if it
misses by more than 3E, or if ks must still exceed σ_R + 7 by more than the D3 margin.

**Lead 5. Δ₁ and Δ₂ exact in the harmonics, an independent check of the series (several days; CPU like the σ_k runs).**
Run the amplitude expansion (README "Without truncation: the amplitude expansion", `solver/inner_amp/`) on the O(1/n)
inner equations of FIRST_ORDER §1, with ω₁ = −S_{1/n}/∂_ωS and S_{1/n} in the adjoint form (L7). Every harmonic is
exact at each order in A, so the M-truncation's 0.3% disappears. **Test:** Δ₁ must land in Track C's slope range
10.37109–10.37529 (LARGE_D_WALL, Track C), and later Δ₂ in −49.16867 to −49.16780. A miss rejects either Conjecture 1 at
O(1/n) or Track C's premise. This is the only check of the 1/n series that shares no machinery with the chain. Shared:
the O(1/n) inner equations (FIRST_ORDER §1, checked symbolically against the outer expansion).

**Lead 6. Exponential integrators for the fast mode's transfer (LARGE_D_BASICS D4; a multishoot change, about a
week).** Treat the fast linear rates nc²/(x(c² − x²)) and (n − 1)/x exactly in each step (Hochbruck and Ostermann,
Acta Numerica 19 (2010) 209). The step count then stops growing with n. The plain plan needs |nλ_F h| ≲ 14, about
23 × (n/261) steps, about 880 at D 10⁴. **Test:** at D 262.57, the exponential plan with the chain's 23 steps against a
plain plan with |nλ_F h| ≤ 4 (T4). Pass: Δ within 10⁻¹². It fails if the slow part's error then dominates.

**Lead 7. Exact WKB for the free field: the error model of the series route (1–2 days of reading and derivation, no
CPU).** Aoki and Tanda's Borel-summed Voros coefficients for the Gauss equation with a large parameter give the free
field's connection coefficient uniformly in n, with its parametric Stokes jumps. **Test:** they must reproduce
`solver/basics/free_field.py`'s ratios at n 60–960 (LARGE_D_BASICS 3.4) and the modulation 2e^(−2πjω) sin²(πn/2). It
bounds lead 1's floor near 10⁻¹⁰ only, so it is low priority for 10⁻⁸.

**Lead 8. The complex-path condition (E2) at very large D (a research project).** Its digit cost is flat in n (70–85),
while (E1)'s grip grows as n^21.7. Relative to D 61, (E1) reaches 10⁷⁰ only near n ≈ 10⁵. So (E2) does not pay below
that, and its basin δ ≲ e^(−πkω) stays. **Test:** BEYOND_LIGHT_CONE's toy at n 1000 and 4000. Pass for (E2): κ_emp
stays flat. Not recommended for #1088.

## 6. References checked

| entry | checked against | verdict |
|---|---|---|
| Emparan, Herzog, RMP 92, 045005 (2020), arXiv:2003.11394 | the e-print's text layer; INSPIRE record per LIT_REVIEW.md #802 | OK |
| Bhattacharyya, De, Minwalla, Mohan, Saha, JHEP 04 (2016) 076, arXiv:1504.06613 | arXiv abstract listing (v4, 8 Mar 2016) and Springer DOI 10.1007/JHEP04(2016)076, from the search; ar5iv text | OK |
| Emparan, Suzuki, Tanabe, PRL 115, 091102 (2015), arXiv:1506.06772 | e-print text layer (v2); the journal reference from LIT_REVIEW.md | OK |
| Emparan, Luna, Martínez, Suzuki, Tanabe, JHEP 05 (2018) 104, arXiv:1802.08191 | e-print text layer (v4); journal per the search's DOAJ and Springer listings | journal *(unverified)* at the publisher |
| Rozali, Way, JHEP 11 (2018) 106, arXiv:1807.10283 | e-print text layer; INSPIRE per LIT_REVIEW.md #802 | OK |
| Clark, Pimentel, JHEP 11 (2025) 017, arXiv:2504.10590 | e-print text layer (v2); INSPIRE per LIT_REVIEW.md #802 | OK |
| Gomide, Guardia, Seara, Zeng, arXiv:2107.14566v3 | e-print text layer | arXiv OK; journal reference not found *(unverified)* |
| Gelfreich, Simó, DCDS-B 10 (2008) 511, doi:10.3934/dcdsb.2008.10.511 | zbMATH/MaRDI record | *(unverified)* at the publisher |
| Chapman, King, Adams, Proc. R. Soc. A 454 (1998) 2733 | the author's Oxford page and MaRDI (search results) | *(unverified)* at the publisher |
| Olde Daalhuis, Proc. R. Soc. A 461 (2005) 2503–2520 and 3005–3021 | search results quoting the papers | *(unverified)* |
| Aoki, Tanda, J. Math. Soc. Japan 68 (2016) 1099–1132 | J-STAGE and Project Euclid listings (search results) | OK |
| Koike, Publ. RIMS 36 (2000) 297–319, doi:10.2977/PRIMS/1195143105 | EMS Press listing (search result) | OK |
| Segur, Kruskal, PRL 58 (1987) 747 | APS issue listing and OSTI (search results) | OK |
| Hochbruck, Ostermann, Acta Numerica 19 (2010) 209–286, doi:10.1017/S0962492910000048 | Crossref and Cambridge listings (search results) | OK |

Chapman–King–Adams and Olde Daalhuis are named here as the standard late-order and hyperasymptotic methods for Stokes
constants of nonlinear ODEs. The project already uses their content (late orders, FIRST_ORDER §5), and the full texts
were not read for this note.

## 7. Commands

From `hp/`, with the venv of SETUP.md (python-flint 0.9.0, mpmath 1.3.0 here), lead 1:

```bash
nice -n 10 python3 -c "
import track_c as T
real = T.load_real(); fits = T.all_fits(real); D0, used = T.levels(fits)
print('points', len(real), 'D0', D0, 'fits used', len(used))
for D in [300, 400, 1000, 2000, 10000, 'inf']:
    e = T.estimate(fits, D)
    v = e['value'].mid().str(16, radius=False)
    print('D %-6s Delta %s  E %.2e  h %.2e  s %.2e' % (D, v, e['E'], e['h'], e['s']))
"
```

Its output on the Mac, 2026-11-25 (about a minute on one core):

```text
points 1349 D0 10 fits used 19
D 300    Delta 1.678171015416520  E 1.25e-10  h 1.25e-10  s 6.89e-12
D 400    Delta 1.669658064110633  E 2.31e-10  h 2.31e-10  s 3.46e-11
D 1000   Delta 1.654240854245244  E 7.84e-10  h 7.84e-10  s 2.79e-11
D 2000   Delta 1.649074313885640  E 1.20e-09  h 1.20e-09  s 3.70e-11
D 10000  Delta 1.644930989933633  E 1.70e-09  h 1.70e-09  s 4.61e-11
D inf    Delta 1.643893745321115  E 1.85e-09  h 1.85e-09  s 4.86e-11
```

Lead 2's ε^(7/2) and lead 3's table (arithmetic on the quoted inputs):

```bash
python3 -c "
import math
g0 = math.log10(3.4e21); n0 = 259.6 - 2
for D in (400, 2000, 10000):
    n = D - 2
    print(D, 'power law 10^%.1f' % (g0 + 21.7 * math.log10(n / n0)),
          'exponential 10^%.1f' % (g0 + 0.41 * (D - 259.6) / math.log(10)), 'eps^3.5 %.3g' % ((1 / (D - 1)) ** 3.5))
od = 424 * math.log10(2)
print('octo-double digits %.1f; exponential law limit with 30 working digits: D %.0f' % (
    od, 259.6 + (od - 30 - g0) * math.log(10) / 0.41))
"
```

The γ comparisons of §4.1 and §4.3, from the repository root:

```bash
python3 -c "
import json
for l in open('hp/data/gamma_dd.jsonl'):
    r = json.loads(l); D = r['D']; g = r['gamma']
    print('D %.4f gamma %.6f EH(10.3) %.6f diff %+.2e mass exponent (D-3)gamma %.3f CP %.3f' % (
        D, g, 0.5 * (1 - 1 / D), g - 0.5 * (1 - 1 / D), (D - 3) * g, 2 * (D - 3) / (D - 2)))
"
```

## 8. Gap statement and discrepancies

**Not read.** The full texts of Gelfreich–Simó, Aoki–Tanda, Koike, Chapman–King–Adams and Olde Daalhuis (abstracts and
reviews only); Clark–Pimentel's appendices on matching (C, D); arXiv:2107.14566 Secs. 4–10; the membrane papers beyond
the first 100k characters of arXiv:1504.06613; the charged and AdS membrane literature. No paper on the large-dimension
limit of self-similar blow-up in other equations (wave maps, Yang–Mills, harmonic map heat flow) was found by the
searches. Biernat and Bizoń's higher-dimensional work (arXiv:1409.8214, 1608.07707) treats fixed d.

**Discrepancies (reported, not fixed).**
- Clark–Pimentel's agreement sentence after (1.5) compares a mass exponent with a length exponent (§4.1).
- Emparan–Herzog's (10.3) is off our γ by 2.46×10⁻³ at D 61.03 and has 0.66 of our slope (§4.3).
- LARGE_D_WALL's literature row for the EFT says "no version for self-similar collapse was found". The inner problem in
  the Rozali–Way scaling is such a version, though not in the gravity literature (§2.2). The row's conclusion, that the
  EFT is no cure for the wall, stands.
- The coordinator's brief for this slice says "beyond our first order". SECOND_ORDER.md already has Δ₂ (§3.1).

**Proposed text** (for the coordinator; agents do not edit the README). For the README's large-D section, after the
sentence on Rozali and Way: "The large-D effective theory of black holes rests on a field localized within r₀/D of the
horizon. The critical solution has no such field: only the linear mass mode, (r/r₀)^−(D−3) with the self-similarity
horizon in the role of r₀, has that form, and the project factors it out exactly. The D-uniform large-D problem of the
critical solution is the inner problem in Rozali and Way's scaling."
