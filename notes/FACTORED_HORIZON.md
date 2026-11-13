# The factored horizon segment: which factor, and what a reformulation can change (#491)

#491 asked for a reformulation of the horizon segment that takes an exponential factor out of the unknowns, so that
the gain between the light cone and the last node "lives in the analytic factor, not in the numbers". Its comments
corrected the premise. The segment holds two exponents of order n, the mass mode's gain G and Proposition 3's fast
mode e^(−nΦ), and Newton's iterates do not change under a linear change of the unknowns. This note says which factor
is tied to the chain's weak directions, what a reformulation can change at all, and designs the factoring of each.

The checks are `solver/factored_horizon_sympy.py` (commands in section 7). Throughout, n = D − 2, x is the
self-similar radius with the light cone at x = 1, u = 1 − x, x_K = 0.7 is the chain's last node and u₀ = 0.3. The
status words are AGENTS.md's.

## Verdicts

| question | answer | status |
|---|---|---|
| which factor is tied to the weak directions | the fast mode, through the horizon series' truncation at ks = 130 | numerical evidence from a linear model (section 4); a conjecture for the chain until #696 and the test of 4.5 |
| can the mass mode's G set the weak σ | no: they follow e_Δᵀ J⁻¹, which no change of the horizon data moves | proved (section 2) given the rows unchanged; the identity checked symbolically (S6) |
| can G be factored out | yes, exactly: B = 1/a² obeys a linear equation, and its datum moves to the last node | checked symbolically (S1–S5) |
| the mass mode's gain before and after | x_K^−(n−1) e^(F²(1−x_K²)/n + O(n⁻²)) before; 1 after, for every n and x_K | checked symbolically (S2, S5) |
| the share of the fast mode's shadow that ks = 130 keeps | ln f = −41.05 at D 237.13 and −63.05 at D 253.80 (μ = 1); f = 1 to 1% from ks ≈ 1.9 Re s | numerical evidence (M3) |

The linear model reproduces three measurements with no fitted parameter (M4):

| quantity | measured | model, truncated at 130 | model, not truncated |
|---|---|---|---|
| change of ln σ₃ from D 237.13 to 242.36 (raw `I488-mac/` at 994f6f21) | −5.833 | −5.851 (μ = 1) | +0.132 |
| change of ln σ₁ over the same | −4.771 | −4.843 (μ = 3) | +0.117 |
| change of ln σ₃ for the last node 0.70 → 0.72 at D 242.36 (#537) | −8.96 | −8.92 (μ = 1), −8.93 (μ = 3) | −13.95 |

The order of work this suggests:
1. #696 (running) and the σ test of 4.5, which need no change to hp's code. If they support the truncation, the chain
   runs at ks ≈ 1.9 Re s.
2. The mass mode factored at the last node (section 3), for the strong side: the ε·G floors, the re-seed's polish and
   the corrector's quadratic jump. It needs an inner solve on the horizon segment, whose cost stage 1 of section 5
   measures.

## 1. The two exponents in the horizon segment

Lemma B2 (proof/C_selection.md §3.2) gives the outer system two fast eigenvalues, −nΦ′(x) for u = p + m, with
Φ = ln x − ½ ln(1 − x²), and −n/x for b − 1. They are the only exponents of order n in the equations.

| mode | form | per unit of n (E0) |
|---|---|---|
| fast mode (Proposition 3) | e^(−nΦ) | −Φ(0.7) = 0.020 at the last node; 1.137 between nodes 0.3 and 0.7 |
| mass mode | x^−(n−1) | 0.357 at 0.7 (1.07 for G³); G = 1.85×10³⁶ at D 237.13 and 1.2×10³⁷ at 242.36 |

**The mass mode.** It solves the linearized b equation, x δb_x = −(n − 1) δb at leading order
([LARGE_D_MIXED.md](LARGE_D_MIXED.md) 10). The horizon series starts from a_h at x = 1 and runs inward, so an error of
a_h, or a rounding error of the series, reaches the last node multiplied by G. The a_h directions of the horizon block
have gains equal to G ([RESEED.md](RESEED.md) 2.1), and the residual's floor is the rounding times G
([HORIZON_RADIUS.md](HORIZON_RADIUS.md) 2).

**The fast mode in the horizon segment.** The horizon series is the regular solution at x = 1. It cannot hold the
fast mode's own branch, (1 − x)^Λ with Λ not an integer (C_selection.md 5.1 and B6, step 3), but it holds its shadow.
In the linear problem (README, Proposition 2) the horizon-regular solution is ₂F₁(α, β; 2 + s − n/2; 1 − x²), whose
third parameter passes near zero at order Re Λ = n/2 − 1. Its terms past that order are large, and for x < 1 they sum
to a multiple of the fast mode (S7, M1). This is #693's transient: the low harmonics' terms grow past order
k_g ≈ Re s, peak near order 190–211 and fall again.
- To leading order in the order j, those terms are those of S(u) = u^k (2 − u)^k (1 − u)^(−2k−2) with k = Re s, the
  sum of C(j, k) w^j with w = 1 − x². Their mean order at u₀ is m = k[1 − u₀/(2 − u₀)] + (2k + 2) u₀/(1 − u₀) (S8):
  192.7 at D 225.13 and 216.8 at D 253.80, against #693's peaks at 189 and 211.
- The model of section 4 puts the peak within 2.6% of #693's at all 33 states (M2).

**The weak directions.** At the D 242.5 starts they span four dimensions with a clean gap (#529). The Δ axis lies in
that span to 0.9995, and the right vectors hold Δ, p1, m_h and node 4's p and m at harmonics 1 and 3. The left vectors
sit on the last matching, half on the Taylor arrival at node 4 and half on the horizon series there, on p and m at
harmonics 1 and 3 and on a at harmonics 0, 2 and 4 (#694, [REDUCED_CUBIC.md](REDUCED_CUBIC.md) 10.6).
- Theorem B counts one complex condition per solved harmonic, μ = 1 and 3 here (kfreeze = 3), on F₁, F₃ and Δ less
  the phase. That is four real directions, so the weak directions are the selection conditions (inferred).
- #693's transient lives on the same harmonics: p and m at 1, 3 and 5, a and c at 0, 2 and 4.

## 2. What a reformulation can change

Let z be the unknowns and F the rows.

- **A linear change**, z = Ty and F → SF (a rescaling, an equilibration, a fixed factor taken out of a field), leaves
  Newton's iterates unchanged in exact arithmetic, since T dy = dz (S6). It changes only the rounding, that is which
  numbers are stored and how the LU pivots.
- **The grip on Δ.** The row vector e_Δᵀ J⁻¹ is the change of Δ per unit change of each row. It is the same under any
  change of the other unknowns, linear or not, as long as Δ and the rows are kept (S6, at a point). Its norm is at
  least |P_V e_Δ|/σ₄, with V the four weak directions, and the new layout's smallest σ is at most the inverse of that
  norm. Since |P_V e_Δ|² = 0.99949 (#694), no reparameterization of the horizon data can lift the smallest σ above
  the old σ₄ by more than 0.03% (proved, given the rows unchanged).
- **A nonlinear change of the unknowns** gives the same first Newton step to first order and different iterates from
  second order on (S6). It changes the rows' curvature in the new unknowns, so Newton's basin and the corrector's jump
  after its first move.
- **A change of what is computed** (the series' truncation, or a closed form in place of a summed series) changes the
  discrete problem and can change everything, the weak σ included.

| what changes | linear rescaling | the mass mode factored (section 3) | ks raised (section 4) |
|---|---|---|---|
| Newton's iterates in exact arithmetic | no | from the second step | yes: another discrete problem |
| e_Δᵀ J⁻¹ and the weak σ | no | no | yes: by 1/f, e^41 at D 237.13 (μ = 1) in the model |
| σ_max and the LU's large pivots, of size G | the pivots only | the factor G leaves them (prediction) | no |
| the floors ε·G | no | removed (S3's demonstration; a prediction for the segment) | no |
| the horizon rows' curvature along the weak directions | no | its G-sized part removed (prediction) | no |
| the solution, Δ included | no | no: the same discrete problem | by the truncation's share (4.3) |

**The step bounds.**
- Newton's local bound in affine-covariant form (the Lipschitz constant of J⁻¹J′) is unchanged by a linear change and
  changed by a nonlinear one.
- The chain's acceptance test on max|F| depends on the rows' scaling. Deuflhard's natural level function ‖J⁻¹F‖ does
  not (Deuflhard 1974; [LIT_REVIEW.md](LIT_REVIEW.md), F4).
- The predictor's step a_max = (ε|z₁|/|z₂₄|)^(1/23) reads the branch's Taylor coefficients, whose largest entries are
  Δ's at D 245.14 (#634). A reparameterization does not change Δ's series, and a raised ks changes the branch only by
  the discretization's share.

## 3. The mass mode factored at the last node

### 3.1 The change of variables

In B = 1/a² and C = c a², hp's a and c equations (the header of `hp/src/model.rs`) become (S1)

```math
x\,C_x = C\,q, \qquad x\,B_x = (n-1)(1-B) - q\,B, \qquad q = p^2 + m^2,
```

with the coupling (n − 1) c (b − 1) = (n − 1) C (1 − B) and c = C B in the p and m equations. The c equation loses
its factor n − 1, and the B
equation is linear in B. Its homogeneous solution is the mass mode, and the equation is regular at x = 1, so B's datum
need not sit at the light cone. #538 used the same variables for the horizon pole ([LARGE_D_WALL.md](LARGE_D_WALL.md)).

### 3.2 The factored equations

The mass mode normalized at the last node is (S2)

```math
\chi(x) = \Big(\frac{x}{x_K}\Big)^{-(n-1)} \exp\!\Big(-\int_{x_K}^{x} \frac{q(t)}{t}\,dt\Big), \qquad \chi(x_K) = 1, \qquad B = B_s + \mu\,\chi ,
```

with B_s any particular solution and μ the mass mode's amplitude at x_K.

**The transport of the amplitude.** Proposition 3's method on the exact equations E1–E4, with Proposition 1's
background as in `solver/largeD_transport_sympy.py`, takes δb = x^(1−n) (β₀ + εβ₁) (S4).
- At leading order δc = −δb and δp = δm = −√ε F x^(1−n) β₀, so δq = 2(p δp + m δm) vanishes there.
- The b equation then gives, along θ = const,

```math
x\,\partial_x \beta = -2\varepsilon\, x^2 F(\theta)^2\, \beta + O(\varepsilon^2).
```

So the mass mode has no transport in θ at this order, where the fast mode runs along θ = θ₀ + ω ln(1 − x²). The factor
agrees with χ's exp(−∫q/t), since q = 2εx²F² + O(ε²) (S5).

**The gain** (S2, S5). From the light cone to the last node the mass mode grows by

```math
G = \frac{\chi(x_K)}{\chi(1)} = x_K^{-(n-1)} \exp\!\Big(\int_{x_K}^{1} \frac{q}{t}\,dt\Big) = x_K^{-(n-1)}\, e^{F^2 (1 - x_K^2)/n + O(n^{-2})} .
```

After the factoring the horizon datum is μ, or equivalently a_K below, whose gain to the node is χ(x_K) = 1 for every
n and x_K. Through the coupling (n − 1) C (1 − B) it reaches the p and m rows as δp = −√ε F μ at x_K (S4), of size
F/√n.

### 3.3 The series at the light cone

hp sums the horizon segment as series in u at x = 1, built by forward recurrences. In u, B's recurrence is

```math
(k+1)\, b_{k+1} = (n-1+k)\, b_k + (q\,b)_k - (n-1)\,\delta_{k0},
```

and hp's a and c series give B's and C's to every order checked (S3, to order 4). The homogeneous solution, the mass
mode, has the coefficients of (1 − u)^(−(n−1)) exp(∫₀ᵘ q/(1 − s) ds) (S3), whose sum at u₀ is G. A forward recurrence
from b₀ therefore carries any error of b₀, and its own rounding, to the node multiplied by G. In 30 digits, at n = 240
and K = 130, its sum at u₀ misses by 3.0×10⁵ (S3).

The factored series splits b_k = B_sl,k + γ χ̂_k.
- **B_sl, the minimal solution.** It solves rows k = 0, …, K of the recurrence with b_(K+1) = 0, one (K + 1) × (K + 1)
  system per τ point. The rows are diagonally dominant (n − 1 + k > k + 1 + Σ|q_j|), so the mass mode decays from high
  orders to low ones. Solving a recurrence backward for its minimal solution is Miller's method, not yet recorded in
  [LIT_REVIEW.md](LIT_REVIEW.md) (a recommended task of #491's closing summary).
- **χ̂**, the mass mode normalized so that Σ_(k≤K) χ̂_k u₀^k = 1, computed forward (the dominant solution is stable
  that way) or from its closed form.
- **γ**, one number per τ point, fixed by the value at the node. With γ = B_K − Σ_k B_sl,k u₀^k, B's sum at u₀ is the
  unknown B_K = 1/a_K².

For any γ this is exactly the forward recurrence's series for b₀ = B_sl,0 + γ χ̂₀ (S3, exact rationals at n = 37,
K = 10). Its sum at u₀ is good to the working precision, 9.8×10⁻³² in 30 digits (S3).

So in exact arithmetic the factored segment computes hp's discrete problem with the horizon datum a_h replaced by
a_K, and a_h = (B_sl,0 + γ χ̂₀)^(−1/2) becomes an output. The map between a_h and a_K depends on m_h and Δ through q,
so it is a nonlinear change of the unknowns. hp sums a's series, not B's, and the two truncated sums differ at the
truncation's level. Fixing γ by a's sum at u₀ equal to a_K (one scalar equation per τ point, by Newton) makes the
factored residual hp's residual composed with that change, exactly; fixing it by B's sum changes the discrete problem
at the truncation's level.

### 3.4 The coupling and its cost

q depends on p and m, which depend on B through the coupling and c, while B_sl depends on q at every order. The
factored segment is therefore a fixed point. Its sweep runs p, m and C forward with B given, then B_sl from q, and the
sweeps repeat to the working precision. By the orders of its terms the loop gain is of order 1/n (inferred). The
coupling's change −(n − 1) C δB drives p − m, then δq = 2p(δp − δm) and δB_sl ≈ −δq/(n − 1). That is about
ln(1/ε)/ln n sweeps, unless an accelerated or Newton inner solve cuts it.
Stage 1 of section 5 measures the gain. The current layout resolves the same coupling in one forward pass, and that
pass is what carries the rounding to the node.

### 3.5 The matching and the layout

- **Unknowns.** The M entries at `off_ah` hold a_K, the segment's a at the last node, in place of a_h. L stays
  4M(K + 1) + 1 = 769, and no other unknown changes.
- **Rows.** The horizon block's a rows become a_K − a(node K − 1), linear in both. Its p, m and c rows keep their
  meaning, the series at u₀ minus the node, computed from the factored fields with c = C B. The centre, Taylor and
  phase rows are unchanged.
- **The LUs.** `horizon_lus` takes c₁ = (n − 1)(1 − a_h²) from the derived a_h, in which a_K enters only at 1/G (S2).
  The LUs keep their double-double factorization.
- **The re-seed.** a_K is the node's own a, so a re-seed inverts only the m rows for m_h: the a_h half of RESEED.md
  2.2's square system disappears.

### 3.6 What it can and cannot change

It cannot change the weak σ or the grip on Δ (section 2), the first Newton move or the discrete solution. Nor can it
change the predictor's a_max while the series' largest entries are not a_h's (at D 245.14 they are Δ's, #634). It can
change, as predictions for the plan's checks:
- **The floors.** The residual's and the re-seed polish's floors ε·G, 4.5×10⁻³⁰ at D 225.13 to 1.8×10⁻²⁵ at D 253.80
  in quad-double (HORIZON_RADIUS.md 2), fall to ε times terms of order one, as S3 shows for B alone.
- **The conditioning.** σ_max, about 2.5 G (LARGE_D_MIXED.md 10), and the LU's large pivots (to 10²⁷ near D 194,
  [ALGORITHM.md](../ALGORITHM.md) 5) lose the factor G, so quad-double's ε·κ falls by about G.
- **The corrector's jump.** The horizon rows' curvature along the weak directions is 3–6×10³² against |J_r| ≈ 3×10³⁶
  (#694), and the residual after the corrector's first move is max|F| ≈ 1.0×10³³ |c|² (#499). Reading (inferred): a
  weak move changes Δ and m_h at fixed a_h, which moves the slaved horizon value of B quadratically, and G carries
  that to the node. With a_K as the unknown, a_h follows the slaved value and the term is absent.
- **The pole of the perturbed family** (QUAD_DOUBLE.md 13.6). A perturbation δ of a_K excites the mass mode with
  amplitude δ at the node, not δG, so 13.6's radius law, which is set by δG, does not apply to it (conjecture until
  P4 of section 5).

## 4. The fast mode's shadow and the truncation at ks

### 4.1 The selection seen through a truncated series

In the linear problem, write the centre-regular solution as y_reg = A y_h + C y_f, with y_h the horizon-regular
solution, y_f the fast mode and C Proposition 2's light-cone coefficient (Abramowitz & Stegun 15.3.6, as in the
README). The matching at x_K needs y_reg parallel to y_h, and det(y_reg, y_h) = −(C/A) det(y_reg, y_f). So the grip is
C's derivative times the fast mode at the node, a power of n times e^(−nΦ(x_K)) by Lemma B5.1's κ_n. A series truncated
at K keeps a share f of y_h's fast part, so (S7)

```math
\det\big(y_{\rm reg},\; y_h^{(K)}\big) = -\,\frac{f\,C}{A}\,\det\big(y_{\rm reg},\; y_f\big),
```

and the grip on Δ is f times the continuum's. f stays near 1 while ks lies past the shadow and falls exponentially as
Re s approaches ks.

### 4.2 The linear model

`factored_horizon_sympy.py model` sums the horizon-regular series of Proposition 2's equation in u to order 700 at
300 digits, at each of #693's 33 states.
- Its near-resonance is set at the state's order: n_lin = 2(Re s + 1), with Re s from `hp/data/horizon_radius.csv`.
- s = iμω, ω = 2π/Δ with Δ from `hp/data/delta_all.csv`, and u₀ = 0.3.
- f = W_K/W_∞ is the share of the continuum's matching determinant that the series truncated at K keeps.
- Nothing is fitted. The series equals ₂F₁ to 9×10⁻¹³⁴ (M1).

| D | Re s | #693's peak order | the model's peak (μ = 1) | ln f at ks 130, μ = 1 | μ = 3 | ln f at ks 200, μ = 1 |
|---|---|---|---|---|---|---|
| 225.13 | 114.16 | 189 | 191 | −29.66 | −27.04 | −0.24 |
| 237.13 | 120.17 | 196 | 201 | −41.05 | −37.02 | −0.80 |
| 241.96 | 122.58 | 204 | 205 | −46.55 | −41.58 | −1.17 |
| 249.01 | 126.11 | 210 | 211 | −55.79 | −48.78 | −1.90 |
| 253.80 | 128.51 | 211 | 215 | −63.05 | −53.96 | −2.53 |

At ks = 250 the model keeps f = 1 to 0.005 at every state. The verdicts' second table compares it with the weak σ,
pairing σ₃ (mostly Δ) with μ = 1 and σ₁ (the harmonic-3 pair) with μ = 3, an assumption of the comparison.
- The fast-mode law alone, −n[Φ(0.72) − Φ(0.70)] = −13.66, is the untruncated column. The truncated series recovers
  5.0 of it, because the shadow peaks at lower orders for the nearer node. That is why #537's shift was 65% of the
  law ([LARGE_D_WALL.md](LARGE_D_WALL.md)).
- With the truncation removed (`--mutate=ks`), both comparisons of M4 fail.

### 4.3 Readings (inferred, not shown)

- **The rate.** The weak σ fall in D at the model's rate d ln f/dD, which grows with D (the table). At a fixed layout
  the equations' exponents cannot give it. The fast mode at the last node changes by 0.020 per unit of n (E0), and G
  cannot set the weak σ (section 2).
- **The node law of D 40–61.** There Re s ≈ n/2 lies far below the series' order, f = 1, and the grip follows
  e^(−nΦ(x_m)) alone. That is the law that matched max|∂Δ/∂F| over 13 orders (LARGE_D_WALL.md); the runs' own ks is
  to be checked.
- **Only the last node is felt.** f depends on u₀ and ks, and e^(−nΦ) on x_K. Neither depends on the other nodes
  (#543, #544).
- **Δ's discretization error.** If the discrete solution meets the selection only up to its other discretization
  errors ε_d in the fast direction, its light-cone coefficient is about −ε_d/f, and Δ's error grows like 1/f. #693's
  transient past ks = 130 then holds that residual, and raising ks moves Δ by about T_omit/σ_∞: T_omit is the omitted
  transient and σ_∞ = σ₃/f the grip without truncation. M5 gives 2.7×10⁻¹² at D 245.14 and 1.4×10⁻⁷ at D 253.80,
  good to perhaps a factor 100 (conjecture; #696 measures it, and a change below 10⁻¹⁴ at D 253.80 refutes it).

### 4.4 Keeping the shadow

The fast mode carries the selection, so it cannot be factored away. "Factoring it out of the horizon segment" can only
mean keeping its shadow in the discrete problem, in one of two ways.
- **Raise ks past the shadow.** |ln f| < 0.01 needs K = 218 (μ = 1) at D 225.13 and 244 at D 253.80, and, with Re s
  extrapolated linearly in D, 285 at D 300 and 374 at D 400 (M3). That is 1.85–1.91 Re s, or the shadow's mean order m
  plus 2.3–2.4 of its standard deviations (S8). It needs no change to hp's code (`anm_run.py --ks`, #696), and it
  costs the horizon block about ks/130 times its present work.
- **Sum the tail in closed form** from Proposition 3's transport, normalized at the last node. It is not designed here,
  and it would change the discrete problem by the WKB error, O(1/n) (C_selection.md B6).

### 4.5 The σ test (criteria fixed before any run)

- **Inputs.** The chain's states at D 237.13 and 242.36 (`I461-win2/crawlI/anm_pin3_D237.1312.npz`,
  `I461-win2/crawlJ/anm_pin3_D242.3642.npz`), re-polished at ks = 130, 160, 200, 250 and 300, then σ₁–σ₄ by
  `hp/near_null.py` as in #488.
- **Supports** the truncation: ln σ₃ rises with ks by the model's −ln f (μ = 1) within 30% at ks = 160 and 200, and
  changes by less than 1 from ks = 250 to 300.
- **Refutes** it: ln σ₃ changes by less than 1 between ks = 130 and 250.
- **Control.** The same at a state whose Re s lies far below 130 (D 60): ln σ changes by less than 0.05.

## 5. Plan for hp's model and the planted tests

hp/src is not changed by this task.

**The truncation (first).**
1. #696 and the σ test of 4.5, on the driver alone.
2. If they support it, ks follows D in the chain: ks ≥ ⌈1.95 Re s⌉ + 10, with Re s = n/(2(1 + c₁)) − 1 from the
   state's c₁ (HORIZON_RADIUS.md). The octo-double jets and the corrector take ks from the context (`Ctx::ks`).

**The mass mode (section 3).**
- **Stage 1, a prototype** of the horizon block alone in mpmath, on #693's two-node layout: the factored segment at a
  stored state, with the coupling solved by the fixed point. It measures the loop gain and runs P1 below.
- **Stage 2, hp's model.** A flag of `Ctx`, off by default and bit-identical when off:
  - `horizon_series` carries B and C, with B = B_sl + γ χ̂: B_sl from the lower-Hessenberg system of 3.3 per lane, χ̂
    forward, and the fixed point over the forward p, m, C pass and the B_sl solve, to the lane's precision. Values,
    duals and jets iterate alike, so the Jacobian and the jets are those of the converged map.
  - `horizon_state` takes the a rows from a_K. `horizon_lus` takes c₁ from the derived a_h.
  - The slot at `off_ah` holds a_K, and the state files say so. multishoot needs no change, since the right endpoint
    map still takes two fields.
  - `hp/reseed.py` sets a_K = a(node) and inverts only m_h.
- **Stage 3, the checks**, each able to fail:
  - **P1, equivalence.** At stored states, with a_K set to the old series' a at u₀, the factored residual equals the
    old one to the old floor ε·G. Control: χ̂ with the transport's sign flipped, as `--mutate=sign`, must fail.
  - **P2, rounding.** The factored residual in quad-double and in octo-double differ by at most 10³ ε_qd, where the
    old ones differ by about ε_qd·G. Control: the forward B recurrence put back must fail it.
  - **P3, the gain (planted).** δ χ̂ added to a converged state's data moves the a rows by δ to 10⁻¹⁰ and the p and m
    rows by −√ε F δ to 1/n, where the old layout's a_h move gives G.
  - **P4, the basin.** Newton from a_K moved by 10⁻⁸, 10⁻⁴ and 10⁻² converges at D 73 and 242.36, where the old
    layout's basin is about exp(−(u₀ − R₀)(D − 3)/C) (LARGE_D_WALL.md).
  - **P5, the strong side at the D 242.5 starts.** `reduced_cubic.py rows` on the factored layout gives the horizon
    rows' q_r along V below 10⁻²⁰ of the old 3–6×10³², and `near_null.py` gives the weak σ within 1% of the old ones.
    A larger change of the σ refutes the premise that the rows and the discrete problem are unchanged.
  - **Bit-identity when off.** `highm_check.py ref`, `scaled` and `jets`, as AGENTS.md lists.

## 6. Discrepancies (reported, not fixed)

1. **The issue's premise.** G = 0.7^−(D−3) is the mass mode's gain, not the fast mode's, as #491's first comment says.
   e^(−nΦ) is O(1) at the last node, and in the horizon series it appears as the transient past order Re s.
2. **"σ₃ falls by about G³ per unit of D"** (#491's first comment). G is a property of the horizon data's
   parameterization and cannot set the weak σ (section 2). The truncated shadow gives the rate (M4).
3. **QUAD_DOUBLE.md 13.3** says the mass mode's floor G "is the problem's conditioning, not the block's". B's equation
   is regular at x = 1 and its homogeneous solution decays outward, so its datum can sit at x_K (S1–S3). G is the
   conditioning of every layout whose horizon data sit at x = 1.
4. **[LARGE_D_LITERATURE.md](LARGE_D_LITERATURE.md) (f)** says no formulation that evaluates the perturbed family on
   [0, u₀] escapes the pole's basin. A family perturbed at x_K excites the mass mode by δ, not δG (S2), so it may
   escape; P4 tests it.
5. **"Neither model holds"** for #537's last-node shift (LARGE_D_WALL.md). The truncated fast mode reproduces it,
   −8.92 against −8.96.
6. **HORIZON_RADIUS.md's discrepancy 3** reads ks = 130 as omitting a tail. By the model it also cuts the selection's
   grip by e^29.7 at D 225.13 to e^63 at D 253.80 (μ = 1), and may amplify Δ's discretization error by as much.
7. **The σ table of #491's second comment** is in no repository file. `I488-mac/` holds D 237.13 and 242.36 only, so
   section 4 compares with those two.

## 7. Commands

From the repository root (sympy 1.14, mpmath 1.3):

```
python3 solver/factored_horizon_sympy.py                      # 30 checks pass, exit 0 (about 70 s)
python3 solver/factored_horizon_sympy.py check --mutate=sign   # 8 checks FAIL, exit 1
python3 solver/factored_horizon_sympy.py model --mutate=ks     # M4's two checks FAIL, exit 1
```

`check` runs E0 and S1–S8 alone, `model` S8 and M1–M5. M2–M4's tolerances were set after a prototype run, so they
describe the agreement rather than test it; the tests are 4.5 and #696. M5 prints the estimate of 4.3 and checks
nothing.

## 8. Prior art used

- Proposition 3 (README): the fast mode's form and transport, and the method of S4.
- Proposition 2 (README) and the connection formula (Abramowitz & Stegun 15.3.6): the linear model and S7.
- proof/C_selection.md: Lemma B2 (the two exponents), B6 (the light-cone exponent is not an integer), Lemma B5.1 (κ_n,
  a power of n) and Theorem B's count.
- [LIT_REVIEW.md](LIT_REVIEW.md) C3 and Q1, the WKB entries: nΦ as the fast mode's WKB exponent (Q1), and the
  Stokes reading of Propositions 3–4, in which the selection is the coefficient of that mode (C3).
- Deuflhard (1974), LIT_REVIEW.md F4: the natural level function ‖J⁻¹F‖, invariant under the rows' scaling.
- Miller's backward recurrence for the minimal solution: named, not yet recorded in LIT_REVIEW.md.
