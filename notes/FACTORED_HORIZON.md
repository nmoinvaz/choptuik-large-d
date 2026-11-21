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
| which factor is tied to the weak directions | the fast mode, through the horizon series' truncation at ks = 130 | numerical evidence from a linear model (section 4); for the chain, order 131 moves Δ by 3.8×10⁻¹³ at D 245.14 and order 132 does not converge (4.6) |
| can the mass mode's G set the weak σ | no: they follow e_Δᵀ J⁻¹, which no change of the horizon data moves | proved (section 2) given the rows unchanged; the identity checked symbolically (S6) |
| can G be factored out | yes, exactly: B = 1/a² obeys a linear equation, and its datum moves to the last node | checked symbolically (S1–S5) |
| the mass mode's gain before and after | x_K^−(n−1) e^(F²(1−x_K²)/n + O(n⁻²)) before; 1 after, for every n and x_K | checked symbolically (S2, S5) |
| the share of the fast mode's shadow that ks = 130 keeps | ln f = −41.05 at D 237.13 and −63.05 at D 253.80 (μ = 1); f = 1 to 1% from ks ≈ 1.9 Re s | numerical evidence (M3) |
| does the factored segment's inner solve reach quad-double in 5 sweeps | no: 29 to 32 from the slaved value and 21 from a warm start at D 242.36; at D 73, 31 from a warm start, and the sweeps diverge from the slaved value | numerical evidence (5.1) |
| does holding a_K remove the G-sized part of the horizon rows' curvature | yes, along the three moves tested at D 73.0 and D 242.36: Q_fact / Q_old is 1.3×10⁻¹⁹ to 2.7×10⁻¹⁶ and 4.5×10⁻⁶⁴ to 5.1×10⁻⁶⁰ | numerical evidence (5.2) |

The linear model reproduces three measurements with no fitted parameter (M4):

| quantity | measured | model, truncated at 130 | model, not truncated |
|---|---|---|---|
| change of ln σ₃ from D 237.13 to 242.36 (raw `I488-mac/` at 994f6f21) | −5.833 | −5.851 (μ = 1) | +0.132 |
| change of ln σ₁ over the same | −4.771 | −4.843 (μ = 3) | +0.117 |
| change of ln σ₃ for the last node 0.70 → 0.72 at D 242.36 (#537) | −8.96 | −8.92 (μ = 1), −8.93 (μ = 3) | −13.95 |

The order of work this suggests:
1. The σ test of 4.5, and a corrector that converges past ks 131 (4.6: #696 stopped at 132). If they support the
   truncation, the chain runs at ks ≈ 1.9 Re s.
2. The mass mode factored at the last node (section 3), for the strong side: the ε·G floors, the re-seed's polish and
   the corrector's quadratic jump. It needs an inner solve on the horizon segment. Stage 1 of section 5 measured it at
   21 to 32 sweeps of the old pass (5.1), so this route is not recommended with that solve. Holding a_K removes the
   horizon rows' G-sized curvature (5.2), so a one-off factored raise is worth building.

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
mode, has the coefficients of (1 − u)^(−(n−1)) exp(∫₀ᵘ q/(1 − s) ds) (S3), whose sum at u₀ is 3.2's G. A forward
recurrence from b₀ therefore carries any error of b₀, and its own rounding, to the node multiplied by G. In 30 digits,
at n = 240 and K = 130, its sum at u₀ misses by 3.0×10⁵ (S3).

The factored series splits b_k = B_sl,k + γ χ̂_k.
- **B_sl, the minimal solution.** It solves rows k = 0, …, K of the recurrence with b_(K+1) = 0, one (K + 1) × (K + 1)
  system per τ point. The rows are diagonally dominant (n − 1 + k > k + 1 + Σ|q_j|), so the mass mode decays from high
  orders to low ones. Solving the rows with b_(K+1) = 0 is the boundary-value form of backward recurrence. For a
  first-order inhomogeneous equation it is Gautschi's backward recursion from y_ν = 0 (1997), and for second order it
  is Olver's algorithm (1967), which he regards as a generalization of Miller's (1952) ([LIT_REVIEW.md](LIT_REVIEW.md),
  addendum of #733, B4 and B7).
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
  e^(−nΦ(x_m)) alone. That is the law that matched max|∂Δ/∂F| over 13 orders (LARGE_D_WALL.md). Its 15 runs used
  ks = 160 and their scan states ks = 100, at least 3.19 Re s from D 40.02 to 60.00, so none lies below 1.9 Re s
  (`hp/ks_vs_res.py`, #734).
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

### 4.6 Raising ks at D 245.14 (#696, #743, #744)

`anm_run.py --ks K` re-converged #603's state at D 245.1366 with the horizon series summed to order K (win2; raw
`I696-win2/` f39f3177, `I743-win2/` e50463e4, `I744-win2/` 2d44081d). Δ is quoted only from converged states.

| run | ks, plan | Newton steps, final max\|F\| | Δ − Δ(chain state) |
|---|---|---|---|
| r130 | 130, stored | 1, 2.0×10⁻⁶⁷ | 0 |
| r130k | 130, planned again | 5, 3.6×10⁻⁸⁶ | +2.374×10⁻¹⁴ |
| r131k | 131, planned again | 9, 1.2×10⁻⁸⁵ | +4.062×10⁻¹³ |
| s132, s132e | 132, from r131k's base, without and with `--elim_switch 0` | 20, 4.1×10⁻²⁴ and 4.5×10⁻²⁴ | not converged |
| r300 | 300, stored | stopped at its start, 1.4×10⁻¹⁷ | not converged |

**The order-131 term moves Δ by 3.8×10⁻¹³ at D 245.14**, sixteen times the plan's own effect (2.4×10⁻¹⁴). Order 132
does not converge: its first full step reaches max|F| 1.18×10²⁰ (|dw| 5×10³⁰), and the corrector never recovers from
it. The failed projections counted at that step all sit at trials that had already blown up, so they are a symptom
(#743). At D 253.80 even the re-planned ks 130 does not converge, so no ladder ran there.

The order-132 term weighted by w (`--ks 132 --ks_w W`, branch `wip/i744`, each rung from the last's base) converges in
5 steps at w = 1/16, 1/8 and 1/4, where Δ − Δ(w = 0) is +3.302×10⁻¹³, +6.452×10⁻¹³ and +9.7×10⁻¹³. At w = 1/2 it
stops at the 20-step cap at max|F| 2.6×10⁻²⁴. Its control holds (w = 0 equals ks 131 to the bit), and the jump from
w = 0 to 1 reproduces s132's first step. So the climb gets further than the jump, and by w = 1/4 order 132 has already
moved Δ by 2.5 times order 131's whole term. So the horizon series' truncation moves Δ at D 245 by 10⁻¹² within a
quarter of one order (numerical evidence, one plan and one layout).

#783 (win2; raw `I783-win2/` 36ed0c10) climbed on from w = 1/4 in smaller steps. w = 3/8, 7/16 and 1/2 each converge in
5 steps (max|F| 3.2×10⁻⁸², 1.6×10⁻⁹⁰, 5.3×10⁻⁹¹, no failed projection), so #744's stall at w = 1/2 came from the size of
its step, not from a missing root. Δ − Δ(w = 1/4) is +8.76×10⁻¹⁴, +9.37×10⁻¹⁴ and +8.80×10⁻¹⁴ there. The order-132
term's effect thus levels off near 1.06×10⁻¹² from w = 0, most of it gained by w = 1/4. Two rungs between (#797; raw
`I797-win2/` 4ecf7009) converge in 5 steps each and put Δ − Δ(w = 1/4) at 8.761, 9.256, 9.368, 9.192 and 8.799×10⁻¹⁴
for w = 3/8, 13/32, 7/16, 15/32 and 1/2: a smooth turn with its peak near w = 7/16, 6×10⁻¹⁵ high, against a last step's
dΔ of at most 4×10⁻⁵¹ at each rung. #796 (raw `I796-win2/` 378653ca and 20070baf) climbed on: w = 5/8 converges in 5 steps; w = 3/4 stalls at the 20-step cap with
θ falling about 8 times per step, and resumed from its own base it converges in 4 more; 7/8 converges in 5 and w = 1 in
19 (λ down to 1/32 in its first 8 steps, 18 failed projections all at blown-up trials). A cold polish of the w = 1 base
at w = 1 converges in 1 step to 6.9×10⁻⁷⁵. **So the climb gives a converged order-132 state at D 245.14, the first past
ks 131, and Δ(132) − Δ(131) = 9.262×10⁻¹³ there**, after Δ(w) peaked near w = 1/2 at 1.058×10⁻¹² (not monotone in w).
Track A climbs the whole tail the same way (4.7).

### 4.7 Track A: the tail homotopy (#800; design, criteria fixed before any code)

**Why the tail and not the jump.** By 4.2–4.3 the weak σ fall in D at the rate the truncation at ks = 130 loses the
selection's grip, so raising ks to about 1.9 Re s (4.4) should restore the grip and the conditioning together. The jump
does not get there: from the ks-130 state the first Newton step at ks 132 reaches max|F| 1e20 and at ks 300 the run never
leaves its start (#696). The weighted climb in one order converges rung by rung (#744, #783, #796). So the first stage
generalizes it from one order to the whole tail.

**The option.** `Ctx` gets `ks_w_from`: the horizon series' orders ks_w_from..ks are multiplied by ks_w (today's
`--ks_w` is the case ks_w_from = ks). At ks_w = 0 the system is the order-(ks_w_from − 1) truncation bit for bit, and at 1
the order-ks one. A rung from w to w' changes the residual by (w' − w) times the tail's terms, about 2×10⁻¹² in the p
rows at D 253.8 (#693), so the rungs stay small where the weak σ is small and may grow as the grip returns.

**Stage 1, at D 245.1366** (win2's build of wip/i744, extended): the tail 131..250 climbed in w from #696's r130k base
(re-planned, converged there). Criteria: (1) w = 0 equals r130k to the bit; (2) each rung converges, or the first that
does not is reported with its projection lines; (3) at w = 1, Δ against the single-order climb's (#796) and the model's
estimate of T_omit/σ_∞ (4.3, M5: 2.7×10⁻¹² at D 245.14, a conjecture good to a factor 100); (4) σ₁–σ₄ (`near_null.py`)
at w = 0, 1/2 and 1: the reading of 4.3 holds if σ₃ rises with w by the model's −ln f (μ = 1) within 30%, and fails if
it changes by less than a factor 2.

**Stage 2, at the fold** (only if stage 1 meets (2) and (4)): the same climb at #603's state D 259.4928 to ks ≈ 1.9 Re s,
then the series there (#776's command) and #782's estimators. The fold is the truncation's if D* moves by more than
0.05 or the exponent leaves the square-root band; it is the continuum's if D* stays within the two plans' 0.01.

**Stage 3** (only if stage 2 removes the fold or moves it past the chain's reach): the chain run on with the climbed ks.

Not chosen for now: the tail summed in closed form from Proposition 3's transport (4.4), which changes the problem by
the WKB error, O(1/n), and would need its own planted tests. It stays the fallback if stage 1 fails (2).

**Stage 1's result: (1) and (2) failed, and the climb stopped at its first rung** (dh370, raw `I800-dh370/` f2bdf5c9).
- **(1), the w = 0 control, failed for a reason of the design.** `--ks 250` plans the base again, because a stored plan
  does not carry a changed `--ks`. So t0 solved order 130's equations on another plan. r130k and t0 both have 23 steps
  but differ in their segments. t0 started at |F| 6.05×10⁻³⁶, and its first correction moved Δ by +4.70×10⁻¹³ through
  pivots down to 1.1×10⁻²⁶. That is a plan's effect at D 245.14, twenty times r130k's own re-plan (2.4×10⁻¹⁴, 4.6),
  as large as order 131's term, and 2% of #805's envelope there (2.0×10⁻¹¹).
- **(2), the first rung, failed in its first step.** The tail of orders 131 to 250 adds 1.3×10⁻¹⁷ to the horizon rows
  at t0's base, in octo-double on the chain's solver (`diag800b.py`), and `horizon_radius.py` reads the same.
  F(w) − F(0) is linear in w. So w = 1/64 starts near 2×10⁻¹⁹. Its first full step reached max|F| 9.4×10⁴⁴ (|dw| 3.9×10⁴⁴), where
  t0's reached 4.2×10⁻²⁶, and every trial from λ = 1 down to 1/128 was refused.
- **The corrector's basin, not the tail's size, ends the climb.** A rung's start residual is Δw times the tail.
  #744's order-132 climb gives the basin's scale. That order's term is about 4×10⁻²⁹ at u₀ (t₁₃₀ at this base), so its
  rungs of 1/16 started near 2.5×10⁻³⁰ and converged, and the rung of 1/4 to w = 1/2 (about 10⁻²⁹) stalled. The
  whole tail is 10¹² times that, so a homotopy in w would need about 10¹² rungs. Stage 1 is not continued. What
  remains is a corrector with a larger basin, or the closed-form tail (4.4).

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

### 5.1 Stage 1's result: the prototype in mpmath (#1035)

**The fixed point needs 21 to 32 sweeps to quad-double, not 5, so stage 1 fails its criterion and stage 2 is not worth
building with this inner solve.** At D 242.36 the sweeps from the slaved start reach eps_qd = 2⁻²¹² in 32 (plain), 29
(Anderson mixing, memory 5) and 30 (memory 20), and from a warm start in 21. At D 73 a warm start takes 31, and the
sweeps diverge from the slaved start. Once the error is small it falls by a factor of order 1/n per sweep, as 3.4
inferred, and the mixing does not change that. P1 holds against hp's series with an exact 1/k. With hp's own 1/k it
holds at D 242.36 and fails at D 73, by that rounding alone, and it fails at D 242.36 too once a_h is moved off the
stored value. All of it is numerical evidence at two stored states, from `hp/factored_horizon.py`.

| | D 73.0 | D 242.36 |
|---|---|---|
| state (raw-data repository) | `T81/conv_pin5_D73.0000_D73.000000.npz` | `I461-win2/crawlJ/anm_pin3_D242.3642.npz` |
| G = x_K^−(n−1) | 7.0×10¹⁰ | 1.2×10³⁷ |
| sweeps to eps_qd from the slaved start (B = 1, γ = 0), plain / Anderson | diverge (1.3, 1.5×10³, 1.5×10³) | 32 / 29 (memory 5, by \|g(y) − y\|), 30 (memory 20) |
| sweeps to eps_qd from a warm start (data moved by 10⁻²⁰), Anderson | 31 | 21 |
| error ratio per sweep, geometric mean of the last six (1/n) | 1.2×10⁻², warm (1.4×10⁻²) | 7.3×10⁻³ warm, 1.2×10⁻² slaved plain (4.2×10⁻³) |
| one sweep of the linearized map on a random perturbation (`loop`), then the next four | 1.5×10⁶, then 0.12, 0.059, 0.037, 0.026 | 8.7×10²⁰, then 0.021, 0.010, 0.0066, 0.0048 |
| γ, the mass mode's amplitude at the node, at the stored state | 9.5×10⁻²⁸ | 4.8×10⁻³⁵ |
| P1 with hp's 1/k: max \|R_fact − R_old\| (eps_qd G) | 1.4×10⁻³⁷ (1.1×10⁻⁵³): fails | 7.3×10⁻³⁹ (1.8×10⁻²⁷): passes |
| P1 with an exact 1/k in both (eps_qd G; 100·2⁻²⁸⁸ G) | 3.2×10⁻⁸⁰ (1.1×10⁻⁵³; 1.4×10⁻⁷⁴): passes both | 4.4×10⁻⁷⁰ (1.8×10⁻²⁷; 2.4×10⁻⁴⁸): passes both |
| P1 with a_h moved by 10⁻⁸/G (γ about 10⁻⁸), hp's 1/k; with an exact 1/k; the control `--mutate=sign` | not run | 6.7×10⁻¹⁸; 9.9×10⁻⁷⁰; 4.5×10⁻¹² |
| one sweep, and the old series, in the prototype at 288 bits | 12 to 14 s, 5 to 10 s | 13 to 18 s, 5 to 10 s |

**The prototype.** It transcribes `horizon_series` and `horizon_state` for the two-node layout into mpmath, with hp's
constants: n and the folded d/dθ rounded to double-double as hpkernel's driver makes them, 2π in double-double,
u₀ = 1 − f64(0.7) exactly, and 1/k as `recip_int` forms it. Its periodic solves are exact at the working precision
(hp refines its LUs to the lane's precision, so `horizon_lus` only preconditions them). The factored segment carries
β = B − 1, so that 1 − B does not cancel, and C. It takes c from its own recursion with the coupling (n − 1) C (1 − B)
and forms p and m as hp does. One sweep is that forward pass, then B_sl (the lower-Hessenberg system of 3.3 with
b_(K+1) = 0), χ̂ (forward) and γ per τ point. γ makes a's sum at u₀ equal to a_K, with a = B^(−1/2) from J. C. P.
Miller's power recurrence, by Newton's method. The slaved start is B = 1 and γ = 0. Its planted checks (`selftest`,
n = 100, M = 8, K = 40) pass, and each of their five controls fails the check it guards.

**The transcription matches hp.** At #603's states, whose octo-double rows #693 recorded (raw `I693-mac/onsol.jsonl`),
the old series' largest m, a and c rows agree with those rows to within 0.2%.

| state (raw `I603-win2/`) | m | a | c | #693's m, a, c |
|---|---|---|---|---|
| `base/base_pin3_D225.1312.npz` | 1.240×10⁻⁹⁵ | 3.228×10⁻⁹⁵ | 1.247×10⁻⁹⁵ | 1.242, 3.228, 1.248 (×10⁻⁹⁵) |
| `chain/anm_pin3_D241.9602.npz` | 2.780×10⁻⁶⁶ | 9.984×10⁻⁶⁶ | 1.996×10⁻⁶⁵ | 2.780×10⁻⁶⁶, 9.984×10⁻⁶⁶, 1.996×10⁻⁶⁵ |

There the p rows on the harmonics k ≤ kfreeze are 4.2×10⁻⁹⁵ and 1.5×10⁻⁶⁶. The control `--mutate=nh` (n/2 times
1 + 10⁻⁸ in the p and m equations) puts the rows at 8.9×10⁵¹. At the two states of the first table the m, a and c rows
are 2.6×10⁻⁵⁴ to 1.3×10⁻⁵³ (D 73, whose record gives max\|F\| 2.5×10⁻⁵³) and 6×10⁻⁷⁴ to 4×10⁻⁷³ (D 242.36). With an
exact 1/k in place of hp's the largest row is 4.4×10⁻³⁰ and 8.4×10⁻⁷ instead, so hp's rounding of 1/k belongs to its
discrete problem (6, item 9).

**Why the loop is slow.** With B given, the forward pass is an initial-value recursion for p and m, and past order Re s
it carries #693's transient, the fast mode's shadow (section 1), which grows. So a change of B near order Re s reaches q
at the top orders magnified. At D 242.36, where Re s is 119 to 126 over τ, `loop` moves y by 10⁻³⁰ at order 100, and p
at order 130 moves by 1.0×10⁻⁷ in u₀^k units. B_sl's backward recursion then returns q's top orders to the lower ones
with weights near 1 in those units, since (k + 1)/((n − 1 + k) u₀) is about 1 near the mass mode's peak order, (n − 1)
u₀/(1 − u₀) ≈ 102 there. The linearized map is far from normal. One sweep magnifies a random perturbation by 8.7×10²⁰ at
D 242.36 and 1.5×10⁶ at D 73, and the later sweeps shrink it. From the slaved value at D 242.36 the sweeps never meet
that growth, and they converge by a factor of order 1/n per sweep. Anderson's mixing saves two or three sweeps at most,
with memory 5 or 20, so the slow part is not a few directions. At D 73, Re s is 35 to 41 over τ, far inside K = 130, and
y moved by 10⁻³⁰ at order 20 moves p by up to 2.4×10⁻²² further up. There the slaved start's error, β itself (2×10⁻³ at
order 0), drives the transient, and the sweeps diverge, as they do from the naive start (q = 0, and γ from the datum,
where \|g(y) − y\| goes 0.56, 34, 34, 34). Started from 1/a² of the old series itself they converge, to eps_qd in 20
sweeps and to 2×10⁻⁷³ in 25.

**P1.** With a_K set to the old series' a at u₀, the factored segment recovers the stored a_h, and its rows equal the
old ones up to hp's rounding of 1/k.
- With an exact 1/k in both series they agree to the fixed point's own floor, 4.4×10⁻⁷⁰ at D 242.36 and 3.2×10⁻⁸⁰ at
  D 73, and the derived a_h equals the stored one to 4×10⁻⁸⁷ at both.
- With hp's 1/k the old a series carries a rounding of 2⁻¹⁰⁶ per order that B's rows do not have. Its mass-mode part
  moves the derived a_h by 4.5×10⁻⁴⁴ at D 242.36 and 3.5×10⁻⁴¹ at D 73. The rest is the difference left, 7.3×10⁻³⁹
  and 1.4×10⁻³⁷, so P1 passes at D 242.36 (eps_qd G = 1.8×10⁻²⁷) and fails at D 73 (1.1×10⁻⁵³).
- With a_h moved by 10⁻⁸/G the old series holds a mass mode of about 10⁻⁸ at the node, and with hp's 1/k the p rows
  then differ by 6.7×10⁻¹⁸ at D 242.36, well above eps_qd G. That is the rounding of 1/k on the mass mode's
  coefficients, magnified by the p series' transient. With an exact 1/k the difference is 9.9×10⁻⁷⁰, and P1 passes
  both bounds.
- The control as written cannot fire at the stored states. Their γ is 9.5×10⁻²⁸ (D 73) and 4.8×10⁻³⁵ (D 242.36), so
  χ̂ barely enters the rows, and `--mutate=sign` changed the difference at D 242.36 only from 7.3×10⁻³⁹ to 1.7×10⁻³⁸.
  With a_h moved by 10⁻⁸/G (`--move 1e-8`) it fires, at 4.5×10⁻¹².

**What this leaves.** The factored segment removes G only through an inner solve, and this one costs 21 to 32 sweeps,
each 1.3 to 2.5 times the old series in the prototype (noisy timings on the shared Mac). Quad-double in place of
octo-double gains 6.5 to 9 per operation (OCTO_DOUBLE.md 3), so the horizon block would cost three to twelve times what it
does now (an estimate from these ratios). Jacobian columns and jet orders iterate alike, and they have no warm start. A
Newton step on the coupled system in place of the sweeps needs the Jacobian of q in B, with (K + 1) M = 4224 unknowns
per evaluation, which costs more than the whole horizon block. Stage 2 as designed is therefore not recommended. The
reach past D ≈ 295 rests on the other routes, the truncation of section 4 or a 16-part type. Two predictions of 3.6 are
untested here, the factored rows' floor of ε·O(1) and the larger basin (P2, P4).

### 5.2 Stage 1b: no direct solve, and the curvature test (#1035)

**Why.** The order raise of #1064 fails from every ks 293 state near D 259–262. There the stored state's residual at
ks 315 is 1.7×10⁻³⁰, but Newton's step lies along a strong direction whose residual grows as 3.2×10⁵¹ per step²
(dh170, #1064). That is 3.6's quadratic jump, which the mass mode's factoring is predicted to remove. Section 2 rules
out any effect on the weak σ, so this bears on the raise and the floors only.

**No direct solve meets the cost gate (design, not run).** Stage 1b was to replace 5.1's sweeps by one Newton solve of
the factored segment over all orders, at most 2 old passes per converged solve. Over all orders the system has
5(K + 1) M ≈ 21 000 unknowns (p, m, c, C and β). Every block is causal in k (lower block-triangular) except B's datum,
which couples all β_k. The τ points couple only through ωD in the p and m rows. Three exact solves exist:
- **A, forward elimination with β₀ (that is, a_h) as the M border unknowns.** This is shooting on the old discrete
  problem, about 3 old passes per evaluation with a chord M × M Jacobian. B is carried forward from β₀, so ε·G returns
  in the values. Its inner Newton on a_h works only inside the pole basin, δ_max(u₀) ≈ 2×10⁻³⁷ at D 241.96
  ([HORIZON_RADIUS.md](HORIZON_RADIUS.md) 3). That is the solution's basin: at ks 130 the series hides the pole, and
  at D 242.36 A converged from a_h off by 1.6×10⁻³⁴ (the result below). A raise-sized move of 10⁻¹⁴ in m_h or Δ
  shifts the slaved a_h by about 10⁻¹⁷ (an estimate), so A cannot follow it alone. That is the re-seed's problem.
- **B, Olver's backward elimination of β with the causal blocks folded in.** Its Schur complement is lower Hessenberg
  with dense M × M blocks, and dense below the diagonal, because the forward response is not local in k: moving y by
  10⁻³⁰ at order 60 moves p at order 130 by 2.1×10⁻⁹ at D 242.36 (`loop`, 5.1). Forming it takes (K + 1) M = 4224
  tangent passes, and its elimination about K²M³/2 ≈ 3×10⁸ multiply-adds.
- **C, Newton–Krylov preconditioned by B's Olver solve.** Its operator is one minus the sweep map's linearization.
  GMRES on it equals Anderson mixing, which took 21–32 sweeps (5.1).

So B and C miss the gate by orders of magnitude, and A keeps ε·G and the pole basin. B is not coded.

**The curvature test (criteria fixed before any run).** The test asks whether the horizon rows' second derivative
along a weak-type move loses its G-sized part when a_K is held instead of a_h.
- **Moves.** Three directions at D 73.0 and D 242.36 (5.1's states): Δ alone (δΔ = 1), and m_h along harmonics 1 and 3
  (δm_h = cos θ_j and cos 3θ_j on the half grid).
- **The measure.** Q(t) = [R(t) + R(−t) − 2R(0)] / t², with R the p, m and c rows of the horizon block (the series at
  u₀ minus the node). Q_old holds a_h fixed, and Q_fact holds a_K fixed (the a row, a_K − a(node), is then constant).
  Each is read as its largest entry.
- **The factored evaluator.** R_fact(t) is the old series at the a_h that keeps a's sum at u₀ equal to a_K. That is
  method A, which 3.3 and P1 (5.1) show equals the factored segment. Here a_h is found by Newton with the M × M
  Jacobian of a(u₀) in a_h, from a predictor quadratic in t. A cross-check runs the stage-1 fixed point at D 242.36 and
  D 73 along Δ at t = ±10⁻²⁰ (warm from 5.1's references, 288 bits). The two Q_fact must agree to 1%, or both lie
  below their own noise.
- **The base.** Both maps use an exact 1/k, so that A and the fixed point solve one discrete problem. With an exact 1/k
  the rows at the stored a_h miss by up to 8.4×10⁻⁷ at D 242.36 (5.1), the a row by 4.2×10⁻⁷. So a_h is first
  re-solved by A until a's sum at u₀ equals the node's a, and that is the base for both maps.
- **The predictor.** A's predictor is linear in t at ±10⁻²⁵, and its quadratic term is taken from those two solves.
- **Precision.** 448 bits for the old series and A. The noise is about 2^−448·G·t⁻² per Q entry, 4×10⁻⁵⁹ at D 242.36
  for t = 10⁻²⁰.

**Gates** (the coordinator's, 2026-11-21):
1. Q_fact / Q_old ≤ 10⁻¹⁰ along every direction at both states: a narrow stage 2, a one-off factored raise, is worth
   building.
2. A ratio ≥ 10⁻² along any direction: the factoring does not help the raise.
3. A ratio in between is reported as not resolved.

**Controls that must fire.**
- **A planted quadratic term is recovered to 1%.** Δ(t) = Δ + t + c t² adds 2c ∂R/∂Δ to Q_fact along Δ. c is set so
  the term's largest entry is 10⁻¹⁰ Q_old, the gate-1 level.
- **The second differences scale as t².** Q at t = 10⁻²⁰ and 10⁻¹⁹ must agree to 10% of the larger, else that
  direction's Q is not in its quadratic range there.
  - Mine: Q_old may leave that range before 10⁻²⁰, since a fixed a_h then lies outside the pole basin. It is also read
    at the pairs 10⁻³⁰/10⁻²⁹, 10⁻⁴⁰/10⁻³⁹ and 10⁻⁵⁰/10⁻⁴⁹, and taken from the largest pair that scales. With no such
    pair, that direction is reported as not resolved.

**Result: gate 1.** Every ratio is below 10⁻¹⁰, along all three moves at both states. Holding a_K instead of a_h
removes the G-sized part of the horizon rows' curvature, so a narrow stage 2 is worth building. At D 242.36 the
verdict rests on the amended plant and the t² checks, not on a plant at the gate level (below). The ratios are
quotients of each Q's largest entry, and each Q holds to the digits shown (its t² check):

| state | move | Q_fact | Q_old | Q_fact / Q_old | Q_fact's t² check | Q_old's t² check at 10⁻²⁰ |
|---|---|---|---|---|---|---|
| D 73.0 | Δ | 3.34×10⁻² | 1.23×10¹⁴ | 2.7×10⁻¹⁶ | 7.8×10⁻³⁹ | 4.9×10⁻²⁴ |
| D 73.0 | m, cos θ | 1.18 | 9.16×10¹⁸ | 1.3×10⁻¹⁹ | 3.6×10⁻³⁸ | 2.7×10⁻¹⁹ |
| D 73.0 | m, cos 3θ | 2.10 | 5.04×10¹⁸ | 4.2×10⁻¹⁹ | 4.5×10⁻³⁸ | 1.9×10⁻¹⁹ |
| D 242.36 | Δ | 5.43×10⁻³ | 1.07×10⁵⁷ | 5.1×10⁻⁶⁰ | 1.9×10⁻³⁶ | 3.2×10⁻¹² |
| D 242.36 | m, cos θ | 1.07 | 2.35×10⁶³ | 4.5×10⁻⁶⁴ | 3.9×10⁻²⁴ | 5.5×10⁻⁶ |
| D 242.36 | m, cos 3θ | 1.21 | 2.26×10⁶³ | 5.4×10⁻⁶⁴ | 2.2×10⁻²⁴ | 4.2×10⁻⁶ |

- **Q_fact is of order one at both states**, between 5×10⁻³ and 2.1. Q_old grows by 43 to 45 orders of magnitude from
  D 73.0 to D 242.36, while G grows by 26.
- **Q_old scales as t² at all four pairs** at both states, so it is read at 10⁻²⁰. My fallback to the smaller pairs
  was not needed.
- **The base.** a's row at the stored a_h with an exact 1/k is 2.3×10⁻³⁰ at D 73.0 and 4.2×10⁻⁷ at D 242.36. A brings
  it to 1.7×10⁻¹²⁴ in 5 passes and to 1.3×10⁻⁹⁰ in 8 (the cap), moving a_h by 3.1×10⁻⁴¹ and 3.5×10⁻⁴⁴ relative. J's
  diagonal lies within 5% of G at both states (6.97–7.28×10¹⁰ and 1.180–1.197×10³⁷), and its largest off-diagonal
  entry is 3.6×10⁻³ and 1.9×10⁻⁴ of G.
- **A's Newton.** At D 73.0 it reaches a's row at 10⁻¹²² to 10⁻¹²⁴ in 5 or 6 passes. At D 242.36 it stops at the cap
  of 8 passes: a's row is 9×10⁻⁸⁰ to 3.5×10⁻⁷⁷ along Δ and 1.2×10⁻⁶⁸ to 1.1×10⁻⁶² along the two m moves. There the chord
  gains a steady 12 digits per pass at every t, which I read as J's own error (inferred: J is a finite difference at
  relative step 10⁻⁵⁰). The residual enters Q_fact as about |F|/t², and Q_fact's t² check bounds it at 3.9×10⁻²⁴
  relative.
- **A converged from outside the solution's pole basin.** At D 242.36 the predictor's first miss of a(u₀) is 0.13 and
  1.3 along Δ (t = 10⁻²⁰, 10⁻¹⁹) and 191 to 1964 along the two m moves. It grows linearly in t, since the predictor's
  linear term carries J's error. A miss of 1964 is a_h off by about 1964/G = 1.6×10⁻³⁴, 800 times δ_max(0.3) ≈
  2×10⁻³⁷. The series truncated at 130 hides that pole until about D 249 ([HORIZON_RADIUS.md](HORIZON_RADIUS.md)), so
  this says nothing about A's basin at the raise's ks 315 near D 260.
- **The planted control, as fixed above, passes at D 73.0 and fails at D 242.36.** At D 73.0, c = 3.2×10⁵ plants
  1.23×10⁴ in Q_fact, and it is recovered to 2.7×10⁻³⁵. At D 242.36 the gate-1 level needs c = 1.65×10⁴⁹, which
  moves Δ by c t² = 1.7×10⁹ at t = 10⁻²⁰. A's first solve then overflows (|F| = 7×10⁹⁵ at the predictor's level
  t = 10⁻²⁵, then infinite). No quadratic plant through the move can reach 10⁻¹⁰ Q_old ≈ 10⁴⁷ there and stay inside
  the map's domain, so this failure is the control's design, not the measure.
- **The amended plant at D 242.36 (amended after the data, with the coordinator's agreement).** c = 10¹⁶, so that c t
  ≤ 10⁻³ at t = 10⁻¹⁹, plants 6.5×10¹³ in Q_fact, and it is recovered to 8.4×10⁻²⁵. So Q_fact at D 242.36 recovers a
  planted curvature 1.2×10¹⁶ times its own reading to 1%. It does not test the measure at the gate level. At D 242.36
  the gate decision rests on that sensitivity and on the t² checks, not on a plant at the gate level.
- **The cross-check.** The stage-1 fixed point at 288 bits gives Q_fact = 5.433×10⁻³ along Δ at D 242.36, method A's
  value to 1.3×10⁻²⁹ relative. Its noise is about 2.7×10⁻³⁰ (10³ times the fixed-point residual, over t²), and it
  took 22, 22 and 15 sweeps at t = 10⁻²⁰, −10⁻²⁰ and 0. At D 73.0 it gives 3.340×10⁻², method A's value to
  3.5×10⁻²⁹, in 33, 33 and 24 sweeps. The two share the p, m and c recursion of `series` and the state's loading. They
  differ in a (the old recursion from a_h, against B_sl + γχ̂ and Miller's power) and in the solve (Newton on a_h,
  against Anderson's sweeps).
- **The controls that must fail do** (D 73.0, along Δ, each exit 1). `--mutate=r0` shifts R(0)'s p, m and c rows by
  10⁻⁴⁰. Q_fact then reads 2.03 at 10⁻²⁰ and 5.3×10⁻² at 10⁻¹⁹, its t² check 0.97, so the move is not resolved. Q_old's
  check fires at the three smaller pairs (0.99) and holds at 10⁻²⁰. `--mutate=plant1` plants the term at +t only, and
  the recovery misses by 0.50.

**Limits.**
- The prototype's two-node layout is not the chain's. So the test reads the mechanism, not dh170's 3.2×10⁵¹ itself.
- A ratio ≤ 10⁻¹⁰ says the horizon rows' curvature drops. Whether the chain's raise then converges is for stage 2.

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
7. **The σ table of #491's second comment** (crawlJ at D 239.54, 240.51, 241.12 and 241.89) is stored in `I734-mac/`
   of the raw-data repository. Section 4 compares only with `I488-mac/`'s D 237.13 and 242.36.
8. **3.4's loop gain.** The sweeps' late ratio is about 1/n at D 242.36, as 3.4 inferred, but 3.4's estimate missed
   the p series' transient past Re s. Through it the linearized map magnifies some perturbations by 8.7×10²⁰
   in one sweep, and at D 73 the sweeps diverge from the slaved start (5.1).
9. **hp's 1/k is part of its discrete problem.** The horizon series multiply by `recip_int(k)`, 1/k rounded by
   multidouble's double-double division, in every lane, the octo-double ones included. The rounding excites the mass
   mode like any error of the a recursion, so the series with an exact 1/k misses the stored nodes by 4.4×10⁻³⁰ at
   D 73 and 8.4×10⁻⁷ at D 242.36 (5.1). The factored segment's B rows have no such rounding, and P1 fails at D 73 by
   what is left of it.
10. **P1's control** (χ̂'s transport sign flipped) cannot fire at a converged state. The mass mode's amplitude at the
   node is then 9.5×10⁻²⁸ (D 73) and 4.8×10⁻³⁵ (D 242.36), so χ̂ hardly enters the rows. It fires once a_h is moved
   by 10⁻⁸/G (5.1).
11. **The p rows on the harmonics k ≤ kfreeze** are 2.6×10⁻³⁴ at T81's D 73.0 state and 1.6×10⁻³⁴ at I461's D 242.36
   state, whose m, a and c rows are at the states' floors. At #603's states they are 1.5×10⁻⁶⁶ and 4.2×10⁻⁹⁵. T139's
   D 73.75 state (`T139-macm3/chain/anm_pin3_D73.7500.npz`, converged to the corrector's 10⁻²⁸) reads 3.1×10⁻³⁴. The
   cause is not known. A guess is that the builds that made the first three formed the frozen rows differently.

## 7. Commands

From the repository root (sympy 1.14, mpmath 1.3):

```
python3 solver/factored_horizon_sympy.py                      # 30 checks pass, exit 0
python3 solver/factored_horizon_sympy.py check --mutate=sign   # 9 of 26 checks FAIL, exit 1
python3 solver/factored_horizon_sympy.py model --mutate=ks     # M4's two checks FAIL, exit 1
```

One process, about a minute on an idle Mac and nine minutes on the shared one.

`check` runs E0 and S1–S8 alone, `model` S8 and M1–M5. M2–M4's tolerances were set after a prototype run, so they
describe the agreement rather than test it; the tests are 4.5 and #696. M5 prints the estimate of 4.3 and checks
nothing.

The prototype of 5.1 (mpmath 1.3, numpy 2.5) runs from `hp/`, with the raw-data repository cloned beside the checkout
(`git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw`), `RAW=../../choptuik-large-d-raw`
and the cached reference in a task folder of it, `OUT=$RAW/I1035-<machine>` (made first, with its README):

```
python3 factored_horizon.py selftest                   # 7 of 7 pass, exit 0 (half a minute)
python3 factored_horizon.py selftest --mutate=sign     # S2, S4, S6 FAIL, exit 1 (also nh: S0; bsl: S1; miller: S3; nofp: S5)
python3 factored_horizon.py check $RAW/I603-win2/chain/anm_pin3_D241.9602.npz --bound 2e-64       # PASS, worst 2.0e-65
python3 factored_horizon.py check $RAW/I603-win2/chain/anm_pin3_D241.9602.npz --bound 2e-64 --mutate=nh   # FAIL, 8.9e51
python3 factored_horizon.py gain $RAW/I461-win2/crawlJ/anm_pin3_D242.3642.npz --prec 288 --start none --ref $OUT/ref242.json
python3 factored_horizon.py gain $RAW/I461-win2/crawlJ/anm_pin3_D242.3642.npz --prec 288 --accel plain --maxit 36 --ref $OUT/ref242.json   # 32
python3 factored_horizon.py p1 $RAW/I461-win2/crawlJ/anm_pin3_D242.3642.npz --prec 288 --ref $OUT/ref242.json --maxit 12   # 7.3e-39
python3 factored_horizon.py p1 $RAW/I461-win2/crawlJ/anm_pin3_D242.3642.npz --prec 288 --ref $OUT/ref242.json --maxit 36 --move 1e-8 --mutate=sign
python3 factored_horizon.py loop $RAW/I461-win2/crawlJ/anm_pin3_D242.3642.npz --prec 512 --h 1e-110
```

One process each. The reference takes about ten minutes, a gain or P1 run three to nine, a check ten seconds. The
D 73 runs take `--ref_start old` for the reference, as the slaved start diverges there. The logs and records of the
runs quoted in 5.1 are for `I1035-mac/` of the raw-data repository.

The curvature test of 5.2 runs the same way, with `OUT=$RAW/I1035-<machine>/stage1b`,
`S73=$RAW/T81/conv_pin5_D73.0000_D73.000000.npz`, `S242=$RAW/I461-win2/crawlJ/anm_pin3_D242.3642.npz` and 5.1's
288-bit references (stored as `I1035-mac/ref_D73_288.json` and `ref_D242_288.json`). curvjac goes first, since it
writes the cache the others read:

```
python3 factored_horizon.py curvjac $S73 --prec 448 --cache $OUT/cache_D73.json      # re-based to 1.73e-124
python3 factored_horizon.py curv $S73 --prec 448 --cache $OUT/cache_D73.json --move D --out $OUT/curv_D73_D.json
                                                       # ratio 2.710e-16, gate 1; --move m1: 1.289e-19; m3: 4.163e-19
python3 factored_horizon.py curvjac $S242 --prec 448 --cache $OUT/cache_D242.json    # re-based to 1.27e-90
python3 factored_horizon.py curv $S242 --prec 448 --cache $OUT/cache_D242.json --move D --out $OUT/curv_D242_D.json
                                                       # ratio 5.091e-60, gate 1; --move m1: 4.543e-64; m3: 5.364e-64
python3 factored_horizon.py curv $S73 --prec 448 --cache $OUT/cache_D73.json --move D --plant $OUT/curv_D73_D.json
                                                       # PASS, recovered to 2.73e-35
python3 factored_horizon.py curv $S242 --prec 448 --cache $OUT/cache_D242.json --move D --plant $OUT/curv_D242_D.json
                                                       # FAIL, exit 1 (A overflows at c = 1.65e49)
python3 factored_horizon.py curv $S242 --prec 448 --cache $OUT/cache_D242.json --move D --plant $OUT/curv_D242_D.json --plant_c 1e16
                                                       # PASS, recovered to 8.41e-25 (amended after the data)
python3 factored_horizon.py curv $S73 --prec 448 --cache $OUT/cache_D73.json --move D --mutate=r0
                                                       # not resolved, exit 1 (Q_fact's t^2 check 0.974)
python3 factored_horizon.py curv $S73 --prec 448 --cache $OUT/cache_D73.json --move D --plant $OUT/curv_D73_D.json --mutate=plant1
                                                       # FAIL, exit 1 (recovered to 0.50)
python3 factored_horizon.py curvx $S242 --prec 288 --ref $RAW/I1035-mac/ref_D242_288.json --t 1e-20 --out $OUT/curvx_D242_1e-20.json
                                                       # and --t -1e-20, --t 0: 22, 22 and 15 sweeps
python3 factored_horizon.py curvxq $OUT/curvx_D242_1e-20.json $OUT/curvx_D242_-1e-20.json $OUT/curvx_D242_0.json --t 1e-20 --curv $OUT/curv_D242_D.json
                                                       # PASS, relative difference 1.31e-29
```

The last two lines with `D73`, `$S73` and `ref_D73_288.json` give the cross-check at D 73.0: PASS, relative
difference 3.48e-29. One process each. On the Mac a curv run took 3 to 12 minutes, a curvjac 3 to 8 and a curvx 3 to 5.

## 8. Prior art used

- Proposition 3 (README): the fast mode's form and transport, and the method of S4.
- Proposition 2 (README) and the connection formula (Abramowitz & Stegun 15.3.6): the linear model and S7.
- proof/C_selection.md: Lemma B2 (the two exponents), B6 (the light-cone exponent is not an integer), Lemma B5.1 (κ_n,
  a power of n) and Theorem B's count.
- [LIT_REVIEW.md](LIT_REVIEW.md) C3 and Q1, the WKB entries: nΦ as the fast mode's WKB exponent (Q1), and the
  Stokes reading of Propositions 3–4, in which the selection is the coefficient of that mode (C3).
- Deuflhard (1974), LIT_REVIEW.md F4: the natural level function ‖J⁻¹F‖, invariant under the rows' scaling.
- Miller (1952), Olver (1967) and Gautschi (1997), [LIT_REVIEW.md](LIT_REVIEW.md) B1–B8 (addendum of #733): backward
  recurrence for the minimal solution, and its boundary-value form with b_(K+1) = 0 (B4, B7).
