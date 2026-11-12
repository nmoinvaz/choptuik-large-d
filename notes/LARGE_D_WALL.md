# Why the continuation stalls at large D: the weak directions, the drop-offs and the horizon-series wall

Moved verbatim from README.md (commit ea42b3e, lines 829–882, 931–977) on 2026-11-04, in the reorganization of
[README_PLAN.md](README_PLAN.md); the README keeps a pointer where each block stood. Links were made relative to this
folder.

![Why the continuation stalls along the weak directions: a model of one Newton step along a weak direction, where a large curvature against a small slope makes Newton land above its start, and each run's step in D diving at the stalls at D = 35.2, 60.96 and 67.75](../hp/newton_stall.svg)

**Why the steps collapse** (`hp/plot_newton_stall.py`, which reads T73's records from folder `T73` of the [raw-data repository](https://github.com/nmoinvaz/choptuik-large-d-raw): run
`git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw` first). Top, a model rather than data: the residual along one weak direction,
F(t) = F₀ − σt + ½ct², and Newton's straight line F₀ − σt. Newton jumps to the line's zero, t = F₀/σ, which is far
out when the slope σ is small, and by then the curvature has added cF₀²/(2σ²). The step lowers the residual only
while cF₀/σ² < 2, that is while the residual Newton starts from is below 2σ²/c. At cF₀/σ² = 2.5 it lands at 1.25 F₀,
and line search ends at the curve's lowest point, 0.8 F₀: the stall. The weak directions' σ falls as D rises (below),
so that bound shrinks, and a shorter step in D hands Newton a smaller starting residual; that is why a run's step
dives before it stops. Bottom, the data: each run's step in D is smallest at the stalls, 0.00235 at D = 35.20, 0.00086
at 60.96 and 0.00010 at 67.75. Above D = 70 the median step is back to 0.0146 in T73, which holds the weakest
harmonics at the predictor's values and corrects Δ for that with one Newton step of the full system. At D = 62–64,
where T48's seeds can be rebuilt from its saved states (`hp/stall_margin.py --attempts`), every step whose seed
residual was at most 1.4×10⁻⁵ (62) or 2.2×10⁻⁵ (64) converged and every larger one failed. Behind the basin is a
precision limit: past D ≈ 60 double-double cannot resolve the weakest direction at all (below, "Why the solver
stalls").

![The drop-offs of the continuation's step and the walls behind them: each run's step in D with the stalls at 35.2 (the junk floor), 60.96 (precision) and 67.78 named, and below it u times kappa at converged states with the full system's fit crossing 1 at D 59.9 and the pinned systems' crossings predicted at 79, 94 and 113](../hp/large_d_dropoffs.svg)

**Where the drop-offs are, and where the next ones would be** (`hp/plot_dropoffs.py`, from the same T76 records as the
next figure). The step collapses at a wall, and the two walls have different causes. At 60.96, with harmonics to 13,
it is precision: u·κ, the unit roundoff times the Jacobian's condition number, was 1.7 at the stall (the fit below puts
u·κ = 1 at 59.9, 1.7 at 60.7; its scatter is ±0.4 in D). At 35.2, with every harmonic kept, u·κ was only 0.010 at
35.02 (`T101-mac/stall_up6.jsonl`, 2026-11-03): that wall was the residual floor of 10⁻¹⁴ to 10⁻¹⁶ which the
undetermined junk directions set, with the basin measure of the stall detector, κ = cF₀/σ² along the weakest direction
(harmonic 31), at 26 against its limit of 2. T48 ran past the precision wall to 67.78 with the harmonics held on their
trend (u·κ = 168 at 67.74). The other dips are not walls: at 45.3 a cloud run stopped with its sensitivity and residual
floor unchanged and the next run continued at once, and below D = 27 the dips are the f64 solver's own limits
([solver/CONTEXT.md](../solver/CONTEXT.md)). With the
same slope, u·κ reaches 1 at D = 79 with harmonics 11–13 held, 94 with 9–13 held and 113 with 7–13 held, and at
372 for the full system in quad-double; the pinned runs stopped before that, at 73.3 and 78.1, at the horizon block
([notes/QUAD_DOUBLE.md](QUAD_DOUBLE.md) 12.4 and 12.7). So the walls belong to the formulation and the arithmetic,
not to the solution: Δ(D) is smooth through every one of them, and each wall moved when the method changed.

![Why the horizon series stops: its radius for data off the solution against the data's accuracy at five D with the matching distance marked, and the iterate accuracy Newton needs against D for three layouts with the walls, the predictor's accuracy and the double-double resolution](../hp/pole_basin.svg)

**What the horizon block's wall is** (`hp/plot_pole_basin.py`, from `hp/pole_test.py` on the stored states;
[notes/QUAD_DOUBLE.md](QUAD_DOUBLE.md) 13.6). The horizon series' radius for horizon data a fraction δ off the
solution is R = R₀ + C ln(1/δ)/(D − 3) with C ≈ 0.9 at every D from 40 to 73 (left). That C is read at the states'
100 series orders and rises with the order count, to 0.947–0.958 at 250 (QUAD_DOUBLE.md 13.6's addendum). So Newton
can only step from an iterate within exp(−(u₀ − R₀)(D − 3)/C) of the solution (right): 3×10⁻¹² at D 73 with the
last node at 0.7, where
the pinned run stopped, and 5×10⁻³¹ at D 200, below double-double's resolution. The walls at 73.3 and 78.1 lie on
these lines. A series that must converge across u₀ cannot carry the branch to 200 in any layout; the collocation
block of section 13 there replaces it.

![Where double-double runs out: u times kappa against D for the full system, crossing 1 near D = 60 at the stalls, and for the pinned systems 8 to 11 orders lower; below, the residual each state reached following the rounding floor u sigma_max |z|](../hp/stall_kappa.svg)

**Where double-double runs out** (`hp/plot_stall_kappa.py`, from the stall detector's runs in folder `T76` of the
raw-data repository:
run `git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw` first). Top: u·κ, double-double's unit roundoff times the Jacobian's
condition number, at converged states. The full system's rises like n^38.9 and crosses 1 at D = 59.9, just before the
stall at 60.96; at 67.74, where T48 stopped, it is 168. The pinned systems the runs solve sit 8 to 11 orders lower
(1.0×10⁻⁸ with harmonics 7–13 held at D = 70.98, 1.5×10⁻⁵ with 9–13 held at 70.63), which is why they still converge;
the single full-system step that corrects for the pin is still taken at the full system's u·κ. Bottom: the residual each
saved full-system state reached against u·σ_max·|z|, scaled by their median ratio 0.045, the rounding floor of
evaluating F at a double-double state.

What the solver has shown:

- **The weak directions.** The near-null directions of the Jacobian that stalled the f64 solver (called "junk" in the
  code and notes) are a ladder of high τ-harmonics of the centre data, one pair per odd harmonic k, with
  singular values falling about like $`e^{-0.7k}`$
  (10⁻¹⁰ at k = 7 down to 7×10⁻¹⁸ at k = 31 at D = 21; the weakest is 10⁻²⁴ at D = 35). They
  are outgoing waves near the centre that barely reach the matching point, and they belong to the boundary-value problem
  itself: moving the first node or the matching point leaves them unchanged. Their own effect on Δ is tiny
  (≤ 10⁻¹² per unit amplitude for k ≥ 17), but second-order errors leak into them amplified by 1/σ,
  so Newton's basin shrinks with D; at D = 8, 80 points in τ already fail to converge. From D = 35 the solver
  pins the centre data's harmonics above 13 at zero and drops their matching on the horizon side: the weakest singular
  value rises from 10⁻²⁴ to 10⁻¹⁷ and plain Newton converges again. The cut moves Δ by 5×10⁻¹⁵ at
  D = 35 and about 10⁻¹⁵ at D = 40–60 (freeing harmonic 15 changes Δ by +1.5, −1.1 and −1.2×10⁻¹⁵ at D = 40, 50 and 60,
  with full Newton steps), which moves Δ∞ by at most 5.5×10⁻¹⁰ ([notes/NUMERICS_REVIEW.md](NUMERICS_REVIEW.md)).
  A cut at 11 does not converge at D = 40.
- **How fast the selection weakens.** $`\max\lvert\partial\Delta/\partial F\rvert`$ grows about 1.8 times per unit of D
  below D = 27, 1.7 at 35–45, 1.6 at 45–56 and 1.5 at 56–61, to 9×10¹³ at D = 60 (table), and the residual floor rises
  about 1.4-fold per unit. Their product, a worst case that ignores the direction of F, bounds the error in Δ by 10⁻¹⁴
  at D = 45 and 2×10⁻⁹ at D = 60. The linearised estimate $`-\partial\Delta/\partial F \cdot F`$, which accounts for it,
  stays between 10⁻²¹ and 10⁻¹⁷ from D = 35.3 to 60.1 and reaches 3×10⁻¹⁵ above, where the tolerance was relaxed. The
  falling per-unit factors are those of a power law, not an exponential: once the harmonic set is fixed (D = 40–61,
  harmonics above 13 frozen) $`\max\lvert\partial\Delta/\partial F\rvert`$ is $`n^{21.7}`$ to 2% in ln, while
  $`e^{n/2} n^p`$ misses by 15–17% (`solver/beyond_lc/sens_fit.py`). The fast mode that carries the selection,
  $`e^{-n\Phi}`$ with $`\Phi = \ln x - \tfrac12\ln(1 - x^2)`$, has modulus 1 on $`\mathrm{Re}\,x^2 = \tfrac12`$, which
  crosses the real axis at x = 1/√2. The solver matches at 0.7, where it is of order one, and moving the last node
  changes $`\max\lvert\partial\Delta/\partial F\rvert`$ by $`e^{n[\Phi(x_m) - \Phi(0.7)]}`$, over 13 orders of magnitude
  with no fitted parameter (`solver/beyond_lc/hp_node_test.py`). What grows is the high-harmonic ladder below
  ([notes/BEYOND_LIGHT_CONE.md](BEYOND_LIGHT_CONE.md)); the residual floor's 1.4-fold rise per unit of D is the growth
  of the Jacobian's largest singular value (next item).
- **Why the solver stalls: double-double runs out of precision.** `hp/stall_margin.py` measures the Jacobian J at a
  solved point: its weakest singular directions, its largest singular value σ_max (from the dense double-double
  Jacobian) and the condition number κ = σ_max/σ_min. Its runs are in folder `T76` of the raw-data repository and `hp/stall_summary.py`
  prints the numbers below; its controls are two planted problems with known answers and two mutations. From
  D = 40 to 67.74, with n = D − 2, the weakest direction is always p1's harmonic 13, the highest the solver keeps;
  σ_min falls like n^−20.8 (to 4% at 9 points) and σ_max grows like n^18.1. Since ∂Δ/∂F is a row of J⁻¹,
  σ_min ≤ 1/max|∂Δ/∂F| (proved), and in fact σ_min·max|∂Δ/∂F| stays between 6.5 and 9.5×10⁻⁹, so the sensitivity
  in the CSV tracks the weakest direction. The product u·κ, with u = 2⁻¹⁰⁴ double-double's unit roundoff, grows
  like n^38.9 and reaches 1 at D = 59.9 (numerical evidence): it is 1.7 at the stall at 60.96 and 168 at 67.74,
  where T48 stopped. A backward-stable solve's error relative to the solution is about u·κ[^21], so beyond D ≈ 60 a
  double-double Newton step's error along the weakest direction can be as large as the step, and the convergence
  theorem for inexact Newton methods, which needs each solve's relative residual to stay uniformly below 1[^22], no
  longer applies. Every remedy so far takes those directions out of the solve: holding them on their trend,
  deflating them, and T73's pin, which lowers u·κ at D = 71 from 1.4×10³ to 1.5×10⁻⁵ (harmonics 9–13 held) and
  1.0×10⁻⁸ (7–13 held). The residual the runs reach is 0.04–0.05 × u·σ_max·|z| (|z| ≈ 19.5) at every state from
  D 40 to 64 (0.11 and 0.2 at 66.0 and 67.5), the rounding floor of evaluating F at a double-double z; σ_max grows
  1.36 times per unit of D at D = 60.

## The octo-double wall near D 232–242

The octo-double chains refuse steps near D 242.5 (win2's crawlJ and its probes) and, on one of two held fits, at
D 231.13 → 232.13 (win_d10; win_d6 accepted the same step). Raw folders below are in the raw-data repository; each
item names its issue.

**Where the first move goes.** Every first Newton move of a corrector attempt lies on the four weakest directions of
the pinned system: share 1.0 for accepted and refused attempts alike, at D 242.5 and at D 232.13 (`hp/dz0_weak.py`,
#499 and #513; `I499-mac/` at ea4cd55e and fe30a3c6). Its rebuild check passes on every attempt, and its control,
the kernel rebuilt at D + 10⁻⁶, fails on every attempt.

**The weak rows are not the obstacle.** On those four directions the reduced system is
B(c − a) + ½Q(c, c) + T(c, c, c)/6 = 0, with a the first move's coefficients.
- Down to h = 10⁻²⁴ the quadratic term Q = UᵀF″(vⱼ, vₖ) is lost in the differences' truncation, which falls like h²
  (6.3×10²⁰, 10¹⁶ and 10¹² at h = 10⁻²⁰, 10⁻²² and 10⁻²⁴). At h = 10⁻³¹ and 10⁻³² it is resolved, agreeing to
  1.3×10⁻⁶ and 2.3×10⁻⁶. Its largest entry is 5.4×10⁴ at the accepted attempt and 2.75×10⁴ at the refused one,
  27 to 54 times the scale at which #527 estimated it would matter (#529; `I529-win2/` at dcde13a4). So Q is not
  zero. The roots below were found with Q set to 0, but each was checked against the actual weak residual, which
  includes every order.
- The weak subspace is four-dimensional with a clean gap: σ₅ lies at least 1.2×10⁶ above σ₄ at both attempts, though
  its own value does not converge (1.2–7.1×10⁻¹⁹ over 16 to 256 iterations; #529).
- The cubic term is resolved: max|UᵀF‴| is 1.63×10³³ at D 242.55 and 4.46×10³¹ at D 232.13, the same at h = 10⁻²³ and
  10⁻²⁴ to 3×10⁻¹⁸.
- At D 242.55 the root nearest the move is real, at about half of |a|, where the actual weak residual is 0.4% of
  |B a|. At D 232.13 the root is the linear guess (4×10⁻⁹ and 3×10⁻¹¹ of |B a|), and win_d10 refused there anyway.
- A corrector that first puts the weak part on that root (`hp/weak_corrector.py`, #499; `I499-mac/` at 19958783)
  is refused like newton_od. The control reproduces the attempt's sequence, 9.04×10⁻³⁶ → 7.27×10⁻²⁶ → 3.63×10⁻²⁶ →
  3.18×10⁻²⁶ with no trial accepted. The weak-first start stands at 2.02×10⁻²⁶ and stops at 1.35×10⁻²⁶.

**The jump after the first move is quadratic in the move and sits in the strong rows.** The residual after the move is
7.27×10⁻²⁶ for a move of 8.32×10⁻³⁰ and 2.02×10⁻²⁶ for one of 4.45×10⁻³⁰. Their ratio, 3.60, is the square of the
moves' ratio, 3.50, so max|F| ≈ 1.0×10³³ |c|². At both points the weak rows' residual is of the order of
|B a| = 9.8×10⁻⁵⁶, 30 orders below, so this residual is in the strong rows.

A third point comes from newton_od's first step from the weak-first start (`I499-mac/` at af80d7a1). That step again
lies entirely on the weak directions (2-norm 3.69×10⁻³⁰, 3.0×10⁻⁵⁶ off them). The law predicts 1.43×10⁻²⁶ after it,
and the run reached 1.35×10⁻²⁶. The coupling back runs through the mixed curvature UᵀF″(Vc, x) with x strong. The
first move's 0.4% off the weak directions raises the weak residual from 0.004 to 0.52 of |B a|, which UᵀJx = 0 cannot
do. So each weak correction raises the strong rows quadratically, and each strong correction feeds back into the weak
rows. Three points, so numerical evidence.

That feedback is the term S of the reduced coefficient T − 3S (Kuznetsov's form, [LIT_REVIEW.md](LIT_REVIEW.md),
#527). #499 measured T alone. #528 measures S, #529 the mixed block, and #530 tests a corrector that solves the strong
rows before the weak ones.

**Where the weakness lives in the layout.** Raw singular values are not comparable across layouts, since a node's
rows carry its own gain. With rows and columns equilibrated (`hp/near_null_eq.py`, #502; `I502-mac/` at 811bbed3), the
0.3–0.8 layout lifts σ₁ by 7.6×10⁴ (1.88×10⁻⁶¹ to 1.42×10⁻⁵⁶). That is the reverse of the raw reading, 2.55×10⁻²⁷ to
8.43×10⁻³⁸ (#493). One node moved at a time at D 242.36 (#537; `I537-win2/` at 923c87c1):

| move | shift of ln σ₃, raw | equilibrated | fast mode between nodes | G^p |
|---|---|---|---|---|
| first node 0.3 → 0.35 | 0.00 | 0.00 | +41 | 0 |
| last node 0.7 → 0.72 | −8.96 | −2.20 | −13.7 | positive |

All four σ shift together, and neither model holds. The −13.7 is n[Φ(0.72) − Φ(0.7)], the node law that matched
max|∂Δ/∂F| over 13 orders at D 40–61 (above). At D 242.36 the raw shift is 65% of it.

Only the last node is felt (`I543-win2/` and `I544-win2/` at adbf158f, `I553-win2/` and `I554-win2/` at 145084e7):
- Moving any interior node by 0.02, or the first node to 0.25, changes no σ (below 0.001 in ln, equilibrated). The
  five-node layout ending at 0.80 gives #502's six-node 0.3–0.8 values to within 0.037 (#544, #543).
- No law a + b x or a + b ln x in the last node's position x holds (#543), and the fit's prediction for 0.8, written
  before the comparison, missed.
- The weakest direction changes at x = 0.7428 (#553): below it is the harmonic-3 pair, and above it the pure Delta
  direction. The Delta direction's raw σ there is near 10⁻³⁷ and is not the norm of J's Delta column (#553).
- The weakest equilibrated σ is smallest at 0.74 (4.0×10⁻⁶³) and still rising at 0.92 (2.04×10⁻⁴², 19 orders above
  0.70). The rise is J's scale, whose largest entry falls like x^−(D−3), from 2.4×10³⁷ at 0.70 to 9.3×10⁸ at 0.92
  (#554). #559 extends the ladder to 0.98.
- At x = 0.80 the Delta direction's raw σ falls with D: 1.59×10⁻³⁷, 1.24×10⁻³⁷ and 8.74×10⁻³⁸ at D 232.63, 237.13 and
  242.36, about 6% per unit of D, while the other weak σ stay within 0.3% (#560; `I560-win2/` at 12e71363).

**Why the 0.8 layout's polish failed.** At the re-noded 0.3–0.8 start of D 242.36, the polish's first step is of order
1 (|dz| 1.73), almost all on v₄. It is the start's horizon residual, 1.62×10⁻³⁴, divided by σ₄ = 1.77×10⁻³⁴ (#501;
`I501-mac/` at 650c9bf8). The outer layout is better conditioned, but its start is not accurate enough for its weakest
direction. #562 refines the start's horizon rows before the polish.

**The freeze projector's rounding.** The frozen rows apply the kfreeze projector to p1, and main forms it in
double-double. With the projector in octo-double (#523's cadcaf1, error 1.15×10⁻¹³²), the chain's states are
1.6–1.7×10⁻³⁴ off their rows in the horizon p rows k ≤ 3, at every D from 215 to 242.36. On main's rounded rows they are
at 5×10⁻³⁶ to 8×10⁻³⁸.
- At the 0.975 start, three of the four weak σ come from that rounding: 7.4×10⁻³⁵, 7.4×10⁻³⁵ and 1.84×10⁻³⁴ on main,
  against 2.8×10⁻³⁹ to 1.9×10⁻³⁷ on cadcaf1 (#562; `I562-mac/` at 690c41a9). The raw σ near 10⁻³⁴ of the outer-layout
  ladders above carry it.
- The pinned step to the exact rows lies on the four weak directions (share 1.0000), and its Δ part grows by about 0.8
  in ln per unit of D (#589; `I589-mac/` at 420f201a):

| D | Δ part of the step to the exact rows | chain_fit row's quoted err |
|---|---|---|
| 215.00 | −4.5×10⁻¹⁹ | 3.4×10⁻¹¹ |
| 225.13 | −7.8×10⁻¹⁶ | 3.8×10⁻¹¹ |
| 235.13 | −3.6×10⁻¹² | 2.6×10⁻¹⁴ |
| 240.03 | −2.0×10⁻¹⁰ | none |
| 242.36 | +2.1×10⁻⁹ | none |

These are first-order figures. No state lies inside its chord window, so none is a measured shift. On the rounded rows,
though, #562's homotopy traces stayed straight in Δ well past their chord windows. **Discrepancy, open:** if the
first-order figure holds, the chain_fit rows of `hp/data/delta_all.csv` are off the exact problem's Δ from about D 233.6
on. At D 235.13 that is the 12th decimal, 140 times the row's quoted err. #595 measures the shift at D 235 and 240 by
tracing to the exact rows, and the rows stay as quoted until then.

**The weak σ oscillate in D.** The equilibrated σ₁ and σ₂ oscillate with a period that shortens as D falls: 2.715 at
D 232.6, 2.525 at 221.6 and 2.440 at 212.8 (#525; `I525-win2/` at 5037ec07). σ₃ and σ₄ have periods of 2.13–2.14
over D 210–215.6.
A drift quadratic in D fits the 51 states D 210–237 to 0.0060 and 0.0052 in ln for σ₁ and σ₂ (#533; `I533-win2/` at
f01f08fe). One of its four hold-outs misses the 0.014 bound (0.0173). It predicts σ₁'s period over D 204–209.6 as
2.404 (sd 0.021, noise only), which #535 tests.

**The held fit's share in one step.** Swapping the held fits between the degree-6 and degree-10 chains for the step
D 229.13 → 230.13 moves Δ by ±1.175×10⁻¹³, equal and opposite. That is 0.506% of the 2.32×10⁻¹¹ gap between the chains
(#514; `I514-win/` at d2208e68).

**The horizon pole is not an apparent horizon.** The variables B = 1/a² and C = c a² keep the pole of
[QUAD_DOUBLE.md](QUAD_DOUBLE.md) 13.6 (`hp/pole_bc.py`, #538; `I538-mac/` at 651f29e6).
- p, m, B and C all have the radius of a. Their log-slope in ln(1/δ) is 0.997–1.000 of 13.6's at D 40–73, and 0.93 at
  D 242.36.
- B stays at 0.90–0.98 near the radius, so no trapped sphere forms there. That contradicts 13.1's reading of the pole
  as g_rr → ∞ at an apparent horizon. The law itself holds.
- The zero of x − c, where the p-equation is singular, is the nearest singularity at moderate D and small δ, and not
  at D 242.36:
  - A real-axis march of the perturbed solution (`hp/pole_march.py`, #545; `I545-mac/` at 6e935ad8) finds no breakdown
    at the radius. For δ > 0 at D 40 and 73, x − c rises to between −0.01 and −0.03 and turns back 1–6% beyond R,
    with a field peak there. For δ < 0 nothing happens on the axis.
  - The nearest zero of x − c from Padé approximants of the series (`hp/pole_sonic.py`, #592; `I592-mac/` at 5e4ec5b6)
    lies within 3% of R at D 40 and 73 for |δ| = 10⁻⁸, and 4–13% beyond R at larger δ. At D 242.36 it lies 15–157%
    beyond R.
  - Numerical evidence. Past R the series' own accuracy limits the reading. At D 73 and 242.36 the double-double series
    and the kernel's Taylor steps disagree far above the march's conditioning (#545), and #591 compares the series with
    an octo-double one.
- 13.6's C depends on the series' order count: 0.918, 0.935 and 0.942 at 100, 130 and 160 orders at D 40. #546 quotes
  it with that range.
