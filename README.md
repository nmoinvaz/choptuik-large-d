# dss-residual-kernel

This repository computes Choptuik's discretely self-similar critical solution of massless scalar collapse in
$`D = n + 2`$ spacetime dimensions, with its echo period $`\Delta(D)`$ and critical exponent $`\gamma(D)`$: an f64 solver in
[`solver/`](solver/) with a Rust kernel for its residual (this crate), and a separate double-double (about 32-digit)
solver in [`hp/`](hp/) for large $`D`$. The physics is in this file; the software is described at the end. Headline results:

- **One formula for all $`D \ge 4`$.** A rational function of $`1/(D-2)`$ gives $`\Delta`$ and $`\gamma`$ within the precision of the
  computed points up to $`D = 30`$, and $`\Delta`$ to $`6\times10^{-8}`$ up to $`D = 61`$ ([The formula](#the-formula), [Values](#values)).
- **The large-D limit.** $`\Delta`$ is computed at 857 points from $`D = 8.25`$ to $`61.03`$, to about $`10^{-16}`$ ($`10^{-15}`$ above
  $`D = 60.1`$). The points support, but do not prove, that $`\Delta`$ is analytic in $`1/D`$; then
  $`\Delta_\infty = 1.64389375 \pm 10^{-8}`$ ($`\omega_\infty = 2\pi/\Delta_\infty = 3.82213589`$), and $`\gamma_\infty = 1/2`$ to about $`4\times10^{-4}`$.
  Nine candidate closed forms for $`\Delta_\infty`$ are ruled out ([Evidence](#evidence)).
- **Why the 1/D expansion cannot give it.** The $`1/D`$ expansion of the field equations can be carried to all orders for any
  period (Proposition 1), so $`\Delta`$, and likewise $`\gamma`$, are selected by effects smaller than every power of $`1/D`$. Without
  gravity the obstruction at the light cone is a closed-form coefficient $`C(n,s)`$ (Proposition 2), which is the Stokes
  multiplier of a mode born far outside the light cone (Propositions 3–4). The working hypothesis is that the selection
  rule is the vanishing of the Stokes multipliers of an $`n`$-free inner problem; its higher-harmonic conditions already
  predict the shape of the echo, $`F_3/F_1 = 0.055`$ against $`0.050`$–$`0.058`$ from the data
  ([Large D](#large-d-what-the-1d-expansion-can-and-cannot-see)).
- **Relation to published work.** Agreement with the accepted values at $`D = 4`$ and at the maximum of $`\Delta`$ near $`D = 3.756`$;
  an unresolved $`6\times10^{-5}`$ discrepancy at $`D = 5`$; and the published conjecture $`\Delta \to 0`$ as $`D \to \infty`$ is not
  supported: a power-law approach to 0 is excluded, a logarithmic one disfavoured
  ([Relation to published work](#relation-to-published-work)).

Contents

- [Results](#results): [The formula](#the-formula) · [Values](#values) ·
  [What the numbers mean physically](#what-the-numbers-mean-physically) ·
  [Candidate closed forms, all ruled out](#candidate-closed-forms-all-ruled-out) ·
  [Relation to published work](#relation-to-published-work)
- [Evidence](#evidence): [The double-double solver](#the-double-double-solver-hp) ·
  [Is the period analytic in 1/D?](#is-the-period-analytic-in-1d)
- [Large D](#large-d-what-the-1d-expansion-can-and-cannot-see): [How the solution scales](#how-the-solution-scales-with-d) ·
  [Proposition 1](#proposition-1-the-1n-expansion-never-fixes-the-period) ·
  [Proposition 2](#proposition-2-the-leading-order-problem-solved-exactly) ·
  [Proposition 3](#proposition-3-what-gravity-does-to-the-non-analytic-piece) ·
  [Proposition 4](#proposition-4-late-orders-of-the-leading-order-series) ·
  [The inner problem](#the-inner-problem-at-large-x) ·
  [What is known and what is open](#what-is-known-about-the-selection-and-what-is-open) · [References](#references)
- [Software: the Rust kernel](#software-the-rust-kernel): [Backends](#backends) · [Use](#use) ·
  [Double-double residual](#double-double-residual) ·
  [Exact Jacobian and growth-mode matrix](#exact-jacobian-and-growth-mode-matrix) · [Build and test](#build-and-test) ·
  [Performance](#performance) · [Layout](#layout) · [Limits](#limits)

## Results

![Echo period Δ(D) and critical exponent γ(D) with one rational formula for all D ≥ 4, and the double-double points to D = 61](solver/delta.svg)

The echo period Δ(D) and critical exponent γ(D), computed with the f64 solver in [`solver/`](solver/) (its residual is the
Rust kernel of [Software](#software-the-rust-kernel)) and the double-double solver in [`hp/`](hp/).
The dashed curves are one rational formula in 1/(D−2) for every D ≥ 4 (below), fitted on the f64 points
4 ≤ D ≤ 30.2 whose integration-step error has been extrapolated out ("step-converged"), with the solver's near-null
directions resolved and enough Fourier modes in time at low D;
hollow points are held out of the fit, and the green diamonds are double-double points up to D = 61, which it matches
to $`6\times10^{-8}`$. As D → ∞, γ → 1/2 − 0.758/n (the limit 1/2 is imposed in the γ fit; free fits give 0.4996–0.4998), and
Δ → Δ∞ = 1.64389375 ± 0.00000001 if Δ is analytic in 1/D (from the double-double points; the f64 points gave
1.64395 ± 0.00008). Details are in
[`solver/CONTEXT.md`](solver/CONTEXT.md). Regenerate with `cd solver/data && python3 ../plot_delta.py`.

### The formula

For massless scalar critical collapse in $`D = n + 2`$ spacetime dimensions, the echo period $`\Delta`$ and the critical
exponent $`\gamma`$ of Choptuik's discretely self-similar solution (Choptuik, PRL 70, 9 (1993)) are fitted, for every
$`D \ge 4`$ and within the tolerances in the table below, by

```math
\Delta(D) = \frac{a_0 + a_1 \varepsilon + a_2 \varepsilon^2 + \cdots + a_8 \varepsilon^8}{1 + b_1 \varepsilon + b_2 \varepsilon^2},
\qquad
\gamma(D) = \frac{\tfrac{1}{2} + e_1 \varepsilon + e_2 \varepsilon^2 + \cdots + e_7 \varepsilon^7}{1 + f_1 \varepsilon + f_2 \varepsilon^2},
\qquad
\varepsilon = \frac{1}{n} = \frac{1}{D-2},
```

| $`k`$ | $`a_k`$ ($`\Delta`$, numerator) | $`b_k`$ ($`\Delta`$, denominator) | $`e_k`$ ($`\gamma`$, numerator) | $`f_k`$ ($`\gamma`$, denominator) |
|---|---|---|---|---|
| 0 | $`+1.6438975180`$ | $`1`$ | $`+0.5`$ (fixed) | $`1`$ |
| 1 | $`+4.0378045077 \times 10^{1}`$ | $`+1.8251453626 \times 10^{1}`$ | $`+1.8083629635 \times 10^{1}`$ | $`+3.7683686025 \times 10^{1}`$ |
| 2 | $`+2.5233285989 \times 10^{2}`$ | $`+6.8182050167 \times 10^{1}`$ | $`+1.5241588758 \times 10^{2}`$ | $`+3.4245505235 \times 10^{2}`$ |
| 3 | $`+2.0673502321 \times 10^{2}`$ |  | $`-5.2341588541 \times 10^{1}`$ |  |
| 4 | $`-3.5709850575 \times 10^{2}`$ |  | $`+9.9259594505`$ |  |
| 5 | $`+3.1347320996 \times 10^{2}`$ |  | $`-9.8386450353 \times 10^{1}`$ |  |
| 6 | $`-4.1891374574 \times 10^{2}`$ |  | $`+7.4529506074 \times 10^{1}`$ |  |
| 7 | $`+3.2695252782 \times 10^{2}`$ |  | $`-4.8272358903 \times 10^{1}`$ |  |
| 8 | $`-1.6050467699 \times 10^{2}`$ |  |  |  |

Eleven numbers for $`\Delta`$ and nine for $`\gamma`$, with no singularity at any physical $`D`$ (the denominators vanish only
at $`D = -3.2, -11.0`$ and $`D = -13.3, -20.4`$). Fitted on every computed point $`4 \le D \le 30.2`$ (`solver/formula_all.py`;
`solver/rational_fit.py` searches the forms: for $`\Delta`$ one numerator term beyond the smallest adequate form makes it
ten times tighter), they reproduce each point within its own precision:

| $`D`$ | $`\Delta`$: formula − computed | precision of the points | $`\gamma`$: formula − computed | precision of the points |
|---|---|---|---|---|
| 4 – 4.3 | $`5\times10^{-10}`$ | $`3\times10^{-7}`$ | $`3.4\times10^{-7}`$ | $`5\times10^{-7}`$ |
| 4.3 – 20 | $`\le 4\times10^{-9}`$ | $`3\times10^{-11}`$ – $`4\times10^{-9}`$ (up to $`4\times10^{-10}`$ at $`D \approx 8`$–$`10`$) | $`\le 3.3\times10^{-8}`$ | $`\le 7\times10^{-8}`$ |
| 20 – 30 | $`\le 1\times10^{-8}`$ | $`\le 3\times10^{-8}`$ | $`\le 2.3\times10^{-9}`$ | $`\sim 10^{-7}`$ (assigned $`10^{-9}`$, but the fits' $`\chi^2/\mathrm{dof}`$ is about 7000 against it) |
| 30.2 – 61, never seen by the fit | $`\le 6\times10^{-8}`$ | $`\le 10^{-14}`$ (double-double) | $`\le 5\times10^{-7}`$ up to $`D = 35.8`$ | |

An independent double-double computation ([below](#the-double-double-solver-hp)) agrees with the formula to
$`2.7\times10^{-9}`$ or better on $`8.25 \le D \le 21`$, $`7.5\times10^{-9}`$ up to $`D = 30`$, and $`5.8\times10^{-8}`$ from there to
$`D = 61`$, which the fit never saw. It also shows that the f64 points at $`D \approx 8`$–$`10`$ (23 Fourier modes in time)
sit up to $`4\times10^{-10}`$ above the true values (from $`1.3\times10^{-11}`$ to $`3.7\times10^{-10}`$ on $`8.25`$–$`9.9`$; within
$`3\times10^{-11}`$ on $`10`$–$`15.2`$ and $`5.6\times10^{-11}`$ on $`15.6`$–$`21`$), so their quoted $`3\times10^{-11}`$ was optimistic
there; that is below the formula's tolerance and does not change it.

At $`D = 4`$ the formula gives $`\Delta = 3.4454524`$ and $`\gamma = 0.3739610`$, the accepted four-dimensional values
($`\Delta = 3.445452402(3)`$, Martín-García & Gundlach, PRD 68, 024011 (2003); $`\gamma = 0.373961`$, Ecker et al. 2026; see
also Gundlach, PRD 55, 695 (1997)).
Near $`D = 4`$ the points themselves set the limit: the solution needs many Fourier modes in time there, and $`\Delta(4)`$
moves by $`-4.3\times10^{-4}`$, $`-3.4\times10^{-5}`$ and $`-2.8\times10^{-6}`$ from 23 to 31 to 39 to 47 modes, so the points
below $`D = 5`$ are extrapolations in the mode count. Above $`D \approx 36`$ the computed $`\gamma`$ scatter by up to
$`6\times10^{-5}`$ around both this formula and the power series below, which agree with each other; the growth-rate solve
is not reliable there. Below $`D = 4`$ the formula is not fitted; $`\Delta(D)`$ turns over there. The cubic through the
47-mode points at $`D = 3.70, 3.75, 3.76, 3.80`$ (`solver/data/scan_down_kc47.jsonl`) puts the maximum at $`D = 3.75570`$,
$`\Delta = 3.466774`$ (a quartic with $`D = 3.9`$ added: the same to $`2\times10^{-6}`$ in $`D`$, $`3\times10^{-8}`$ in $`\Delta`$), in agreement to $`2\times10^{-6}`$ with
$`D = 3.755726`$, $`\Delta = 3.466772`$ of Ecker, Ecker, Grumiller & Jechtl 2026 (below); the rational formula, extended
outside its range, peaks at $`D = 3.75558`$, $`\Delta = 3.466781`$.

As $`D \to \infty`$ ($`\varepsilon \to 0`$):

```math
\Delta(D) = a_0 + \frac{a_1 - a_0 b_1}{n} + \cdots \approx 1.6439 + \frac{10.37}{n} + \cdots,
\qquad
\gamma(D) = \frac{1}{2} + \frac{e_1 - f_1/2}{n} + \cdots \approx \frac{1}{2} - \frac{0.758}{n} + \cdots
```

```math
\Delta_\infty = 1.64389375 \pm 0.00000001,
\qquad
\omega_\infty = \frac{2\pi}{\Delta_\infty} = 3.82213589 \pm 0.00000002,
\qquad
\gamma_\infty = \frac{1}{2} .
```

$`\gamma_\infty = 1/2`$ is imposed in both $`\gamma`$ fits (their constant term is fixed), not derived. Unconstrained fits on
$`D \ge 8`$–$`12`$ give $`\gamma_\infty = 0.4996`$–$`0.4998`$, so the data support $`1/2`$ to about $`4\times10^{-4}`$; the f64 $`\gamma`$ values
reach only $`D = 40`$ ($`\gamma(40) = 0.4848`$). $`1/2`$ is also the value conjectured in the literature
([below](#relation-to-published-work)), and like $`\Delta_\infty`$ it is not fixed at any finite order of the $`1/n`$ expansion
([Large D](#what-is-known-about-the-selection-and-what-is-open)).

$`\Delta_\infty`$ comes from fits of all the double-double points, which give limits from $`1.643893744`$ to $`1.643893752`$
([Is the period analytic in 1/D?](#is-the-period-analytic-in-1d), where the fits and their residuals are listed). The f64
points gave $`1.64395 \pm 0.00008`$, and this formula's own limit is $`a_0 = 1.643898`$, $`4\times10^{-6}`$ high, which its fitting
range could not see. The coefficients of one fit are not
physical constants (they are correlated); only the function is.

**The limit rests on an assumption, which the double-double points now support.** $`\Delta_\infty = 1.64389375`$ holds if $`\Delta(D)`$ is analytic in $`1/D`$ at
$`D = \infty`$: no half-integer powers $`n^{-1/2}`$, no $`\log n`$, and no new scale far beyond the computed range. The f64
data could not test that. Formulas with those features reproduce every f64 point from $`D = 4`$ to $`40`$ (fit and held
out) just as well: a series in $`n^{-1/2}`$ gives limits from $`1.607`$ to $`1.691`$ depending on its degree; adding one
$`n^{-1/2}`$ or $`\log n / n`$ term to the $`1/D`$ series moves the limit to $`1.6437`$–$`1.6448`$; and the rational form with
its limit forced to $`\pi^2/6`$ or $`\sqrt e`$ fits too, by placing a pole at $`D \approx -440`$ or $`-5900`$, a slow bend
that starts in the hundreds. All of these agree with the formula above to $`10^{-5}`$ up to $`D \approx 60`$ and fan out
above $`D \approx 100`$. The double-double points test it: fitted on $`D \le 45`$, the analytic form predicts the points
up to $`D = 60`$ to $`5\times10^{-13}`$, 40 to 300 times better than series in half-integer powers, and no added
non-analytic term improves it ([below](#is-the-period-analytic-in-1d)).

![Formulas that fit the f64 points, extended to D = 100000; the double-double points rule out all but the analytic ones](solver/extrapolation.svg)

The analytic form is the natural default (the large-$`D`$ expansion of the equations has no other scale, and the only
structure the data show is a singularity on the negative $`D`$ axis, near $`D \approx -9`$ to $`-11`$), but it is not proven:
the period is selected by an exponentially small effect that the $`1/D`$ expansion never sees, and Ecker, Ecker and
Grumiller conjecture $`\Delta \to 0`$ ([below](#relation-to-published-work)). The double-double points up to $`D = 61`$
favour the analytic form strongly; a proof needs the $`D \to \infty`$ equations themselves.
`solver/plot_extrapolation.py` draws the graph; the faded curves are the ones the double-double points rule out.

<details>
<summary>Power series in 1/D for D ≥ 10 (the earlier form)</summary>

```math
\Delta(D) = \Delta_\infty + \sum_{k=1}^{10} \frac{c_k}{D^k},
\qquad
\gamma(D) = \frac{1}{2} + \sum_{k=1}^{10} \frac{g_k}{D^k},
\qquad
\Delta(D) \approx 1.64389 + \frac{10.37}{D} - \frac{28.4}{D^2} + \frac{239}{D^3} - \cdots
```

Fitted on $`10 \le D \le 30.2`$ (`solver/formula.py`); it reproduces the points there to
$`1.6\times10^{-8}`$ ($`\Delta`$) and $`2.3\times10^{-9}`$ ($`\gamma`$; the $`\gamma`$ points themselves are good to about $`10^{-7}`$). The coefficients grow about ninefold per order with
alternating signs: a singularity near $`D \approx -9`$ on the negative axis, which the rational form represents by the
zero of its denominator at $`D = -11.0`$. That singularity
limits the series to $`D \gtrsim 10`$ and forbids cutting it short (five terms are off by $`4\times10^{-3}`$ at $`D \ge 20`$).

| $`k`$ | $`c_k`$ (for $`\Delta`$) | $`g_k`$ (for $`\gamma`$) |
|---|---|---|
| 0 | $`+1.6438931360`$ | $`+0.5`$ (fixed) |
| 1 | $`+1.0374964007 \times 10^{1}`$ | $`-7.5854191958 \times 10^{-1}`$ |
| 2 | $`-2.8355347834 \times 10^{1}`$ | $`+8.2843462006`$ |
| 3 | $`+2.3900833130 \times 10^{2}`$ | $`-1.2603029673 \times 10^{2}`$ |
| 4 | $`-2.2446858552 \times 10^{3}`$ | $`+1.8446582182 \times 10^{3}`$ |
| 5 | $`+2.0424472078 \times 10^{4}`$ | $`-2.5165496284 \times 10^{4}`$ |
| 6 | $`-1.6230686158 \times 10^{5}`$ | $`+2.8434887038 \times 10^{5}`$ |
| 7 | $`+1.0037347608 \times 10^{6}`$ | $`-2.4157799331 \times 10^{6}`$ |
| 8 | $`-4.4127962953 \times 10^{6}`$ | $`+1.4042658549 \times 10^{7}`$ |
| 9 | $`+1.2046388629 \times 10^{7}`$ | $`-4.9128734266 \times 10^{7}`$ |
| 10 | $`-1.5266460427 \times 10^{7}`$ | $`+7.7494021741 \times 10^{7}`$ |

</details>

### Values

| $`D`$ | $`\Delta`$ | $`\gamma`$ |
|---|---|---|
| 4 | 3.4454524 | 0.3739610 |
| 5 | 3.2217019 | 0.4131269 |
| 6 | 3.0094832 | 0.4300795 |
| 7 | 2.8427532 | 0.4399912 |
| 8 | 2.711790054444 | 0.4467164 |
| 10 | 2.521078919562 | 0.4555879 |
| 12 | 2.389188553641 | 0.4614056 |
| 15 | 2.253098407194 | 0.4673656 |
| 20 | 2.112080154580 | 0.4737107 |
| 30 | 1.964957899258 | 0.4807927 |
| 40 | 1.888568197910 | 0.484776 |
| 50 | 1.841659503007 | 0.48736 |
| 60 | 1.809889459975 | |
| 70 | 1.78693476494 | |
| 100 | 1.7450264844 | 0.49313 |
| 1000 | 1.65424086 | 0.4992 |
| $`\infty`$ | $`1.64389375 \pm 0.00000001`$ | $`1/2`$ (assumed) |

$`\Delta`$ for $`D = 10`$–$`60`$ is double-double, interpolated to $`10^{-14}`$ from [`hp/data/delta_dd.csv`](hp/data/delta_dd.csv)
where $`D`$ is not itself a computed point; $`\Delta(8)`$ is the solver's separate check run at $`D = 8`$ (Newton to
$`\lvert F\rvert = 5\times10^{-31}`$, checks table below), which is not in the data file. $`\Delta`$ for $`D = 4`$–$`7`$ and $`\gamma`$ up to
$`D = 40`$ are from the f64 solver, within the precision of its points; $`\gamma(50)`$, $`\gamma(100)`$ and $`\gamma(1000)`$ are the
rational formula's values (the solver was not run there, and its $`\gamma`$ is unreliable above $`D \approx 36`$).
$`\Delta(70)`$ and $`\Delta(100)`$ are extrapolations that every model fitting the double-double points agrees on,
to $`10^{-11}`$ and $`2\times10^{-8}`$ (the half-integer series included). $`D = 1000`$ and $`\infty`$ assume analyticity in
$`1/D`$; the half-integer series differ there by $`3\times10^{-6}`$ and more.

The physics figure below and the candidate test use these values; the double-double points behind them are described in
[The double-double solver](#the-double-double-solver-hp).

### What the numbers mean physically

![e^Δ, the fine-tuning per echo and the strength of the field and its gravity, from D = 4 to D = infinity](solver/physics.svg)

The same two numbers, read as physics (`solver/plot_physics.py`):

- **Each echo is a factor $`e^{\Delta}`$ smaller than the last:** 31 at $`D = 4`$, 12 at $`D = 10`$, 5.2 at $`D = \infty`$. In higher
  dimensions the echoes come closer together in scale.
- **One more echo costs $`\Delta/(\gamma\ln 10)`$ digits of fine-tuning.** A collapse whose initial strength $`p`$ is tuned to
  $`\lvert p - p^*\rvert`$ makes about $`\gamma\ln(1/\lvert p - p^*\rvert)/\Delta`$ echoes before it either forms a black hole or disperses,
  so each extra echo needs $`\lvert p - p^*\rvert`$ smaller by $`e^{\Delta/\gamma}`$: 4.0 digits at $`D = 4`$, 2.4 at $`D = 10`$, 1.43 at $`D = \infty`$.
- **The field and its gravity weaken as $`D`$ grows:** the centre slope of the field falls like $`2.72/\sqrt n`$ and the
  departure of spacetime from flat at the centre like $`3.65/n`$ (double-double solutions, $`D = 20`$–$`60`$). At $`D = \infty`$ the
  echo is a free wave on flat space, which on its own cannot echo (Proposition 2); the selection of $`\Delta`$ is gravity's
  doing even there ([Large D](#large-d-what-the-1d-expansion-can-and-cannot-see)).

The phenomenon is the same at every $`D \ge 4`$: the same echoing structure and a smooth $`\Delta(D)`$, with one unstable mode
as in four dimensions (assumed here: only that mode's growth rate is computed, not the full spectrum). Only the numbers
change.

### Candidate closed forms, all ruled out

Nine closed forms were tried as the limit, first on
the f64 points and then on the double-double points. None of them passes the second test, so none of them is
$`\Delta_\infty`$, given that $`\Delta`$ is analytic in $`1/D`$ (which the double-double points support,
[below](#is-the-period-analytic-in-1d)).

The closest candidate sits $`1.3\times10^{-5}`$ from the limit, more than 1000 times its uncertainty, and forcing any of them makes the fit predict the unseen points at least 100 times worse. A search over products of standard constants finds nothing inside the double-double
window either.

<details>
<summary>The two tests, the table, the influence function and the search</summary>

- **f64 test** ($`D \le 39.5`$, `solver/candidates.py`): each candidate forced as $`c_0`$ in 12 fits (variable $`1/D`$,
  $`1/(D-1)`$, $`1/(D-2)`$; degree 9 and 10; lower cut $`D \ge 10`$ or $`12`$). The table gives $`\chi^2/\mathrm{dof}`$ of the best
  and the worst of the 12 (0.62 is the unconstrained fit) and the rms error on the held-out points at $`D = 34`$–$`39.5`$.
  Five candidates passed it; four were already excluded.
- **Double-double test** (made on the points then available, up to $`D = 59.8`$; `hp/dd_candidates.py 45 --dmax=59.8`
  repeats it on today's file, where later runs from the second machine have added a few points in the same range, so
  counts can differ by one or two): each candidate forced as the limit of a power series in $`1/D`$ fitted on
  $`10 \le D \le 45`$ at degree 14 to 26, and compared with the 351 cleaned points from $`45`$ to $`59.8`$ that the fit never
  saw. The table gives the best of those degrees and how many times worse it is than the free fit ($`4.7\times10^{-13}`$
  at degree 16 on this range; held out to $`60.1`$ the same fit gives $`5.2\times10^{-13}`$, [below](#is-the-period-analytic-in-1d)).

| Candidate | Value | − double-double limit | f64 test: $`\chi^2/\mathrm{dof}`$; held-out rms | Double-double test: held-out error (× free) | Verdict |
|---|---|---|---|---|---|
| none (free fit) | $`1.64389375 \pm 0.00000001`$ | | 0.62 .. 3.4; $`1.4\times10^{-6}`$ | $`4.7\times10^{-13}`$ | |
| $`\frac{5}{7}\sqrt{G}\, j_{0,1}`$ | 1.6439747 | $`+8.1\times10^{-5}`$ | 0.62 .. 8.6; $`1.3\times10^{-6}`$ | $`6.2\times10^{-11}`$ (130×) | **ruled out** (passed the f64 test) |
| $`\frac{7}{5}\,(\ln 2)^{-2}\,\pi^{-1/2}`$ | 1.6440014 | $`+1.1\times10^{-4}`$ | 0.62 .. 18; $`1.2\times10^{-6}`$ | $`1.1\times10^{-10}`$ (220×) | **ruled out** (passed the f64 test) |
| $`\frac{7}{2}\sqrt{\ln 2/\pi}`$ | 1.6440152 | $`+1.2\times10^{-4}`$ | 0.62 .. 26; $`1.2\times10^{-6}`$ | $`1.3\times10^{-10}`$ (270×) | **ruled out** (passed the f64 test) |
| $`\frac{9}{8}\sqrt{j_{2,1}/j_{0,1}}`$ | 1.6440207 | $`+1.3\times10^{-4}`$ | 0.62 .. 30; $`1.2\times10^{-6}`$ | $`1.4\times10^{-10}`$ (290×) | **ruled out** (passed the f64 test) |
| $`\frac{10}{9}\sqrt{j_{0,1}/\ln 3}`$ | 1.6439063 | $`+1.3\times10^{-5}`$ | 0.62 .. 118; $`1.5\times10^{-6}`$ | $`5.4\times10^{-11}`$ (110×) | **ruled out** (passed the f64 test) |
| $`\pi^2/6 = \zeta(2)`$ | 1.6449341 | $`+1.0\times10^{-3}`$ | 3.6 .. 13000; $`1.9\times10^{-6}`$ | $`1.7\times10^{-9}`$ (3500×) | **ruled out** by both |
| $`2\pi/j_{1,1}`$ | 1.6397878 | $`-4.1\times10^{-3}`$ | 67 .. $`3\times10^{5}`$; $`1.3\times10^{-5}`$ | $`7.0\times10^{-9}`$ (15000×) | **ruled out** by both |
| $`\sqrt{e}`$ | 1.6487213 | $`+4.8\times10^{-3}`$ | 82 .. $`3\times10^{5}`$; $`1.3\times10^{-5}`$ | $`8.1\times10^{-9}`$ (17000×) | **ruled out** by both |
| $`\varphi = (1+\sqrt5)/2`$ | 1.6180340 | $`-2.6\times10^{-2}`$ | 2500 .. $`10^{7}`$; $`7.7\times10^{-5}`$ | $`4.4\times10^{-8}`$ (92000×) | **ruled out** by both |

The f64 test could not separate the first five: they move $`\Delta(D)`$ by less than $`10^{-6}`$ for $`D \le 40`$, below the
precision of those points. The double-double points can. A
forced fit can only bend toward its candidate by spending degrees, so it improves with degree (the table gives each
one's best) until it reaches the level where the free fits overfit too (about $`5\times10^{-11}`$ at degree 24–26), which is
still about 100 times the free fit's best.

![How far each candidate constant moves Δ(D) over the computed range](solver/candidates.svg)

The graph is from the f64 test. It shows, for every candidate, how much fixing $`\Delta_\infty`$ to it changes the fitted $`\Delta(D)`$ over the computed
range ($`|\text{forced fit} - \text{free fit}|`$, log scale), against the precision of the computed points (shaded).
The five forms inside the window stay at the level of the data; $`\pi^2/6`$ sits 10–100 times above it at
$`D = 10`$–$`30`$, and $`2\pi/j_{1,1}`$, $`\sqrt e`$ and $`\varphi`$ climb further out. Every curve has the same shape: forcing the
limit is a linear constraint on a weighted least-squares fit, so forced minus free is (candidate − free limit) times one
influence function $`h(D)`$ with $`h(\infty) = 1`$. It is $`\sim 10^{-7}`$ where the points are known to about $`3\times10^{-11}`$
($`D \le 22`$; up to $`4\times10^{-10}`$ at $`D \approx 8`$–$`10`$), rises where their error bars step up ($`D = 22`$, $`29.5`$), and climbs toward 1 past the data:
$`h(40) = 3.8\times10^{-3}`$, $`h(100) = 0.17`$, $`h(1000) = 0.85`$. Regenerate with
`cd solver/data && python3 ../plot_candidates.py`.

$`G`$ is Catalan's constant and $`j_{\nu,k}`$ the $`k`$-th zero of the Bessel function $`J_\nu`$. The five that passed the f64
test are all that a search finds in the f64-era window $`1.64388`$–$`1.64403`$ (about $`\pm 8\times10^{-5}`$ around the f64 estimate) among products $`r\, b_1^{p_1} b_2^{p_2}`$
($`r = i/j`$ with $`i, j \le 12`$; $`p = \pm\tfrac12, \pm1, \pm2`$; $`b`$ from $`\pi`$, $`e`$, $`\sqrt2`$, $`\sqrt3`$, $`\ln 2`$, $`\ln 3`$, Euler's
$`\gamma`$, $`\zeta(3)`$, $`\varphi`$, $`G`$ and Bessel zeros: $`j_{0,1}, j_{0,2}, j_{1,1}, j_{1,2}, j_{2,1}, j_{3/2,1}, j_{5/2,1}`$ and $`y_{0,1}`$, as in
`solver/candidates.py`), $`2\pi/(r b)`$ and $`r \ln b`$. The same search finds nothing in the double-double window
$`1.6438936`$–$`1.6438939`$ (deliberately $`\pm1.5\times10^{-7}`$, 15 times the stated error, so that a constant just outside
the error bar would still show), nor in one about 27 times wider ($`1.643890`$–$`1.643898`$). Whether
$`\Delta_\infty`$ has a closed form at all is open; the $`D \to \infty`$ limit of the equations is the way to find out.

</details>

### Relation to published work

From a literature search (October 2026); the references are listed at the end of the [Large D](#references) section.

- **D = 4.** $`\Delta = 3.445452402(3)`$ (Martín-García & Gundlach 2003) and $`\gamma = 0.373961 \pm 10^{-6}`$ (Ecker, Ecker,
  Grumiller & Jechtl 2026, who also give $`\Delta = 3.445453`$). The formula's $`3.4454524`$ and $`0.3739610`$ agree to about
  $`10^{-8}`$ and $`10^{-7}`$.
- **The maximum below D = 4.** Ecker et al. 2026, Eq. (70): $`D = 3.755726 \pm 10^{-6}`$, $`\Delta = 3.466772 \pm 10^{-6}`$. The 47-mode
  points here give $`D = 3.75570`$, $`\Delta = 3.466774`$ (agreement to about $`2\times10^{-6}`$).
- **D = 5: an unresolved discrepancy.** Ecker et al. 2026 give $`\Delta = 3.22176`$, $`\gamma = 0.41322`$, which are
  $`5.8\times10^{-5}`$ and $`9.3\times10^{-5}`$ above the values here ($`3.2217019`$, $`0.4131269`$). Ours are stable across 31, 39 and 47
  Fourier modes to about $`10^{-7}`$, and Bland & Kunstatter (2007) give $`\gamma(5) = 0.4131 \pm 0.0001`$, matching ours. The cause
  is not known.
- **Below D = 4.** $`\gamma`$ extrapolated from the 47-mode points ($`\gamma(3.7) = 0.34865`$) suggests $`\gamma(3.5) \lesssim 0.33`$, against
  $`0.349 \pm 0.003`$ from Bland et al. (2005). The points here below $`D = 4`$ are under-resolved in the Fourier modes; this is
  an open check.
- **Large D.** Ecker, Ecker & Grumiller (2026) and Ecker et al. (2026) conjecture $`\Delta \to 0`$ as $`D \to \infty`$; Bland's 2007
  PhD thesis (University of Manitoba) hypothesizes $`\Delta_\infty = 5/2`$, which the data exclude ($`\Delta(14) = 2.29`$ and
  decreasing). The double-double points give a finite $`\Delta_\infty = 1.64389375`$ if $`\Delta`$ is analytic in $`1/D`$. Up to $`D = 61`$
  they show no sign of a turn toward 0, but they cannot exclude non-analytic behaviour beyond the data. $`\gamma_\infty = 1/2`$
  agrees with the conjectures there; it is not new.
- **Leading order.** That at leading order any periodic profile with any period solves the equations is in Ecker, Ecker
  & Grumiller 2026, and the free wave equation in similarity coordinates was analysed in 3+1 by Donninger (2008). Their
  next-to-leading-order condition is discussed next.

#### Relation to Ecker, Ecker & Grumiller

Ecker, Ecker & Grumiller (PRL 136, 191401 (2026), [arXiv:2601.14358](https://arxiv.org/abs/2601.14358)) use the same gauge in
their large-$`D`$ expansion: their $`f = \alpha/a`$ is our $`c`$, with $`f = 1`$ at the self-similarity horizon $`x = 1`$ for all $`\tau`$.
At leading order they find the same freedom as Proposition 1: any periodic profile
$`\beta(\tau) = \Pi(0,\tau) \approx \sqrt{2/n}\,F(\omega\tau)`$, with any period. At next order they require $`f \le 1`$ near the horizon at the zeros of $`\beta`$, which
gives $`\beta'' + 3\beta' = 0`$ there (their Eq. (20)); in our variables, $`\omega^2F'' + 3\omega F' = 0`$ at the zeros of $`F`$.

The inequality $`f \le 1`$ holds for every regular solution, whatever the period. From $`x b_x = b(p^2+m^2) - (n-1)b(b-1)`$ with
$`b(0) = 1`$, $`b`$ can never drop below 1, since at $`b = 1`$ the slope is $`p^2+m^2 \ge 0`$; then $`x c_x = (n-1)c(b-1) \ge 0`$, so
$`c \le c(1) = 1`$. Its apparent violation at next-to-leading order comes from truncating a perfect square in a window of
width $`O(1/n)`$ around each zero. In the formal family of Proposition 1,

```math
b - 1 = 2\varepsilon^2x^2F(\theta-\delta\theta)^2 + 2\varepsilon^4\omega^2x^4F'^2 + \ldots, \qquad \delta\theta(x) = -\varepsilon\omega x^2\,\frac{3F'+\omega F''}{2F'} .
```

Their next-to-leading-order term is the cross term, which shifts the field's zero; the square term arrives one order
later. The discriminant $`16\omega^2x^6F'^4`$ is positive for every $`F`$ and $`\omega`$. The double-double solutions bear this out:

- They violate Eq. (20) completely: $`(\omega^2F''+3\omega F')/(3\omega F') = 0.98`$–$`1.06`$ at $`D = 12`$–$`60`$, since $`F`$ is nearly
  antisymmetric about its zeros.
- The zero line of $`p - m`$ tilts as $`\delta\theta(x)`$ predicts: measured over predicted at $`x = 1`$ is 0.69, 0.78, 0.84, 0.88 at
  $`D = 20, 30, 45, 60`$, and the agreement is about 1% for $`x \le 0.3`$.
- The maximum of $`c`$ at the centre approaches $`1 - (\varepsilon^3/2)\,\omega^2F'^2`$, their Eq. (22) in this scaling (ratio 0.36
  rising to 0.86 from $`D = 20`$ to $`60`$).

So Eq. (20) does not select $`\Delta`$, and Proposition 1 stands.

Their conjecture $`\Delta \to 0`$ comes with no rate. Fitted on $`10 \le D \le 45`$ and compared with $`45 < D \le 61`$:

- a power-law approach to 0 ($`D^{-p}`$ times a series in $`1/D`$, $`p \ge 0.01`$) misses the held-out points by $`6\times10^{-8}`$ or
  more, against $`7\times10^{-13}`$ for the analytic series;
- a series in $`1/\ln D`$ with zero limit misses them by $`1.3\times10^{-11}`$ and still gives $`\Delta \approx 0.88`$ at $`D = 10^{30}`$;
- adding a $`c/\ln D`$ term to the analytic series gives $`c \approx 4\times10^{-7}`$ and leaves the limit at $`1.6438937`$.

A power-law approach to 0 is excluded; a logarithmic one is disfavoured but cannot be excluded from $`D \le 61`$ alone.
Scripts: [`solver/eeg20/`](solver/eeg20/).

## Evidence

### The double-double solver (`hp/`)

Above $`D \approx 26`$ the f64 points lose precision: the period is selected by an effect that weakens exponentially with
$`D`$, so rounding at $`10^{-16}`$ in the residual moves $`\Delta`$ by $`10^{-7}`$–$`10^{-6}`$ at $`D = 30`$–$`40`$ (through the sensitivity
$`\max\lvert\partial\Delta/\partial F\rvert`$ of the table below), and points far above $`D = 40`$ are out of
reach. [`hp/`](hp/) (crate `hpkernel`) is a separate solver built for that regime, to settle the large-$`D`$ behaviour
with computed points rather than an assumed form. Its points are used in [The formula](#the-formula) (agreement and
$`\Delta_\infty`$), [Values](#values), the [candidate test](#candidate-closed-forms-all-ruled-out), the
[analyticity tests](#is-the-period-analytic-in-1d) and the large-$`D`$ scaling ([How the solution scales with D](#how-the-solution-scales-with-d)).

- **Arithmetic:** double-double (hi + lo, about 32 digits) on 8-lane vectors: one AVX-512 register per lane group, two
  AVX2 registers, or a portable array, chosen at compile time (`hp/build.sh` builds with `-C target-cpu=native`).
  Error-free two-sum and FMA two-product; sums of products accumulated Sum2-style.
- **No fixed-step error:** Taylor series in $`x`$ (order 36, step points planned once per $`D`$ so the truncation term is
  $`10^{-32}`$), centre and horizon series run until their tails are below $`10^{-33}`$, and collocation in $`\tau`$ on the
  half period ($`p, m`$ odd harmonics, $`a, c`$ even harmonics in $`\tau`$) with no spectral truncation of the products
  (from $`D = 35`$ on, the centre data's harmonics above 13 are frozen; below).
- **Exact Jacobian:** forward-mode duals over double-double, double-double LU for Newton, a transpose solve that gives
  $`\partial\Delta/\partial F`$ at every point, and a Levenberg–Marquardt fallback.
- **Continuation in $`D`$:** prediction by the cubic through the last four converged points (a tangent from the
  Jacobian is wrong in the weak directions, below), Newton with the weakest directions deflated, and from $`D = 35`$ on
  the centre data's $`\tau`$-harmonics above 13 frozen (below). 32 collocation points in $`\tau`$, nodes at
  $`x = 0.3, 0.4, \ldots, 0.7`$ (the node checks in the table below moved the first node closer to the centre); 30–50 s per
  point on two cores up to $`D = 60`$. The tolerance on $`\lvert F\rvert`$ goes from $`10^{-25}`$ at $`D = 35`$ to $`10^{-22}`$ at $`D = 56`$,
  following the residual floor, which rises about 1.4-fold per unit of $`D`$. Above $`D = 60.1`$, on the second machine, it
  was relaxed to $`3\times10^{-19}`$ for part of the climb ($`\lvert F\rvert`$ up to $`2.9\times10^{-19}`$ there).

<details>
<summary>Checks of the solver</summary>

| Check | Result |
|---|---|
| double-double $`+, \times, \div`$ against mpmath | $`1.3`$–$`1.6\times10^{-32}`$ relative |
| horizon series against 40-digit mpmath | $`2\times10^{-32}`$ |
| Jacobian against double-double differences | agrees to the difference truncation |
| Newton at $`D = 8`$ | quadratic, $`\lvert F\rvert = 5\times10^{-31}`$ in 4 steps |
| $`\Delta(7)`$, $`\Delta(7.75)`$ against the 39-mode references ($`\pm3\times10^{-11}`$) | $`+3.1\times10^{-12}`$, $`-2.8\times10^{-12}`$ |
| 32 / 48 / 64 collocation points in $`\tau`$, $`D = 7`$–$`17`$ | $`\Delta`$ agrees to $`10^{-16}`$ |
| first node at $`x = 0.02`$ / $`0.05`$ / $`0.1`$, $`D = 16.75`$ | $`\Delta`$ identical to $`10^{-16}`$ |
| 32 / 48 collocation points in $`\tau`$, $`D = 40`$ | $`\Delta`$ agrees to $`9\times10^{-22}`$ |
| harmonics frozen above 13 / above 15, $`D = 40`$ | $`\Delta`$ agrees to $`3\times10^{-17}`$ |
| second machine (Windows, i7-1185G7, own build), $`D = 59.84`$–$`60.05`$ | agrees with the cloud points to $`2`$–$`4\times10^{-16}`$ |
| the 32 points at $`D = 60.95`$–$`61.026`$ (25 from the second machine, 7 from the cloud) against the $`1/D`$ fit on $`D \le 60.1`$ (degree 18) | agree to $`2.6\times10^{-15}`$ |

</details>

<details>
<summary>A sample of the computed points</summary>

A sample of the computed points (all of them, from $`D = 8.25`$ up, are in
[`hp/data/delta_dd.csv`](hp/data/delta_dd.csv)), with the residual and the largest entry of $`\partial\Delta/\partial F`$, the factor by which a residual error moves $`\Delta`$:

| $`D`$ | $`\Delta`$ | $`\lvert F\rvert`$ | $`\max\lvert\partial\Delta/\partial F\rvert`$ |
|---|---|---|---|
| 8.25 | 2.683434296073424 | $`5\times10^{-31}`$ | $`1.2\times10^{2}`$ |
| 12 | 2.389188553640616 | $`8\times10^{-31}`$ | $`5.9\times10^{2}`$ |
| 16 | 2.218342998377885 | $`4\times10^{-30}`$ | $`6.8\times10^{3}`$ |
| 21.19141 | 2.087734162683867 | $`4\times10^{-29}`$ | $`2.0\times10^{5}`$ |
| 25.07972 | 2.023511166632091 | $`1\times10^{-28}`$ | $`2.1\times10^{6}`$ |
| 29.98475 | 1.965111240898741 | $`1\times10^{-15}`$ \* | $`8.0\times10^{7}`$ |
| 35.35323 | 1.918943717094251 | $`4\times10^{-27}`$ | $`7.0\times10^{8}`$ |
| 40.02007 | 1.888451664490614 | $`2\times10^{-26}`$ | $`9.6\times10^{9}`$ |
| 44.96341 | 1.862784197662323 | $`9\times10^{-26}`$ | $`1.3\times10^{11}`$ |
| 50.00749 | 1.841631133202264 | $`1\times10^{-24}`$ | $`1.5\times10^{12}`$ |
| 54.99609 | 1.824394337272626 | $`6\times10^{-24}`$ | $`1.2\times10^{13}`$ |
| 60.00195 | 1.809884258001715 | $`2\times10^{-23}`$ | $`8.9\times10^{13}`$ |

\* From $`D = 27`$ to $`35.2`$, before the freezing, points were accepted once the linearised correction to $`\Delta`$,
$`-\partial\Delta/\partial F \cdot F`$, was below $`10^{-14}`$ with the weak directions held on their trend; their residual
sits in those directions.

</details>

What the solver has shown:

- **The weak directions.** The near-null directions of the Jacobian that stalled the f64 solver (called "junk" in the
  code and notes) are a ladder of high $`\tau`$-harmonics of the centre data, one pair per odd harmonic $`k`$, with
  singular values falling about like $`e^{-0.7k}`$
  ($`10^{-10}`$ at $`k = 7`$ down to $`7\times10^{-18}`$ at $`k = 31`$ at $`D = 21`$; the weakest is $`10^{-24}`$ at $`D = 35`$). They
  are outgoing waves near the centre that barely reach the matching point, and they belong to the boundary-value problem
  itself: moving the first node or the matching point leaves them unchanged. Their own effect on $`\Delta`$ is tiny
  ($`\le 10^{-12}`$ per unit amplitude for $`k \ge 17`$), but second-order errors leak into them amplified by $`1/\sigma`$,
  so Newton's basin shrinks with $`D`$; at $`D = 8`$, 80 points in $`\tau`$ already fail to converge. From $`D = 35`$ the solver
  pins the centre data's harmonics above 13 at zero and drops their matching on the horizon side: the weakest singular
  value rises from $`10^{-24}`$ to $`10^{-17}`$ and plain Newton converges again. The cut moves $`\Delta`$ by $`5\times10^{-15}`$ at
  $`D = 35`$ and $`3\times10^{-17}`$ at $`D = 40`$; a cut at 11 does not converge at $`D = 40`$.
- **How fast the selection weakens.** $`\max\lvert\partial\Delta/\partial F\rvert`$ grows about $`1.8`$ times per unit of $`D`$ below
  $`D = 27`$, $`1.7`$ at $`35`$–$`45`$, $`1.6`$ at $`45`$–$`56`$ and $`1.5`$ at $`56`$–$`61`$, to $`9\times10^{13}`$ at $`D = 60`$ (table), and the
  residual floor rises about 1.4-fold per unit. Their product, a
  worst case that ignores the direction of $`F`$, bounds the error in $`\Delta`$ by $`10^{-14}`$ at $`D = 45`$ and $`2\times10^{-9}`$ at
  $`D = 60`$. The linearised estimate $`-\partial\Delta/\partial F \cdot F`$, which accounts for it, stays between
  $`10^{-21}`$ and $`10^{-17}`$ from $`D = 35.3`$ to $`60.1`$ and reaches $`3\times10^{-15}`$ above, where the tolerance was relaxed;
  the points themselves fit smooth curves to about $`4\times10^{-15}`$ (degree 18–20) all the way to $`D = 60.1`$.
- **How good the values are.** The points fit a smooth curve to about $`4\times10^{-15}`$, not $`10^{-20}`$. Points between
  $`D = 27`$ and $`35.2`$, solved before the freezing, carry a few $`10^{-15}`$ from the weak directions; about 25 of them sit
  visibly off their neighbours and `hp/dd_fits_mp.py --clean` drops them (26 on today's file with `--dmax=60.1`). The frozen points from $`35.35`$ to $`37`$ carry the cut's $`\le 5\times10^{-15}`$, which falls off
  with $`D`$. Elsewhere up to $`D = 60.1`$ the error is about $`10^{-16}`$; the 94 points above $`60.1`$, solved to the looser
  tolerance, carry up to about $`3\times10^{-15}`$.
- **The f64 points.** Against the double-double values they are within $`5.5\times10^{-11}`$ at $`D = 15.6`$–$`21`$ (quoted
  $`3\times10^{-11}`$), typically $`3.6\times10^{-9}`$ and at most $`6.4\times10^{-9}`$ ($`D = 24.5`$) at $`22`$–$`29.3`$ (quoted
  $`1.5`$–$`2\times10^{-9}`$), $`1.8\times10^{-8}`$ at $`30.1`$ and
  $`7\times10^{-8}`$ at $`34.2`$, and $`0.6`$–$`2.1\times10^{-6}`$ high from $`35`$ to $`40.3`$ (quoted $`1`$–$`3\times10^{-6}`$). How well the
  rational formula holds against these points is in [The formula](#the-formula).

### Is the period analytic in 1/D?

Yes, as far as the points up to $`D = 61`$ can tell. Each model below is a polynomial in one variable, fitted at 60 digits
on the cleaned double-double points $`10 \le D \le 45`$ (or $`\le 37.5`$) and compared with every point above, up to
$`D = 60.1`$, which it never saw. The table and the figure use the points up to $`D = 60.1`$, the data when they were made
(`hp/dd_fits_mp.py 10 45 --clean --dmax=60.1`, and `10 37.5 --clean --dmax=60.1` for the fourth column, on today's file).
The 94 points above $`60.1`$, solved to a looser tolerance, are tested separately: they lie within $`2.6\times10^{-15}`$ of the
degree-18 fit on $`D \le 60.1`$ (checks table above).

![How well each model predicts double-double points it never saw](hp/heldout.svg)

| Model | max residual on $`10 \le D \le 45`$ | worst error on $`45 < D \le 60.1`$ | fitted on $`D \le 37.5`$: worst error up to $`60.1`$ | limit |
|---|---|---|---|---|
| $`1/D`$, degree 16 | $`4.1\times10^{-15}`$ | $`5.2\times10^{-13}`$ | $`4.2\times10^{-11}`$ | 1.64389375 |
| $`1/D`$, degree 18 | $`4.0\times10^{-15}`$ | $`1.8\times10^{-12}`$ | $`1.5\times10^{-10}`$ | 1.64389377 |
| $`1/(D-2)`$, degree 18 | $`4.0\times10^{-15}`$ | $`2.8\times10^{-12}`$ | $`7.9\times10^{-11}`$ | 1.64389377 |
| $`(D-2)^{-1/2}`$, degree 20 | $`3.9\times10^{-15}`$ | $`2.0\times10^{-11}`$ | $`3.9\times10^{-10}`$ | 1.636 |
| $`D^{-1/2}`$, degree 20 | $`3.8\times10^{-15}`$ | $`1.6\times10^{-10}`$ | $`6.2\times10^{-9}`$ | 1.547 |

Every model fits its range to the level of the points, so the fit alone decides nothing; the unseen points do. Fitted
on $`D \le 45`$, the $`1/D`$ series predicts $`\Delta`$ fifteen units of $`D`$ ahead to $`5\times10^{-13}`$, about 100 times the
noise of the points, while the series in half-integer powers miss by 40 to 300 times more. Their limits also jump with
the degree (1.636 to 1.694 for $`(D-2)^{-1/2}`$ and 1.54 to 2.75 for $`D^{-1/2}`$ at degree 20–24 on all the points to $`D = 60`$), so they
give no usable $`\Delta_\infty`$. Adding one non-analytic term ($`D^{-1/2}`$, $`D^{-3/2}`$, $`\ln D / D`$, $`\ln D / D^2`$ or
$`e^{-D/4}`$) to the degree-16 series changes neither the prediction ($`1.6`$–$`5\times10^{-13}`$ at $`D = 60`$) nor the limit
(1.64389375), and the term's coefficient is not determined: $`D^{-1/2}`$ gets $`3\times10^{-5}`$ at degree 14 and
$`1\times10^{-7}`$ at degree 16 (`hp/dd_fits_mp.py 10 45 --clean --mixed --dmax=60.1`).

Fitted on all the cleaned points, power series in $`1/D`$ of degree 16–20 and in $`1/(D-2)`$ of degree 18–20 (the ones that fit
to the noise; $`1/(D-2)`$ at degree 16 leaves $`3\times10^{-13}`$) fit the points from $`D = 10`$ to $`61.03`$ to $`4.2\times10^{-15}`$
($`8.6\times10^{-15}`$ for $`1/D`$ at degree 16, $`1.5\times10^{-14}`$ for $`1/(D-2)`$ at degree 18) and give limits from $`1.643893744`$ to $`1.643893752`$ on
$`10 \le D \le 60.1`$ (`hp/dd_fits_mp.py 10 1e9 --clean --dmax=60.1`) and from $`1.643893745`$ to $`1.643893752`$ on all 823
cleaned points to $`D = 61.03`$, so $`\Delta_\infty = 1.64389375 \pm 0.00000001`$. The analytic form is not proven (the period is
selected by an effect the $`1/D`$ expansion does not see), but nothing in the points up to $`D = 61`$ asks for anything else.

Data range: the cloud runs reach $`D = 60.06`$ (and a few points at $`60.96`$–$`60.97`$); the climb above that was done
mostly on a second machine (runs in [`hp/data/local/`](hp/data/local/), coordinated through [`comms.txt`](comms.txt)),
to $`D = 61.03`$. It was stopped there because the steps in $`D`$ had shrunk to $`0.0015`$–$`0.005`$ and further points add
little to $`\Delta_\infty`$; why the steps shrink, and the solver options used (`hp/scan_hp.py --junk_hold`,
`--junk_trend`, `--corrector dnp`), are in [`solver/notes.md`](solver/notes.md).

These limits are the $`\Delta_\infty`$ quoted in [The formula](#the-formula) and [Values](#values).

## Large D: what the 1/D expansion can and cannot see

The double-double points fix $`\Delta_\infty = 1.64389375`$ numerically. This section asks where the number comes from
in the equations. The answer so far: **the period is not determined at any finite order of the $`1/D`$ expansion of
the field equations** (within the natural ansatz of Proposition 1: integer powers of $`1/n`$ with the measured scalings).
For every choice of the period and of a free time profile, the expansion can be carried to all
orders (Proposition 1). The leading-order problem is solved exactly by hypergeometric functions, and those are never
analytic at the light cone (Proposition 2). So the period is selected by effects smaller than every power of $`1/D`$.
Finding that selection rule is open; the last part of the section lists what is known about it.

The results, in order:

- [Proposition 1](#proposition-1-the-1n-expansion-never-fixes-the-period): the $`1/n`$ expansion can be carried to all
  orders for any period and any time profile, so it never selects the period.
- [Proposition 2](#proposition-2-the-leading-order-problem-solved-exactly): without gravity the problem is solved by
  hypergeometric functions, and the solution is never analytic at the light cone; the obstruction is a coefficient
  $`C(n,s)`$ in closed form.
- [Proposition 3](#proposition-3-what-gravity-does-to-the-non-analytic-piece): gravity rescales the non-analytic piece by an
  $`n`$-independent factor that never vanishes (about 40 over the tested interval), so inside $`0 < x < 1`$ it cannot cancel it
  on the way to the light cone.
- [Proposition 4](#proposition-4-late-orders-of-the-leading-order-series): a derivation (not yet a full proof) that the
  divergence of the $`1/n`$ series without gravity is that piece; its singulant, exponent and constant are derived and
  checked numerically, which identifies $`C(n,s)`$ as the Stokes multiplier of a mode born at $`x = \infty`$.
- [The inner problem at large x](#the-inner-problem-at-large-x): near $`x \sim \sqrt n`$ the equations lose $`n`$ altogether;
  without gravity they reduce to Kummer's equation, whose Stokes constant gives $`C(n,s)`$ asymptotically (times a power of $`n`$
  and a phase). With gravity, the working hypothesis is that this multiplier vanishing is the selection rule; computing
  it is in progress (one and two harmonics so far, no selection of $`\omega_\infty`$ yet).

What is new here, as far as a literature search shows ([Relation to published work](#relation-to-published-work)):
the leading-order freedom (any periodic profile and any period solve the equations at leading order) is known in
substance (Ecker, Ecker & Grumiller 2026), and the free wave equation in similarity coordinates has been studied
in 3+1 (Donninger 2008). The all-orders statement (Proposition 1), the exact light-cone coefficient in every dimension
(Proposition 2) and the Stokes reading (Propositions 3–4 and the inner problem) were not found in print.

Everything below uses the scaling measured on the double-double solutions in
[How the solution scales with D](#how-the-solution-scales-with-d), and the results are checked against those solutions
where possible.

Notation: $`n = D - 2`$, $`\varepsilon = 1/n`$, $`\omega = 2\pi/\Delta`$. The solution depends on two variables: $`x`$, the
self-similar radius (radius over time to the singularity, normalized so that the past light cone of the singularity is at
$`x = 1`$; the code and the solver sections call that surface the "horizon"), and $`\tau = -\ln(-t)`$, the logarithm of the
time to the singularity, or the phase $`\theta = \omega\tau`$ of the echoing (period $`2\pi`$). $`p`$ and $`m`$ are the
derivatives of the scalar field along the two radial null directions (scaled), $`a`$ is the radial metric function
($`b = a^2`$), and $`c`$ is the lapse-type function that sets the light cones (the past light cone is where $`x = c`$). The
equations are the ones at the top of [`hp/src/model.rs`](hp/src/model.rs), with $`G = (n-1)\,c\,(b-1)`$:

```math
x c_x = (n-1)\,c\,(b-1), \qquad
x b_x = b\,(p^2+m^2) - (n-1)\,b\,(b-1),
```
```math
x(x-c)\,p_x = -\omega x\,p_\theta + G p - c p + \tfrac{n}{2}\,c\,(p+m), \qquad
x(x+c)\,m_x = -\omega x\,m_\theta - G m + c m - \tfrac{n}{2}\,c\,(p+m),
```

on $`0 \le x \le 1`$, with only odd harmonics of $`\theta`$ in $`p, m`$ and only even ones in $`b, c`$ ("odd" and "even" for
functions of $`\theta`$ refer to harmonics below; for polynomials in $`x`$ they refer to parity in $`x`$), regular at the centre
$`x = 0`$, analytic at the past light cone $`x = 1`$, and $`c(1) = 1`$ (the gauge that puts the light cone at $`x = 1`$).

### How the solution scales with D

Measured on the double-double solutions at $`D = 12`$–$`60`$ (`hp/largeD_scaling.py`, `hp/largeD_F.py`):

- $`p \approx x\,F(\theta)/\sqrt n`$ near the centre, and $`p + m`$ is smaller than $`p`$ by another factor of order $`1/n`$;
- $`c = 1 - O(1/n)`$, and $`b - 1 = (p^2+m^2)/(n-1)`$ to about 3%;
- near the light cone $`c`$ stays within $`10^{-3}`$ of 1, so $`x - c \approx x - 1`$ there;
- the centre series has the exact symmetry $`m(x) = p(-x)`$.

The profile $`F = \sqrt n\, \partial_x p\,(x = 0)`$ converges as $`D`$ grows:

| $`D`$ | first harmonic of $`F`$ | third / first | fifth / first | $`n\,(1 - \langle c(0)\rangle)`$ | $`\langle F^2\rangle`$ |
|---|---|---|---|---|---|
| 20 | 3.480 | 0.130 | 0.024 | 3.021 | 6.162 |
| 30 | 3.212 | 0.105 | 0.016 | 3.201 | 5.218 |
| 40 | 3.086 | 0.093 | 0.012 | 3.304 | 4.804 |
| 50 | 3.011 | 0.085 | 0.0097 | 3.372 | 4.568 |
| 60 | 2.962 | 0.080 | 0.0085 | 3.421 | 4.416 |
| $`\infty`$ (extrapolated) | $`\approx 2.72`$ | | | $`\approx 3.65`$ | $`\approx 3.7`$ |

The last two columns approach the same limit, as Proposition 1 predicts ($`c = 1 - \varepsilon F^2 (1-x^2) + \ldots`$).

### Proposition 1: the 1/n expansion never fixes the period

Write $`p = \varepsilon^{1/2} P`$, $`m = \varepsilon^{1/2} M`$, $`S = P + M`$, $`\Delta_P = P - M`$, and expand
$`S = \sum_{k \ge 1} \varepsilon^k S_k`$, $`\Delta_P = \sum_{k \ge 0} \varepsilon^k D_k`$,
$`c = 1 - \sum_{k\ge1} \varepsilon^k G_k`$, $`b = 1 + \sum_{k \ge 2} \varepsilon^k B_k`$,
$`\omega = \sum_k \varepsilon^k \omega_k`$. The sum and difference of the $`p`$ and $`m`$ equations become

```math
x\,c\,\partial_x \Delta_P - c\,\Delta_P = x^2 \partial_x S + \omega x\,\partial_\theta S - \tfrac{1-\varepsilon}{\varepsilon}\, c\,(b-1)\,\Delta_P ,
\qquad
S = \varepsilon\,\frac{x^2 \partial_x \Delta_P - x c\,\partial_x S + \omega x\,\partial_\theta \Delta_P}{c\,(1-\varepsilon)\,b} .
```

The ansatz (integer powers of $`\varepsilon`$, $`p, m \propto \varepsilon^{1/2}`$, $`c - 1 = O(\varepsilon)`$, $`b - 1 = O(\varepsilon^2)`$) is the
scaling measured in the table above; Proposition 1 and its consequence hold within it.

**Proposition 1.** Let $`F(\theta)`$ be any smooth $`2\pi`$-periodic function with odd harmonics only, $`\omega_0 \ne 0`$ (the
hypothesis only excludes the static case; the proof does not use it),
$`\omega_1, \omega_2, \ldots`$ any numbers and $`K_1(\theta), K_2(\theta), \ldots`$ any smooth odd-harmonic functions.
Then there is exactly one formal solution with $`D_0 = 2xF`$ in which, at every order,
$`S_k`$ is an even polynomial in $`x`$ starting at $`x^2`$, $`D_k - xK_k`$ is an odd polynomial starting at $`x^3`$, and $`G_k`$
and $`B_{k+1}`$ are even polynomials ($`B`$ starting at $`x^2`$) with $`G_k(1) = 0`$. Every order is regular at the centre and
analytic at the light cone, and $`m(x) = p(-x)`$ holds order by order.

<details>
<summary>Proof of Proposition 1</summary>

*Proof.* By induction on $`k`$, with the structure above as the hypothesis for all orders below $`k`$. Write $`E`$ for even
polynomials in $`x`$ with smooth periodic coefficients.

1. $`B_{k+1}`$: The $`b`$ equation at order $`k+1`$ gives $`B_{k+1}`$ algebraically (its coefficient is $`(n-1)b \approx 1/\varepsilon`$)
   from $`x\partial_x B_k`$, $`(1+B)\,(P^2+M^2)`$ and $`(1-\varepsilon)(1+B)B`$ at lower orders. Since $`P^2 + M^2 = (S^2 + \Delta_P^2)/2`$
   with $`S \in x^2E`$ and $`\Delta_P \in xE`$, every term is in $`x^2E`$.
2. $`G_k`$: The $`c`$ equation at order $`k+1`$ reads $`-x\,\partial_x G_k = [(1-\varepsilon)\,c\,(b-1)]_{k+1} \in x^2 E`$.
   Dividing by $`x`$ leaves only odd powers $`x^{2j-1}`$, $`j \ge 1`$, so $`G_k = -\sum_j r_{2j} x^{2j}/(2j) + \text{const}`$ is an even
   polynomial with no $`\log x`$. The constant is fixed by $`G_k(1) = 0`$.
3. $`S_k`$: The numerator $`x^2\partial_x\Delta_P - xc\,\partial_x S + \omega x\,\partial_\theta\Delta_P`$ maps
   $`\Delta_P \in xE`$, $`S \in x^2E`$ into $`x^2E`$. The denominator is $`1 + O(\varepsilon)`$ with coefficients in $`E`$, so its inverse is
   a power series in $`\varepsilon`$ with coefficients in $`E`$. Hence $`S_k \in x^2 E`$.
4. $`D_k`$: The sum equation at order $`k`$ is $`x\,\partial_x D_k - D_k = R_k`$, where $`R_k`$ collects
   $`x^2\partial_x S_k + \omega x\,\partial_\theta S_k \in x^3E`$, terms $`c\,(b-1)\,\Delta_P \in x^3E`$, and terms
   $`G_i\,(x\,\partial_x D_j - D_j)`$. The last are in $`x^3E`$ because $`x\partial_x - 1`$ annihilates $`xK_j`$. So $`R_k`$ is odd and starts
   at $`x^3`$, and $`D_k = x\int R_k\,x^{-2}\,dx = \sum_{j \ge 1} R_{k,2j+1}\, x^{2j+1}/(2j) + xK_k`$. The only obstruction,
   a term $`x^1`$ in $`R_k`$ that would produce $`x \log x`$ (not allowed at the centre), never occurs.
5. Symmetry: $`S`$ even and $`\Delta_P`$ odd in $`x`$ is the same as $`p(-x) = m(x)`$.
6. Harmonic content: if $`P, M`$ (so $`S, \Delta_P`$) contain only odd harmonics of $`\theta`$ and $`B, G`$ only even ones at the
   lower orders, every step above preserves it (products of two odd-harmonic factors are even-harmonic, of an odd and an
   even one odd-harmonic, and $`\partial_\theta`$ keeps the harmonic), so the next order has it too.

Polynomials are analytic everywhere, so the light-cone condition is automatic, and $`c(1) = 1`$ holds by construction.
Nothing in the construction constrains $`F`$, $`\omega_0`$ or the $`\omega_k`$. $`\square`$

</details>

The first orders, from `solver/largeD_expand.py` (sympy, run to fourth order):

```math
P_0 = xF, \quad B_2 = 2x^2F^2, \quad G_1 = F^2\,(1-x^2), \quad S_1 = 2x^2\,(F + \omega_0 F'),
\quad D_1 = x^3\,(\omega_0^2 F'' + 3\omega_0 F' + 2F - 2F^3) + xK_1 .
```

*Check.* `solver/largeD_formal_check.py` substitutes the expansion (through $`S_3`$, $`G_3`$, $`B_4`$) into the exact equations,
with $`F`$ taken from the $`D = 60`$ solution. The relative residual of every equation falls like $`n^{-3}`$ for
$`n = 50, \ldots, 800`$, both for $`\omega_0 = 2\pi/1.64389`$ and for a wrong $`\omega_0 = 2\pi/1.4`$. The expansion is right
to that order, and it does not care what the period is.

**Consequence.** If $`\Delta(D)`$ is analytic in $`1/D`$, as the double-double points support, it has a $`1/D`$ expansion, but its
coefficients cannot be computed order by order from the field equations (within the ansatz above). The selection works
through terms smaller than any power of $`1/n`$.

*Comparison.* Ecker, Ecker & Grumiller (PRL 136, 191401 (2026)) derive a next-to-leading-order condition on the period
(their Eq. (20)). It turns out not to select $`\Delta`$, and Proposition 1 stands; see
[Relation to published work](#relation-to-ecker-ecker--grumiller).

### Proposition 2: the leading-order problem, solved exactly

At leading order the equations are linear (the scalar wave equation on flat space in self-similar form, $`c = b = 1`$).
For one harmonic $`e^{i\mu\theta}`$, put $`s = i\mu\omega`$.

**Proposition 2.** The solution regular at the centre is, up to a constant,

```math
p - m = x\;{}_2F_1\!\left(\tfrac{1+s}{2},\,\tfrac{2+s}{2};\,\tfrac{n+1}{2};\,x^2\right),
\qquad
p + m = \frac{x(1-x^2)\,\partial_x(p-m) - (1+sx^2)\,(p-m)}{x\,(s-n+1)} .
```

Near the light cone it contains the non-analytic branch $`(1-x^2)^{\,n/2-1-s}`$ with the coefficient

```math
C(n, s) = \frac{\Gamma\!\left(\frac{n+1}{2}\right)\,\Gamma\!\left(s + 1 - \frac n2\right)}{\Gamma\!\left(\frac{1+s}{2}\right)\Gamma\!\left(\frac{2+s}{2}\right)} ,
```

which is finite and non-zero for every real $`\omega \ne 0`$, odd $`\mu`$ and $`n \ge 2`$. So no time-periodic self-similar
solution of the linear problem is analytic at the light cone, in any dimension $`D \ge 4`$.

<details>
<summary>Proof of Proposition 2</summary>

*Proof.* With $`c = b = 1`$ the difference equation gives $`(n - 1)\,(p+m) = x^2\,\partial_x(p-m) - x\,\partial_x(p+m) + s x\,(p-m)`$,
and the sum gives $`x\,\partial_x(p-m) - (p-m) = x^2\,\partial_x(p+m) + sx\,(p+m)`$. Eliminating $`p+m`$ gives

```math
x^2(1-x^2)\,y'' + x\,\big(n - 2 - 2(s+1)x^2\big)\,y' - \big(n - 2 + s(s+1)x^2\big)\,y = 0 , \qquad y = p - m ,
```

and with $`y = x\,w(z)`$, $`z = x^2`$, this is Gauss's equation $`z(1-z)w'' + [\kappa - (\alpha+\beta+1)z]\,w' - \alpha\beta\,w = 0`$ with
$`\alpha = (1+s)/2`$, $`\beta = (2+s)/2`$, $`\kappa = (n+1)/2`$. Its exponents are $`0, 1-\kappa`$ at $`z = 0`$ (so $`y \sim x`$ or $`x^{2-n}`$; for $`n > 2`$
regularity, and for $`n = 2`$ parity ($`y = p - m`$ is odd in $`x`$ because $`m(x) = p(-x)`$, while $`x^{0}`$ is even), picks
$`{}_2F_1(\alpha,\beta;\kappa;z)`$) and $`0,\ \kappa-\alpha-\beta = n/2 - 1 - s`$ at $`z = 1`$. The connection formula (Abramowitz & Stegun 15.3.6)

```math
{}_2F_1(\alpha,\beta;\kappa;z) = \frac{\Gamma(\kappa)\Gamma(\kappa-\alpha-\beta)}{\Gamma(\kappa-\alpha)\Gamma(\kappa-\beta)}\,{}_2F_1(\alpha,\beta;\alpha+\beta-\kappa+1;1-z)
+ (1-z)^{\kappa-\alpha-\beta}\,\frac{\Gamma(\kappa)\Gamma(\alpha+\beta-\kappa)}{\Gamma(\alpha)\Gamma(\beta)}\,{}_2F_1(\kappa-\alpha,\kappa-\beta;\kappa-\alpha-\beta+1;1-z)
```

holds because $`\kappa - \alpha - \beta`$ is not an integer ($`s`$ is imaginary and non-zero). The second coefficient is $`C(n,s)`$. The
Gamma function has no zeros, and its arguments $`\alpha`$, $`\beta`$, $`\alpha+\beta-\kappa`$ are not poles because their imaginary parts
are $`\mu\omega/2`$, $`\mu\omega/2`$, $`\mu\omega \ne 0`$. $`\square`$

</details>

*Check.* At 40 digits (mpmath), the formula satisfies the original first-order equations to $`10^{-40}`$ for $`n = 18`$–$`1000`$. For the
fundamental at $`\omega = \omega_\infty`$, $`\lvert C\rvert = 0.0121,\ 0.0114,\ 0.0124,\ 0.0192,\ 0.0405`$ at $`n = 18, 38, 58, 200, 1000`$.
It grows like $`\sqrt n`$ ($`\lvert C\rvert/\sqrt n = 0.00128`$ at $`n = 1000`$, tending to
$`2\pi\,2^{-1/2}e^{-\pi\omega}/\lvert\Gamma(\alpha)\Gamma(\beta)\rvert = 0.00126`$), small only through the factor of order $`e^{-\pi\mu\omega/2}`$.

The free wave equation in similarity coordinates has been analysed in 3+1 dimensions before (Donninger,
arXiv:0809.5177); what Proposition 2 adds is the closed-form light-cone coefficient for every $`D`$ and harmonic.

**What this means.** At leading order the self-similar oscillation is a free scalar wave, and a free wave cannot be
both regular at the centre and analytic at the light cone. The non-analytic piece $`(1-x^2)^{n/2-1-s}`$ is smaller than any
power of $`1/n`$ at every fixed $`x < 1`$, which is why Proposition 1 never sees it. Gravity, through the $`O(1/n)`$ couplings,
has to cancel its coefficient harmonic by harmonic. That cancellation is what must select $`\Delta`$ and the profile $`F`$.
The next two subsections show that gravity cannot do it while the wave travels to the light cone inside $`0 < x < 1`$,
and point to where it has to happen instead.

### Proposition 3: what gravity does to the non-analytic piece

The non-analytic piece is a fast WKB mode: in the exact solution of Proposition 2 it is
$`x^{2-n}(1-x^2)^{n/2-1-s}`$, which varies on the scale $`1/n`$. Linearize the field equations around the formal solution
of Proposition 1 and look for perturbations of that kind.

**Proposition 3.** Let $`F(\theta)`$ and $`\omega`$ be given and $`c = 1 - \varepsilon F^2(1-x^2) + O(\varepsilon^2)`$, as in
Proposition 1. Put

```math
\delta p = e^{-n\Phi(x)}\,\big(A(x,\theta) + O(1/n)\big), \qquad
\delta m = e^{-n\Phi(x)}\,\Big(\frac{1-x}{1+x}\,A(x,\theta) + O(1/n)\Big), \qquad
\Phi(x) = \ln x - \tfrac12 \ln(1-x^2) .
```

This solves the linearized equations on $`0 < x < 1`$ to the order shown if and only if

```math
x(1-x^2)\,\partial_x A - 2\omega x^2\,\partial_\theta A = \big(1 + x - 2x^2F(\theta)^2\big)\,A ,
```

that is, along the characteristics $`\theta = \theta_0 + \omega\ln(1-x^2)`$,

```math
A = A_0(\theta_0)\,\frac{x}{1-x}\,\exp\!\Big(-\int_0^x \frac{2t\,F\big(\theta_0 + \omega\ln(1-t^2)\big)^2}{1-t^2}\,dt\Big) .
```

<details>
<summary>Proof of Proposition 3</summary>

*Proof.* Substitute into the linearized equations. The terms of order $`n`$ come from $`x(x\mp c)\,\partial_x`$ acting on
$`e^{-n\Phi}`$ and from $`\tfrac n2 c\,(\delta p+\delta m)`$. With $`B`$ the amplitude of $`\delta m`$ they read

```math
-x(x-1)\,\Phi'\,A = \tfrac12 (A+B), \qquad -x(x+1)\,\Phi'\,B = -\tfrac12 (A+B) .
```

A non-zero solution needs $`x(1-x^2)\,\Phi' = 1`$, which gives $`\Phi`$, and then $`B = A(1-x)/(1+x)`$. At order 1 the
corrections $`(a_1, b_1)`$ satisfy a linear system with the same singular matrix,

```math
\frac12\begin{pmatrix} \dfrac{1-x}{1+x} & -1 \\[4pt] 1 & -\dfrac{1+x}{1-x} \end{pmatrix}
\begin{pmatrix} a_1 \\ b_1 \end{pmatrix} = \begin{pmatrix} R_p \\ R_m \end{pmatrix},
```

whose left null vector is $`(1+x,\ -(1-x))`$. The right-hand sides collect $`x(x\mp1)\,\partial_x`$ of $`A`$ and $`B`$, the
$`\omega x\,\partial_\theta`$ terms, the $`\mp c`$ terms, and the gravity terms from $`c = 1 - \varepsilon G_1`$ inside
$`x(x\mp c)\,n\Phi'`$ and $`\tfrac n2 c`$. The perturbations of $`b`$ and $`c`$, and $`G\,\delta p`$, are smaller by $`1/n`$ and drop
out. Solvability, $`(1+x)R_p - (1-x)R_m = 0`$, multiplied by $`(1+x)/2`$, is the transport equation; with
$`G_1 = F^2(1-x^2)`$ the gravity terms add up to $`-2x^2F^2A`$. A first-order equation integrates along its characteristics,
and $`\int (1+x)/(x(1-x^2))\,dx = \ln\big(x/(1-x)\big)`$ gives the prefactor. $`\square`$

</details>

Without gravity, a harmonic $`A = \alpha(x)e^{i\mu\theta}`$ gives $`\alpha = x(1-x)^{-1}(1-x^2)^{-s}`$ and
$`\delta p - \delta m = 2x^{2-n}(1-x^2)^{n/2-1-s}`$: exactly the second solution of Proposition 2, with its exponents at the
centre and at the light cone.

*Check.* `solver/largeD_fast_transport.py` integrates the linearized equations ($`b = 1`$, $`c = 1 - F^2(1-x^2)/n`$,
$`F = 2.72\sin\theta + 0.22\sin3\theta`$, $`\omega = \omega_\infty`$, 64 points in $`\theta`$) from $`x = 0.9`$ to $`0.72`$, starting
from the WKB data, and compares the result with the transported $`A`$:

| $`n`$ | 100 | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|---|
| relative error, no gravity ($`F = 0`$ in $`c`$) | $`5.6\times10^{-2}`$ | $`2.9\times10^{-2}`$ | $`1.5\times10^{-2}`$ | $`7.5\times10^{-3}`$ | $`3.8\times10^{-3}`$ |
| relative error, with gravity | $`0.57`$ | $`0.26`$ | $`0.13`$ | $`0.062`$ | $`0.031`$ |
| relative error, full background ($`b`$, $`G`$, $`p`$ of Proposition 1, with $`\delta b`$, $`\delta c`$) | $`0.62`$ | $`0.28`$ | $`0.135`$ | $`0.066`$ | $`0.032`$ |

All fall like $`1/n`$. Over this interval gravity multiplies the fast mode by about 40. The last row is
`solver/largeD_fast_transport_full.py`. `solver/largeD_transport_sympy.py` repeats the derivation symbolically from the
exact equations with $`\delta b`$ and $`\delta c`$ kept: they come out at order $`\varepsilon^{3/2}e^{-n\Phi}`$, the half-integer orders vanish,
the leading order gives $`B = A(1-x)/(1+x)`$, $`a_1, b_1`$ drop out of the projection, and the projected order-1 equation is
$`-2/(1+x)`$ times the transport equation.

**Consequences.**

1. Gravity acts on the non-analytic piece at order 1, not at order $`1/n`$, through a factor that contains every power of
   $`F^2`$. Near the light cone the characteristic winds through all phases, $`F^2`$ averages, and the factor behaves like
   $`(1-x^2)^{\langle F^2\rangle}`$: the light-cone exponent becomes $`n/2 - 1 - s + \langle F^2\rangle`$. The exact local analysis at
   $`x = 1`$ agrees: there $`x - c = -(1-x)\,(1 - c_x(1))`$ with $`c_x(1) = G(1)`$ (because $`c(1) = 1`$), $`m`$ enters only one power
   later, and periodicity in $`\theta`$ gives the exponent $`(n/2 - 1 - s + \langle G(1)\rangle)/(1 - \langle G(1)\rangle)`$. On the double-double
   solutions (`hp/largeD_exponent.py`; $`\lvert c_x(1) - G(1)\rvert \le 1.7\times10^{-14}`$):

   | $`D`$ | 20 | 30 | 40 | 50 | 60 | $`\infty`$ (fit in $`1/n`$) |
   |---|---|---|---|---|---|---|
   | $`n\langle G(1)\rangle/2`$ | 2.038 | 2.435 | 2.678 | 2.844 | 2.964 | 3.59–3.70 |
   | $`\langle F^2\rangle`$ | 6.162 | 5.218 | 4.804 | 4.568 | 4.416 | 3.70–3.73 |
   | exponent shift, real part | 2.63 | 2.95 | 3.12 | 3.23 | 3.30 | |

   The two agree only in the limit (both extrapolate to about 3.70); at finite $`D`$ the $`1/n`$ corrections are large. With
   a shift of about 3.7 at $`D = \infty`$, an expansion in the amplitude of the wave is not a route to the selection.
2. The factor never vanishes. Gravity rescales the fast mode along each characteristic, but it cannot cancel it on the
   way to the light cone; the light-cone coefficient is zero only where the mode is never created.
3. Every characteristic $`\theta = \theta_0 + \omega\ln(1-x^2)`$ reaches the centre, and there the fast mode is the singular
   branch $`x^{2-n}`$ (the second solution of Proposition 2). A solution regular at $`x = 0`$ therefore has $`A_0 = 0`$ on every
   characteristic: at any order of this WKB expansion it carries no fast mode on $`0 < x < 1`$. (A size argument,
   $`e^{-n\Phi}`$ being exponentially large for $`x < 1/\sqrt2`$, would not exclude an exponentially small $`A_0`$; regularity
   does.) The non-analytic piece comes from a Stokes phenomenon, which the next subsection locates.

### Proposition 4: late orders of the leading-order series

A divergent series records the exponentially small terms it switches on in its late orders (Dingle, *Asymptotic
Expansions: Their Derivation and Interpretation*, 1973): if the switched-on
term is $`n^{-g}e^{-n\chi(x)}`$, the coefficients behave like $`y_k \sim K\,Q(x)\,\Gamma(k-g)/\chi(x)^{k-g}`$. At leading order this can
be checked exactly, because the $`1/n`$ expansion of $`x\,{}_2F_1`$ from Proposition 2 is explicit. `solver/largeD_linear_late.py`
computes it to order 122 at 90 digits and reads $`\chi`$ and $`g`$ off consecutive ratios of $`y_k`$:

| $`x`$ | $`\chi`$ measured (orders 100–120) | $`\Phi(x) - i\pi/2`$ | $`g`$ measured |
|---|---|---|---|
| 0.6 | $`-0.27 - 1.57i`$ to $`-0.30 - 1.58i`$ | $`-0.288 - 1.571i`$ | scattered |
| $`1/\sqrt2`$ | $`0.0069 - 1.5703i`$ | $`-1.571i`$ | $`-0.52 - 4.50i`$ |
| 0.8 | $`0.291 - 1.5704i`$ | $`0.288 - 1.571i`$ | $`-0.55 - 4.27i`$ |
| 0.9 | $`0.728 - 1.5694i`$ | $`0.725 - 1.571i`$ | $`-0.47 - 4.26i`$ |

So the singulant is $`\chi(x) = \Phi(x) - i\pi/2`$ to the accuracy of 120 orders. The exponent read off two ratios carries
a $`1/k`$ error; with the constant below fixed, the data are consistent with $`g = -(s + \tfrac12) = -0.5 - 3.82i`$ exactly, the
power $`n^{s+1/2}`$ that $`C(n,s)`$ carries for large $`n`$. Below $`x = 1/\sqrt2`$ the readings scatter. The next singulant,
$`\Phi + i\pi/2`$, has the same modulus but a small weight ($`8\times10^{-5}`$ at $`x = 0.5`$, $`e^{-\pi\omega} \approx 6\times10^{-6}`$ at $`1/\sqrt2`$,
$`e^{-2\pi\omega}`$ as $`x \to 1`$); the two-ratio estimator amplifies it by about $`k\,\lvert e^{i\varphi}-1\rvert^2`$, and at $`x = 0.6`$, $`k \ge 110`$ the
direct series also loses 70–84 of its 90 digits to cancellation. The prefactor and the constant are predicted too (next
subsection):

```math
y_k(x) \sim K\,x^2(1-x^2)^{-1-s}\,\frac{\Gamma(k+s+\frac12)}{\chi(x)^{\,k+s+\frac12}}, \qquad
K = e^{-i\pi(1+s)}\,\frac{2^{-s-1/2}}{\Gamma(\alpha)\Gamma(\beta)} .
```

`solver/largeD_dingle_K.py` divides the measured $`y_k`$ (orders 92–122) by this form and extrapolates in $`1/k`$:

| $`x`$ | 0.6 | $`1/\sqrt2`$ | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|
| $`K_{\text{measured}}/K`$ | $`1.08 + 0.06i`$ | $`1.01 - 0.10i`$ | $`0.9936 + 0.0022i`$ | $`1.0006 - 0.0002i`$ | $`1.0010 - 0.0015i`$ |

(the first two columns carry the estimator's interference). The singulant and the exponent also follow from a
derivation, not yet a full proof:

**Proposition 4 (derivation; late orders of the leading-order series).** Let $`y(\varepsilon) = x\,{}_2F_1(\alpha, \beta; \kappa; z)`$, $`z = x^2`$,
$`\kappa = (1 + 1/\varepsilon)/2`$, with $`y \sim \sum_k y_k\varepsilon^k`$. Then, exactly,

```math
y_k = 2x\,(-1)^{k-1}\sum_{J \ge 0} R_J\,(2J+1)^{k-1}, \qquad
R_J = \frac{(-1)^J(\alpha)_{J+1}(\beta)_{J+1}}{J!\,(J+1)!}\,z^{J+1}\,{}_2F_1(\alpha+J+1, \beta+J+1; J+2; z),
```

the sum converging for $`x < 1/\sqrt2`$ and continued analytically above, and as $`k \to \infty`$

```math
y_k \approx \sum_{\ell} K_\ell(x)\,\frac{\Gamma\big(k + s + \frac12\big)}{\big(\Phi(x) + i\pi(\ell + \frac12)\big)^{k+s+\frac12}} ,
```

with the $`\ell = -1`$ term, $`\chi = \Phi - i\pi/2`$, dominant for an echo with $`\omega > 0`$.

<details>
<summary>Derivation of Proposition 4, and the gaps a full proof must close</summary>

*Proof sketch.* As a function of $`\kappa`$, $`{}_2F_1`$ is meromorphic with simple poles at $`\kappa = -J`$ and residues $`R_J`$; with
$`1/(\kappa + J) = 2\varepsilon/\big(1 + (2J+1)\varepsilon\big)`$ the pole expansion gives the first formula term by term. For large $`J`$,
$`R_J \approx (-1)^J x^2(1-x^2)^{-\alpha-\beta}\,e^{2\Phi J}J^{s+1/2}/\big(\Gamma(\alpha)\Gamma(\beta)\big)`$. Poisson summation over $`J`$ turns
$`\sum_J (-1)^J e^{2\Phi J}J^{s+1/2}(2J+1)^{k-1}`$ into $`\sum_\ell \int_0^\infty e^{(2\Phi + i\pi(2\ell+1))J}J^{k+s-1/2}\,dJ`$ (to leading order), and each
integral is a Gamma function: singulants $`\Phi + i\pi(\ell+\tfrac12)`$ and exponent $`g = -(s+\tfrac12)`$. The tower of $`\ell`$ is the tower
in $`1/\sin(\pi(\kappa-\alpha-\beta))`$ of $`C(n,s)`$, and the prefactor carries the same $`1/(\Gamma(\alpha)\Gamma(\beta))`$. (End of sketch.)

Gaps a full proof must close: the pole expansion in $`\kappa`$ needs a growth bound (and its constant term); exchanging the
$`J`$-sum with the expansion of $`1/(1 + (2J+1)\varepsilon)`$, which is not uniform in $`J`$; error bounds for the large-$`J`$ form
of $`R_J`$ inside the Poisson sum; the continuation to $`x > 1/\sqrt2`$, where the $`J`$-sum diverges (and where the $`K`$ checks
above are made); and fixing the branch of $`\chi^{k+s+1/2}`$ that makes $`\ell = -1`$ dominant.

</details>

*Check.* `solver/largeD_linear_poles.py`: the pole sum equals the direct series to $`10^{-49}`$–$`10^{-91}`$ at $`x = 0.3, 0.5, 0.6`$,
$`k \le 60`$. `solver/largeD_linear_poisson.py`: at $`x = 0.5`$, $`k = 30`$ the Poisson pieces add up to the exact $`y_k`$ within
$`10^{-19}`$, with relative sizes $`1`$ ($`\ell = -1`$), $`1.6\times10^{-4}`$ ($`\ell = 0`$), $`10^{-14}`$ ($`\ell = -2`$).

What this says:

- $`e^{-n\chi} = e^{in\pi/2}\,x^{-n}(1-x^2)^{n/2}`$ is the fast mode of Proposition 3, and $`x^2(1-x^2)^{-1-s}`$ is its transported
  amplitude at $`F = 0`$. The divergence of the $`1/n`$ series and the non-analytic piece are the same object.
- $`\chi`$ vanishes only at $`x = \infty`$ ($`\Phi(\infty) = i\pi/2`$ on this branch): the fast mode is born at the outer singular point
  of the equations, not in $`0 \le x \le 1`$.
- Its Stokes line ($`\mathrm{Im}\,\chi = 0`$, $`\mathrm{Re}\,\chi > 0`$) is the real axis $`x > 1`$ approached from above, where
  $`\chi = \ln\big(x/\sqrt{x^2-1}\big)`$. On $`0 < x < 1`$, $`\mathrm{Im}\,\chi = -\pi/2`$, so nothing is switched on there (as
  Consequence 3 requires). The Stokes line ends at the light cone, so a loop around $`x = 1`$ crosses it, and that is where
  the branch $`(1-x^2)^{n/2-1-s}`$ and its coefficient $`C(n,s)`$, the Stokes multiplier, come from. ($`\mathrm{Im}\,\chi = 0`$ also on the
  imaginary axis, but there $`\mathrm{Re}\,\chi < 0`$, so it does not switch this exponential.)
- Directly: the $`1/n`$ series truncated near its smallest term reproduces the exact $`x\,{}_2F_1`$ inside $`0 < x < 1`$ to far
  below the size the singular branch would have there (`solver/largeD_linear_truncation.py`): at $`x = 0.9`$, $`n = 60`$ the
  smallest difference is $`4\times10^{-40}`$ against $`\lvert C\rvert\times\text{branch} = 7\times10^{-21}`$; at $`x = 0.8`$, $`n = 40`$, $`10^{-23}`$ against
  $`2\times10^{-7}`$. So the regular solution carries no fast mode inside the light cone.

With gravity this suggests a **working hypothesis** (not proved): if the transport of Proposition 3 continues to $`x > 1`$,
the light-cone coefficient is the multiplier times a factor that does not vanish, and **the selection rule at
$`D = \infty`$ is that the Stokes multiplier of the mode born at $`x = \infty`$ vanishes.** Proposition 3 is proved only on
$`0 < x < 1`$, and its background (the polynomials of Proposition 1) is not uniform at $`x \sim \sqrt n`$; the inner problem
of the next subsection is where the hypothesis has to be made precise.

### The inner problem at large x

Near $`x = \infty`$ the large-$`n`$ expansion is not uniform: at $`x \sim \sqrt n`$, $`\varepsilon x^2F^2 = O(1)`$, so $`c - 1 = O(1)`$
(gravity is no longer weak); $`p \approx \varepsilon^{1/2}xF = O(1)`$; and $`e^{-n\Phi} \to e^{\mp in\pi/2}e^{-n/(2x^2)}`$, so fast and slow
modes are no longer separated. Put $`\xi = x/\sqrt n`$, $`y = p - m`$ (unscaled, as in Propositions 2 and 4; `Dd` in the scripts), $`\sigma = \sqrt n\,(p+m)`$, $`b - 1 = \eta/n`$, and
$`L = \xi^2\partial_\xi + \omega\xi\,\partial_\theta`$. The difference and sum of the $`p, m`$ equations and the $`b, c`$ equations give, at
leading order,

```math
L\,y = c\,\sigma, \qquad
L\,\sigma = c\,\xi\,\partial_\xi{y} + c\,\big(\tfrac12 y^2 - 1\big)\,{y}, \qquad
\xi\,\partial_\xi c = \tfrac12\,c\,{y}^2 , \qquad \eta = \tfrac12 y^2 .
```

No large parameter is left: this problem is the same for every $`n`$. (Derivation: in the difference equation the terms
of order $`\sqrt n`$ are $`x^2\partial_x{y}`$, $`\omega x\,\partial_\theta{y}`$ and $`n c\,(p+m)`$; in the sum, the order-1 terms are
$`x^2\partial_x(p+m)`$, $`x c\,\partial_x{y}`$, $`\omega x\,\partial_\theta(p+m)`$ and $`(G - c)\,y`$ with $`G = c\eta`$; the $`b`$ equation becomes
algebraic, $`\eta = p^2 + m^2`$.)

- **It resums the outer expansion.** Its formal solution regular at $`\xi = 0`$, $`y = \sum_j d_j\xi^{2j+1}`$,
  $`\sigma = \sum_j s_j\xi^{2j+2}`$, $`c = \sum_j g_j\xi^{2j}`$ with $`d_0 = 2F`$, $`g_0 = 1`$, has $`g_1 = F^2`$, $`s_0 = 2(F + \omega F')`$,
  $`d_1 = \omega^2F'' + 3\omega F' + 2F - 2F^3`$: the top-degree coefficients of $`G_1`$, $`S_1`$, $`D_1`$ of Proposition 1. `solver/largeD_inner.py`
  runs the recursion and finds $`d_k`$, $`-g_k`$, $`s_{k-1}`$ equal to the top-degree coefficients of the outer $`D_k`$, $`G_k`$, $`S_k`$
  to $`2\times10^{-13}`$ at orders 1–10 ($`F = 2.72\sin\theta`$, $`\omega = \omega_\infty`$).
- **Without gravity it is Kummer's equation, and $`C(n,s)`$ is asymptotically its Stokes constant.** With $`c = 1`$ and one harmonic, the regular
  solution is $`y = 2\xi\,{}_2F_0\big(\alpha, \beta; ; 2\xi^2\big)`$, i.e. $`d_j = 2^j(\alpha)_j(\beta)_j/j!\;d_0`$, and the second solution
  carries $`e^{-1/(2\xi^2)}`$, the outer fast mode $`e^{-n/(2x^2)}`$. The Stokes line of $`{}_2F_0(\cdot;2\xi^2)`$ is real $`\xi`$, i.e. real
  $`x > 1`$, and its Stokes constant is proportional to $`1/(\Gamma(\alpha)\Gamma(\beta))`$ (DLMF §2.11 and §13.7 for the
  Stokes phenomenon of $`{}_2F_0`$, i.e. Kummer's $`U`$). Matching gives the constant $`K`$ above and the large-$`n`$ form of
  $`C(n,s)`$ displayed after this list.
- **The selection rule as an inner problem (working hypothesis).** With gravity the rule becomes: the fast-type Stokes multipliers of the formal
  inner solution vanish, one complex condition per odd harmonic $`\mu`$. The unknowns are $`F`$'s odd harmonics (less one phase)
  and $`\omega`$, so the count balances, and the problem contains $`F`$ and $`\omega`$ only (the free functions $`K_k`$ of
  Proposition 1 multiply $`x`$, the lowest degree, so they never reach the top-degree terms the inner problem resums). This
  formulation is derived from the matching above but not yet proved.
- **What the inner series shows with gravity.** On the Stokes line the inner variable is real and the fast characteristic
  is $`\theta_0 + 2\omega\ln\xi`$, also real, so the multiplier needs $`F`$ on the real $`\theta`$ axis only. (In the outer labels the same
  characteristic carries a shift $`-i\pi\omega`$; that is branch bookkeeping between $`0 < x < 1`$ and $`x > 1`$, and it is the phase
  $`e^{-i\pi\Lambda}`$ in $`C`$ below.) Without gravity, harmonic 1 of the inner series grows at ratio/$`j \to 2`$ (the Borel singularity at
  $`\zeta = \tfrac12`$, dual to $`1/\xi^2`$). With gravity ($`F = 2.72\sin\theta`$, 60 orders) the growth is dominated by high harmonics,
  ratio/$`j`$ 6.8, 4.9, 4.2 at $`j`$ = 10, 30, 50. That growth is not a second singularity: it is the same fast singularity at
  $`\zeta = \tfrac12`$ (Borel variable $`u = \xi^2`$) carried by the higher harmonics, whose exponents have large imaginary parts
  ($`\lvert\mathrm{Im}\,g\rvert \approx 8.45`$ for the $`\nu = \pm3`$ modes of the $`M = 3`$ truncation; untruncated it would be
  $`3\omega_\infty = 11.47`$) and whose constants are $`10^4`$ times larger; at $`M = 3`$ the four modes at
  $`\zeta = \tfrac12`$ alone fit the series to $`2.5\times10^{-8}`$. Gravity couples the $`\pm\mu`$ fast modes, so the multiplier is
  a matrix over harmonics.

The large-$`n`$ form of the light-cone coefficient, from the matching in the second bullet ($`\Lambda = n/2 - 1 - s`$ is the
light-cone exponent of Proposition 2):

```math
C(n,s) = -2\pi i\,\frac{2^{-s-1/2}}{\Gamma(\alpha)\Gamma(\beta)}\,(n+1)^{s+1/2}\,e^{-i\pi\Lambda}\,\big(1 + O(s^2/n)\big), \qquad \Lambda = \tfrac n2 - 1 - s .
```

`solver/largeD_stokes_C.py`: the ratio of the two sides is $`1.00014`$ at $`n = 10^5`$ for $`\mu = 1`$ ($`1.0013`$ for $`\mu = 3`$), with
$`n\,(\text{ratio} - 1) \to 13.9 - 7.6i`$, of the size of $`\lvert s^2\rvert = 14.6`$. At $`D = 20`$–$`60`$ the ratio is still about $`2.1`$–$`1.27`$ for
$`\mu = 1`$: the asymptotics need $`\mu^2\omega^2 \ll n`$. So, asymptotically for large $`n`$, the light-cone coefficient of
Proposition 2 is the $`n`$-independent Stokes constant $`2^{-s-1/2}/(\Gamma(\alpha)\Gamma(\beta))`$ of a $`{}_2F_0`$ times $`(n+1)^{s+1/2}`$
and a phase.

**First computation with gravity** (harmonics $`\lvert\mu\rvert \le M`$), by two independent implementations that share no
code: *lateral solutions* (`solver/inner_stokes/`: the inner equations integrated from just above and just
below the Stokes line and differenced) and *Borel–Darboux* (`solver/inner_borel/`, ball arithmetic with python-flint: the
late orders of the inner series fitted with the predicted fast-mode exponents).

- *Both checked without gravity.* Lateral: the jump divided by $`2\pi i\,2^{-s-1/2}/(\Gamma(\alpha)\Gamma(\beta))`$ times the linear fast solution is
  $`1.000000000002 - 7\times10^{-12}i`$, the same at two matching radii (`solver/inner_stokes/lateral_np.py check`). Darboux:
  the constant $`d_0/(\Gamma(\alpha)\Gamma(\beta))`$ to $`6\times10^{-27}`$ from 200 orders (`solver/inner_borel/inner_stokes.py linear`;
  despite its name this script belongs to the Borel–Darboux implementation).
- *The fast mode with gravity, in the inner region* (derived): with $`\delta y = e^{-1/(2\xi^2)}a`$, the order-$`\xi^{-2}`$ terms cancel and
  $`\xi\,\partial_\xi a + 2\omega\,\partial_\theta a = 2F^2a`$, the large-$`x`$ limit of Proposition 3. So the multiplier is a function of the
  characteristic label, $`S(\theta_0) = \sum_\nu S_\nu e^{i\nu\theta_0}`$, multiplying $`e^{-1/(2\xi^2)}\xi^{2\langle F^2\rangle}e^{\Pi(\theta)}`$ with
  $`\Pi = \omega^{-1}\!\int(F^2 - \langle F^2\rangle)`$. The extra power $`\xi^{2\langle F^2\rangle}`$ is seen in the computed jumps. In Borel terms
  the singularity stays at $`\zeta = \tfrac12`$ and the Dingle exponents become $`g_\nu = -(\tfrac12 + i\nu\omega) + \langle F^2\rangle`$ (the convention
  of $`g`$ in Proposition 4, where $`g = -(s + \tfrac12)`$ without gravity; the late orders grow like $`\Gamma(j - g_\nu)`$), each mode spread over
  harmonics $`\nu + 2k`$ by $`e^{\Pi}`$; Darboux fits with these exponents leave residuals of $`5\times10^{-8}`$ ($`M = 1`$).
- *Harmonic truncation is not a controlled approximation* (derived and seen). The linear multipliers grow like
  $`S_\nu \propto F_\nu\,e^{\pi\lvert\nu\rvert\omega/2}`$ ($`1/\lvert\Gamma(\alpha)\Gamma(\beta)\rvert \approx 47`$ for $`\nu = 1`$, $`4.4\times10^6`$ for $`\nu = 3`$), so
  $`S(\theta_0)`$ involves $`F`$ continued to $`\mathrm{Im}\,\theta = \mp\pi\omega/2 \approx \mp6`$, and a harmonic generated only by the
  nonlinearity already carries a jump $`10^2`$–$`10^4`$ times that of harmonic 1.
- *One harmonic* ($`F = A\sin\theta`$): the multiplier has a genuine, isolated zero at $`A = 2.542148`$, $`\omega = 2.969777`$
  ($`\Delta = 2.11571`$). Both methods find it: lateral (`solver/inner_stokes/zero_M1.py`, the same at four matching radii) and
  Darboux (`solver/inner_borel/inner_zero.py`, Newton converging quadratically, $`\lvert S\rvert = 3\times10^{-14}`$ at 220 orders; it is the only
  clean zero for $`0.5 \le A \le 3`$, $`0.5 \le \omega/\omega_\infty \le 1.5`$). The amplitude is 6.5% below $`F_1 \approx 2.72`$; $`\omega`$ is 22% below
  $`\omega_\infty`$. Adding harmonic 3 (`solver/inner_stokes/zero_M3.py`) destroys that zero, and the zero found there is unphysical
  ($`\omega = 0.198\,\omega_\infty`$, $`F_3/F_1 = 0.65`$); at the one-harmonic zero the $`M = 3`$ constant for $`\nu = \pm1`$ is
  $`421 + 392i`$, nowhere near zero. ($`\Delta = 2.1157`$ is close to $`\Delta(20) = 2.1121`$, almost certainly a coincidence.) So no selection of $`\omega_\infty`$ follows yet; the next attempt needs rescaled
  unknowns $`F_\nu e^{\pi\lvert\nu\rvert\omega/2}`$, seven or more harmonics, and more precision along the integration arc.

**Without truncation: the amplitude expansion** (`solver/inner_amp/`).

*The inner problem is a Weber (parabolic-cylinder) problem* (derived). With $`t = 1/\xi`$ and the characteristic coordinate
$`\phi = \theta + \omega\ln t`$, $`L = -\partial_t`$ at fixed $`\phi`$, and eliminating $`\sigma`$ gives

```math
y'' + \Big(c^2t - \frac{c'}{c}\Big)\,y' + c^2\Big[\omega\,\partial_\phi y + \big(1 - \tfrac12 y^2\big)\,y\Big] = 0, \qquad t\,c' + \omega\,\partial_\phi c = -\tfrac12\,c\,y^2
```

($`' = \partial_t`$ at fixed $`\phi`$): along each characteristic, a softening anharmonic oscillator with friction $`c^2t`$. Without
gravity, $`w = t^{-s}y`$ obeys Hermite's equation $`w'' + tw' + (1+s)\,w = 0`$, and $`w = e^{-t^2/4}v`$ gives Weber's
$`v'' + (\tfrac12 + s - \tfrac14t^2)\,v = 0`$. By the duplication formula $`2^{-s-1/2}/(\Gamma(\alpha)\Gamma(\beta)) = 1/(\sqrt{2\pi}\,\Gamma(1+s))`$
(checked numerically for $`\nu = 1, 3`$), so the large-$`n`$ coefficient is

```math
C(n,s) \approx -\,i\,\frac{\sqrt{2\pi}}{\Gamma(1+s)}\,(n+1)^{s+1/2}\,e^{-i\pi\Lambda},
```

Weber's classical Stokes constant $`\sqrt{2\pi}/\Gamma(-\nu_W)`$ with $`\nu_W = -1-s`$.

- *Every harmonic exact.* Put $`F = A\sin\theta + \sum_{\nu\ge3} f_\nu(A)\,e^{i\nu\theta}`$ (+ c.c.). Harmonic $`\mu`$ first appears at
  order $`A^{\lvert\mu\rvert}`$, so keeping $`\lvert\mu\rvert \le K`$ at each order in $`A`$ is exact: no truncation error, unlike fixed $`M`$
  (`amp_series.py` matches `solver/largeD_inner.py` to $`2`$–$`4\times10^{-16}`$ at orders 0–3). The late orders of each harmonic are
  normalized by its exact linear form $`\Gamma(\alpha + j - \delta/2)\,\Gamma(\beta + j - \delta/2)`$, $`\delta = \langle F^2\rangle`$, and de-mixed by
  $`e^{-\Pi}`$; without that, harmonics $`\ge 5`$ never reach their asymptotic regime (corrections $`\sim(\nu\omega)^2/j`$), which is
  probably part of why fixed-$`M`$ truncation misbehaves.
- *The higher-harmonic conditions predict $`F_\infty`$'s shape* (computed; compared with data). Tuning $`f_\nu`$ order by order
  so that $`S_\nu = 0`$ for $`\lvert\nu\rvert \ge 3`$ (residuals $`\le 10^{-9}`$ of the linear sizes), at $`A = 2.72`$ and $`\omega = \omega_\infty`$,
  gives $`\lvert F_3/F_1\rvert = 0.055`$ and $`\lvert F_5/F_1\rvert = 0.0030`$ (the $`f_3`$ series converges: 0.0855, 0.0131, 0.0016). The double-double
  solutions at $`D = 20`$–$`60`$, extrapolated in $`1/n`$ (degree 1–3), give 0.050–0.058 and 0.0013–0.0046. This is the first number
  the D = ∞ selection rule has produced that can be checked against the solutions, and it agrees.
- *Leading structure and a prediction* (conjectured). The tuned ratios are $`f_{\nu+2}/f_\nu \approx q(\omega)\,A^2`$ with $`q`$ nearly the same
  for every $`\nu`$ (at $`\omega_\infty`$: $`q = 0.00850 + 0.00114i`$, $`0.00851 + 0.00085i`$, $`0.00852 + 0.00082i`$ for $`\nu = 1, 3, 5`$;
  roughly $`1/q \approx 8\omega^2 - 4i\omega - 1.3`$), so $`F \approx 2\,\mathrm{Re}\big[F_1e^{i\theta}/(1 - qA^2e^{2i\theta})\big]`$ with singularities at
  $`\lvert\mathrm{Im}\,\theta\rvert \approx \tfrac12\ln\big(1/\lvert q\rvertA^2\big) \approx 1.4`$, well inside $`\pi\omega/2 \approx 6`$. If so, $`F_\infty`$ is not entire, and
  the harmonic decay rate (1.02 at $`D = 60`$, growing) levels off near 1.4 instead of growing like $`\tfrac14\ln n`$.
- *The harmonic-1 condition is still open.* With the higher harmonics tuned, $`S_1/S_1^{(1)} = 1 + \sigma_1A^2 + \sigma_2A^4 + \ldots`$ with
  $`\sigma_1 = 0.743587 + 0.524476i`$, $`\sigma_2 = 0.118797 + 0.398074i`$, $`\sigma_3 = -0.053969 + 0.116587i`$, $`\sigma_4 \approx -0.0290 + 0.0109i`$
  (stable to $`10^{-5}`$–$`4\times10^{-4}`$ across fit windows). $`\ln S_1`$ is nearly linear in $`A^2`$, and four terms do not locate its
  zero; whether $`S_1 = 0`$ at $`A \approx 2.72`$, $`\omega = \omega_\infty`$ needs the series to about $`A^{25}`$ in extended precision
  (in progress). $`\sigma_1(\omega)`$ alone never vanishes.

### What is known about the selection, and what is open

- **The expansion diverges.** `solver/largeD_late_orders.py` runs the recursion of Proposition 1 numerically to order 36,
  as polynomials in $`x`$ with coefficients sampled in $`\theta`$. It reproduces sympy at orders 1–2 to $`10^{-11}`$, and roundoff
  harmonics are projected out. For $`F = 2.72\sin\theta`$ and $`\omega = \omega_\infty`$, the terms grow factorially (ratio of
  successive orders $`\approx 7`$–$`8.6\,k`$), dominated by ever-higher harmonics. The rate depends on the amplitude and on
  $`\omega`$. This is the setting of exponential asymptotics: the series is divergent and its Borel-type remainders carry the
  selection.
- **The weak directions look like the selection, measured.** The near-null directions of the Jacobian (the "junk"
  directions of the solver sections) are variations of $`F`$'s high harmonics, which Proposition 1 leaves free, so their
  singular values plausibly measure how strongly the hidden condition pins each harmonic; that identification has not
  been checked directly. For harmonics $`k = 7, 9, 11, 13`$ they fall between $`D = 20`$ and $`60`$ about like
  $`n^{-10.5}, n^{-14}, n^{-17.5}, n^{-21}`$, roughly $`n^{-1.6k}`$; data at $`n = 18`$–$`58`$ cannot tell such power laws from the
  exponential weakening seen in $`\max\lvert\partial\Delta/\partial F\rvert`$.
- **$`F`$'s singularities recede.** The decay rate of $`F`$'s harmonics grows roughly like $`\tfrac14 \ln n`$ (0.58 at
  $`D = 12`$, 1.02 at $`D = 60`$), so $`F_\infty`$ looks entire, and its harmonics beyond the first shrink as $`D`$ grows. (The amplitude
  expansion of the inner problem, above, predicts instead that the rate levels off near 1.4; the data cannot yet tell.)
- **The nonlinear late orders are dominated by something else.** In the nonlinear recursion the ratio of successive
  orders grows like $`7k`$–$`8.6k`$ (a singulant of modulus about $`0.13`$), against $`\lvert\chi\rvert \ge \pi/2`$ for the fast mode, so the fast
  singulant is not directly visible. `solver/largeD_late_structure.py` characterizes the dominant growth: the rate is the
  same at $`x = 0.25, 0.5, 0.75, 1`$ (at $`x = 0.25`$ it is carried by the $`x^3`$ coefficient; the top-degree share at $`x = 1`$
  falls to 0.008 by order 27), and the dominant harmonic grows like $`0.55k`$, almost independently of the amplitude
  (13, 15, 17 at order 25 for $`A = 1.36, 2.72, 5.44`$) and of $`\omega`$, while the rate does depend on them (about 6, 8.5,
  12 for those amplitudes, 25 at $`2\omega`$). An $`x`$-dependent singulant of either eikonal branch is ruled out: a heat-like
  slow-branch singulant would give a rate that depends on $`x`$, a fast one $`\lvert\chi\rvert \ge \pi/2`$. The divergence lives in the free data
  ($`\theta`$-harmonics of $`F`$ and the $`K_k`$), consistent with an $`x`$-independent Borel singularity per harmonic that moves
  towards 0 as the harmonic grows. Its ambiguities are exponentially small changes of $`F`$'s high harmonics, like the
  solver's weak directions, not light-cone terms. (In the inner series the analogous fast high-harmonic growth turned out
  to be the fast singularity itself, carried by harmonics with large exponents; whether the same holds for this outer
  growth is not settled.)
- **The critical exponent too** (`solver/inner_pert/`). The $`1/n`$ expansion of growing perturbations $`e^{\lambda\tau}`$ of the formal
  solution gives no condition on $`\lambda`$ either: Proposition 1's parity induction goes through with every $`\tau`$-derivative of a
  perturbation replaced by $`\lambda + \omega\partial_\theta`$ (`expand_pert.py`, sympy, through order 2). So the unstable mode, and hence
  $`\gamma_\infty = 1/\lambda_\infty`$, is also selected beyond all orders, by the same kind of fast-mode multiplier. In the inner problem two
  eigenvalues are exact, from symmetries: $`\lambda = 0`$ (a shift of $`\theta`$; `pert_check.py`, to $`6\times10^{-30}`$) and $`\lambda = 1`$ (a shift
  of the time of the singularity, $`\delta Z = (\xi\partial_\xi + \omega\partial_\theta)Z`$; `pert_check2.py`, with gravity, to $`10^{-177}`$), the
  D = ∞ limit of the solver's gauge mode $`1 - \langle G(1)\rangle`$. No symmetry gives $`\lambda = 2`$. The fast exponents of a perturbation
  are $`g + \lambda`$. In the one-harmonic model at its zero the physical root is $`\lambda = 1.765 \pm 0.005`$ ($`\gamma = 0.567`$;
  `pert_ratio.py`), as far from 2 as that model's $`\omega`$ is from $`\omega_\infty`$. Free fits of the data give
  $`\gamma_\infty = 0.4996`$–$`0.4998`$ (`g_fits.py`): why $`\lambda_\infty = 2`$ to $`10^{-3}`$ is open.
- **Open:** the D = ∞ selection rule itself. Proposition 2 gives the multiplier without gravity, $`C(n,s)`$, and the inner
  problem shows it is asymptotically a $`{}_2F_0`$ Stokes constant; Proposition 3 shows that propagation inside the light cone
  cannot change whether it vanishes. The multipliers with gravity have been computed for one and two harmonics (above),
  with no selection of $`\omega_\infty`$ yet. What remains is to go past $`M = 3`$ with rescaled unknowns
  $`F_\nu e^{\pi\lvert\nu\rvert\omega/2}`$, or without truncation by the amplitude expansion above (whose higher-harmonic conditions
  already reproduce $`F_3/F_1`$), to require the harmonic-1 multiplier to vanish as well,
  and to prove the working hypothesis that this is the selection rule. That would be an equation for $`\omega_\infty`$ and
  $`F_\infty`$ alone, from which $`\Delta_\infty`$ would follow to any precision.

### References

- M. W. Choptuik, PRL 70, 9 (1993).
- C. Gundlach, PRD 55, 695 (1997), [gr-qc/9604019](https://arxiv.org/abs/gr-qc/9604019).
- J. M. Martín-García and C. Gundlach, PRD 68, 024011 (2003), [gr-qc/0304070](https://arxiv.org/abs/gr-qc/0304070).
- J. Bland et al., CQG 22, 5355 (2005), [gr-qc/0507088](https://arxiv.org/abs/gr-qc/0507088).
- E. Sorkin and Y. Oren, [hep-th/0502034](https://arxiv.org/abs/hep-th/0502034).
- J. Bland and G. Kunstatter, PRD 75, 101501 (2007), [hep-th/0702226](https://arxiv.org/abs/hep-th/0702226); J. Bland, PhD thesis,
  University of Manitoba (2007).
- R. Donninger, [arXiv:0809.5177](https://arxiv.org/abs/0809.5177).
- R. Emparan and C. P. Herzog, large-$`D`$ review, [arXiv:2003.11394](https://arxiv.org/abs/2003.11394).
- Review of critical collapse (2025), [arXiv:2507.07636](https://arxiv.org/abs/2507.07636).
- C. Ecker, F. Ecker and D. Grumiller, PRL 136, 191401 (2026), [arXiv:2601.14358](https://arxiv.org/abs/2601.14358).
- C. Ecker, F. Ecker, D. Grumiller and T. Jechtl, SciPost Phys. 20, 176 (2026), [arXiv:2602.10185](https://arxiv.org/abs/2602.10185).
- R. B. Dingle, *Asymptotic Expansions: Their Derivation and Interpretation*, Academic Press (1973).
- NIST Digital Library of Mathematical Functions, §2.11 and §13.7 ([dlmf.nist.gov](https://dlmf.nist.gov)).

## Software: the Rust kernel

Rust port of the multiple-shooting residual `F(z)` used to compute Choptuik's discretely self-similar critical solution in D = n + 2 dimensions. It covers the centre series, the RK4 shooting segments and the horizon series on the Python solver's spectral layout (941 unknowns in the production configuration; [`solver/`](solver/)), and is
called from Python through PyO3 as a drop-in for `S.residual`, the Python reference residual.

### Backends

The build targets the architecture's baseline, so one wheel runs on any CPU of that architecture. The instruction set is detected at runtime, and each residual block runs inside a `#[target_feature]` entry point for it.

| Backend | Kernels | Instruction set | f64 lanes |
|---|---|---|---|
| `avx512` | explicit intrinsics | AVX-512F | 8 |
| `avx2` | explicit intrinsics | AVX2 + FMA | 4 |
| `neon` | explicit intrinsics | aarch64 Advanced SIMD | 2 |
| `auto` | plain loops, compiler-vectorised | best available | – |
| `baseline` | plain loops | SSE2 on x86_64 | – |

Every backend performs the same operations in the same order, so all of them give bitwise identical results. The tests check this on each backend the CPU supports.

```
backend="simd" (default)
   │
   ├── AVX-512F ............ avx512
   ├── AVX2 + FMA .......... avx2
   ├── aarch64 ............. neon
   └── none ................ RustResidual falls back to "auto"
```

`available_backends()` lists what this CPU can run, and `auto:avx2` style names pick plain loops for a given instruction set. Setting `RUSTKERNEL_ISA=avx2` (or `baseline`) caps the instruction set for the whole process, which turns AVX-512 off without a rebuild.

### Use

```python
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual

S = freeze_steps(make_long(23, D - 2, nodes=[0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6], Ks=60), z[-1])
R = RustResidual(S)                           # parity-folded layout, 2 threads, best SIMD backend
F = R(Zb)                                     # (B, 941) -> (B, 941), same as S.residual(Zb)
F = R(Zb, backend="avx2")                     # force a backend for this call
F = R(Zb, dedup=True)                         # block reuse for finite-difference batches
Fc = R.residual_complex(Zb.astype(complex))   # complex-step derivatives
Fh, Fl = R.residual_dd(z_hi, z_lo)            # double-double residual (~32 digits)
F0, J = R.jacobian(z)                         # exact Jacobian, (941, 941)
M = R.mode_matrix(z, lam)                     # growth-mode matrix, as DSSSpec.mode_matrix(z, lam)
R5 = RustResidual(S, match_node=5)            # segments beyond node 5 integrate inward (see below)
```

`match_node=m` moves the point where the centre and horizon integrations meet to node m: segments
b ≤ m integrate outward (node b−1 → b), segments b > m inward (node b → b−1), so the horizon data are
carried towards the centre instead of the centre data towards the horizon. The unknowns and the solution
are the same; only the conditioning of the matching rows changes. The default (`None`) is the last node,
identical to the Python reference.

### Double-double residual

Above D ≈ 26 the period Δ is selected by an effect that falls below f64 rounding noise, so the f64
residual can no longer pin it. `residual_dd` evaluates the same residual in double-double arithmetic
(`scalar::DD`: hi + lo with two-sum / FMA two-product, about 32 digits) on plain loops. At D = 30 the
residual's rounding noise (the maximum over the residual's entries) drops from 1.3e-11 (f64) to about 1e-27, at 8–9×
the cost of the SIMD f64 residual in that measurement (15 ms per double-double evaluation; a different configuration
from the Performance table's synthetic one).

### Exact Jacobian and growth-mode matrix

`jacobian` and `mode_matrix` differentiate the residual in forward mode with a dual-number scalar
(`scalar::Dual`): one evaluation of block b per unknown that block depends on, run on rayon threads.
The value part reproduces the f64 residual bit for bit; the tangent is the exact derivative, with no
finite-difference step.

`mode_matrix(z, lam)` is the linearisation for perturbations ~ exp(lam τ): every τ-derivative of the
perturbation gains `+ lam`, as in `DSS.D` and `DSS.solve_periodic` with `lam` set. Its determinant
changes sign at the growth rate of the unstable mode, and the critical exponent is γ = 1/λ.

Checked against the Python reference (`spec_c.jacobian_cs`, `spec_c.mode_matrix_cs`) at D = 24.235:

| | Python | Rust | Agreement |
|---|---|---|---|
| Jacobian | 3.5 s | 0.16 s | 1.7e-10 relative |
| M(λ) | 3.0 s | 0.25 s | 1e-10 relative |
| Growth rate λ (det root) | ~4 min per D | ~3.5 s per D | 5e-10 at D = 11.06, 13.98, 24.24 |


### Build and test

Python side: numpy, scipy, mpmath and sympy; python-flint for `solver/inner_borel/` (`pip install python-flint`).

```sh
pip install .                     # maturin build, enables the python feature
cargo test --release              # backend equivalence tests, no Python needed
python tests/test_bindings.py     # bindings test, after pip install .
./build.sh main                   # wheel into dist/, plus ablation variants with ./build.sh all
hp/build.sh                       # double-double solver, built for this machine's instruction set
```

### Performance

One residual of a synthetic production-sized configuration (`tests/test_bindings.py --time`), single thread, on an i7-1185G7.

| Build | Backend | ms per residual |
|---|---|---|
| runtime dispatch | `avx512` | 0.84 |
| runtime dispatch | `auto:avx512` | 0.83 |
| runtime dispatch | `avx2` | 1.05 |
| runtime dispatch | `auto:avx2` | 1.17 |
| runtime dispatch | `baseline` | 28.2 |
| `-C target-cpu=native` (earlier build, before runtime dispatch) | `simd` | 0.86 |
| `-C target-cpu=native` (earlier build, before runtime dispatch) | `auto` | 1.50 |

`baseline` is slow because x86_64 without FMA turns every fused multiply-add into a libm call, which keeps it bitwise identical to the other backends. Only pre-2013 x86 CPUs land there.

### Layout

```
src/
├── model.rs        residual blocks, centre and horizon series, RK4, LU, plain-loop backend,
│                   forward-mode linearization (Jacobian, growth-mode matrix)
├── scalar.rs       f64 / Complex64 / Dual / DD scalar trait, 64-byte aligned field type
├── dispatch.rs     runtime instruction-set detection and target_feature entry points
├── avx512.rs       hand-written AVX-512 backend
├── simd/
│   ├── mod.rs      lane-generic kernels shared by AVX2 and NEON
│   ├── avx2.rs     4-lane AVX2 + FMA
│   └── neon.rs     2-lane NEON
├── python.rs       PyO3 bindings (python feature)
└── tests.rs        bitwise backend equivalence
python/rustkernel/  Python wrapper, RustResidual
tests/              bindings test
hp/                 double-double solver (crate hpkernel)
├── src/v8.rs       8-lane f64 vector: AVX-512 / AVX2 / portable
├── src/dv.rs       8-lane double-double, Sum2 dot accumulator
├── src/lane.rs     value and forward-mode dual lanes
├── src/model.rs    centre series, Taylor stepping, horizon series, residual, Jacobian
├── src/linalg.rs   double-double LU, transpose solve, Gram matrix
├── python/hpkernel/driver.py   grid, seeds, Newton, deflated Newton, Levenberg-Marquardt, re-noding
├── src/dd.rs       scalar double-double (constants, set-up)
├── src/field.rs    tau fields on the half grid, folded spectral derivative
├── src/lib.rs      crate root
├── src/python.rs   PyO3 bindings (Solver)
├── build.sh        builds the crate for this machine's instruction set
├── scan_hp.py      continuation in D
├── export_dd.py    collects the runs' points into data/delta_dd.csv
├── dd_fits_mp.py   large-D model fits at 60 digits, held-out errors, predictions (--dmax for a data snapshot)
├── dd_candidates.py  candidate constants forced as the limit, held-out test (--dmax as above)
├── plot_heldout.py heldout.svg (python3 plot_heldout.py; drawn before the points above D = 60.1 were added)
├── data/delta_dd.csv   every double-double point: D, Delta (hi + lo), |F|, max dDelta/dF
├── data/local/     the second machine's runs (scan_up*.jsonl), folded into delta_dd.csv by export_dd.py
└── data/seeds/     saved solver states to restart continuation from
hp/largeD_*.py      large-D scaling of the solutions (profiles, F(theta), harmonic content, light-cone exponent)
comms.txt           task log between the cloud and the second machine
solver/README.md    the Python solver, scans and fits; solver/CONTEXT.md (summary), solver/notes.md (lab notebook)
solver/formula_all.py  the rational formulas (README coefficient table): cd solver/data && python3 ../formula_all.py
solver/rational_fit.py the search over rational forms; solver/formula.py the 1/D power series
solver/candidates.py   the f64 candidate-constant test: cd solver/data && python3 ../candidates.py
solver/plot_delta.py, plot_extrapolation.py, plot_candidates.py  delta.svg, extrapolation.svg, candidates.svg
                       (cd solver/data && python3 ../plot_delta.py, and likewise)
solver/plot_physics.py        physics.svg: e^Delta, fine-tuning digits per echo, field and gravity strength vs D
solver/largeD_expand.py       the 1/n expansion symbolically (sympy, Proposition 1)
solver/largeD_formal_check.py the expansion substituted into the exact equations (residual ~ n^-3 for any omega)
solver/largeD_late_orders.py  the same recursion numerically to high order (divergence of the series)
solver/largeD_fast_transport.py  the fast WKB mode with gravity against direct integration (Proposition 3)
solver/largeD_linear_late.py  late orders of the exact leading-order 1/n series (singulant Phi - i pi/2)
solver/largeD_dingle_K.py     their Dingle constant against the inner 2F0 prediction
solver/largeD_stokes_C.py     C(n,s) as the 2F0 Stokes constant times n^(s+1/2) and a phase
solver/largeD_inner.py        the inner problem at x ~ sqrt(n): formal series, check against the outer top degrees
solver/inner_stokes/          the inner Stokes multiplier by lateral solutions (lateral_np.py check|scan, zero_M1.py, zero_M3.py)
solver/inner_borel/           the same by Borel-Darboux in ball arithmetic (python-flint): inner_stokes.py linear|grav, inner_zero.py
solver/eeg20/                 the comparison with Ecker-Ecker-Grumiller's Eq. (20) (window identity, tilt, max c, Delta -> 0 fits)
solver/inner_amp/             the inner multipliers by an amplitude expansion with every harmonic exact (amp_series.py check, stokes_amp2.py wf K J)
solver/inner_pert/            growing modes of the inner problem (exact lambda = 0, 1 modes; one-harmonic root) and free gamma fits
solver/largeD_transport_sympy.py  Proposition 3 derived symbolically from the exact equations (b, c perturbed too)
solver/largeD_fast_transport_full.py  the transport check with the full O(1/n) background
solver/largeD_linear_poles.py  Proposition 4: the late orders as an exact sum over the poles in kappa
solver/largeD_linear_poisson.py  Proposition 4: Poisson summation of the pole sum (the singulant tower)
solver/largeD_linear_truncation.py  optimally truncated series against the exact solution (no fast mode inside x < 1)
solver/largeD_late_structure.py  where the nonlinear late-order growth lives (x, harmonic, amplitude)
```

### Limits

- The Python solver, scans and fits live in `solver/` (see `solver/README.md`).
- `jacobian` and `mode_matrix` use plain loops (no explicit SIMD) and the parity-folded layout's
  generic path; the residual itself uses the SIMD backends.
- Only the long-horizon-series variant with frozen RK4 step counts is ported.
- Complex-step evaluation uses plain loops, without explicit SIMD.
