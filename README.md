# dss-residual-kernel

Rust port of the multiple-shooting residual `F(z)` used to compute Choptuik's discretely self-similar critical solution in D = n + 2 dimensions. It covers the centre series, the RK4 shooting segments and the horizon series on a 941-unknown spectral layout, and is called from Python through PyO3 as a drop-in for `S.residual`.

[`hp/`](hp/) is a separate double-double (~32 digit) solver built to reach large D, where f64 can no longer pin the
period ([below](#double-double-solver-hp-in-progress)). It has computed Δ from D = 8 to 60 to about $`10^{-14}`$, and the
points beyond the fitted range show that Δ is analytic in 1/D at large D
([below](#is-the-period-analytic-in-1d)). Why the 1/D expansion of the equations cannot fix the period, with proofs,
is in [Large D](#large-d-what-the-1d-expansion-can-and-cannot-see).

![Echo period Δ(D) and critical exponent γ(D) with one rational formula for all D ≥ 4, and the double-double points to D = 61](solver/delta.svg)

The echo period Δ(D) and critical exponent γ(D), computed with this kernel and the solver in [`solver/`](solver/).
The dashed curves are one rational formula in 1/(D−2) for every D ≥ 4 (below), fitted on the step-converged points
4 ≤ D ≤ 30.2 (step error extrapolated out, near-null directions solved, enough Fourier modes in time at low D);
hollow points are held out of the fit, and the green diamonds are double-double points up to D = 61, which it matches
to 6e-8. As D → ∞, γ → 1/2 − 0.758/n, and Δ → Δ∞ = 1.64389375 ± 0.00000001 if Δ is
analytic in 1/D (from the double-double points; the f64 points gave 1.64395 ± 0.00008). Details are in
[`solver/CONTEXT.md`](solver/CONTEXT.md). Regenerate with `cd solver/data && python3 ../plot_delta.py`.

## The formula

For massless scalar critical collapse in $`D = n + 2`$ spacetime dimensions, the echo period $`\Delta`$ and the critical
exponent $`\gamma`$ of Choptuik's discretely self-similar solution are, for every $`D \ge 4`$,

```math
\Delta(D) = \frac{a_0 + a_1 x + a_2 x^2 + \cdots + a_8 x^8}{1 + b_1 x + b_2 x^2},
\qquad
\gamma(D) = \frac{\tfrac{1}{2} + e_1 x + e_2 x^2 + \cdots + e_7 x^7}{1 + f_1 x + f_2 x^2},
\qquad
x = \frac{1}{n} = \frac{1}{D-2},
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
| 4.3 – 20 | $`\le 4\times10^{-9}`$ | $`3\times10^{-11}`$ – $`4\times10^{-9}`$ | $`\le 3.3\times10^{-8}`$ | $`\le 7\times10^{-8}`$ |
| 20 – 30 | $`\le 1\times10^{-8}`$ | $`\le 3\times10^{-8}`$ | $`\le 2.3\times10^{-9}`$ | $`\sim 10^{-9}`$ |
| 30.2 – 60, never seen by the fit | $`\le 6\times10^{-8}`$ | $`\sim10^{-14}`$ (double-double) | $`\le 5\times10^{-7}`$ up to $`D = 35.8`$ | |

An independent double-double computation ([below](#double-double-solver-hp-in-progress)) agrees with the formula to
$`2.7\times10^{-9}`$ or better on $`8 \le D \le 21`$, $`8\times10^{-9}`$ up to $`D = 30`$, and $`6\times10^{-8}`$ from there to
$`D = 60`$, which the fit never saw. It also shows that the f64 points from $`D = 8`$ upward (23 Fourier modes in time)
sit $`2`$–$`4\times10^{-10}`$ above the true values, so their quoted $`3\times10^{-11}`$ was optimistic by about ten; that is
below the formula's tolerance and does not change it.

At $`D = 4`$ the formula gives $`\Delta = 3.4454524`$ and $`\gamma = 0.3739610`$, the accepted four-dimensional values.
Near $`D = 4`$ the points themselves set the limit: the solution needs many Fourier modes in time there, and $`\Delta(4)`$
moves by $`-4.3\times10^{-4}`$, $`-3.4\times10^{-5}`$ and $`-2.8\times10^{-6}`$ from 23 to 31 to 39 to 47 modes, so the points
below $`D = 5`$ are extrapolations in the mode count. Above $`D \approx 36`$ the computed $`\gamma`$ scatter by up to
$`6\times10^{-5}`$ around both this formula and the power series below, which agree with each other; the growth-rate solve
is not reliable there. Below $`D \approx 3.8`$, where $`\Delta(D)`$ turns over (maximum $`\approx 3.46612`$ near $`D = 3.79`$),
the formula does not apply.

As $`D \to \infty`$ ($`x \to 0`$):

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

$`\Delta_\infty`$ comes from the double-double points ([below](#is-the-period-analytic-in-1d)): power series in $`1/D`$ and
$`1/(D-2)`$ of degree 16–20 fit all of them from $`D = 10`$ to $`60`$ to $`4\times10^{-15}`$ and give limits from
$`1.643893744`$ to $`1.643893752`$. The f64 points gave $`1.64395 \pm 0.00008`$, and this formula's own limit is
$`a_0 = 1.643898`$, $`4\times10^{-6}`$ high, which its fitting range could not see. The coefficients of one fit are not
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
structure the data show is the pole near $`D \approx -9`$), but it is not proven: the period is selected by an
exponentially small effect that the $`1/D`$ expansion never sees, and Ecker, Ecker and Grumiller argue for $`\Delta \to 0`$.
The double-double points up to $`D = 60`$ favour it strongly; a proof needs the $`D \to \infty`$ equations themselves.
`solver/plot_extrapolation.py` draws the graph; the faded curves are the ones the double-double points rule out.

**Values.**

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
| $`\infty`$ | $`1.64389375 \pm 0.00000001`$ | $`1/2`$ |

$`\Delta`$ for $`D = 8`$–$`60`$ is double-double (interpolated to $`10^{-14}`$ where $`D`$ is not itself a computed point); the
rest of the table, and every $`\gamma`$, is from the f64 solver and is within the precision of its points up to
$`D = 40`$. $`\Delta(70)`$ and $`\Delta(100)`$ are extrapolations that every model fitting the double-double points agrees on,
to $`10^{-11}`$ and $`2\times10^{-8}`$ (the half-integer series included). $`D = 1000`$ and $`\infty`$ assume analyticity in
$`1/D`$; the half-integer series differ there by $`3\times10^{-6}`$ and more.

<details>
<summary>Power series in 1/D for D ≥ 10 (the earlier form)</summary>

```math
\Delta(D) = \Delta_\infty + \sum_{k=1}^{10} \frac{c_k}{D^k},
\qquad
\gamma(D) = \frac{1}{2} + \sum_{k=1}^{10} \frac{g_k}{D^k},
\qquad
\Delta(D) \approx 1.64395 + \frac{10.4}{D} - \frac{28.4}{D^2} + \frac{239}{D^3} - \cdots
```

Fitted on $`10 \le D \le 30.2`$ (`solver/formula.py`); it reproduces the points there to
$`1.6\times10^{-8}`$ ($`\Delta`$) and $`2.3\times10^{-9}`$ ($`\gamma`$). The coefficients grow about ninefold per order with
alternating signs: a pole near $`D \approx -9`$, the same one the rational form carries in its denominator. That pole
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

**Candidate constants for $`\Delta_\infty`$: tried, all ruled out.** Nine closed forms were tried as the limit, first on
the f64 points and then on the double-double points. None of them passes the second test, so none of them is
$`\Delta_\infty`$, given that $`\Delta`$ is analytic in $`1/D`$ (which the double-double points support,
[below](#is-the-period-analytic-in-1d)).

- **f64 test** ($`D \le 39.5`$, `solver/candidates.py`): each candidate forced as $`c_0`$ in 12 fits (variable $`1/D`$,
  $`1/(D-1)`$, $`1/(D-2)`$; degree 9 and 10; lower cut $`D \ge 10`$ or $`12`$). The table gives $`\chi^2/\mathrm{dof}`$ of the best
  and the worst of the 12 (0.62 is the unconstrained fit) and the rms error on the held-out points at $`D = 34`$–$`39.5`$.
  Five candidates passed it; four were already excluded.
- **Double-double test** ($`D \le 59.8`$, `hp/dd_candidates.py`): each candidate forced as the limit of a power series in
  $`1/D`$ fitted on $`10 \le D \le 45`$ at degree 14 to 26, and compared with the 351 points from $`45`$ to $`59.8`$ that the fit
  never saw. The table gives the best of those degrees and how many times worse it is than the free fit ($`4.7\times10^{-13}`$
  at degree 16).

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
precision of those points. The double-double points can. The closest candidate sits $`1.3\times10^{-5}`$ from the limit,
more than 1000 times its uncertainty, and forcing any of them makes the fit predict the unseen points at least 100 times worse. A
forced fit can only bend toward its candidate by spending degrees, so it improves with degree (the table gives each
one's best) until it reaches the level where the free fits overfit too (about $`5\times10^{-11}`$ at degree 24–26), which is
still about 100 times the free fit's best.

![How far each candidate constant moves Δ(D) over the computed range](solver/candidates.svg)

The graph is from the f64 test. It shows, for every candidate, how much fixing $`\Delta_\infty`$ to it changes the fitted $`\Delta(D)`$ over the computed
range ($`|\text{forced fit} - \text{free fit}|`$, log scale), against the precision of the computed points (shaded).
The five forms inside the window stay at the level of the data; $`\pi^2/6`$ sits 10–100 times above it at
$`D = 10`$–$`30`$, and $`2\pi/j_{1,1}`$, $`\sqrt e`$ and $`\varphi`$ climb further out. Every curve has the same shape: forcing the
limit is a linear constraint on a weighted least-squares fit, so forced minus free is (candidate − free limit) times one
influence function $`h(D)`$ with $`h(\infty) = 1`$. It is $`\sim 10^{-7}`$ where the points are known to $`3\times10^{-11}`$
($`D \le 22`$), rises where their error bars step up ($`D = 22`$, $`29.5`$), and climbs toward 1 past the data:
$`h(40) = 3.8\times10^{-3}`$, $`h(100) = 0.17`$, $`h(1000) = 0.85`$. Regenerate with
`cd solver/data && python3 ../plot_candidates.py`.

$`G`$ is Catalan's constant and $`j_{\nu,k}`$ the $`k`$-th zero of the Bessel function $`J_\nu`$. The five that passed the f64
test are all that a search finds in the old window $`1.64388`$–$`1.64403`$ among products $`r\, b_1^{p_1} b_2^{p_2}`$
($`r = i/j`$ with $`i, j \le 12`$; $`p = \pm\tfrac12, \pm1, \pm2`$; $`b`$ from $`\pi`$, $`e`$, $`\sqrt2`$, $`\sqrt3`$, $`\ln 2`$, $`\ln 3`$, Euler's
$`\gamma`$, $`\zeta(3)`$, $`\varphi`$, $`G`$ and the first Bessel zeros), $`2\pi/(r b)`$ and $`r \ln b`$. The same search finds nothing in
the double-double window $`1.6438936`$–$`1.6438939`$, nor in one 20 times wider ($`1.643890`$–$`1.643898`$). Whether
$`\Delta_\infty`$ has a closed form at all is open; the $`D \to \infty`$ limit of the equations is the way to find out.

## Double-double solver (`hp/`, in progress)

Above $`D \approx 26`$ the f64 points lose precision: the period is selected by an effect that weakens exponentially with
$`D`$, so rounding at $`10^{-16}`$ moves $`\Delta`$ by $`10^{-7}`$–$`10^{-6}`$ at $`D = 30`$–$`40`$, and points far above $`D = 40`$ are out of
reach. [`hp/`](hp/) (crate `hpkernel`) is a separate solver built for that regime, to settle the large-$`D`$ behaviour
with computed points rather than an assumed form.

- **Arithmetic:** double-double (hi + lo, about 32 digits) on 8-lane vectors: one AVX-512 register per lane group, two
  AVX2 registers, or a portable array, chosen at compile time (`hp/build.sh` builds with `-C target-cpu=native`).
  Error-free two-sum and FMA two-product; sums of products accumulated Sum2-style.
- **No fixed-step error:** Taylor series in $`x`$ (order 36, step points planned once per $`D`$ so the truncation term is
  $`10^{-32}`$), centre and horizon series run until their tails are below $`10^{-33}`$, and pure collocation in $`\tau`$ on the
  half period ($`p, m`$ odd, $`a, c`$ even) with no spectral truncation.
- **Exact Jacobian:** forward-mode duals over double-double, double-double LU for Newton, a transpose solve that gives
  $`\partial\Delta/\partial F`$ at every point, and a Levenberg–Marquardt fallback.
- **Continuation in $`D`$:** prediction by the cubic through the last four converged points (a tangent from the
  Jacobian is wrong in the weak directions, below), Newton with the weakest directions deflated, and from $`D = 35`$ on
  the centre data's $`\tau`$-harmonics above 13 frozen (below). 32 collocation points in $`\tau`$, nodes at
  $`x = 0.3, 0.4, \ldots, 0.7`$; 30–50 s per point on two cores up to $`D = 60`$. The tolerance on $`\lvert F\rvert`$ goes
  from $`10^{-25}`$ at $`D = 35`$ to $`10^{-22}`$ at $`D = 56`$, following the residual floor, which rises about twofold per unit
  of $`D`$.

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

A sample of the computed points (all of them, from $`D = 8.25`$ up, are in
[`hp/data/delta_dd.csv`](hp/data/delta_dd.csv)), with the residual and the largest entry of $`\partial\Delta/\partial F`$, the factor by which a residual error moves $`\Delta`$:

| $`D`$ | $`\Delta`$ | $`\lvert F\rvert`$ | $`\max\lvert\partial\Delta/\partial F\rvert`$ |
|---|---|---|---|
| 8.25 | 2.683434296073425 | $`5\times10^{-31}`$ | $`1.2\times10^{2}`$ |
| 12 | 2.389188553640616 | $`8\times10^{-31}`$ | $`5.9\times10^{2}`$ |
| 16 | 2.218342998377886 | $`4\times10^{-30}`$ | $`6.8\times10^{3}`$ |
| 21.19141 | 2.087734162683867 | $`4\times10^{-29}`$ | $`2.0\times10^{5}`$ |
| 25.07972 | 2.023511166632090 | $`1\times10^{-28}`$ | $`2.1\times10^{6}`$ |
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

What the solver has shown:

- **The weak directions.** The near-null directions that stalled the f64 solver ("junk") are a ladder of high
  $`\tau`$-harmonics of the centre data, one pair per odd harmonic $`k`$ with singular values about $`e^{-0.8k}`$
  ($`10^{-10}`$ at $`k = 7`$ down to $`7\times10^{-18}`$ at $`k = 31`$ at $`D = 21`$; the weakest is $`10^{-24}`$ at $`D = 35`$). They
  are outgoing waves near the centre that barely reach the matching point, and they belong to the boundary-value problem
  itself: moving the first node or the matching point leaves them unchanged. Their own effect on $`\Delta`$ is tiny
  ($`\le 10^{-12}`$ per unit amplitude for $`k \ge 17`$), but second-order errors leak into them amplified by $`1/\sigma`$,
  so Newton's basin shrinks with $`D`$; at $`D = 8`$, 80 points in $`\tau`$ already fail to converge. From $`D = 35`$ the solver
  pins the centre data's harmonics above 13 at zero and drops their matching on the horizon side: the weakest singular
  value rises from $`10^{-24}`$ to $`10^{-17}`$ and plain Newton converges again. The cut moves $`\Delta`$ by $`5\times10^{-15}`$ at
  $`D = 35`$ and $`3\times10^{-17}`$ at $`D = 40`$; a cut at 11 does not converge at $`D = 40`$.
- **How fast the selection weakens.** $`\max\lvert\partial\Delta/\partial F\rvert`$ grows about $`1.7`$–$`1.8`$ times per unit
  of $`D`$, to $`9\times10^{13}`$ at $`D = 60`$ (table), and the residual floor rises about twofold per unit. Their product, a
  worst case that ignores the direction of $`F`$, bounds the error in $`\Delta`$ by $`10^{-14}`$ at $`D = 45`$ and $`2\times10^{-9}`$ at
  $`D = 60`$. The linearised estimate $`-\partial\Delta/\partial F \cdot F`$, which accounts for it, stays between
  $`10^{-21}`$ and $`10^{-17}`$, and the points themselves fit smooth curves to $`4\times10^{-15}`$ all the way to $`D = 60`$.
- **How good the values are.** The points fit a smooth curve to $`4\times10^{-15}`$, not $`10^{-20}`$. Points between
  $`D = 27`$ and $`35.2`$, solved before the freezing, carry a few $`10^{-15}`$ from the weak directions; 25 of them sit
  visibly off their neighbours and `hp/dd_fits_mp.py --clean` drops them. The frozen points from $`35.35`$ to $`37`$ carry
  the cut's $`\le 5\times10^{-15}`$, which falls off with $`D`$. Elsewhere the error is about $`10^{-16}`$.
- **The f64 points.** Against the double-double values they are within $`5.5\times10^{-11}`$ at $`D = 15.6`$–$`21`$ (quoted
  $`3\times10^{-11}`$), $`3.6\times10^{-9}`$ at $`22`$–$`29.3`$ (quoted $`1.5`$–$`2\times10^{-9}`$), $`1.8\times10^{-8}`$ at $`30.1`$ and
  $`7\times10^{-8}`$ at $`34.2`$, and $`0.6`$–$`2.1\times10^{-6}`$ high from $`35`$ to $`40.3`$ (quoted $`1`$–$`3\times10^{-6}`$). The rational
  formula above holds to $`2.7\times10^{-9}`$ on $`8`$–$`21`$, $`8\times10^{-9}`$ on $`21`$–$`30`$ and $`6\times10^{-8}`$ on $`30`$–$`60`$.

### Is the period analytic in 1/D?

Yes, as far as the points up to $`D = 60`$ can tell. Each model below is a polynomial in one variable, fitted at 60 digits
on the cleaned double-double points $`10 \le D \le 45`$ (or $`\le 37.5`$) and compared with every point above, up to
$`D = 60`$, which it never saw (`hp/dd_fits_mp.py 10 45 --clean`):

![How well each model predicts double-double points it never saw](hp/heldout.svg)

| Model | max residual on $`10 \le D \le 45`$ | worst error on $`45 < D \le 60`$ | fitted on $`D \le 37.5`$: worst error up to $`60`$ | limit |
|---|---|---|---|---|
| $`1/D`$, degree 16 | $`4.1\times10^{-15}`$ | $`5.1\times10^{-13}`$ | $`4.1\times10^{-11}`$ | 1.64389375 |
| $`1/D`$, degree 18 | $`4.0\times10^{-15}`$ | $`1.8\times10^{-12}`$ | $`1.5\times10^{-10}`$ | 1.64389377 |
| $`1/(D-2)`$, degree 18 | $`4.0\times10^{-15}`$ | $`2.7\times10^{-12}`$ | $`7.8\times10^{-11}`$ | 1.64389377 |
| $`(D-2)^{-1/2}`$, degree 20 | $`3.9\times10^{-15}`$ | $`1.9\times10^{-11}`$ | $`3.9\times10^{-10}`$ | 1.636 |
| $`D^{-1/2}`$, degree 20 | $`3.8\times10^{-15}`$ | $`1.5\times10^{-10}`$ | $`6.1\times10^{-9}`$ | 1.547 |

Every model fits its range to the level of the points, so the fit alone decides nothing; the unseen points do. Fitted
on $`D \le 45`$, the $`1/D`$ series predicts $`\Delta`$ fifteen units of $`D`$ ahead to $`5\times10^{-13}`$, about 100 times the
noise of the points, while the series in half-integer powers miss by 40 to 300 times more. Their limits also jump with
the degree (1.636 to 1.694 for $`(D-2)^{-1/2}`$ and 1.54 to 2.75 for $`D^{-1/2}`$ at degree 20–24 on all the points), so they
give no usable $`\Delta_\infty`$. Adding one non-analytic term ($`D^{-1/2}`$, $`D^{-3/2}`$, $`\ln D / D`$, $`\ln D / D^2`$ or
$`e^{-D/4}`$) to the degree-16 series changes neither the prediction ($`1.6`$–$`5\times10^{-13}`$ at $`D = 60`$) nor the limit
(1.64389375), and the term's coefficient is not determined: $`D^{-1/2}`$ gets $`3\times10^{-5}`$ at degree 14 and
$`1\times10^{-7}`$ at degree 16 (`hp/dd_fits_mp.py 10 45 --clean --mixed`).

Fitted on all the cleaned points ($`10 \le D \le 60`$), power series in $`1/D`$ and $`1/(D-2)`$ of degree 16–20 give limits
from $`1.643893744`$ to $`1.643893752`$, so $`\Delta_\infty = 1.64389375 \pm 0.00000001`$. The analytic form is not proven (the
period is selected by an effect the $`1/D`$ expansion does not see), but nothing in the points up to $`D = 60`$ asks for
anything else.

Status: the cloud climb stopped at $`D = 60.06`$ and the local machine carried it to $`D = 60.96`$ (task T12 in
[`comms.txt`](comms.txt); its points are in [`hp/data/local/`](hp/data/local/) and in the data file). There the climb
stalls, and not because of anything in the problem: from a converged point Newton reaches the usual residual floor
at the same $`D`$, and the Jacobian's weak spectrum changes smoothly. The two weakest directions, harmonics 11 and 13
of the centre data, are pinned only to about $`\lvert F\rvert / \sigma`$, so their amplitudes jitter by about
$`5\times10^{-9}`$ from point to point; the cubic predictor amplifies that 30-fold, past Newton's basin along exactly
those directions. Carrying those harmonics on a straight-line trend instead (`scan_hp.py --junk_hold 9
--junk_trend N`, with `--corrector dnp` to converge every point to the bottom) gets past $`D = 60.96`$, but only in
steps of $`0.002`$–$`0.003`$. Points above $`D = 61`$ add little to $`\Delta_\infty`$ now, so the climb is paused there.

## Large D: what the 1/D expansion can and cannot see

The double-double points fix $`\Delta_\infty = 1.64389375`$ numerically. This section asks where the number comes from
in the equations. The answer so far: **the period is not determined at any finite order of the $`1/D`$ expansion of
the field equations.** For every choice of the period and of a free time profile, the expansion can be carried to all
orders (Proposition 1). The leading-order problem is solved exactly by hypergeometric functions, and those are never
analytic at the light cone (Proposition 2). So the period is selected by effects smaller than every power of $`1/D`$.
Finding that selection rule is open; the last part of the section lists what is known about it.

Notation: $`n = D - 2`$, $`\varepsilon = 1/n`$, $`\omega = 2\pi/\Delta`$, and $`\theta`$ the phase of the echoing (period
$`2\pi`$). The equations are the ones at the top of [`hp/src/model.rs`](hp/src/model.rs), with $`b = a^2`$ and
$`G = (n-1)\,c\,(b-1)`$:

```math
x c_x = (n-1)\,c\,(b-1), \qquad
x b_x = b\,(p^2+m^2) - (n-1)\,b\,(b-1),
```
```math
x(x-c)\,p_x = -\omega x\,p_\theta + G p - c p + \tfrac{n}{2}\,c\,(p+m), \qquad
x(x+c)\,m_x = -\omega x\,m_\theta - G m + c m - \tfrac{n}{2}\,c\,(p+m),
```

on $`0 \le x \le 1`$, with $`p, m`$ odd and $`b, c`$ even in $`\theta`$, regular at the centre $`x = 0`$, analytic at the past
light cone $`x = 1`$, and $`c(1) = 1`$ (the gauge that puts the light cone at $`x = 1`$).

### How the solution scales with D

Measured on the double-double solutions at $`D = 12`$–$`60`$ (`hp/largeD_scaling.py`, `hp/largeD_F.py`):

- $`p \approx x\,F(\theta)/\sqrt n`$ near the centre, and $`p + m`$ is smaller than $`p`$ by another factor of order $`1/n`$;
- $`c = 1 - O(1/n)`$, and $`b - 1 = (p^2+m^2)/(n-1)`$ to about 3%;
- the light cone stays at $`x - c = x - 1`$ to $`10^{-3}`$;
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

**Proposition 1.** Let $`F(\theta)`$ be any smooth $`2\pi`$-periodic function with odd harmonics only, $`\omega_0 \ne 0`$,
$`\omega_1, \omega_2, \ldots`$ any numbers and $`K_1(\theta), K_2(\theta), \ldots`$ any smooth odd-harmonic functions.
Then there is exactly one formal solution with $`D_0 = 2xF`$ in which, at every order,
$`S_k`$ is an even polynomial in $`x`$ starting at $`x^2`$, $`D_k - xK_k`$ is an odd polynomial starting at $`x^3`$, and $`G_k`$
and $`B_{k+1}`$ are even polynomials ($`B`$ starting at $`x^2`$) with $`G_k(1) = 0`$. Every order is regular at the centre and
analytic at the light cone, and $`m(x) = p(-x)`$ holds order by order.

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

Polynomials are analytic everywhere, so the light-cone condition is automatic, and $`c(1) = 1`$ holds by construction.
Nothing in the construction constrains $`F`$, $`\omega_0`$ or the $`\omega_k`$. $`\square`$

The first orders, from `solver/largeD_expand.py` (sympy, run to fourth order):

```math
P_0 = xF, \quad B_2 = 2x^2F^2, \quad G_1 = F^2\,(1-x^2), \quad S_1 = 2x^2\,(F + \omega_0 F'),
\quad D_1 = x^3\,(\omega_0^2 F'' + 3\omega_0 F' + 2F - 2F^3) + xK_1 .
```

*Check.* `solver/largeD_formal_check.py` substitutes the expansion (through $`S_3`$, $`G_3`$, $`B_4`$) into the exact equations,
with $`F`$ taken from the $`D = 60`$ solution. The relative residual of every equation falls exactly like $`n^{-3}`$ for
$`n = 50, \ldots, 800`$, both for $`\omega_0 = 2\pi/1.64389`$ and for a wrong $`\omega_0 = 2\pi/1.4`$. The expansion is right
to that order, and it does not care what the period is.

**Consequence.** $`\Delta(D)`$ itself does have a $`1/D`$ expansion: it is analytic in $`1/D`$, as the double-double points
show. But its coefficients cannot be computed order by order from the field equations. The selection works through
terms smaller than any power of $`1/n`$.

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

which is finite and non-zero for every real $`\omega \ne 0`$, odd $`\mu`$ and $`n > 2`$. So no time-periodic self-similar
solution of the linear problem is analytic at the light cone, in any dimension.

*Proof.* With $`c = b = 1`$ the difference equation gives $`(n - 1)\,(p+m) = x^2\,\partial_x(p-m) - x\,\partial_x(p+m) + s x\,(p-m)`$,
and the sum gives $`x\,\partial_x(p-m) - (p-m) = x^2\,\partial_x(p+m) + sx\,(p+m)`$. Eliminating $`p+m`$ gives

```math
x^2(1-x^2)\,y'' + x\,\big(n - 2 - 2(s+1)x^2\big)\,y' - \big(n - 2 + s(s+1)x^2\big)\,y = 0 , \qquad y = p - m ,
```

and with $`y = x\,w(z)`$, $`z = x^2`$, this is Gauss's equation $`z(1-z)w'' + [\kappa - (\alpha+\beta+1)z]\,w' - \alpha\beta\,w = 0`$ with
$`\alpha = (1+s)/2`$, $`\beta = (2+s)/2`$, $`\kappa = (n+1)/2`$. Its exponents are $`0, 1-\kappa`$ at $`z = 0`$ (so $`y \sim x`$ or $`x^{2-n}`$; regularity
picks $`{}_2F_1(\alpha,\beta;\kappa;z)`$) and $`0,\ \kappa-\alpha-\beta = n/2 - 1 - s`$ at $`z = 1`$. The connection formula (Abramowitz & Stegun 15.3.6)

```math
{}_2F_1(\alpha,\beta;\kappa;z) = \frac{\Gamma(\kappa)\Gamma(\kappa-\alpha-\beta)}{\Gamma(\kappa-\alpha)\Gamma(\kappa-\beta)}\,{}_2F_1(\alpha,\beta;\alpha+\beta-\kappa+1;1-z)
+ (1-z)^{\kappa-\alpha-\beta}\,\frac{\Gamma(\kappa)\Gamma(\alpha+\beta-\kappa)}{\Gamma(\alpha)\Gamma(\beta)}\,{}_2F_1(\kappa-\alpha,\kappa-\beta;\kappa-\alpha-\beta+1;1-z)
```

holds because $`\kappa - \alpha - \beta`$ is not an integer ($`s`$ is imaginary and non-zero). The second coefficient is $`C(n,s)`$. The
Gamma function has no zeros, and its arguments $`\alpha`$, $`\beta`$, $`\alpha+\beta-\kappa`$ are not poles because their imaginary parts
are $`\mu\omega/2`$, $`\mu\omega/2`$, $`\mu\omega \ne 0`$. $`\square`$

*Check.* At 40 digits (mpmath), the formula satisfies the original first-order equations to $`10^{-40}`$ for $`n = 18`$–$`1000`$. For the
fundamental at $`\omega = \omega_\infty`$, $`\lvert C\rvert = 0.0121,\ 0.0114,\ 0.0124,\ 0.0192,\ 0.0405`$ at $`n = 18, 38, 58, 200, 1000`$.
It grows like $`\sqrt n`$ ($`\lvert C\rvert/\sqrt n \to 0.00128`$), small only through the factor of order $`e^{-\pi\mu\omega/2}`$.

**What this means.** At leading order the self-similar oscillation is a free scalar wave, and a free wave cannot be
both regular at the centre and analytic at the light cone. The non-analytic piece $`(1-x^2)^{n/2-1-s}`$ is smaller than any
power of $`1/n`$ at every fixed $`x < 1`$, which is why Proposition 1 never sees it. Gravity, through the $`O(1/n)`$ couplings,
has to cancel its coefficient harmonic by harmonic. That cancellation is the selection rule for $`\Delta`$ and for the
profile $`F`$.

### What is known about the selection, and what is open

- **The expansion diverges.** `solver/largeD_late_orders.py` runs the recursion of Proposition 1 numerically to order 36,
  as polynomials in $`x`$ with coefficients sampled in $`\theta`$. It reproduces sympy at orders 1–2 to $`10^{-11}`$, and roundoff
  harmonics are projected out. For $`F = 2.72\sin\theta`$ and $`\omega = \omega_\infty`$, the terms grow factorially (ratio of
  successive orders $`\approx 7`$–$`8.6\,k`$), dominated by ever-higher harmonics. The rate depends on the amplitude and on
  $`\omega`$. This is the setting of exponential asymptotics: the series is divergent and its Borel-type remainders carry the
  selection.
- **The weak directions are the selection, measured.** The near-null directions of the Jacobian, the "junk" of the
  solver sections, are exactly the variations of $`F`$'s harmonics that Proposition 1 leaves free. Their singular values measure
  how strongly the hidden condition pins each harmonic. For harmonics $`k = 7, 9, 11, 13`$ they fall like
  $`n^{-10.5}, n^{-14}, n^{-17.5}, n^{-21}`$ between $`D = 20`$ and $`60`$, roughly $`n^{-1.6k}`$.
- **$`F`$'s singularities recede.** The decay rate of $`F`$'s harmonics grows roughly like $`\tfrac14 \ln n`$ (0.58 at
  $`D = 12`$, 1.02 at $`D = 60`$), so $`F_\infty`$ looks entire, and its harmonics beyond the first shrink as $`D`$ grows.
- **Open:** the cancellation condition at $`D = \infty`$. Proposition 2 gives the linear half exactly. The missing half is the
  coefficient of $`(1-x^2)^{n/2-1-s}`$ produced by the gravitational couplings, to the order in $`1/n`$ at which it competes with
  $`C(n,s)`$. With both halves, the selection would become an equation for $`\omega_\infty`$ and $`F_\infty`$ alone, and $`\Delta_\infty`$
  would follow from it to any precision.

## Backends

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

## Use

```python
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual

S = freeze_steps(make_long(23, D - 2, nodes=[0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6], Ks=60), z[-1])
R = RustResidual(S)                           # folded, 2 threads, best SIMD backend
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

## Double-double residual

Above D ≈ 26 the period Δ is selected by an effect that falls below f64 rounding noise, so the f64
residual can no longer pin it. `residual_dd` evaluates the same residual in double-double arithmetic
(`scalar::DD`: hi + lo with two-sum / FMA two-product, about 32 digits) on plain loops. At D = 30 the
residual's rounding noise drops from 1.3e-11 (f64) to about 1e-27, at 8–9× the cost of the SIMD f64
residual (15 ms per evaluation).

## Exact Jacobian and growth-mode matrix

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


## Build and test

```sh
pip install .                     # maturin build, enables the python feature
cargo test --release              # backend equivalence tests, no Python needed
python tests/test_bindings.py     # bindings test, after pip install .
./build.sh main                   # wheel into dist/, plus ablation variants with ./build.sh all
hp/build.sh                       # double-double solver, built for this machine's instruction set
```

## Performance

One residual of a synthetic production-sized configuration (`tests/test_bindings.py --time`), single thread, on an i7-1185G7.

| Build | Backend | ms per residual |
|---|---|---|
| runtime dispatch | `avx512` | 0.84 |
| runtime dispatch | `auto:avx512` | 0.83 |
| runtime dispatch | `avx2` | 1.05 |
| runtime dispatch | `auto:avx2` | 1.17 |
| runtime dispatch | `baseline` | 28.2 |
| `-C target-cpu=native` (before) | `simd` | 0.86 |
| `-C target-cpu=native` (before) | `auto` | 1.50 |

`baseline` is slow because x86_64 without FMA turns every fused multiply-add into a libm call, which keeps it bitwise identical to the other backends. Only pre-2013 x86 CPUs land there.

## Layout

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
├── scan_hp.py      continuation in D
├── export_dd.py    collects the runs' points into data/delta_dd.csv
├── dd_fits_mp.py   large-D model fits at 60 digits, held-out errors, predictions
└── data/delta_dd.csv   every double-double point: D, Delta (hi + lo), |F|, max dDelta/dF
hp/largeD_*.py      large-D scaling of the solutions (profiles, F(theta), harmonic content)
solver/largeD_expand.py       the 1/n expansion symbolically (sympy, Proposition 1)
solver/largeD_formal_check.py the expansion substituted into the exact equations (residual ~ n^-3 for any omega)
solver/largeD_late_orders.py  the same recursion numerically to high order (divergence of the series)
```

## Limits

- The Python solver, scans and fits live in `solver/` (see `solver/README.md`).
- `jacobian` and `mode_matrix` use plain loops (no explicit SIMD) and the parity-folded layout's
  generic path; the residual itself uses the SIMD backends.
- Only the long-horizon-series variant with frozen RK4 step counts is ported.
- Complex-step evaluation uses plain loops, without explicit SIMD.
