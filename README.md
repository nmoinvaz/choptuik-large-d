# dss-residual-kernel

Rust port of the multiple-shooting residual `F(z)` used to compute Choptuik's discretely self-similar critical solution in D = n + 2 dimensions. It covers the centre series, the RK4 shooting segments and the horizon series on a 941-unknown spectral layout, and is called from Python through PyO3 as a drop-in for `S.residual`.

![Echo period Δ(D) and critical exponent γ(D) with one rational formula for all D ≥ 4](solver/delta.svg)

The echo period Δ(D) and critical exponent γ(D), computed with this kernel and the solver in [`solver/`](solver/).
The dashed curves are one rational formula in 1/(D−2) for every D ≥ 4 (below), fitted on the step-converged points
4 ≤ D ≤ 30.2 (step error extrapolated out, near-null directions solved, enough Fourier modes in time at low D);
hollow points are held out of the fit. Δ → Δ∞ = 1.64395 ± 0.00008 and γ → 1/2 − 0.758/n as D → ∞ (details in
[`solver/CONTEXT.md`](solver/CONTEXT.md)). Regenerate with `cd solver/data && python3 ../plot_delta.py`.

## The formula

For massless scalar critical collapse in $D = n + 2$ spacetime dimensions, the echo period $\Delta$ and the critical
exponent $\gamma$ of Choptuik's discretely self-similar solution are, for every $D \ge 4$,

$$
\Delta(D) = \frac{a_0 + a_1 x + a_2 x^2 + \cdots + a_8 x^8}{1 + b_1 x + b_2 x^2},
\qquad
\gamma(D) = \frac{\tfrac{1}{2} + e_1 x + e_2 x^2 + \cdots + e_7 x^7}{1 + f_1 x + f_2 x^2},
\qquad
x = \frac{1}{n} = \frac{1}{D-2},
$$

| $k$ | $a_k$ ($\Delta$, numerator) | $b_k$ ($\Delta$, denominator) | $e_k$ ($\gamma$, numerator) | $f_k$ ($\gamma$, denominator) |
|---|---|---|---|---|
| 0 | $+1.6438975180$ | $1$ | $+0.5$ (fixed) | $1$ |
| 1 | $+4.0378045077 \times 10^{1}$ | $+1.8251453626 \times 10^{1}$ | $+1.8083629635 \times 10^{1}$ | $+3.7683686025 \times 10^{1}$ |
| 2 | $+2.5233285989 \times 10^{2}$ | $+6.8182050167 \times 10^{1}$ | $+1.5241588758 \times 10^{2}$ | $+3.4245505235 \times 10^{2}$ |
| 3 | $+2.0673502321 \times 10^{2}$ |  | $-5.2341588541 \times 10^{1}$ |  |
| 4 | $-3.5709850575 \times 10^{2}$ |  | $+9.9259594505$ |  |
| 5 | $+3.1347320996 \times 10^{2}$ |  | $-9.8386450353 \times 10^{1}$ |  |
| 6 | $-4.1891374574 \times 10^{2}$ |  | $+7.4529506074 \times 10^{1}$ |  |
| 7 | $+3.2695252782 \times 10^{2}$ |  | $-4.8272358903 \times 10^{1}$ |  |
| 8 | $-1.6050467699 \times 10^{2}$ |  |  |  |

Eleven numbers for $\Delta$ and nine for $\gamma$, with no singularity at any physical $D$ (the denominators vanish only
at $D = -3.2, -11.0$ and $D = -13.3, -20.4$). Fitted on every computed point $4 \le D \le 30.2$ (`solver/formula_all.py`;
`solver/rational_fit.py` searches the forms: for $\Delta$ one numerator term beyond the smallest adequate form makes it
ten times tighter), they reproduce each point within its own precision:

| $D$ | $\Delta$: formula − computed | precision of the points | $\gamma$: formula − computed | precision of the points |
|---|---|---|---|---|
| 4 – 4.3 | $5\times10^{-10}$ | $3\times10^{-7}$ | $3.4\times10^{-7}$ | $5\times10^{-7}$ |
| 4.3 – 20 | $\le 4\times10^{-9}$ | $3\times10^{-11}$ – $4\times10^{-9}$ | $\le 3.3\times10^{-8}$ | $\le 7\times10^{-8}$ |
| 20 – 30 | $\le 1\times10^{-8}$ | $\le 3\times10^{-8}$ | $\le 2.3\times10^{-9}$ | $\sim 10^{-9}$ |
| 34 – 40, never seen by the fit | $\le 2.1\times10^{-6}$ | $1$ – $3\times10^{-6}$ | $\le 5\times10^{-7}$ up to $D = 35.8$ | |

At $D = 4$ the formula gives $\Delta = 3.4454524$ and $\gamma = 0.3739610$, the accepted four-dimensional values.
Near $D = 4$ the points themselves set the limit: the solution needs many Fourier modes in time there, and $\Delta(4)$
moves by $-4.3\times10^{-4}$, $-3.4\times10^{-5}$ and $-2.8\times10^{-6}$ from 23 to 31 to 39 to 47 modes, so the points
below $D = 5$ are extrapolations in the mode count. Above $D \approx 36$ the computed $\gamma$ scatter by up to
$6\times10^{-5}$ around both this formula and the power series below, which agree with each other; the growth-rate solve
is not reliable there. Below $D \approx 3.8$, where $\Delta(D)$ turns over (maximum $\approx 3.46612$ near $D = 3.79$),
the formula does not apply.

As $D \to \infty$ ($x \to 0$):

$$
\Delta(D) = a_0 + \frac{a_1 - a_0 b_1}{n} + \cdots \approx 1.6439 + \frac{10.37}{n} + \cdots,
\qquad
\gamma(D) = \frac{1}{2} + \frac{e_1 - f_1/2}{n} + \cdots \approx \frac{1}{2} - \frac{0.758}{n} + \cdots
$$

$$
\Delta_\infty = 1.64395 \pm 0.00008,
\qquad
\omega_\infty = \frac{2\pi}{\Delta_\infty} = 3.8220 \pm 0.0002,
\qquad
\gamma_\infty = \frac{1}{2} .
$$

This formula's own limit is $a_0 = 1.643898$; the quoted range $1.64388$–$1.64403$ covers all acceptable fits (power
series and rational forms in $1/D$, $1/(D-1)$, $1/(D-2)$ over several degrees and ranges, $\chi^2/\mathrm{dof} \le 1$).
The coefficients of one fit are not physical constants (they are correlated); only the function is.

**The limit rests on an assumption.** $\Delta_\infty = 1.64395 \pm 0.00008$ holds if $\Delta(D)$ is analytic in $1/D$
at $D = \infty$: no half-integer powers $n^{-1/2}$, no $\log n$, and no new scale far beyond the computed range. The
data do not test that. Formulas with those features reproduce every computed point from $D = 4$ to $40$ (fit and held
out) just as well: a series in $n^{-1/2}$ gives limits from $1.607$ to $1.691$ depending on its degree; adding one
$n^{-1/2}$ or $\log n / n$ term to the $1/D$ series moves the limit to $1.6437$–$1.6448$; and the rational form with
its limit forced to $\pi^2/6$ or $\sqrt e$ fits too, by placing a pole at $D \approx -440$ or $-5900$, a slow bend
that starts in the hundreds. All of these agree with the formula above to $10^{-5}$ up to $D \approx 60$ and fan out
above $D \approx 100$:

![Formulas that all fit the computed points, extended to D = 100000](solver/extrapolation.svg)

The analytic form is the natural default (the large-$D$ expansion of the equations has no other scale, and the only
structure the data show is the pole near $D \approx -9$), but it is not proven: the period is selected by an
exponentially small effect that the $1/D$ expansion never sees, and Ecker, Ecker and Grumiller argue for $\Delta \to 0$.
Settling it needs either points at $D$ in the hundreds or the $D \to \infty$ equations themselves.
`solver/plot_extrapolation.py` draws the graph.

**Values.**

| $D$ | $\Delta$ | $\gamma$ |
|---|---|---|
| 4 | 3.4454524 | 0.3739610 |
| 5 | 3.2217019 | 0.4131269 |
| 6 | 3.0094832 | 0.4300795 |
| 7 | 2.8427532 | 0.4399912 |
| 8 | 2.7117901 | 0.4467164 |
| 10 | 2.5210789 | 0.4555879 |
| 12 | 2.3891886 | 0.4614056 |
| 15 | 2.2530984 | 0.4673656 |
| 20 | 2.1120802 | 0.4737107 |
| 30 | 1.9649579 | 0.4807927 |
| 40 | 1.888568 | 0.484776 |
| 50 | 1.84166 | 0.48736 |
| 100 | 1.74503 | 0.49313 |
| 1000 | 1.6542 | 0.4992 |
| $\infty$ | $1.64395 \pm 0.00008$ | $1/2$ |

Up to $D = 40$ these are within the precision of the computed points. $D = 50$ is an extrapolation that every form
above agrees on to $10^{-5}$; $D = 100$ and $1000$ assume analyticity in $1/D$ (the other forms differ by up to
$2\times10^{-4}$ at $D = 100$ and $0.02$ at $D = 1000$).

<details>
<summary>Power series in 1/D for D ≥ 10 (the earlier form)</summary>

$$
\Delta(D) = \Delta_\infty + \sum_{k=1}^{10} \frac{c_k}{D^k},
\qquad
\gamma(D) = \frac{1}{2} + \sum_{k=1}^{10} \frac{g_k}{D^k},
\qquad
\Delta(D) \approx 1.64395 + \frac{10.4}{D} - \frac{28.4}{D^2} + \frac{239}{D^3} - \cdots
$$

Fitted on $10 \le D \le 30.2$ (`solver/formula.py`); it reproduces the points there to
$1.6\times10^{-8}$ ($\Delta$) and $2.3\times10^{-9}$ ($\gamma$). The coefficients grow about ninefold per order with
alternating signs: a pole near $D \approx -9$, the same one the rational form carries in its denominator. That pole
limits the series to $D \gtrsim 10$ and forbids cutting it short (five terms are off by $4\times10^{-3}$ at $D \ge 20$).

| $k$ | $c_k$ (for $\Delta$) | $g_k$ (for $\gamma$) |
|---|---|---|
| 0 | $+1.6438931360$ | $+0.5$ (fixed) |
| 1 | $+1.0374964007 \times 10^{1}$ | $-7.5854191958 \times 10^{-1}$ |
| 2 | $-2.8355347834 \times 10^{1}$ | $+8.2843462006$ |
| 3 | $+2.3900833130 \times 10^{2}$ | $-1.2603029673 \times 10^{2}$ |
| 4 | $-2.2446858552 \times 10^{3}$ | $+1.8446582182 \times 10^{3}$ |
| 5 | $+2.0424472078 \times 10^{4}$ | $-2.5165496284 \times 10^{4}$ |
| 6 | $-1.6230686158 \times 10^{5}$ | $+2.8434887038 \times 10^{5}$ |
| 7 | $+1.0037347608 \times 10^{6}$ | $-2.4157799331 \times 10^{6}$ |
| 8 | $-4.4127962953 \times 10^{6}$ | $+1.4042658549 \times 10^{7}$ |
| 9 | $+1.2046388629 \times 10^{7}$ | $-4.9128734266 \times 10^{7}$ |
| 10 | $-1.5266460427 \times 10^{7}$ | $+7.7494021741 \times 10^{7}$ |

</details>

**Candidate constants for $\Delta_\infty$.** Each candidate is forced as $c_0$ in 12 fits (variable $1/D$, $1/(D-1)$,
$1/(D-2)$; degree 9 and 10; lower cut $D \ge 10$ or $12$), so the test assumes analyticity in $1/D$ (above). The
table gives the fit quality, as $\chi^2/\mathrm{dof}$ of the
best and the worst of the 12 (0.62 is the unconstrained fit), and the rms error on the held-out points at
$D = 34$–$39.5$, which the fits never saw.

| Candidate | Value | $\chi^2/\mathrm{dof}$ (best .. worst) | Held-out rms | Verdict |
|---|---|---|---|---|
| none (free fit) | $1.64395 \pm 0.00008$ | 0.62 .. 3.4 | $1.4\times10^{-6}$ | |
| $\frac{5}{7}\sqrt{G}\, j_{0,1}$ | 1.6439747 | 0.62 .. 8.6 | $1.3\times10^{-6}$ | fits, contrived |
| $\frac{7}{5}\,(\ln 2)^{-2}\,\pi^{-1/2}$ | 1.6440014 | 0.62 .. 18 | $1.2\times10^{-6}$ | fits, contrived |
| $\frac{7}{2}\sqrt{\ln 2/\pi}$ | 1.6440152 | 0.62 .. 26 | $1.2\times10^{-6}$ | fits, contrived |
| $\frac{9}{8}\sqrt{j_{2,1}/j_{0,1}}$ | 1.6440207 | 0.62 .. 30 | $1.2\times10^{-6}$ | fits, contrived |
| $\frac{10}{9}\sqrt{j_{0,1}/\ln 3}$ | 1.6439063 | 0.62 .. 118 | $1.5\times10^{-6}$ | fits, contrived |
| $\pi^2/6 = \zeta(2)$ | 1.6449341 | 3.6 .. 13000 | $1.9\times10^{-6}$ | excluded if analytic in $1/D$ |
| $2\pi/j_{1,1}$ | 1.6397878 | 67 .. $3\times10^{5}$ | $1.3\times10^{-5}$ | excluded if analytic in $1/D$ |
| $\sqrt{e}$ | 1.6487213 | 82 .. $3\times10^{5}$ | $1.3\times10^{-5}$ | excluded if analytic in $1/D$ |
| $\varphi = (1+\sqrt5)/2$ | 1.6180340 | 2500 .. $10^{7}$ | $7.7\times10^{-5}$ | excluded if analytic in $1/D$ |

![How far each candidate constant moves Δ(D) over the computed range](solver/candidates.svg)

The graph shows, for every candidate, how much fixing $\Delta_\infty$ to it changes the fitted $\Delta(D)$ over the computed
range ($|\text{forced fit} - \text{free fit}|$, log scale), against the precision of the computed points (shaded).
The five forms inside the window stay at the level of the data; $\pi^2/6$ sits 10–100 times above it at
$D = 10$–$30$, and $2\pi/j_{1,1}$, $\sqrt e$ and $\varphi$ climb further out. Every curve has the same shape: forcing the
limit is a linear constraint on a weighted least-squares fit, so forced minus free is (candidate − free limit) times one
influence function $h(D)$ with $h(\infty) = 1$. It is $\sim 10^{-7}$ where the points are known to $3\times10^{-11}$
($D \le 22$), rises where their error bars step up ($D = 22$, $29.5$), and climbs toward 1 past the data:
$h(40) = 3.8\times10^{-3}$, $h(100) = 0.17$, $h(1000) = 0.85$. Regenerate with
`cd solver/data && python3 ../plot_candidates.py`.

$G$ is Catalan's constant and $j_{\nu,k}$ the $k$-th zero of the Bessel function $J_\nu$. The five that fit are all that a
search finds in the window $1.64388$–$1.64403$ among products $r\, b_1^{p_1} b_2^{p_2}$ ($r = i/j$ with $i, j \le 12$;
$p = \pm\tfrac12, \pm1, \pm2$; $b$ from $\pi$, $e$, $\sqrt2$, $\sqrt3$, $\ln 2$, $\ln 3$, Euler's $\gamma$, $\zeta(3)$, $\varphi$, $G$
and the first Bessel zeros), $2\pi/(r b)$ and $r \ln b$. None has a reason to appear here, and inside the window the data
cannot tell them apart: they move $\Delta(D)$ by less than $10^{-6}$ for $D \le 40$, below the precision of the held-out
points. Forcing a constant can exclude it but not select it, and only under the analytic assumption: with a
non-analytic or slow large-$D$ term allowed, $\pi^2/6$ and $\sqrt e$ fit as well (graph above). Identifying
$\Delta_\infty$ would take about eight digits and a settled large-$D$ form, from the $D \to \infty$ limit of the equations
rather than from larger $D$.

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
```

## Limits

- The Python solver, scans and fits live in `solver/` (see `solver/README.md`).
- `jacobian` and `mode_matrix` use plain loops (no explicit SIMD) and the parity-folded layout's
  generic path; the residual itself uses the SIMD backends.
- Only the long-horizon-series variant with frozen RK4 step counts is ported.
- Complex-step evaluation uses plain loops, without explicit SIMD.
