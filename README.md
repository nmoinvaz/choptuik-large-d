# dss-residual-kernel

Rust port of the multiple-shooting residual `F(z)` used to compute Choptuik's discretely self-similar critical solution in D = n + 2 dimensions. It covers the centre series, the RK4 shooting segments and the horizon series on a 941-unknown spectral layout, and is called from Python through PyO3 as a drop-in for `S.residual`.

![Echo period Δ(D) and critical exponent γ(D) with their power series in 1/(D−2)](solver/delta.svg)

The echo period Δ(D) and critical exponent γ(D), computed with this kernel and the solver in [`solver/`](solver/).
Both are fitted as power series in 1/D on the step-converged points (2x RK steps, near-null directions solved,
independent setups agreeing to ~1e-8); hollow points are held out of the fits. Δ → Δ∞ = 1.64395 ± 0.00008 and
γ → 1/2 − 0.758/D as D → ∞ (formula below; details in [`solver/CONTEXT.md`](solver/CONTEXT.md)). Regenerate with `cd solver/data && python3 ../plot_delta.py`.

## The formula

For massless scalar critical collapse in $D = n + 2$ spacetime dimensions, the echo period $\Delta$ and the critical
exponent $\gamma$ of Choptuik's discretely self-similar solution are

$$
\Delta(D) = \Delta_\infty + \sum_{k=1}^{10} \frac{c_k}{D^k},
\qquad
\gamma(D) = \frac{1}{2} + \sum_{k=1}^{10} \frac{g_k}{D^k},
$$

with the limits

$$
\Delta_\infty = 1.64395 \pm 0.00008,
\qquad
\omega_\infty = \frac{2\pi}{\Delta_\infty} = 3.8220 \pm 0.0002,
\qquad
\gamma_\infty = \frac{1}{2},
\qquad
g_1 = -0.758 \pm 0.003 .
$$

The leading terms are

$$
\Delta(D) \approx 1.64395 + \frac{10.4}{D} - \frac{28.4}{D^2} + \frac{239}{D^3} - \frac{2245}{D^4} + \cdots,
\qquad
\gamma(D) \approx \frac{1}{2} - \frac{0.758}{D} + \frac{8.28}{D^2} - \frac{126}{D^3} + \frac{1845}{D^4} - \cdots
$$

The coefficients grow about tenfold per order with alternating signs (a radius of convergence near $D \approx 10$),
so the series must not be cut short: the first five terms are off by up to $4\times10^{-3}$ even at $D \ge 20$. Use
all eleven coefficients (the degree-10 fit drawn in the graph, `solver/formula.py`):

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

**Accuracy.** On the fitted range $10 \le D \le 30.2$ the two polynomials reproduce the computed values to
$1.6\times10^{-8}$ ($\Delta$) and $2.3\times10^{-9}$ ($\gamma$). Points computed later at $D = 34$–$38$, never seen by
the fit, agree with the $\Delta$ formula to $2\times10^{-6}$, which is their own precision there. Below $D \approx 10$
the series does not apply; use the computed values. The individual $c_k$ of one fit are not physical constants (they
are correlated); $\Delta_\infty$ is quoted as the range over all acceptable fits (variable $1/D$, $1/(D-1)$, $1/(D-2)$,
degree 8–11, lower cut $D \ge 10$ or $12$, $\chi^2/\mathrm{dof} \le 1$): $1.64388$–$1.64403$.

**Excluded limits.** $\pi^2/6 = 1.64493$ (forcing it raises $\chi^2/\mathrm{dof}$ from 0.6 to between 3.6 and 800),
$2\pi/j_{1,1} = 1.63979$ and $\sqrt{e} = 1.64872$. Whether $\Delta_\infty$ has a closed form is open; four to five digits
cannot decide it.

**Values.**

| $D$ | $\Delta$ | $\gamma$ | |
|---|---|---|---|
| 4 | 3.4454524 | 0.3739609 | computed |
| 5 | 3.2217019 | 0.4131269 | computed |
| 6 | 3.0094832 | 0.4300795 | computed |
| 7 | 2.8427541 | 0.4399913 | computed |
| 8 | 2.7117900 | 0.4467164 | computed |
| 10 | 2.521079 | 0.455588 | formula |
| 12 | 2.3891886 | 0.4614056 | formula |
| 15 | 2.2530984 | 0.4673656 | formula |
| 20 | 2.1120802 | 0.4737107 | formula |
| 30 | 1.9649579 | 0.4807927 | formula |
| 40 | 1.888568 | 0.48478 | formula (computed points to $D = 38$ agree to $2\times10^{-6}$) |
| 50 | 1.84166 | 0.4874 | extrapolated |
| 100 | 1.7450 | 0.4931 | extrapolated |
| $\infty$ | $1.64395 \pm 0.00008$ | $1/2$ | limit |

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
