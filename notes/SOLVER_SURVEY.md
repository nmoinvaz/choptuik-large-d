# Survey: the double-double solver (`hp/`) against existing tools, and whether to split it out

Written 2026-10-23 at the author's request ("see if it would be worth splitting it out into its own repo and if it has
been done in Rust; also what might make our solver more complete"). Web survey of crates.io / docs.rs / lib.rs, the
Julia and C++ ecosystems and the literature cited below. Small or unpublished crates can be missed by such a search;
"not found" below means not found in this survey, not "does not exist". No claim here goes into the README or the
papers (AGENTS.md, novelty claims).

## 1. What the solver is, in layers

About 10,900 lines: 10,160 of Rust in `hp/src/`, 750 of Python in `hp/python/hpkernel/driver.py`; dependencies
only `rayon` (and `pyo3`/`numpy` for the bindings).

| Layer | Files | What it does | Specific to this problem? |
|---|---|---|---|
| Arithmetic | `dd.rs`, `v8.rs`, `dv.rs`, `qd.rs`, `qv.rs`, `exact.rs` | double-double and quad-double (Hida–Li–Bailey algorithms), scalar and 8-lane SIMD (AVX-512, AVX2 + FMA, portable), bitwise the same results on every backend; an exact reference for the tests | no |
| Automatic differentiation | `lane.rs` | forward-mode dual lanes over double-double and f64, batched tangents (several Jacobian columns per pass) | no |
| Linear algebra | `linalg.rs`, `block.rs` | double-double LU, f64 LU, transpose solves, Gram matrices; block storage and a staircase LU for multiple-shooting Jacobians | no (the block shape is that of any multiple-shooting problem) |
| Solver engine | `field.rs`, parts of `model.rs` | Fourier collocation on a half grid in τ (antiperiodic fields), Taylor-series stepping in x with planned steps, multiple shooting, Newton and its mixed-precision variants (f64 Jacobian, double-double horizon block, GMRES correction with exact double-double Jacobian–vector products) | partly: the τ fields' parity and the half grid are this problem's |
| Model | most of `model.rs` | the Einstein–scalar equations in self-similar coordinates at real D, the centre series, the light-cone (horizon) series with its Frobenius structure, the phase condition and the freezing of high harmonics | yes |
| Driver | `driver.py`, `scan_hp.py`, … | seeds, re-noding, deflated Newton, Levenberg–Marquardt, continuation in D, step control | mostly generic Newton/continuation logic around the model |

## 2. Has it been done in Rust?

Pieces have; the combination was not found.

| Need | Rust crates found | Gap against `hp/` |
|---|---|---|
| double-double scalar | [twofloat](https://docs.rs/twofloat) (0.8.4, scalar, error bounds after Joldes–Muller–Popescu; math functions "preliminary"); [qd](https://docs.rs/crate/qd/latest) (0.8.0, 2025-11, the faer author's, based on Jack Poulson's C++ code, depends on `pulp`, almost undocumented) | no SIMD lanes, no bitwise cross-backend guarantee, no quad-double |
| quad-double | [qd on lib.rs](https://lib.rs/crates/qd) lists quad-double as "not yet implemented" | ours (`qd.rs`, `qv.rs`) has the accurate add and the sqrt repaired (notes/QUAD_DOUBLE.md) |
| double-float on GPUs | [df64](https://github.com/jesedv/df64) (pairs of f32, WGSL kernels; 6 commits) | a different target (f32 hardware) |
| dual numbers / forward AD | [num-dual](https://lib.rs/num_dual) (generalized hyper-dual numbers, f64-based; whether it wraps a custom float type is not documented); [Peroxide](https://joss.theoj.org/papers/10.21105/joss.10366.pdf) (`Jet<N>` Taylor-mode AD, f64); [ad_trait](https://arxiv.org/html/2504.15976v1) | none over double-double SIMD lanes |
| Taylor-series ODE integration | Peroxide's ODE solvers are Runge–Kutta, not Taylor; [diffsol](https://docs.rs/crate/diffsol) is an f64 IVP solver (BDF, SDIRK) | no high-order Taylor integrator in Rust found |
| boundary-value problems | [RustedSciThe](https://rust-digger.code-maven.com/crates/RustedSciThe) (damped Newton, 4th-order collocation, symbolic Jacobians, f64; its BVP chapter "under development") | f64 only; no periodic direction, no singular endpoints |
| spectral methods | [rustpde](https://github.com/preiter93/rustpde) (Chebyshev/Fourier Galerkin for 2-D Rayleigh–Bénard, f64, no releases) | f64; an evolution code, not a Newton solver |
| dense / sparse linear algebra | [faer](https://docs.rs/faer) (0.24.4; LU, QR, SVD, eigen, sparse LU; SIMD through `pulp`; generic `Mat<T>`, but extended-precision scalar support is not documented) | no double-double LU documented; no block-staircase LU |
| interval arithmetic | [inari](https://lib.rs/crates/inari) (f64 intervals, nightly Rust, last release 2021) | no double-double balls; little maintenance |
| arbitrary precision | `rug` (MPFR bindings), `astro-float`, `num-bigfloat` | about 100× slower than f64 near our precision ([van der Hoeven](https://www.texmacs.org/joris/fpsimd/fpsimd.html)) |
| portable SIMD | `std::simd` is still nightly-only; on stable, `wide` and `pulp` ([overview](https://pythonspeed.com/articles/simd-stable-rust)) | `hp/` uses `core::arch` intrinsics with a portable fallback, no NEON path |

Outside Rust, the nearest tools:

- **Taylor integration with SIMD and extended precision:** [heyoka](https://github.com/bluescarni/heyoka) (C++/Python, LLVM JIT, batch mode on AVX/AVX-512/NEON, 80- and 128-bit and arbitrary precision, variational equations; [paper](https://arxiv.org/pdf/2105.00800)). It is an initial-value integrator with no boundary-value or periodic-orbit support. Also TaylorIntegration.jl (Julia, jet transport).
- **Multi-double arithmetic:**
  - QD (C++), Boost.Multiprecision's `cpp_double_fp_backend`.
  - [MultiFloats.jl](https://docs.juliahub.com/MultiFloats/k11Ei/0.4.0/) (Float64x2…x8; about 2× DoubleFloats.jl and 40× BigFloat in QR; only +, −, ×, ÷, √ fully supported).
  - Kouya's branch-free SIMD multi-component arithmetic ([arXiv:2603.14926](https://arxiv.org/pdf/2603.14926v1), C with AVX2/NEON: branch-free pays for triple and quad-double, not for double-double, which matches what `hp/` found).
  - Verschelde's multiple-double linear algebra on GPUs.
- **Continuation and periodic orbits:** [BifurcationKit.jl](https://docs.sciml.ai/BifurcationKit/stable/) (pseudo-arclength and deflated continuation, periodic orbits by shooting, trapezoid or collocation, Newton–Krylov, branch switching), AUTO-07p, MATCONT, pde2path. These are double precision in practice.
- **Validated numerics:** CAPD, Arb/FLINT (already used here through python-flint), INTLAB, the radii-polynomial method for Fourier–Chebyshev periodic orbits of PDEs (Lessard, van den Berg, Castelli; e.g. [arXiv:1509.08648](https://arxiv.org/pdf/1509.08648)).
- **The same physics:** Gundlach's pseudo-spectral boundary-value formulation of the discretely self-similar solution ([gr-qc/9604019](https://arxiv.org/abs/gr-qc/9604019)) and Martín-García & Gundlach ([gr-qc/0304070](https://arxiv.org/abs/gr-qc/0304070)), in double precision; Reiterer & Trubowitz's computer-assisted existence proof at D = 4 (80 digits; [CMP 2019](https://link.springer.com/article/10.1007/s00220-019-03413-8)).

So in this survey, no Rust crate, and no tool in another language, combines:
- a periodic-in-one-variable boundary-value problem with singular endpoints;
- Taylor stepping and an exact Jacobian in double-double on SIMD lanes, bitwise across instruction sets;
- a mixed-precision Newton that converges to the double-double solution.

The closest in spirit are heyoka (the integrator side) and MultiFloats.jl (the arithmetic side).

## 3. Is it worth splitting out?

Two different questions, with different answers.

**The arithmetic layer (double-double and quad-double, scalar and 8-lane, with duals): yes, as a crate, once.**
- **Where the gap is.** Its gap in Rust is the clearest of the survey. The scalar crates have no SIMD lanes and no quad-double, and the one that depends on `pulp` documents almost nothing.
- **Tests worth showing.** It already has the tests a published crate needs: bitwise agreement across AVX-512, AVX2 and portable builds, an exact reference, error bounds against exact arithmetic, and mutation controls (notes/QUAD_DOUBLE.md).
- **A track record.** Its tests found two bugs in the QD library itself (the PRs to the QD fork).
- **Low coupling.** It does not depend on the model.
- **Costs:**
  - an API designed for others, with operator traits (`num-traits` compatibility), `Debug`/`Display`, documentation and examples;
  - a CI matrix (x86-64 with and without AVX-512, aarch64 once a NEON path exists);
  - crates.io releases, semver and issue handling.
- **Audience.** Plausible: anyone doing extended-precision linear algebra or ODEs in Rust (the faer ecosystem's own double-double crate shows the interest), astrodynamics, computational number theory.

**The solver (engine and model): not yet.**
- **Single-model code.** It is a one-model code. The engine's generic parts (Taylor stepping, collocation, multiple shooting, the Newton variants) are written against this model's state layout and parity.
- **No model trait.** A useful solver crate would need a model trait (state, right-hand side, endpoint series, extra conditions) and at least a second model to prove the interface. Without that it would be a repository nobody else could use.
- **Ordering.** First define the trait inside this repository. Port a second model through it, the ℓ-multiplet matter model (`solver/matter/dss_ell.py`) or a planted test with a known exact solution, such as the sine-Gordon breather. Then decide.

**Timing and mechanics.**
- **When.** Not while runs depend on the build. The local machine builds `hp/` from this repository (T43 now, T44 next), and the cloud's low-D run uses it.
- **How.** When the time comes: `git subtree split` (history kept), the solver depending on the new crate by path or git revision, and a check that every bit-identity reference still passes.

## 4. What would make the solver more complete

Ordered by expected value for this project, with the evidence for each.

1. **A validated (ball-arithmetic) mode for one point.**
   - *What:* the residual and Jacobian with rigorous enclosures (truncation of the centre and horizon series, Taylor remainders), and a Newton–Kantorovich or radii-polynomial test at a given D. This turns a double-double solution into an existence proof with an error bar.
   - *Why:* Corollary C of PROOF.md needs exactly this (proof queue P7), and Reiterer & Trubowitz did it at D = 4.
   - *Cost:* the largest item; the arithmetic layer would need interval or ball versions of the lanes.
2. **Double-double steps above M = 128.**
   - *What:* the exact double-double Jacobian, the Levenberg–Marquardt fallback and γ stop at M = 128 (notes/HIGH_M.md). The pieces exist: the f64 block LU plus the GMRES correction with exact Jacobian–vector products gives double-double steps without a double-double LU, and γ could use the same exact products in an Arnoldi iteration.
   - *Why:* the low-D push needs M = 256–512.
3. **FFT for the τ derivative.**
   - *What:* the spectral derivative is a dense matrix product. At M = 256 it is 57% of the Jacobian's time (notes/HIGH_M.md), and an FFT would save about 1.7× there, more at M = 384–512.
   - *Cost:* in double-double this needs an FFT over double-double lanes. No Rust crate found does that, so it would be written here.
4. **A real step controller for continuation.**
   - *What:* step size from the Newton contraction rate and the predictor's error (Deuflhard-style adaptive continuation) instead of fixed growth factors and a cap.
   - *Why:* the T43 pilot shows the scans' cap of 0.03 cost speed at D = 60 (edge about 0.04). Pseudo-arclength continuation is not needed (Δ(D) has no folds), but a step controller is.
   - *T43's result* (notes/STEP_TEST.md): Newton's edge is 0.12 at D = 45 and 0.034 at D = 60; Lagrange predictors through 6–8 points are 4–10 times faster than the cubic up to D = 55 and fail at 60. The controller should also adapt the predictor's degree.
5. **Certification in a sub-layout as a standard check.** `scan_hp.py --certify_nodes` (notes/BEYOND_LIGHT_CONE.md §5.1) lowers the first-order Δ bound 230–270 times at D = 60 for about 15 s a point. It belongs in the default large-D recipe.
6. **A quad-double residual when the double-double floor binds.**
   - *What:* the floor rises about 1.4× per unit of D.
   - *Why:* a scan toward D = 200 would hit it. The quad-double arithmetic exists, and a residual would cost about 7× (notes/QUAD_DOUBLE.md).
   - *Cost:* the Jacobian could stay double-double or f64 with the correction loop.
7. **Portability and packaging.**
   - *What:* a NEON path (Apple silicon and ARM servers fall back to the portable code now), wheels built by `maturin` in CI (the local machine had to build by hand because maturin was missing), a CI matrix running the bit-identity references.
   - *Why:* cheap; needed before any split.
8. **A model trait** (section 3): the step that makes the engine reusable for other critical-collapse models, self-similar blow-up, or time-periodic solutions such as oscillons and breathers.
9. **Smaller items:**
   - solution deflation to look for other branches (the existing deflation is of weak directions, not of solutions);
   - checkpoint and restart in the Rust layer for multi-hour M = 384–512 solves (the driver does this in Python now);
   - matrix-free Newton–Krylov with the exact Jacobian–vector products, if M grows past what the block LU's memory allows.

## 5. Recommendation

- Leave the solver in this repository for now.
- Do items 2, 3, 5 and 7 inside it, since they serve the current work (low D, large D).
- Define the model trait (item 8) when a second model is wanted.
- Split out the arithmetic layer as its own crate when there is time to support it publicly: it is the most clearly missing piece in Rust and the most self-contained.
- The validated mode (item 1) is the largest and most valuable addition for the proof work; it should be planned with proof queue item P7, not as a side project.
