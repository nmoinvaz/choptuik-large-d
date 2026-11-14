# Software: the Rust kernel

Moved verbatim from README.md (commit ea42b3e, lines 2359–2713) on 2026-11-04, in the reorganization of
[README_PLAN.md](notes/README_PLAN.md); the README keeps a pointer where each block stood.

## Software: the Rust kernel

Rust port of the multiple-shooting[^18] residual `F(z)` used to compute Choptuik's discretely self-similar critical
solution in D = n + 2 dimensions. It covers the centre series, the RK4 shooting segments and the horizon series on the
Python solver's spectral layout (941 unknowns in the production configuration; [`solver/`](solver/)), and is called from
Python through PyO3 as a drop-in for `S.residual`, the Python reference residual.

### Backends

<details>
<summary>Which backend runs where</summary>

The build targets the architecture's baseline, so one wheel runs on any CPU of that architecture. The instruction set is
detected at runtime, and each residual block runs inside a `#[target_feature]` entry point for it.

| Backend | Kernels | Instruction set | f64 lanes |
|---|---|---|---|
| `avx512` | explicit intrinsics | AVX-512F | 8 |
| `avx2` | explicit intrinsics | AVX2 + FMA | 4 |
| `neon` | explicit intrinsics | aarch64 Advanced SIMD | 2 |
| `auto` | plain loops, compiler-vectorised | best available | – |
| `baseline` | plain loops | SSE2 on x86_64 | – |

Every backend performs the same operations in the same order, so all of them give bitwise identical results. The tests
check this on each backend the CPU supports.

```
backend="simd" (default)
   │
   ├── AVX-512F ............ avx512
   ├── AVX2 + FMA .......... avx2
   ├── aarch64 ............. neon
   └── none ................ RustResidual falls back to "auto"
```

`available_backends()` lists what this CPU can run, and `auto:avx2` style names pick plain loops for a given instruction
set. Setting `RUSTKERNEL_ISA=avx2` (or `baseline`) caps the instruction set for the whole process, which turns AVX-512
off without a rebuild.

</details>

### Use

<details>
<summary>Calling the kernel from Python</summary>

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

</details>

### Double-double residual

<details>
<summary>The double-double residual</summary>

`residual_dd` evaluates the same residual in double-double arithmetic
(`scalar::DD`: hi + lo with two-sum / FMA two-product, about 32 digits) on plain loops. At D = 30 the
residual's rounding noise (the maximum over the residual's entries) drops from 1.3e-11 (f64) to about 1e-27, at 8–9×
the cost of the SIMD f64 residual in that measurement (15 ms per double-double evaluation; a different configuration
from the Performance table's synthetic one).

</details>

### Exact Jacobian and growth-mode matrix

<details>
<summary>How the Jacobian and the growth-mode matrix are computed</summary>

`jacobian` and `mode_matrix` differentiate the residual in forward mode with a dual-number scalar[^17]
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

The double-double kernel (`hp/`) has the same construction: with the growth rate `Ctx::lam` set, `jacobian` returns the
exact mode matrix M(λ) in double-double (Δ held fixed, phase row dropped), and `Solver.mode_matrix` and
`Solver.mode_det` (sign and ln|det| from a double-double LU) expose it to Python; at λ = 0 the residual and Jacobian are
bit-identical to before. `hp/gamma_dd.py` brackets the root of det M(λ) around the formula's value and refines it with
brentq; above D = 35 it drops the weak directions with the kfreeze-13 cut, without which they bias λ by about 10⁻⁹ at D
= 60. `hp/gamma_batch.py` runs every saved state on N processes (resumable), and `hp/gamma_compare.py` (given both
`gamma_dd.jsonl` files as arguments) compares with the f64 values and the formula.

</details>


### Build and test

<details>
<summary>Dependencies, build and test commands</summary>

Python side: numpy, scipy, mpmath, sympy and python-flint (`pip install python-flint`; needed by `check_numbers.py`,
`solver/proof/`, `solver/inner_amp/` and `solver/inner_borel/`).

```sh
pip install .                     # maturin build, enables the python feature
cargo test --release              # backend equivalence tests, no Python needed
python tests/test_bindings.py     # bindings test, after pip install .
./build.sh main                   # wheel into dist/, plus ablation variants with ./build.sh all
hp/build.sh                       # double-double solver, built for this machine's instruction set
```

</details>

### Performance

<details>
<summary>Timings of the residual</summary>

One residual of a synthetic production-sized configuration (`tests/test_bindings.py --time`), single thread, on an
i7-1185G7.

| Build | Backend | ms per residual |
|---|---|---|
| runtime dispatch | `avx512` | 0.84 |
| runtime dispatch | `auto:avx512` | 0.83 |
| runtime dispatch | `avx2` | 1.05 |
| runtime dispatch | `auto:avx2` | 1.17 |
| runtime dispatch | `baseline` | 28.2 |
| `-C target-cpu=native` (earlier build, before runtime dispatch) | `simd` | 0.86 |
| `-C target-cpu=native` (earlier build, before runtime dispatch) | `auto` | 1.50 |

`baseline` is slow because x86_64 without FMA turns every fused multiply-add into a libm call, which keeps it bitwise
identical to the other backends. Only pre-2013 x86 CPUs land there.

The double-double solver's Newton step (`hp/`): a blocked LU[^19] that skips exact zeros (panels of 32 columns, parallel
trailing update; 18–34× faster than the dense LU on one thread) and batched solves (`Solver.lu_solve_many`) for the
deflation subspace bring one D-step from 15.4 to 9.5 s on one thread and from 10.1 to 5.5 s on two (D = 20, L = 1153),
with every output bitwise identical; on 8 threads the estimate is about 2.3 s against 6.5 s. The step is now almost all
Jacobian (its columns already run in parallel). The mixed-precision step computes the Jacobian in f64 instead, with
several columns sharing one pass over the base trajectory (1.2–1.35× on top of the f64 lanes); at D = 3.7 the Jacobian
goes from 7.5 to 0.72 s at M = 48 and from 31 to 3.3 s at M = 80 on two threads
([notes/MIXED_PRECISION.md](notes/MIXED_PRECISION.md), `hp/profile_newton.py --mixed --solve`). Continuation in D stays
sequential, since each step starts from the previous solution inside Newton's basin; past D = 61 it costs about 22 hours
per unit of D.

</details>

### Layout

<details>
<summary>The repository's files, folder by folder</summary>

```
src/
├── lib.rs          crate root: module list (the Python bindings come with the python feature)
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
├── Cargo.toml      the arithmetic comes from the multidouble crate (crates.io, the version named there; on the
│                   compile-time backend): double-double and quad-double, scalar and 8-lane SIMD (AVX-512 / AVX2 /
│                   SSE2 / NEON / portable), dual lanes (with the complex tangent of modecount_cx), the double-double
│                   LU (and the complex determinant), the staircase LU and the tau-field kernels (with the early end
│                   of a series for the tangents); hp re-exports its modules (hpkernel::dd, ::dv, ...)
├── src/lane.rs     hp's Lane trait: every multidouble lane type on the compile-time backend
├── src/model.rs    the equations on multishoot's engine: centre series, Taylor steps, horizon series; residual,
│                   Jacobians (double-double, quad-, octo-double, f64, staircase storage), jets, mode matrix
├── python/hpkernel/driver.py   grid, seeds, Newton, deflated Newton, Levenberg-Marquardt, re-noding (its numpy
│                   double-double helpers come from multidouble's Python package, which build.sh installs)
├── src/field.rs    tau fields on the half grid (multidouble's field module: folded spectral derivative, Cauchy
│                   products, the many-right-hand-side LU solves)
├── src/lib.rs      crate root
├── src/python.rs   PyO3 bindings (Solver)
├── build.sh        builds the crate for this machine's instruction set
├── scan_hp.py      continuation in D (--kc_auto 30: the centre-series order the solution needs, bitwise safe)
├── kc_check.py     the centre-series order at large D: bitwise checks and timing
├── profile_newton.py   time of a Newton step by part; --mixed --solve for the mixed-precision step
├── mixed_check.py  checks of the mixed-precision Newton: Jacobian agreement, controls, convergence, bit-identity
├── mixed_diag.py   why the f64 Jacobian agrees only to 1e-7 at M >= 48 (step size, harmonics, M)
├── mixed_largeD*.py  the mixed-precision step at large D: where it fails (horizon block), the correction-loop prototype
├── bench_variants.py, plot_bench_variants.py  the Newton variants timed against D, and bench_variants.svg
├── plot_solve_times.py  solve_times.svg, how long each D took (--export writes data/solve_times.csv from the solve records)
├── plot_point_spacing.py  point_spacing.svg, the distance in D between our solved points
├── plot_newton_stall.py   newton_stall.svg, why the steps collapse (a model of one Newton step; reads the raw-data folder T73)
├── plot_stall_kappa.py    stall_kappa.svg, u*kappa against D and the residual floor (reads the raw-data folder T76)
├── plot_light_cone.py     light_cone.svg, the past light cone and the fast mode's size inside it
├── stall_margin.py   the stall detector: weakest directions, sigma_max and u*kappa at a state or a rebuilt seed
├── stall_summary.py  its results from the raw-data folder T76 (the power laws, the floor, the pinned systems)
├── highm_check.py, lowD_highM.py  checks of the larger tau grids, and the continuation toward D = 3 on them
├── lowD_highM_leg.py  a new leg of a finished lowD_highM.py run (its progress file, new targets)
├── step_test.py    the largest step in D that Newton tolerates at D = 45-60, for several solver variants (local task T43)
├── qd_check.py     checks of the quad-double arithmetic against mpmath, and across the SIMD backends
├── scan_qd.py, qd_planted.py  continuation in D with the quad-double Newton, and its planted-root check
├── export_dd.py    collects the runs' points into data/delta_dd.csv
├── t79_rows.py     T79's points (D 4-8.25, folder T79-win of the raw-data repository) into the low-D exporter's inputs
├── export_qd.py    the quad-double values of Delta at D 68.5-73 (folders T84-mac and T81 of the raw-data repository) into data/delta_qd.csv
├── export_pinned_qd.py  the quad-double Delta_full1 of the pinned solutions at D 71-73.27 (folder T84-mac), exact sums, into data/delta_pinned_qd.csv
├── qd_remainder.py  the quad-double points against the 1/D series fitted on the double-double data, and the bound on an e^(-aD) term
├── dd_fits_mp.py   large-D model fits at 60 digits, held-out errors, predictions (--dmax for a data snapshot)
├── short_forms.py  the shortest 1/D series per range of D with its error margin (README "The formula"; --selftest)
├── dd_candidates.py  candidate constants forced as the limit, held-out test (--dmax as above)
├── plot_heldout.py heldout.svg (python3 plot_heldout.py --dmax=60.1: the points up to D = 60.1)
├── fit_residuals.py  each point's distance from the degree-18 fit in 1/(D-1), into data/fit_residuals.csv
├── plot_errors.py  lowD_errors.svg and largeD_errors.svg, the error graphs (run fit_residuals.py first)
├── data_graphs.py  SVG graph primitives, vendored unchanged from github.com/nmoinvaz/speedy-gonzales (skills/graph-data)
├── data/delta_dd.csv   every double-double point: D, Delta (hi + lo; Delta_lo is the low word, not an error), |F|,
│                       max dDelta/dF, run, an error estimate and its kind (Delta_err, err_kind: the change from the next
│                       lower tau resolution at low D, the size of the nearby checks elsewhere), and whether a published
│                       value at that D agrees within its stated uncertainty (published_ref, published_diff,
│                       published_match; data/published_delta.csv); the continuation points (857 for D = 8.25-61.03, 342 from
│                       the continuation toward 70 for 61.04-67.77, runs T48-pilotB and T48-up70) and the
│                       best low-D value per D (run lowD-M<M>, D = 3.045-4.0), which the large-D fits skip;
│                       dDelta_exact on T48-up70's rows (the remaining correction with a quad-double horizon residual)
├── data/delta_dd_lowD.csv   the low-D points at every M, with gamma and status (export_lowD.py)
├── data/fit_residuals.csv   each continuation point from D = 10 up minus the degree-18 1/(D-1) fit (fit_residuals.py)
├── data/local/     the second machine's runs (scan_up*.jsonl), folded into delta_dd.csv by export_dd.py
└── data/seeds/     saved solver states to restart continuation from
hp/largeD_*.py      large-D scaling of the solutions (profiles, F(theta), harmonic content, light-cone exponent)
comms.txt           task log between the cloud and the local machines (win, mac)
comms_watch.py      waits for comms entries addressed to one local machine (its session runs it in the background)
util_report.py      cpu and memory of a runner's machine for its posts
AGENTS.md           conventions for the AI sessions that run computations here (which scripts to use, thread budget, comms)
paper/physics/      the physics paper (main.tex, built as dss_physics.pdf with make)
paper/math/         the mathematics paper (main.tex, built as dss_math.pdf with make)
paper/figures/      the PDF figures both papers share (conversions of the repo's SVG figures; make figures in paper/)
PROOF.md            roadmap for a proof of the selection rule (computer-assisted inner zero; analytic selection theorem)
GAMMA.md            why gamma_inf = 1/2 (solver/inner_pert/heat_form.py: the heat-equation form; rw_map.py: it is Rozali-Way's
                    system; pert_check3.py: the exact lambda = 2 mode)
matter/             other matter models at large D (LITERATURE.md: survey; STRUCTURE.md: large-D structure and predictions;
                    scripts in solver/matter/)
notes/              audits and assessments: REF_CHECK.md (bibliography), LIT_REVIEW.md (novelty claims), REFEREE_SIM.md
                    (simulated referees), BOOTLOOPS.md (the BootLoops toolkit); proof/FORMAL.md (Lean formalization)
solver/proof/       feasibility numbers for the proof roadmap (feasibility.py: sigma ratios, conditioning, tail size;
                    entire_cert.py: S_1 as an entire function of A^2, the factored series and certificate margins;
                    exact_sigma.py: sigma_1 rigorously enclosed by an exact telescoped formula (no fit);
                    fast_eigenvalues.py: the outer system's fast eigenvalues, Lemma B2; certify_M1.py, lin_check.py,
                    series_arb.py, residual_poly.py, pm_dryrun.py, fixed_point.py, krawczyk.py, validated_int.py: the
                    M = 1 certificate in ball arithmetic (one jump component's zero certified in a 1e-9 box);
                    review/: the independent review's test scripts R1-R8 (proof/A_review.md);
                    capd_check/: the independent C++/CAPD re-check (build.sh pins CAPD; A_certify.md section 10);
                    mutation_controls.py: deliberately broken versions of the final test;
                    lemma6_*.py: lemma 6a, the start-value lemma near the real axis, the lower-path certificate,
                    and the tube sweep (superseded); lemma6b1_*.py: lemma 6b1 by validated Taylor discs and the
                    covering check (A_certify.md section 12);
                    tail_harmonics.py, reconcile.py, amp_trunc.py, family_amps.py: the harmonic-tail analysis;
                    outer_borel.py, inner_corrections.py, lightcone_exponents.py: checks for the selection theorem)
proof/              the proof roadmap's parts (A_certify.md: the computer-assisted certificate, prototyped at M = 1;
                    A_review.md: its independent review (both lemmas it asked for are since proved);
                    B_tail.md: gap (b), the harmonic tail;
                    C_selection.md: the selection theorem for M harmonics, lemma by lemma;
                    D_entire.md: the entire-function route to a certified inner zero)
solver/README.md    the Python solver, scans and fits; solver/CONTEXT.md (summary), solver/notes.md (lab notebook)
solver/formula_all.py  the rational formulas (README coefficient table): cd solver/data && python3 ../formula_all.py
solver/rational_fit.py the search over rational forms; solver/formula.py the 1/D power series
solver/candidates.py   the f64 candidate-constant test: cd solver/data && python3 ../candidates.py
solver/plot_delta.py, plot_extrapolation.py, plot_candidates.py  delta.svg, extrapolation.svg, candidates.svg
                       (cd solver/data && python3 ../plot_delta.py, and likewise)
reproduce.sh, reproduce.py    one-command check of the headline numbers (PASS/FAIL/SKIP; --full adds Proposition 4)
check_numbers.py              ties the numbers quoted in the README, paper and proof notes to their source files
                              (--selftest: every check must reject a one-unit shift; --full: the Delta_inf fit grid)
hp/states.py                  where saved solutions are read from (hp/data/states, else the runs' directory)
hp/planted_dinf.py            planted-truth test of the Delta_inf error bar (log in hp/data/planted_dinf.log)
hp/dd_fits.py                 an earlier, double-precision version of the large-D fits (dd_fits_mp.py supersedes it)
hp/pinning.py                 pin-to-trend for the weak p1 harmonics (used by scan_hp.py --pin, --predictor lagrange_dd)
hp/compare_interp.py, hp/compare_old.py   the f64 data set against the double-double scans
hp/data/states/               53 double-double solutions, about one per unit of D from 8.25 to 60, for the profile scripts
hp/dinf_budget.py             error budget for Delta_inf: model x degree x fit range x cleaning x snapshot
hp/gamma_dd.py                gamma(D) in double-double: root of det M(lambda) (gamma_batch.py --procs N; gamma_compare.py;
                              results in hp/data/gamma_dd.jsonl, 28 points D = 10-61,
                              and hp/data/local/gamma_dd.jsonl, all 67 points D = 8.25-61)
solver/plot_physics.py        physics.svg: e^Delta, fine-tuning digits per echo, field and gravity strength vs D
solver/plot_echo.py           echo.svg: the echo at D = 4, 10, 20, 60 (data: solver/echo_data.py -> solver/data/echo_spectrum.npz)
solver/plot_spectrum.py       spectrum.svg: the echo's harmonic spectrum and decay rates against the D = infinity prediction
solver/plot_selection.py      selection.svg: the selection rule order by order (bracket, truncation sequence, exponent)
solver/plot_selection_zoom.py selection_zoom.svg: the same estimates as distance from 1 (log), with the order-14 projections
solver/plot_matter.py         matter.svg: gamma(D) across matter models, against 1/D
solver/plot_first_order.py    first_order.svg: the first 1/D corrections from the D = infinity theory against M
solver/supercrit/plot_supercrit.py  supercrit.svg: the supercritical front, the lapse behind it, finite-n CSS checks
hp/plot_modecount.py          modecount.svg: the unstable-mode count in the lambda plane, and the real rates against 1/D
hp/plot_lowD.py               lowD.svg: below D = 4: the maximum, the f64 bias, the tau modes needed
hp/plot_lowD_values.py        lowD_values.svg: every double-double Delta and gamma below D = 4, and the change with M
solver/plot_gamma.py          gamma.svg: lambda = 1/gamma against 1/n with the lambda = 2 limit, and the free fits
solver/simplify_math.py       converts trivial inline math in README.md to Unicode (GitHub renders at most ~1400 expressions)
solver/largeD_expand.py       the 1/n expansion symbolically (sympy, Proposition 1)
solver/largeD_formal_check.py the expansion substituted into the exact equations (residual ~ n^-3 for any omega)
solver/largeD_late_orders.py  the same recursion numerically to high order (divergence of the series)
solver/largeD_fast_transport.py  the fast WKB mode with gravity against direct integration (Proposition 3)
solver/largeD_linear_late.py  late orders of the exact leading-order 1/n series (singulant Phi - i pi/2)
solver/largeD_dingle_K.py     their Dingle constant against the inner 2F0 prediction
solver/largeD_stokes_C.py     C(n,s) as the 2F0 Stokes constant times n^(s+1/2) and a phase
solver/largeD_inner.py        the inner problem at x ~ sqrt(n): formal series, check against the outer top degrees
solver/inner_stokes/          the inner Stokes multiplier by lateral solutions (lateral_np.py check|scan, zero_M1.py, zero_M3.py);
                              the M-harmonic selection by nested solve (nested.py M = 3, full_M.py, fixedA.py) and, for
                              M >= 5, by hierarchical solve (hier.py M A wf [--fixA] [--procs 2]); conditioning (m3_diag.py)
solver/inner_borel/           the same by Borel-Darboux in ball arithmetic (python-flint): inner_stokes.py linear|grav, inner_zero.py
solver/eeg20/                 the comparison with Ecker-Ecker-Grumiller's Eq. (20) (window identity, tilt, max c, Delta -> 0 fits)
solver/inner_amp/             the inner multipliers by an amplitude expansion with every harmonic exact (amp_series.py check, stokes_amp2.py;
                              amp_flint.py wf K J in ball arithmetic; amp_flint_par.py, the same 26x faster with --procs N;
                              scan_omega.py for an N-process scan in omega; zero_selection.py; arg_crossing.py, where
                              across omega the zero becomes real, order by order)
solver/inner_pert/            growing modes of the inner problem (exact lambda = 0, 1 modes; one-harmonic root) and free gamma fits
hp/largeD_Fratios.py          harmonic ratios F_{nu+2}/F_nu of the double-double solutions, extrapolated to D = infinity
hp/compare/                   D = 5 by double-double (dd_D5.py) and the comparison with Ecker et al.'s published data (cmp_ecker.py)
solver/largeD_transport_sympy.py  Proposition 3 derived symbolically from the exact equations (b, c perturbed too)
solver/largeD_fast_transport_full.py  the transport check with the full O(1/n) background
solver/largeD_linear_poles.py  Proposition 4: the late orders as an exact sum over the poles in kappa
solver/largeD_linear_poisson.py  Proposition 4: Poisson summation of the pole sum (the singulant tower)
solver/largeD_prop4/          Proposition 4's proof checked: late-order prediction (euler_late.py), constant K (kcheck.py), switch near x = 1 (switch2.py)
solver/largeD_linear_truncation.py  optimally truncated series against the exact solution (no fast mode inside x < 1)
solver/largeD_late_structure.py  where the nonlinear late-order growth lives (x, harmonic, amplitude)
```

</details>

### Limits

<details>
<summary>What the kernel does not do</summary>

- The Python solver, scans and fits live in `solver/` (see `solver/README.md`).
- `jacobian` and `mode_matrix` use plain loops (no explicit SIMD) and the parity-folded layout's
  generic path; the residual itself uses the SIMD backends.
- Only the long-horizon-series variant with frozen RK4 step counts is ported.
- Complex-step evaluation uses plain loops, without explicit SIMD.
- The kernel is compiled for three sizes of the parity-folded τ grid, nh ≤ 40, 72 and 96 lanes (`Ctx<40>`, `Ctx<72>`,
  `Ctx<96>`; nh = N/2 with N = 4⌈(3kc + 2)/4⌉), so kc ≤ 63; kc 49–63 run padded to 96 lanes. Larger kc needs another
  size (for example `Ctx<128>` with `MAXV = 16` in `src/avx512.rs`). Adding the 96-lane size left the 40- and
  72-lane results bitwise identical (residuals on all backends and thread counts, and the exact Jacobian).

</details>
