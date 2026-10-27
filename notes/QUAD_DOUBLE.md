# Quad-double arithmetic in the Rust kernel (`hp/`)

> **Moved.** The arithmetic described here (`dd.rs`, `v8.rs`, `dv.rs`, `qd.rs`, `qv.rs`, `exact.rs`, `qd_tests.rs`, the
> dual lanes and the LU) now lives in the separate crate [multidouble](https://github.com/nmoinvaz/multidouble) (crates.io), which
> `hp/Cargo.toml` names by version. Paths `hp/src/<file>` below are that crate's `src/<file>`; the feature `qd_mutate`
> is its feature `mutate`; `hp/qd_check.py` runs the Rust tests in a checkout of it. The logs below were made before the
> move; the code is the same.

Quad-double numbers (an unevaluated sum of four f64, about 64 significant digits) are now in the `hpkernel` crate:
a scalar type `QD` and an 8-lane SIMD type `QV` with bitwise the same results, on the AVX-512, AVX2 and portable
backends of `v8.rs`. Nothing in the solver uses them yet. Every existing output is bit-identical (section 9). The
logs behind every number here are in `hp/data/quad_double/`, and the commands run in `hp/`.

**Summary.**

- **Accuracy** against exact arithmetic (and, independently, mpmath): every operation is within its threshold in all
  seven input categories. The thresholds were written before the runs (section 2).
  - The accurate operations (accurate add, accurate mul, qd × f64, accurate div) err by at most 0.083 u.
  - The library's defaults (sloppy add, sloppy mul, sloppy div) and sqrt err by at most 0.016, 0.38, 2.1 and 0.38 u.

  Here u = 2⁻²⁰⁹ ≈ 1.2×10⁻⁶³, so 2.1 u is 63 correct digits and 0.02 u is 65. The literature's ε_qd is 0.25 u
  (section 3).
- **Two repairs of the QD library's algorithms**, each confirmed independently:
  - The accurate (IEEE-style) add drops the accumulator's low part v in one branch: 16.8 u in 2 of 20 000
    near-cancelling sums here, and up to 86 u in a later check of the current C++ library, against the stated 0.5 u.
    The first repair (x₃ = u + v) was incomplete: it still left up to 11 u on some sums. The repair now in the code
    renormalizes five terms, with a worst case of 0.016 u over 180 000 such sums (Finding 1, addendum).
  - The square root loses all but 53 bits above about 2⁸⁶⁰ and below about 2⁻⁸⁶⁰. Its operand is now scaled by an
    exact power of four.

  Also, the sloppy add, whose stated bound is only Cray-style, never exceeded 0.016 u relative to the result.
- **Cost** (one thread, AVX-512, section 6). Quad-double over double-double:

  | add | multiply | divide | a·b + c | dot-product term |
  |---|---|---|---|---|
  | 8.3× | 22.5× | 12× | 16× | 17× |

  AVX2 runs quad-double 1.3–1.9 times slower than AVX-512.
- **A quad-double residual** at D = 3.7, M = 80 (section 7):
  - Its lane work measures 10–11 times the double-double one. The model's own generic code with `QV` lanes takes
    0.97–1.00 s against 0.092–0.100 s on one thread.
  - With the horizon LUs kept in double-double and refined in quad-double, a residual would take about 1.05–1.08 s
    against 0.15 s, about 7× at the same discretization.
  - The Taylor and series orders for a smaller truncation error would add a factor of about 2 (tolerance 10⁻⁴²)
    to 4 (10⁻⁶²).
- **Stage 2** (section 8): `QV` implements `Lane`, and the generic residual code compiles and runs with it. A
  quad-double residual needs the constants, D, n and the phase row in quad-double, a scalar-type parameter in the
  model, iterative refinement of the horizon solves, and a four-part state in the Python bindings. Effort: about
  6–9 agent-days.

## 1. What was built

Code in `hp/src`:

| file | contents |
|---|---|
| `qd.rs` | `QD` (scalar): QD's two-sum, FMA two-product, three-sum, renormalization (four and five parts), sloppy and accurate add, qd ± f64 and ± dd, sloppy and accurate multiply, qd × f64 and × dd, square, sloppy and accurate divide, reciprocal, sqrt, integer power, `mul_pwr2`, abs; conversions from f64 and dd (exact) and to f64 and dd; constants π, 2π, ε (exact); exact decimal parsing (`parse`, `FromStr`) and printing (`to_sci`, `Display` with 66 digits); `QdAcc`, the scalar sum-of-products accumulator; the mutation hooks of the controls (`qd::mutate`, feature `qd_mutate`) |
| `qv.rs` | `QV` (8 lanes, four `V8` registers, structure of arrays): every `QD` operation with the same algorithm, the data-dependent branches run on all lanes with masks; load, store, splat, broadcast, lanes; `QDotAcc`; `impl Lane for QV` (stage 2) |
| `v8.rs` | added: the mask type `M8` (compare `ne`, `eq`, `gt`, `ge`; `and`, `or`, `not`; per-lane select `sel`), `sqrt`, `pow2_floor`, for all three backends (`_mm512_cmp_pd_mask` / `_mm512_mask_blend_pd`, `_mm256_cmp_pd` / `_mm256_blendv_pd`, arrays) |
| `exact.rs` | exact arithmetic on dyadic rationals (big integers, no dependency): the decimal input/output and the tests' reference |
| `qd_tests.rs` | the accuracy, agreement, decimal and control tests, the mpmath dump, the benchmark, the two diagnostics of section 3 |
| `qd_residual.rs` | `QV` through the model's blocks (test), a counting lane, the residual's operation mix and timing (bench) |

Also `hp/qd_check.py` (independent mpmath check; the three-backend check) and the cargo feature `qd_mutate`. No new
dependency and no inline assembly: the SIMD code is `std::arch` intrinsics only, as in `v8.rs`.

**Defaults.** The operators follow the QD library's default configuration (`qd_config.h`: `QD_IEEE_ADD` unset,
`QD_SLOPPY_MUL` and `QD_SLOPPY_DIV` set): `+`, `-` the sloppy add, `*` the sloppy multiply, `/` the sloppy divide.
`ieee_add`, `accurate_mul` and `accurate_div` are the others. (`DV`'s `+` is the accurate double-double add, which
costs about as much as the sloppy one; for quad-double the accurate add costs 4.5 times the sloppy one, section 6.)

**How the branches run on 8 lanes.** QD's renormalization skips zero error terms, and its accurate add merges the
eight parts by magnitude. Both branch on the data, and both are emulated exactly with masks:

- The renormalization's second pass, written with a slot index k (the part being filled), is
  (t, e) = quick_two_sum(s[k], c); s[k] = t; s[k+1] = e; k += (e ≠ 0). The last step at k = 3 keeps t only (QD's
  `s3 += c4`). k is held as threshold masks, and the slots are updated by selects.
- The merge add keeps per-lane read positions i, j and output position k. It reads a[i] and b[j] by selects, freezes
  lanes that have finished, runs at most seven steps, and stops once all lanes are done.

**Departures from QD, each for a reason found or stated here:**

1. The accurate add keeps v as a fifth term and renormalizes five terms where QD keeps u alone (section 3).
2. sqrt scales its argument by an exact power of four into [1, 4) (section 3).
3. The O(ε³) and O(ε⁴) partial products that QD sums in plain arithmetic are summed by FMA, in QD's order. A C
   compiler contracting QD's expression would do the same.
4. Decimal input and output are exact instead of QD's digit-by-digit loops. `parse` rounds the exact decimal value
   to four parts by greedy round-to-nearest (x₀ = RN(v), x₁ = RN(v − x₀), ...; error ≤ ulp(x₃)/2 ≤ 2⁻²¹²|v|).
   `to_sci` prints the exact four-part sum correctly rounded (half to even).
5. `to_f64` returns x₀ + (x₁ + (x₂ + x₃)) where QD returns x₀, as `DD::to_f64` returns hi + lo.
6. `QDotAcc` is new (QD has no accumulator); section 3.

The model never calls exp, log, sin or cos: its set-up (D matrix, phase weights, n) is computed in Python with
mpmath and passed in. So no transcendental functions were written; a quad-double set-up needs those values to about
70 digits from Python (section 8).

**Inlining.** On the SIMD backends the hot operations are forced inline: the renormalizations, the sloppy add and
multiply, × and + f64, and the accumulator. Done on the portable backend's 8-element arrays as well, that made its
test build exceed 3 GB of memory (killed) after 17 minutes. There the compiler now decides, and the build takes
16 CPU-minutes and 1.2 GB. The other large operations are never forced.

## 2. Algorithms, sources, and the bounds expected before the runs

- **Error-free transformations:** two-sum (Knuth), quick-two-sum (Dekker), and the FMA two-product. These are the
  same functions as in `dd.rs` and `dv.rs`.
- **Quad-double:** Y. Hida, X. S. Li, D. H. Bailey, *Algorithms for quad-double precision floating point arithmetic*,
  ARITH-15 (2001); LBNL report, https://escholarship.org/uc/item/69q5t2mj; library QD 2.3.
  - The texts of `renorm` (both), `three_sum`, `three_sum2`, `quick_three_accum`, `ieee_add`, `sloppy_add`, qd + f64
    and qd + dd were compared with `qd_inline.h` as served by an LXR mirror
    (https://eic-code-browser.sdcc.bnl.gov/lxr/source/include/qd/qd_inline.h). They were read through WebFetch, as
    quotes and line-by-line extracts.
  - The sloppy multiply, qd × f64, the square and the sloppy divide were checked step by step against a summary of
    the oct Lisp port (https://people.eecs.berkeley.edu/~fateman/generic/newer-versions/newoct/oct-3-arg/qd.lisp).
  - The configuration defaults come from `qd_config.h` on the same mirror.
  - The accurate multiply, qd × dd, the accurate divide and sqrt follow my reading of QD 2.3 and the paper's
    description: the O(ε⁴) partial products, five quotient terms, three Newton steps. They were not compared line by
    line with the C++ source.
- **The accumulator:** T. Ogita, S. M. Rump, S. Oishi, *Accurate sum and dot product*, SIAM J. Sci. Comput. 26 (2005)
  (SumK). *(Recalled, not re-read for this note.)*
- **Prior SIMD quad-double:** T. Kouya, *Acceleration of multiple precision matrix multiplication based on
  multi-component floating-point arithmetic using AVX2* (arXiv:2101.06584). It uses QD's sloppy add and multiply on
  AVX2 and runs the renormalization's branches as a scalar loop over the lanes.

The bounds expected before any test ran, and the pass thresholds, were committed first (e789a42). They are repeated
here. The paper states:

- the sloppy add: a ⊕ b = (1 + δ₁)a + (1 + δ₂)b with |δᵢ| ≤ ε_qd = 2⁻²¹¹ = 0.25 u, a Cray-style bound;
- the accurate add: |δ| ≤ 2ε_qd "or so" relative = 0.5 u;
- multiplication: |δ| ≤ ε_qd = 0.25 u;
- nothing for division or sqrt; and "Necessary conditions for this renormalization algorithm to work correctly are,
  unfortunately, not known."

My expectations, derived from the algorithms and not proved:

| operation | expected | threshold |
|---|---|---|
| accurate add | 0.5 u relative | 1 u |
| sloppy add, qd + f64, qd + dd | 0.25–0.5 u × (\|a\| + \|b\|) | 1 u × (\|a\| + \|b\|) |
| sloppy mul, qd × dd, square | ≲ 1 u relative | 2 u |
| accurate mul, qd × f64 | 0.25 u, ≲ 1 u | 1 u |
| sloppy div | ≲ 2 u relative | 4 u |
| accurate div | ≲ 1 u relative | 2 u |
| sqrt | ≲ 2 u relative | 4 u |
| parse | ≤ 0.125 u relative | 0.25 u |
| dot accumulator, n terms | ≲ (8 + n)/8 u × Σ\|aᵢbᵢ\| | (8 + n)/8 u |

Near underflow an absolute allowance of 16·2⁻¹⁰⁷⁴ was planned.

**Changed in the tests after the first runs** (each reported, none to make a failure pass):

- The near-underflow allowance is now defined on the quantity each test checks. For add and mul that is the result.
  For division it is the residual r·b − a: 16·2⁻¹⁰⁷⁴·(1 + |b|), because the quotient's parts cannot go below 2⁻¹⁰⁷⁴
  and the remainders are formed at the dividend's scale. For sqrt it is r² − a: 16·2⁻¹⁰⁷⁴·(1 + 2|r|). The first run
  put 16·2⁻¹⁰⁷⁴·|b| on the residual and failed 210 division cases; the second put 16·2⁻¹⁰⁷⁴ and failed 489 others.
  The cases were the two floors.
- The square got its own input ranges: the first run reused the multiply's, and a² overflowed.
- A seventh input category, *computed* operands, was added (section 3).
- `QD::EPS` was the decimal literal of QD (1.21543267145725e-63), which is not 2⁻²⁰⁹; it is now exact.

## 3. Accuracy

**How it is measured.** Every input is built from an exact random value by greedy rounding (`exact.rs`), without
any quad-double arithmetic, except the computed category, which is generated with no mutation active. Every
reference is exact.

- Add and mul: the result against the exact sum or product.
- Div: the residual |r·b − a| / |a|, which equals the relative error of r.
- sqrt: |r² − a| / (2|a|).
- The Cray-style operations are checked against |a| + |b|, and their error relative to the result is reported too.

`python3 qd_check.py mpmath` recomputes all of it with mpmath at 3000 bits (the 120 digits asked for are a lower
bound, needed so that sums of parts are exact under deep cancellation). It shares no code with `exact.rs`. On 2000
inputs per row it gives the same maxima and 99.9th percentiles as the Rust reference
(`hp/data/quad_double/mpmath_check.log`, exit 0).

The input categories, for every operation:

| category | inputs |
|---|---|
| wide | exponents of a and b uniform in ±400 (sqrt ±800) |
| cancel | adds: b = −a + d with \|d\| = 2⁻ᵏ\|a\|, k = 1..260; mul: (1 + d₁)(1 − d₂); div: b = a(1 + d); sqrt: 1 + d |
| overflow | results and operands up to 2¹⁰²² |
| underflow | results between 2⁻¹⁰²⁰ and 2⁻⁸⁶⁰, so that the last parts are subnormal or zero |
| pow2 | b (sqrt: a) within 2⁻ʲ of a power of two, j around the part boundaries (52–54, 105–107, 158–159, 211–212) or random, and exact powers of two |
| sparse | sums of 1–4 small integers with random binary gaps (exact sums, zero error terms: the renormalization's branches) |
| computed | operands from quad-double arithmetic (x·y·(1/y)), so with QD-renormalized parts instead of greedy ones; for the adds b = −x·y·(1/y) or b = −(a + d), so a + b is about 2⁻²¹⁰\|a\|, as in a residual's subtraction |

**Results** (in `hp/`, `QD_N=20000 QD_REPORT=1 cargo test --release --lib qd_`, with RUSTFLAGS choosing the backend;
`hp/data/quad_double/accuracy_n20000.log`;
20 000 inputs per operation and category; max and 99.9th percentile over the six categories other than underflow;
units of u = 2⁻²⁰⁹):

| operation | measure | expected | threshold | max | 99.9% | worst category |
|---|---|---|---|---|---|---|
| accurate add | relative | 0.5 | 1 | 0.083 | 0.015 | cancel |
| sloppy add | / (\|a\| + \|b\|) | 0.25 | 1 | 0.016 (relative: 0.016) | 0.015 | pow2 |
| qd + f64 | / (\|a\| + \|b\|) | ≲ 0.5 | 1 | 0.016 | 0.015 | overflow |
| qd + dd | / (\|a\| + \|b\|) | ≲ 0.5 | 1 | 0.016 | 0.015 | pow2 |
| accurate mul | relative | 0.25 | 1 | 0.016 | 0.015 | wide |
| sloppy mul | relative | ≲ 1 | 2 | 0.38 | 0.18 | overflow |
| qd × f64 | relative | ≲ 1 | 1 | 0.024 | 0.020 | overflow |
| qd × dd | relative | ≲ 1 | 2 | 0.20 | 0.074 | overflow |
| square | relative | ≲ 1 | 2 | 0.15 | 0.10 | computed |
| accurate div | relative | ≲ 1 | 2 | 0.036 | 0.022 | overflow |
| sloppy div | relative | ≲ 2 | 4 | 2.07 | 0.56 | cancel |
| sqrt | relative | ≲ 2 | 4 | 0.38 | 0.19 | wide |
| parse | relative | 0.125 | 0.25 | 0.015 | 0.014 | |
| dot, n = 16 / 128 | / Σ\|aᵢbᵢ\| | 3 / 17 | 3 / 17 | 0.24 / 0.58 | 0.21 / 0.35 | |

- **In decimal digits:** 0.016 u is a relative error of 2×10⁻⁶⁵ and 2.07 u one of 2.5×10⁻⁶³.
- **Where the errors come from.** A result's last part, correctly rounded, contributes up to ulp(x₃)/2. For the
  parts of random numbers that is typically 2⁻²¹⁵ to 2⁻²¹² relative (0.015–0.125 u). The accurate operations stay at
  that level, below the literature's ε_qd by a factor of 3 to 15. The sloppy operations sit 10–100 times higher.
- **Near underflow** (every row passes with the allowance): add and accurate add are exact or 0.01 u; mul, square
  and the divisions are limited by the subnormal floor of their last parts (relative errors up to 2⁻⁵¹, absolute
  within 16·2⁻¹⁰⁷⁴ of the result or residual); sqrt keeps 0.28 u.
- **Decimal I/O:**
  - greedy round-to-nearest of the parse held in 20 000 of 20 000 cases (exact check);
  - 45 000 printed strings (1–100 digits) are all correctly rounded (exact check, and the same as mpmath's `nstr`);
  - 5000 round trips through the exact digits and 5000 through 70 digits give back the same four parts.
  - `to_dd` is within 2⁻¹⁰⁵ relative and `to_f64` within half an ulp in all 20 000 cases.
- **Scalar and 8-lane** agree bitwise, lane by lane, in every operation, category and input (the "lanes" column of
  the tables, 0 everywhere), for the conversions, and for the accumulator.

### Finding 1: QD's accurate add drops half of its last part when all eight parts are read with three emitted

With 20 000 near-cancelling inputs, 2 accurate adds had relative errors of 5.7 and 16.8 u, against the stated 2ε_qd
(0.5 u) (`accuracy_n20000_before_repairs.log`). mpmath confirms both independently. The sloppy add gets these inputs
right. A step-by-step trace of the second case, in a Python transcription of the same operations, shows the
mechanism:

- the leading parts cancel by 2¹⁵ (|a|/|a + b| = 4.1×10⁴);
- the merge emits three parts and then has read all eight;
- QD's branch `x[k] = u; if (k < 3) x[++k] = v;` then stores u and drops v;
- after the cancellation, the accumulator's u has only about 41 significant bits, and v holds the next 12.

This branch (lines 305–310 of `qd_inline.h`) was quoted from the source. `QD::ieee_add` keeps x₃ = u + v in that
branch. `cargo test --release --lib qd_diag -- --ignored --nocapture` (`diag.log`; 200 000 sums a + b = 2⁻ᵏa, by k)
compares a literal copy of QD's algorithm with it:

| cancellation k (bits) | 1–7 | 8–15 | 16–23 | ≥ 24 |
|---|---|---|---|---|
| QD 2.3: max relative error (cases above 1 u) | 1.17 u (1) | 10.1 u (3) | 4.4 u (1) | 0 |
| `QD::ieee_add` | 0.12 u | 0.015 u | 0 | 0 |

A first hypothesis, that the merge's leftover accumulator is dropped when four parts are emitted before all are
read, was tested the same way and changed nothing (`diag_first_design.log`, "remainder kept").

**Addendum (2026-10-22, coordinator): the u + v repair was incomplete.** A later check of the current C++ library
(github.com/BL-highprecision/qd at b527a4b, whose `ieee_add` is unchanged) used sums a = round(x), b = round(−x + d)
with x and d of 424 random bits, |d| ≈ 2⁻ᵏ|x|, k = 1–30, checked against mpmath. QD itself reached 86 u (about 1 sum
in 6 000 above 1 u). Keeping x₃ = u + v still left 1.3, 3.9 and 10.8 u on some of them. The cause: after cancellation
the emitted parts can overlap, so the four slots hold fewer than 212 bits, and v can lie above the last bit of the
result while still falling below the rounding of u + v. The repair is now x₃ = u with v as a fifth term, and the five
are renormalized with `renorm5` (scalar `QD::ieee_add`, and the 8-lane `QV::ieee_add` on the lanes that take that
branch, selected by mask). Over 180 000 such sums the worst error is 0.016 u. `qd_ieee_add_regression` in
`qd_tests.rs` holds 11 of these sums: each must be within 1 u, the 8 lanes must equal the scalar results bit for bit,
and QD's literal algorithm must miss by more than 1 u (the control). The three-backend check passes with a new digest
(`data/quad_double/backends/`), since the repaired add changes some results.

### Finding 2: QD's sqrt loses accuracy near the ends of the exponent range

Every input of the first run's near-overflow sqrt category (a between 2⁹⁹⁰ and 2¹⁰²²) failed, with relative error
2⁻⁵²·⁶: the result was correct to its first part only. The iteration works on r ≈ 1/√a and forms r² ≈ 1/a, whose
lower parts underflow once a exceeds about 2⁸⁶⁰. Near underflow, h = a/2 drops the last bit of subnormal parts (457
of 800 failed). `QD::sqrt` now scales a by an exact power of four, 4⁻ᵏ with 2ᵏ = `pow2_floor`(√a₀), into [1, 4) in
two exact steps, runs QD's three Newton steps and scales back. All categories pass, with ≤ 0.38 u. (These first runs
were on the scratchpad and not kept; the repair's effect is in the overflow and underflow rows of
`accuracy_n20000.log`.)

### Finding 3: the sloppy add stays accurate relative to the result

QD's sloppy add has only a Cray-style bound, error ≤ ε_qd(|a| + |b|), and so may fail relative to a small result.
In none of 140 000 inputs did its relative error exceed 0.016 u: not in near cancellation down to 2⁻²⁶⁰, nor with
operands computed along different routes. Its dropped terms are third-order errors (of the errors of a₂ + b₂ and
a₃ + b₃). For QD-normalized operands, total cancellation forces the parts to pair up almost exactly, so those terms
are zero or tiny. This is numerical evidence over these categories, not a proof, and no counterexample was searched
for beyond them. For a residual, where the error matters relative to the operands, the Cray-style bound is the
relevant one anyway.

### The dot accumulator (`QDotAcc`), and why it has five levels

Each product is expanded into its partial products by order:

- order 0: a₀b₀;
- order 1: a₀b₁, a₁b₀ and the error of a₀b₀;
- order 2: a₀b₂, a₁b₁, a₂b₀ and the errors of the order-1 products;
- order 3: the four O(ε³) products by FMA and the errors of the order-2 products.

The O(ε⁴) products are dropped, as in the sloppy multiply. Level l of the accumulator sums the order-l terms with
two-sums, and their errors go to level l + 1.

The first design had four levels, the last summed in plain f64. At n = 128 it exceeded the pre-registered threshold
(22.8 u against 17, in 1 of 312 sums; `accuracy_n20000_before_repairs.log`). `qd_diag_accumulator` separates the
sources (maxima over 4000·16/n random sums of n products):

| | n = 16 | 128 | 512 |
|---|---|---|---|
| first design, as built (`diag_first_design.log`) | 1.9 u | 16.8 u | 72 u |
| its level 3 replaced by the exact sum of its inputs | 0.042 u | 0.012 u | 0.003 u |
| five levels, every running-sum error by its own two-sum (`diag.log`) | 0.57 u | 0.37 u | 0.88 u |
| the same, its levels summed exactly (no `finish`) | 0.57 u | 0.37 u | 0.88 u |

So the products' truncation costs below 0.12 u, and `finish` costs nothing; the first design's error was level 3's
plain rounding. The two-sum errors of adding into a running sum are relative to that sum, not to the terms, so
level 3 grows to about n^1.5·2⁻¹⁵⁹ Σ|aᵢbᵢ| (2⁻¹⁵³ at n = 128), and each plain addition into it rounds at about 8 u.
Now level 3 is error-free as well, the errors of adding into the running level-2 and level-3 sums enter the next
level by their own two-sums, and level 4 is the only plain sum.

The remaining error, below 1 u up to n = 512, is the plain summation of per-term errors that still carry the
running sums' lower-level errors. Carrying those separately too would cost three more two-sums per term and was not
done.

## 4. Controls

`cargo test --release --lib --features qd_mutate qd_controls` (`controls.log`) breaks one algorithm at a time, in
scalar and 8-lane alike unless marked, and reruns every accuracy, agreement and decimal check (1000 inputs per row).
All 16 fire, each on its own operation's rows:

| # | breakage | rows failing | lane mismatches |
|---|---|---|---|
| 1 | renormalization: last first-pass two-sum in plain f64 | 104 | 0 |
| 2 | sloppy add: error of a₀ + b₀ dropped | 13 | 0 |
| 3 | accurate add: merge takes the smaller part first | 7 | 0 |
| 4 | sloppy mul: error of a₀b₀ dropped | 14 | 0 |
| 5 | sloppy mul: error of a₀b₂ (an O(ε³) term) dropped | 14 | 0 |
| 6 | accurate mul: error of a₀b₁ dropped | 7 | 0 |
| 7 | qd × f64: error of a₁b dropped | 21 | 0 |
| 8 | qd × dd: error of a₀·b.lo dropped | 7 | 0 |
| 9 | qd + f64: error of a₀ + b dropped | 25 | 0 |
| 10 | qd + dd: error of a₁ + b.lo dropped | 7 | 0 |
| 11 | divide: third remainder update skipped | 14 | 0 |
| 12 | sqrt: one Newton step instead of three | 7 | 0 |
| 13 | accumulator: level-2 errors dropped | 20 | 0 |
| 14 | parse: last part truncated instead of rounded | 3 | 0 |
| 15 | 8-lane only: renormalization's zero test ignored (step c2) | 29 | 18 413 |
| 16 | 8-lane only: merge add reads the next part of a one late | 7 | 6354 |

- The two 8-lane-only breakages are caught by the bitwise lane check. No. 16 leaves the accuracy intact (0.015 u),
  which shows why that check is needed.
- The parse truncation is caught by the exact greedy-rounding check and the round trips, not by the 0.25 u threshold.
  Truncating the last part errs by less than 2⁻²¹¹, inside it.
- The independent mpmath check fails, as it must, on a dump made with a breakage:
  - `python3 qd_check.py mpmath --mutate=5`: exit 1, 8482 failures (`mpmath_control_mutate5.log`);
  - `--mutate=14`: exit 1, 470 greedy-rounding violations (`mpmath_control_mutate14.log`).
- The QV-through-model test's control (Δ moved by 10⁻²⁰) fires at 9.7×10⁻²³ against the 10⁻²⁶ threshold.

## 5. Backends

`python3 qd_check.py backends` builds and runs the quad-double tests (4000 inputs per row) with each `v8` backend:

- AVX-512: `-C target-cpu=x86-64-v4`;
- AVX2: `-C target-feature=+avx2,+fma`;
- portable: no target features.

All three pass, and the digest of all scalar results is the same: cc1ea59faf5a7550 (`backends.log`, `backends/`).
Each build also checks its 8-lane results against the scalar ones bit for bit. Since the scalar code is the same IEEE
operations everywhere, all three backends give the same bits. The check writes each backend's result to
`data/quad_double/backends/<isa>.txt`, and a rerun skips the ones already recorded as passed, because the container
restarted twice during this work.

The AVX-512 build uses x86-64-v4 instead of `native`. After a restart the container moved to a host without AVX-512
VBMI/IFMA, and a test binary built with `native` on the earlier host stopped with SIGILL. A bare
`+avx512f,+avx2,+fma` feature list made LLVM fail to select an instruction.

## 6. Benchmarks

`cargo test --release --lib bench_qd -- --ignored --nocapture` measures one thread, in ns per lane and operation, 16
independent vectors per pass (throughput), minimum of 5 timings per row. The figures are the per-row minimum of two
runs (`bench_avx512_run2.log`, `_run3.log`, `bench_avx2_run2.log`, `_run3.log`). The host is an Intel Xeon @ 2.80
GHz with AVX-512 F/BW/CD/DQ/VL; another session's single-threaded job ran on the second core. "a·b + c" is a
multiply then an add; the dot term is a 16-term sum through the accumulator, finished, per term.

| AVX-512 (x86-64-v4) | f64 | double-double | quad-double | qd/dd |
|---|---|---|---|---|
| add | 0.12 | 0.55 | 4.55 | 8.3 |
| mul | 0.11 | 0.28 | 6.20 | 22.5 |
| div | 0.64 | 4.17 | 50.0 | 12.0 |
| a·b + c | 0.11 (FMA) | 0.87 | 14.3 | 16.3 |
| dot term | 0.05 | 0.35 | 5.81 | 16.6 |
| accurate add / accurate mul / accurate div / qd × f64 / sqrt | | | 20.6 / 7.8 / 71 / 4.3 / 144 | |

| AVX2 | f64 | double-double | quad-double | qd/dd |
|---|---|---|---|---|
| add | 0.10 | 0.75 | 8.76 | 11.7 |
| mul | 0.09 | 0.34 | 9.62 | 28.7 |
| div | 0.60 | 4.90 | 66.3 | 13.5 |
| a·b + c | 0.10 (FMA) | 1.40 | 21.1 | 15.0 |
| dot term | 0.08 | 0.55 | 7.79 | 14.2 |
| accurate add / accurate mul / accurate div / qd × f64 / sqrt | | | 45.1 / 13.1 / 88 / 7.9 / 172 | |

So:

- **A quad-double add costs about 8–12 double-double adds, and a multiply about 23–29 double-double multiplies.** The
  double-double multiply is very cheap (7 instructions), while the quad-double one does 6 two-products, 3 three-sums
  and a masked five-part renormalization (about 170 instructions per 8 lanes).
- **In sums of products, the form the model uses most, the ratio is 14–17×.**
- **The masked branches cost.** The accurate add (the merge) is 4.5 times the sloppy one, and AVX2's blend
  instructions cost more than AVX-512's mask registers: quad-double runs 1.3–1.9 times slower on AVX2, against
  1.2–1.6 for double-double.
- **The machine matters by about 20%.** On the host used before the container moved (AVX-512 with VBMI/IFMA,
  `native` builds; `bench_previous_host.log`), quad-double over double-double was 9.8× for add, 27.5× for multiply
  and 13.2× for divide.

## 7. What a quad-double residual would cost

Measured by `FILE=ctx.bin cargo test --release --lib bench_qd_residual -- --ignored --nocapture` (runs 2 and 3:
`residual_M80_run2.log`, `_run3.log`). The state is D = 3.7, M = 80 (`data/lowD_states/lowDv_D3.7000_M80.npz` via
`hp/dump_ctx.py`: kt 36, kc 100, ks 250, 13 nodes, 86 Taylor steps), one thread, AVX-512, same host. Two methods.

**Operation mix** (a counting lane through the model's generic code; 8-lane operations, over all 14 blocks):

| Cauchy-sum products (`acc a b`) | τ-derivative products by D (`acc a c`) | × integer (`x f64`) | adds | × constant | `finish` | multiplies | divides |
|---|---|---|---|---|---|---|---|
| 11.9 M | 5.5 M | 2.0 M | 0.77 M | 0.51 M | 0.49 M | 0.14 M | 1754 |

Weighted by each operation's quad-double cost (run 2), the shares are: Cauchy products 48%, D products 22%, `finish`
11%, × integer 9%, the rest 10%. Counts times costs predict the blocks at 0.062–0.073 s in double-double and
0.78–0.80 s in quad-double (ratio 11–13). That is 18–38% below the measured times, from the memory traffic and
latency a throughput benchmark does not see.

**Direct measurement:** `QV` implements `Lane` (section 8), so the same blocks run with quad-double lanes:

| part (one thread) | double-double | quad-double | |
|---|---|---|---|
| lane work of the 14 blocks (centre, 12 Taylor segments, horizon series) | 0.092–0.100 s | 0.97–1.00 s | measured, 9.8–10.9× |
| of which 251 horizon solves | 0.006 s | 0.006 s | double-double LU in both |
| 251 horizon LU factorizations (80 × 80) | 0.052–0.067 s | the same | kept double-double (section 8) |
| refinement of the horizon solves | | about 0.012 s | a second double-double solve and a quad-double D-product each (estimate) |
| the rest of `residual` | ≤ 0.004 s | about 0.01 s | estimate |
| residual | 0.153–0.156 s | about 1.05–1.08 s (about 7×) | |

With two threads the double-double residual took 0.1 s of the 3.5 s mixed Newton step at M = 80 on the earlier host
(`hp/data/mixed_precision/profile.log`). At 7× the quad-double one would take about 0.7 s, about 17% of a step,
if the Jacobian stays f64.

**At a smaller truncation error the discretization costs more.** The Taylor plan's step is
h ∝ (tol/|c_kt|)^(1/kt), and the series orders kc and ks are set for 10⁻³⁰. These are estimates from that step rule,
not measurements:

- At tol 10⁻⁴², enough for the 2× per unit of D growth of rounding effects from D = 61 to about 100, kt = 48 gives
  about the same steps for 1.6–1.8 times the work per step, a factor of about 2.
- At tol 10⁻⁶², kt = 60–80 minimizes the work at about 4 times.

The series orders kc and ks grow in proportion to the digits; ks also multiplies the number of horizon LUs. So a
quad-double residual is about 7 times a double-double one at the same discretization, and about 15 to 30 times at
10⁻⁴² to 10⁻⁶².

**Cheap savings not taken:**

- the accumulator's `finish` (five qd + f64 additions): 11% of the quad-double lane work;
- the integer weights of `cs_w` multiplied in quad-double: 9%; folding them into the accumulator would avoid this;
- a multiply by a double-double constant with qd × dd instead of qd × qd: 4%.

## 8. Stage 2: readiness for a quad-double residual

**Done.**

- `impl Lane for QV` (in `qv.rs`): `Acc = QDotAcc`, and the model's constants and inputs (`DV`) enter exactly with
  `QV::from_dv`.
- `qd_residual::qv_lane_blocks_agree` runs every block of a small synthetic context (M = 8, nodes 0.005–0.1, ks 30)
  through `model::block_state::<QV, _>`: the centre series, the Taylor stepping and the horizon series with its
  periodic solves. QV agrees with DV to 5.8×10⁻³²; the check is 10⁻²⁶, and Δ moved by 10⁻²⁰ gives 9.7×10⁻²³. It runs
  in `cargo test` and on every backend.
- `bench_qd_residual` runs the real M = 80 context (section 7).
- No solver or Python function uses it.

What this instantiation is not: a quad-double residual. The trait passes every constant as double-double (`cst`,
`mulc`, `addc`, `accc` take `DV`; `splat`, `muls` take `DD`); `val()` rounds back to double-double; the horizon
solves are double-double (`DDLu`). So it computes in quad-double with double-double data. Its cost is that of a real
quad-double residual, since the constants' zero lower parts are not exploited.

**What a quad-double residual needs**, with my estimate of the work for an agent session:

1. **A scalar type in the model** (1.5–2 days).
   - About 70 lines of `model.rs` compute constants in `DD`: n − 1, n/2, 1/k, 1/x₀, x₀², the centre series' 2 × 2
     inverse, 2π/Δ, the step h, u₀, κ.
   - `Lane` gets an associated scalar type (`DD` for the existing lanes, `QD` for `QV`) and a constant vector type
     (`DV` or `QV`). `cst`, `mulc`, `addc`, `accc`, `splat`, `muls`, `val` and `from_vt` use them. A small `Scalar`
     trait over `DD` and `QD` (+ − × ÷, `mul_f`, `from_f64`, `ONE`, `TWO_PI`) lets the model's functions be written
     once.
   - The existing lanes keep `DD` and `DV`, so they perform the same operations, and `mixed_check.py ref` guards
     their bits.
2. **Constants in quad-double** (1 day).
   - `Ctx` needs n, the phase weights and λ in quad-double.
   - `DMat` needs quad-double columns. At M = 80 that is 205 KB against 102 KB in double-double, still L2-resident.
     `Lane::dmat_apply` for `QV` would use them.
   - Python computes them as it does now (`dmat_odd`, `phase_weights`, n = D − 2), at 80 digits instead of 40, split
     into four parts.
   - `field::DMat::from_rows` and the freeze projectors take four arrays.
3. **The horizon solves: double-double LU plus quad-double refinement, not a quad-double LU** (1 day).
   - Each periodic solve (ωD + diag κ)f = r uses the double-double LU already factored, then one or two refinement
     steps f += LU⁻¹(r − (ωD + κ)f), with the residual formed in quad-double by `dmat_apply`.
   - A double-double LU gives about 32 digits times the condition number per step. The pivots here range over
     10⁻³–10², so two steps reach quad-double.
   - Cost: about 0.012 s at M = 80, against 0.7–1.1 s for quad-double LUs of the 251 matrices at the measured
     ratios, which would also need a new blocked LU.
   - For the tangent lanes nothing changes.
4. **The residual function and the state** (1 day).
   - `residual_qd(z0, z1, z2, z3) -> (F0, F1, F2, F3)`: the blocks with `QV`, the targets read as four parts, the
     phase row in quad-double, the frozen-harmonic projectors in quad-double.
   - `Fl<QV, B>::load/store` for four arrays.
5. **Python** (1–2 days).
   - `Solver.residual_qd`, and the state as four arrays (`z0..z3` in the `.npz`; double-double states load with
     z2 = z3 = 0).
   - Driver helpers `qd_from_mp`, `qd_add`, `qd_norm`.
   - A Newton loop for quad-double: chord steps with the f64 Jacobian's LU from the double-double solve, dz from F
     rounded to f64, and z ← z − dz in quad-double.
   - With the measured contraction of about 10⁻⁷ per step (notes/MIXED_PRECISION.md), going from |F| ≈ 10⁻³⁰ to
     10⁻⁶⁰ takes about five chord steps. Each is one quad-double residual and one f64 solve, about 3.5 s at M = 80 on
     two threads at the same orders.
6. **Tests** (1–2 days).
   - At a converged double-double state, the quad-double residual must equal the double-double one to 10⁻³⁰ and be
     computable to 10⁻⁵⁰ and below.
   - A planted solution, a state built by integrating with known data, must give |F| at the quad-double floor.
   - Convergence at M = 16 to |F| ≈ 10⁻⁶⁰.
   - The truncation orders (kt, kc, ks, plan tolerance) for 10⁻⁴² and 10⁻⁶², with the cost of section 7.

Total about 6–9 agent-days. The predicted residual cost is about 7 times the double-double one at the same
discretization (about 1.05 s on one thread at M = 80, 0.7 s on two). At the orders needed for a 10⁻⁴²–10⁻⁶²
truncation error it is 2–4 times that. The Jacobian stays f64.

## 9. Existing outputs are bit-identical

- `python3 mixed_check.py ref --save` with main's build (f91dafe, built privately in the scratchpad; the system
  `hpkernel` was not touched), then `--compare` with this branch's final build: all 17 arrays bit-identical.
  - Covered: the residual and the Jacobian, the Newton step and its pivots, the chord step, `lu_solve_t`,
    `mode_matrix` at λ = 0.3, and the frozen-harmonic and inward-node Jacobians at D = 3.7, M = 48.
  - Logs: `ref_save_main.log`, `ref_compare.log`.
- `hp/proto/test_dd.py` gives 1.24, 1.28 and 1.64×10⁻³², and 9.8×10⁻³¹ for `dd_solve` (`proto_test_dd.log`), the
  same digits as `hp/data/mixed_precision/proto_test_dd.log`.
- `cargo test --release` passes 8 tests, the three existing LU tests among them, and ignores 7 benches and
  diagnostics (`cargo_test.log`).
- Changes to existing files: additions only in `v8.rs` (masks, `sqrt`, `pow2_floor`) and `lib.rs` (modules), and the
  `qd_mutate` feature in `Cargo.toml`. No existing function changed.

## 10. Proposed README text (for the coordinator)

In "The double-double solver" list:

> - **Quad-double arithmetic (not yet used by the solver):** `hp/src/qd.rs` and `qv.rs` add quad-double numbers
>   (four f64, about 64 digits), scalar and 8-lane SIMD with bitwise the same results on AVX-512, AVX2 and portable
>   builds. They use the QD library's algorithms[^qd], with two repairs found by the tests (the accurate add's last
>   part, the square root near the ends of the exponent range). Against exact arithmetic they err by at most
>   2.1×2⁻²⁰⁹ (63 digits). They cost 8–29 times double-double per operation and 14–17 times in sums of products. A
>   quad-double residual would cost about 7 times the double-double one at the same orders
>   ([notes/QUAD_DOUBLE.md](notes/QUAD_DOUBLE.md); `hp/qd_check.py`).

[^qd]: Y. Hida, X. S. Li, D. H. Bailey, *Algorithms for quad-double precision floating point arithmetic*, ARITH-15
(2001).

In the Layout block under `hp/`:

```
├── src/qd.rs, qv.rs  quad-double: scalar and 8-lane (not used by the solver yet)
├── qd_check.py     checks of the quad-double arithmetic against mpmath, and across the SIMD backends
```

## 12. Plan: a fully quad-double solver past D ≈ 73 (2026-10-27, the author's choice)

**Why.** The pinned double-double continuation (T72–T80, `raw/T73`, `raw/T75`, `raw/T80`) ends near D = 73.3: steps of
6×10⁻⁴ in D stall at |F| ≈ 10⁻¹⁹ against a tolerance of 1.1×10⁻²⁰, and the two pin choices' corrected Δ drift apart
(10⁻¹⁶ at D = 71, 8.6×10⁻¹⁵ at 73.1). The full system's u·κ is 10³ there (README, "Why the solver stalls"): the
double-double Jacobian's rounding, u·σ_max ≈ 10⁻²¹, exceeds its weakest singular value, 3×10⁻²⁴, so no double-double
Jacobian or LU resolves that direction. Done on `wip/qd-horizon` (be10434, 37468e6): `residual_qd`, the residual with
every block in quad-double (kernel test `residual_qd_agrees`), `QdSolver` (Newton–GCR on it, preconditioned by the
double-double LU) and `hp/full_step_check.py`. At D = 72.80 one such step with 60 GCR products (not converged) raised
|F| from 6×10⁻²² to 10⁻¹⁷, and Newton with line search made no progress (`raw/T76` 7ca90fe, ab929a7).

**First, the deciding experiment.** The same step with GCR converged (up to 300 products):
- if it lowers |F|, the linear solve was the limit: Newton–GCR on `residual_qd` is the solver, no quad-double Jacobian;
- if it still raises |F|, the pinned state lies outside Newton's basin along the weak directions (basin 2σ²/c), and
  the solver must release the pinned harmonics one at a time in quad-double (pin 13 → 11 → 9 → 7 → none), each
  release solved from the last.

**Then, if a quad-double Jacobian is needed** (in this order; arithmetic in multidouble first, per AGENTS.md):
1. multidouble: a forward-mode dual over quad-double lanes (as `dual::Du` over double-double) and a quad-double dense
   LU with solve and transpose solve (as `linalg`'s double-double LU); tests and mutation controls; a release.
2. hp: `jacobian_lanes::<QDu, B>` for M = 32 and 48, a binding, and the Newton step on it; the bit-identity checks
   (`highm_check.py ref` and `scaled`) for the unchanged paths.
3. The driver: a quad-double Newton (deflated as `deflated_newton`) and `scan_hp --solver qd`; the stall detector's
   u·κ with u = 2⁻²⁰⁸ to check that the weakest direction is resolved (u·κ ≪ 1).
4. A pilot from T75's last states, then the run toward D = 100 on `raw/<task>`.

### 12.1 Built (2026-10-27, feat/qd-solver; raw/T81)

The arithmetic is multidouble 1.7.0 (PRs #23 and #24): `Qu`, a quad-double value and tangent, and `QDLu`, a dense
quad-double LU with partial pivoting. In hp, `qsolve.rs` holds the residual with every block in quad-double (the
horizon solves refined to quad-double, value and tangent together), its exact Jacobian by `Qu` lanes and the Newton
step, for M = 32 and 48. `driver.newton_qd` is Newton on it, `scan_qd.py` the continuation in D, and `qd_planted.py`
the planted-root check.

**Precision tests.** The kernel tests (`cargo test --release --lib qsolve_`) run on the synthetic context.

| check | result | control |
|---|---|---|
| quad-double residual and Jacobian against the double-double ones | 5.5e-32, 5.9e-31 | |
| Jacobian against central differences (h = 2^-70) of the quad-double residual | 1.5e-41 | horizon solves unrefined: 6.6e-4 |
| two refinement steps against four | 3.3e-63 | |
| planted root from 1e-6 | 1e-6, 6e-8, 1e-14, 4e-28, 3e-55, 4e-62 | |
| a change of 2^-180 seen (J d to 1.6e-11) and undone (8.9e-8 of it left) | pass | state read in double-double: 1.0, 1.0 |

`qd_planted.py` at T75's D = 72.8 state (full system, kfreeze 13) solves G(z) = F(z) - F(z*), whose root is the state.
The pass criterion is at eps = 1e-40.

| start eps | quad-double max \|z - z*\| | \|Delta - Delta*\| | control (state in dd) | double-double \|Delta - Delta*\| |
|---|---|---|---|---|
| 1e-40 | 8.0e-42 | 1.0e-48 | 3.0e-27 | 2.2e-16 |
| 1e-36 | 2.1e-42 | 1.0e-49 | 3.0e-27 | 6.2e-16 |
| 1e-32 | 1.3e-41 | 2.4e-48 | 4.6e-27 | 1.2e-16 |
| 1e-28 | 3.8e-42 | 7.1e-49 | 5.5e-27 | 5.1e-17 |

Started at the root, the double-double Newton drifts to Delta errors near 1e-16. This is the 1e-16 A-B spread of
T75 at D = 71, so that spread is the double-double limit.

**Newton's basin.** A first run from eps = 1e-24 failed the criterion. Every solver, double-double included, took the
same first step to 2.3e-11. The cause is the nonlinearity, not precision. Newton's quadratic constant is
K = 2.3e-11 / (1e-24)^2, about 2e37, so the radius of quadratic convergence is near 4e-38. That is below
double-double's resolution of the state (about 1e-31 of its entries). A double-double iteration cannot enter it, which
is the large-D stall. A quad-double one can.

**The deciding experiment of section 12 was flawed.** `full_step_check.py` loaded only the state's first two parts, so
it started from the double-double truncation. Its quad-double residual was 6.4e-22. The state itself, with all four
parts, has a full-system residual of 1.6e-29.

**Step acceptance.** From that state the full-system Newton step has |dz| = 4.5e-9, mostly p1's harmonics 13, 11
and 9. Along it |F(z + lam dz)| = (1 - lam) 1.6e-29 + 5.2e-18 lam^2 exactly, so a decrease of max |F| accepts no step.
Deuflhard's natural monotonicity test with the old Jacobian rejects the full step too (ratio 748). The weak direction
turns between iterates. Full steps converge in four: 1.6e-29, 5.2e-18, 1.3e-31, 8.9e-32, 2.3e-53. `newton_qd`
therefore accepts a trial when the next correction, with its own Jacobian, is smaller.

**Convergence of T75's states in quad-double (full system).**

| state | result | Delta |
|---|---|---|
| pin 7, D = 72.8 | 2.3e-53 in 5 steps | 1.7816051756032902 - 4.7345207382657e-17 |
| pin 5, D = 72.8 | stalled at 8.2e-18 | (1.7816051756032902 - 4.7300989e-17) |
| pin 5, D = 73.0 | 2.5e-53 in 7 steps | 1.7812396819309262 + 3.9940754733380e-18 |
| pin 7, D = 73.0 | stalled at 7.8e-19 | (1.7812396819309262 + 3.993650e-18) |

The stalled runs halve |F| and |dz| at lam = 1/2 each step while the smallest pivot falls. That is Newton's
behaviour at a nearly double root: along the weak direction the residual is close to a perfect square. The converged
solutions have the smallest pivots, 2.3e-16 and 2.9e-16 against 1e-9 for the largest.

**Taylor plan.** The plan (Solver.plan, tol_step 1e-32) is computed from the state. Replanning at the converged
D = 72.8 state moved Delta by 9.1e-28, so with that tolerance Delta is defined to about 1e-27. `scan_qd.py` keeps a
plan until D moves --replan (0.5).

**Bit-identity of the bump from multidouble 1.5.1 to 1.7.0.** `highm_check.py ref` and `scaled` were saved with a
build of main (6014cb1, 1.5.1) and compared with a build of this branch (086a95e, 1.7.0), on the Mac. All 168 `ref`
outputs (M = 8 to 128) and all 22 `scaled` outputs (M = 192 and 256) are bit-identical, and the `scaled` control
(the scale wrong at order 7) fires. The logs and hashes are on raw/T81 (fd8e038).

### 12.2 What stops the continuation past D = 73: the horizon block (2026-10-27, raw/T81)

The quad-double solutions at D = 72.8 and 73.0 lie on one smooth branch. p1's harmonics 9, 11 and 13 are -1.707234e-6,
2.659647e-7 and -4.136447e-8 at D = 72.8, and -1.700991e-6, 2.646266e-7 and -4.109847e-8 at D = 73.0, with a geometric
ratio of 6.4. Continuing it fails at once. Every block but the horizon block responds to D linearly, about 4e-5 per
unit of D at the D = 72.8 state; the horizon block's residual explodes, and more horizon orders make it worse.

| horizon orders ks | horizon block at D + 1e-6 | at D + 1e-4 |
|---|---|---|
| 130 | 9.6e3 | 3.8e15 |
| 160 | 1.5e5 | 2.6e19 |
| 200 | 4.8e6 | 2.9e24 |

At the solution the horizon series converges cleanly (terms 1e-45 at order 130, 2e-70 at order 200). Off it they grow
about 1.34-fold per order. That suggests a radius near 0.22 for generic horizon data at D = 73, against the matching
distance u0 = 0.3; the solution's own radius is 0.75 at D = 60 (BEYOND_LIGHT_CONE section 2.3). Even linear
interpolation between the two solutions, at D = 72.9, leaves the horizon block at 4.6e18 with every other block near
1e-8. This one structure gives the quadratic constant of about 2e37 (12.1), the response of the residual to D like
dD^6.7, and the nearly double roots where Newton halves. A finite-difference tangent in D is meaningless here: it
scales like h^2, its truncation error. These are measurements; the mechanism (a radius for generic data below u0, or a
nearly singular horizon solve amplifying the deviation) is not established.

The candidate remedy is to match closer to the horizon, with the last node at x = 0.75 or 0.8 (u0 = 0.25 or 0.2), so
that the series converges for data near the solution. That changes the discrete system, so Delta must agree between
layouts to the quoted precision before it is used.

### 12.3 Past D = 73: the layout, single jumps, and the price (2026-10-27; folders T81, T84 and T85 of the raw-data repository)

**The quad-double Newton at T75's frontier (old layout, last node 0.7).** T75's states at D 73.10 (both runs),
73.2657 (pin 7) and 73.1211 (pin 5) all stall the way the stalled 72.8 and 73.0 runs did: |F| and the correction
halve at lam = 1/2 each step while the smallest pivot falls, to 7e-18, 5e-16, 7e-17 and 1e-16 (T81, frontier.log).
Precision does not step past 73 in this layout.

**One node added beyond 0.7** (T81, layout_test.py: the D 73.0 quad-double solution re-noded, newton_qd at D 73.0,
then the blocks' response to D with the state held, then Newton at D + dD).

| last node | newton_qd at D 73 | Delta - the 0.7 layout's | horizon block at dD 1e-4 | at dD 1e-2 |
|---|---|---|---|---|
| 0.7 | converged (12.1) | | 3.8e15 | |
| 0.75 | 1.5e-55 in 5 steps | +4.3e-17 | 6.0e5 | 8.6e21 |
| 0.8 | stalls at 4.5e-18 (lam 1/2 halving) | +1.3e-13 | 1.9e-2 | 4.3e9 |
| 0.85 | stops at 7.0e-14 (the first step moves Delta 1.3e-9) | +9.7e-9 | 2.8e-4 | 2.8e-2 |
| 0.9 | wanders (|F| 2.6e-3 after 7 steps; stopped) | | | |

At 0.85 (u0 = 0.15) the horizon block is linear in dD out to 1e-2, with ks 130 and 200 alike, as the radius
estimate of 12.2 predicts; at 0.75 it is not. But past 0.75 the Newton no longer converges: its first step leaves
along one direction with Delta content, which is nearly singular in these layouts (below). Newton at D + 1e-3 fails
from every layout.

**The pinned double-double continuation in the new layouts** (T84, scan_hp of wip/qd-horizon dd0186e with the pin of
T73-T80, steps of 0.005 from T75's frontier): it converges points at 73.27-73.28 in both layouts, but the full
system's sensitivity dDelta/dF there is 1e22-1e25 (T75: 1.3e11) and the linearised full-system correction
dDelta_pin is 2e-8 to 1.6e-7 (T75: 1e-12); a third 0.85 point at 73.2832 lies 5e-3 off. Delta is undetermined at
double-double precision in these layouts. The pilot was stopped (its criterion was |dDelta_pin| <= 1e-10).

**Single jumps with the state held** (T84, jump_resid.py: the full-system residual at D + dD, the converged state
unchanged):

| dD | last node 0.7 | last node 0.85 |
|---|---|---|
| 0.05 | 2e39 | 14 |
| 0.2 | 4e47 | 1e9 |
| 1 | 1e59 | 2e20 |
| 5 | 1e76 | 1e37 |

The horizon series' radius shrinks with the deviation from the solution: at 0.85 the block is linear for deviations
of about 1e-6 (dD 1e-2 at the state's 4e-5 per unit of D) and exponential beyond. No jump lands from the state held,
in either layout, and T68's jumps of 2.3 and 4.3 from D 67.72 (|F| 1.7e17 and 1.7e27) were the same thing.

**Single jumps with a predictor through wide-spaced points** (T84, run_jump.sh): the pinned scan in the 0.85 layout
from the frontier 73.2657, the cubic predictor through T75's waypoints 70.65, 71.5 and 72.4 and the frontier (spacing
0.87), one step of 0.5, 1, 2 or 5. All four converge. Delta_full1 is the pinned Delta plus the linearised
full-system correction (T75's quantity); the rational formula's value is a plausibility check, not a reference
(T75's own Delta_full1 lies 5.0e-8 to 6.2e-8 above it).

| jump | D | \|F\| | steps | pinned Delta - formula | Delta_full1 - formula |
|---|---|---|---|---|---|
| 0.5 | 73.7657 | 1.3e-27 | 39 | +1.1e-6 | -6.6e-7 |
| 1 | 74.2657 | 5.8e-21 | 6 | -6.6e-8 | -1.0e-6 |
| 2 | 75.2657 | 3.1e-28 | 13 | +2.1e-5 | -9e-9 |
| 5 | 78.2657 | 1.6e-27 | 10 | +4.3e-4 | +4.8e-7 |

So the steps land on the branch when the predictor's error is below the horizon series' tolerance, which the new
layout raises from about 1e-13 (0.7) to about 1e-7 (0.85), and the predictor through points 0.87 apart stays below
it over a step of 5. In the 0.8 layout the pilot's three points have Delta_full1 - formula = +4.9e-8, +4.9e-8,
+4.7e-8, on T75's trend, so that layout may hold Delta better. T85 runs both layouts to D = 200 in steps of 1 on
shared waypoints; their difference per waypoint is the uncertainty of Delta_full1 there.

**The price, and the quad-double solver's part.** Both new layouts make one direction of the full system nearly
singular (the sensitivities above). Double-double leaves Delta there at 1e-6 to 1e-8. Quad-double, with residuals of
1e-50, can resolve a direction of singular value 1e-20 to about 1e-30 in Delta, but the plain quad-double Newton
leaves along it (the 0.85 row of the layout table). `driver.newton_qd` therefore now has the double-double solver's
deflation (`w > 0`: the step and the corrections it is judged by are projected off the w weakest right singular
directions of the quad-double Jacobian, `weak_subspace_q`, then plain Newton finishes; `junk=True`: only the weak
directions without Delta content are frozen, `junk_subspace_q`, for the whole iteration). Its tests on T84's states,
with the undeflated Newton as the control, are in T84-mac/qd_deflated*.

**What the quad-double Newton found at D 73.27 (T84-mac, qd_*.py; 2026-10-27 evening).** Every variant stalls, on
every system and layout tried:

| system, layout | start |F| | ends at | behaviour |
|---|---|---|---|
| full (kfreeze 13), last node 0.85: deflated, junk frozen, plain | 1.5e-23 | 4.7e-14, 1.8e-14, 3.2e-17 | step halving at lam 1/2 |
| full, last node 0.8: deflated, plain | 6.9e-26 | 4.6e-16, 8.1e-17 | the same |
| harmonics above 7 zeroed (kfreeze 7), 0.7 and 0.8 | 6e-6, 3e-5 | 3e-6, 1.9e-11 | lam 1/32; halving |
| the pinned system of T73-T85 (`PinnedQ`, T84-mac/qd_pinned.py), 0.7 and 0.8 | 3.5e-23, 6.9e-26 | 7.4e-18, 4.6e-17 | the first step raises |F| by 5-10 orders |

The arithmetic is not at fault: at the 0.8 state the quad-double LU's plain and transposed solves and the Newton
step satisfy their equations to the quad-double roundoff of |J| |x| (python-flint at 320 bits, qd_luexact.py:
relative residuals 9e-36, 2e-29, 2e-33 with |x| 1.5e29 for a unit right-hand side), and the Jacobian matches finite
differences (qd_jvcheck.py). That |J^-1 b| is 1e29 says the full system's smallest singular value is about 1e-29
there. Inverse iteration on the LU (qd_weak.py) finds twelve directions with sigma 1e-29 to 1e-24, p1's harmonics 13,
11, 9, 7, 5, 3 (cos/sin pairs), with Delta content 1e-5 to 1e-2: in the layouts with the last node at 0.8-0.85
the junk extends down to harmonic 3 (in the 0.7 layout it was 9-13). Two further directions with Delta content
0.6-0.8 at sigma 2e-22 and 5e-22 were f64 Rayleigh-Ritz noise on matrices with entries 1e58 (|J v| = 1.4 along one,
equal to the finite difference, qd_scan1d.py and qd_jvcheck.py) and are withdrawn.

Why no variant converges: the pinned system's step of 1.3e-12 from |F| 3.5e-23 lands at |F| 7.4e-18, so the
residual's second-order term is about 1e7 |dz|^2 (the horizon block's exponential sensitivity, 12.2), while the
pinned Jacobian's smallest pivot in quad-double is 9e-12. Newton converges only where c |J^-1 F|^2 < |F|, i.e. |F|
below about sigma^2 / c = 1e-29 here, and the best start any double-double run can give is its floor, 1e-21 to 1e-26.
That is also what the double-double floor of T73-T80, rising 1.45-fold per unit of D, was: not roundoff but the edge
of the basin. Precision, deflation and the pin do not move that edge; only a discretisation with a smaller c (the
horizon block) or a larger sigma (the junk) does.

**The inward match** (the kernel's `match_node`: segments beyond it integrate inward from the horizon side; used
before only at D = 3.7). T84-mac/qd_inward.py: the frontier state re-noded into 0.3..0.7 plus 0.85 and 0.95, matched
at 0.7, the series summed at u = 0.05. Before any step the two inward segments' residuals are 0.5 and 11 (the node
states are the old series' values, f64-rounded by renode, and the inward integration amplifies that by about 1e17
over 0.1 in x), they respond to dD = 1e-8 by rising to 34 and 47, and the Newton step is 2.6e4 with no trial accepted.
Integrating away from the horizon brings back the growing mode the series suppresses, about 1e9 per unit of D at
D = 73, so the two-radius form of the horizon treatment is out at large D as well.

### 12.4 Newton's basin and the horizon series' radius against D (2026-10-27 evening, T84-mac/qd_basin.py, horizon_radius.py)

**The basin.** At the saved pinned states of T73 and T75 (the 0.7 layout), for the pinned system (kfreeze 7 with the
band, `PinnedQ`) and the full system (kfreeze 13) in quad-double: the LU's smallest pivot sigma, the Newton step
|dz|, the residual after the full step and c = |F(z + dz)| / |dz|^2, and the basin sigma^2 / c that |F| must be inside
for Newton to converge quadratically (the double-double polish of a converged point leaves |F| at the floor column).

| D | system | sigma | c | basin sigma^2/c | the state's |F| | inside |
|---|---|---|---|---|---|---|
| 67.74 | pinned | 1.6e-11 | 4e4 | 6e-27 | 1.3e-30 | yes |
| 70.5 | pinned | 1.2e-11 | 1.4e5 | 1e-27 | 2.6e-30 | yes |
| 72.4 | pinned | 1.3e-11 | 4e5 | 4e-28 | 3.1e-30 | yes |
| 73.27 | pinned | 9.2e-12 | 4e6 | 2e-29 | 3.5e-23 (the frontier, never polished) | no |
| 67.74 | full | 1.3e-15 | 0.07 | 2e-29 | 9e-28 | no |
| 68.5 | full | 8.6e-16 | 0.03 | 2e-29 | 6e-31 | yes |
| 70.5 | full | 8.4e-16 | 0.08 | 9e-30 | 3e-30 | yes |
| 71.5 | full | 3.8e-16 | 0.14 | 1e-30 | 1.4e-30 | marginal |
| 72.4 | full | 4.9e-16 | 0.19 | 1.3e-30 | 3e-30 | no |
| 73.0 | full | 2.9e-16 | 0.28 | 3e-31 | 1e-29 | no |
| 73.27 | full | 2.3e-16 | 0.31 | 1.7e-31 | 3.5e-23 | no |

The full system's basin halves every 0.8 in D (sigma falls 0.73-fold and c grows 1.3-fold per unit) and the
states' floors cross it near D 71: the quad-double Newton converged the 72.8 and 73.0 states from two starts of
four (12.1), and fails at the frontier. The pinned system's basin is 2-3 orders wider and its polished states sit
inside it: from the 72.4 state the pinned quad-double Newton (`PinnedQ`, qd_pinned.py) converges in two full steps,
3.1e-30, 5.9e-33, 3.4e-53, Delta 1.78234204349234227 + 4.615237e-17 + 2.3e-34, the first quad-double solution of the
pinned system past D 70. From the frontier state (3.5e-23, outside) it fails, and a backtracking line search on
|F|_2 (newton_qd ls="F") creeps by 1e-4 of |F| per step, as the quadratic model predicts (lam about F0 / (c |dz|^2)).
So the recipe that works is the double-double deflated Newton and its polish to the floor, then quad-double from the
floor, and its reach is where the double-double floor still falls inside the basin.

**The pin ladder at the frontier** (qd_pinned.py with the pin at 1, 3, 5, 7: p1's harmonics above the pin held at
the double-double solution's values, the band subtracted as PinnedSolver does, the solver built with kfreeze = pin):

| pin (harmonics held) | result | Delta |
|---|---|---|
| 1 (3-13) | converged, 2.3e-53 in 3 steps | 1.78075708073508832 - 1.539722e-17 + 2.7e-34 |
| 3 (5-13) | converged, 3.4e-53 in 5 steps | 1.78075708073509054 + 1.341929e-17 - 6.1e-35 |
| 5 (7-13) | stalls at 3.9e-18 after one step | |
| 7 (9-13) | stalls at 7.4e-18 after one step | |

At D 73.27 the quad-double Newton converges once harmonics 5 and up are held, and not with 5 or 7 free: those two
are junk there as well (their directions' sigma against c). The two converged Delta differ by 3e-17, the cost of
holding harmonic 3 at its double-double value; the full-system correction from the pinned solution, in
quad-double, is the next quantity (qd_fullcorr.py).

Freezing p1's harmonics 3-13 or 9-13 as bare band fields (qd_freezeh.py) does not deflate: the first projected step
of 5e-7 leaves a residual of 5e-7, identical for both sets, because a bare harmonic is not a weak direction of J
(the junk directions mix p1 with the other centre field, qd_weak.py's masses), so the projection breaks the step
instead of removing its weak part. The kernel's pin removes those unknowns and their rows, which is the reduction
done right.

**The horizon series' radius** (horizon_radius.py: the growth of the terms |c_k| u0^k over the last 40 orders,
R = u0 / growth, at the state held while D is moved by dD, i.e. for data about 4e-5 dD off the solution; a
different estimator from BEYOND_LIGHT_CONE 2.3's root test, which gives larger on-solution radii):

| D | R on the solution | R at dD 1e-8 | 1e-6 | 1e-4 | 1e-2 |
|---|---|---|---|---|---|
| 40.02 | 0.73 | 0.50 | 0.43 | 0.36 | 0.26 |
| 45.05 | 0.70 | 0.46 | 0.40 | 0.33 | 0.24 |
| 50.01 | 0.68 | 0.43 | 0.37 | 0.30 | 0.22 |
| 55.04 | 0.62 | 0.40 | 0.34 | 0.28 | 0.21 |
| 60.00 | 0.59 | 0.38 | 0.32 | 0.26 | 0.19 |
| 67.74 | 0.61 | 0.34 | 0.29 | 0.24 | 0.18 |
| 70.5 | 0.62 | 0.33 | 0.28 | 0.23 | 0.17 |
| 72.4 | 0.62 | 0.32 | 0.27 | 0.22 | 0.17 |
| 73.27 | 0.56 | 0.32 | 0.27 | 0.22 | 0.17 |

The matching node is at u0 = 0.30. For data a given distance off the solution the radius falls through u0 at
D about 50 for dD 1e-4, 63 for 1e-6 (the double-double stalls at 61 and 67.8) and, extrapolating the 0.004 per unit
of D, about 78 for 1e-8, the predictor error of T75's steps. Each decade of predictor accuracy buys about 15 in D,
so the series' wall moves and is not the binding constraint at 73; the basin above is, and its c is the series'
sensitivity near its radius. A horizon discretisation without a radius (collocation in u) removes both.
