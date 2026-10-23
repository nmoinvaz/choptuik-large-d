# Quad-double arithmetic in the Rust kernel (`hp/`)

Quad-double numbers (an unevaluated sum of four f64, about 64 significant digits) are now in the `hpkernel` crate:
a scalar type `QD` and an 8-lane SIMD type `QV` with bitwise the same results, on the AVX-512, AVX2 and portable
backends of `v8.rs`. Nothing in the solver uses them yet. Every existing output is bit-identical (section 9).

**Summary.**

- **Accuracy** against exact arithmetic (and, independently, mpmath): every operation is within its threshold in all
  seven input categories. The thresholds were written before the runs (section 2).
  - The accurate operations (accurate add, accurate mul, qd × f64, accurate div) err by at most 0.083 u.
  - The library's defaults (sloppy add, sloppy mul, sloppy div) and sqrt err by at most 0.016, 0.38, 2.1 and 0.38 u.

  Here u = 2⁻²⁰⁹ ≈ 1.2×10⁻⁶³, so 2.1 u is 63 correct digits and 0.02 u is 65. The literature's ε_qd is 0.25 u
  (section 3).
- **Three findings in the QD library's algorithms**, each confirmed independently and changed here:
  - The accurate (IEEE-style) add drops half of its last part in one branch: 16.8 u in 2 of 20 000 near-cancelling
    sums, against the stated 0.5 u. It is repaired with a one-line change, after which the maximum is 0.12 u in
    200 000 sums.
  - The square root loses all but 53 bits above about 2⁸⁶⁰ and below about 2⁻⁸⁶⁰. Its operand is now scaled by an
    exact power of four.
  - The sloppy add, whose stated bound is only Cray-style, never exceeded 0.016 u relative to the result, even on
    near-cancelling and computed operands.
- **Cost** (one thread, ns per lane, AVX-512; section 6):
  - quad-double over double-double: add 9.8×, mul 27×, div 13×, a·b + c 14×, a dot-product term 11.5×;
  - AVX2 runs quad-double 1.5–1.9 times slower than AVX-512.
- **A quad-double residual** at D = 3.7, M = 80 (section 7):
  - Its lane work measures 9.0 times the double-double one. Running the model's own generic code with `QV` lanes
    takes 0.74 s against 0.083 s on one thread, and the operation-count model predicts 9.7×.
  - With the horizon LUs kept in double-double and refined in quad-double, a residual would take about 0.85 s
    against 0.15 s on one thread, 5.6× at the same discretization.
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
costs about as much as the sloppy one; for quad-double the accurate add costs 5 times the sloppy one, section 6.)

**How the branches run on 8 lanes.** QD's renormalization skips zero error terms, and its accurate add merges the
eight parts by magnitude. Both branch on the data, and both are emulated exactly with masks:

- The renormalization's second pass, written with a slot index k (the part being filled), is
  (t, e) = quick_two_sum(s[k], c); s[k] = t; s[k+1] = e; k += (e ≠ 0). The last step at k = 3 keeps t only (QD's
  `s3 += c4`). k is held as threshold masks, and the slots are updated by selects.
- The merge add keeps per-lane read positions i, j and output position k. It reads a[i] and b[j] by selects, freezes
  lanes that have finished, runs at most seven steps, and stops once all lanes are done.

**Departures from QD, each for a reason found or stated here:**

1. The accurate add keeps u + v where QD keeps u (section 3).
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
  AVX2 and runs the renormalization's branches as a scalar loop over the lanes. It reports AVX2 QD add 4× and
  multiply 2× faster than scalar QD.

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
| sloppy mul, qd × dd | ≲ 1 u relative | 2 u |
| accurate mul, qd × f64 | 0.25 u, ≲ 1 u | 1 u |
| square | (written later with the sloppy mul's) | 2 u |
| sloppy div | ≲ 2 u relative | 4 u |
| accurate div | ≲ 1 u relative | 2 u |
| sqrt | ≲ 2 u relative | 4 u |
| parse | ≤ 0.125 u relative | 0.25 u |
| dot accumulator, n terms | ≲ (8 + n)/8 u × Σ\|aᵢbᵢ\| | (8 + n)/8 u |

Near underflow an absolute allowance of 16·2⁻¹⁰⁷⁴ was planned.

**Changed after the first runs** (each reported, none to make a failure pass):

- The near-underflow allowance is now defined on the quantity each test checks. For add and mul that is the result.
  For division it is the residual r·b − a: 16·2⁻¹⁰⁷⁴·(1 + |b|), because the quotient's parts cannot go below 2⁻¹⁰⁷⁴
  and the remainders are formed at the dividend's scale. For sqrt it is r² − a: 16·2⁻¹⁰⁷⁴·(1 + 2|r|). The first run
  put 16·2⁻¹⁰⁷⁴·|b| on the residual and failed 210 division cases; the second put 16·2⁻¹⁰⁷⁴ and failed 489 others.
  The cases were the two floors.
- The square got its own input ranges: the first run reused the multiply's, and a² overflowed.
- A seventh input category, *computed* operands, was added (below).
- `QD::EPS` was the decimal literal of QD (1.21543267145725e-63), which is not 2⁻²⁰⁹; it is now exact.

## 3. Accuracy

**How it is measured.** Every input is built from an exact random value by greedy rounding (`exact.rs`), without
any quad-double arithmetic, except the computed category, which is generated with no mutation active. Every
reference is exact.

- Add and mul: the result against the exact sum or product.
- Div: the residual |r·b − a| / |a|, which equals the relative error of r.
- sqrt: |r² − a| / (2|a|).
- The Cray-style operations are checked against |a| + |b|, and their error relative to the result is reported too.

`hp/qd_check.py mpmath` recomputes all of it with mpmath at 3000 bits (the 120 digits asked for are a lower bound,
needed so that sums of parts are exact under deep cancellation). It shares no code with `exact.rs`. On the same
dumped results (2000 per row) it gives the same maxima and 99.9th percentiles as the Rust reference, row for row.

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

**Results** (`QD_N=20000 QD_REPORT=1 cargo test --release --lib qd_`, AVX-512; 20 000 inputs per operation and
category; max and 99.9th percentile over the six categories other than underflow; units of u = 2⁻²⁰⁹):

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
(0.5 u). mpmath confirms both independently. The sloppy add gets these inputs right. A step-by-step trace of the
second case, in a Python transcription of the same operations, shows the mechanism:

- the leading parts cancel by 2¹⁵ (|a|/|a + b| = 4.1×10⁴);
- the merge emits three parts and then has read all eight;
- QD's branch `x[k] = u; if (k < 3) x[++k] = v;` then stores u and drops v;
- after the cancellation, the accumulator's u has only about 41 significant bits, and v holds the next 12.

The texts of this branch (lines 305–310 of `qd_inline.h`) were quoted from the source. `QD::ieee_add` keeps x₃ = u + v
in that branch. `qd_diag_accurate_add` (200 000 sums a + b = 2⁻ᵏa, by k) compares a literal copy of QD's algorithm
with it:

| cancellation k (bits) | 1–7 | 8–15 | 16–23 | ≥ 24 |
|---|---|---|---|---|
| QD 2.3: max relative error (cases above 1 u) | 1.17 u (1) | 10.1 u (3) | 4.4 u (1) | 0 |
| `QD::ieee_add` | 0.12 u | 0.015 u | 0 | 0 |

A first hypothesis, that the merge's leftover accumulator is dropped when four parts are emitted before all are
read, was tested the same way and changed nothing.

### Finding 2: QD's sqrt loses accuracy near the ends of the exponent range

Every input of the first run's near-overflow sqrt category (a between 2⁹⁹⁰ and 2¹⁰²²) failed, with relative error
2⁻⁵²·⁶: the result was correct to its first part only. The iteration works on r ≈ 1/√a and forms r² ≈ 1/a, whose
lower parts underflow once a exceeds about 2⁸⁶⁰. Near underflow, h = a/2 drops the last bit of subnormal parts (457
of 800 failed). `QD::sqrt` now scales a by an exact power of four, 4⁻ᵏ with 2ᵏ = `pow2_floor`(√a₀), into [1, 4) in
two exact steps, runs QD's three Newton steps and scales back. All categories pass, with ≤ 0.38 u.

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
(22.8 u against 17, in 1 of 312 sums). `qd_diag_accumulator` separates the sources:

| | n = 16 | 128 | 512 |
|---|---|---|---|
| four levels, as built | 1.9 u | 16.8 u | 72 u |
| the same levels summed exactly (no `finish`) | 1.9 u | 16.8 u | 72 u |
| level 3 replaced by its exact sum (only the products' truncation left) | 0.04 u | 0.012 u | 0.003 u |
| five levels, every running-sum error by its own two-sum (as built now) | 0.57 u | 0.37 u | 0.88 u |

(Maxima over 4000·16/n random sums; the "wide" inputs behave the same.) So the truncation and `finish` cost
nothing, and the error was level 3's plain rounding. The two-sum errors of adding into a running sum are relative to
that sum, not to the terms, so level 3 grows to about n^1.5·2⁻¹⁵⁹ Σ|aᵢbᵢ| (2⁻¹⁵³ at n = 128), and each plain addition
into it rounds at about 8 u. Now level 3 is error-free as well, and each running-sum error enters the next level by
its own two-sum.

The remaining error, below 1 u up to n = 512, is the plain summation of per-term errors that still carry the
running sums' lower-level errors. Carrying those separately too would cost three more two-sums per term and was not
done.

## 4. Controls

`cargo test --release --lib --features qd_mutate qd_controls` breaks one algorithm at a time, in scalar and 8-lane
alike unless marked, and reruns every accuracy, agreement and decimal check (1000 inputs per row). All 16 fire, each
on its own operation's rows (CONTROLS_TABLE).

- The two 8-lane-only breakages are caught by the bitwise lane check alone. The accuracy of control 16 is not
  visibly hurt (0.015 u), which shows why the bitwise check is needed.
- The parse truncation is caught by the exact greedy-rounding check and the round trips, not by the 0.25 u threshold.
  Truncating the last part errs by less than 2⁻²¹¹, inside it.
- `python3 qd_check.py mpmath --mutate=5` (sloppy mul without the error of a₀b₂) makes the independent mpmath check
  fail (MPMATH_CONTROL), and `--mutate=14` (parse truncation) does too.
- The QV-through-model test's control (Δ moved by 10⁻²⁰) fires at 9.7×10⁻²³.

## 5. Backends

`python3 qd_check.py backends` builds and runs the quad-double tests with each `v8` backend: AVX-512
(`-C target-cpu=native` here), AVX2 (`-C target-feature=+avx2,+fma`) and portable (no target features). Every build
passes, and the digest of all scalar results is the same on the three (BACKENDS_LINE). Each build also checks its
8-lane results against the scalar ones bit for bit. Since the scalar code is the same IEEE operations everywhere,
all three backends give the same bits.

The portable test build first took 17 minutes and, with the residual tests added, was killed at 3 GB of memory.
Forced inlining of the large 8-lane bodies over 8-element arrays was the cause. On the portable backend the compiler
now decides; on AVX-512 and AVX2 the hot operations stay forced inline (PORTABLE_BUILD).

## 6. Benchmarks

`cargo test --release --lib bench_qd -- --ignored --nocapture`, one thread, ns per lane and operation, 16
independent vectors per pass (throughput), minimum of 5 timings per row. "a·b + c" is a multiply then an add; the
dot term is a 16-term sum through the accumulator, finished, per term.

BENCH_TABLE

So:

- **A quad-double add costs about 10 double-double adds, and a multiply about 27 double-double multiplies.** The
  double-double multiply is very cheap (7 instructions), while the quad-double one does 6 two-products, 3 three-sums
  and a masked five-part renormalization (about 170 instructions per 8 lanes).
- **In sums of products, the form the model uses most, the ratio is 11.5–15×.**
- **The masked branches cost.** The accurate add (the merge) is 5 times the sloppy one, and AVX2's blend
  instructions cost more than AVX-512's mask registers: quad-double runs 1.5–1.9 times slower on AVX2, against 1.2–1.7
  for double-double.

## 7. What a quad-double residual would cost

Measured by `FILE=ctx.bin cargo test --release --lib bench_qd_residual -- --ignored --nocapture`. The state is
D = 3.7, M = 80 (`data/lowD_states/lowDv_D3.7000_M80.npz` via `hp/dump_ctx.py`: kt 36, kc 100, ks 250, 13 nodes,
86 Taylor steps), one thread, AVX-512. Two methods, which agree.

**Operation mix** (a counting lane through the model's generic code): RESIDUAL_COUNTS.

- The Cauchy sums' products dominate (`acc a b`), then the τ-derivative's matrix products (`acc a c`), the
  integer-weight multiplies (`x f64`) and the accumulators' `finish`.
- Counts times each operation's measured cost predict the blocks' double-double time to within 25% and a
  quad-double/double-double ratio of 9.7.

**Direct measurement:** `QV` implements `Lane` (section 8), so the same blocks run with quad-double lanes
(RESIDUAL_TIMES):

| part (one thread) | double-double | quad-double | |
|---|---|---|---|
| lane work of the 14 blocks (centre, 12 Taylor segments, horizon series) | 0.083 s | 0.74 s | measured, 9.0× |
| 251 horizon LU factorizations (80 × 80) | 0.046 s | 0.046 s + refinement | kept double-double (section 8) |
| 251 horizon solves | 0.006 s | about 0.02 s | two double-double solves and a quad-double D-product each (estimate) |
| assembly, the rest of `residual` | 0.02 s | 0.03 s | |
| residual | 0.153 s | about 0.85 s (5.6×) | |

With two threads the double-double residual takes 0.1 s of the 3.5 s mixed Newton step at M = 80
(`hp/data/mixed_precision/profile.log`). The quad-double one would take about 0.5 s, about 14% of the step, if the
Jacobian stays f64.

**At a smaller truncation error the discretization costs more.** The Taylor plan's step is
h ∝ (tol/|c_kt|)^(1/kt), and the series orders kc and ks are set for 10⁻³⁰. These are estimates from that step rule,
not measurements:

- At tol 10⁻⁴², enough for the 2× per unit of D growth of rounding effects from D = 61 to about 100, kt = 48 gives
  about the same steps for 1.8 times the work per step, a factor of about 2.
- At tol 10⁻⁶², kt = 60–80 minimizes the work at about 4 times.

The series orders kc and ks grow in proportion to the digits; ks also multiplies the number of horizon LUs. So a
quad-double residual is about 5.6 times a double-double one at the same discretization, and 10–25 times at 10⁻⁴² to
10⁻⁶².

**Cheap savings not taken:**

- `finish` of the quad-double accumulator: four qd + f64 additions, 15% of the quad-double lane work here;
- the integer weights of `cs_w` multiplied in quad-double (`x f64`, 2 million times): 10%;
- using the double-double constants' zero lower parts: the τ-derivative's products by D;
- a multiply by a double-double constant (`x const`, 0.5 million times) with qd × dd instead of qd × qd.

## 8. Stage 2: readiness for a quad-double residual

**Done.**

- `impl Lane for QV` (in `qv.rs`): `Acc = QDotAcc`, and the model's constants and inputs (`DV`) enter exactly with
  `QV::from_dv`.
- `qd_residual::qv_lane_blocks_agree` runs every block of a small synthetic context (M = 8, nodes 0.005–0.1, ks 30)
  through `model::block_state::<QV, _>`: the centre series, the Taylor stepping and the horizon series with its
  periodic solves. QV agrees with DV to 5.8×10⁻³²; the check is 10⁻²⁶, and Δ moved by 10⁻²⁰ gives 9.7×10⁻²³.
- `bench_qd_residual` runs the real M = 80 context (section 7).
- No solver or Python function uses it.

What this instantiation is not: a quad-double residual. The trait passes every constant as double-double (`cst`,
`mulc`, `addc`, `accc` take `DV`; `splat`, `muls` take `DD`); `val()` rounds back to double-double; the horizon
solves are double-double (`DDLu`). So it computes in quad-double with double-double data.

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
   - Cost: two more double-double solves and one quad-double D-product per order, about 0.015 s at M = 80, instead of
     a quad-double LU of 251 matrices: 0.5 s at the measured ratio, plus a new blocked LU.
   - For the tangent path (`Du`) nothing changes.
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
     10⁻⁶⁰ takes about five chord steps. Each is one quad-double residual and one f64 solve, about 2.5 s at M = 80 on
     two threads.
6. **Tests** (1–2 days).
   - At a converged double-double state, the quad-double residual must equal the double-double one to 10⁻³⁰ and be
     computable to 10⁻⁵⁰ and below.
   - A planted solution, a state built by integrating with known data, must give |F| at the quad-double floor.
   - Convergence at M = 16 to |F| ≈ 10⁻⁶⁰.
   - The truncation orders (kt, kc, ks, plan tolerance) for 10⁻⁴² and 10⁻⁶², with the cost of section 7.

Total about 6–9 agent-days. The predicted residual cost is about 5.6 times the double-double one at the same
discretization (0.85 s on one thread at M = 80, about 0.5 s on two). At the orders needed for a 10⁻⁴²–10⁻⁶²
truncation error it is 2–4 times that. The Jacobian stays f64.

## 9. Existing outputs are bit-identical

- `python3 mixed_check.py ref --save` with main's build (f91dafe, built privately into the scratchpad; the system
  `hpkernel` was not touched), then `--compare` with this branch's build: all 17 arrays bit-identical.
  - Covered: the residual and the Jacobian, the Newton step and its pivots, the chord step, `lu_solve_t`,
    `mode_matrix` at λ = 0.3, and the frozen-harmonic and inward-node Jacobians at D = 3.7, M = 48.
  - Logs: `hp/data/quad_double/ref_save_main.log`, `ref_compare.log`.
- `hp/proto/test_dd.py` gives 1.24, 1.28 and 1.64×10⁻³², and 9.8×10⁻³¹ for `dd_solve`, the same digits as
  `hp/data/mixed_precision/proto_test_dd.log`.
- `cargo test --release` passes (CARGO_TEST).
- Changes to existing files: additions only in `v8.rs` (masks, `sqrt`, `pow2_floor`) and `lib.rs` (modules), and the
  `qd_mutate` feature in `Cargo.toml`. No existing function changed.

## 10. Proposed README text (for the coordinator)

In "The double-double solver" list:

> - **Quad-double arithmetic (not yet used by the solver):** `hp/src/qd.rs` and `qv.rs` add quad-double numbers
>   (four f64, about 64 digits), scalar and 8-lane SIMD with bitwise the same results on AVX-512, AVX2 and portable
>   builds. They use the QD library's algorithms[^qd], with two repairs found by the tests (the accurate add's last
>   part, the square root near the ends of the exponent range). Against exact arithmetic they err by at most
>   2.1×2⁻²⁰⁹ (63 digits). They cost 10–27 times double-double per operation and 11–15 times in sums of products. A
>   quad-double residual would cost about 5.6 times the double-double one at the same orders
>   ([notes/QUAD_DOUBLE.md](notes/QUAD_DOUBLE.md); `hp/qd_check.py`).

[^qd]: Y. Hida, X. S. Li, D. H. Bailey, *Algorithms for quad-double precision floating point arithmetic*, ARITH-15
(2001).

In the Layout block under `hp/`:

```
├── src/qd.rs, qv.rs  quad-double: scalar and 8-lane (not used by the solver yet)
├── qd_check.py     checks of the quad-double arithmetic against mpmath, and across the SIMD backends
```
