# Quad-double arithmetic in the Rust kernel (`hp/`)

Work in progress on branch `wip/quad-double`. This first commit records, before any accuracy test is run, the error
bounds expected from the literature and the pass thresholds the tests will use. Results follow in later commits.

## Expected bounds (written before the runs)

Units: u = 2⁻²⁰⁹ ≈ 1.2×10⁻⁶³ (the QD library's `qd_real::_eps`). The algorithms paper writes ε_qd = 2⁻²¹¹ = u/4
for "the precision one expects for quad-double numbers".

Source: Y. Hida, X. S. Li, D. H. Bailey, *Algorithms for quad-double precision floating point arithmetic*, ARITH-15
(2001); LBNL report, https://escholarship.org/uc/item/69q5t2mj (read through WebFetch; quotes as the tool returned
them). What it states:

| operation | stated bound | in units of u |
|---|---|---|
| sloppy add (the library's default `+`) | Cray-style: a ⊕ b = (1 + δ₁)a + (1 + δ₂)b, \|δᵢ\| ≤ ε_qd | 0.25 (\|a\| + \|b\|), no bound relative to the result |
| accurate (IEEE-style) add | (1 + δ)(a + b), \|δ\| ≤ 2ε_qd "or so"; "computes the first four components of the result correctly" | 0.5 relative |
| multiplication | (1 + δ)ab, δ "bounded by ε_qd" (stated for the algorithm with the O(ε⁴) terms, i.e. the accurate one) | 0.25 relative |
| division | none stated | — |
| square root | none stated (Newton on 1/√a, three iterations) | — |
| renormalization | "Necessary conditions for this renormalization algorithm to work correctly are, unfortunately, not known." | — |

My own expectations where the paper states none (derived from the algorithms, not proved):

- **Sloppy mul**: drops a₁b₃ + a₂b₂ + a₃b₁ (each ≤ 2⁻²¹²\|a₀b₀\| for normalized operands) and sums the O(ε³) terms in
  plain f64; expect ≲ 1 u relative.
- **Sloppy div** (four quotient terms) ≲ 2 u, **accurate div** (five) ≲ 1 u relative.
- **sqrt** ≲ 2 u relative.
- **qd × f64, qd × dd** ≲ 1 u relative; **qd + f64, qd + dd**: Cray-style, ≲ 0.5 u (\|a\| + \|b\|).
- **Parsing** (exact decimal → greedy round-to-nearest of the four components): ≤ 2⁻²¹² relative = 0.125 u.
  **Printing**: exact (correctly rounded to the digits asked for).
- **Dot accumulator** (`QDotAcc`, a four-level Sum2/SumK-style cascade, products truncated as in the sloppy mul):
  \|error\| ≲ (8 + n)·2⁻²¹²·Σ\|aᵢbᵢ\| for n terms.
- **Near underflow** of the components (leading part below about 2⁻⁸⁶⁰, so the last component is subnormal) the
  relative bounds fail by design; expect an absolute floor of a few × 2⁻¹⁰⁷⁴. **Near overflow**: the bounds hold as
  long as no intermediate overflows.

## Pass thresholds of the cargo tests (fixed before the runs)

Each is about 2–4 times the expectation above, so that a broken algorithm (errors of 2⁵⁰ u and more for a dropped
error term at order ε³ or above) fails by a wide margin while the correct one passes.

| operation | measure | threshold |
|---|---|---|
| accurate add | relative | 1 u |
| sloppy add, qd + f64, qd + dd | \|error\| / (\|a\| + \|b\|) | 1 u |
| sloppy mul, qd × dd | relative | 2 u |
| accurate mul, qd × f64 | relative | 1 u |
| sloppy div | relative | 4 u |
| accurate div | relative | 2 u |
| sqrt | relative | 4 u |
| parse | relative | 0.25 u |
| dot accumulator | \|error\| / Σ\|aᵢbᵢ\| | (8 + n)/8 u |

In the near-underflow category an absolute allowance of 16·2⁻¹⁰⁷⁴ is added to every threshold. Scalar and 8-lane
results must agree bitwise, lane by lane, for every operation and input.
