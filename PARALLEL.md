# How the computations parallelize

This page covers which of the project's computations can be split across threads or processes and recombined exactly,
the mathematics that makes the recombination exact, and what cannot be split. The model throughout is CRC-32, whose
`crc32_combine`, braided and folded (PCLMUL) implementations are the cleanest examples of the same idea.

The short version:

- A computation can be cut into chunks and recombined exactly when each chunk's effect on the running state is an
  **affine map with a compact representation**. Maps of that kind compose associatively, so a prefix scan or tree
  reduction recombines the chunks, the way `crc32_combine` does.
- In this project that holds for:
  - the linearized shooting problem (the Newton step);
  - the Jacobian columns;
  - the j-recursion of the inner amplitude series;
  - every scan over independent parameters.
- It does not hold for the nonlinear parts: the primal Taylor steps, the order-by-order series recursions, the Newton
  iteration and continuation in D. These stay sequential.

Agent S's measurements for the amplitude series are in §4; agent N's for the Newton Jacobian will be filled in where
marked *(pending)*.

Contents: [1. The principle](#1-the-principle-crc-32-as-the-model) ·
[2. Multiple shooting is chunking](#2-multiple-shooting-is-chunking) ·
[3. The Newton Jacobian and LU](#3-the-newton-jacobian-and-its-lu) ·
[4. The inner amplitude series](#4-the-inner-amplitude-series-solverinner_ampamp_flintpy) ·
[5. Embarrassingly parallel scans](#5-embarrassingly-parallel-scans) ·
[6. What cannot be split](#6-what-cannot-be-split-and-why) · [7. Summary](#7-summary)

## 1. The principle: CRC-32 as the model

**The algebra.** A message M of N bits is a polynomial M(x) over GF(2). With the register started at s, the CRC
register after the message is

```math
R(s, M) = s\,x^{N} + M(x)\,x^{32} \bmod P(x),
```

which is affine in the state s. zlib's pre- and post-conditioning (init = xorout = all ones) cancels in the
difference, so for the published checksum

```math
\mathrm{crc}(A \,\|\, B) = \bigl(x^{8|B|} \bmod P\bigr)\cdot \mathrm{crc}(A) \;\oplus\; \mathrm{crc}(B) \pmod P ,
```

with |B| in bytes. This is `crc32_combine(crc1, crc2, len2) = multmodp(x2nmodp(len2, 3), crc1) ^ crc2`.

Multiplication by $`x^{8n} \bmod P`$ is a 32 × 32 matrix over GF(2): the shift operator for n zero bytes. Every
chunk acts on the state as an affine map s ↦ A s ⊕ b, with A the shift and b the chunk's own CRC.

**Why that is enough to parallelize.** Affine maps compose as

```math
(A_2, b_2) \circ (A_1, b_1) = (A_2 A_1,\; A_2 b_1 + b_2),
```

which is associative, and the pair (A, b) has a fixed size however long the chunk. A recurrence
$`x_{j+1} = A_j x_j + b_j`$ of length J therefore splits into P chunks:

1. Compose each chunk's maps into one (A, b), in parallel.
2. Run the P chunk maps in order on the start value, sequentially. This step is cheap.
3. Optionally, recompute the interior values of each chunk from its now-known start value, in parallel.

A chunk's b is the ordinary recurrence run from the zero state, at the serial cost c_v (one matrix–vector product)
per step. Its $`A = A_{j_1-1}\cdots A_{j_0}`$ either has a closed form or has to be multiplied out:

- CRC: x^{8n} mod P by square-and-multiply in O(log n).
- §4: a ratio of hypergeometric products.
- Otherwise: d × d matrix products, c_A ~ d³ against c_v ~ d² per step.

With a closed-form A, on P workers:

```math
W \approx 2J\,c_v, \qquad T_P \approx \frac{2J}{P}\,c_v + P\,c_v ,
```

where W is the work and T_P the span. The speedup tends to P/2, or to P when only the final state is needed, as for
a CRC. A tree combine replaces the P term by log₂ P. Without a closed form the span is about (J/P)(c_A + c_v), and
the speedup is about P c_v/c_A ~ P/d for dense maps: a loss unless d < P. So the scan pays off when the map is
scalar or diagonal, or when A has a closed form.

**The three CRC techniques and their counterparts here:**

| CRC technique | What it does | Counterpart in this project |
|---|---|---|
| combine | split the message across threads, join the CRCs with the shift x^{8n} | multiple-shooting condensation (§2): join segments with their transfer matrices |
| braid | N interleaved registers in one thread, joined at the end | independent Jacobian columns (§3) and independent ω or D runs (§5) |
| fold (PCLMUL) | multiply the accumulator by precomputed constants x^k mod P, add new data | precomputed linear operators applied to accumulated state: transfer matrices (§2), horizon-series LUs (§3), hypergeometric tables T_j (§4) |

One more CRC identity carries over. The CRC of a single set bit at distance n from the end is $`x^{n} \bmod P`$, so
the full matrix of a linear map is its action on unit vectors, one column each, all independent. This is how the
Jacobian is built (§3).

The monoid has to be **compact**. Nonlinear maps also compose associatively, but a composition of nonlinear maps has
no finite representation: you cannot store "Taylor steps 1–20" as a fixed-size object and apply it to a new start
state. So §6 is sequential.

## 2. Multiple shooting is chunking

The double-double solver (`hp/src/model.rs`) solves the DSS equations in the self-similar radius x by multiple
shooting. M τ points per field (M = 48), K nodes $`x_0 < \dots < x_{K-1}`$ (K = 10 in production: 0.02, 0.05, …, 0.7).

**Unknowns.** The length is L = 4M(K+1) + 1, which is 2113 for M = 48, K = 10:

- centre data $`y_c = (p_1, c_0)`$ (2M);
- each node's state $`z_b = (p, m, a, c)`$ (4M each);
- horizon data $`y_h = (m_h, a_h)`$ (2M);
- Δ.

**Residual blocks**, all-outward layout (the default `match_node`):

```math
\begin{aligned}
F_0 &= \mathcal C(y_c;\Delta) - z_0 &&\text{centre series evaluated at } x_0,\\
F_b &= \Phi_b(z_{b-1};\Delta) - z_b, \quad b = 1,\dots,K-1 &&\text{Taylor stepping } x_{b-1}\to x_b,\\
F_K &= \mathcal H(y_h;\Delta) - z_{K-1} &&\text{horizon series evaluated at } u = 1-x_{K-1},\\
F_{\rm ph} &= w\cdot p_1 &&\text{phase condition.}
\end{aligned}
```

Each block depends on its own inputs and Δ only, so **given the start states, the K + 1 blocks are independent**,
like CRCs of separate chunks. `model::residual` evaluates them with `into_par_iter` over blocks.

**The Newton system.** Write $`G_b = \partial\Phi_b/\partial z_{b-1}`$ (the segment's 4M × 4M transfer matrix),
$`g_b = \partial\Phi_b/\partial\Delta`$, and $`C_y, c, H_y, h`$ for the centre and horizon derivatives. The
Jacobian is block-bidiagonal with a border (the Δ column and the phase row):

```math
J = \begin{pmatrix}
C_y & -I  &        &         &        &      & c \\
    & G_1 & -I     &         &        &      & g_1 \\
    &     & \ddots & \ddots  &        &      & \vdots \\
    &     &        & G_{K-1} & -I     &      & g_{K-1} \\
    &     &        &         & -I     & H_y  & h \\
w^{T} &   &        &         &        &      & 0
\end{pmatrix}
```

The columns are y_c, z_0, z_1, …, z_{K−2}, z_{K−1}, y_h and Δ; the rows are F_0, F_1, …, F_{K−1}, F_K and F_ph.
w^T acts on the p₁ half of y_c.

Row b of J δz = −F reads

```math
\delta z_b = G_b\,\delta z_{b-1} + g_b\,\delta\Delta + F_b ,
```

an affine recurrence over the segments. This is the CRC register update with the segment's transfer matrix as the
shift: the linearized problem combines exactly the way `crc32_combine` does. Define the composed transfer matrix
$`G_{b:a} = G_b G_{b-1}\cdots G_a`$ (the identity when b < a). Eliminating the interior nodes gives

```math
\delta z_{K-1} = \Psi\,\delta y_c + \psi\,\delta\Delta + r,\qquad
\Psi = G_{K-1:1} C_y,\quad
\psi = G_{K-1:1}\,c + \sum_{b=1}^{K-1} G_{K-1:b+1}\, g_b,\quad
r = \sum_{b=0}^{K-1} G_{K-1:b+1} F_b .
```

Substituting into the horizon row and the phase row leaves a **condensed system** of size 4M + 1 = 193 in
(δy_c, δy_h, δΔ):

```math
\begin{pmatrix} -\Psi & H_y & h-\psi \\ w^{T} & 0 & 0 \end{pmatrix}
\begin{pmatrix} \delta y_c \\ \delta y_h \\ \delta\Delta \end{pmatrix}
= \begin{pmatrix} r - F_K \\ -F_{\rm ph} \end{pmatrix}.
```

Here w acts on the p₁ half of y_c. The interior δz_b then follow from the recurrence, in parallel per chunk (step 3
of §1).

The triples (G_b, g_b, F_b) compose by the rule of §1, so Ψ, ψ and r form a tree reduction of depth log₂ K. The
d³ cost per product is affordable here because d = 4M is fixed and K is small.

**Conditioning.** The explicit products $`G_{b:a}`$ are what multiple shooting exists to avoid:

- They contain the exponentially growing and decaying modes of the radial ODE (the fast WKB mode of Proposition 3).
- They contain the weak, near-null directions that already limit the Newton basin.

Forming Ψ explicitly returns the conditioning of single shooting. The stable way to use the same block structure is
a **structured factorization**. Examples are block cyclic reduction with pivoting, Wright's structured QR for
boundary-value problems (S. J. Wright, *Stable parallel algorithms for two-point boundary value problems*, SIAM J.
Sci. Stat. Comput. 13, 1992) and alternate row/column elimination for almost-block-diagonal systems. These work on
pairs of adjacent block rows with pivoting, so they keep the stability of the dense LU. Their parallel depth is
log₂ K block steps.

**Cost, operation counts (estimates, not measurements).** One LU of a 4M-sized block costs about $`(4M)^3`$ DD
operations, 7.1 × 10⁶ at M = 48.

- The dense LU in `hp/src/linalg.rs` does (2/3)L³ ≈ 6.3 × 10⁹ DD multiply-adds at L = 2113.
- A structured elimination does a few block operations per segment, about 4(4M)³(K+1) ≈ 3 × 10⁸, roughly
  20 times less.

The dense factorization does not exploit the zeros. Its trailing update in `DDLu::factor` runs over every trailing
column even where the pivot-row entry is exactly zero. Two cheap changes move it toward the structured count while
staying a dense LU:

- skip columns whose pivot-row entry is zero;
- restrict the update to the rows where the multiplier column is nonzero, which is a band of about two block rows
  plus the phase row.

**What stays sequential here:**

- the Newton iteration itself, since each step needs the previous iterate;
- inside each segment, the primal Taylor steps, a nonlinear composition (§6).

The linear algebra of one Newton step is the part with the CRC structure.

## 3. The Newton Jacobian and its LU

**Columns are independent.** By the unit-vector identity of §1, column i of J is the directional derivative of F
along e_i. `model::jacobian` builds it by forward-mode duals. `Du = (v, d)` carries the value and one tangent, each a
lane vector of double-doubles (`DV`, 8 doubles hi + lo), so 8 τ points travel per SIMD register.

A column task is (block b, input field f, τ point j): seed the tangent with a unit vector, run `block_state` on duals
and store the block's 4M tangent outputs. The Δ column is one task per block. The number of tasks is

```math
N_{\rm col} = 2(2M+1) + (K-1)(4M+1) = 1931 \quad (M = 48,\ K = 10).
```

The tasks are independent and run with `par_iter`. On 8 threads they saturate the machine, and rayon's work stealing
balances the unequal task costs (segments with different step counts; the horizon block with ks + 1 orders). The
target-node −I blocks and the phase row are inserted directly, not differentiated.

In CRC terms there are two levels of parallelism:

- Within one task, the 8 lanes process 8 τ points at once (data parallelism, like PCLMUL handling 128 bits per
  instruction).
- Across tasks, independent columns run on different threads. This is the braid: independent registers, joined at the
  end by placing each column in its slot, with no shift needed because the columns are disjoint.

**Where the work is (model).** One Taylor step of order kt on M points costs:

- per order: two applications of the folded d/dθ matrix (2M² each);
- per order: about nine Cauchy products of length k (≈ 9kM).

That is $`\approx 2 k_t M^2 + 4.5\,k_t^2 M \approx 4.5\times 10^5`$ DD multiply-adds for kt = 36, M = 48.

A tangent step costs 2–3× a primal step: a dual multiply is three plain multiplies, while the linear d/dθ
application costs the same on both parts. So the Jacobian of one Taylor block costs about 3(4M + 1) ≈ 580 primal
evaluations of that block. Two options cut total work, not just span:

- **Vector-mode tangents that share the primal.** Each column task now recomputes the primal values v. A dual with
  r tangents per pass costs about c_p + r·c_t instead of r(c_p + c_t). With c_t ≈ 1–2 c_p this saves between a third
  and a half of the Jacobian time at large r. It also turns the d/dθ application into a matrix–matrix product
  (M × M times M × r), which uses the cache and vector units better.
- **Step-parallel transfer matrices.** The tangent of a Taylor step along a fixed primal trajectory is linear:
  $`\delta s_{i+1} = G^{(i)}\delta s_i + g^{(i)}\delta\Delta`$, and $`G_b = G^{(S)}\cdots G^{(1)}`$.
  1. Run the primal path once, sequentially.
  2. Form the per-step $`G^{(i)}`$ independently in parallel.
  3. Multiply them in a tree.

  The extra cost is one 4M × 4M product per step, 2(4M)³ ≈ 1.4 × 10⁷, about 5% of the step's 4M + 1 tangents. The
  total work is unchanged, but the span drops from S tangent steps to one tangent step plus log₂ S products. This matters only
  with many more cores than the 1931 column tasks; on 8 threads it gains nothing.

**The LU.** `DDLu::factor` is LAPACK getf2 order with partial pivoting. Two steps are serial and one is parallel:

- The pivot search and multiplier scaling at step k are serial, O(L) each.
- The rank-one trailing update runs over columns with rayon once ncols × rows > 32768, on 8-row lane vectors (`axpy8`).
- `solve` and `solve_t` are serial triangular sweeps, O(L²), small next to the O(L³) factorization.

Independent right-hand sides can be solved in parallel; for example, `delta_correction` solves with $`J^{-T}e_\Delta`$
while chord steps reuse one LU. The two levers on the LU are the structure-aware ones of §2 (skip exact zeros, or
factor block by block). A blocked right-looking LU, panel then GEMM-like update, also cuts memory traffic. The
f64 kernel (`src/model.rs`, `linearization`) has the same structure: tasks (block, unknown) with a scalar `Dual`,
run with `par_iter`, and the −spec∘phys target blocks added directly.

**Measured time breakdown and speedups** *(pending: agent N measurements)*:

| Stage (M = 48, K = 10, one Newton step) | 1 thread | 8 threads | speedup | notes |
|---|---|---|---|---|
| residual (K + 1 blocks) | (pending) | (pending) | (pending) | span = slowest block |
| Jacobian (1931 column tasks) | (pending) | (pending) | (pending) | |
| LU factor (L = 2113) | (pending) | (pending) | (pending) | |
| solve | (pending) | (pending) | (pending) | serial |

## 4. The inner amplitude series (`solver/inner_amp/amp_flint.py`)

The inner problem at x ~ √n (README, "The inner problem at large x") is expanded in powers of the amplitude:
$`F = \varepsilon\sin\theta + \sum_{\nu\ge3} f_\nu e^{i\nu\theta}`$. The fields are series in ξ:
y = Σ d_j ξ^{2j+1}, σ = Σ s_j ξ^{2j+2}, c = Σ g_j ξ^{2j}. The code stores X̃_j = X_j/2^j so that every product is
a pure convolution in j. Each X_j is a row over harmonics |μ| ≤ H = K, and each field is further expanded in powers
ε^k.

**The order structure.** Order k is triangular. Every product at order k pairs lower orders, except the g⁽⁰⁾ = 1
term, which supplies the linear operator. So order k is a fixed linear problem with a forcing built from orders < k:

- Even k (`pass_even`): $`e^{(k)}_m = \tfrac14[d\,d]^{(k)}_{m-1}`$ and
  $`g^{(k)}_m = \frac{1}{2m}\bigl(e^{(k)}_m + \sum_{k_1\ge2}[g^{(k_1)} e^{(k-k_1)}]_m\bigr)`$. There is no
  recursion in m, so all m are independent once the convolutions are done.
- Odd k (`pass_odd`): with $`S^\sigma_j = \sum_{k_1\ge2}[g^{(k_1)}s^{(k-k_1)}]_j`$,
  $`S^d_j = \sum_{k_1\ge2}[g^{(k_1)}d^{(k-k_1)}]_j`$ (lower orders only) and s = iμω for harmonic μ:

```math
s_j = (2j+1+s)\,d_j - S^\sigma_j, \qquad d_{j+1} = \frac{(2j+2+s)\,s_j}{4(j+1)} - S^d_{j+1}.
```

**It is an affine recurrence, and scalar.** Eliminating s_j gives

```math
d_{j+1} = a_j\,d_j + b_j,\qquad a_j = \frac{(j+\alpha)(j+\beta)}{j+1},\qquad
b_j = -\frac{(2j+2+s)\,S^\sigma_j}{4(j+1)} - S^d_{j+1},
```

with α = (1+s)/2, β = (2+s)/2. The harmonics do not couple inside the recurrence: the operator is diagonal in μ.
All the coupling between harmonics sits in the forcing b_j, which the convolutions of lower orders provide. So the
"matrix" A_j of §1 is the scalar a_j (or a 2 × 2 block if (d_j, s_j) is kept as the state), for each μ separately.
The recurrence even has a closed form. The homogeneous solution is the hypergeometric coefficient that `add_linear`
already uses:

```math
T_j = \prod_{i<j} a_i = \frac{(\alpha)_j(\beta)_j}{j!},\qquad
d_j = T_j\Bigl(d_0 + \sum_{i=0}^{j-1}\frac{b_i}{T_{i+1}}\Bigr).
```

So the j-recursion at fixed (k, μ) is a **prefix sum** of b_i/T_{i+1}, with T_j a precomputed table independent of
the order k. That table is the counterpart of CRC's folding constants: a fixed linear operator applied to new data.

Checked against the code's own recursion at 40 digits, J = 60, μ = 1, 3, −5, with random forcing:

- the closed form agrees to 1e-40;
- a 4-chunk compose-then-combine agrees to 1e-40.

The T_j grow factorially (|T₆₀| ~ 10⁸²). This is the Gevrey divergence the late-order fits measure, not a
conditioning problem: the chunk maps are scalars and the arithmetic is ball arithmetic. Results of a reordered
evaluation agree within the balls but are not bit-identical.

**What this means for speed.** The scan, chunked or closed form, is O(J·H) work per order, small in any case. Its
parallel version gains nothing measurable. The work is elsewhere.

**Where the time goes.** Agent S's first profile, a serial run at K = 7, J = 300 on one cloud core:

| Stage | Calls | Time | Share |
|---|---|---|---|
| late-order fits `S_at` | 10 | 160 s | 62% |
| of which mpmath `qr_solve` (Householder) | 10 | 125 s | 48% |
| convolutions `conv` (acb_poly products) | 21 | 98 s (4.7 s each) | 38% |
| everything else (recursion, packing) | | ~1 s | <1% |
| total | | 259 s | |

The counts follow from the order structure. Order k runs k − 1 products, each over the full packed (j, μ) grid:

- odd k: (k−1)/2 each for S^σ and S^d;
- even k: k/2 for [dd] and k/2 − 1 for [g e].

The product count through order K is therefore

```math
N_{\rm conv}(K) = \sum_{k=2}^{K} (k-1) = \frac{K(K-1)}{2}
```

(21 at K = 7). Each product has length about (J+1)(4K+1), which is 8 700 here and 26 500 at J = 500, K = 13. So the
products grow roughly like K³J and take a larger share at production sizes. Through order K there are
$`N_{\rm fit} = \sum_{k\ \rm odd \le K} (k+1)/2`$ fits (10 at K = 7), about K²/4.

**The fits are a fixed linear functional, so precompute them.** This is the folding-constant idea again, and it is
the biggest single saving. `S_at(ν, k)` solves the least-squares problem min‖A c − y‖ and keeps c₀. The basis matrix
A (`Amp.basis(ν)`: rows j = jlo … J, columns for the Γ-ratio corrections and the neighbour families) depends on ν
but **not** on the order k, and it is already cached. Only y changes from call to call. So

```math
c_0 = a_\nu^{T} y, \qquad a_\nu^{T} = e_0^{T} A^{+} \ \ (\text{the first row of the pseudo-inverse}),
```

is one dot product of length J − jlo + 1 once a_ν is computed: one QR per ν instead of one per (ν, k). The fit
residual ‖(I − AA⁺)y‖ follows from the same factorization, as the norm of the part of Qᴴy outside the column space
(complete QR). This removes most of the 48% now spent in Householder QR. It changes no number, beyond rounding at
60 digits.

**Kronecker substitution, the folding step.** A product over (j, μ) with |μ| ≤ H becomes one univariate polynomial
product by the map (j, μ) ↦ jW + μ + H, with W = 4H + 1 (`Amp.poly`, `Amp.conv`). Two harmonics μ₁ + H and μ₂ + H in
[0, 2H] sum to at most 4H < W, so rows never overlap. The product row j is read back at offsets jW + H … jW + 3H,
which is |μ| ≤ H after truncation. One large FFT-class multiply replaces (J+1)²(2H+1)² scalar products, the same
packing trick as one PCLMUL instruction doing 64 × 64 bit products.

**What parallelizes in the series:**

1. **The products at one order.** The k − 1 products at order k are independent and summed, a linear combine with
   no shift. With P workers the span at order k is ⌈(k−1)/P⌉ products; the bounds on 8 workers are in the table
   below.
2. **The late-order fits at one order.** At odd k the fits S_ν^(k) for ν = 1, 3, …, k are independent:
   - The order-k data enter S_ν^(k) only through the ν-th harmonic of d⁽ᵏ⁾ (the E⁽⁰⁾ = δ term).
   - `add_linear(k, ν)` changes only harmonics ±ν.
   - The θ-series E and ⟨F²⟩ at order k use F only through order k − 1.

   So the (k+1)/2 fits can run in parallel, and inside each fit the rows j = jlo … J are independent.

3. **The j-recursion.** A prefix sum, as above: parallel, but negligible.

4. **Fewer, larger products.** Summing over k₁ is the ε^k coefficient of a product in a third Kronecker variable.
   Relaxed (online) multiplication (van der Hoeven) computes all orders with O(K log K) block products instead of
   K²/2, while the inputs arrive order by order. At K ≤ 21 this saves at most a factor of about 2. It needs block
   sizes in ε that respect the tuning of f_ν after each order. Not implemented.

Bounds for the product part on 8 workers (item 1), ignoring the cost of shipping polynomials between processes:

| K | products (serial) | span in products, P = 8 | bound |
|---|---|---|---|
| 9 | 36 | 8 | 4.5× |
| 13 | 78 | 16 | 4.9× |
| 17 | 136 | 24 | 5.7× |
| 21 | 210 | 36 | 5.8× |

For an ω scan, items 1–2 are not needed: running different ω in different processes (§5) gives the full factor P
with no communication. Order-level parallelism and the fit precomputation matter for the **latency of one deep run**,
for example one ω at K = 21.

**Measured results** (agent S, `solver/inner_amp/amp_flint_par.py`, on the cloud container's 2 shared cores). Most of
the gain is algorithmic, not parallel:

- *Products packed by order.* Order k carries only the harmonics −k, −k+2, …, k, so a product of orders k₁ + k₂ = k is
  packed with stride k + 1 instead of 4K + 1, and only the first (J + 1)(k + 1) coefficients are formed (a truncated
  product). The two mirror-image d·d products of the even pass are formed once.
- *The fit functional precomputed* (item 2): G = (AᴴA)⁻¹Aᴴ once per ν in ball arithmetic, so each fit is c = G y; the
  ψ tables by recurrence, and the normalisation exponential through N′ = L′N.
- *Processes, not threads.* python-flint holds the GIL during a product (two threads take 2.3× the time of one
  product), so `--procs N` spawns N worker processes (Windows-safe). Each keeps its own copy of the finished tables, which
  are sent to it once, when an order is done.

| Run | serial `amp_flint.py` | 1 process | 2 processes |
|---|---|---|---|
| K = 7, J = 300 | 90 s | 3.4 s | 2.7 s |
| K = 9, J = 500 | — | 17 s | — |
| K = 11, J = 500 | — | 38 s | 23 s (products 2.06×) |
| K = 17, J = 500 | — | 147 s | 132 s (one core shared with a build) |

So one process is 26× faster than the serial code at K = 7, and products are now about 95% of the run. The cost grows
like about K³·⁵ J²·⁵. On 8 processes the model from the measured per-order times (round-robin, about 80% efficiency,
5 s overhead) gives about 30 s for K = 17, J = 500. Agreement: K = 7, J = 300 matches the serial code's tuned f_ν
to every printed digit and σ₁–σ₃ to 8×10⁻³⁵ (the serial code's 60-digit QR; ball radii ≤ 7×10⁻⁵⁰); 1 and 2 processes
are bit-identical; the stored `data/sigF_*` files (K = 9 and 11 at J = 500, K = 9 at J = 750) are reproduced to all
20 stored digits.

One finding from the faster code: **J = 500 is not enough from order 13 on.** The ν = 3 tuning at order 13 leaves
|f₃| = 1.1×10⁻³ at J = 500, 7.7×10⁻⁵ at J = 750 and 1.3×10⁻⁵ at J = 1000, and σ₆ moves from 0.0138 − 0.0080i
(J = 500) to −2.7×10⁻⁴ − 1.09×10⁻³i (J = 1000); at J = 500, σ₇ and σ₈ blow up to about 10⁶ and 10¹¹. The ball
radii stay small, so this is the fit's truncation, not rounding. σ₅ is converged (J = 750 and 1000 agree to 10⁻⁸). Runs
with K ≥ 13 need J ≥ 1000 and a second J to compare.

## 5. Embarrassingly parallel scans

These need no combine at all: independent runs with no shared state. They correspond to CRCs of unrelated messages.

- **ω scans** (`solver/inner_amp/scan_omega.py WF_LO WF_HI N K J PROCS`): one `Amp(wf, K, J)` per ω, a process pool,
  resumable (one JSON per ω). The speedup is min(P, N_ω) up to memory and load balance. All ω cost the same at fixed
  (K, J), so 16 ω on 8 processes take two rounds. `zero_selection.py` then finds the Padé zeros of S₁ in ε² and
  the ω where the zero is real.
- **Seeds in D**, each solved independently from its own seed: low-D sweeps at several kc (`solver/scan_ne.py` with
  NE_SEED_KC), D = 5 checks, the per-integer-D states in `hp/data/states/`.
- **Finite-difference Jacobian columns** for the lateral-solution solves (`solver/inner_stokes/continue_M.py procs`,
  and `solver/inner_stokes/full_M.py` / `fixedA.py --procs N`); the formula follows the list.

The unknowns are u = (A, ω, B₃, C₃, …, B_M, C_M) ∈ ℝⁿ, with n = M + 1. Column i of the Jacobian is

```math
J_{:,i} \approx \frac{r(u + h_i e_i) - r(u)}{h_i}, \qquad i = 1,\dots,n .
```

The n + 1 residual evaluations are independent. With one residual taking time t_r, the wall time per Jacobian is
t_r(1 + ⌈n/P⌉) instead of t_r(n + 1): n = 6 (M = 5) on P = 8 gives 3.5×. Central differences double both the count
and the attainable speedup. The nested solve in `full_M.py` parallelizes the same way: an outer 2 × 2 Newton on
(A, ω) around inner solves for the higher harmonics, with the three inner solves per outer Jacobian independent.

## 6. What cannot be split, and why

The criterion from §1: the chunk maps must form a monoid with a compact representation. These do not:

- **Continuation in D** (`hp/scan_hp.py`). The map from the solution at D to the solution at D + δD is the nonlinear
  solve F(z; D + δD) = 0 started from a prediction. It is defined only inside the Newton basin, which along the weak
  directions limits the step. Past D = 61 this costs about 22 hours per unit D, and better double-double predictors
  (agent L) did not lengthen the step. Two consequences:
  - The composition of steps has no fixed-size representation, so there is nothing to combine.
  - Each step needs the previous converged solution.

  Running P points at D + δD, D + 2δD, … in parallel from extrapolated seeds works only while the extrapolation error
  stays inside the basin, so the gain is about 1.
- **Routes that remain:**
  - Independent anchors. Two chains, one climbing from D = 61 and one descending from a seed at much larger D (if a
    large-D asymptotic profile is ever accurate enough to land in the basin), would meet in the middle for a factor
    of 2. This is the same idea as the shooting `match_node`.
  - Parareal on the Davidenko equation dz/dD = −J⁻¹ ∂F/∂D: a cheap coarse propagator run sequentially, with fine
    solves in parallel, iterated. The speedup is at most P/k_iter, and the coarse predictions must still land in the
    basin.
  - Neither is implemented.
- **The Newton iteration.** Each iterate needs the previous one. Chord steps reuse an LU but stay sequential.
- **The primal Taylor steps within a segment.** These are a nonlinear composition. The segments themselves are
  parallel given their start states (§2), and the tangent along a fixed primal path is linear (§3).
- **Order-by-order series recursions**: the centre series (kc orders), the horizon series (ks orders) and the
  ε-orders of §4. Order k is a nonlinear function of orders < k, so the order loop is sequential. Within an order,
  the linear operator is fixed and known in advance, so its factorizations can be precomputed in parallel.
  `horizon_lus` already factors the ks + 1 periodic M × M systems with `into_par_iter`. The forcing products and the
  per-order solves are data-parallel.

## 7. Summary

| Computation | Structure | Parallel scheme | Status | Expected on 8 threads |
|---|---|---|---|---|
| Residual blocks (hp, f64 kernel) | independent given start states | blocks with `par_iter` | done | ≤ (K+1)/⌈(K+1)/8⌉ ≈ 5.5× for K = 10, span = slowest block |
| Newton Jacobian columns | linear in the seed (forward-mode duals) | column tasks with `par_iter`, 8 τ lanes per SIMD register | done | near 8× (1931 tasks) *(pending: agent N measurements)* |
| Vector-mode tangents | linear, shared primal | r tangents per dual pass | proposed (agent N auditing) | 1.5–2× less work *(pending)* |
| Newton LU | dense, block-bidiagonal + border | parallel trailing update (done); structure-aware or block LU | partly done, in progress (agent N) | dense: memory-bound; structured: ~20× fewer operations (estimate) *(pending)* |
| Condensation (shooting combine) | affine over segments | tree of (G_b, g_b, F_b) or structured QR/LU | not implemented | depth log₂ K; stability needs the structured form |
| Taylor tangent per step | linear along a fixed primal path | per-step transfer matrices, tree product | not implemented | only with ≫ 1931 cores |
| Horizon series LUs | independent per order | `into_par_iter` over ks + 1 orders | done | ~8× on that stage |
| amp_flint convolutions at order k | independent products, linear sum | packed by order (stride k + 1, truncated); process pool over products | done (`amp_flint_par.py`) | 26× on one process (algorithmic); products 2.06× on 2 processes; bound 4.5–5.8× on 8 |
| amp_flint late-order fits at order k | fixed linear functional per ν; independent over ν | precompute G = (AᴴA)⁻¹Aᴴ once per ν; setups on the pool | done (`amp_flint_par.py`) | the 48% QR share removed; at K = 17 products are about 95% of a run |
| amp_flint j-recursion | scalar affine per μ, closed form prefix sum | prefix scan or closed form | possible, not needed | negligible share of the time |
| ω scan of the inner series | independent | `scan_omega.py`, one process per ω | done | 8× |
| Finite-difference Jacobian (lateral solves) | independent columns | `--procs N` (continue_M.py, full_M.py, fixedA.py) | done | (n+1)/(1+⌈n/8⌉): 3.5× at n = 6 |
| Low-D sweeps, D seeds | independent | separate runs | done | 8× |
| Continuation in D | nonlinear, basin-limited | none (independent anchors at most) | not possible | ~1× (2× with two anchors) |
| Newton iteration, Taylor primal, series orders | nonlinear prefix | none | not possible | 1× |
