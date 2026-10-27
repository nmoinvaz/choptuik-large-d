# Ghost admixture: growth rates of a test scalar field on the DSS solution

**Question.** Add to the matter a second massless scalar field ψ, with ψ = 0 in the critical solution: either a
ghost (kinetic term of the opposite sign) or an ordinary scalar (for instance the imaginary part of a complex field).
Does the critical solution stay critical, or does it gain an unstable mode in the ψ direction? At linear order
around ψ = 0 the stress tensor is quadratic in ψ, so ψ's perturbations do not act back on the metric or on φ. They
obey the wave equation □ψ = 0 on the critical spacetime, which is the same equation for either sign of the kinetic
term. So the question is the spectrum of a test scalar field on the DSS solution, with perturbations
$`\psi \propto e^{\lambda\tau}`$ × (periodic in τ, regular at the centre and at the past light cone x = 1). A zero
with Re λ > 0 would be an additional unstable mode of the critical solution in the model with ψ.

**One exact zero.** ψ = φ₀, the critical solution's own scalar field, solves the test equation with λ = 0. In the
two-field model it is the symmetry that rotates φ into ψ: an O(2) rotation for an ordinary ψ, a hyperbolic O(1,1)
rotation (φ, ψ) → (φ cosh s, φ sinh s) for a ghost, which leaves the stress tensor φ'² − ψ'² unchanged. Both give
a one-parameter family of critical solutions with the same metric, so λ = 0 is marginal, not unstable.

## 1. How det M_ψ(λ) is computed

The ψ equations are the p, m equations of `hp/src/model.rs` with a and c held at the background:

```math
x(x-c)\,p_x = -x\,(\partial_\tau + \lambda)\,p + G p - c p + \tfrac{n}{2}\,c\,(p+m),\qquad
x(x+c)\,m_x = -x\,(\partial_\tau + \lambda)\,m - G m + c m - \tfrac{n}{2}\,c\,(p+m),
```

with G = (n − 1)(c a² − c) from the background. They are linear in (p, m), and with a, c given they are exactly
the scalar field's equations. `hp/modecount_cx` (the complex-λ mode matrix of the full problem, `notes/MODE_COUNT.md`)
computes them with `Cx::test = TEST_ON`: wherever the centre series, the Taylor steps and the horizon series form a
coefficient of a or c (the block inputs included), its tangent is set to zero, so the tangents of p and m are the
test field's. `test_matrix_c` keeps the ψ rows (p, m of every block) and the ψ columns (p₁, the nodes' p and m, m_h):
a square matrix of size 2M(K + 1), the multiple-shooting matrix of the test field, whose determinant vanishes at its
growth rates. Selecting the ψ rows and columns of the full mode matrix would not do: inside each block p feeds a
(through H = a(p² + m²) + …) and a feeds back into p, so the a and c tangents have to be removed during the
integration, not afterwards.

The crate was ported to the current kernel first (power-of-two scaled series, multidouble's lanes); its structure,
the complex coupling of λ and the complex LU are unchanged. At li = 0 the horizon solves are hpkernel's M × M solves of
A + λ for the real and imaginary parts separately, so the real-λ matrix is hpkernel's bit for bit.

## 2. Checks (`hp/modecount_test_check.py`, records in `hp/data/modecount_test_check.jsonl`)

| | what | D = 8.25 (M 32) | D = 30.07 (M 32) | D = 45.05 (M 32) |
|---|---|---|---|---|
| 1 | full mode matrix (test field off) at λ = 0 and 1.7 against hpkernel's, entry by entry | 0 of 1,982,464 entries differ | 0 of 589,824 | 0 of 589,824 |
| | control: λ one double above 1.7 | 163,840 differ | 81,920 differ | 81,920 differ |
| 2 | complex λ against the cached evaluations of the count in MODE_COUNT.md (old crate) | (see below) | exact (difference 0.0) at 6.182 + 0.40i, 6.182 + 0.80i | (no cached values) |
| 3 | identity M_ψ(0; z) z_ψ = F_ψ(z) at a perturbed state (every row of F moved by about 10⁻⁸): max \|M_ψ z_ψ − F_ψ\|, and the bound 10ν | 5.2×10⁻²⁸ (7.5×10⁻²⁷) | 8.1×10⁻²⁶ (2.2×10⁻²⁴) | 3.5×10⁻²² (4.5×10⁻²¹) |
| | control: a's tangent kept (mode 2), must exceed 10³ × the bound | 0.13 | 0.021 | 0.011 |
| 4 | max \|M_ψ(0) z_ψ\| at the solution (λ = 0 is a zero), and the bound 10 max(ν₀, \|F_ψ\|) | 8.5×10⁻²⁸ (8.4×10⁻²⁷) | 5.3×10⁻²⁵ (5.3×10⁻²⁴) | 3.3×10⁻²² (3.9×10⁻²¹) |
| | control: (p, −m) in place of (p, m), must exceed 10³ × the bound | 1.2 | 0.67 | 0.54 |
| | smallest pivot of M_ψ at λ = 0, and at λ = ∓0.1 | 1.7×10⁻²⁶; 1.7×10⁻⁴, 3.5×10⁻⁴ | 2.7×10⁻²⁸; 7.4×10⁻¹¹, 1.2×10⁻¹⁰ | 9.1×10⁻²⁶; 4.5×10⁻¹³, 5.6×10⁻¹³ |
| 5 | Cauchy–Riemann on ln det M_ψ at λ = 1.3 | 2.1×10⁻⁷ | 2.0×10⁻⁷ | 2.8×10⁻⁷ |
| | control: det M_ψ(λ̄) (anti-analytic) | 2.0 | 2.0 | 2.0 |

Check 3 is the strongest: the p, m equations are homogeneous of degree one in (p, m) with a, c fixed, so the test
field's λ = 0 matrix applied to the state's own p, m gives the p, m rows of the residual, at any state. It holds to the
rounding of the two computations, and misses by 0.01–0.1 when a's tangent is not removed.

**The yardstick ν, and why the checks changed.** In exact arithmetic M_ψ z_ψ − F_ψ is zero at every z, so its change
under a move of z by 10⁻³¹ (relative, two random moves, Δ kept) is the rounding of the two computations: that change
is ν (at least ε_dd max |F_ψ|). Checks 3 and 4 pass when they are within 10ν. They were first written with fixed
yardsticks, and those failed outside D = 8–30:

- Check 3 perturbed every unknown by 10⁻⁶ and compared with |F_ψ|; its control had to miss by 10⁻⁸ of |F_ψ|. At
  D = 45 the horizon block amplifies that perturbation to |F_ψ| = 8.1×10⁷, and the control's miss (0.011) was only
  1.3×10⁻¹⁰ of it; at D = 67 (the Mac) |F_ψ| was 1.4×10²⁵ and neither mode's miss showed. Each unknown is now moved by
  10⁻⁸ divided by its column's largest entry in the mode matrix, so that every row of F moves by about 10⁻⁸.
- Check 4 compared |M_ψ(0) z_ψ| / |z_ψ| with 10³ |F|. It failed at D = 45 (7.7×10⁻²² against |F| = 9.4×10⁻²⁶) and
  at D = 4 (the Mac: 1.5×10⁻²⁶ against 3.3×10⁻³⁰). A second form after the first D = 45 run, with the product's
  rounding ε_dd max_i Σ_j |M_ij||z_j| (3.8×10⁻³⁰ at D = 45), failed too. At D = 45 the excess sits only in the horizon
  block's ψ rows (3.3×10⁻²², the other blocks at most 3×10⁻²⁸), whose entries are at most 18. It is rounding noise of
  the mode matrix there: moving z by 10⁻³¹ relative moves M_ψ z_ψ by 2–5×10⁻²² while F_ψ moves by 4×10⁻²⁵ (a smooth
  change). That is what ν measures.

The records of the earlier forms are in `modecount_test_check.jsonl`: its first four lines (cloud) and the Mac's D = 67 and
D = 4 runs (`modecount_test_check_D4_mac.log`, `_D67_mac.log`). The counts do not use
checks 3 and 4.

**A discrepancy found on the way.** The count at D = 8.25 committed with the mode count (`hp/data/modecount_D8.2500_M32*`,
not used in MODE_COUNT.md's results) does not reproduce: its cached determinants differ from the current kernel's by
0.1–0.5 in ln|det M| at real λ (the signs agree), and the kernel and the crate of that commit, rebuilt, give the
current values bit for bit. At D = 30 the cached values reproduce exactly. So those cached D = 8.25 values came from a
different state or layout than the one saved with them; that count should be rerun before it is quoted.

## 3. Counts

`hp/modecount_test.py` runs the argument-principle count of `hp/modecount_wind.py` (the same contour network,
sampling, candidate poles and trend; notes/MODE_COUNT.md section 2) on det M_ψ, with λ = 0 (and its copies 2ikω) as
the only expected zero. The candidate poles are the full problem's: the test field's p goes through the same horizon
solves. Records: `hp/data/modecount_test_results.jsonl`; logs `hp/data/modecount_test_<tag>.log`.

| D | M | region | windings of g (B0, U0, B1a, B1b, B1c, B1, U, full) | zeros of det M_ψ | evaluations |
|---|---|---|---|---|---|
| 8.25 | 32 | −0.08 < Re λ < 6.70, \|Im λ\| < 3.51 (1.5ω) | 0, 0, 0, 0, 0, 0, 0, 0 | λ = 0 only | 87 (2.5 s each) |
| 16.76 | 32 | −0.06 < Re λ < 6.26, \|Im λ\| < 4.30 | 0, 0, 0, 0, 0, 0, 0, 0 | λ = 0 only | 92 |
| 30.07 | 32 | −0.15 < Re λ < 6.18, \|Im λ\| < 4.80 | 0, 0, 0, 0, 0, 0, 0, 0 | λ = 0 only | 94 |
| 45.05 | 32 | −0.13 < Re λ < 6.28, \|Im λ\| < 5.06 | 0, 0, 0, 0, 0, 0, 0, 0 | λ = 0 only | 101 (4–7 s each) |

On the real axis det M_ψ keeps one sign from aL to aR at every D (no real growth rate in the symmetric sector), and
the strips U count none in the other sector.

**Planted controls** on the same det M_ψ (`--plant`, the evaluations shared), D = 8.25: a pair planted at 3.1 ± 1.0i
is counted in B1c, B1 and full (2 each, 0 elsewhere); a pair at 1.0 ± 3.0i in U (1, its upper member) and full (2).

**Recounted with refined edges, and in a second layout** (T78, the Mac, 2026-10-27; `raw/T78` bdc6627). The count
now refines each edge until a verification pass adds nothing new (`hp/modecount_wind.py`, main 5c76ade); at D = 4 it
finds the pair −0.1123 ± 0.1003i in B0 that the single pass missed, and `--mutate=one_verify` restores the miss
(`raw/T77`). Each state was recounted in its own layout and moved to a second node layout by `hp/relayout.py`
(4eff588), which reproduced Δ to the last digit:

| D | first layout | second layout | windings of g, both layouts | largest step (first, second) | relayout \|F\| |
|---|---|---|---|---|---|
| 8.25 | 10 nodes, 0.02 … 0.7 | 6 nodes, 0.1, 0.2, 0.3, 0.4, 0.55, 0.7 | 0 in every box | 0.39, 0.37 rad | 9.0×10⁻²⁸ |
| 16.76 | 8 nodes, 0.1 … 0.7 | 6 nodes, 0.05, 0.15, 0.3, 0.45, 0.6, 0.7 | 0 in every box | 0.22, 0.22 rad | 1.1×10⁻²⁸ |
| 30.07 | 5 nodes, 0.3 … 0.7 | 7 nodes, 0.1 … 0.7 | 0 in every box | 0.23, 0.23 rad | 4.6×10⁻²³ (its 10⁻²⁴ not reached) |
| 45.05 | 5 nodes, 0.3 … 0.7 | 7 nodes, 0.15 … 0.7 | 0 in every box | 0.20, 0.20 rad | 1.1×10⁻²⁵ |

Every edge is resolved (every step below π/4, no verification mismatch), and B0 holds only λ = 0 at every D. The
candidate poles inside U and full, and one inside U0 at D = 30.07, make det M's own winding negative there; g's is 0.
A pair planted 0.002 inside B0's left edge at D = 8.25 (−0.0785 ± 0.3i) is counted by the refined count (B0: 3)
and also by the single pass (3), so that plant does not separate the two; the pair at D = 4 does.

## 4. D = 4: the zeros located in two layouts

At D = 4 the count found zeros where D = 8.25–45 have none. They were located with `hp/modecount_test_zeros.py`
(Delves–Lyness power sums on Gauss–Legendre panels refined until every step of arg g is below π/4, Newton on g to
polish; its `--selftest` and a planted zero per box as controls), on the state `lowD_D4.0000_M48` (K = 9 nodes,
|F| = 3.3×10⁻³⁰, ω = 1.823617). Records: `hp/data/modecount_test_zeros.jsonl`; logs
`hp/data/modecount_test_zeros_D4_*_mac.log` (the Mac, 2026-10-25/26).

| box | zeros of det M_ψ | control |
|---|---|---|
| B0: −0.114 < Re λ < 0.193, \|Im λ\| < 0.912 | 0 (the exact zero), −0.112341115908 ± 0.100296125832i | planted 0.1 ± 0.5i found, the others unchanged |
| U0: −0.114 < Re λ < 0.193, 0.912 < Im λ < 2.735 | −0.0866036 + 0.9604110i, −0.0344254 + 1.8308635i, **+0.0388691 + 2.6865263i** | planted 0.05 + 1.5i found, the others unchanged |
| U1: −0.114 < Re λ < 0.193, 2.735 < Im λ < 4.558 | 2iω (3.6472334i), **+0.1052019 + 3.5262063i**, **+0.1622992 + 4.3643366i** | none run |

Two scans of g along lines (`hp/modecount_test_axis.py`, `hp/data/modecount_test_axis_s{0,1}_D4_mac.out`), with
Re λ from −0.7 to −0.02, find one zero each: −0.3651202796 on the real axis, and −0.0344254228 + 1.8308635107i on
Im λ = ω, the U0 zero (Im λ / ω = 1.00397). Gundlach and Martín-García's slowest decaying mode, −0.362 ± 0.012, is
within their error of the first. The revised checks of section 2 pass at D = 4
(`hp/data/modecount_test_check2_D4_mac.out`) and at D = 67.0115 (`_D67_mac.out`).

**Was a discrepancy.** The zeros with Re λ > 0 (+0.0389 + 2.6865i in U0, two more in U1) would be growing modes of
the test field, so in the two-field model extra unstable modes of the critical solution. That disagrees with Gundlach
(1997, Phys. Rev. D 55, 695, arXiv:gr-qc/9604019), whose review (arXiv:gr-qc/9712084) states that of the Choptuik
solution's perturbations as a free complex scalar field "only one of them, purely real, has Re λ > 0" (the 1997
paper's body has not been read here, only its abstract and the review). The U0 zeros' real parts rise with Im λ
(−0.087, −0.034, +0.039 at Im λ = 0.96, 1.83, 2.69), as spurious modes of a truncated discretization often do.

**The second layout.** The 9-node state does not converge at M = 64 (|F| stalls at 6.8×10⁻⁴,
`hp/data/modecount_test_zeros_D4_U0_M64_mac.log`), so the boxes were repeated on states with 11 nodes, 0.01 … 0.7
(kc 80, ks 200): `lowDn_D4.0000_M48` (|F| = 3.8×10⁻³¹) and `lowDn_D4.0000_M64` (|F| = 9.1×10⁻³⁰), ω = 1.8236169.
Logs `hp/data/modecount_test_zeros_D4_*_lowDn_D4.0000_M*_mac.log` (the Mac, 2026-10-25/26).

| box | zeros, M = 48 | zeros, M = 64 | control (M = 64) |
|---|---|---|---|
| B0 | 0 only | 0 only | none run |
| U0 | none | none | planted 0.05 + 1.5i found |
| U1 | not run | 2iω only (3.647233843506i) | planted 0.05 + 3.6i found, 2iω unchanged |
| U2: −0.114 < Re λ < 0.193, 4.558 < Im λ < 6.382 | none | none | planted 0.05 + 5.5i found |

The real-axis scan at M = 64 (`hp/data/modecount_test_axis_s0_lowDn_D4.0000_M64_mac.out`) finds −0.3651202372,
4×10⁻⁸ from the 9-node value. The scan along Im λ = ω (`_s1_lowDn_D4.0000_M64_mac.out`) finds no zero, where the
9-node layout had −0.0344 + 1.8309i.

**Resolved.** Every zero of the 9-node layout near the imaginary axis (the pair −0.1123 ± 0.1003i, the three in U0
and the two growing ones in U1) is absent in the 11-node layout at both M, while the exact zeros 0 and 2iω and the
real zero −0.3651 stay. Those zeros are artifacts of the 9-node layout. The U0 zero −0.0344 + 1.8309i was found there
by both the box and the line scan, so it was a zero of that discretization's determinant, not an error of the finder.
In the 11-node layout the test field has no growing mode in B0, U0, U1 or U2 (|Im λ| < 6.382), as the review quoted
above states. This is numerical evidence at two resolutions of one layout. It does not cover Im λ > 6.382, nor
Re λ < −0.114 off the lines Im λ = 0 and ω.

**The count's under-refinement.** The winding count of section 3 verifies each edge by one bisection and does not
refine further. At D = 4 its B0 and U0 edges kept steps of arg g of 1.2–1.8 rad (above its π/4), and it counted B0 as
holding only the exact zero, missing the pair at −0.1123 ± 0.1003i, which lies 0.0017 inside B0's left edge. The
planted count, which sampled the same edges more finely, counted it. The counts at D = 8.25–45 in section 3 were made
the same way, so their zeros near the boxes' edges are not certain either; the zero finder above refines until every
step is below π/4. Those counts also used one layout each, and after D = 4 each deserves a second-layout check before
it is quoted. *(Done, 2026-10-27: the recount in section 3.)*
