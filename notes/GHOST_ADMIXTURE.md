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
| 3 | identity M_ψ(0; z) z_ψ = F_ψ(z) at a state perturbed by 10⁻⁶ | 1.0×10⁻²² of \|F_ψ\| | 2.4×10⁻²³ | 8.5×10⁻²⁵ |
| | control: a's tangent kept (mode 2) | 1.6×10⁴ | 1.6 | 1.3×10⁻¹⁰ (see below) |
| 4 | \|M_ψ(0) z_ψ\| / \|z_ψ\| at the solution (λ = 0 is a zero) | 4.9×10⁻²⁸ (\|F\| 8.6×10⁻²⁸) | 9.7×10⁻²⁵ (\|F\| 6.0×10⁻²⁵) | **7.7×10⁻²² (\|F\| 9.4×10⁻²⁶): fails** |
| 5 | Cauchy–Riemann on ln det M_ψ at λ = 1.3 | 2.1×10⁻⁷ | 2.0×10⁻⁷ | 2.8×10⁻⁷ |
| | control: det M_ψ(λ̄) (anti-analytic) | 2.0 | 2.0 | 2.0 |

Check 3 is the strongest: the p, m equations are homogeneous of degree one in (p, m) with a, c fixed, so the test
field's λ = 0 matrix applied to the state's own p, m gives the p, m rows of the residual, at any state. It holds to
double-double rounding, and fails at O(1) when a's tangent is not removed.

**Check 4 fails at D = 45, and control 3 needed a new threshold there.** Two discrepancies, reported here and not
resolved:

- *Control 3.* Its threshold was a fixed 10⁻⁸ of |F_ψ|. At D = 45 the perturbed state's |F_ψ| is 8.1×10⁷ (the
  horizon block amplifies the 10⁻⁶ perturbation), so the control's miss, 0.011, is 1.3×10⁻¹⁰ of it: under that
  threshold the control did not fire. After that run the threshold was changed to 10³ times the check's own ratio
  (8.5×10⁻²⁵ at D = 45); the control's ratio is 14 orders of magnitude above the check's. The first D = 45 record in
  `modecount_test_check.jsonl` is the run under the old threshold.
- *Check 4.* |M_ψ(0) z_ψ| / |z_ψ| is 7.7×10⁻²² at D = 45, against |F| = 9.4×10⁻²⁶; it fails both the original
  criterion (within 10³ of |F|) and the one added after the first D = 45 run (within 10³ of the product's rounding,
  ε_dd max_i Σ_j |M_ij||z_j| = 3.8×10⁻³⁰). The excess sits only in the horizon block's ψ rows: there
  max |M_ψ z_ψ| = 3.3×10⁻²², while the other five blocks give at most 3×10⁻²⁸, and the horizon block's ψ entries are
  at most 18 (row sums at most 33). So it is not the rounding of the product. The residual's own horizon rows at the
  same state are 9.4×10⁻²⁶, and moving a_h by 10⁻³¹ moves them by 1.5×10⁻²⁴. The suspected cause is the horizon
  series' amplification (x_m^−(n−1) = 3.3×10⁶ at D = 45 against 1.6×10⁴ at D = 30, with x_m = 0.7) acting on the mode
  matrix's tangent computation, the mechanism behind the double-double floor of the continuation at large D. It is not
  shown: the null residual grew 800 times from D = 30 to 45, the amplification 210 times.

λ = 0 is still a zero by the determinant: the smallest pivot is 9.1×10⁻²⁶ at λ = 0 against 4.5×10⁻¹³ and 5.6×10⁻¹³ at
λ = ∓0.1, and det M_ψ changes sign between them. The count below does not use check 4.

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
