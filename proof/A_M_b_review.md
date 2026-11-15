# Independent review of the second start-value implementation (fixed_point_M_b.py)

Task (#867): an independent review of `solver/proof/fixed_point_M_b.py` and `solver/proof/fixed_point_M_b_check.py`,
the second implementation of the start-value contraction and its harness ([A_lemmas_M.md](A_lemmas_M.md) §2.5), in
the form of [A_M_review.md](A_M_review.md). Reviewer: the agent that wrote `start_cover_M2.py` (#484, #862). It wrote
neither file under review. The review began on the files at main 51dceb1f and was redone at main ecfc1b92, after #866
added `start_data` and the harness's `cert` command (6b92ee40). Every finding refers to ecfc1b92. The reviewer's
scripts are in [`solver/proof/review_867/`](../solver/proof/review_867/), and their logs in raw `I867-mac/`, f2c232d2.

## Contents

- [Verdict](#verdict)
- [Summary of the checks](#summary-of-the-checks)
- [1. The system, the series and the residual](#1-the-system-the-series-and-the-residual)
- [2. The majorants, the forcing and the Lipschitz matrix](#2-the-majorants-the-forcing-and-the-lipschitz-matrix)
- [3. The kernel bound](#3-the-kernel-bound)
- [4. The componentwise test and the weight exponent](#4-the-componentwise-test-and-the-weight-exponent)
- [5. The start set, the start data and the comparisons](#5-the-start-set-the-start-data-and-the-comparisons)
- [6. What the two implementations share](#6-what-the-two-implementations-share)
- [7. The authors' runs and mutations, rerun](#7-the-authors-runs-and-mutations-rerun)
- [8. Findings](#8-findings)
- [9. What I did not check](#9-what-i-did-not-check)
- [10. Reproduce](#10-reproduce)

## Verdict

The claim of A_lemmas_M.md §2.5 holds. I found no error in any bound the proof needs: the system in z, the series, the
residual, the forcing, the majorants, the kernel bound, the componentwise test and the start data built from it. What
the code proves is what §2.5 says:

> Lemma FP(M) with the diag splitting on a ray, or on every ray of a sector, at a point, on a real box and on a
> complex box of parameters, with the fixed point bounded componentwise, |η_i(z)| ≤ w_i|z|^{k_i} on 0 < |z| ≤ R.

The start set of lemma 6b₁ and the point certificates' start data follow from it by jets of Y_N and Cauchy's estimate,
as §2.5 describes, and they overlap the first implementation's in every component in my reruns.

No finding changes a number of §2.5 or the verdict of a stored run. Two findings need action:

- F1: none of the authors' checks tests the new kernel bound. With one integration by parts too many in it, a
  bound that the true integral exceeds by up to 0.68%, all five of their checks that use it still pass. The bound
  itself is right: I derived it again, and quadrature stays below it in all 1417 cases the module asks for.
- F2: the plain command line reads its point at 53 bits. At the M = 5 certificate's centre it then proves the
  lemma on a ball of radius 6×10⁻¹⁷ with N = 78, not at the point with N = 105. No stored run uses that path.

F3 is a gap in the kernel bound's stated hypotheses. F4 to F7 are about the text and the review's own independence.

## Summary of the checks

| check | how | verdict |
|---|---|---|
| the system in z is `inner_np.py`'s | by hand (§1); K1 (a), (b) | OK: 3×10⁻¹⁵⁰ against the reviewer's transcription, 2×10⁻¹⁶ against `inner_np.Inner.rhs` |
| the series recursion | reading; K1 (c), (d) | OK: low orders of the residual vanish to 4.5×10⁻¹⁵³, sums agree with `inner_np.series_at` to 6×10⁻¹⁷ |
| the residual numerators and their structural orders | by hand; K1 | OK |
| the forcing | reading; K3 | OK: its true size is 0.85 to 0.9998 of F_i |
| the majorants and the Lipschitz matrix | reading; K3 | OK: 0 of 283 entries below the true derivative; the largest ratio is 0.9995 |
| the kernel bound, its exponent on complex boxes, the supremum over r | by hand (§3); K2 | OK: 0 of 11336 values above the bound, the largest 0.999815 of it; F3 |
| the componentwise test and the refined set | reading (§4) | OK |
| k = N − 2 and the identification with y⁺ | by hand (§4); K4 | OK: the guard refuses k = N − 1 |
| `start_set`, `compare` and the stored set's layout | reading `lemma6b1_cover_M.py`, `validated_int.py` | OK |
| `start_data` (#866) | reading | OK |
| the harness, `m1`, `lip` and `cert` | reading; rerun | OK; F6 |
| the command line | K4 | F2 |
| §2.5's list of what is shared | reading both implementations | accurate; F7 |
| the authors' 38 runs | rerun (§7) | all 38 exit as required; §2.5's numbers reproduced |
| the reviewer's mutation | K2 `--mutate=offbyone` | fires; the authors' checks do not notice it (F1) |

## 1. The system, the series and the residual

I derived the system in z from `inner_np.py`'s equations in ξ, which read

```
ξ² D′ + iμω ξ D = [C S],    ξ² S′ + iμω ξ S = [C ξD′] + [[C E] D] − [C D],    ξ C′ = [C E],    E = [D D]/2.
```

With z = ξ², D = ξP, S = zQ and d/dξ = 2ξ d/dz they become 2zP′ = [CQ] − (1 + s)P, 2zQ′ = −(2 + s)Q + 2[CP′] + [GP]
and C′ = G/2 with G = [CE], E = [PP]/2, which is the module's docstring. The passage to V = Q − (1 + s)P and
W = V + [(C − 1)Q], the principal part 1/(2z²) − (3 + 2s)/(2z) and the remainder Φ_V are right as well: the identity
(2 + s)Q + (1 + s)W = (3 + 2s)V + (2 + s)(1 + s)P + (1 + s)[(C − 1)Q] holds. The projections agree, since each binary
product is projected onto its target set in both codes. The recursion for d_j, q_j, g_j follows from these equations
order by order, and it is the recursion of `inner_np.series_at`.

K1 (`rv_system.py`) checks this without relying on my reading of the module. For any polynomials P, Q, C the defects in
the three ξ-equations are, in the module's residual numerators,

```
ξ²D′ + sξD − [CS] = −z N_P,     ξC′ − [CE] = −2z N_C,
ξ²S′ + sξS − [CξD′] − [[CE]D] + [CD] = ξ(−N_V − (1 + s) z N_P + [C N_P]).
```

The [C N_P] term appears because the module writes 2[CP′] as [CW]/z.

- (a) With random polynomials of degree 6 and a complex ω, where every term is of size 1, the three identities hold to
  3×10⁻¹⁵⁰ relative at a point on the ray and at a point off it.
- (b) The same polynomials against `inner_np.Inner.rhs`, the solver's own code, agree to 2×10⁻¹⁶.
- (c) For the module's series at the M = 5 zero (N = 105) the low orders of the residual contain 0, with balls of size
  4.5×10⁻¹⁵³ at |z| = R, and the true defect at |ξ| = 0.9 r₀, of size 1.1×10⁻⁵⁹, satisfies the identities to 4×10⁻⁹⁵.
- (d) The module's 105 terms agree with `inner_np.series_at`'s sums to 6×10⁻¹⁷ at that point.

Its controls fire: with the module's own factor on [GP] set to 1 + 10⁻⁹, (a) fails, and with −1, (a) and (b) fail.
The factor 1 + 10⁻⁹ is below the double-precision resolution of (b). M = 3 passes too.

So the residual numerators are the defects of the solver's system, and a series whose low-order residual vanishes is
that system's formal solution. The structural orders are right: N_P and N_V vanish below order N, and N_C below
order N − 1, because order j of the C equation needs g_{j+1}.

## 2. The majorants, the forcing and the Lipschitz matrix

A majorant is a polynomial in t = |z| with nonnegative coefficients, carried with its derivatives in the components of
η. I read every line of `MJ`, `mconv`, `Setup._sups` and `Setup.lipschitz` against the system.

- |P_N + p| ≤ Σ|d_j|R^j + w_p t^k, |V_N + v| ≤ t Σ_{j≥1}|V_j|R^{j−1} + w_v t^k and |C_N − δ + c| ≤ t Σ_{j≥1}|g_j|R^{j−1} +
  w_c t^{k+1} hold on |z| ≤ R. The constant terms of V_N and of C_N − δ vanish identically, as the code assumes.
- The product rule in `MJ.__mul__` and the rows of Φ_P, Φ_V and Φ_C are those of §1, with |1 + s| and |(2 + s)(1 + s)|
  taken as upper bounds over the parameter set. The label term (3 + 2s)V is in the principal part and not in the rows.
- The slow rows integrate term by term, and `slow_norm` refuses an exponent below the weight.
- The forcing sums the residual's coefficients from the structural order on, after checking that the lower ones
  contain 0.

K3 (`rv_lipschitz.py`) tests the result from below. I linearized the operator at η = 0 by hand, applied it to the
direction η_j = z^{k_j}, whose norm is 1, and evaluated the image at |z| = R, the fast rows by quadrature. That is a
lower bound of the true norm, so it must not exceed L_ij.

| | M = 5, N = 105, k = 103 | M = 3, N = 53, k = 51 |
|---|---|---|
| positive entries of L | 283 | 133 |
| entries below the true derivative | 0 | 0 |
| largest true derivative / bound | 0.9995 | 0.9977 |
| entries where it is above half the bound | 277 | 121 |
| entries where L = 0 and the derivative is not | 0 | 0 |
| size of T(0)_i at \|z\| = R, as a fraction of F_i | 0.847 to 0.9998 | 0.867 to 0.9991 |
| the same as a fraction of the proved radius of η_i | at most 0.922 | at most 0.883 |

So the matrix is not only an upper bound by reading: nearly every entry is within a factor 2 of the truth, and a
missing term would show. Two controls confirm that. Without the label term (2 + s)(1 + s)P in the majorant
(`--mutate=nolabel`, written down before the run), four entries fall below the true derivative, v₋₅ ← p₋₅ by a factor
3.4, while the module still reports a contraction, 0.0723. Without [GP] (`--mutate=gpdrop`), 17 entries fall below it,
by up to a factor 1.8, and the authors' Lipschitz comparison notices this mutation as well.

## 3. The kernel bound

On a ray |K_μ(t, z)| = e^{−a(1/ϱ − 1/r)}(ϱ/r)^e with e = 3/2 − μ Im ω, exactly, since t/z is real and positive. With
1/ϱ = 1/r + τ the weighted integral of ϱ^m is r^{m+2} I_n, I_n = ∫₀^∞ e^{−aτ}(1 + rτ)^{−n} dτ, n = m + 2 + e. One
integration by parts gives 1 = a I_{n−1} + (n − 1) r I_n, and I_{n−1} ≥ I_n, so (a + (n − 1)r) I_n ≤ 1. I confirm the
derivation, with one condition that the text omits (F3).

The code is right where the text is loose.

- `fast_norm` requires m ≥ −1 for the integrand, so the denominator a + (m + 1 + e)R is at least a.
- It uses the smallest exponent over the labels and the parameter set, e_min = 3/2 − M sup|Im ω|, and refuses a
  negative one. A larger e makes the integral smaller, so this is the right direction. On the M = 5 Cauchy box it is
  1.4809.
- The norm takes the supremum over r ≤ R of r^{pw}/(a + (m + 1 + e)r), where pw is the surplus of the integrand's
  exponent over the weight. For pw ≥ 1 that function increases with r, and the code evaluates it at R. For pw = 0 it
  decreases, and the code uses its limit 1/a.
- On a sector of rays it uses the smallest rate a, at an end of the sector, and checks that the sector stays inside
  π/2 < arg z < 3π/2.

K2 (`rv_kernel.py`) tests `fast_norm` against quadrature. The module's own calls are recorded during five runs: the
M = 5 point, complex box, sector and k = 40 runs, and the M = 3 run. That gives 1417 distinct cases, with pw from 1
to 377. For each, the true value at r = R, R/2, R/100 and R/10⁴, with e = e_min and e_min + 0.05, must not exceed the
bound.

| | the module's bound | the reviewer's mutation |
|---|---|---|
| values above the bound, of 11336 | 0 | 2834 |
| largest true value / bound | 0.999815 | 1.006761 |
| the identity 1 = a I_{n−1} + (n − 1) r I_n | holds to 2×10⁻⁴¹ | the same |

The bound is sharp to 0.02%. The mutation is the reviewer's own: the denominator a + (m + 2 + e)R, one integration by
parts too many. It was written down before the run that the quadrature must exceed it, and it does. What the
authors' checks make of it is F1.

## 4. The componentwise test and the weight exponent

The test is sound. For η in X_w the mean value inequality on the convex set X_w gives ‖(Tη)_i − (T0)_i‖ ≤ (L(w)w)_i, so
F + L(w)w < w makes T map X_w into itself, and it contracts in the norm max_i ‖η_i‖/w_i with constant
max_i (L(w)w)_i/w_i. The refinement is sound too: if w′ ≤ w and F + L(w)w′ ≤ w′, the Picard iterates from 0 stay in the
closed set X_w′, so the fixed point lies in it. The code checks both inequalities, w′ ≤ w and the constant below 1 in
ball arithmetic. The Perron vector and the scale of w come from floating point and only propose the vectors.

The weight exponent k = N − 2 is the largest that keeps y⁺ in the uniqueness class. The c components carry the weight
k + 1, and y⁺ − Y_N = O(z^N) must be strictly smaller than |z|^{k+1} near 0, which needs k + 1 < N. The identification
then runs as in the first implementation: on a small enough radius R′ the difference y⁺ − Y_N lies in X_w, the
restriction of the fixed point to |z| ≤ R′ is the fixed point there, and the two agree. That step needs every bound to
be non-decreasing in R, which holds for the sharp kernel bound as well (§3: increasing for pw ≥ 1, constant for
pw = 0). The module refuses any k above N − 2: with N = 105 it proves at k = 103 and raises at k = 104 (K4).

## 5. The start set, the start data and the comparisons

`start_set` proves the contraction at the centre and on the complex box, whose real parts reach h + c and whose
imaginary parts reach c. The values are those of Y_N at the centre with the η discs of the point run, and the columns
are jets of Y_N over the real box plus Cauchy's estimate from the complex box. The variables P, V and κ(C − δ) with
κ = 100, and the two last states ω and A, are those of `lemma6b1_cover_M.start_data_u`. `stored_set` reads the stored
Lohner set as c + Qq with the real parts stacked above the imaginary ones, which is `validated_int.Lohner.box` and
`var_box`. So an overlap of the two is a meaningful test.

`start_data` (#866) gives the state (D, S, C) at ξ₀ = r₀e^{iθ}, on the edge |z| = R, over a real set and its Cauchy
box. Its radii are right: |D − D_N| ≤ r₀|p|, |S − S_N| ≤ R(|v| + |1 + s||p|) with |1 + s| taken over the set in
question, and |C − C_N| = |c|. Cauchy's estimate is applied to S − S_N as a whole, so the ω in 1 + s is covered.

The radius of η near the zero depends strongly on the point. §2.5's first table is at the stored start set's centre,
which is the certificate's p̂ cut to 38 decimals, 9×10⁻³⁹ away. There sup|η_p| is 7.5×10⁻⁷⁵. At the certificate's
50-digit p̂ the same run gives 4.3×10⁻⁸⁷ (K4). Both are right, and §2.5 names its centre.

## 6. What the two implementations share

§2.5's list is accurate. `fixed_point_M_b.py` imports python-flint and the standard library only. Beyond the list:

- both take the kernel rate a = −cos(arg z)/2 and its minimum over a sector at an end, from the same hand argument;
- both drop the residual's low-order coefficients after testing only that their balls contain 0, which §2.5 says;
- the harness, not the proof code, imports `fixed_point.py`, `fixed_point_M.py` and, for `cert`, `start_cover_M2.py`,
  `certify_M.py` and `certify_M2.py`.

The comparison of the Lipschitz matrices (`lip`) runs the second implementation with the kernel bounds of §4.4
(`kernel="classic"`), as its docstring says. So it compares the majorants and not the new kernel bound.

## 7. The authors' runs and mutations, rerun

`review_867/rerun_runs.sh` reran the 32 runs of `I427-mac/run_i427.sh` and the 6 of `I866-mac/run_i866.sh`. All 38 exit
with the status they must have, which is the authors' status in every case. The 32 gave the same statuses on the
files before #866.

Every number of §2.5 that I looked for is in my logs with the same digits:

| §2.5 | reviewer's rerun |
|---|---|
| contraction constants 0.1000, 0.1002, 0.1135, 0.1137, 0.2355 | the same |
| start set at s₀: 21 of 21 values and 126 of 126 derivative entries overlap, 21 balls inside | the same |
| its radii, 3.2×10⁻¹⁰³ and 2.7×10⁻¹⁶ | 3.22×10⁻¹⁰³, 2.72×10⁻¹⁶ |
| M = 3: 0.1674, 0.1681, 15 of 15, 60 of 60 | the same |
| Lipschitz ratios from 0.88, 0.86, 0.81 | 0.8823, 0.8623, 0.8142 |
| planted: radius 3.2 to 3.4 times the distance, 1.05 to 1.16 near \|z\| = R | 3.38 and 3.43, 1.13 and 1.16 at μ = 5 |
| two truncations: 0.83 and 0.68 | 0.830, 0.682 |
| the certificates: 0.0984, 0.0987 and 0.0886, 0.0888; 1197 of 1197 jets | the same |
| mutations: row sums 19.3, a miss by 2.3×10⁻¹³, a factor 820 | 19.276, 2.29×10⁻¹³, 819.6 |

Under `--mutate=sign`, `scale` and `rho0` my runs report 2 of 21 values and 12 of 126 derivative entries overlapping.
Those are the ω and A rows, which do not depend on the series, so this is §2.5's "0 of 19 and 0 of 114".

The planted check tests P and V of the linear problem at one point per run. The nonlinear rows are tested by the
comparisons with the first implementation, and now by K3.

## 8. Findings

No finding changes a number of §2.5 or a stored run's verdict.

### F1. None of the authors' checks tests the new kernel bound

Check coverage. The bound I_n ≤ 1/(a + (m + 1 + e)r) is the one piece of the proof that is new in this implementation
and not in the first. With the reviewer's mutation in `fast_norm`, a + (m + 2 + e)R in the denominator, the true
integral exceeds the bound in 2834 of 11336 values, by up to 0.68%. The authors' checks that use `fast_norm` then give:

| authors' check | with the mutated bound |
|---|---|
| planted, M = 5, at the start point | PASS |
| planted, M = 5, at \|z\| = R | PASS |
| two truncations, N = 60 and 105 | PASS, ratio 0.837 |
| two truncations, N = 95 and 105 | PASS, ratio 0.686 |
| the start set against the stored one | PASS |

The contraction constants move in the third digit: 0.1000 to 0.0997, 0.1002 to 0.0999, 0.1135 to 0.1130, 0.2355 to
0.2338, and 0.1674 to 0.1662 at M = 3. The Lipschitz comparison does not use the sharp bound (§6). The reason is
margin: the planted enclosure is 5% to 16% wider than the truth and the bound is sharp to 0.02%, so only a direct test
can see an error of this size. As it stands the bound rests on its derivation, which is right (§3). *Proposed:* add
the quadrature test to the module's own checks, with this mutation. `rv_kernel.py` can be taken over.

### F2. The plain command line reads its point at 53 bits

A documented command. `main()` builds `x = [acb(arb(v)) ...]` from `--x` before `Setup` sets the working precision.
On the plain path, `python3 fixed_point_M_b.py --M=5 --x=...`, which the docstring calls "the contraction at a point",
the point is therefore a ball of radius 6×10⁻¹⁷. At the M = 5 certificate's centre (K4):

| `--x` read at | N | k | contraction constant | sup\|η_p\| at \|z\| = R |
|---|---|---|---|---|
| 53 bits, the plain command line | 78 | 76 | 0.1250 | 1.2×10⁻³³ |
| 512 bits, as `--start`, `--planted` and `--two` do | 105 | 103 | 0.1000 | 4.3×10⁻⁸⁷ |

The run still proves what it prints, for the ball and so for the point. It is not the run the docstring describes, and
a user comparing its N with §2.5's 105 would be misled. The half-widths of `--box` and `--cbox` are read the same
way. None of the 38 stored runs uses this path, and the harness sets the precision first. This is the defect of
`fixed_point_M.data_table` again (#484, discrepancy 2). *Proposed:* set the precision at the top of `main()`.

### F3. The kernel bound's hypotheses omit one condition

Proof text. §2.5 and the module's docstring state the bound "for a > 0, e ≥ 0 and m ≥ −2". The step from
(a + (n − 1)r) I_n ≤ 1 to the bound needs a + (m + 1 + e)r > 0, which fails for m = −2, e < 1 and r ≥ a/(1 − e). The
code never uses that case: it requires m ≥ −1 and treats the weight-equal case by 1/a (§3). The text also does not say
that the supremum over r ≤ R is at r = R only when the integrand's exponent exceeds the weight. *Proposed:* state the
bound for m ≥ −1, and add one sentence on the supremum.

### F4. A comparison in §2.5 mixes two centres

Text. §2.5 says of its last table row, at p̂, that `fixed_point_M.py` gives 0.3671 for it. That number is from the
Lipschitz comparison at x̂ = `fixed_point_M.ZEROS[5]` (`lip_M5_sector.log`), in the first implementation's weights,
where the second gives 0.3668 with the kernel bounds of §4.4. *Proposed:* name the centre.

### F5. The run's label for the complex box is loose

Cosmetic. `start_set` prints "complex box +-(h + c)(1 + i)", and §2.5 writes p̂ ± (10⁻¹⁸ + 10⁻³)(1 + i). The imaginary
half-width is c, not h + c. The module also imports `math` and does not use it.

### F6. The `cert` command takes the certificate's sets from the reviewer's code

Independence. `fixed_point_M_b_check.py cert` rebuilds the four parameter sets with `start_cover_M2.cert_data`, which I
wrote. I am therefore not an independent reviewer of that function, and its two users now share it.

### F7. `Setup` picks N from bounds over the set

Minor. With N not given, `Setup` takes the smallest term of sup|d_j|R^j over the parameter set, not at its centre as
Lemma FP(M) says. Any N is admissible, so nothing is unsound. `start_set` and `cert` take N from a point run first, so
§2.5's N are the centre's. F2 is where the difference shows.

## 9. What I did not check

- Braaksma's theorem, and the hand argument that the residual vanishes below its structural orders. I checked only that
  those balls contain 0 and are of size 10⁻¹⁵³.
- The first implementation, `fixed_point_M.py`, and the stored Lohner sets beyond their layout. `lohner_c1` is not
  recomputed by either side.
- `start_cover_M2.cert_data` (F6), `certify_M2.series2`, and the comparison of the second-order jets beyond rerunning
  it.
- python-flint and FLINT, Kummer's U in arb, and mpmath's quadrature, which K2 and K3 use. K2 checks it against the
  identity of §3 to 2×10⁻⁴¹.
- An integration of y⁺ to the start point by other means, as K4 of A_M_review.md did at M = 3 for the first
  implementation.

Two of my checks are one-sided. K3 bounds the true Lipschitz norm from below, at η = 0, in monomial directions and on
the central ray. It can show a bound that is too small, and it shows the bounds are tight, but that L is an upper
bound over all of X_w rests on reading the majorant calculus (§2). K2 covers the cases the module asked for in five
runs, not every case it could ask for.

## 10. Reproduce

From `solver/proof/review_867/`, with python-flint 0.9.0, mpmath and numpy, one process. RAW is a clone of the raw-data
repository.

```
python3 rv_system.py                      # K1, seconds; RV_SYSTEM: PASS, exit 0; also --M=3
python3 rv_system.py --mutate=scale       # exit 1; also --mutate=sign
python3 rv_kernel.py                      # K2, 4 min; RV_KERNEL: PASS, exit 0
python3 rv_kernel.py --mutate=offbyone --ray=RAW/I38-macm3/lemma6b1_runs_M5_cert/ray.json    # exit 1, 5 min; F1
python3 rv_lipschitz.py                   # K3, 2 min; RV_LIPSCHITZ: PASS, exit 0; also --M=3
python3 rv_lipschitz.py --mutate=nolabel  # exit 1; also --mutate=gpdrop
python3 rv_cli_precision.py --ray=RAW/I38-macm3/lemma6b1_runs_M5_cert/ray.json               # K4, seconds; F2
```

From `solver/proof/`, the authors' 38 runs into a new folder OUT, 8 min:

```
bash review_867/rerun_runs.sh python3 RAW OUT     # RERUN: 38 runs, 0 with another exit status than required
```
