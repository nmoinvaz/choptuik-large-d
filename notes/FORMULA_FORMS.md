# The formula's shorter forms by range, and the earlier power series

Moved verbatim from README.md (commit ea42b3e, lines 210–245, 414–445) on 2026-11-04, in the reorganization of
[README_PLAN.md](README_PLAN.md); the README keeps a pointer where each block stood. Links were made relative to this
folder.

<details>
<summary>Shorter forms for one range of D: plain series in 1/D with fewer terms, per range</summary>

On a narrower range a plain series Δ ≈ c₀ + c₁/D + c₂/D² + … with fewer terms does as well as the rational formula. The
table gives, per range, the fewest terms that reach each accuracy, the coefficients, and the error margin. Up to D = 61
the margin is the largest miss over the computed points in the range (an interpolation error, since the series is fitted
on those points). Above D = 61 there are no computed points: the series is fitted on the points from half the range's
lower end up, and the margin is the spread of its values across fits that differ in the lower cut and in one extra term,
plus the fit's own residual (`cd hp && python3 short_forms.py`; `--selftest` runs it on a planted analytic function with
controls). That is a fit spread, not a bound; it assumes Δ is analytic in 1/D
([Is the period analytic in 1/D?](../README.md#is-the-period-analytic-in-1d)), and `--selftest` shows it widens a
hundredfold when a non-analytic term is present. Coefficients are given to two digits past the margin.

| D | accuracy | terms | Δ ≈ | error margin |
|---|---|---|---|---|
| 10–20 | 10⁻⁴ | 4 | 1.651778 + 9.85735/D − 14.4024/D² + 27.6111/D³ | 2.0×10⁻⁵ |
| | 10⁻⁶ | 5 | 1.64636229 + 10.169508/D − 21.027033/D² + 88.933312/D³ − 208.97821/D⁴ | 6.3×10⁻⁷ |
| | 10⁻⁸ | 7 | 1.6443656219 + 10.3223902/D − 25.77585357/D² + 165.2685938/D³ − 875.4410611/D⁴ + 2973.759667/D⁵ − 5229.273842/D⁶ | 3.9×10⁻⁹ |
| 20–40 | 10⁻⁴ | 3 | 1.648396 + 9.94459/D − 13.442/D² | 5.7×10⁻⁵ |
| | 10⁻⁶ | 5 | 1.64418027 + 10.331218/D − 25.622295/D² + 148.39562/D³ − 504.78463/D⁴ | 1.4×10⁻⁷ |
| | 10⁻⁸ | 6 | 1.6439764681 + 10.36020445/D − 27.24883172/D² + 193.4099753/D³ − 1119.311266/D⁴ + 3311.999159/D⁵ | 6.7×10⁻⁹ |
| 40–61 | 10⁻⁴ | 3 | 1.6450284 + 10.19239/D − 18.04099/D² | 5.7×10⁻⁶ |
| | 10⁻⁶ | 4 | 1.644077767 + 10.3363896/D − 25.2423218/D² + 118.863223/D³ | 9.2×10⁻⁸ |
| | 10⁻⁸ | 5 | 1.6439252935 + 10.3670057/D − 27.53181976/D² + 194.4269583/D³ − 928.7010788/D⁴ | 1.8×10⁻⁹ |
| 70–100 (fit spread) | 10⁻⁴ | 4 | 1.644121 + 10.3299/D − 24.9162/D² + 113.486/D³ | 8.6×10⁻⁵ |
| | 10⁻⁶ | 7 | 1.64389549 + 10.37469/D − 28.330517/D² + 236.72558/D³ − 2104.6691/D⁴ + 15445.597/D⁵ − 63511.93/D⁶ | 2.8×10⁻⁷ |
| | 10⁻⁸ | 9 | 1.6438938283 + 10.37524957/D − 28.41118719/D² + 243.2171123/D³ − 2421.974291/D⁴ + 25011.26095/D⁵ − 234683.1799/D⁶ + 1615767.159/D⁷ − 5754580.832/D⁸ | 8.0×10⁻⁹ |
| 100–200 (fit spread) | 10⁻⁴ | 5 | 1.643914 + 10.3694/D − 27.7262/D² + 201.332/D³ − 1020.47/D⁴ | 4.4×10⁻⁵ |
| | 10⁻⁶ | 8 | 1.64389387 + 10.37523/D − 28.407016/D² + 242.7195/D³ − 2385.3469/D⁴ + 23303.804/D⁵ − 185367.37/D⁶ + 807821.97/D⁷ | 4.3×10⁻⁷ |
| | 10⁻⁸ | | not reached with 12 terms | |

Above D = 61 the series need more terms than below it, not fewer, because each fit is an extrapolation whose
uncertainty grows with the distance from the last computed point; the coefficients of the longer forms are not
physical constants (they are correlated, and shift with the range), only the function is.

</details>

<details>
<summary>Power series in 1/D for D ≥ 10 (the earlier form)</summary>

```math
\Delta(D) = \Delta_\infty + \sum_{k=1}^{10} \frac{c_k}{D^k},
\qquad
\gamma(D) = \frac{1}{2} + \sum_{k=1}^{10} \frac{g_k}{D^k},
\qquad
\Delta(D) \approx 1.64389 + \frac{10.37}{D} - \frac{28.4}{D^2} + \frac{239}{D^3} - \cdots
```

Fitted on 10 ≤ D ≤ 30.2 (`solver/formula.py`); it reproduces the points there to 1.6×10⁻⁸ (Δ) and 2.3×10⁻⁹ (γ; the γ
points themselves are good to about 10⁻⁷). The coefficients grow about ninefold per order with alternating signs: a
singularity near D ≈ −9 on the negative axis, which the rational form represents by the zero of its denominator at D =
−11.0. That singularity limits the series to $`D \gtrsim 10`$ and forbids cutting it short (five terms are off by 4×10⁻³
at D ≥ 20).

| k | $`c_k`$ (for Δ) | $`g_k`$ (for γ) |
|---|---|---|
| 0 | +1.6438931360 | +0.5 (fixed) |
| 1 | +1.0374964007 × 10¹ | −7.5854191958 × 10⁻¹ |
| 2 | −2.8355347834 × 10¹ | +8.2843462006 |
| 3 | +2.3900833130 × 10² | −1.2603029673 × 10² |
| 4 | −2.2446858552 × 10³ | +1.8446582182 × 10³ |
| 5 | +2.0424472078 × 10⁴ | −2.5165496284 × 10⁴ |
| 6 | −1.6230686158 × 10⁵ | +2.8434887038 × 10⁵ |
| 7 | +1.0037347608 × 10⁶ | −2.4157799331 × 10⁶ |
| 8 | −4.4127962953 × 10⁶ | +1.4042658549 × 10⁷ |
| 9 | +1.2046388629 × 10⁷ | −4.9128734266 × 10⁷ |
| 10 | −1.5266460427 × 10⁷ | +7.7494021741 × 10⁷ |

</details>
