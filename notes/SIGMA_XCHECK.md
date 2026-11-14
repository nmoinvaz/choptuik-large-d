# Cross-check of #82's amplitude coefficients against K 33 runs (#395)

#82 computed σ₀…σ₂₂ (K 45, J 1500, 1280 bits) at ω/ω∞ = 1.09, 1.10, 1.11 on win2 while win2 had native faults. This
page compares σ₀…σ₁₆ with the K 33, J 1000, 960-bit files of other runs at the same nodes (T155-win2 at 1.09 and
1.11, T153-macm3 at 1.10), against a tolerance set from other ω before the comparison. Data only, no new runs.
Script `solver/inner_amp/sigma_xcheck.py`, raw `I395-mac/` of the raw-data repository.

## Result

**No difference beyond the J 1000 truncation of the comparison files.** Of the 51 entries (3 nodes × σ₀…σ₁₆), 46 are
OK, 4 WATCH and 1 BAD under the rule fixed beforehand. The BAD entry, σ₉ at 1.09, lies in the K 33 J 1000 file: the
K 45 J 1500 value agrees with the independent K 45 J 1250 run of #213 to 2.8×10⁻²⁰, while the J 1000 value is
reproduced to all 20 digits by #256's K 39 J 1000 run. The four WATCH entries (σ₇ and σ₉ at 1.10 and 1.11) have the
same structure. The three nodes lie on the K 33 ladder's interpolant (ω/ω∞ = 1.06 … 1.14) exactly as well as the K 33
values do.

## Tolerance

r_n = |σ_n(A) − σ_n(B)| / |σ_n(A)|, from the 20-digit strings, A the K 45 file. R_n is the largest r_n over ω/ω∞ =
0.99, 1.00, 1.01 between T99-win2 (K 45, J 1500, 1280 bits) and T83-mac (K 33, J 1000, 960 bits), the only ω where
both exist. floor_n is one unit in the 20th digit of each component of both strings, over |σ_n|. A verdict is OK when
r_n ≤ max(R_n, floor_n), WATCH when r_n ≤ 10 max(R_n, floor_n), BAD otherwise. The rule and R_n were committed
(raw 15706a6c, script 705f07e1) before the target comparison ran. It was not blind: #82, #202 and #256 had already
posted the target differences at n = 6–10 and 14–16. R_n and r_n are differences between two truncations, not error
bounds.

| n | R_n | 1.09 | ratio | 1.10 | ratio | 1.11 | ratio |
|---|---|---|---|---|---|---|---|
| 1–5 | 0 | 0 | | 0 | | 0 | |
| 6 | 0 | 0 | | 9.7×10⁻²¹ | 0.25 | 0 | |
| 7 | 1.7×10⁻²⁰ | 6.7×10⁻²⁰ | 0.49 | 1.2×10⁻¹⁸ | 8.7 WATCH | 3.6×10⁻¹⁹ | 2.6 WATCH |
| 8 | 3.5×10⁻¹⁷ | 8.5×10⁻²⁰ | 0.002 | 6.1×10⁻²⁰ | 0.002 | 9.9×10⁻²⁰ | 0.003 |
| 9 | 2.4×10⁻¹⁸ | 3.2×10⁻¹⁷ | **13 BAD** | 9.4×10⁻¹⁸ | 3.9 WATCH | 3.3×10⁻¹⁸ | 1.4 WATCH |
| 10 | 2.6×10⁻¹⁷ | 7.8×10⁻¹⁸ | 0.30 | 8.9×10⁻¹⁸ | 0.34 | 1.0×10⁻¹⁷ | 0.39 |
| 11 | 2.2×10⁻¹⁴ | 4.6×10⁻¹⁸ | 2×10⁻⁴ | 5.2×10⁻¹⁸ | 2×10⁻⁴ | 5.7×10⁻¹⁸ | 3×10⁻⁴ |
| 12 | 6.4×10⁻¹⁶ | 7.5×10⁻¹⁸ | 0.012 | 9.0×10⁻¹⁸ | 0.014 | 1.1×10⁻¹⁷ | 0.017 |
| 13 | 1.6×10⁻¹⁵ | 3.8×10⁻¹⁷ | 0.023 | 4.4×10⁻¹⁷ | 0.027 | 4.8×10⁻¹⁷ | 0.030 |
| 14 | 3.4×10⁻¹⁵ | 9.3×10⁻¹⁷ | 0.027 | 1.3×10⁻¹⁶ | 0.037 | 1.3×10⁻¹⁶ | 0.037 |
| 15 | 6.6×10⁻¹⁵ | 2.5×10⁻¹⁶ | 0.037 | 4.0×10⁻¹⁶ | 0.061 | 2.8×10⁻¹⁶ | 0.042 |
| 16 | 1.2×10⁻¹⁴ | 3.5×10⁻¹⁶ | 0.029 | 1.2×10⁻¹⁵ | 0.10 | 4.4×10⁻¹⁶ | 0.037 |

σ₀ = 1 by normalization, and its imaginary part is noise of 10⁻²⁹⁰. The ratio is r_n / max(R_n, floor_n).

**Where the excess sits** (`sigma_xcheck.py jseq`, the K 45 J 1250 files of #213). At every WATCH and BAD entry the
J 1250 and J 1500 values agree to 6.4×10⁻²¹ or better, so the whole difference is in the J 1000 value.

| ω/ω∞, n | J 1000 vs J 1500 | J 1000 vs J 1250 | J 1250 vs J 1500 |
|---|---|---|---|
| 1.09, 9 | 3.2×10⁻¹⁷ | 3.2×10⁻¹⁷ | 2.8×10⁻²⁰ |
| 1.10, 7 | 1.2×10⁻¹⁸ | 1.2×10⁻¹⁸ | 0 |
| 1.10, 9 | 9.4×10⁻¹⁸ | 9.4×10⁻¹⁸ | 6.4×10⁻²¹ |
| 1.11, 7 | 3.6×10⁻¹⁹ | 3.6×10⁻¹⁹ | 0 |
| 1.11, 9 | 3.3×10⁻¹⁸ | 3.3×10⁻¹⁸ | 0 |

The rule's factor 10 is smaller than the variation of the J 1000 truncation that the reference itself shows. Between
ω/ω∞ = 0.99 and 1.01, R_n's three values differ by 27× at n = 8 and 32× at n = 11. The BAD verdict stands as the
rule's output; its cause is the comparison file, not #82's node.

## Smoothness

At each node t the K 45 value is compared with the degree-5 interpolant through the six other nodes of the ladder
1.06, 1.08, 1.09, 1.10, 1.11, 1.12, 1.14 (K 45 at the other two targets, K 33 J 1000 elsewhere). The reference is the
same interpolant through the pure K 33 ladder. The two residuals agree to the six digits stored, from 6×10⁻¹² (σ₁) to
1.1×10⁻⁸ (σ₁₆) relative, and no entry is BAD. That residual is the check's resolution. The OK/WATCH split (36/15) is
decided far below it and carries no information. This check is cross-machine for 1.09 and 1.11 through macm3's 1.08
and 1.12, at that resolution only.

## Controls

| control | result |
|---|---|
| replica: T103-mac against T99-win2, ω/ω∞ 1.00, K 45 J 1500, through the same check with tolerance 0 | 23 of 23 equal |
| the same with one digit of T103's σ_k changed, k = 3, 16, 22 | found at k only, each time |
| #82's posted K 45 − K 33 table (22 values, two digits) | recomputed, 22 of 22 |
| change of 30 max(R_n, floor_n) planted in 1.10's σ_n, n = 2, 9, 16 | BAD at n (plus the unplanted 1.09, σ₉) |
| change of 0.3 max(R_n, floor_n) at the same n | silent |
| smoothness: 100 times the residual planted in 1.10's σ_n, n = 2, 9, 16 | BAD (at all three nodes, since 1.10 enters the other two interpolants) |

## Independence

The comparison reads only `sigma_str` and shares no code with `amp_flint_par.py` or the comparison scripts of #202,
#256 and #295. The files share `amp_flint_par.py`. For 1.09 and 1.11 every file at the 10⁻¹⁵ level is from win2:
T155-win2 (2026-11-07, its run overlapping the first fault at 18:37 UTC), I213-win2 and I256-win2 (new firmware). Only
1.10 is compared with another machine at that level (T153-macm3). The K 33 values at 1.09 and 1.11 are reproduced to
20 digits by #256's K 39 J 1000 runs for σ₁…σ₁₅.

## An observation

The J truncation is not monotone in J. At 1.10, σ₁₃ of J 1250 differs from J 1500 by 3.4×10⁻¹⁶ while J 1000 differs
by 4.4×10⁻¹⁷. At 1.00, σ₁₆ of J 1250 differs from J 1500 by 6.1×10⁻¹⁴ while J 1000 differs by 1.1×10⁻¹⁴. So a
difference between two J is a sample of an oscillating truncation, not a bound on either value.

## Reproduce

With RAW a clone of choptuik-large-d-raw, from `solver/inner_amp/`:

```
python3 sigma_xcheck.py tol RAW --out tol.json             # R_n, the table's second column
python3 sigma_xcheck.py selftest RAW --tol RAW/I395-mac/tol_ref.json   # "selftest ok"
python3 sigma_xcheck.py compare RAW --tol RAW/I395-mac/tol_ref.json    # 46 OK, 4 WATCH, 1 BAD; exit 1
python3 sigma_xcheck.py jseq RAW
python3 sigma_xcheck.py smooth RAW                          # no BAD; exit 0
```

## Proposed text for notes/HARMONY.md

After "… leave σ₀–σ₁₅ unchanged (#229, #236, #247).": "A cross-check against a tolerance fixed from ω/ω∞ = 0.99–1.01
(#395, notes/SIGMA_XCHECK.md) finds no difference of σ₀…σ₁₆ at the three nodes beyond the J 1000 truncation of the
K 33 files. Its one entry outside ten times the reference, σ₉ at 1.09, is in the J 1000 file, since #213's J 1250 run
agrees with the J 1500 value to 3×10⁻²⁰."
