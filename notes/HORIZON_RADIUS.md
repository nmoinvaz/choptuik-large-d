# The horizon series' radius at the chain's front (#693)

Does the radius of the horizon series reach the matching distance u₀ = 1 − x_last = 0.3 near D 253, where #603's chain
slows to steps of 0.02–0.03 and the horizon re-seed's last rung fails? The code is `hp/horizon_radius.py`, the derived
table `hp/data/horizon_radius.csv`, and the raw output folder `I693-mac/` of the raw-data repository (commit 2227dbf;
its `run_693.sh` holds every command). Everything here is numerical evidence.

## Answer

**The series of each state's own horizon data does not approach u₀.** It was read in octo-double at 33 states from
D 225.13 to 253.80. Its terms at u₀ fall to a minimum of 1.4×10⁻³⁶ at D 225 and 8.4×10⁻⁴¹ at D 253.8, and the decaying
part's radius rises from 0.71 to 0.77. No D is extrapolated for a crossing of 0.3, since there is no trend toward it.

**The re-seed fails because its quad-double polish starts from data rounded to double-double.** The predictions carry
a_h to about 2×10⁻³⁹ (relative; the median after steps of a < 0.6). The re-seed's double-double ladder hands the polish
data off by 1.2–1.4×10⁻³² instead. Near D 250 the series' response at u₀ to an error of that size turns nonlinear: for
δ = 10⁻³² the horizon residual is H = 1.0 δG at D 225, 1.4 δG at 249.0 and 1.6 δG at 253.8. Where the polish fails, its
first Newton step lands at |R| of 10¹⁴²–10²⁶⁰. Started from the prediction's own four parts instead, the polish
converges at the three failing steps tried (section 4).

13.6's pole law, which holds here where it can be read, says the same: data off by 10⁻³² put the pole at 0.280
(D 225) and 0.248 (D 253.8), inside u₀ at every D of the chain. Its δ_max(0.3) falls from 6.7×10⁻³⁵ to 3.8×10⁻³⁹. The
series truncated at ks = 130 hides that pole until about D 249.

So the wall near D 253 is not the series' radius on the solution. The re-seed's failures come from the hand-off between
its double-double ladder and its quad-double polish, inside the series formulation's basin for data of double-double
accuracy. The step's collapse is not all the re-seed's, though: #603 also refused steps whose re-seed succeeded
(D 253.07 at a = 0.568 and 0.284), the corrector's refusals of #634.

Two other things grow with D, and neither is the radius.

- **The arithmetic's floor at u₀.** In double- and quad-double the summed series misses the node by ε·G, the
  rounding times the mass mode's gain G = x_last^−(n−1). In double-double it is 6×10² at D 225 and 1.5×10⁷ at
  D 253.8, the size of the double-double rung's |R| in all 70 of #603's re-seeds, ok or failed. In quad-double it is
  4.5×10⁻³⁰ and 1.8×10⁻²⁵, the size of the polish's converged |R|. The polish accepts below 10⁻²⁴, which this floor
  reaches near D 258.6 (an extrapolation).
- **A transient past order Re s ≈ n/2.** Past k_g ≈ Re s (the exponent of the singular solutions, QUAD_DOUBLE 13.1) the
  low harmonics' terms grow, peak near order 190–211 and fall again. The terms past the chain's ks = 130, summed at u₀,
  are 6.7×10⁻²¹ at D 225 and 2.2×10⁻¹² at D 253.8 in the p rows. The chain's discrete problem leaves them out, and that
  omission grows 4.6-fold per unit of D since D 249. Its effect on Δ was not measured (recommended task).

## Method

The kernel's `Solver.series` is double-double only. The quad- and octo-double terms c_k u₀^k come from the horizon
block's residual on a two-node layout (the centre and the last node). Its rows are S_ks(u₀) − node, the series summed
to order ks minus the node, so S_k − S_(k−1) is one term per field and τ point in the residual's precision.

The estimators each read a window of 40 orders [K − 40, K]:
- E1, 13.6's ratio test, R = u₀ (t_(K−40) / t_K)^(1/40), with t_k the largest |c_k| u₀^k over the τ points;
- E2, the least-squares slope of ln t_k (the root test as a fit);
- E3, Domb–Sykes per τ point, read only where the ratios keep one sign, with its exponent.

Each is also applied per field (p, m, a, c) and per harmonic (the first four the field's parity allows). The spread over
E1–E3 is an estimator range, not an error bar.

## Checks and controls

| check | result | control |
|---|---|---|
| planted: poles at u = 0.25 and 0.4 (the issue's) | E1, E2, E3 read 0.2500 in the windows ending at 130 and 160 | `--mutate drop_term` (the 0.25 pole dropped) reads 0.4000 and fails, exit 1 |
| planted: a pole at 0.4 alone; a complex pair at 0.4; a pole at 0.15 under a decaying part | 0.4000; E2 0.3997–0.4002 with no Domb–Sykes lane read; 0.1500 | |
| planted: the mass mode (1 − u)^−(n−1), n − 1 = 250 | E3 0.998 with exponent 249; E1 and E2 read 0.306 | |
| C1: the two-node solver's horizon rows at kc = kt = 4 and at the state's kc, kt | bit for bit at orders 40 and 130 | |
| C2a, C2b: octo-double terms against double-double (orders 1–5) and quad-double (orders 1–20) | 2×10⁻¹⁶ (the f64 sum of the double-double parts), 0 | `--mutate shift` fails both, exit 1 |
| octo- against quad-double past the chain's orders, D 253.73 | 4×10⁻⁴ to 8×10⁻⁴ relative at orders 150–240 | |
| the re-seed reproduced on this Mac (`ladder`, six predictions) | the same outcome as #603's records at all six; D 249.01 converges to 8.69×10⁻²⁷ and D 249.39 fails at 1.61×10⁶, the records' values | |

C2 as first written compared the quad- and octo-double terms with the double-double series at orders 1–30, to 10⁻²⁴,
and it fired at D 226 and 253.7. The double-double series leaves the octo-double one from order 7 on and is wrong by
O(1) at order 16–17. Its two `horizon_alpha` modes differ by the same amounts, so it is the double-double rounding,
amplified by the mass mode. C2a and C2b replace it.

## 1. The series on the solution, in octo-double

R_dec is E2 of the a field over orders 20–60 (Domb–Sykes reads no lane there: the ratios change sign, a complex pair).
13.6's estimator is E1 / E2 of the a field over orders 90–130, the chain's last 40. k_g is the first order past the
minimum with a term 100 times larger.

| D | R_dec | 13.6's E1 / E2 | t_min (order) | k_g | Re s | transient peak (order) | terms past 130 at u₀, p |
|---|---|---|---|---|---|---|---|
| 225.13 | 0.710 | 0.274 / 0.296 | 1.4e-36 (104) | 118 | 114.2 | 3.0e-22 (189) | 6.7e-21 |
| 231.13 | 0.722 | 0.273 / 0.292 | 8.4e-38 (98) | 118 | 117.2 | 1.1e-21 (192) | 2.6e-20 |
| 237.13 | 0.734 | 0.272 / 0.288 | 1.2e-38 (100) | 120 | 120.2 | 2.4e-21 (196) | 5.6e-20 |
| 241.96 | 0.743 | 0.272 / 0.288 | 2.2e-39 (104) | 121 | 122.6 | 8.5e-20 (204) | 2.2e-18 |
| 244.64 | 0.748 | 0.276 / 0.292 | 8.9e-40 (104) | 122 | 123.9 | 7.0e-19 (203) | 1.9e-17 |
| 247.04 | 0.753 | 0.282 / 0.297 | 3.9e-40 (104) | 123 | 125.1 | 1.2e-17 (205) | 3.2e-16 |
| 249.01 | 0.756 | 0.288 / 0.301 | 2.2e-40 (106) | 123 | 126.1 | 4.6e-17 (210) | 1.2e-15 |
| 249.39 | 0.757 | 0.289 / 0.301 | 1.8e-40 (106) | 123 | 126.3 | 9.4e-17 (211) | 2.5e-15 |
| 250.84 | 0.760 | 0.293 / 0.303 | 1.2e-40 (104) | 124 | 127.0 | 1.0e-15 (211) | 2.8e-14 |
| 251.89 | 0.762 | 0.296 / 0.304 | 9.8e-41 (103) | 125 | 127.5 | 5.1e-15 (211) | 1.4e-13 |
| 252.61 | 0.763 | 0.298 / 0.304 | 9.3e-41 (111) | 125 | 127.9 | 1.4e-14 (211) | 3.9e-13 |
| 253.07 | 0.764 | 0.299 / 0.304 | 9.0e-41 (112) | 126 | 128.1 | 2.9e-14 (211) | 7.8e-13 |
| 253.64 | 0.766 | 0.300 / 0.304 | 8.7e-41 (100) | 126 | 128.4 | 6.4e-14 (211) | 1.7e-12 |
| 253.80 | 0.766 | 0.300 / 0.304 | 8.4e-41 (114) | 126 | 128.5 | 8.0e-14 (211) | 2.2e-12 |

The transient's columns come from quad-double runs to order 300. At D 253.73 the octo-double run to order 240 agrees
with them to 8×10⁻⁴.

- **13.6's estimator reads about u₀ at every state, and that is not a radius.** Its window holds the minimum and the
  transient's onset. It read 0.27–0.30 already at D 225, so it marks no wall at 253. In quad-double the same window
  holds the mass mode excited by the rounding: Domb–Sykes reads 0.990 with exponent 219 at D 225 (n − 1 = 222.1) and
  1.000 with exponent 251 at D 253.8 (n − 1 = 250.8). That is 13.6's Tangherlini term (1 − u)^−(n−1), of radius 1.
- **The transient is not a radius either.** Past its peak the local ratio t_K / t_(K−1) falls back to 0.60 (D 225)
  and 0.66 (D 253.73) by order 300, and E2 over orders 260–300 is 0.477 and 0.425, still rising with the order. It lives
  in the low harmonics (j ≤ 5 for p and m, j ≤ 4 for a and c), while the decaying part is carried by harmonics 13–17.
  Its onset tracks Re s = n / (2(1 + c₁)) − 1, the order where the p recurrence's κ_k changes sign.
- **The transient's tail is the chain's truncation error at the node.** The chain sums to ks = 130 and leaves out the
  terms past it: 2.2×10⁻¹² in the p rows at D 253.8, from 3.2×10⁻¹⁶ at D 247.

## 2. The arithmetic's floors at u₀, against #603's re-seed records

H is the largest |S_130(u₀) − node| over the m and a rows (the re-seed rung's rows) for each state's own data rounded
to the arithmetic. G = 0.7^−(n−1). The last column is #603's re-seed at the same D: the double-double u₀ rung's |R|,
then the quad-double polish.

| D | G | H, double-double | H, quad-double | H, octo-double | #603's re-seed at this D |
|---|---|---|---|---|---|
| 225.13 | 2.56e34 | 6.0e2 | 4.5e-30 | 3.2e-95 | (226.13: 8.7e2, converged 2.9e-30, reproduced here) |
| 237.13 | 1.85e36 | 3.1e4 | 3.2e-28 | 9.1e-78 | |
| 244.64 | 2.69e37 | 2.8e5 | 2.5e-27 | 4.9e-90 | 6.6e5, converged 2.6e-27 |
| 249.01 | 1.28e38 | 3.1e6 | 2.7e-26 | 9.8e-62 | 3.2e6, converged 8.7e-27 |
| 249.39 | 1.47e38 | 1.9e6 | 1.7e-26 | 2.2e-61 | 5.1e6, converged 1.7e-26 (a 0.763); 2.65e6, failed (a 0.381) |
| 251.89 | 3.57e38 | 5.5e6 | 6.4e-26 | 3.7e-79 | 6.7e6, converged 6.3e-26 |
| 253.64 | 6.68e38 | 1.8e7 | 9.1e-26 | 1.5e-61 | 2.2e7, converged 6.6e-26 |
| 253.80 | 7.08e38 | 1.5e7 | 1.8e-25 | 8.0e-62 | 2.05e7, converged 5.9e-26 |

- H / G is 0.7–3×10⁻³² in double-double and 0.7–4×10⁻⁶⁴ in quad-double at every D: the floors are the rounding times
  the gain, nothing more.
- The double-double u₀ rung was at its floor in all 70 re-seeds of #603's records, from D 226 on: one step, no trial
  accepted, not converged. So it marks nothing. Its |R| since D 249.3 is 2.3×10⁶–2.8×10⁷ in the 18 that succeeded and
  2.6×10⁶–3.0×10⁷ in the 14 that failed.
- The quad-double polish accepts below 10⁻²⁴ (`reseed.polish_q`). Its floor reaches that at G = 4×10³⁹, near D 258.6.
  This is an extrapolation with the measured 2.5×10⁻⁶⁴ and G's exact dependence on D.

## 3. 13.6's pole law at the front

Here a_h is scaled by 1 + δ, the series is summed in quad-double, and the response b(δ) − b(0) is read (`summary
--coef`). 13.6's line is R = (0.88 ln(1/δ) − 2.6) / (n − 1).

| D | H / (δG) at δ = 10⁻⁴⁰, 10⁻³⁶, 10⁻³², 10⁻²⁸, 10⁻²⁴, 10⁻²⁰ | E2 of a over orders 120–160, δ = 10⁻²⁸, 10⁻²⁴, 10⁻²⁰ | 13.6's line, same δ | δ_max(0.3), 13.6's line |
|---|---|---|---|---|
| 225.13 | 1.0, 1.0, 1.0, 1.9e2, 1.9e6, 4.3e11 | 0.240, 0.231, 0.178 | 0.244, 0.207, 0.171 | 6.7e-35 |
| 241.96 | 1.0, 1.0, 1.2, 1.7e3, 1.7e7, 4.1e12 | 0.227, 0.217, 0.167 | 0.227, 0.193, 0.159 | 2.2e-37 |
| 249.01 | 1.0, 1.0, 1.4, 3.8e3, 3.8e7, 9.1e12 | 0.222, 0.211, 0.163 | 0.220, 0.187, 0.154 | 2.0e-38 |
| 253.80 | 1.0, 1.0, 1.6, 6.2e3, 6.2e7, 1.5e13 | 0.219, 0.207, 0.160 | 0.216, 0.184, 0.151 | 3.8e-39 |

- Where the response is nonlinear (δ ≥ 10⁻²⁸), the measured radius agrees with 13.6's line within 2% at δ = 10⁻²⁸, 6%
  at 10⁻²⁰ and 13% at 10⁻²⁴. So the law holds at D 225–254.
- Where it is linear (δ ≤ 10⁻³²), the late orders hold the linear response (the mass mode and the transient), not the
  pole. The planned fit over all six δ gives C = 0.32–0.66, against 13.6's 0.88, because it mixes the two regimes; it is
  not used (discrepancy 2).
- By 13.6's line, the pole of data off by 10⁻³² sits at 0.280 at D 225 and 0.248 at D 253.8, inside u₀ at every state.
  The series truncated at 130 orders does not see it until its response turns nonlinear: 1.4 δG at D 249.

## 4. The re-seed's start

`pdelta` compares each stored prediction (`pred_pin3_D*.npz`, 54 with a converged state at the same D) with that state,
exactly.

- The prediction's a_h, all four parts, is off by 3.8×10⁻⁴⁰ to 5.6×10⁻³⁵ (median 1.7×10⁻³⁹) after the 34 steps with
  a < 0.6, and by up to 1.5×10⁻³³ after the 20 larger steps (the jet's truncation).
- Its two leading parts, where the double-double ladder starts, are off by 4.8–6.8×10⁻³³ at every D, whether the
  re-seed there succeeded or failed. The ladder's rungs then move them by about as much: the polish's start was off by
  1.35×10⁻³² (failed, D 253.79) and 1.19×10⁻³² (converged, D 253.80).

So the start's error does not grow with D; the response to it does (section 3). The failed polishes all diverge on their
first Newton step:

| D | outcome (`ladder`) | the polish's \|R\|: start, then each step |
|---|---|---|
| 226.13 | converged | 8.7e2, 3.3e1, 5.4e-2, 1.5e-7, 1.1e-18, 1.0e-27, ... 2.9e-30 |
| 249.01 | converged | 4.2e6, 8.7e5, 1.1e5, 3.2e3, 3.1, 2.7e-6, 2.2e-18, ... 8.7e-27 |
| 249.39 | failed | 1.6e6, the a_h and m_h steps 2.3e6, Newton 9.4e141: stopped |
| 253.76 | failed | 3.9e7, 9.1e6, 8.4e6, 2.1e6, Newton 1.8e223: stopped |
| 253.79 | failed | 2.4e7, 5.4e6, 2.0e6, Newton 7.0e260: stopped |
| 253.80 | converged | 3.1e7, 7.1e6, 1.3e6, 9.6e4, 7.4e2, 4.6e-2, 1.8e-10, 5.7e-19, ... 5.9e-26 |

`polish4` runs the same polish (`Reseeder.invert_q`) from the prediction's own four-part data instead:

| D | the start's a_h error | the polish's \|R\|: start, then the first steps | outcome |
|---|---|---|---|
| 249.39 | 2.1e-39 | 3.4e-4, 2.7e-14, 2.7e-30, then 1.5–2.8e-26 | converged (best 2.7e-30), 144 s |
| 253.76 | 2.3e-38 | 7.9e-26, 1.4e-25, 1.8e-29, then 0.7–1.4e-25 | converged (best 1.8e-29), 141 s |
| 253.79 | 6.8e-40 | 1.2e-25, 8.9e-26, 9.2e-30, then 0.9–2.6e-25 | converged (best 9.2e-30), 152 s |

All three steps whose re-seed failed converge from the prediction's data. At D 253.8 those data already match the node
at u₀ to the quad-double floor. This tests the re-seed only, not whether the corrector then accepts the step.

## Discrepancies

1. C2 as first written fired on real data (above). Its cause is known, the double-double series' own rounding past
   order 7, and C2a and C2b replace it.
2. 13.6's pole law, fitted over the six δ as planned, gives C = 0.32–0.66 and δ_max(0.3) of 10⁻⁵⁰ to 10⁻⁴², against
   13.6's C ≈ 0.88. The fit's points at δ ≤ 10⁻³² read the linear response, not the pole. On the three nonlinear points
   13.6's line holds within 2–13%.
3. The chain's horizon block omits the transient's terms past ks = 130: 2.2×10⁻¹² at the node in the p rows at D 253.8,
   growing 4.6-fold per unit of D since D 249 (2.2-fold over D 225–254). No error of #603's Δ accounts for it.
4. The issue takes the failing rung's |R| (2.7×10⁶ growing to 1.85×10⁷) as the failure's measure. That rung is the
   double-double u₀ rung, which converged in none of the 70 re-seeds. Its |R| is the floor ε·G, the same in ok and
   failed re-seeds, and grows with G alone. The step that fails is the quad-double polish after it.
