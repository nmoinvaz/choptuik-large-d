# LARGE_D_INNER: a horizon variable whose rounding does not grow with D (#1090)

Issue #1090 (quest P1 of #1088). Written by a subagent of the coordinator on the Mac, 2026-11-25. Code in
[`solver/inner_large_d/`](../solver/inner_large_d/); raw output in the folder `I1090-mac/` of the raw-data repository.
Status words as in AGENTS.md. Every number below comes from a log in `I1090-mac/` named beside it.

## 0. Verdicts

| question | answer | status |
|---|---|---|
| Does form S need about 0.06n more digits at fixed precision, from the scalar mode's shadow? (#1090's premise) | **No.** The loss is a forward-recurrence instability of Track B's horizon variable, affine in z; a Moebius variable removes it. At the float64 solutions, the float64 error of the horizon sum at x* is 1e-14 to 2.5e-14 from D 100 to 600 with a = 8, against 5e-6 at D 200 and 1e24 at D 400 in the affine variable (§4) | numerical evidence |
| Where does the affine variable lose its digits? | from the low orders on: the critical solution's horizon series is the minimal solution of its recurrence, and the resonance at σ_R injects the rounding into the shadow. At D 200 the float64 relative error passes 1 at order 35 of σ_R ≈ 102 (§2) | numerical evidence |
| Which Moebius variable? | t = (1 − z)/(1 + a z) with a = max(8, n/400), and products with z by their exact two-term recurrence. a = 1 leaves a growth of about 10^(0.011n) in the resonance's band and underflows at large D; a fixed a = 8 runs out of exponent range near D 10⁴. With the rule the rounding amplification stays at 3.5–120 from D 1000 to 10⁴ (§3) | numerical evidence |
| The inner problem at x ~ √n with the Stokes constant as an unknown | not the place for the selection. At finite n it lies beyond the SSH: through x = 1 it costs 2^(n/2), around it e^(πkω) per harmonic (BEYOND_LIGHT_CONE §3, §7). The selection stays at x*, where the shadow comes out of the stable series with no extra unknown (§6) | argued from measured weights |
| Cost against D | linear: ks ≈ n + 14√n + 80 orders, banded; one float64 residual 0.07 s at D 100, 3.6 s at D 10⁴ (§5) | measured |
| Conditioning against D | the harmonic ladder of the selection, which no layout at x* changes: σ_min 1e-13 to 1e-15 at M 6 from D 100 to 600 (§4) | numerical evidence |
| Δ | float64 M 6 to D 600+: Δ − Track C = +3.7e-3 to +1.8e-3, M 6's truncation. Double-double M 10 at D 100: +1.76e-6. M ≥ 8 off D 100 is blocked by the weak directions' basin, not by precision (§4) | numerical evidence |

## 1. The problem and the layout

Form S of notes/LARGE_D_FACTORED.md (Track B, `wip/large-d-factored`): parity variables U = p + m, V = (p − m)/x, a
slaved by the momentum constraint (gr-qc/9604019 Eq. 19), c, all series in z = x². With n = D − 2 and θ = ωτ,

```
A = 2z U_z − 2cz V_z + ω U_θ − G V = 0
B = 2z² V_z − 2cz U_z + ω z V_θ + z V − (G + (n−1) c) U = 0
G = 2z c_z,   (ω∂_θ − (n−1) − q) X = q,   q = (U² + zV²)/2 − cUV,   X = 1/a² − 1,   G = −(n−1) c X/(1 + X)
```

(checked symbolically by Track B's `derive_factored.py`). The layout is Track B's: the centre series in z with data p₁
and c₀, summed at x* = 1/√2; the horizon series at the SSH with c = 1 and Q₀ = U − V = 2m_h; the rows U, V, c matched
at x* and the phase row. Unknowns p₁, c₀, Q₀ on M collocation points each, and Δ. The only change here is the variable
of the horizon series.

## 2. Why the affine horizon series loses digits

Track B's horizon series runs in s = (1 − z)/(1 − z*), affine in z. On the z-plane the SSH is z = 1, x* is z = ½ and
the centre is z = 0, where every SSH-analytic solution except the critical one is singular (the fast scalar mode x⁻ⁿ
and its relatives). In any variable affine in z, x* lies halfway between the SSH and the centre. The critical solution
is regular at the centre, so its horizon series has a radius larger than the centre's distance: it is the *minimal*
solution of the horizon recurrence, and forward recurrence of a minimal solution is unstable (W. Gautschi, SIAM Rev. 9
(1967) 24–82, doi:10.1137/1009002; F. W. J. Olver, J. Res. NBS 71B (1967) 111–129, doi:10.6028/jres.071B.018). Rounding made at order k excites solutions whose coefficients decay
like 2⁻ᵏ in units where x* is at 1, while the critical solution's decay much faster at low orders (the slaving of p by
the (n/2)c(p + m) term). At the order σ_R ≈ n/2 + 2.7 the near-resonance of the SSH exponent injects whatever is there
into the shadow, whose sum at x* is O(1).

Per order, float64 against double-double, at Track B's D-200 seed (`diag/affine_orders.log`; σ_R ≈ 101.7; |U_k| in
units where x* is at 1, so the term at x* is the coefficient):

| k | 7 | 21 | 35 | 49 | 77 | 98 | 118 | 166 | 214 |
|---|---|---|---|---|---|---|---|---|---|
| affine: \|U_k\| | 3.8e-10 | 1.4e-21 | 3.7e-31 | 7.0e-38 | 1.4e-42 | 1.1e-38 | 3.6e-24 | 2.2e-12 | 4.6e-11 |
| affine: float64 error, relative | 6.0e-11 | 3.1e-5 | 2.4 | 3.8e2 | 2.8e-2 | 0.21 | 16 | 29 | 17 |
| Moebius: \|U_k\| | 3.7e-6 | 3.9e-13 | 8.7e-20 | 7.7e-26 | 4.4e-30 | 1.8e-26 | 1.4e-14 | 2.6e-11 | 1.0e-18 |
| Moebius: float64 error, relative | 1.4e-14 | 4.4e-14 | 5.8e-13 | 2.6e-11 | 3.6e-11 | 2.5e-12 | 1.1e-11 | 1.3e-11 | 9.3e-12 |

- In the affine variable the relative error passes 1 at order 35, a third of the way to σ_R.
- The shadow (orders 118 to 214, where the coefficients grow again) then comes out 16 to 29 times its true size, and
  the sum at x* is off by 1.2e-5 relative at D 200 and by a factor 30 at D 245.
- In the Moebius variable no coefficient is worse than 6e-11 relative, and the sum at x* is right to 2.0e-14 at D 200
  and 1.2e-14 at D 245.
- At D 100 (the solution itself) both sums are right to 1.4e-14 and 1.2e-14.

Relative noise η put on any stored array of the recurrence (U, V, X, G, c, q) after every order is amplified in the sum
at x* by the same factor, about 10^(0.05–0.13 n) between D 100 and 245 (scratch diagnostic on Track B's code, logged
in `diag/noise_variable.log` for U, V and c):

| D | 100 | 150 | 200 | 245 | 300 | 400 |
|---|---|---|---|---|---|---|
| affine, amplification of noise on U, V | 7.2e1 | 2.5e4 | 7.2e9 | 8.4e14 | 1.2e15 | 4.3e14 |
| affine, float64 max\|U(x*)\| | 1.2245e-2 | 6.5295e-3 | 4.1931e-3 | 9.2888e-2 | 6.6564e6 | 2.1156e22 |
| Moebius (a = 1), amplification | 7.8 | 4.0 | 11 | 3.1 | 5.1 | 83 |
| Moebius, float64 max\|U(x*)\| | 1.2245e-2 | 6.5295e-3 | 4.1930e-3 | 3.0731e-3 | 2.2563e-3 | 1.4571e-3 |

The affine amplification saturates near 1e15 at D ≥ 245 only because the float64 sum itself is then garbage (row 2).
The seeds are Track B's D-100 solution rescaled (`seed_precision.py`'s), so they are not solutions; the noise test
measures the recurrence's linear response there. §4 repeats it at solutions.

## 3. The Moebius variable

Put the centre far from x* in the expansion variable while keeping the SSH at 0:

```
t = (1 − z)/(1 + a z),    τ = t/t*,    t* = 1/(2 + a)
```

so x* is at τ = 1, the centre at τ = 2 + a and z = ∞ at τ = −(2 + a)/a. With z d/dz = −h d/dt,
h = (1 − t)(1 + at)/(1 + a), and z² d/dz = −e d/dt, e = (1 − t)²/(1 + a):

```
A + B = 2h(c − 1) U_t + 2(hc − e) V_t + ω (U + zV)_θ − G (U + V) + zV − (n−1) c U
A     = −2h (U_t − c V_t) + ω U_θ − G V,            c_t = −G/(2h)
```

At the SSH (t = 0, c = 1, z = 1) both derivative terms of A + B carry a factor t, so order k of A + B fixes
P_k = U_k + V_k through

```
L_k = ω D + k (1 + a + 2c₁)/(1 + a) − G₀ − (n/2 − 1),     c₁ = −G₀ (1 + a)/2,
```

which is resonant at k = (n/2 − 1 + G₀)/(1 − G₀) = σ_R, the SSH exponent of LARGE_D_BASICS §2, for every a. Order k
of A fixes Q_{k+1} = U_{k+1} − V_{k+1}, the c equation fixes c_{k+1}, and q_k needs only Q_k because z₀ = c₀ = 1.
a = 0 is Track B's recurrence in w = 1 − z. The recurrence is `mobius_s.py` (float64 and complex128),
`mobius_sdd.py` (double-double with exact forward-mode Jacobians) and `mobius_mp.py` (element by element, any
precision; written first, kept as the independent check).

**Checks** (each able to fail):

| check | result | threshold | control |
|---|---|---|---|
| float64, a = 0 and a = 1, against Track B's sums at x* at its D-100 solution (`check_mobius.log`) | 2.1e-14, 2.2e-14 relative | 1e-13 | a wrong h₁ (`--mutate=Lk`): 5.6e-3, FAIL (`check_mobius_mutate_Lk.log`) |
| double-double, a = 1, against the mpmath reference at 40 digits (`check_mobius_dd.log`; after the exact z products `check_mobius_dd_v2.log`) | 9.4e-31, 9.6e-31 relative | 1e-29 | t* moved by 1e-20 relative (`--mutate=ts`): 5.3e-21, FAIL (`check_mobius_dd_mutate_ts.log`) |
| forward-mode dd Jacobian against the float64 complex-step one | 1.0e-15 relative | 1e-12 | |
| noise amplification below 1e3 at every D, Moebius (`noise_variable.log`) | 83 at most | 1e3 | the affine variable (`--mutate=affine`): 1.2e15, FAIL |

a = 0 and a = 1 are different recurrences for the same function, so their agreement checks the derivation, not only
the code.

**The choice of a** (`diag/noise_at_solution_a.log`, `diag/noise_at_solution_bigA.log`, at the float64 M-6
solutions of §4). Relative noise put only on the orders σ_R/2 ≤ k < 1.5σ_R, where the resonance injects into the
shadow, shows that a = 1 still leaves a slow growth there; a larger a lowers it, until z = ∞ (at τ = −(2 + a)/a) comes
near x* and the series' tail grows:

| a | D 200 | 263.6 | 300 | 350 | 400 | tail at ks, D 200 / 400 |
|---|---|---|---|---|---|---|
| 1 | 0.46 | 2.7 | 15 | 32 | 55 | |
| 2 | 9.1e-3 | 3.4e-2 | 0.22 | 0.77 | 1.9 | |
| 3 | 5.0e-4 | 1.8e-3 | 5.3e-3 | 3.4e-2 | 0.13 | |
| 4 | 5.0e-4 | 7.4e-4 | 1.8e-3 | 2.2e-3 | 9.2e-3 | |
| 6 | 1.0e-3 | | | | 2.0e-3 | |
| 8 | 2.0e-3 | | | | 6.5e-4 | 7.6e-50 / 5.4e-77 |
| 16 | 7.3e-3 | | | | 1.3e-3 | 4.3e-28 / 4.1e-42 |
| 32 | 9.9e-2 | | | | 1.3e-2 | 2.7e-16 / 3.0e-23 |

- For fixed a ≤ 3 the injection channel grows about 10^(0.011n) from D 200 to 400. That is far below the affine
  variable's growth, but it is not flat.
- At a = 6–8 it is about 1e-3 at both D, and the total amplification, set by the low orders, is 3.3–3.6.
- The residual of the same solution is the same for every a (5.91e-12 at D 200, 7.04e-14 at D 400, a = 1 and 8),
  as it must be.
- The float64 continuation runs at a = 8 from D 400 on.

**Two constraints at larger D** (`diag/seed_test*.log`, seeds rescaled from the D-624 solution, Δ from Track C):

1. **Products with z must not be banded.** z = (1 − t)/(1 + at) has coefficients that fall only like (a/(2 + a))ʲ,
   0.8ʲ at a = 8. So the banded convolution (kb = 150) drops terms of about 1e-15 at every order, and that error grew
   to garbage at D ≥ 5000. The series now forms zW from (1 + a t*τ)(zW) = (1 − t*τ)W, that is
   zW_k = W_k − t* W_{k−1} − a t* zW_{k−1}, exact and stable for a t* < 1. The checks of §3 pass unchanged
   (`check_mobius_v2.log`: 2.3e-14; `check_mobius_dd_v2.log`: 9.6e-31).
2. **The coefficients must stay inside the floating exponent range.** The critical solution's coefficients fall like
   \|τ_∞\|⁻ᵏ = ((2 + a)/a)⁻ᵏ, where τ_∞ is the image of z = ∞, until the shadow rises from σ_R ≈ n/2.
   - At a = 8 and D 10⁴ they reach 6e-296 near order 3000. The shadow then grows from underflow-damaged values,
     and max\|U(x*)\| comes out as 3.4e109.
   - At a = 1 they underflow to zero before σ_R. The shadow is then computed as exactly zero, so the selection
     signal is lost, even though the sum looks right.
   - So a must grow with n. With a = max(8, n/400), (1 + 2/a)^(n/2) ≈ e^(n/a) stays below e^400 ≈ 1e174 (1e155 at
     n = 3200 with a = 8), inside float64's 1e±308.

With that rule:

| D | 1000 | 2000 | 3200 | 5000 | 6500 | 8000 | 10000 |
|---|---|---|---|---|---|---|---|
| a | 8 | 8 | 8 | 12.5 | 16.2 | 20.0 | 25.0 |
| ks | 1520 | 2703 | 4069 | 6067 | 7706 | 9330 | 11477 |
| noise amplification | 3.5 | 16 | 120 | 14 | 7.4 | 13 | 22 |
| float64 error against dd | 5.8e-14 | 5.1e-14 | | 2.7e-13 | | | |

(`diag/seed_test_rule.log`.) The amplification stays within two decades from D 1000 to 10⁴; D 3200, at the rule's
lower edge a = 8, is the largest, so n/300 may be the better constant. In the affine variable the same quantity is
1e15 already at D 245.

## 4. Solutions against D

**Float64, M 6** (harmonics 1, 3, 5), continuation in D (`solve_cont.py`, Newton with the complex-step Jacobian and
the float64 split Newton of `lsnewton64.py`), a = 1 to D 400 and a = 8 from D 400 on (`f64_M6_cont/`). The table is
`tabulate.py`'s, the noise columns the amplification of relative noise on U, V at the solution (§2's measure), Δ
against Track C at the same D (`diag/trackc_at.json`):

<!-- f64 table: solver/inner_large_d/tabulate.py, regenerated -->
**`f64_M6_cont/` (20 records)**

| D | M | a | ks | max\|F\| | Δ | Δ − Track C | Track C's E | σ_min (scaled) | grip | noise, Moebius | noise, affine | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 6 | 1 | 316 | 5.2e-14 | 1.74872893839 | +3.70e-03 | 8.4e-13 | 1.6e-13 | 2.4e+11 | 8.2e+00 | 2.3e+01 | 108 |
| 110 | 6 | 1 | 333 | 5.0e-12 | 1.73994605761 | +3.91e-03 | 2.0e-12 | 1.1e-13 | 2.5e+11 | 3.8e+00 |  | 143 |
| 125 | 6 | 1 | 358 | 5.2e-12 | 1.72926763689 | +4.07e-03 | 4.8e-12 | 6.4e-14 | 6.4e+11 | 3.6e+00 |  | 202 |
| 150 | 6 | 1 | 398 | 3.9e-12 | 1.71607436594 | +4.21e-03 | 1.3e-11 | 1.0e-14 | 4.8e+12 | 6.3e+00 | 2.8e+03 | 140 |
| 175 | 6 | 1 | 437 | 3.5e-12 | 1.70661436727 | +4.32e-03 | 2.6e-11 | 3.0e-14 | 1.7e+12 | 4.6e+00 |  | 458 |
| 186.66 | 6 | 1 | 454 | 2.9e-12 | 1.70296278021 | +4.27e-03 | 3.3e-11 | 3.1e-14 | 1.6e+12 | 6.4e+00 |  | 94 |
| 200 | 6 | 1 | 474 | 5.9e-12 | 1.69925028995 | +4.16e-03 | 4.2e-11 | 2.9e-13 | 1.9e+11 | 2.7e+00 | 4.5e+09 | 98 |
| 225 | 6 | 1 | 512 | 7.0e-11 | 1.69317125687 | +3.71e-03 | 6.1e-11 | 2.5e-12 | 9.8e+09 | 7.2e+00 |  | 122 |
| 245.64 | 6 | 1 | 542 | 7.1e-13 | 1.68964912352 | +3.97e-03 | 7.8e-11 | 2.2e-14 | 2.8e+12 | 5.2e+00 | 2.9e+15 | 146 |
| 263.6 | 6 | 1 | 568 | 4.9e-13 | 1.6867653049 | +3.91e-03 | 9.3e-11 | 8.9e-15 | 7.7e+12 | 3.6e+00 |  | 136 |
| 300 | 6 | 1 | 619 | 2.6e-13 | 1.68197169023 | +3.80e-03 | 1.2e-10 | 1.3e-14 | 6.1e+12 | 1.7e+01 | 2.6e+16 | 219 |
| 350 | 6 | 1 | 689 | 6.0e-14 | 1.67697951943 | +3.67e-03 | 1.8e-10 | 5.5e-16 | 1.9e+14 | 3.5e+01 |  | 173 |
| 400 | 6 | 8 | 757 | 6.2e-14 | 1.67323717561 | +3.58e-03 | 2.3e-10 | 6.2e-15 | 2.1e+13 | 3.4e+00 |  | 184 |
| 400 | 6 | 1 | 757 | 7.0e-14 | 1.67323887255 | +3.58e-03 | 2.3e-10 | 8.4e-15 | 8.8e+12 | 5.6e+01 | 2.7e+16 | 211 |
| 450 | 6 | 8 | 824 | 4.4e-11 | 1.6714732384 | +4.66e-03 | 2.9e-10 | 1.0e-11 | 1.7e+10 | 3.5e+00 |  | 121 |
| 500 | 6 | 8 | 890 | 9.9e-14 | 1.66700386426 | +2.47e-03 | 3.4e-10 | 1.4e-14 | 1.5e+13 | 6.1e+00 |  | 191 |
| 600 | 6 | 8 | 1020 | 8.1e-14 | 1.66287327803 | +1.77e-03 | 4.5e-10 | 1.5e-16 | 1.6e+15 | 3.5e+00 |  | 221 |
| 624 | 6 | 8 | 1051 | 6.6e-14 | 1.66284118249 | +2.39e-03 | 4.7e-10 | 6.7e-15 | 3.1e+13 | 6.6e+00 |  | 354 |
| 636.73 | 6 | 8 | 1067 | 3.9e-09 | 1.65258986038 | -7.53e-03 | 4.9e-10 | 1.3e-10 | 7.2e+08 | 4.5e+00 |  | 169 |
| 650 | 6 | 8 | 1084 | 1.1e-07 | 1.66558191808 | +5.79e-03 | 5.0e-10 | 2.9e-09 | 3.3e+07 | 6.3e+00 |  | 153 |

**`f64_M6_cont2/` (4 records)**

| D | M | a | ks | max\|F\| | Δ | Δ − Track C | Track C's E | σ_min (scaled) | grip | noise, Moebius | noise, affine | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 624 | 6 | 8 | 1051 | 8.2e-14 | 1.66284118249 | +2.39e-03 | 4.7e-10 | 3.1e-13 | 5.1e+11 | 6.6e+00 |  | 308 |
| 632 | 6 | 8 | 1061 | 7.3e-14 | 1.66252139716 | +2.28e-03 | 4.8e-10 | 2.0e-14 | 1.6e+13 | 3.5e+00 |  | 328 |
| 640 | 6 | 8 | 1071 | 7.5e-14 | 1.66202416442 | +1.99e-03 | 4.9e-10 | 2.4e-14 | 1.4e+13 | 3.4e+00 |  | 298 |
| 650 | 6 | 8 | 1084 | 4.8e-10 | 1.66491572558 | +5.13e-03 | 5.0e-10 | 3.0e-13 | 4.2e+10 | 6.3e+00 |  | 343 |

**`f64_M6_cont3/` (3 records)**

| D | M | a | ks | max\|F\| | Δ | Δ − Track C | Track C's E | σ_min (scaled) | grip | noise, Moebius | noise, affine | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 640 | 6 | 8 | 1071 | 6.7e-14 | 1.66201800008 | +1.98e-03 | 4.9e-10 | 5.0e-15 | 5.6e+13 | 3.5e+00 |  | 348 |
| 646 | 6 | 8 | 1079 | 7.1e-14 | 1.66201367551 | +2.13e-03 | 5.0e-10 | 3.5e-14 | 7.7e+12 | 3.5e+00 |  | 361 |
| 652 | 6 | 8 | 1086 | 1.7e-10 | 1.6609189278 | +1.18e-03 | 5.0e-10 | 5.6e-12 | 3.2e+10 | 3.5e+00 |  | 346 |

**`f64_M6_cont4/` (3 records)**

| D | M | a | ks | max\|F\| | Δ | Δ − Track C | Track C's E | σ_min (scaled) | grip | noise, Moebius | noise, affine | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 646 | 6 | 8 | 1079 | 6.6e-14 | 1.66199937859 | +2.11e-03 | 5.0e-10 | 7.5e-15 | 2.8e+13 | 3.5e+00 |  | 206 |
| 660 | 6 | 8 | 1097 | 1.5e-11 | 1.67652832979 | +1.70e-02 | 5.1e-10 | 2.7e-12 | 3.4e+10 | 5.5e+00 |  | 255 |
| 680 | 6 | 8 | 1122 | 9.8e-07 | 1.69619961867 | +nan | nan | 8.5e-09 | 7.6e+07 | 3.7e+00 |  | 48 |
<!-- end f64 table -->

Every attempt is a row. A row whose max\|F\| is above its run's acceptance is a failed attempt after which the driver
inserted a midpoint in 1/n. The acceptances were 1e-11 to D 186.66, then 1e-9 in `f64_M6_cont/` and `f64_M6_cont2/`, and
1e-10 in `f64_M6_cont3/`.

D 650 in `f64_M6_cont2/` passed 1e-9 but is off the branch: Δ − Track C is +5.1e-3 there, against +2.0e-3 at D 640. It
is a point of small residual in a weak valley, so `f64_M6_cont3/` restarts from D 640, refusing any solve whose Δ
moves more than 1e-4 from its seed's extrapolated Δ.
- That refused D 652, whose Δ moved 1.1e-3.
- `f64_M6_cont4/` from D 646 allowed 5e-3. It took D 660 with Δ = 1.6765, Track C + 1.7e-2, because the check needs two
  points, and its next seed (D 680) failed.

So float64 at M 6 has no well-defined branch past D ≈ 650: there the residual floor (1e-11 to 1e-13) is reached in a
valley along which Δ moves by 1e-2. The float64 M-6 solutions on the branch end at D 646.

**The float64 error at the solutions themselves** (`diag/f64_vs_dd.log`; the horizon sums at x* in float64 against
double-double, the larger of U, V and c):

| D | 100 | 200 | 263.6 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|---|
| affine (a = 0) | 1.1e-14 | 5.0e-6 | 4.6e4 | 2.7e9 | 1.4e24 | 5.7e31 | 5.2e32 |
| a = 1 | 4.7e-15 | 8.0e-15 | 1.7e-14 | 8.6e-15 | 2.0e-14 | 4.1e-14 | 1.1e-13 |
| a = 8 | 6.9e-15 | 7.4e-15 | 1.6e-14 | 7.4e-15 | 9.4e-15 | 2.0e-14 | 2.5e-14 |

From D 500 on even the affine double-double sum is garbage: its max\|U(x*)\| is 5.3e4 and 3.2e19 there, against
1.17e-3 and 8.9e-4.

**What the table shows.**
- The residual reaches the float64 floor of this solver, 1e-13 to 1e-11, at every D. The noise amplification at a = 8
  stays at 3.4–6.1. These are the layout's numerics, and they do not grow with D.
- σ_min and the grip are the harmonic ladder of the M-6 truncation: 1e-13 to 1e-16 and 1e11 to 1e15, from D 100 to
  640. These are the problem's conditioning in its harmonics, the same in any layout at x* (§6). At M 6 in float64
  they leave Δ pinned to about 1e-6 at D 400 (the a = 1 and a = 8 solves differ by 1.7e-6), and only to about 3e-4 at
  D 600–640, where Δ − Track C scatters as +1.77e-3, +2.39e-3, +2.28e-3, +1.99e-3. Beyond D 640 the float64 solve
  hardly moves Δ from its seed (4e-6 from D 640 to 646, while Track C falls by 1.3e-4), since grip × residual is
  about 0.5 there. So past D 600 this continuation shows that a float64 solution exists and that the horizon's
  numerics stay flat; it does not give Δ.
- Δ − Track C is M 6's τ-truncation error: +3.7e-3 at D 100, at most +4.3e-3 near D 175, about +2e-3 at D 600–640.
  For comparison only, the D = ∞ inner problem's M = 5 truncation is +1.27e-3 (SECOND_ORDER.md §4). Its M-harmonic
  truncation is a different one, so this is not a check.

**Double-double, M ≥ 8.** The dd form (`mobius_sdd.py`) solves Track B's D-100 M-10 point to max|F| = 2.6e-24, with
Δ_M10(100) = 1.7450282484478268, which is Track C + 1.76e-6 (`dd_M10_cont_try*/`). Moving off it failed, in D and in M:

| attempt | result | log |
|---|---|---|
| M 10, D 100 → 102–110, Track B's split Newton (difference step 1e-3 \|g\|/σ_min) | its step leaves the linear range; the reduced Jacobian is garbage | `dd_M10_cont_try2/`, `_try3/` |
| the same with difference step 1e-9 | a weak step of 3.5e-3 lowers \|g\| but raises max\|F\| to 2.2e-8 | `dd_M10_cont_try4/` |
| M 6 → M 8 at D 200 | no step along the weak directions | `dd_Mcont/` |
| hold h5, h7 at D 200, M 8, then release (Track A's method) | every pairing of dropped rows tried gives a held system worse conditioned than the full one at this start (row-scaled 2.3e6 to 5.9e8, against 3.6e5) | `release/`, `diag/pairing_test_D200_M8.log` |
| plain damped Newton, M 8, D 200, from M 6 | after one step to 9.4e-9, only λ ≈ 5e-4 lowers \|F\|; Δ creeps by 1.5e-6 per step | `diag/newton_M8_D200_a.log` |
| M 8, D 200, harmonic 7 from the closed form (F + K₁/2n + K₂/2n², SECOND_ORDER §1.2) and Δ from Track C (`seed_closed.py`) | Newton still crawls (λ ≈ 0.03 by step 11); the float64 split Newton reaches max\|F\| 1.6e-12 with Δ = 1.696049, Track C + 9.6e-4, which is 1e-4 from the other float64 M-8 value (1.695951); float64 does not pin Δ at M 8 | `seed_closed/sc_D200_M8.log` |

What the last row shows is the obstacle. M 8's Δ at D 200 lies about 4e-3 from M 6's, which is M 6's τ-truncation error. That distance lies along the direction that harmonic 7's selection grips weakly (σ_min 1.2e-10 after the first step), far outside Newton's basin there, about σ_min²/curvature. Track A's held harmonics start from a fitted guess (the chain's held fit), so its Δ starts close and each release moves it little. A start from M 6 resampled has harmonic 7 at zero. a = 1 and a = 8 give identical Newton histories here, as they must: the residual is the same function of the unknowns in any horizon variable.

So Δ to 1e-8 needs:
- the high harmonics started from a fit (the chain's, or the closed form's F, K₁, K₂ of SECOND_ORDER §1.2);
- a precision above double-double for M ≥ 12.

That is hp's machinery (Track A, P0), which with the Moebius horizon no longer has D as its limit. It is not done here.

## 5. Cost against D

One residual, banded at kb = 150 (`diag/cost_vs_D.log`, M 6, the Mac, one process):

| D | 100 | 400 | 1000 | 2000 | 5000 | 10000 |
|---|---|---|---|---|---|---|
| ks | 316 | 757 | 1520 | 2703 | 6067 | 11477 |
| float64, s | 0.07 | 0.20 | 0.48 | 0.83 | 1.66 | 3.55 |
| double-double, s | 0.50 | 1.25 | 2.75 | 4.45 | | |

About 2–3×10⁻⁴ s per horizon order in float64 and 1.6×10⁻³ s in double-double: linear in n through ks. A float64
continuation step costs 2–4 min up to D 600 (Jacobians and the split Newton).

## 6. The inner problem at x ~ √n, and why the selection stays at x*

#1090 proposed the finite-n inner problem in x = ξ√n, where the fast mode is O(1) (nΦ = 1/(2ξ²) + 1/(4nξ⁴) + O(n⁻²),
notes/LIT_GAPS_LARGED.md L6), with the shadow's amplitude, the slow manifold's Stokes constant, as an unknown.

- **At finite n the inner region lies beyond the SSH.** The physical domain 0 ≤ x ≤ 1 is 0 ≤ ξ ≤ n^(−1/2) (BEYOND_LIGHT_CONE
  §2.5). The exact finite-n equations in ξ are the exact equations; they are singular at the SSH, ξ² = c²/n. The D = ∞
  inner problem is well posed because the SSH and the centre merge into ξ = 0 and the outer region supplies only the
  formal series. At finite n the inner region is reached from the physical one only through x = 1.
- **Through x = 1 on the real axis** the continuation is a Cauchy evolution in x (x is timelike there), and the fast mode
  grows outward from |1 − x⁻²|^(n/2): rounding near the SSH is amplified by at least 2^(n/2) (BEYOND_LIGHT_CONE §3(a),
  measured e^(0.546n) in its toy).
- **Around x = 1 in the complex plane** (condition E2, reality on the Stokes line) the modulus is flat in n, but the
  half-turn weights harmonic k by e^(πkω): the measured jump is 1.4e4 for k = 1 and 8.2e64 for k = 13 at every n, the
  basin is δ ≲ e^(−πkω), and the half-turn's label shift πω′ ≈ 12 exceeds the profile's analyticity strip, so for any
  iterate that is not the solution the harmonic series diverges after the turn (BEYOND_LIGHT_CONE §7). That is 70–85
  digits at every D, against 20–50 for the same harmonics at x*.
- **At x*** (condition E1) the fast mode is O(1) too, |F(x*)| = 1 for every n, and the harmonic ladder levels off at its
  large-n limit, of order e^(−πkω/2)√n (BEYOND_LIGHT_CONE §6), half E2's exponent.

So the place where the selection is O(1) and the harmonics are least weighted is x*, which is the layout already used
here, and the shadow's amplitude need not be an unknown: once the horizon series is computed stably (§3), its shadow
is computed with it. What an amplitude unknown could still buy is cost: the shadow's coefficients from σ_R on in a
factored form smooth in k/n would replace the ~n orders of the series by an O(1) number of samples. That is not built
here (§7).

## 7. What is not done, and what it would take

- **Cost flat in n.** The horizon order still grows like n: the shadow occupies orders σ_R ≈ n/2 to about n, and the
  series must reach them to impose analyticity (LARGE_D_BASICS D3). A factored tail, C(k, σ)λᵏ × an amplitude smooth
  in k/n with its continuum limit an ODE in κ = k/n, would make it O(1); with the banded recurrence the present cost is
  linear (§5), which reaches D 10⁴ in float64 on one core.
- **Δ to 1e-10.** That needs the M sequence to M ≈ 14–16 (each two harmonics cut the τ-truncation error by about 50,
  Track B's M 6–10 at D 100) and the precision for the harmonic ladder's weak directions at that M (double-double to
  about M 12, quad-double beyond). Not limited by D.
- **The hp port.** hp's form S (P0, `wip/large-d-form-s`) sums its horizon series in u = 1 − x, and Track A measured
  that it loses digits: amplification 78 at D 400 and about 4e14 at D 1000 (#1088). Track A is porting the Moebius
  variable into hp's form-S horizon series. Three things carry over:
  - a grown with n;
  - products with z (or with x, in a map of x) by their exact recurrence;
  - multidouble's lanes have double's exponent range, so the exponent-range constraint of §3 holds in octo-double too.
- **Δ at M ≥ 12** in hp, with the weak harmonics held at a fitted guess and released one at a time (Track A's method),
  now that the horizon does not limit D (§4).

## 8. Discrepancies

- **A retracted inference.** In #1088 (comment 7152680300) I inferred that hp's u = 1 − x should not lose digits,
  because x* sits nearer the SSH than the centre in u. Track A's measurement (78 at D 400, about 4e14 at D 1000) shows
  that it does; the inference was wrong and was corrected there (7153606604). The band-split test of §3 shows why the
  geometric argument is not enough: the loss sits in the resonance's band, whose growth depends on more than the
  ratio of distances.

- **Track B's double-double horizon rounds a constant to float64.** `solver/factored/fseries_sdd.py` forms
  c[k + 1] = (c[k] (2k) − G[k]) × (lam/(2(k + 1))), where lam/(2(k + 1)) is a float64 quotient. At its D-100 M-8
  solution its horizon sums differ from the mpmath reference by 5.8e-21 relative, and from this note's double-double
  by the same; with the quotient formed in double-double the two agree to 7.2e-31 (`check_mobius_dd.log`, and a
  patched copy in the scratch run). So Track B's dd solutions solve a problem perturbed at about 1e-21 at D 100. In
  the affine variable a relative error in c is amplified about 5e8 at D 200 (§2), which may be part of Track B's D-200
  stall at a weak residual of 4.4e-14; not tested.
- **Track B's M-6 Δ at D 100** is 1.7487024 (float64), this note's 1.748728938 (float64, Moebius): 2.6e-5 apart. At
  M 6 the grip on Δ is 2.4e11 with unit rows, so a float64 residual of 5e-14 leaves Δ uncertain to about 1e-5 in the
  worst case; the two are consistent with that, and neither pins Δ at M 6.

## 9. Commands

From `solver/inner_large_d/`, with the raw-data repository cloned beside the checkout
(`git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw`), so that
`R=../../../choptuik-large-d-raw`, and Track B's `solver/factored/` checked out from `wip/large-d-factored` into a
directory `FACT` (for the two checks that compare with it):

```
python3 check_mobius.py FACT $R/I1088-mac/Sdd_D100_M8.json                    # exit 0; --mutate=Lk: exit 1
python3 check_mobius_dd.py FACT $R/I1088-mac/Sdd_D100_M8.json --kb=150        # exit 0 (about 2 min)
python3 noise_variable.py $R/I1088-mac/Sdd_D100_M8.json --Ds=100,150,200,245,300,400   # exit 0; --mutate=affine: exit 1
python3 affine_orders.py $R/I1088-mac/Sdd_D100_M8.json --Ds=100,200,245
python3 solve_cont.py $R/I1088-mac/Sdd_D100_M8.json --M=6 --Ds=100,110,125,... --out=DIR --noise0=100,150,200,245.64,300,400
python3 solve_dd_cont.py $R/I1088-mac/Sdd_D100_M10.json --M=10 --Ds=100,110,125,... --out=DIR
python3 noise_at_solution.py SOL1.json SOL2.json --as=1,2,3,4 --bands=all,mid          # the band split of §3
python3 f64_vs_dd.py SOL1.json ... --as=0,1,8                                         # §4's float64 errors
python3 seed_test.py SOL.json TRACKC.json --Ds=1000,...,10000 --as=8 --a_rule=400 --Ddd=1000,2000,5000
python3 cost_vs_D.py $R/I1088-mac/Sdd_D100_M8.json --M=6                              # §5
python3 pairing_test.py SOL.json --D=200 --M=8 --hold=5,7                             # §4's held systems
(from hp/) python3 ../solver/inner_large_d/trackc_at.py OUT.json D1,D2,...            # Track C at the run D
python3 tabulate.py TRACKC.json DIR --md                                              # §4's table
```
SOL.json are the `sol_*.json` records of `I1090-mac/f64_M6_cont/`; TRACKC.json is `I1090-mac/diag/trackc_at.json`.

The exact commands of the runs, with their D lists, are in `I1090-mac/README.md`.

## 10. Independent check of hp's Moebius horizon

hp's port (`hp/src/form_s.rs` `horizon_mob`, `wip/large-d-form-s` f53e6077) against `mobius_sdd.py`. Both evaluate the
horizon sum at x_l = 0.7 from the same float64 m_h and Δ, with M = 32, ks = ⌈1.5n⌉ + 20 and a = max(8, n/400). The
checks and bars of `hp_mob_check.py` were committed before the runs (4dfefe69); the logs are in `hp_mob/`.

| D | m_h from | max\|S\| | hp − mine | gain from m_h, hp / mine | mine at a and 1.25a | verdict |
|---|---|---|---|---|---|---|
| 1000 | Track A's jump guess | 1.0 | 7.9e-31 | 0.55643 / 0.55643 | 1.9e-30 | PASS |
| 1000 | the D-646 solution rescaled | 1.0 | 1.9e-28 | 0.54604 / 0.54604 | 1.2e-28 | PASS |
| 2000 | Track A's jump guess | 40.7 | 9.3e-20 | 2.164e5 / 2.165e5 | 8.8e-20 | FAIL (bars) |
| 2000 | rescaled | 7.5e3 | 3.2e-20 | 3.836e5 / 3.834e5 | 3.1e-20 | FAIL (bars) |
| 5000 | rescaled | 1.0e53 | 3.3e-14 | | | FAIL (bars) |

- At D 1000 hp's octo-double agrees with my double-double at my dd's floor.
- From D 2000 the inputs on hand are far from a solution: horizon sums of 41 to 1e53 where a solution's is about 0.06.
  There my dd is consistent only to its C3 value. hp's own sums at a = 8 and 10 agree to 3.0e-25 (D 2000, rescaled),
  so the failures are the inputs' and my dd's, not hp's.
- Both controls fire: hp's mutate 52 misses by 11, and my z series at 1.25a by 1.0.
- hp has both corrections of §3 in effect:
  - its z products are full Cauchy sums, exact at O(ks²) cost;
  - a is the caller's, and 8 is the rule's value up to D 3200.
- The D 5000 rows with Track A's guess and the D 10⁴ rows were dropped on the coordinator's word: with m_h far from a
  solution they cannot pass or fail hp, and they cost hours at O(ks²). The check is to be rerun from Track A's
  converged states with `hp_mob_check.py --state=NPZ`:
  - D 2000 (`I1088-dh610`, j2000m);
  - later D 5000 and 10⁴, after Track A's port of the z recurrence.
