# LARGE_D_INNER: a horizon variable whose rounding does not grow with D (#1090)

Issue #1090 (quest P1 of #1088). Written by a subagent of the coordinator on the Mac, 2026-11-25. Code in
[`solver/inner_large_d/`](../solver/inner_large_d/); raw output in the folder `I1090-mac/` of the raw-data repository.
Status words as in AGENTS.md. Every number below comes from a log in `I1090-mac/` named beside it.

## 0. Verdicts

| question | answer | status |
|---|---|---|
| Does form S need about 0.06n more digits at fixed precision, from the scalar mode's shadow? (#1090's premise) | **No.** The loss is a forward-recurrence instability of the affine horizon variable, which a Moebius variable removes (§2–§3) | numerical evidence, D 100–400 at seeds (§3), and at solutions (§4) |
| Where does the affine variable lose its digits? | at low orders already: the critical solution's horizon series is the minimal solution of its recurrence; at D 200 the float64 relative error passes 1 at order 35 of σ_R ≈ 102, and the resonance at σ_R injects it into the shadow (§2) | numerical evidence |
| The inner problem at x ~ √n with the Stokes constant as an unknown, at finite n | not the place to impose the selection; the selection is read at x* (§6) | argued from measured weights (BEYOND_LIGHT_CONE §7) |
| Cost against D | the horizon order still grows like n (ks ≈ n + 14√n + 80), banded: linear in n | measured (§5) |
| Conditioning against D | set by the harmonic ladder of the selection, which the layout does not change (§4) | numerical evidence |
| Δ | §4–§5 | numerical evidence |

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
| double-double, a = 1, against the mpmath reference at 40 digits (`check_mobius_dd.log`) | 9.4e-31 relative | 1e-29 | |
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
- The float64 continuation runs at a = 8 from D 400 on. Whether a must grow with n beyond that is what its per-D
  noise test at a = 1, 8 and 32 measures (§4).

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
- **The hp port.** hp's form S (P0, `wip/large-d-form-s`) sums its horizon series in u = 1 − x. Whether u loses digits
  the same way is not measured; the test is `noise_variable.py`'s on hp's series. If it does, the Moebius map is a
  change of the series' coefficient functions only.

## 8. Discrepancies

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
```

The exact commands of the runs, with their D lists, are in `I1090-mac/README.md`.
