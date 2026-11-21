# LOW_D_SONIC: getting the low-D solution below the sonic closing (#997, experiment (b))

Subagent of the coordinator on the Mac, 2026-11-21. Code in [`solver/lowd_sonic/`](../solver/lowd_sonic/), raw output in
folder `I997-mac-sonic/` of the raw-data repository. Status words as in AGENTS.md; every number here is *numerical
evidence* unless marked.

RESULTS_PLACEHOLDER

## Pass criteria, written before the runs (2026-11-21 18:55 UTC)

Planted truth (the warped equations are an exact change of the curves one marches on, so they must reproduce hp):

- **P1, segment.** hp's converged D 3.7 state at M 256 (`hp/data/lowD_states/highm15_D3.7000_M256.npz`), its node state
  at x 0.01 marched to 0.02 on warped curves (the test warp, max |φ| 1.74×10⁻³): the end state equals hp's node state at
  0.02 to 10⁻¹³ relative. Control `--mutate nowarp` (the term ω h_θ dropped from both gaps): above 10⁻⁶.
- **P2, solve.** hp's D 3.7 state at M 128 (Δ 3.465200482271642), the band 0.01 → 0.02 replaced by the warped band
  with the test warp, the chord iteration: converges (|F| < 10⁻¹²) with |Δ − 3.465200482271642| < 2×10⁻⁹ (the warped
  band at M 128 differs from hp's at 1.7×10⁻⁹ of the fields, the τ truncation of M 128). With φ = 0: |Δ − Δ_hp| <
  10⁻¹². Control `--mutate nowarp`: Δ off by more than 10⁻⁶, or no convergence; exit 1.
- **P3, solve at D 4.** hp's D 4 state at M 96 (Δ 3.445452402227911), the same: |Δ − 3.4454524022| < 2×10⁻⁹.

The real test (issue #997's bar):

- **R1.** hp's D 3.045 state at M 320, the band 0.01 → 0.02 warped (ε₀ = 2×10⁻³, the warp designed from that state):
  converges, with |Δ − 2.4464747274661773| < 10⁻⁸ (hp's M 384 value).
- **R2.** Continued to D 3.044, 3.043, 3.042, 3.041, 3.040 from the same LU: converges at D ≤ 3.043 (below the closing
  at 3.04312), with min g_p > 0 on the band and min (c − x) < 0 there (the patch where c < x exists and is crossed).

Two changes after the criteria were written, with the reason: P3 runs on the stored D 4 state with nodes 0.8 and 0.9
added (its last node 0.7 makes hp's horizon block fail for a reason outside the band, section 3), and the real test
lists the D of hp's own last points (3.04375, 3.04335, 3.04315, each checked against hp's M 384 value to 10⁻⁸) and then
3.0430, 3.0428, 3.0425, 3.042, 3.041, 3.040.

## 1. Stage 0: does the closing exist in each candidate's coordinates?

hp marches x(x − c) p_x = −ωx p_θ + G p − c p + (n/2) c (p + m) in x on curves of constant x. A curve of constant x is
null where x = c (the norm of ∂_τ at fixed x is proportional to x² − c²), and there the p equation is singular. The
closing is the first point where such a curve touches a characteristic inside the domain (notes/LOW_D.md §17).

| candidate | marching surfaces | closing near D 3.04? | why |
|---|---|---|---|
| (i) Ecker, Ecker, Grumiller & Jechtl (Vienna), their code and data | curves of constant x in the same gauge | **yes, the same one** | Their V equation has the denominator f − x (`code/delta/inner/derivs.f`), with f_x = (D − 3) f (1 − 1/a²)/x and f = 1 at x = 1 (`rightdata.f`): f is our c and x our x. Their exported D 3.046 fields have min (f − x) = 1.640×10⁻⁴ at x 0.014596, where hp's gaps give 1.62×10⁻⁴. Their data stop at 3.046. |
| (ii) Reiterer–Trubowitz (`solver/rt/`) | none: a global Fourier × Chebyshev collocation in (t, ξ) | **no, by construction** | The curves of constant ξ have tangent ∂_t, with g_tt = −e^{−2ζ}(1 − ξ²) < 0 for \|ξ\| < 1 (RT_FORMULATION.md §1): timelike everywhere inside, whatever the solution. The characteristics dξ/dt = ±μ(1 ± ξ) are tangent to them only at ξ = ∓1, the horizon. |
| (iii) hp's gauge marched on warped curves x = ζ + b(ζ) φ(θ) (`solver/lowd_sonic/band.py`) | curves chosen to stay off the characteristics | **no, where a warp exists** | The marching is singular only where g_p = c − x + ω h_θ or g_m = c + x − ω h_θ vanishes. Through a valley or a small patch of c < x a periodic φ with ω b φ_θ > x − c keeps g_p > 0. The device is Gundlach's τ-dependent radial coordinate ζ = ln x − ξ₀(τ) (PRD 55, 695 (1997), Sec. II.3), there used to put the horizon on ζ = 0. |

**(i) in detail** (`ecker_gap.py`, raw `I997-mac-sonic/ecker_gap.log`; their repository at f002d0af, read only).

- *Solutions below 3.046:* none. Their lowest D is 3.046 (`complete_data/CppData/data_lowD_1024_4.json`, 2048 τ points,
  converged, mismatch 1.3×10⁻¹²). The only rows below 3.046 in `plots/*.dat` are Bland's thesis point at 3.02 (an
  evolution value) and three points of a fitted line (`delta_sing_fit.dat`).
- *The (D − 3)^0.15 curve:* the notebook draws y = 3.87819 (x − 3)^0.15 on x = linspace(3.01, 3.05, 20) from
  hard-coded coefficients (`plots/plots.ipynb`, the cell that saves `extrapolateDelta.pdf`). It fits nothing below 3.046.
- *Resolution:* 2048 τ points from 3.046 to 3.096; the only resolution pairs are at 3.094–3.096 (1024 → 2048 points:
  −9.9×10⁻¹⁰, 6.9×10⁻¹¹, 1.1×10⁻¹²). Their left integration steps grow as D falls: 20000 at 3.081, 67000 at 3.051,
  120000 at 3.0475, 200000 at 3.046 (matching point 0.019, just above the valley at 0.0146). That is the stiffness of
  the valley, as for hp's plans.
- *Agreement with hp* (Vienna − hp, hp's best row of `hp/data/delta_dd_lowD.csv`): +6.9×10⁻¹⁰ at 3.0475, +5.0×10⁻¹⁰ at
  3.05, +4.1×10⁻¹⁰ at 3.1.
- *The valley in their data:* at 3.046 their min (f − x) over 104 τ samples and 559 x samples is 1.640×10⁻⁴ at x
  0.014596. A sampled minimum lies above the true one; hp's grid gaps at 3.045 and 3.0475 (LOW_D.md §17) interpolate
  linearly to 1.615×10⁻⁴ at 3.046 (computed in section 5's reproduction). So an independent code, which shares only the
  equations and the gauge with hp, sees the same valley closing at the same rate. At D 4 and 5.5 their smallest f − x
  is at the horizon (x 0.999, 0.992), not inside.

**(ii) in detail.** RT's gauge has no interior sonic point, so the closing is not in it. The present code does not reach
low D, for two reasons that were not worked around here. Its seed map (`gg_to_rt.GGMap`) interpolates hp's fields on a
global Chebyshev grid in x of 48 points and solves the transport w⁻_τ + (x − c) w⁻_x = λ w⁻, whose coefficient vanishes
at x = c: the same singular point, so the map from hp's states needs a valley-aware rewrite. The D 3.045 fields vary on
scales far below what 48 global points hold (a rises from 1.0001 to 6.03 between x 1.57×10⁻⁴ and 0.01, and p/x reaches
666 at the centre). And its Newton factors a dense Jacobian of L = 3.5 N_t (N + 1) unknowns; at D 4 it needed 32 × 61
for 1.2×10⁻⁹ (RT_FORMULATION.md §3), and the grid needed at 3.04 is not known.

**(iii)** passes the desk check and is the one prototyped: it is the smallest change to hp (one segment), it reuses hp
everywhere else, and its cost is that of hp's own low-D solve.

## 2. The formulation

On the band x_A ≤ ζ ≤ x_B (here hp's nodes 0.01 and 0.02, around the valley at 0.0143) the fields of hp's gauge are
marched in ζ on the curves x = h(θ, ζ) = ζ + b(ζ) φ(θ), b = sin²(π(ζ − x_A)/(x_B − x_A)), so the band's ends are hp's
curves of constant x and every other segment is hp's. φ has period π in θ, which keeps the parities of p, m (odd) and
a, c (even). With p_x = p_ζ/h_ζ and p_θ|_x = p_θ|_ζ − (h_θ/h_ζ) p_ζ:

```math
p_\zeta = -\frac{h_\zeta\,(R_p - \omega x\, p_\theta)}{x\, g_p},\qquad m_\zeta = \frac{h_\zeta\,(R_m - \omega x\, m_\theta)}{x\, g_m},\qquad g_p = c - x + \omega h_\theta,\quad g_m = c + x - \omega h_\theta,
```

```math
a_\zeta = \frac{h_\zeta\,[a(p^2 + m^2) + (n-1)(a - a^3)]}{2x},\qquad c_\zeta = \frac{h_\zeta\, G}{x},
```

with R_p = G p − c p + (n/2) c (p + m), R_m = −G m + c m − (n/2) c (p + m), G = (n − 1) c (a² − 1). φ = 0 is hp's
marching. The warp used at low D is designed from the state above the closing: ω φ_θ = (S(ε₀ − g*) − mean)/0.85, with
g*(θ) the smallest c − x over the band at each θ and S a softplus of width ε₀/4 (`band.design_phi`). At D 3.045, M 320,
ε₀ = 2×10⁻³: max |φ| 1.58×10⁻⁴, and the band's smallest g_p rises from 1.06×10⁻⁴ (hp's curves) to 1.48×10⁻³. The
integrator (DOP853's eighth-order weights, fixed steps of 1/rate) then needs 3632 steps in the band instead of 16802.

The solve (`hybrid.py`) is hp's multiple shooting (hpkernel.Solver: centre series, Taylor steps, horizon series, phase
row) with the 4M rows of the band segment replaced by Φ_warp(s_A) − s_B in f64. Newton's matrix is one LU at the start
D. Two forms:

- `--jac band` (M ≤ 128, where hp gives its dense Jacobian): hp's f64 Jacobian with the band rows replaced by the warped
  band's own (forward differences on the band's plan).
- `--jac hp --aa_m 8` (any M): hp's own f64 Jacobian and LU (Solver.newton_mixed), whose band rows are those of hp's
  unwarped band, with Anderson mixing of the last 8 chord steps (Anderson, J. ACM 12 (1965) 547; Walker and Ni, SIAM J.
  Numer. Anal. 49 (2011) 1715). The warped and unwarped band maps differ in the harmonics near the top of the τ grid
  (they agree to rounding once those are resolved, P1), and a plain chord step with hp's band rows diverges along them
  once the warp is large: at D 3.7, M 128 it converges for the test warp at A 1.5×10⁻⁴ and diverges at 5×10⁻⁴ and
  1.5×10⁻³. With Anderson mixing it converges at both (11 and 14 steps).

## 3. Planted truth

| check | case | result | bar | verdict |
|---|---|---|---|---|
| P1 | D 3.7, M 256, segment 0.01 → 0.02, test warp (max \|φ\| 1.74×10⁻³) | 1.04×10⁻¹⁴ relative, 82 steps | < 10⁻¹³ | OK |
| P1 | φ = 0 | 7.9×10⁻¹⁵ | < 10⁻¹³ | OK |
| P1 control | `--mutate nowarp` | 2.0×10⁻², exit 1 | > 10⁻⁶ | fires |
| P2 | D 3.7, M 128, solve with the test warp (`--jac band`) | Δ 3.465200482271642, \|F\| 4.8×10⁻¹⁵ in 2 steps | \|ΔΔ\| < 2×10⁻⁹ | OK (difference 0) |
| P2 | φ = 0 | Δ − Δ_hp = 0 | < 10⁻¹² | OK |
| P2 control | `--mutate nowarp` | Δ − Δ_hp = −3.7×10⁻⁴, no convergence, exit 1 | > 10⁻⁶ | fires |
| P3 | D 4, M 96, nodes + 0.8, 0.9, test warp | Δ 3.4454524022279087 (2.8×10⁻¹¹ from 3.4454524022; −2.2×10⁻¹⁵ from hp's M 96) | < 2×10⁻⁹ | OK |
| P3 control | `--mutate nowarp` | converges to Δ − 3.4454524022 = −3.3×10⁻⁵, exit 1 | > 10⁻⁶ | fires |

The warped map's difference from hp's next node falls with M, as a τ truncation does: at D 3.7 9.3×10⁻⁸ (M 96),
1.7×10⁻⁹ (M 128), 3.8×10⁻¹¹ (M 160), 4.8×10⁻¹⁵ (M 256); at D 3.045 with the designed warp 1.4×10⁻⁸ (M 320) and
5.7×10⁻¹⁰ (M 384), against 5.8×10⁻¹⁴ unwarped (raw `seg3045.jsonl`). Δ does not see it: at D 3.7, M 128 the solve
with the test warp returns hp's Δ to the last bit.

The brief's D 3.7 value, 3.4652004835, is hp's M 80 value, 1.2×10⁻⁹ above the converged 3.465200482271639 (M ≥ 160).

**The stored D 4 state's horizon block** (`d4hm_D4.0000_M96.npz`, last node 0.7, horizon series of 200 terms at u 0.3)
is hypersensitive: a change of 10⁻⁹ in one horizon unknown changes that block by 1.3×10⁻⁷, and a change of 10⁻⁶ by
6×10¹⁸. The first chord step, of 3×10⁻⁹ in the horizon data, took the block to 5.2 (raw `p3_lastnode0.7.log`). With
nodes 0.8 and 0.9 added (the horizon series at u 0.1) P3 converges. On that state hp's f64 Jacobian also differs from
its double-double one by 2.3×10⁻⁶ in the horizon block and by at most 9×10⁻¹³ elsewhere.

## 4. The real test

REAL_PLACEHOLDER

## 5. What it means for hp, and what is proposed

PROPOSALS_PLACEHOLDER

## 6. Discrepancies with the notes (reported, not fixed)

- **Bland's D 3.02 point** (notes/LIT_REVIEW.md, the 2026-11-04 entry): "They cite Bland's thesis (2007) for D = 3.02:
  Δ = 0.1379 ± 0.0042". In their repository 0.1379 ± 0.0042 is Bland's γ at 3.02 (`plots/gBlandThesis.dat`), and his Δ
  there is 2.083 ± 0.024 (`plots/DeltaBlandThesis.dat`). Their paper's own sentence was not reread here.
- **"fits Δ ≈ 3.88 (D − 3)^0.15 on 3.01–3.05"** (notes/LOW_D.md §6; notes/DIRECTION_REVIEW.md §2.6: "Ecker et al.'s
  plotting notebook fits values on 3.01–3.05"): the notebook plots a curve with hard-coded coefficients over 3.01–3.05;
  it fits no values there, and the repository has no solution below 3.046 (section 1). DIRECTION_REVIEW's open question,
  whether their values below 3.043 are converged, has the answer that there are none.

## 7. Reproduction

From `solver/lowd_sonic/` with `PYTHONPATH` naming an hpkernel site (here #1043's build), the raw-data repository
cloned beside the checkout as `../choptuik-large-d-raw`, and hp's states from it (`T82-win/states/`, `T105-win2/states/`,
`I343-win2/states/`, `I945-win2/leg/states/`):

    python3 segment.py OUT.jsonl ../../hp/data/lowD_states/highm15_D3.7000_M256.npz --warp test           # P1
    python3 segment.py OUT.jsonl ../../hp/data/lowD_states/highm15_D3.7000_M256.npz --warp test --mutate nowarp
    python3 hybrid.py OUT.jsonl --state ../../hp/data/lowD_states/highm15_D3.7000_M128.npz --Ds 3.7 --warp test \
        --expect 3.7:3.465200482271642:2e-9                                                              # P2
    python3 hybrid.py OUT.jsonl --state ../../hp/data/lowD_states/d4hm_D4.0000_M96.npz --add_nodes 0.8,0.9 \
        --Ds 4.0 --warp test --expect 4.0:3.4454524022:2e-9                                              # P3
    python3 -I ecker_gap.py <clone of github.com/EckerChristian/CriticalSpacetimeCrystal>

(each with `--allow_unraw` or an output path in a task folder of the raw-data repository; add `--mutate nowarp` for the
controls). The real test's command is in raw `I997-mac-sonic/real/run_real.sh`.
