# LOW_D_SONIC: getting the low-D solution below the sonic closing (#997, experiment (b))

Subagent of the coordinator on the Mac, 2026-11-21. Code in [`solver/lowd_sonic/`](../solver/lowd_sonic/), raw output in
folder `I997-mac-sonic/` of the raw-data repository. Status words as in AGENTS.md.

## Pass criteria, written before the runs (2026-11-21 18:55 UTC)

Planted truth (the warped equations are an exact change of the curves one marches on, so they must reproduce hp):

- **P1, segment.** hp's converged D 3.7 state at M 256 (`hp/data/lowD_states/highm15_D3.7000_M256.npz`), its node state
  at x 0.01 marched to 0.02 on warped curves (the test warp, max |φ| 1.95×10⁻³): the end state equals hp's node state at
  0.02 to 10⁻¹³ relative. Control `--mutate nowarp` (the term ω h_θ dropped from both gaps): above 10⁻⁶.
- **P2, solve.** hp's D 3.7 state at M 128 (Δ 3.465200482271642), the band 0.01 → 0.02 replaced by the warped band
  with the test warp, the chord iteration from hp's Jacobian: converges (|F| < 10⁻¹²) with |Δ − 3.465200482271642| <
  2×10⁻⁹ (the warped band at M 128 differs from hp's at 1.7×10⁻⁹ of the fields, the τ truncation of M 128). With
  φ = 0: |Δ − Δ_hp| < 10⁻¹². Control `--mutate nowarp`: Δ off by more than 10⁻⁶, or no convergence; exit 1.
- **P3, solve at D 4.** hp's D 4 state at M 96 (Δ 3.445452402227911), the same: |Δ − 3.4454524022| < 2×10⁻⁹.

The real test (issue #997's bar):

- **R1.** hp's D 3.045 state at M 320, the band 0.01 → 0.02 warped (ε₀ = 2×10⁻³, the warp designed from that state):
  converges, with |Δ − 2.4464747274661773| < 10⁻⁸ (hp's M 384 value).
- **R2.** Continued to D 3.044, 3.043, 3.042, 3.041, 3.040 from the same LU: converges at D ≤ 3.043 (below the closing
  at 3.04312), with min g_p > 0 on the band and min (c − x) < 0 there (the patch where c < x exists and is crossed).
