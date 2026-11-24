# L1_BRANCH_OVERLAP: is the falling Jacobian mode of the ℓ = 1 legs the λ₁ eigenmode?

Issue #1085, from #1025. On the ℓ = 1 legs past D 21, one regular singular value of the DSS solver's Jacobian J falls
at every point. It is the mode tracker's m1, at k_c 71, 79 and 87, and it is nearly orthogonal to the continuation
tangent. Meanwhile λ₁ falls toward 0.

The hypothesis H is that this mode is the λ₁ eigenmode. Then J would become singular at the D where λ₁ = 0, a branch
point of the ℓ = 1 family.

This note tests H on stored states with `solver/matter/ell_overlap.py` (branch `wip/l1-branch-overlap`). Raw output is
in `I1085-mac/` of the raw-data repository. Written 2026-11-24 by Claude (Anthropic) under Nathan Moinvaziri's
direction; not reviewed. Status words as in AGENTS.md.

## Result

- **J becomes singular exactly when λ₁ reaches 0.** This is *proved* given the relation of §1, which is checked
  numerically (M(0) equals J's spectral block bit for bit). The proof uses the bordered determinant and Jacobi's
  formula: det J = (2π/Δ²)(e_p·t) (d/dλ) det M(λ) at λ = 0, and that derivative carries the factor (0 − λ₁).
- **H as stated is not supported** (*numerical evidence*, k_c 87 at D 21.125 and 21.75, k_c 79 at D 21.375). Two
  findings stand against it:
  - The falling mode's spectral part is not near-null for M(λ₁). |M(λ₁) x_m1| = 8.1e-2, against 1.7e-12 for the λ₁
    eigenvector, and |M(λ) x_m1| stays between 6e-2 and 9e-2 for λ in [0, 1.2] (§3).
  - The smallest singular value of M(λ) on the span of J's four weakest modes and the translation mode has no dip at
    λ₁ at any of the three states (§4). So the λ₁ eigenmode does not lie in J's weak band.
- **The overlap test posted on #1085 does not decide, by construction.** In float64 the λ₁ eigenvector cannot be
  separated from M's numerical null space: about 80 directions at the rounding floor, which turn with λ. Its controls
  fired, at 0.65 to 0.93 against a bar of 0.3 (§2).
- **The falling mode is J's period mode.** It is the only one of the four weakest modes with a sizable Δ component
  (|δ| = 0.137, 0.145 and 0.138 at the three states; 0.002 to 0.010 for the others, with one exception at k_c 79 noted
  in §3). It carries 95 to 99% of |J⁻ᵀ e_Δ|², the records' `dDelta_dF` (§3).
- **The period mode's link to λ₁ is *conditional*.** By Cramer's rule, |J⁻ᵀ e_Δ| = |c_Δ| / |det J|, with c_Δ the
  cofactors of J's Δ column. So σ_m1 ≈ |δ_m1| |det J| / |c_Δ| carries (0 − λ₁) through det J. The falling mode would then
  vanish at λ₁ = 0 if two conditions hold: |c_Δ| stays finite there, which is the generic, Jordan-type crossing, and the
  mode keeps carrying |J⁻ᵀ e_Δ|. At k_c 79 a second mode with a Δ component sits at the numerical cutoff (§3), so the
  second condition is not secure.
- **Crossing D: an extrapolation, not a result.** λ₁ = 0 extrapolates to D ≈ (see §5) at k_c 87. Under the conditional
  mechanism, the falling mode vanishes there too.

## 1. How J and M(λ) are related

In `dss_ell.py`, with n spectral unknowns:
- `jacobian(z)` is (n+1)×(n+1): the shooting mismatch rows and the phase row, over the spectral columns and Δ.
- `mode_matrix(z, λ)` is n×n: the same blocks at fixed Δ, with every τ-derivative of a perturbed member replaced by
  D f + λ (f − f₀).

From the code:
- (R1) J[:n, :n] = M(0).
- (R2) J's last row is the unit phase row e_p (Im of the centre datum's first harmonic), and J[n, n] = 0.
- (R3) J[:n, n] = b = ∂F/∂Δ = −(2π/Δ²) M′(0) t, with t = ∂_θ z the translation mode. ω enters only through ω ∂_θ, and λ
  shifts exactly that term, so ∂F/∂ω = M′(0) t when the discrete equations commute with translations.
- (R4) M(0) t = O(|F|): the translation mode is a null vector of M(0), up to the residual and aliasing.

With M(0) t = 0 and M(0) of rank n − 1, the adjugate of M(0) is t ŵᵀ. The bordered determinant det J = −e_pᵀ adj M(0) b
and Jacobi's formula (d/dλ) det M = tr(adj M · M′) then give

```math
\det J = \frac{2\pi}{\Delta^2}\,(e_p\cdot t)\,\frac{d}{d\lambda}\det M(\lambda)\Big|_{\lambda=0} .
```

The derivative carries the factor (0 − λ₁). So J becomes singular exactly when a root of det M reaches the translation
root at λ = 0. The determinant does not say which singular value of J carries the factor.

On the state checked (k_c 87, D 21.125; `floor_kc87_D21.125.log`, `run_kc87_D21.125.log`):
- R1: the full M(0) equals J[:n, :n] bitwise. The matvec form of R1, 7.8e-11, missed its first bar (1e-12), but it lies
  below its own rounding floor, 1.25e-10 (the same matvec with the complex step scaled by 3).
- R2 holds exactly.
- R3: 2.0e-8 relative.
- R4: 4.6e-9 relative to |J| |t|.
- e_p·t/|t| = 7.3e-3: the phase row pins the translation only weakly.

## 2. The overlap test as posted, and why it does not decide

The plan and the reading were posted on #1085 before any overlap was computed (comments 7138890561 and 7139117917).
The λ₁ null vector v₁ was the eigenvector of the linearized pencil M(λ_r) v = −μ M′(λ_r) v with μ nearest 0, the method
of successive linear problems.[^ruhe] It was compared with J's falling mode through its J vector (P_e v₁, 0).

At k_c 87, D 21.125 (`run_kc87_D21.125.log`):
- The eigenpair is sound: λ₁ = 0.877342, inside the scan's bracket (0.875, 0.880). |M(λ) v₁| is 1.07e-5 at λ₁ ± 0.005
  and 1.66e-12 at λ₁.
- The statistic reads no verdict. Its overlap with m1 is 0.351 and with m0 0.925, and the wrong-λ controls give 0.651
  and 0.723, above their bar of 0.3. So the controls fired and the work paused.

The cause has two parts, found by the diagnostics `diag` and `geom`.
- **J's weak vectors carry Δ components.** A near-null vector w = (x, δ) of J has M(0) x ≈ −b δ, so its spectral part x
  is not near-null for M(0). For the four weakest modes, |M(0) x| equals |b| |δ| to the digits printed. The posted J
  vector (P_e v₁, 0) had no Δ component.
- **The λ₁ eigenvector is inseparable from M's numerical null space.**
  - v₁ lies 100.00% in M(λ₁)'s null space (81 vectors below 1e-12 σ_max) at D 21.125.
  - It lies 1.000000 in M(λ₁ − 0.05)'s null space (80 vectors) at D 21.75. What remains, normalized, is not V-shaped:
    |M(λ) x| = 0.336 at λ₁ and λ₁ ± 0.005.
  - At k_c 79, D 21.375 the weight is 0.999999 (72 vectors), and the remainder gives 0.133 flat.
  - The null space turns with λ, so the λ₁ mode cannot be cut from it in float64. The Δ-corrected statistic O*
    therefore tracks its control as well (0.899 against 0.735 at D 21.75).

The planted test did not catch either part. In the planted pencil the λ₁ mode was decoupled from the translation, and
the junk was fixed in λ.

## 3. What the falling mode is

Rows from `geom_*.log`. δ is the Δ component of J's right singular vector, and |M(0) x| is for the unit spectral part.

| state | mode | σ | δ | |M(0) x| |
|---|---|---|---|---|
| k_c 87, D 21.125 | m0 | 4.327e-11 | +0.0048 | 2.1e-3 |
| | **m1 (falling)** | 1.391e-10 | **+0.137** | 6.1e-2 |
| | m2, m3 | 1.887e-10, 1.894e-10 | +0.0073, +0.0044 | 3.2e-3, 1.9e-3 |
| k_c 87, D 21.75 | m0 | 5.547e-11 | −0.0066 | 2.9e-3 |
| | **m1 (falling)** | 8.080e-11 | **−0.145** | 6.5e-2 |
| | m2, m3 | 3.212e-10, 3.227e-10 | +0.0028, +0.0023 | 1.3e-3, 1.0e-3 |
| k_c 79, D 21.375 | (cutoff mode) | 1.442e-11 | −0.075 | 3.3e-2 |
| | record's m0 | 6.583e-11 | −0.0103 | 4.6e-3 |
| | **falling** | 1.387e-10 | **−0.138** | 6.2e-2 |
| | next | 3.360e-10 | −0.0048 | 2.1e-3 |

At k_c 79, D 21.375, this Mac finds n_null = 71, where win2's record has 72. The mode at 1.442e-11 sits just above the
cutoff here (1e-12 σ_max = 1.427e-11) and just below it on win2. So the falling mode is index 2 here, and the run's
`--mode 1` statistics at that state refer to the record's m0, not the falling mode. The four J rows above are printed in
full, so they do not depend on the index.

`dDelta_dF` is ell_leap.py's |J⁻ᵀ e_Δ| = (Σ_i (δ_i / σ_i)²)^{1/2}, over σ above lstsq's cutoff, n ε σ_max. The falling
mode's δ/σ against it:

| state | δ/σ of the falling mode | dDelta_dF (record) | share of the square |
|---|---|---|---|
| k_c 87, D 21.125 | 9.82e8 | 9.99e8 | 0.97 |
| k_c 87, D 21.75 | 1.80e9 | 1.81e9 | 0.99 |
| k_c 79, D 21.375 | 9.93e8 | 1.02e9 | 0.95 |

Along the legs, σ × dDelta_dF of the tracked falling mode (`refit_*.log`) stays near |δ|. At k_c 87 it holds at every
point; at k_c 79 it holds from D 21.25 on. Before D 21.25, k_c 79's dDelta_dF was 3.2 to 3.9e9, carried by the mode
then just above lstsq's cutoff (n ε σ_max ≈ 1.6e-11), and it fell to 9.3e8 when that mode dropped below it.

By Cramer's rule, J⁻ᵀ e_Δ = c_Δ / det J, with c_Δ the vector of cofactors of J's Δ column. With §1:

```math
\sigma_{m1} \approx \frac{|\delta_{m1}|}{|J^{-T}e_\Delta|} = |\delta_{m1}|\,\frac{2\pi}{\Delta^2}\,\frac{|e_p\cdot t|\,\big|\tfrac{d}{d\lambda}\det M(0)\big|}{|c_\Delta|} .
```

The first equality holds to the share in the table. The falling mode's singular value therefore carries (0 − λ₁). The
statement that it reaches 0 with λ₁ is *conditional* on two things:
- |c_Δ| stays finite at the crossing. That is the generic case, a Jordan block of the translation root. c_Δ vanishes
  only if the null space at the crossing is two-dimensional.
- The falling mode keeps carrying |J⁻ᵀ e_Δ|. The k_c 79 cutoff mode (δ = −0.075, σ = 1.44e-11, so δ/σ = 5.2e9) shows
  that a second Δ-carrying mode exists near the numerical floor. lstsq's cutoff hides it, and it may be noise-dominated.

This is not H. The falling mode is the period mode of the bordered system, (x, δ) with M(0) x ≈ −b δ. Its x is the
candidate generalized eigenvector of the translation root, not the λ₁ eigenvector at λ₁ ≈ 0.65 to 0.88.

## 4. No dip at λ₁ on J's weak band

Let Q be an orthonormal basis of span{x_m0, …, x_m3, t}, with x the unit spectral parts of J's four weakest right
singular vectors. s(λ) is the smallest singular value of M(λ) Q. If the λ₁ mode lay in that span, s would dip at λ₁.
The criterion, written before the run (#1085 comment 7139667226), was s(λ₁) ≤ 0.1 min s(λ₁ ± 0.05).

| state | s(0) | λ₁ − 0.05 | λ₁ − 0.005 | λ₁ | λ₁ + 0.005 | λ₁ + 0.05 |
|---|---|---|---|---|---|---|
| k_c 87, D 21.125 | 4.35e-11 | 2.257e-4 | 2.380e-4 | 2.394e-4 | 2.407e-4 | 2.530e-4 |
| k_c 87, D 21.75 | 5.55e-11 | 2.335e-4 | 2.513e-4 | 2.533e-4 | 2.552e-4 | 2.730e-4 |
| k_c 79, D 21.375 | 6.28e-11 | 4.186e-4 | 4.443e-4 | 4.472e-4 | 4.500e-4 | 4.758e-4 |

s rises monotonically through λ₁ with no dip. At λ = 0 the minimizer is m0 with a little m1 (s = 4e-11 to 6e-11): the
combination whose Δ components cancel is near-null for M(0).

## 5. σ against λ₁, refitted from the records

(from `refit_kc87.log`, `refit_kc79.log`)

## 6. What is open

- Following the mode carrying |J⁻ᵀ e_Δ| toward the crossing would test the conditional statement of §3 directly. The
  leg records would need the Δ components of the four weakest modes, which ell_leap.py already holds in Vt. Then σ × δ⁻¹
  × dDelta_dF ≈ 1 must hold for the falling mode, and dDelta_dF × λ₁ must stay bounded.
- A clean overlap with the λ₁ eigenvector needs that vector modulo M's null space, which float64 does not give here.
  Double-double arithmetic for the pencil solve would.
- The tracker's identity rule follows n_null, which differs between machines when a mode sits at the cutoff (k_c 79,
  D 21.375: 71 here, 72 on win2).

[^ruhe]: A. Ruhe, Algorithms for the nonlinear eigenvalue problem, SIAM J. Numer. Anal. 10 (1973) 674.
