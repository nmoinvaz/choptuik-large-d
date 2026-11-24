# The mode determinant near D 3: where its sign is lost, and a matched determinant (#1083)

Status: design and pass criteria written before the prototype's checks; prototype `hp/mode_matched.py` in progress.
Branch `wip/mode-horizon-1083`. This note starts from #1078's measurements (dh370). The status words are AGENTS.md's.

Throughout, n = D − 2, x is the self-similar radius with the horizon at x = 1, and u = 1 − x. M is the number of τ
points on the half grid (M′ 96 here). K is the number of nodes, and the state is the one stored as raw blob 6a0f83d2
(D 3.044375, 21 nodes from x 0.000157 to 0.975, τ-map s₀ 1.9635, L 0.4) unless another state is named. Numbers
marked "Mac" come from the hpkernel build of 2026-11-08 (venv-314), on the plan `Solver.plan(z, 1e-32, 0.05)` (1269
steps), as `hp/lowD_gamma.py` uses it.

## 1. M(λ) is an Evans matrix in multiple-shooting form

The unknowns of the mode matrix are the centre data c = (p₁, c₀), 2M of them; the node states n₀ … n_{K−1} (p, m, a, c
on the τ grid, 4M each); and the horizon data h = (m_h, a_h), 2M. On the default layout every segment is integrated
outward, and the block rows are

```math
C_0 c - n_0, \qquad \Phi_b n_{b-1} - n_b \;\; (b = 1, \dots, K-1), \qquad H h - n_{K-1}.
```

C₀ is the centre series' derivative at the first node, Φ_b the Taylor map's derivative over segment b, and H the
horizon series' derivative at u₀ = 1 − x_{K−1}, all for perturbations proportional to e^{λτ}. Eliminating the node
states (each −I block has determinant (−1)^{4M} = 1, and moving c past the node columns is an even permutation) gives

```math
\det M(\lambda) = \det E(\lambda), \qquad E = \big[\, \Phi_{K-1} \cdots \Phi_1 C_0 \;\big|\; H \,\big].
```

E is the Evans matrix at the last node. Its first 2M columns are the perturbations regular at the centre, carried out
to x_{K−1}, and its last 2M columns are those regular at the horizon. λ is a growth rate exactly when the two sets are
linearly dependent.

## 2. Where the sign is lost: the sonic valley, not the horizon block

**The valley.** The p equation, x(x − c) p_x = −x(ωD + λ) p + (G − c + nc/2) p + (n/2) c m, carries the perturbation's
rate λ with the coefficient 1/(c − x). Near D 3 the state's c − x has a narrow valley ([LOW_D.md](LOW_D.md) §17): on
the Mac, along the plan, min over τ of c − x is 7.26×10⁻⁵ at x 0.01444 (τ point 10) in the segment 0.01 → 0.02, which
takes 675 of the plan's 1269 steps. On the D 3.15 state (blob 9e683a14) the smallest c − x is 5.0×10⁻³, at x 0.0254.

**The growth across it.** For a p perturbation at the τ point j, the λ term alone gives a growth of exp(λ T_j) across
a segment, with T_j = ∫ dx / (c_j − x). `mode_matched.py valley` evaluates these integrals from the background (Simpson's
rule on the plan's steps):

| segment | x | steps | max over j of T_j | Σ_j T_j | Σ_j ∫ dx/(x + c_j) |
|---|---|---|---|---|---|
| 6 | 0.005 → 0.01 | 92 | 1.23 | 27.5 | 11.5 |
| **7** | **0.01 → 0.02** | **675** | **21.6** | **103.9** | **14.1** |
| 8 | 0.02 → 0.05 | 51 | 0.49 | 25.3 | 15.6 |
| 18–20 | 0.8 → 0.975 | 40, 40, 39 | 1.02–1.05 | 97–100 each | 1.2–5.3 |
| all others | | | at most 0.63 | | |

A fixed τ point overstates the growth, since the outgoing characteristics cross τ as they cross the valley. #1078's
smallest equilibrated pivot falls by about 13 in its logarithm per unit λ. Either way the valley's transfer Φ₇
stretches the p directions by e^{13λ} to e^{21λ} relative to the others: 10²⁹ or more at λ 5.14. Its double-double
entries carry absolute errors of about ε_dd times that stretch. Those errors swamp the part of Φ₇ that the sign of
det E depends on, the directions that do not grow. D 3.15 has no such valley, so its determinant is resolved.

**Why the small pivots sit in the horizon columns.** The LU eliminates the unknowns in the order c, n₀, …, n_{K−1},
h. Whatever the earlier pivots lose is left to the last block, the 2M horizon columns. These are the Schur complement
of everything else, and that is where #1078 found all 192 small pivots. The horizon series itself is well
conditioned (u₀/r 0.51, horizon LUs with κ + λ ≥ 5.3).

**So candidates (a) and (b), as posed on the horizon block, cannot help.** The quad-double LU of the double-double
entries reproduced the noisy determinant (comment 7120204793). The determinant of those entries is itself the noise.
Eliminating the horizon unknowns through the series, or any exact factorization (a Schur complement, a Riccati or
compound transfer computed from M's blocks), is exact algebra on the same Φ₇. It returns the same number. A remedy
must change which transfer the kernel integrates across the valley.

## 3. The matched determinant

**Layout.** The kernel's `match_node` = m integrates every segment past node m inward. Block row b > m becomes
Ψ_b n_b − n_{b−1}, with Ψ_b the Taylor map from node b back to node b − 1. With m the valley's inner node (node 6, x
0.01), the valley is crossed inward, the direction in which the p perturbations decay. The outward part keeps at
most max_j T_j = 1.23 per segment. That is candidate (b)'s "transfer from the horizon inward", in multiple-shooting
form, with no Riccati variable needed because the inward transfer does not grow. It needs no kernel change.
Call its determinant D_m(λ), and the present one D_out(λ).

**Identity.** If Ψ_b = Φ_b⁻¹ for every b > m, then

```math
D_{\rm out}(\lambda) = D_m(\lambda) \prod_{b > m} \det \Phi_b(\lambda).
```

*Proof.* Multiply block row b > m of M_in on the left by −Φ_b: −Φ_b (Ψ_b n_b − n_{b−1}) = Φ_b n_{b−1} − n_b, which is
row b of M_out. So diag(I, …, −Φ_{m+1}, …, −Φ_{K−1}, I) M_in = M_out. The factor's determinant is
∏ det(−Φ_b) = ∏ det Φ_b, since 4M is even. ∎

**The factor is positive.** Φ_b is the transfer matrix of the linear system n′ = A(x; λ) n over the segment.
Liouville's formula gives det Φ_b = exp(∫ tr A dx) > 0. The λ part of tr A is λ Σ_j [1/(c_j − x) − 1/(x + c_j)]: λ
enters only the p and m equations, through their τ derivative, and the folded d/dθ has a zero diagonal on the half
grid, mapped or not. So

```math
\ln \prod_{b>m} \det \Phi_b(\lambda) = \lambda\, I_m + \text{const}, \qquad
I_m = \sum_{b>m} \sum_j \int_b \Big[ \frac{1}{c_j - x} - \frac{1}{x + c_j} \Big] dx .
```

On the Mac, I_6 = 585.2 at D 3.044375, from the table's integrals. The discrete maps are not exact inverses. Ψ_b Φ_b =
I + O(the integration's error, about 10⁻²⁶ here: the stored state's inward residual at node 6 is 1.0×10⁻²⁶, against
1.8×10⁻²⁹ outward). So the identity holds to that order, and D_m and D_out have the same zeros and the same sign at
every real λ to that accuracy.

**Which quantity carries the root, and which factor is the near-null one.** D_m(λ) carries the roots. It is an Evans
function matched at x_m. The positive factor ∏ det Φ_b = e^{λ I_m + const} carries the exponential growth of
ln|D_out|: about 600 per unit λ, so ln|det| ≈ 3600 at λ 5.14. Inside it, the valley's ratio of growing to non-growing
directions, e^{−λT} with T ≈ 13 to 21, is the λ-dependent near-null direction that the outward LU meets. D_m never
forms it. That factor never vanishes, so dropping it changes no sign.

**Limits.** For λ below the valley's κ (the λ-independent part of the p rate, about 0.6 there), p perturbations decay
outward through the valley, and the inward transfer would grow. The outward layout is the right one there. γ's roots
(λ ≈ 4 to 5.2 for 3.04 < D < 3.15) lie far above. As the gap closes (D → 3.0431), T grows like the gap^(−1/2). The
outward determinant then loses more digits, about 13 per unit λ at D 3.044375. The matched one does not.

**An alternative, not recommended.** Splitting the valley segment by k extra nodes, all outward, leaves the
determinant unchanged (the chain rule for the same steps). It divides the stretch per transfer to e^{λT/k}, at 4M
unknowns per node, and k must grow as the gap closes. It would be an independent check of the diagnosis.

## 4. Pass criteria (fixed before the prototype's checks run)

Notation: D_out(λ) is the present determinant, `Solver.mode_det` on the state's own layout (every segment integrated
outward, `match_node` = K − 1). D_m(λ) is the matched determinant of section 3, the same kernel call on the layout whose
segments past node m are integrated inward (`match_node` = m), m the last node before the sonic valley. Both are
evaluated on the state's plan `Solver.plan(z, 1e-32, 0.05)`, as `hp/lowD_gamma.py` does.

1. **The four λ of comment 7120204793** at D 3.044375, M′ 96 (state blob 6a0f83d2): λ = 5.1408197887857,
   5.140819788786, 5.1408197887863025 and 5.1408197887866. With f_i = sign_i · exp(ln|D_m(λ_i)| − ln|D_m(λ_1)|),
   the four f_i lie on one straight line in λ to within 10⁻⁶ of max |f_i| (a straight line over 9×10⁻¹³ in λ is what a
   smooth function with at most one simple root gives there), and the double-double LU of D_m has its pivot ratio
   min |u_kk| / max |u_kk| above 10⁻²⁰. The present D_out fails both there (ln|det| moves by 4.0, 3.1 and 4.7 between
   neighbours, and the sign flips twice).
2. **D 3.15**, M′ 96 (state blob 9e683a14, its own layout and map): the root of D_m near λ 4.3977 equals the root of
   D_out, each found by `lowD_gamma.find_lam` with xtol 10⁻¹¹ from λ₀ 4.397684106690, to within 10⁻⁸ in λ, and both
   searches see the same number of sign changes in their first bracket.
3. **A planted root.** At D 3.044375 (state 6a0f83d2), on the matched layout, one horizon column j of the mode matrix
   is replaced by its value at λ minus its value at λ* (the column of a_h at the τ point of the valley, λ* = 5.0), so
   that the matrix has a zero column, and its determinant an exact root, at λ*. The prototype's root search on the
   planted determinant returns λ* to within 10⁻⁹. Controls, each of which must fail the same test: (i) the column
   replaced by its value at λ minus its value at λ* + 10⁻⁶ (the root moves by about 10⁻⁶); (ii) the same plant on the
   outward layout, where the signs at λ* − δ and λ* + δ, δ = 10⁻³, 10⁻⁴, 10⁻⁵ and 10⁻⁶, must not form one clean
   change (four equal signs on the left, four equal and opposite on the right).
4. **The identity of section 3.** Where both are resolved (D 3.15 at λ 4.3 and 4.5; D 3.044375 at λ 1.0 and 1.1), the
   signs of D_out and D_m agree, and the change of ln|D_out| − ln|D_m| between the two λ equals the Liouville trace
   integral of section 3 over the segments past m, computed from the background alone, to within 10⁻⁶ of that change.

The control of criterion 1 is the same command on the outward layout (`--match=out`), which must fail it.

## 5. The prototype and its runs

`hp/mode_matched.py` uses only the kernel's existing bindings: `Solver(..., match_node)`, `plan`, `residual`,
`integrate`, `mode_det` and `mode_matrix`. No kernel function is needed. It picks m from the background: the valley
segment is the one with the largest max_j T_j, and m is its inner node (node 6 at D 3.044375). Its subcommands are
listed in its docstring; the verdict subcommands exit 1 on FAIL.

**Commands for a Linux runner** (a checkout of this branch, hp/ as the working directory, the raw-data clone at `$R`,
the output folder `$O = $R/I1083-<machine>/` made first with its README):

```bash
S1=$R/I1072-dh370/states3/tmap_tmap_tmap_lowDhm_x0.000157-0.975_D3.0444_M384_L0.4_M160_s1.9635_L0.4_M128_s1.9635_L0.4_M96_s1.9635.npz
S2=$R/I1072-dh370/states2/tmap_tmap_lowDhm_x0.000625-0.9_D3.1500_M160_L0.7_M128_s1.8326_L0.7_M96_s1.8326.npz
python3 mode_matched.py toy                                                   # planted toy, seconds: PASS
python3 mode_matched.py toy --mutate=shift                                    # its control: must FAIL (exit 1)
python3 mode_matched.py valley $S1 --threads=6                                # m = 6 (x 0.01)
python3 mode_matched.py four $O/four.jsonl $S1 --threads=6                    # criterion 1: 4 dets
python3 mode_matched.py four $O/four_out.jsonl $S1 --threads=6 --match=out    # its control: must FAIL (exit 1)
python3 mode_matched.py d315 $O/d315.jsonl $S2 --threads=6                    # criterion 2
python3 mode_matched.py identity $O/identity_D3.15.jsonl $S2 4.3 4.5 --threads=6     # criterion 4
python3 mode_matched.py identity $O/identity_D3.0444.jsonl $S1 1.0 1.1 --threads=6   # criterion 4
python3 mode_matched.py planted $O/planted.jsonl $S1 --threads=6              # criterion 3
python3 mode_matched.py planted $O/planted_shift.jsonl $S1 --threads=6 --mutate=shift      # must FAIL (exit 1)
python3 mode_matched.py planted $O/planted_outward.jsonl $S1 --threads=6 --mutate=outward  # exit 0: the control fired
```

Then, if criteria 1 to 4 pass, γ at D 3.044375 from D_m: `python3 mode_matched.py root $O/root.jsonl $S1
--threads=6 --lam0=5.14082 --xtol=1e-11`, and the sign scan of #1078 again on D_m (`det` over 5.14060–5.14100).
