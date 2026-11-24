# The mode determinant near D 3: where its sign is lost, and a matched determinant (#1083)

Status: design done; prototype `hp/mode_matched.py` on branch `wip/mode-horizon-1083`, checked on the Mac (its
hpkernel build of 2026-11-08). The criteria on main's build are a Linux runner's (section 5). This note starts from
#1078's measurements (dh370). The status words are AGENTS.md's.

## Verdicts

| question | answer | status |
|---|---|---|
| where the sign of det M(λ) is lost at D 3.044375 | in the transfer across the sonic valley, segment 0.01 → 0.02 (min c − x 7.26×10⁻⁵), which stretches p perturbations by e^{13λ} to e^{21λ}; the horizon columns only receive the deficit, being eliminated last | numerical evidence (section 2) |
| can a reformulation of the horizon block fix it | no: any exact algebra on M's entries returns the determinant of those entries, which #1078 showed to be the noise | follows from comment 7120204793's quad-double LU |
| the remedy | the matched determinant D_m: the kernel's `match_node` at the valley's inner node, every segment past it integrated inward; no kernel change | design (section 3) |
| D_out = D_m ∏_{b>m} det Φ_b, with det Φ_b = exp(∫ tr A) > 0 | proved for exact inverse transfers; holds for the kernel's discrete blocks to about 10⁻⁶ in ln\|det\| at D 3.15 | proved (section 3); numerical evidence (section 6) |
| criterion 1, the four λ | PASS on the Mac build: all four −1, ln\|D_m\| = 605.35172667508 to …52266, on a line to 4.4×10⁻¹³ (bound 10⁻⁶), pivot ratio 5.2×10⁻⁸ (bound 10⁻²⁰) | numerical evidence; main's build to confirm |
| criterion 2, D 3.15 | D_m's root 4.397786345516482 on the Mac against #1072's D_out root 4.397786345516483 on dh370: 1×10⁻¹⁵ apart (bound 10⁻⁸); the D_out half on one build not run | numerical evidence; the full check is a Linux run |
| criterion 3, planted | the kernel plant is a Linux run (about 16 mode matrices); the toy plant passes, with its control failing | numerical evidence (toy only) |
| criterion 4, the identity against Liouville | FAIL as written at D 3.15 (4.9×10⁻³ against 10⁻⁶); the cause is the hmax 0.05 plan's outer segments (x ≥ 0.5), whose discrete blocks depart from exp(∫ tr A) until hmax 0.001 | numerical evidence (section 6) |
| is the mode matrix converged on the background's plan | no, in the outer segments: ln\|D_out(4.3)\| at D 3.15 is 1974.537, 1962.459 and 1958.389 at hmax 0.05, 0.005 and 0.001; whether the root moves is not measured | numerical evidence (section 6); bears on #1078 |

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

**The growth across it.** For a p perturbation at the τ point j, the λ term alone gives a growth of exp(λ T_j) across a
segment, with T_j = ∫ dx / (c_j − x). `mode_matched.py valley` evaluates these integrals from the background (Simpson's
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

## 6. Results on the Mac (hpkernel build of 2026-11-08)

These ran on the shared Mac at 2 threads, one M′ 96 mode matrix at D 3.044375 taking about 15 minutes. The numbers
below are from that build, not from main's (multishoot 4.1.0); the criteria themselves are for a Linux runner on
main's build (section 5).

**The matched determinant is smooth where the outward one is noise.** D 3.044375, state 6a0f83d2, m = 6
(`mode_matched.py det --match=6`, raw records in section 7):

| λ | sign | ln\|D_m\| | pivots min..max (double-double LU) |
|---|---|---|---|
| 1.0 | −1 | −64.123250972 | 1.17×10⁻⁴..5.71 |
| 3.0 | −1 | 260.126141749 | 1.18×10⁻⁴..20.2 |
| 5.1408197887857 | −1 | 605.3517266750824 | 1.19×10⁻⁴..2.27×10³ |
| 5.140819788786 | −1 | 605.3517266751303 | 1.19×10⁻⁴..2.27×10³ |
| 5.1408197887863025 | −1 | 605.3517266751795 | 1.19×10⁻⁴..2.27×10³ |
| 5.1408197887866 | −1 | 605.3517266752266 | 1.19×10⁻⁴..2.27×10³ |

Across the four λ (9×10⁻¹³), ln|D_m| rises by 1.44×10⁻¹⁰, with slopes 159.9, 162.5 and 158.2 between neighbours (the
last digits of ln|D_m| near 605 are 1.1×10⁻¹³ apart). The secant from λ 1 to 3 is 162.1. There the present determinant,
in quad-double from the same double-double entries, read (+, 3600.328311), (+, 3604.297374), (−, 3601.178104) and (+,
3596.436366). Criterion 1's line test gives a residual of 4.4×10⁻¹³ of max |f_i| (bound 10⁻⁶), with pivot ratio 5.2×10⁻⁸
(bound 10⁻²⁰): **PASS on the Mac build**. So no root of D_m lies among the four λ: #1072's root there was a sign flip of
noise.

**Criterion 2, half of it.** At D 3.15 (state 9e683a14, hmax 0.05 plan), the valley rule picks m = 5 (x 0.02,
the segment 0.02 → 0.05 with max_j T_j 2.9). `mode_matched.py root --lam0=4.397684106690` finds D_m's root at
λ = 4.397786345516482 in 10 determinants, with one sign change in its first bracket. #1072 found D_out's root on
dh370 on the same plan at 4.397786345516483 (raw `I1072-dh370/gamma_L07_M96.log`), also with one sign change. The
roots differ by 1×10⁻¹⁵, against the bound 10⁻⁸. D_m's pivots stay at 3.85×10⁻⁴..51 (1.68×10⁻⁶ at the λ nearest the
root), against D_out's 3.85×10⁻⁴..1.55×10³. The roots come from two builds, so the full criterion is the Linux
`d315` run.

**The planted toy** (`mode_matched.py toy`, no kernel, float64): four fields, a valley that stretches field 0 by
e^{(λ − 0.6) T}, T = 10.1, and a root planted at λ* = 5 along the centre solutions' non-growing combination,
computed at 60 digits. The matched root is 5 − 2.7×10⁻¹⁵; the outward root is off by 4.7×10⁻³. The control
`--mutate=shift` (the plant at λ* + 10⁻⁶) moves the matched root by 1.0×10⁻⁶ and fails, as it must. Without the
valley (`--gap=0.05`, T = 1.35) both roots are within 1.1×10⁻¹⁴ of λ*, so the outward failure is the valley's.
The identity of section 3 holds there to 6×10⁻¹⁵ in ln|det|.

**Criterion 4 fails at D 3.15 as written, and the cause is the plan, not the identity.** On the state's plan
(hmax 0.05, 292 steps), at λ 4.3 and 4.5 the signs of D_out and D_m agree, −1 and +1 (the root 4.3978 lies between).
But ln|D_out| − ln|D_m| changes by −78.807976 while the Liouville integral over the inward segments gives −78.422082,
a relative difference of 4.9×10⁻³ against the bound 10⁻⁶. To find the cause, I compared each outward block Φ_b
(float64 log-determinant of the mode matrix's block) with that segment's trace integral, on three plans:

| segments | hmax 0.05 (292 steps) | hmax 0.005 (346 steps) | hmax 0.001 (1012 steps) |
|---|---|---|---|
| 1–12 (x ≤ 0.5) | ≤ 1×10⁻⁶ (printed to 6 decimals) | ≤ 2×10⁻⁹ | ≤ 2.2×10⁻¹⁰ |
| 13 (0.5 → 0.6) | −3.3×10⁻³ | 3.5×10⁻¹² | −1.2×10⁻¹⁴ |
| 14 (0.6 → 0.7) | −0.104 | 1.0×10⁻¹¹ | −3.5×10⁻¹³ |
| 15 (0.7 → 0.8) | −0.113 | 4.3×10⁻⁷ | −4.1×10⁻¹³ |
| 16 (0.8 → 0.9) | −0.166 | −0.0999 | 2.6×10⁻¹² |

The entries are the change of ln det Φ_b from λ 4.5 to 4.3 minus the integral's. On the coarse plan, the four outer
segments' changes sum to −0.385895. That is the criterion's whole discrepancy (−78.807976 + 78.422082 = −0.385894).
So D_out = D_m ∏ det Φ_b holds for the discrete maps to about 10⁻⁶ in ln|det|, while the discrete outer transfers
depart from the ODE's determinant until hmax 0.001. Segment 16's largest singular value at λ 4.3 is 1.13×10³ at
hmax 0.005 and 1.62×10² at 0.001.

The plan is built from the background's Taylor coefficients, and these steps under-resolve the perturbations there.
ln|D_out(4.3)| is 1974.537 at hmax 0.05, 1962.459 at 0.005 and 1958.389 at 0.001. **This bears on #1078's question
"is it the step plan?".** The mode determinant at hmax 0.05 is not converged in the outer segments, at D 3.15 too.
Whether its root moves at the 10⁻⁸ level is not yet measured. Criterion 4 is to be run again on a plan with
hmax 0.001, and criterion 2 there too if it fails at hmax 0.05. These reruns are diagnostics; the criteria as written
stay failed or passed as they ran.

## 7. Raw records

The Mac's runs are in the raw-data repository's folder `I1083-mac/` (its README lists each file with the command that
wrote it). The exploration scripts there (`explore*.py`) import `hp/mode_matched.py`. Their per-segment tables are
also produced by `mode_matched.py blocks`, whose records are in the same folder.

## 8. For the coordinator

**Recommended design:** the matched determinant D_m, with `match_node` at the sonic valley's inner node. It is
candidate (b), a transfer from the horizon inward, in the kernel's own multiple-shooting form. It needs no kernel
change and no extra precision. It never forms the valley's stretch, which grows like gap^(−1/2) as D → 3.0431, while
quad-double entries (#1082) only move the outward determinant's limit from λT ≈ 74 to about 147. Candidate (a), the
analytic elimination of the horizon block, does not help (section 2).

Decisions:

1. **Run section 5's commands on main's build** (Linux), criteria 1 to 3 and their controls. On dh370 a determinant
   at D 3.044375, M′ 96 is about 8 minutes, so `four` and its control take about 35 minutes each, `d315` about an
   hour, and each `planted` run 2–3 hours.
2. **The plan.** The mode matrix on the background's plan is not converged in the outer segments (section 6). The
   check is `root --both --hmax=0.001` at D 3.15 against hmax 0.05: whether γ's root moves by more than 10⁻⁸. If it
   does, mode determinants need hmax ≤ 0.001 (or a plan made from the perturbations' series), at a cost of about 3.5
   times the steps at D 3.15. Criterion 4 can then be rerun on that plan (`identity ... --hmax=0.001`).
3. **γ at D 3.044375.** After 1 and 2, `lowD_gamma.py` takes the matched layout (the diff below), and #1078's scan
   over 5.14060–5.14100 is repeated on D_m. D_m is −1 and smooth at all four λ of comment 7120204793, so its root
   is elsewhere.
4. **#1082** (the quad-double mode matrix) is not needed for this sign if criteria 1 to 3 pass on main's build.
5. **The premise of #1083's title** is corrected here: the horizon block is not formed by cancellation. The horizon
   columns receive the valley's deficit because they are eliminated last.

The diff for `hp/lowD_gamma.py` (not applied: this branch writes only the note and the prototype):

```diff
-def build_solver(D, M, meta, threads, mutate=None):
+def build_solver(D, M, meta, threads, mutate=None, match_node=None):
@@
-        return make_solver(D, M, **solver_kw(meta, threads)), None
+        return make_solver(D, M, match_node=match_node, **solver_kw(meta, threads)), None
@@
-    return tau_map.make_solver_mapped(D, M, s0, L, mutate=mutate, **solver_kw(meta, threads)), [s0, L]
+    return tau_map.make_solver_mapped(D, M, s0, L, mutate=mutate, match_node=match_node,
+                                      **solver_kw(meta, threads)), [s0, L]
@@ def main():
         s.set_segs(s.plan(zh, zl, 1e-32, hmax))
+        if opts.get("match") == "valley":            # #1083: segments past the sonic valley integrated inward
+            import mode_matched
+            m, _ = mode_matched.valley_node(mode_matched.trace_integrals(s, zh, zl, meta, s.get_segs()))
+            s, grid = build_solver(D, M, meta, threads, mutate=opts.get("mutate_map"), match_node=m)
+            s.set_segs(s.plan(zh, zl, 1e-32, hmax))
         if "polish" in opts:
```

## References

- R. M. M. Mattheij, "Decoupling and stability of algorithms for boundary value problems", SIAM Rev. 27 (1985)
  1–44, doi:10.1137/1027001: multiple shooting and the other stable methods owe their stability to decoupling the
  growing and decaying modes along the solution's dichotomy. Section 3's choice of the matching point is that rule.
  Checked against SIAM's and TU Eindhoven's records.
- J. Humpherys and K. Zumbrun, "An efficient shooting algorithm for Evans function calculations in large systems",
  Physica D (2006), doi:10.1016/j.physd.2006.07.003, arXiv:math/0508020: Evans functions of large systems by
  shooting, with a polar-coordinate form of continuous orthogonalization that keeps the function analytic in its
  parameter (from the arXiv abstract; the paper itself not read here). Checked against arXiv (title, authors, related
  DOI); the volume and pages are not checked. The term "Evans function" is from J. W. Evans's
  nerve-axon papers (Indiana Univ. Math. J., 1972–1975) *(unverified)*.
- Liouville's formula for the determinant of a fundamental matrix, det Φ = exp(∫ tr A): a textbook result.
