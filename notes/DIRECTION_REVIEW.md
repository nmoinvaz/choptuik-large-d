# DIRECTION_REVIEW: which lines can still produce a discovery, and what has not been tried

An independent review of the project's direction, asked for by Nathan Moinvaziri ("check to see we are headed in the
right direction, we want: discoveries. Also look for anything that hasn't been thought of before"). Written 2026-11-19
by Claude (Anthropic), from the repository, the issues and the last week's commits, without the coordinator's
reasoning. Not reviewed. Status words as in [AGENTS.md](../AGENTS.md). Every number is quoted from the file or issue
named beside it, or marked as this review's own arithmetic or probe. Nothing heavy was run. Two probes ran on the Mac
(the one-harmonic multiplier on a grid, about 10 minutes of CPU, and finite differences of stored Δ values), and their
scripts and outputs stayed in the session's scratchpad, so they are described here, not stored.

## Summary

- **The discoveries the project already has are unpublished, and the nearest competitor is closing in.** Ecker, Ecker
  and Grumiller (arXiv:2611.07007, 4 Nov 2026) now extrapolate their D ≤ 200 solutions to a finite Δ∞ = 1.643911
  ([DINF_ANALYTIC.md](DINF_ANALYTIC.md)), and they expect "instanton corrections of the form e^{−1/ε}"
  ([LARGE_D_WALL.md](LARGE_D_WALL.md)), which is the beyond-all-orders road to the selection rule. The finite limit is
  no longer ours alone. The selection rule, Propositions 1–5 and the certificates still are, and only publication
  protects them.
- **Most of the fleet's effort is maintenance.** Of the 450 issues closed since 2026-11-11, 185 are the D 400 chain
  and 110 are digits and error bars, 65% together. Papers and docs got 16. None of the chain's last week changed a
  physical conclusion.
- **Stop the chain, the speed work, the low-D legs and the RT diagnostics.** The chain past D 245 leaves the continuum
  (its own conclusion), the low-D legs are within 3×10⁻⁵ in D of a coordinate wall they cannot cross, and the RT
  diagnostics measure floor noise of a formulation that is not ours. This contradicts Nathan's "Keep going to 400" of
  2026-11-07, so the decision is his.
- **Four blind spots could each give a discovery.** In order of value per cost:
  1. **The ℓ = 2 non-spherical mode.** At D = 4 it is the least damped mode, Re λ ≈ −0.02 (Martín-García and Gundlach
     1998), so it is almost marginal. A short argument of this review (§2.1, not checked) says that at D = ∞ a harmonic
     deformation of r² gives exact modes λ = 2 − ℓ, so ℓ = 2 would be exactly marginal there, and the sign of its 1/n
     coefficient decides whether Choptuik's solution acquires a second, non-spherical unstable mode at large D. Either
     answer is new: nothing on non-spherical stability above D = 5 was found.
  2. **The ℓ = 1 multiplet's (and so the σ-model's) D = ∞ selection, computed directly.** The inner problem is
     derived and n-free ([matter/STRUCTURE.md](../matter/STRUCTURE.md)). Its M = 1 and M = 3 zeros and its growth
     rate would turn "γ∞ ≠ 1/2 for ℓ ≥ 1" from a conjecture into a prediction, cheaply, instead of pushing finite-D
     solves past D = 13.
  3. **A countable family of D = ∞ solutions.** The Saffman–Taylor template that PROOF.md adopts has a discrete
     infinity of selected solutions. Here S₁ is numerically entire of exponential type, so it generically has
     infinitely many zeros, and each one that becomes real at some ω is a candidate excited echoing solution. No note
     asks this. A ten-minute probe of the one-harmonic multiplier finds the known zero, no other zero where its grid
     resolves the phase, and two unresolved valleys at ω/ω∞ < 0.7 and A > 3.
  4. **The amplitude as a second sharp test of Conjecture 1.** The gauge rate's fitted c₁ = 7.4165644 ± 4.4×10⁻⁶ is
     2⟨F∞²⟩ by Ecker et al.'s Eq. (15), so the finite-D data fix the D = ∞ amplitude to about 10⁻⁶, against the
     README's "measured 2.70–2.72".
- **One unflagged tension.** The truncation sequence extrapolated with a constant ratio gives ω/ω∞ = 0.999985–0.999991
  ([P6_scope.md](P6_scope.md) §2.3), 1.1×10⁻⁵ below the amplitude expansion and the finite-D limit, at about twice the
  M = 7 noise. M = 9 decides it, and the amplitude expansion of hier's own truncated models may decide it sooner.

## 1. Direction, line by line

The verdict words: **discovery** (can produce a result a referee would call new), **maintenance** (keeps the machinery
or the record correct), **polishing** (more digits or error bars on a known number), **stuck** (blocked by a wall the
line cannot pass as designed).

| Line | Issues (open) | Verdict | Reason | What would change my mind |
|---|---|---|---|---|
| Chain in D toward 400 | #903, #949, #966, #971, #973, #984, #782, #834, #842, #800, #899, #900, #884, #738, #939, #938, #935, #936, and about 30 blocked Sprint 1–2 items | maintenance, stuck past D 245 | Its physics result is in hand. To D 243.6 its rows carry errors of 1–3×10⁻¹¹ (`hp/data/delta_all.csv`, kind chain_fit), and to D 215 they agree with the 1/n series within the series' spread, which is the test of analyticity the chain can give. Past D 245 it leaves the series, its errors grow to 1.5×10⁻⁵ at 254 and 0.44 at 259.86 as the weak pivot falls like e^{−1.44D}, and it folds at D* = 259.9, which its own notes assign to the discrete system ([WALLS.md](WALLS.md)). So each further unit of D costs exponentially more and measures the method. Track C already gives Δ(250–400) as series rows at about 10⁻¹⁰ (#801, #804). Ecker et al. have solutions to D = 200. | A stated physics question that only D > 250 can answer, with a pass criterion that can fail. None is written in the notes or issues. |
| Speed work | #954, #957, #961, #924; multidouble, multishoot, multisolver PRs | maintenance | It serves the chain's octo-double Newton steps. Without the chain it serves only bit-identity checks. | The chain continuing for a physics reason. |
| Digits and error bars | #444–#453, #467, #471, #498, #508, #569, #678, #723, #549 | polishing | The extra zeros at D 54–61 behave as truncation artefacts (README, "Extra zeros"; MODE_COUNT.md §5), and γ's root floors and λ_g biases change no conclusion. | An extra zero that survives M 64 at the same place. |
| Low D near D 3 | #932, #945, #931, #947 | stuck | The legs stall at D 3.04315, and the sonic gap min(c − x) closes at D 3.04312 ([LOW_D.md](LOW_D.md) §17). The D → 3 question needs points at D ≤ 3.02 (README, "The two readings"), which x-marching cannot reach. §2.6 argues that the closing is a coordinate wall, not an end of the family. | A converged point below 3.0431 from the current method. |
| Inner selection, M = 7 and 9 | #980, #989, #974 | discovery (one test), otherwise polishing | M = 9 decides between a truncation limit of 0.99999 and one of 1 (§2.5), which is a real test of the rule's internal consistency. The radius pairs (#989) test the numerics. | M = 9 landing on 1.00000 ± 5×10⁻⁶ would make the radius work pure polishing. |
| RT formulation at D 40 | #972, #985, #983 | stuck, and not ours | P7 needs ln n₀ near 5.5, and Theorem B(1) has 89.75, with no lever shown to close 84 e-folds ([P7_scope.md](P7_scope.md) §4). The bound ratios 0.98, 6.06 and 51.8 come from floor-level changes of one root ([RT_FORMULATION.md](RT_FORMULATION.md) §14), so they measure noise. The one finding worth keeping is that the RT system imposes neither the light-cone condition nor the constraint, which bears on Ecker et al.'s D = 200 numbers. | A lever that brings ln n₀ under about 10. |
| P6 routes | #990, #982, #986, #987 | discovery, low probability | A proof of uniformity in the harmonics would be a real theorem, but P6_scope.md says it "needs a new idea". Route 3's polynomial growth in ν is progress. | Route 3 closing the range j ≲ ν. |
| P5, the entire-function route | none open | discovery, unfinished | The fixed-ω certificates pass at K = 17–22, and the step across ω (H_ω) was measured on 2026-10-31 and not finished ([PROOF_QUEUE.md](PROOF_QUEUE.md) P5). A conditional Theorem A(∞) is closer here than through P6. | The type bound failing at the next order. |
| P2's unchecked items | #988 | maintenance, needed | Theorem B(1) needs its review before it is submitted. | — |
| σ-model and ℓ = 1 in D | #649, #651; [matter/SIGMA_DSS.md](../matter/SIGMA_DSS.md) §11 | polishing at D = 4, stuck at D 10–13 | Δ(η)'s minimum at η = 6.98 and γ(η) at D = 4 refine a published model. The push in D meets a near-null band and a drifting gauge root. The question it serves, γ∞ ≠ 1/2, is cheaper at D = ∞ (§2.2). | — |
| Twins and far Padé poles | #668, #690–#692, #704–#706, #719, #746–#748 | polishing | They sharpen the ring exclusion of the selection estimator. The ring-free sums already put the zero at 1.0000000019–1.0000000046 of ω∞ (README, order 22), inside the finite-D bar. | A twin class that moves the zero outside 0.9999988–1.0000004. |
| Papers | #663, #474, #475 | the highest-value maintenance | The summary's first point. | — |

## 2. Blind spots

Each has what it is, why it matters, the smallest first experiment with a criterion that can fail, the cost, and
whether it is new. "Not found" means not found in the searches listed in §4, which are not exhaustive.

### 2.1 The ℓ = 2 mode: near-marginal at D = 4, and argued to be exactly marginal at D = ∞

**What.** Martín-García and Gundlach (PRD 59, 064031 (1999), gr-qc/9809059, Table 1) give κΔ = −0.07 and ωΔ/2π = 0.3 for
the least damped non-spherical mode, polar ℓ = 2, that is λ ≈ −0.02 + 0.55i at D = 4. Every other non-spherical mode is
damped by κΔ ≤ −0.30. In D = 5, Porto Veronese and Gundlach (arXiv:2209.08404) find that weak gravitational-wave
perturbations of the scalar critical solution decay, in a cohomogeneity-two ansatz whose wave is a squashing of S³,
transverse and traceless on the sphere, so tensor-type as this review reads it (abstract only). The scalar-type ℓ = 2
sector above D = 4 has not been computed by anyone found.

**An argument for λ = 0 at D = ∞ (this review's; not checked).** In the inner region r ~ √n the field depends on x
through s = r²/(2n), and the flat Laplacian is ∂_s at leading order because (n/r)∂_r dominates ∂_r². For a field φ(S(x),
t), Δφ = φ″|∇S|² + φ′ΔS. If S = r²/(2n) + h(x)/(2n) with h a harmonic polynomial of degree ℓ, then ΔS = 1 + O(1/n) and
|∇S|² = O(1/n), so Δφ = φ_S at leading order, the same as for S = s. If the gravitational equations also depend on x
only through S at leading order, then the deformed solution is again a solution, and its derivative in h is a mode. That
is the step to check, and it can fail: the level sets of S are ellipsoids, whose curvature varies with angle at relative
order |C|, and the spheres' curvature n(n − 1)/r² enters the lapse equation multiplied by a² − 1 = O(1/n), so at order
one. Whether the variation cancels there is not settled by this argument. Since h scales as r^ℓ ∝ s^{ℓ/2}, the mode's
rate is λ = 2 − ℓ:

| ℓ | h | λ | status |
|---|---|---|---|
| 0 | a constant: the r² shift | 2 | Proposition 5, exact at D = ∞ |
| 1 | a·x: translation | 1 | exact at every D (a gauge mode) |
| 2 | x·Cx with C traceless | 0 | checked symbolically at leading order in 1/n, gravity included ([L2_MODE.md](L2_MODE.md), #993) |
| 3 | a cubic harmonic | −1 | this argument |

The ℓ = 0 and ℓ = 1 rows are known. The ℓ = 2 row was checked by experiment (a): the lapse and scalar equations
need no correction, and only the O(1/n) spatial metric along the normal adjusts (L2_MODE.md). Its sign at finite
D, c₂ in λ₂ = c₂/n, is experiment (b). If it holds, ℓ = 2 is a zero mode at
D = ∞, the analogue of the free shape of a large-D black-hole membrane, and λ₂(D) = c₂/n + O(1/n²). The sign of c₂
decides whether Choptuik's solution has a second unstable mode at large D. If c₂ > 0, it would cross zero at some
finite D_c, and above D_c critical collapse would have codimension two, and spherical tuning would no longer find the
threshold solution. If c₂ < 0, the solution is non-spherically stable at large D, with an ℓ = 2 mode that becomes
ever slower, which would also explain why it is so slow already at D = 4.

**First experiment.**
- (a) Symbolic, at leading order in 1/n: the Einstein–scalar equations in an O(p) × O(q)-invariant ansatz
  (p + q = n + 1, the standard cohomogeneity-two reduction), with h = q|y|² − p|z|², which is harmonic. Check that
  S = s + εh/(2n) solves the inner equations with the metric to first order in ε. **Control that must fail:**
  h = |y|² alone, which is not harmonic.
- (b) If (a) passes, compute c₂ by the method of [FIRST_ORDER.md](FIRST_ORDER.md), which gave λ₁ = 3.0324 for ℓ = 0
  to 0.3% at M = 5.
- (c) Independently, the scalar-type ℓ = 2 perturbation equations (Kodama–Ishibashi–Seto's gauge-invariant
  formalism for a matter background) on hp's stored states. **Pass:** λ ≈ −0.02 + 0.55i at D = 4 within Martín-García
  and Gundlach's stated noise, and the ℓ = 1 gauge mode at its known place. Then D = 5, 6, 8, 12, 20, 40. A cheaper
  start is the tensor-type sector, which decouples (its matter source vanishes for a scalar field) and must decay at
  D = 5 as in Porto Veronese and Gundlach.

**Cost.** (a) 1–2 agent-days. (b) 2–4 agent-days. (c) 1–2 weeks of agent time and modest CPU, since the problems are
linear on stored states.

**New?** Non-spherical perturbations of the scalar critical solution above D = 5, and any zero mode of it at large D:
not found (§4). The argument is new here.

### 2.2 The ℓ = 1 multiplet's selection and growth rate at D = ∞

**What.** [matter/STRUCTURE.md](../matter/STRUCTURE.md) derives the n-free inner problem ψ_t = cQ, Q_t = cψ_s,
c_s = 2s^ℓcQ², and shows that the s-translation, and with it λ = 2, is broken for ℓ ≥ 1. It predicts γ∞ ≠ 1/2. The
finite-D data to D = 9 cannot decide it: free fits put λ∞ anywhere in 0.55–3.95, and the heuristic bracket from λ₁ − λ₀
is 2.09–2.31 ([matter/ELL.md](../matter/ELL.md) §7). The σ-model at η = ∞ is this model, so its D = ∞ limit is the
same.

**Why.** For the axion–dilaton and the complex scalar, the D = ∞ Stokes zero was computed and then matched by finite D
to about 10⁻⁴ (README, "Other matter"). Doing the same for ℓ = 1 gives Δ∞(ℓ = 1) and λ∞(ℓ = 1) as predictions made
before the data, which the finite-D runs can then confirm or refute. A λ∞ of exactly 2 despite the broken symmetry would
point to a hidden one. A λ∞ ≠ 2 would be the first case where γ∞ ≠ 1/2 is derived, against the folklore that
γ → 1/2 is universal.

**First experiment.** Adapt `nested.py` (M = 1, then M = 3) to the ℓ-weighted gravity term. **Pass:** at ℓ = 0 the
adapted code returns the scalar's M = 1 zero, A = 2.54215 and ω/ω∞ = 0.77699 (`zero_M1.py`). Then the ℓ = 1 zero. Its
Δ must be finite and below the finite-D maximum 0.988 if the ℓ = 1 data's trend is right, so the comparison can fail.
Then λ∞(ℓ = 1) from the linearized multiplier, as GAMMA.md §5 describes.

**Cost.** 3–5 agent-days, laptop CPU. It replaces the σ-model's chains past D = 10 (#649, #651 can finish at D = 4).

**New?** No large-D study of critical collapse for matter other than the real scalar was found by matter/LITERATURE.md.
This review did not search further.

### 2.3 A countable family of selected solutions

**What.** PROOF.md takes Saffman–Taylor finger selection and dendrite selection as the template. In those problems the
solvability condition has a discrete infinity of solutions (finger widths λ_n), of which only the first is stable. Here
S₁ is numerically an entire function of A² of exponential type ([proof/D_entire.md](../proof/D_entire.md)), and such a
function generically has infinitely many zeros. Each zero x_j(ω) that becomes real and positive at some ω_j gives a
candidate D = ∞ echoing solution (A_j = √x_j, ω_j), likely with more than one unstable mode: an excited Choptuik
solution. The rings of zeros tested in PLANTED_SELECTION.md were artificial alternatives near x₀, not this question. No
note mentions excited branches, countable families or global uniqueness.

**Why.** If such solutions exist at D = ∞, they can be continued to finite D with hp, and an excited DSS solution of the
massless scalar would be new at any D. If none exists in a large box, that is evidence for the global uniqueness that
P7's link L5 needs.

**Probe (this review).** The one-harmonic multiplier R₁ ξ₁^{−A²} from `lateral_np.jump` (0.6 s per point) on a 28 × 28
grid, A = 0.6–6.0, ω/ω∞ = 0.3–3.0, counted by the winding around each cell. It recovers the known zero (winding +1 in
the cell A 2.4–2.6, ω/ω∞ 0.7–0.8). For ω/ω∞ ≥ 1.2 with A ≤ 4.6, and for A ≤ 2.4 everywhere, the phase moves by less than
1.5 rad between grid points and no other zero appears. For ω/ω∞ < 1 with A > 2.6, and for A > 4.6, the phase turns by up
to π between points, so the coarse grid can neither count nor exclude zeros there. A refined grid of that region,
A = 2.6–4.0 in steps of 0.05 and ω/ω∞ = 0.3–1.0 in steps of 0.025, has net winding 0 on its boundary. Inside it, two
valleys where |R₁| drops about tenfold below its neighbours run along ω/ω∞ ≈ 0.36 (A 3.1–3.85) and along a diagonal
from (A, ω/ω∞) = (3.0, 0.56) to (3.95, 0.69). Along both, cells of winding +1 and −1 alternate, with phase steps of
2–3 rad between points. That is either a chain of zeros of opposite orientation (R₁ is a map of two real variables, so
both signs occur) or a valley still under-resolved. The probe does not decide which.

**First experiment.** An adaptive argument-principle count of the M = 1 and M = 3 multipliers on A ≤ 6,
ω/ω∞ = 0.2–3, refining every cell edge until the phase step is below 0.5 rad. **Pass:** the known M = 1 and M = 3 zeros
are found, a planted factor (z − z₀) is counted, and any new zero at M = 1 reappears at M = 3 within the shift the
known zero shows (0.777 → 1.016). **Then:** seed hp at D = 40 from a surviving zero.

**Cost.** Hours of CPU, which a runner can absorb, and 1–2 agent-days.

**New?** Excited echoing solutions of the real massless scalar: not found (§4).

### 2.4 The amplitude as a second test of Conjecture 1

**What.** Ecker et al.'s Eq. (15) (arXiv:2611.07007, as read in [EEG_PROFILE.md](EEG_PROFILE.md) §1) says
(Δ₁ − 4πμ₁)/(4πμ₀) = ⟨β̃²⟩. With λ_g = 4πμ/Δ and β̃ = √2 F this is n(1 − λ_g) → 2⟨F∞²⟩. MODE_COUNT.md §10 fits
c₁ = 7.4165644 ± 4.4×10⁻⁶ over D 20–245. At the M = 7 zero ([proof/B_tail_M7.md](../proof/B_tail_M7.md), A = 2.719219
and the harmonic coefficients of its `--c7` line), A² + Σ(B_ν² + C_ν²) = 7.41664, 7×10⁻⁵ above the fit (this review's
arithmetic). With the M = 7 harmonics held, the identity gives A∞ = 2.719205. M = 7's own A is 2.71922–2.71924, and its
next truncation increment is about −5×10⁻⁵ (the ratio −0.082 per pair times A's increment of 6.4×10⁻⁴, both from
P6_scope.md §2.3), so the two agree within the truncation.

**Why.** The selection rule gives two real numbers, ω and A. ω is compared at 10⁻⁸. A is compared in the README at 1%
("2.720, against a measured 2.70–2.72"). The identity makes the amplitude a test at about 10⁻⁶, using only data the
repository has.

**First experiment.** Issue #638, with the identity taken from Ecker et al. (credited) and derived here from the outer
1/n expansion. Compare 7.4165644 with A₀²(1 + Σ|F_ν/F₁|²) at the amplitude series' zero at K = 22. **Pass:** agreement
within the series' estimator range for A₀. **Control that must fail:** ⟨F²⟩ in place of 2⟨F²⟩, and #638's n − 1 → n.

**Cost.** Hours.

**New?** The identity is Ecker et al.'s. Its use as a test of the selection rule is new here.

**Result (#638, [EEG_PROFILE.md](EEG_PROFILE.md) §3.5.2).** The frozen test fails narrowly: c₁ lies 6.2×10⁻⁶ below the
range of 2⟨F²⟩ at the K = 22 zero. Through the identity c₁ implies A = 2.71920533, which matches the states' own
A∞ = 2.71920535 to 1.9×10⁻⁸, so the gap lies in the selection's x₀ estimate (1.1×10⁻⁶ in A, under its imaginary
residue of 6×10⁻⁶). The two agree to 4×10⁻⁷ relative. Settling the rest needs σ₂₃ onward.

### 2.5 The truncation sequence against the amplitude expansion

**What.** The truncation zeros are 0.777, 1.0156, 0.999228 and 1.000022–1.000028 (rtol-converged,
P6_scope.md §2.3). The ratio per pair is κ₅ = −0.0686 and κ₇ = −0.0485. Held constant from M = 7, it gives a limit of
0.999985–0.999991. The amplitude expansion gives 0.9999988–1.0000004 (estimator range) and its ring-free sums
1.0000000019–1.0000000046 (README, order 22), and the finite-D limit is 1 ± 10⁻⁸. So the constant-ratio limit is
1.1×10⁻⁵ low, at about twice the M = 7 noise of 5×10⁻⁶. The notes quote the limit as "0.999985 ± 2×10⁻⁵", a bar that the
rtol-converged runs have since shrunk, and do not report the tension.

**Why.** If the ratio stays near −0.049, the truncated models converge to a frequency that is not the full problem's,
and Lemma T's (T3) and the truncation route to Theorem A(∞) need rethinking (the even-harmonic cut is one known way the
limit can change, [proof/B_tail_M7.md](../proof/B_tail_M7.md) §4). If |κ| keeps shrinking, as from κ₅ to κ₇, the limit
is 1 and the rule is consistent across two methods to 10⁻⁶.

**First experiment.** Two predictions for M = 9, written down before the run: 0.99998–0.99999 for a constant ratio,
and about 0.9999995 for κ₉ ≈ −0.032. They are 1.2×10⁻⁵ apart. M = 9 is blocked at level 5 (#974). A cheaper route:
`amp_trunc_hiermodel.py` runs the amplitude expansion of hier's model 𝔐_M itself, and its coefficients differ from the
full ones only from σ_{M+1} on (Δσ₆ = 7.9×10⁻⁹ at M = 5). The shift p_M − p_∞ then follows from the differences
Δσ_k^{[M]}, with no resummation of the common part. **Pass:** the shifts reproduce p₃ − 1, p₅ − 1 and p₇ − 1 within the
hier.py values' noise. Then they predict p₉ without hier.py.

**Cost.** Amplitude runs of 𝔐₇ and 𝔐₉ to K of about 17–19: about a day on one runner. The analysis: 1 agent-day.

**New?** Internal consistency, new here only.

### 2.6 Low D: the sonic closing is a coordinate wall, and D → 3 needs another route

**What the data say.** Δ shows no singularity where the gap closes (this review's finite differences of the
double-double values in `hp/data/delta_all.csv`):

| D (midpoint) | Δ′ | (D − 3)Δ′/Δ |
|---|---|---|
| 3.04345 | 8.1139 | 0.14484 |
| 3.04406 | 8.0464 | 0.14537 |
| 3.04875 | 7.5660 | 0.14900 |
| 3.06563 | 6.2283 | 0.15776 |
| 3.10625 | 4.3069 | 0.16342 |

Δ″ runs smoothly from −33 at D 3.11 to −111 at 3.0436. A square-root term b√(D − D*) at D* = 3.04312 would raise Δ′ by a
factor 4.3 between the first and third rows, against the 1.07 seen, so b ≲ 5×10⁻⁴, against 0.035 for the large-D fold.

**Why it is likely a coordinate wall.** x = c is where the curves of constant x become null. In a discretely
self-similar spacetime there is no homothetic vector field, and ∂_τ at fixed x is a field of the solver's coordinates
(areal radius, and the time fixed by c = 1 at the light cone). So the closing is a property of those coordinates, like
the walls of WALLS.md, and not by itself an end of the family. A formulation that does not march in x, or another time
coordinate, should pass it. Ecker et al.'s plotting notebook fits values on 3.01–3.05 ([LOW_D.md](LOW_D.md) §6), in
their own gauge. Whether those are converged solutions below 3.043 is not recorded.

**First experiments.**
- (a) The Taylor series of Δ in D at D₀ ≈ 3.10–3.15, with the jets that exist at low D
  ([LOW_D_ANM.md](LOW_D_ANM.md)), to order 15–25 at an M of 160–256. Dlog-Padé and a ratio test for the nearest
  singularity, with `hp/singularity_type.py planted` first. Pre-registered readings: a singularity at D = 3.000 ± 0.003
  with exponent 0.50 ± 0.05 is the finite-Δ(3) reading. Exponent 0.15 ± 0.05 with no constant term is Ecker et al.'s
  power law. A nearest singularity at 3.043 would mean the family does end at D*.
- (b) The repository's RT code (`solver/rt/`) at D = 3.04, seeded from hp's state at 3.045. **Pass:** it converges below
  3.0431 and matches hp at 3.045 to 10⁻⁸.
- (c) Longer: the D → 3 limit as a singular perturbation, as the inner problem is for D → ∞. |F₁| grows like
  (D − 3)^{−1.09} and gravity couples only through n − 1 ([LOW_D.md](LOW_D.md) §6), so a rescaled limit problem
  should exist. Its selection condition would give Δ(3), or the law by which Δ → 0.

**Cost.** (a) 1–3 days on one runner. (b) 1 agent-day. (c) open-ended, days to start.

**New?** Taylor coefficients of Δ in D, and a D → 3 limit problem: not checked in the literature, new as far as the
notes show.

**First reading (#997, `hp/series_singularity.py`), corrected.** At D 3.15, M 256 and M 160 agree only through order 6,
and a planted singularity at D 3.000 reads 2.35 from orders 3–6 there, so that base gives no reading. The earlier
D 2.998 used orders 7–9, which are not converged in M. At D 3.0475 (M 256, one M) the ratio fit over orders 3–6 puts
the nearest singularity at D 2.994, where plants at 3.000 and at the sonic closing 3.0431 are both recovered. It is
not yet numerical evidence: a second M at D 3.0475 has to agree first. The global fits of
notes/QUADRATIC_FORMULA.md put a branch point at 2.990–3.003.

### 2.7 Smaller items

- **The 1199 double-double states already hold the τ-periodic profiles.** The ℓ = 2 perturbations (§2.1), a test field
  with angular momentum and the gauge-rate identity (§2.4) all run on them, with no new continuation.
- **The README's D → 3 table** treats the readings as finite Δ(3) against Δ → 0. If (a) of §2.6 finds the nearest
  singularity elsewhere, the table needs a third row.

## 3. Ranking

| # | Proposal | Expected value | Cost | Make room by stopping or slowing |
|---|---|---|---|---|
| 1 | Submit the papers (#663 first) | protects every result above | days of Nathan's time, one agent | the chain's issue stream |
| 2 | Amplitude test of Conjecture 1 (§2.4, #638) | a second number of the rule tested at about 10⁻⁶ | hours | — |
| 3 | ℓ = 2 zero mode, symbolic check (§2.1 (a)) | a possible non-spherical instability at large D, new either way | 1–2 agent-days | RT at D 40 (#972, #985, #983) |
| 4 | Δ's Taylor series in D at D 3.1, and RT below D* (§2.6) | decides the D → 3 question, or finds that the family ends | 2–4 days | low-D legs (#932, #945, #931, #947) |
| 5 | ℓ = 1 selection at D = ∞ (§2.2) | the first derived γ∞ ≠ 1/2, or a hidden symmetry | 3–5 agent-days | σ-model chains past D 10 (#649, #651 finish at D = 4) |
| 6 | Excited branches, adaptive count (§2.3) | excited echoing solutions, or global uniqueness for P7 | hours of CPU, 1–2 agent-days | mode-count extras (#444–#453, #471, #498, #569, #723) |
| 7 | 𝔐₇ and 𝔐₉ amplitude series, then M = 9 (§2.5) | resolves an unreported 10⁻⁵ tension | about 2 days | radius pairs (#989); twins and far Padé (#668, #690–#692, #704–#706, #719, #746–#748) |
| 8 | ℓ = 2 spectrum across D (§2.1 (c)), if 3 passes | the answer to 3 at finite D | 1–2 weeks | the chain (#903, #949, #966, #971, #973, #984) and the speed work (#954, #957, #961, #924) |
| 9 | P5's step across ω | a conditional Theorem A(∞) | a few agent-days | P6 to one agent at low intensity |

The CPU-heavy items, 4(a), 6, 7 and 8, can take the runners that the chain frees.

## 4. Searches made for the novelty statements

All on 2026-11-19, with WebSearch, read as data.

| Query | What it found |
|---|---|
| nonspherical perturbations Choptuik critical solution higher dimensions stability l=2 mode | D = 4 only: Martín-García–Gundlach, Choptuik et al. 2003, Baumgarte 2018, a 2024 null-coordinate study |
| "all nonspherical perturbations of the Choptuik spacetime decay" least damped mode | gr-qc/9809059; its Table 1 was read on ar5iv for the ℓ = 2 values above |
| critical collapse scalar field higher dimensions non-spherical perturbations Kodama-Ishibashi self-similar | Porto Veronese–Gundlach, arXiv:2209.08404 (D = 5, gravitational-wave sector decays); nothing on the scalar-type sector or D > 5 |
| excited discretely self-similar solutions massless scalar more than one unstable mode | nothing for the real scalar; the complex scalar's extra modes (Hirschmann–Eardley) are a different solution class |
| large D limit critical gravitational collapse non-spherical deformation zero mode emergent symmetry | the large-D critical-collapse papers already in LIT_REVIEW.md; none treats non-spherical modes |

The searches are not exhaustive. Per AGENTS.md, nothing here may enter the README or the papers as "new" until the
closest work is read and recorded in [LIT_REVIEW.md](LIT_REVIEW.md).
