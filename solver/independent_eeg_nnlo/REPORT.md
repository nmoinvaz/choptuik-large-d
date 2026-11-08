# Independent check of the NNLO series of arXiv:2611.07007v1 (issue #142)

The claim is confirmed. The printed Π₃ of (S46) violates the paper's own (S39) at order ε^{5/2}, and nothing at a lower order fails. The printed Ω₃ (S45) and f₃ (S44, S48–S51) satisfy (S36) at ε² and (S37) at ε³ *with the printed Π₃*, so they carry its error through 2β̃Π₃ and through (S37) with f(τ,1) = 1. The claimed Π₃, Ω₃ and f₃ pass every order they fix. They equal, term by term, this check's own order-by-order solution of (S36)–(S39), which is unique once the paper's conventions (S29) and (S34) are imposed. Status: *checked symbolically* (sympy 1.14.0), with a numerical cross-check of the ε-scaling (mpmath 1.3.0).

The work used the paper alone. No file of this repository about this expansion was read.

## Source and versions

- arXiv:2611.07007, C. Ecker, F. Ecker, D. Grumiller, *Critical collapse remains discretely self-similar at large D*. The abstract page lists v1 only (Wed 4 Nov 2026 19:00:01 UTC), "6+8pp, 4+1figs", no journal reference, DOI 10.48550/arXiv.2611.07007 (pending registration). `arxiv.org/abs/2611.07007v2` returns 404. Web searches found no journal version and no erratum (2026-11-08). Search engines do not index the paper yet.
- The section is on PDF page 11. The PDF's text layer is garbled, so the page was rendered to images locally (PDFKit) and read from them. The HTML version was read through WebFetch, equation by equation in LaTeX. The two renderings agree on every coefficient, sign, exponent, prime and tilde of (S35)–(S51).

## Transcription and its ambiguities

1. **β against β̃.** (S35)–(S42) print β̃. (S43)–(S51), including the factor 2βΠ₃ of (S45), print a plain β in both renderings, after the section defines β = √ε β̃ (main-text Eq. (10)). Two readings were tested.
   - Reading A, plain β means β̃. The NLO, (S43) included, then satisfies (S36)–(S39) at every order it fixes.
   - Reading B, plain β means √ε β̃ literally. (S43) then fails (S38) at ε^{3/2} already, and the NNLO fails (S36) at ε², (S37) at ε³ and (S39) at ε^{5/2}. Each coefficient would also mix orders of ε.

   Reading A is the only one under which the printed NLO holds. No reading found rescues the printed NNLO.
2. **(S35b)** √ε^{-1} is read as ε^{-1/2}, so Π = √ε β̃ + Σ Π_n ε^{n−1/2}, consistent with main-text Eq. (14).
3. **Primes** are τ-derivatives, as stated before (S35).
4. **Integration functions.** (S29) imposes f(τ,1) = 1 at the self-similar horizon. (S34) defines β̃ = Π(τ,0)/√ε exactly, so Π_n(τ,0) = 0 for n ≥ 2. Both are in the transformation section, not the perturbation section. The printed f₁, f₂, f₃ vanish at x = 1 and the printed Π₂, Π₃ vanish at x = 0, so the printed series follows both. The section states no range for x. The check takes the center x = 0 to the horizon x = 1.
5. **(S37)** is used as printed, (1/x)∂ₓ ln f = Ω, with ln f expanded to ε⁴.
6. **(S45)** is a formula in Π₃. The claim's Ω₃ is (S45) with the claimed Π₃.
7. **Ψ₃** is "not needed" (S47) and is set to 0, so (S38) is testable only through ε^{3/2}. No NNLO term needs it.

## Verdicts

Each residual is LHS − RHS expanded in √ε. The orders the given terms fix are (S36) ε⁰–ε², (S37) ε¹–ε³, (S38) ε^{1/2}–ε^{3/2} and (S39) ε^{1/2}–ε^{5/2}. The next order needs a term not given (Ω₄, f₄, Ψ₃, Π₄). Every such open order is nonzero in every set, so the residuals are not trivially zero.

| set | (S36) | (S37) | (S38) | (S39) | `check.py` exit |
|---|---|---|---|---|---|
| printed NLO (S35)–(S43) | pass to ε¹ | pass to ε² | pass to ε^{3/2} | pass to ε^{3/2} | `--nlo` 0 |
| printed NNLO, reading A | pass to ε² | pass to ε³ | pass | **fails at ε^{5/2}** | `--set=printed` 1 |
| printed, reading B | fails at ε² | fails at ε³ | **fails at ε^{3/2}** | fails at ε^{5/2} | `--set=printedB` 1 |
| claimed NNLO | pass to ε² | pass to ε³ | pass | pass to ε^{5/2} | `--set=claimed` 0 |
| own derivation | pass to ε² | pass to ε³ | pass | pass to ε^{5/2} | `--set=derived` 0 |

Summary mode (`check.py`, no flags) exits 0, meaning the NLO passes, the claimed set passes, derived equals claimed, and the printed set fails first and only at (S39) ε^{5/2}.

| term | derived = claimed | derived = printed |
|---|---|---|
| Π₂, Ψ₂, Ω₂, f₂ | yes | yes |
| Π₃ | yes | no |
| Ω₃ | yes | no, differs by 2β̃ ΔΠ₃ exactly |
| f₃ | yes | no, c₂ agrees, c₀, c₄, c₆ differ |

The printed minus the derived Π₃ is

```math
\Delta\Pi_3=x^2\left(2\tilde\beta^3+\tfrac12\tilde\beta^2\tilde\beta'-\tfrac12\tilde\beta\tilde\beta'^2\right)+\frac{x^4}{24}\left(3\tilde\beta^5+8\tilde\beta^3+40\tilde\beta^2\tilde\beta'+8\tilde\beta^2\tilde\beta''+24\tilde\beta\tilde\beta'^2-24\tilde\beta-62\tilde\beta'-57\tilde\beta''-22\tilde\beta'''-3\tilde\beta''''\right)
```

One term shows the error without long algebra. (S38) at ε^{3/2} puts x²β̃‴/2 into Ψ₂, as (S43) prints. (S39) at ε^{5/2} contains ∂_τΨ₂, so Π₃ must carry x⁴β̃⁗/8, and nothing else at that order contains β̃⁗. The printed Π₃ stops at β̃‴. A hand derivation of Π₃ from (S39) at ε^{5/2}, done before the script ran, gives the claimed Π₃.

**Uniqueness.** At each NNLO order the residual is affine in its unknown. The script measures the operator rather than assuming it: −(1/x)∂ₓ on Π₃ in (S39), the identity on Ω₃ in (S36), (1/x)∂ₓ on f₃ in (S37). So Ω₃ is unique, and Π₃ and f₃ are unique up to a function of τ, which (S34) and (S29) set to zero. Given the printed NLO, the claimed terms are the only fix within the paper's conventions. Dropping (S34) adds only a free C(τ) to Π₃, which cannot repair the printed x² and x⁴ terms.

## Numerical cross-check

`check.py --numeric` evaluates the (S39) residual for β̃ = 0.7 sin τ + 0.4 cos 2τ + 0.2 at τ = 0.3, x = 0.8, with mpmath at 60 digits and numerical derivatives. It shares the transcription and sympy's lambdify with the symbolic path, not the series extraction. It passes when the tested set's last local exponent exceeds 3.4.

| ε | printed | claimed | derived |
|---|---|---|---|
| 1e-3 | 1.112e-7 | 3.269e-13 | 3.269e-13 |
| 1e-4 | 3.517e-10 | 1.06e-16 | 1.06e-16 |
| 1e-5 | 1.112e-12 | 3.359e-20 | 3.359e-20 |
| 1e-6 | 3.517e-15 | 1.062e-23 | 1.062e-23 |
| local exponent | 2.500 | 3.500 | 3.500 |

## Controls

`controls.py --full` runs each row and compares the exit code and the set of failing (order, equation) pairs with an expectation written before the run. All rows behaved as expected.

| control (`check.py` flags) | what it perturbs | expected and observed exit, failing orders |
|---|---|---|
| `--nlo` | nothing | 0, none |
| `--nlo --mutate=S40` | f₂: 4(1−x²) → 5(1−x²) | 1, (S37) ε² |
| `--nlo --mutate=S41` | Ω₂: 3β̃′ → 4β̃′ | 1, (S36) ε¹, (S37) ε² |
| `--nlo --mutate=S42` | Π₂: 2β̃ → 3β̃ | 1, (S36) ε¹, (S38) ε^{3/2}, (S39) ε^{3/2} |
| `--nlo --mutate=S43` | Ψ₂: 11β′ → 12β′ | 1, (S38) ε^{3/2} |
| `--set=claimed` | nothing | 0, none |
| `--set=claimed --mutate=claimPi3` | claimed Π₃: 50b₁ → 49b₁ | 1, (S39) ε^{5/2}, (S37) ε³ |
| `--set=claimed --mutate=claimC4` | claimed c₄: 5/4 b₀⁴ → 9/4 b₀⁴ | 1, (S37) ε³ |
| `--set=claimed --mutate=S49` | c₂: 3/16 → 1/16 | 1, (S37) ε³ |
| `--set=claimed --mutate=S45` | Ω₃: −7β² → −6β² | 1, (S36) ε², (S37) ε³ |
| `--set=printed` | nothing | 1, (S39) ε^{5/2} |
| `--set=printed --mutate=S45` | Ω₃: −7β² → −6β² | 1, adds (S36) ε², (S37) ε³ |
| `--set=printed --mutate=S46` | printed Π₃: 24β³ → 12β³ | 1, adds (S37) ε³ |
| `--set=printed --mutate=S49` | c₂: 3/16 → 1/16 | 1, adds (S37) ε³ |
| `--set=printedB` | reading B | 1, (S36) ε², (S37) ε³, (S38) ε^{3/2}, (S39) ε^{5/2} |
| `--set=derived` | nothing | 0, none |
| `--numeric` | nothing | 0 |
| `--numeric --set=printed` | printed set tested | 1, exponent 2.500 |
| `--numeric --mutate=claimPi3` | claimed Π₃: 50b₁ → 49b₁ | 1 |
| (summary) | nothing | 0 |
| `--mutate=claimPi3` (summary) | claimed Π₃: 50b₁ → 49b₁ | 6 = 2 (claimed fails) + 4 (derived ≠ claimed) |

`controls.py --full` exited 0, with 21 of 21 rows as expected. The printed-set mutations matter most here. They show that (S36) at ε² and (S37) at ε³, which pass for the printed set, can fail. The S46 row shows that the printed f₃ is tied to the printed Π₃ through Ω₃.

## What the checks share

The derived solution and the residual checks share one residual routine. The numerical check shares the transcription. The hand derivation of Π₃ shares only the reading of the equations. No check shares code with whatever produced the claim, which this checker has not seen.


## Files and commits

- `solver/independent_eeg_nnlo/check.py` and `controls.py`, commit a006244 on branch `wip/i142`.
- Raw logs and JSON: folder `I142-mac/` of the raw-data repository, commit ce5f6e7. The paper's PDF is not stored, only its SHA-256.
- `solver/independent_eeg_nnlo/REPORT.md` is not written. The subagent harness refused the file ("Subagents should return findings as text, not write report files"), and I did not route around that. Its intended text is this comment from the first heading on.

## Recommended tasks

1. Commit this comment's report text as `solver/independent_eeg_nnlo/REPORT.md` on `wip/i142` (coordinator).
2. Compare this verdict and the derived Π₃, Ω₃, f₃ with the repository's own NNLO work, which this check did not read, and record agreement or disagreement in the notes.
3. A second, fully independent implementation (another CAS, or a direct numerical solve of (S36)–(S39) at small ε that does not use the transcribed NNLO) would remove the shared transcription from the cross-checks.
4. If the project needs Ψ₃ or the next order, extend `check.py` to (S38) at ε^{5/2} and (S39) at ε^{7/2}. The current controls carry over.
5. Whether to tell the authors is Nathan's decision. No public action was taken.

