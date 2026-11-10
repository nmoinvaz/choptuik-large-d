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
- Raw logs and JSON: folder `I142-mac/` of the raw-data repository, commit 551833a. The paper's PDF is not stored, only its SHA-256.
- `solver/independent_eeg_nnlo/REPORT.md` is not written. The subagent harness refused the file ("Subagents should return findings as text, not write report files"), and I did not route around that. Its intended text is this comment from the first heading on.

## Recommended tasks

1. Commit this comment's report text as `solver/independent_eeg_nnlo/REPORT.md` on `wip/i142` (coordinator).
2. Compare this verdict and the derived Π₃, Ω₃, f₃ with the repository's own NNLO work, which this check did not read, and record agreement or disagreement in the notes.
3. A second, fully independent implementation (another CAS, or a direct numerical solve of (S36)–(S39) at small ε that does not use the transcribed NNLO) would remove the shared transcription from the cross-checks.
4. If the project needs Ψ₃ or the next order, extend `check.py` to (S38) at ε^{5/2} and (S39) at ε^{7/2}. The current controls carry over.
5. Whether to tell the authors is Nathan's decision. No public action was taken.

## One order past the NNLO: Ψ₃, Π₄, Ω₄ and f₄ (issue agent-comms#40)

The order past the NNLO closes from (S36)–(S39) with the paper's own conventions, and two codes built on different
equations agree on it. Ψ₃ follows algebraically from (S38) at ε^{5/2}. Π₄ follows from (S39) at ε^{7/2}, with (S34)
fixing its free function of τ. Ω₄ follows algebraically from (S36) at ε³, and f₄ from (S37) at ε⁴ with (S29). No
assumption beyond the two conventions the NNLO already used is needed. With these terms, (S38) holds through ε^{5/2}
and (S39) through ε^{7/2}, where the printed series is testable only through ε^{3/2} and ε^{5/2}. Status: *checked
symbolically* (sympy 1.14.0), with a numerical cross-check of the ε-scaling (mpmath 1.3.0).

The terms build on the corrected NNLO of the sections above (the claimed set, which `check.py` derives again). The
paper gives nothing at this order ((S47) says Ψ₃ is not needed), so there is nothing printed to compare with. This
section carries out recommended task 4 above.

### The terms

β̃ and its τ-derivatives are as in (S35). Each term is an even polynomial in x. Ψ₃ has the powers x⁰, x², x⁴, Π₄ and
Ω₄ have x², x⁴, x⁶, and f₄ has x⁰ to x⁸. As in (S45), Ω₄ is written as 2β̃Π₄ plus a remainder. The machine-readable
source is `closed()` in `psi3.py`, and these blocks are generated from its `--print` output.

```math
\begin{aligned}
\Psi_3 &= \frac{1}{8}\Big(- \tilde\beta^{5} - \tilde\beta^{4}\tilde\beta' - 4\tilde\beta^{3} + 2\tilde\beta^{2}\tilde\beta' + 2\tilde\beta^{2}\tilde\beta'' + 6\tilde\beta\tilde\beta'^{2} + 2\tilde\beta\tilde\beta'\tilde\beta''\Big) \\
&\quad + \frac{x^{2}}{4}\Big(- 4\tilde\beta^{5} - 4\tilde\beta^{4}\tilde\beta' + 46\tilde\beta^{3} + 75\tilde\beta^{2}\tilde\beta' + 20\tilde\beta^{2}\tilde\beta'' + 3\tilde\beta^{2}\tilde\beta''' + 22\tilde\beta\tilde\beta'^{2} \\
&\qquad + 8\tilde\beta\tilde\beta'\tilde\beta'' + 2\tilde\beta'^{3} - 24\tilde\beta - 44\tilde\beta' - 24\tilde\beta'' - 4\tilde\beta'''\Big) \\
&\quad + \frac{x^{4}}{8}\Big(24\tilde\beta^{5} + 24\tilde\beta^{4}\tilde\beta' - 136\tilde\beta^{3} - 304\tilde\beta^{2}\tilde\beta' - 94\tilde\beta^{2}\tilde\beta'' - 10\tilde\beta^{2}\tilde\beta''' - 126\tilde\beta\tilde\beta'^{2} \\
&\qquad - 34\tilde\beta\tilde\beta'\tilde\beta'' - 8\tilde\beta'^{3} + 120\tilde\beta + 274\tilde\beta' + 225\tilde\beta'' + 85\tilde\beta''' + 15\tilde\beta'''' \\
&\qquad + \tilde\beta^{(5)}\Big)
\end{aligned}
```

```math
\begin{aligned}
\Pi_4 &= \frac{x^{2}}{8}\Big(- 8\tilde\beta^{3} - 4\tilde\beta^{2}\tilde\beta' + 3\tilde\beta^{2}\tilde\beta'' + \tilde\beta^{2}\tilde\beta''' + 17\tilde\beta\tilde\beta'^{2} + 16\tilde\beta\tilde\beta'\tilde\beta'' + \tilde\beta\tilde\beta'\tilde\beta''' \\
&\qquad + 2\tilde\beta\tilde\beta''^{2} + 3\tilde\beta'^{3} + \tilde\beta'^{2}\tilde\beta''\Big) \\
&\quad + \frac{x^{4}}{8}\Big(- 40\tilde\beta^{5} - 48\tilde\beta^{4}\tilde\beta' - 8\tilde\beta^{4}\tilde\beta'' - 16\tilde\beta^{3}\tilde\beta'^{2} + 126\tilde\beta^{3} + 276\tilde\beta^{2}\tilde\beta' + 107\tilde\beta^{2}\tilde\beta'' \\
&\qquad + 21\tilde\beta^{2}\tilde\beta''' + 2\tilde\beta^{2}\tilde\beta'''' + 117\tilde\beta\tilde\beta'^{2} + 58\tilde\beta\tilde\beta'\tilde\beta'' + 7\tilde\beta\tilde\beta'\tilde\beta''' + 4\tilde\beta\tilde\beta''^{2} + 15\tilde\beta'^{3} \\
&\qquad + 7\tilde\beta'^{2}\tilde\beta'' - 48\tilde\beta - 100\tilde\beta' - 70\tilde\beta'' - 20\tilde\beta''' - 2\tilde\beta''''\Big) \\
&\quad + \frac{x^{6}}{48}\Big(- 15\tilde\beta^{7} + 278\tilde\beta^{5} + 485\tilde\beta^{4}\tilde\beta' + 71\tilde\beta^{4}\tilde\beta'' + 136\tilde\beta^{3}\tilde\beta'^{2} - 968\tilde\beta^{3} - 2578\tilde\beta^{2}\tilde\beta' \\
&\qquad - 1075\tilde\beta^{2}\tilde\beta'' - 204\tilde\beta^{2}\tilde\beta''' - 15\tilde\beta^{2}\tilde\beta'''' - 1436\tilde\beta\tilde\beta'^{2} - 692\tilde\beta\tilde\beta'\tilde\beta'' - 54\tilde\beta\tilde\beta'\tilde\beta''' - 42\tilde\beta\tilde\beta''^{2} \\
&\qquad - 174\tilde\beta'^{3} - 58\tilde\beta'^{2}\tilde\beta'' + 720\tilde\beta + 1764\tilde\beta' + 1624\tilde\beta'' + 735\tilde\beta''' + 175\tilde\beta'''' \\
&\qquad + 21\tilde\beta^{(5)} + \tilde\beta^{(6)}\Big)
\end{aligned}
```

```math
\begin{aligned}
\Omega_4 - 2\tilde\beta\,\Pi_4 &= x^{2}\Big(- 21\tilde\beta^{4} - 14\tilde\beta^{3}\tilde\beta' - 4\tilde\beta^{3}\tilde\beta'' - 3\tilde\beta^{2}\tilde\beta'^{2} + 12\tilde\beta^{2} + 16\tilde\beta\tilde\beta' + 8\tilde\beta\tilde\beta'' \\
&\qquad - 4\tilde\beta'^{2}\Big) \\
&\quad + \frac{x^{4}}{2}\Big(- 16\tilde\beta^{6} - 4\tilde\beta^{5}\tilde\beta' - \tilde\beta^{5}\tilde\beta'' - \tilde\beta^{4}\tilde\beta'^{2} + 84\tilde\beta^{4} + 142\tilde\beta^{3}\tilde\beta' + 36\tilde\beta^{3}\tilde\beta'' \\
&\qquad + 30\tilde\beta^{2}\tilde\beta'^{2} + 7\tilde\beta^{2}\tilde\beta'\tilde\beta'' + \tilde\beta^{2}\tilde\beta''^{2} + 3\tilde\beta\tilde\beta'^{3} + \tilde\beta\tilde\beta'^{2}\tilde\beta'' - 72\tilde\beta^{2} - 152\tilde\beta\tilde\beta' \\
&\qquad - 105\tilde\beta\tilde\beta'' - 28\tilde\beta\tilde\beta''' - 3\tilde\beta\tilde\beta'''' - 5\tilde\beta'^{2} - 6\tilde\beta'\tilde\beta'' + 2\tilde\beta'\tilde\beta''' - 3\tilde\beta''^{2}\Big) \\
&\quad + \frac{x^{6}}{8}\Big(- 3\tilde\beta^{8} + 30\tilde\beta^{6} + 49\tilde\beta^{5}\tilde\beta' + 11\tilde\beta^{5}\tilde\beta'' + 8\tilde\beta^{4}\tilde\beta'^{2} - 72\tilde\beta^{4} - 202\tilde\beta^{3}\tilde\beta' \\
&\qquad - 75\tilde\beta^{3}\tilde\beta'' - 10\tilde\beta^{3}\tilde\beta''' - \tilde\beta^{3}\tilde\beta'''' - 136\tilde\beta^{2}\tilde\beta'^{2} - 64\tilde\beta^{2}\tilde\beta'\tilde\beta'' - 8\tilde\beta^{2}\tilde\beta''^{2} - 24\tilde\beta\tilde\beta'^{3} \\
&\qquad - 8\tilde\beta\tilde\beta'^{2}\tilde\beta'' + 48\tilde\beta^{2} + 172\tilde\beta\tilde\beta' + 94\tilde\beta\tilde\beta'' + 20\tilde\beta\tilde\beta''' + 2\tilde\beta\tilde\beta'''' + 150\tilde\beta'^{2} \\
&\qquad + 155\tilde\beta'\tilde\beta'' + 30\tilde\beta'\tilde\beta''' + 3\tilde\beta'\tilde\beta'''' + 35\tilde\beta''^{2} + 10\tilde\beta''\tilde\beta''' + \tilde\beta''\tilde\beta''''\Big)
\end{aligned}
```

```math
\begin{aligned}
f_4 &= \frac{1}{384}\Big(105\tilde\beta^{8} + 648\tilde\beta^{6} - 652\tilde\beta^{5}\tilde\beta' - 116\tilde\beta^{5}\tilde\beta'' - 48\tilde\beta^{4}\tilde\beta'^{2} - 192\tilde\beta^{4} - 1280\tilde\beta^{3}\tilde\beta' \\
&\qquad + 120\tilde\beta^{3}\tilde\beta'' + 188\tilde\beta^{3}\tilde\beta''' + 12\tilde\beta^{3}\tilde\beta'''' + 964\tilde\beta^{2}\tilde\beta'^{2} + 352\tilde\beta^{2}\tilde\beta'\tilde\beta'' - 28\tilde\beta^{2}\tilde\beta'\tilde\beta''' + 8\tilde\beta^{2}\tilde\beta''^{2} \\
&\qquad + 84\tilde\beta\tilde\beta'^{3} - 4\tilde\beta\tilde\beta'^{2}\tilde\beta'' + 192\tilde\beta^{2} + 368\tilde\beta\tilde\beta' - 100\tilde\beta\tilde\beta'' - 374\tilde\beta\tilde\beta''' - 234\tilde\beta\tilde\beta'''' \\
&\qquad - 42\tilde\beta\tilde\beta^{(5)} - 2\tilde\beta\tilde\beta^{(6)} - 356\tilde\beta'^{2} - 738\tilde\beta'\tilde\beta'' - 244\tilde\beta'\tilde\beta''' - 18\tilde\beta'\tilde\beta'''' - 114\tilde\beta''^{2} \\
&\qquad - 60\tilde\beta''\tilde\beta''' - 6\tilde\beta''\tilde\beta''''\Big) \\
&\quad + \frac{x^{2}}{96}\Big(- 15\tilde\beta^{8} - 88\tilde\beta^{6} + 62\tilde\beta^{5}\tilde\beta' + 14\tilde\beta^{5}\tilde\beta'' + 4\tilde\beta^{4}\tilde\beta'^{2} - 20\tilde\beta^{4} + 68\tilde\beta^{3}\tilde\beta' \\
&\qquad - 6\tilde\beta^{3}\tilde\beta'' - 20\tilde\beta^{3}\tilde\beta''' - 2\tilde\beta^{3}\tilde\beta'''' - 30\tilde\beta^{2}\tilde\beta'^{2} - 12\tilde\beta^{2}\tilde\beta'\tilde\beta'' - 2\tilde\beta^{2}\tilde\beta''^{2}\Big) \\
&\quad + \frac{x^{4}}{64}\Big(- 3\tilde\beta^{8} - 40\tilde\beta^{6} - 8\tilde\beta^{5}\tilde\beta' - 8\tilde\beta^{4}\tilde\beta'^{2} - 264\tilde\beta^{4} - 160\tilde\beta^{3}\tilde\beta' - 20\tilde\beta^{3}\tilde\beta'' \\
&\qquad + 4\tilde\beta^{3}\tilde\beta''' - 24\tilde\beta^{2}\tilde\beta'^{2} + 40\tilde\beta^{2}\tilde\beta'\tilde\beta'' + 4\tilde\beta^{2}\tilde\beta'\tilde\beta''' + 4\tilde\beta^{2}\tilde\beta''^{2} + 12\tilde\beta\tilde\beta'^{3} + 4\tilde\beta\tilde\beta'^{2}\tilde\beta'' \\
&\qquad + 192\tilde\beta^{2} + 256\tilde\beta\tilde\beta' + 128\tilde\beta\tilde\beta'' - 64\tilde\beta'^{2}\Big) \\
&\quad + \frac{x^{6}}{96}\Big(- 3\tilde\beta^{8} - 136\tilde\beta^{6} - 102\tilde\beta^{5}\tilde\beta' - 14\tilde\beta^{5}\tilde\beta'' - 44\tilde\beta^{4}\tilde\beta'^{2} + 988\tilde\beta^{4} + 1924\tilde\beta^{3}\tilde\beta' \\
&\qquad + 566\tilde\beta^{3}\tilde\beta'' + 64\tilde\beta^{3}\tilde\beta''' + 6\tilde\beta^{3}\tilde\beta'''' + 702\tilde\beta^{2}\tilde\beta'^{2} + 276\tilde\beta^{2}\tilde\beta'\tilde\beta'' + 28\tilde\beta^{2}\tilde\beta'\tilde\beta''' + 22\tilde\beta^{2}\tilde\beta''^{2} \\
&\qquad + 84\tilde\beta\tilde\beta'^{3} + 36\tilde\beta\tilde\beta'^{2}\tilde\beta'' - 768\tilde\beta^{2} - 1616\tilde\beta\tilde\beta' - 1120\tilde\beta\tilde\beta'' - 304\tilde\beta\tilde\beta''' - 32\tilde\beta\tilde\beta'''' \\
&\qquad - 40\tilde\beta'^{2} - 48\tilde\beta'\tilde\beta'' + 16\tilde\beta'\tilde\beta''' - 24\tilde\beta''^{2}\Big) \\
&\quad + \frac{x^{8}}{384}\Big(- 15\tilde\beta^{8} + 488\tilde\beta^{6} + 860\tilde\beta^{5}\tilde\beta' + 116\tilde\beta^{5}\tilde\beta'' + 256\tilde\beta^{4}\tilde\beta'^{2} - 2096\tilde\beta^{4} - 5728\tilde\beta^{3}\tilde\beta' \\
&\qquad - 2240\tilde\beta^{3}\tilde\beta'' - 388\tilde\beta^{3}\tilde\beta''' - 28\tilde\beta^{3}\tilde\beta'''' - 3508\tilde\beta^{2}\tilde\beta'^{2} - 1648\tilde\beta^{2}\tilde\beta'\tilde\beta'' - 108\tilde\beta^{2}\tilde\beta'\tilde\beta''' - 112\tilde\beta^{2}\tilde\beta''^{2} \\
&\qquad - 492\tilde\beta\tilde\beta'^{3} - 164\tilde\beta\tilde\beta'^{2}\tilde\beta'' + 1728\tilde\beta^{2} + 4560\tilde\beta\tilde\beta' + 3812\tilde\beta\tilde\beta'' + 1590\tilde\beta\tilde\beta''' + 362\tilde\beta\tilde\beta'''' \\
&\qquad + 42\tilde\beta\tilde\beta^{(5)} + 2\tilde\beta\tilde\beta^{(6)} + 900\tilde\beta'^{2} + 930\tilde\beta'\tilde\beta'' + 180\tilde\beta'\tilde\beta''' + 18\tilde\beta'\tilde\beta'''' + 210\tilde\beta''^{2} \\
&\qquad + 60\tilde\beta''\tilde\beta''' + 6\tilde\beta''\tilde\beta''''\Big)
\end{aligned}
```

**The linear part.** Linearized in β̃ (f = 1, Ω = 0, D = d/dτ), (S38) and (S39) give, order by order,
S_k = (D + x∂ₓ + 1)P_k − x∂ₓS_{k−1} and (1/x)∂ₓP_{k+1} = (D + x∂ₓ + 2)S_k, with P₁ = β̃. This reproduces the linear
parts of Π₂, Ψ₂ and the corrected Π₃, and gives

```math
\Psi_3 \supset \frac{x^4}{8}(D+1)\cdots(D+5)\,\tilde\beta - x^2(D+1)(D+2)(D+3)\,\tilde\beta,
\qquad
\Pi_4 \supset \frac{x^6}{48}(D+1)\cdots(D+6)\,\tilde\beta - \frac{x^4}{4}(D+1)\cdots(D+4)\,\tilde\beta .
```

This hand derivation was posted on the issue before any new code ran, and `psi3.py` checks the closed forms against it.

### Code 1: `psi3.py`, from the paper's equations

(S36)–(S39) are typed again in `psi3.py` from the paper's HTML rendering, read again on 2026-11-10. They agree with
`check.py`'s transcription. The residuals are formed by truncated series arithmetic in √ε, exact at every power they
report, and the NLO and NNLO below the new terms are `check.py`'s claimed set. The default run checks the closed forms,
derives the four terms again order by order, and measures the operator each unknown enters through.

| equation | vanishes through | next order (needs a term not given) | unknown fixed there | operator, measured |
|---|---|---|---|---|
| (S36) | ε³ | ε⁴, 166 terms | Ω₄ | identity |
| (S37) | ε⁴ | ε⁵, 160 terms | f₄ | (1/x)∂ₓ, f₄(τ,1) = 0 |
| (S38) | ε^{5/2} | ε^{7/2}, 113 terms | Ψ₃ | −identity |
| (S39) | ε^{7/2} | ε^{9/2}, 119 terms | Π₄ | −(1/x)∂ₓ, Π₄(τ,0) = 0 |

Every power below the next order vanishes, including the half-integer ones (S35) never populates. The next order is
nonzero in every equation, so the residuals are not trivially zero. Ψ₃ and Ω₄ are unique, and Π₄ and f₄ are unique up
to a function of τ, which (S34) and (S29) set to zero. The derivation equals the closed forms, and both linear parts
equal the hand derivation. `--crosscheck` compares this file's truncated residuals with `check.py`'s fully expanded
ones on the NNLO set, at every power `check.py` forms, and they are equal.

`--numeric` evaluates all four residuals with mpmath at 80 digits, with numerical derivatives and the exact ln f. It
uses β̃ = 0.7 sin τ + 0.4 cos 2τ + 0.2 at τ = 0.3, x = 0.8, as `check.py --numeric` does. It passes when each local
exponent with the new terms is within 0.1 of the expected one, and each without them is more than 0.6 below it.

| equation | expected | local exponent, new terms (ε = 1e-5 to 1e-6) | without them |
|---|---|---|---|
| (S36) | 4 | 4.000 | 3.000 |
| (S37) | 5 | 5.000 | 4.000 |
| (S38) | 3.5 | 3.500 | 2.500 |
| (S39) | 4.5 | 4.500 | 3.500 |

### Code 2: `third_order_eeg.py`, against Proposition 1

`solver/inner_pert/third_order_eeg.py` carries Proposition 1's recursion (`second_order_eeg.outer`, written again with
K₃ and ω₃ free and longer derivative chains) one order further, to S₃, D₃, G₄ and B₅. Its lower orders equal
`second_order_eeg.outer(3)`. EEG's series, with the NLO and corrected NNLO of `second_order_eeg.eeg_coeffs` and the new
terms typed in this file, is re-expanded in e = 1/n under the map of `second_order_eeg.py` (f = c, x²Ω = (n − 1)(b − 1),
Π = y/(√2x), Ψ = (p + m)/(√2x²), ε = e/(1 + e), β̃ = √(2(1 + e))F_n, d/dτ = ω d/dθ). It equals Proposition 1 at every
order:

| field | orders compared | terms at the new order (ours) |
|---|---|---|
| f = c | e⁰–e⁴ | 250 at e⁴ |
| Ω = u/x² | e⁰–e⁴ | 141 at e⁴ |
| Π/√e = y/(√2 x √e) | e⁰–e³ | 102 at e³ |
| Ψ/√e = (p + m)/(√2 x² √e) | e⁰–e³ | 59 at e³ |

Setting any one new term to zero breaks its order (check N). The map determines each new term uniquely from its order,
since distinct monomials in β̃ and its derivatives go to distinct monomials in F, its θ-derivatives and ω₀. So this
comparison fixes the four terms by itself, from this repository's equations E1–E4. Check I finds the NLO and NNLO it
takes from `second_order_eeg.py` equal to `check.py`'s claimed set, the other transcription. The run takes 8–17 s.

### Controls

`controls.py --full` exited 0, with 40 of 40 rows as expected: the 21 rows of `check.py`, unchanged, and 19 rows of
`psi3.py`. The failing sets of the `psi3.py` rows were written in `controls.py` before their first run. `--quick` forms
only the residuals, the boundary values and the linear part, so a row's failing orders are those of its mutated term.

| `psi3.py` flags | what it perturbs | exit, failing orders (expected = observed) |
|---|---|---|
| (none) | nothing | 0, none |
| `--quick` | nothing | 0, none |
| `--quick --mutate=Psi3` | Ψ₃, x⁴ coefficient of β̃²β̃″, −47/4 → −43/4 | 1, (S38) ε^{5/2}, (S39) ε^{7/2} |
| `--quick --mutate=Psi3c` | Ψ₃, x⁰ coefficient of β̃⁴β̃′, −1/8 → 7/8 | 1, (S38) ε^{5/2}, (S39) ε^{7/2} |
| `--quick --mutate=Pi4` | Π₄, x⁶ coefficient of β̃⁽⁶⁾, 1/48 → 2/48 | 1, (S36) ε³, (S39) ε^{7/2} |
| `--quick --mutate=Pi4b` | Π₄, x² coefficient of β̃³, −1 → 0 | 1, (S36) ε³, (S39) ε^{7/2} |
| `--quick --mutate=Pi4c` | Π₄ + β̃ (a constant in x) | 1, (S36) ε³ only, and Π₄(τ,0) ≠ 0 |
| `--quick --mutate=Omega4` | Ω₄, x² coefficient of β̃⁴, −23 → −22 | 1, (S36) ε³, (S37) ε⁴ |
| `--quick --mutate=f4` | f₄, x² coefficient of β̃⁸, −5/32 → 27/32 | 1, (S37) ε⁴ |
| `--quick --mutate=f4c` | f₄, x⁰ coefficient of β̃⁸, 35/128 → 163/128 | 1, no residual, only f₄(τ,1) ≠ 0 |
| `--mutate=Psi3` | as above, full mode | 1, (S38) ε^{5/2}, (S39) ε^{7/2} |
| `--quick --mutate=eqS36` | this file's (S36): (1 − 2ε)Π² → (1 − 3ε)Π² | 1, (S36) ε¹, ε², ε³ |
| `--crosscheck` | nothing | 0, none |
| `--crosscheck --mutate=eqS36` | as above | 1, (S36) ε¹, ε², ε³ |
| `--numeric` | nothing | 0 |
| `--numeric --mutate=Psi3`, `Pi4`, `Omega4`, `f4` | as above | 1 each |

The `Pi4c` and `f4c` rows show that the boundary checks can fail on their own. A constant in Π₄ leaves (S39) alone and
is caught by (S34) and by (S36) through 2β̃Π₄, and a constant in f₄ leaves every residual alone and is caught only by
(S29).

`third_order_eeg.py --controls` exited 0, with 8 of 8 rows as expected. The first seven were written before the first
run, and the `Psi2` row before check I was added:

| `third_order_eeg.py` flags | what it perturbs | exit, failing (field, order) (expected = observed) |
|---|---|---|
| (none) | nothing | 0, none |
| `--mutate=Psi3` | as in `psi3.py` | 1, Ψ e³ |
| `--mutate=Pi4` | as in `psi3.py` | 1, Π e³ |
| `--mutate=Omega4` | as in `psi3.py` | 1, Ω e⁴ |
| `--mutate=f4` | as in `psi3.py` | 1, f e⁴ |
| `--mutate=K3` | β̃ without K₃ (a map error at the new order only) | 1, Π e³, f e⁴, Ω e⁴ |
| `--mutate=map` | ε = e instead of e/(1 + e) | 1, f, Ω e²–e⁴, Π e¹–e³, Ψ e²–e³ |
| `--mutate=Psi2` | `second_order_eeg`'s own (S43) mutation, 11 → 10 | 1, Ψ e², e³, and check I |

The baseline before any change, and again at the end, was `check.py` exit 0, `controls.py --full` exit 0 (21 of 21)
and `second_order_eeg.py` exit 0.

### What the two codes share

Both were written by the same agent in one session and run on sympy 1.14.0.

- **The new terms' text.** `closed()` in `psi3.py` and `new_terms()` in `third_order_eeg.py` hold the same sympy
  output of `psi3.py --print`, pasted into each file, not imported. A paste error in either copy fails that code's
  check. Each code tests its copy against different equations.
- **The equations.** Code 1 uses the paper's (S36)–(S39), as read for this task and by `check.py`'s author. Code 2 uses
  none of them. It uses this repository's E1–E4 through Proposition 1's recursion and the map of
  `solver/supercrit/eqmap.py`, which is *checked symbolically* there.
- **The lower orders.** Code 1 takes the NLO and NNLO from `check.py` (claimed set). Code 2 takes them from
  `second_order_eeg.eeg_coeffs` (printed coefficients plus `delta_Pi3`, `delta_f3`). These are two transcriptions of
  (S40)–(S51) by different agents, each confirmed by its own check. Check I of code 2 compares them and finds them
  equal; it is the only place where code 2 reads `check.py`.
- **Within code 1.** The derivation and the residual checks share the residual routine. The numerical check shares the
  closed forms and the reading of the equations, not the series arithmetic. The `--crosscheck` compares that routine
  with `check.py`'s, which expands fully instead of truncating. The hand derivation of the linear part shares only the
  reading of (S38) and (S39).
- **Within code 2.** The recursion is a copy of `second_order_eeg.outer`, so the two share its logic. They agree at
  every order the older one computes.

### Discrepancies

None. The two codes agree on every coefficient, and nothing at this order is printed in the paper to disagree with.
Two observations for the notes: Ω₄(τ,0) = 0, as for Ω₃, while Ω₂(τ,0) = −2β̃², and Ψ₃ has an x⁰ term, so Ψ(τ,0) changes
at ε^{7/2}.

### Proposed text

For notes/SECOND_ORDER.md §1, after the corrected NNLO (for the coordinator; that file was not edited here). The task
is private, so this waits for Nathan's word.

> **One order further.** (S38) at ε^{5/2}, (S39) at ε^{7/2}, (S36) at ε³ and (S37) at ε⁴ fix Ψ₃, Π₄, Ω₄ and f₄
> uniquely under (S29) and (S34). They are written in `solver/independent_eeg_nnlo/psi3.py` (`closed()`, derived there
> from EEG's equations) and its REPORT.md. EEG's series with them equals Proposition 1 through 1/n⁴ for f and Ω and
> through 1/n³ for Π/√ε and Ψ/√ε (`third_order_eeg.py`). *Checked symbolically.* Their linear parts are
> (x⁴/8)(D+1)⋯(D+5)β̃ − x²(D+1)(D+2)(D+3)β̃ in Ψ₃ and (x⁶/48)(D+1)⋯(D+6)β̃ − (x⁴/4)(D+1)⋯(D+4)β̃ in Π₄.

In §1.1, the sentence "Ψ₃ is 'not needed' in EEG, so Ψ is the same in both versions" stays true of EEG's text. A
comparison of the N³LO with the states would test Ψ₃ there (recommended task 1).

### Files and commits

- Branch `wip/ac40-psi3` of the worktree `/Users/nathan/Source/choptuik-ac40`, local only, not pushed. Commits 7781543
  (scaffold), 1e67b36 (`psi3.py`, `controls.py`), 70ac34e (`third_order_eeg.py`) and the commit of this section, which
  also adds check I and the `Psi2` control to `third_order_eeg.py`.
- The logs of the final runs are in the session's scratchpad on the Mac: `controls.py --full`, `psi3.py` default and
  `--print` and `second_order_eeg.py` at commit 70ac34e, and `third_order_eeg.py` default and `--controls` after
  check I. They were not put in the raw-data repository, which is public. Every log is reproducible in about 25
  minutes.

### Commands (from the repository root)

```
python3 solver/independent_eeg_nnlo/psi3.py                  # exit 0, 3 s
python3 solver/independent_eeg_nnlo/psi3.py --crosscheck     # exit 0, 45 s
python3 solver/independent_eeg_nnlo/psi3.py --numeric        # exit 0, 10 s
python3 solver/independent_eeg_nnlo/controls.py --psi3       # 19 of 19, 4 min
python3 solver/independent_eeg_nnlo/controls.py --full       # 40 of 40, 20 min
(cd solver/inner_pert && python3 third_order_eeg.py)         # exit 0, 17 s
(cd solver/inner_pert && python3 third_order_eeg.py --controls)   # 8 of 8, 1 min
```

### Recommended tasks

1. **The N³LO against the finite-D solutions.** Add Ψ₃, Π₄, Ω₄ and f₄ to the models of
   `solver/inner_pert/nnlo_vs_solution.py` (issue #126) and refit R(D) = max |F_sol − F_model| over its states. Why:
   it tests the terms against solutions, sharing nothing with either derivation, it tests Ψ₃ where EEG gave none, and
   it tests the notes' estimate that the ε^{7/2} term of Π is about 30 times Π₃ (§1.1), now computable as Π₄.
   Acceptance: a planted run first recovers plants at ε⁵, ε⁵, ε^{9/2}, ε^{9/2} (f, Ω, Π, Ψ) within 0.25. Then the
   fitted exponents are within 0.25 of 5, 5, 4.5 and 4.5 and farther than 0.5 from 4, 4, 3.5 and 3.5, and a mutated
   Π₄ coefficient moves the Π exponent out of its window. The near-null held harmonics at D ≥ 150 (§1.1, criterion C6)
   enter through six τ-derivatives here and may limit the fit. Depends on: this issue, the states of I126-mac and
   T138-win2. Whether it is public or private is the coordinator's decision, since it extends the check of EEG's NNLO.
2. **Merge and the notes' text.** Review `wip/ac40-psi3`, merge it when Nathan allows, add the proposed paragraph to
   notes/SECOND_ORDER.md §1, and give this run's logs a private home. Acceptance: on main, `controls.py --full` gives
   40 of 40 and `third_order_eeg.py --controls` gives 8 of 8. Depends on: Nathan's word, since the task is private.
3. **Orders further, and their growth (low priority).** `psi3.py`'s truncated solve is generic in the order. Carry it
   to Π₆ and f₆, checking each order against (S36)–(S39) and Proposition 1, and evaluate each order's largest
   coefficient on the tuned D = ∞ profile. Why: the notes measure a factor of 12–34 per order (§1.1), and the exact
   ratios would show whether the series' growth is set by the τ-derivatives of β̃ as read there. Acceptance: every
   order passes both codes with its mutations firing, and a planted geometric series is recovered by the ratio
   estimate. Depends on: this issue.
