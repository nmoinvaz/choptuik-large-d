# Review of the M = 5 second-order certificate (A_M2.md)

An independent, adversarial review of [A_M2.md](A_M2.md) and its new code at commit b0ed77e of
`wip/m5-second-order`. Written by Claude (Anthropic) at Nathan Moinvaziri's request. The reviewer did not write the code
under review and had not seen it before. The review followed AGENTS.md, "How results are checked" and
"Computer-assisted proofs".

**Verdict.** VERDICT_PLACEHOLDER

Contents

- [1. The claim and the trusted code](#1-the-claim-and-the-trusted-code)
- [2. The mathematics against the code](#2-the-mathematics-against-the-code)
- [3. Issues](#3-issues)
- [4. Making it fail: mutations and what they reach](#4-making-it-fail-mutations-and-what-they-reach)
- [5. The independent computation](#5-the-independent-computation)
- [6. Reruns](#6-reruns)
- [7. What was checked and what was not](#7-what-was-checked-and-what-was-not)
- [8. Reproduce](#8-reproduce)

## 1. The claim and the trusted code

A_M2.md §4 and §6.4 claim, given Braaksma's theorem and the start-value lemma `fixed_point_M.start_data_M`:

- g, the D-jumps at ξ₁ = 0.3 of the M = 5 truncation, has exactly one zero in the curved parallelepiped
  p(X) = {p̂ + Tx + ½(0, Γ[x_o, x_o]) : x ∈ X}, X = [−10⁻¹⁰, 10⁻¹⁰]² × [−10⁻¹⁸, 10⁻¹⁸]² × [−10⁻¹⁹, 10⁻¹⁹]²;
- Dg is non-singular there;
- the zero has A ∈ [2.71857918921174, 2.71857918921175] and ω/ω∞ ∈ [0.999228403206567, 0.999228403206568] (and
  the four inner coordinates of §6.4).

The new trusted code is `certify_M2_lohner.py` (661 lines), `certify_M2.py` (749 lines) and `main2` of
`certify_M_recheck.py`. It rests on the reviewed `certify_M_lohner.py`, `certify_M_sys.py`, `validated_int.py` and
`lemma6_common.py` (A_review.md, A_M_review.md). `certify_M2_fd.py` is not trusted: it supplies the chart's Γ, which
can be anything, and the control D2.
