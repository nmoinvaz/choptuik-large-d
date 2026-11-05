# The flat Yang–Mills family as a test of the D = ∞ inner selection (queued 2026-11-05, not started)

**The question.** In odd d ≥ 11, flat SO(d) Yang–Mills with the magnetic ansatz has smooth self-similar blowup profiles
in one-to-one correspondence with the zeros in (0, 1) of an explicit polynomial P_m(z) of degree m = (d − 5)/2, the
light-cone resonance condition (proved, [BGW26] in [LITERATURE.md](LITERATURE.md)). There are at least three [BK26].
Two are explicit, u₊ and u₋, and the large-d limits of both solve the D = ∞ flat inner equation
W_tt = W_s + W(1 − W²)/(2s) exactly (LITERATURE.md §4.2, `solver/matter/lit_remarks_check.py` part 4). The third, u*, is
known only numerically. Does u* have a large-d limit, and is that limit picked out by a condition of the inner equation
alone, as the scalar's Δ∞ is fixed by its inner problem (README, Proposition 5)? This tests the inner-selection idea on
a continuously self-similar family with exact data in every odd d. The idea is from notes/YANG_MILLS_AI.md §4, lead 2.

**Procedure (criteria fixed before any run).**
1. *Planted truth.* Implement P_m(z) from [BGW26] in exact rational arithmetic. Reproduce three zeros in (0, 1) for
   every odd d from 11 to 35 [BGW26], the zeros of u₊ and u₋ at their closed-form values, and the count of three to
   d = 101 [BK26]. Pass only if every one is reproduced exactly. Otherwise stop and report.
2. *The roots to large d.* Isolate the zeros in (0, 1) with a certified method (Sturm sequences, or python-flint's
   real-root isolation) for odd d up to 301, and record z₊, z₋ and z* with enclosures. Fit the large-d behaviour of z*
   and of the profile parameters it fixes. Run the same fitting code first on z₊ and z₋, whose limits are known, and
   accept the fit only if it returns those limits within its stated error.
3. *The inner limit.* From u* at the largest d, build its profile in the inner variables s = r²/(2(d − 3)) and t, and
   test whether the limit solves the inner equation. If it does, find which condition of the inner equation selects it
   (regularity at s = 0 together with its behaviour at large s, or its analyticity at the light cone). Write the
   criteria for this step into this file before step 3 runs.

**Controls.** A P_m with one coefficient changed must change the count of zeros at some d ≤ 35. The fit of step 2 must
recover the known limits of z₊ and z₋. Step 3's inner test must fail on a profile that is not a solution (u₊ with its
α off by 1%, as in lit_remarks_check.py).

**How it runs.** One coordinator subagent (the paper's formulas, then exact computation; small CPU on the Mac), in its
own worktree. Raw output in the folder `ym-family-mac/` of the raw-data repository.
