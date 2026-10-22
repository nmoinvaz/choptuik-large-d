# Referee simulation: "Choptuik's echo period at large dimension"

Audit of [paper/main.tex](../paper/main.tex) (the build of 2026-10-21) and of the README statements it summarises. It
follows the `referee-sim` protocol of the BootLoops skills repository. The closed-form statements get a separate
`constant-recognition` audit in §4.

- **Context.** This pass ran in a separate agent, but not a context free of the authors' framing: I read the README,
  PROOF.md and the paper before simulating the referees. The objections were sharpened against that bias with the
  protocol's sting, knowledge and lead tests.
- **Sources.** Literature facts used here come from [LIT_REVIEW.md](LIT_REVIEW.md) and [REF_CHECK.md](REF_CHECK.md),
  where each has its source and locator.
- **No edits.** No paper text was changed (outside the bibliography). Everything below is a proposal.

Severity:

- **HIGH**: an objection a referee would lead with that the paper does not answer, or a claim that the project's own
  numbers contradict.
- **MEDIUM**: answered in the body but not where the impression forms (title, abstract, intro).
- **LOW**: answered, could be more prominent.

## 1. Audiences

| Audience | Why it gets a row |
|---|---|
| Numerical relativity / critical collapse | the incumbent community for $`\Delta(D)`$, $`\gamma(D)`$; the paper corrects a published value ($`D = 5`$) and a published conjecture ($`\Delta\to0`$) |
| The incumbent large-$`D`$ DSS group (Ecker, Ecker, Grumiller, Jechtl) | the paper says their NLO condition does not select $`\Delta`$ and that their $`D = 5`$ value is under-resolved |
| Large-$`D`$ gravity (Emparan school; Rozali–Way; Clark–Pimentel) | the $`1/D`$ expansion, the inner region and Prop. 5 borrow its methods |
| Exponential asymptotics / resurgence | Props. 3–4, the Stokes reading, Conjecture 1 and the amplitude-series estimators borrow its methods and standards |
| Computer-assisted proofs | the $`M = 1`$ certificate is described in the paper; PROOF.md positions the project against this community |
| Experimental mathematics / constant recognition | the closed-form candidates and the "nothing found" search for $`\Delta_\infty`$ |
| Practitioner (uses $`\Delta(D)`$, $`\gamma(D)`$ or the formula) | the rational formula and the value table |
| General PRD reader / abstract-only reader | the abstract carries the headline claims |

## 2. Objections and where the paper answers them

| # | Audience | Axis | Objection (in the referee's voice) | Answered where? | Severity |
|---|---|---|---|---|---|
| 1 | NR / critical collapse | correctness | "Your $`\gamma`$ is defined by $`M\propto(p-p^*)^\gamma`$ (intro) and then set to $`1/\lambda`$. In $`D`$ dimensions $`M\propto r^{D-3}`$, so the mass exponent is $`(D-3)/\lambda`$, which grows without bound. Your $`\gamma_\infty = 1/2`$ and Prop. 5's $`r_{\rm BH}^2\propto p-p^*`$ are radius exponents. Which is it?" | Not stated. The intro uses the mass form; Prop. 5's discussion uses the radius form. | **HIGH** (headline-number convention) |
| 2 | NR / critical collapse | correctness | "You continue a DSS solution from $`D = 4`$ to $`61`$, but nobody has evolved collapse above $`D = 14`$, and you compute one growth rate, not the spectrum. How do you know this branch is still *the* critical solution, a codimension-one attractor with a single unstable mode? It could have acquired a second unstable mode, or lost the competition to a CSS solution (Clark–Pimentel construct CSS critical solutions at large $`D`$; Kol conjectured critical dimensions). Without this, $`\gamma = 1/\lambda`$ is not the critical exponent." | Not in the paper. The README says "assumed here: only that mode's growth rate is computed, not the full spectrum". The paper's "Open" list mentions only uniqueness *at $`D=\infty`$*. | **HIGH** |
| 3 | NR / critical collapse | correctness | "$`\pm1.5\times10^{-8}`$ on $`\Delta_\infty`$ is the spread of 48 fit choices, not an uncertainty with a stated coverage. Has the pipeline recovered a planted limit from synthetic data with your noise?" | Partly: Sec. limit says what the spread is ("Over 48 choices of model..."). No planted-truth test (also flagged in notes/BOOTLOOPS.md). | MEDIUM |
| 4 | NR / critical collapse | correctness | "Above $`D \approx 30`$–$`40`$ your only check is the double-double code against itself (a second build on a second machine, resolution and pinning tests). The independent $`f64`$ solver stops at $`D = 40`$ with $`10^{-7}`$ noise. What independent evidence supports $`10^{-16}`$ per point at $`D = 55`$?" | Table I lists the internal checks; the abstract says the independent solver covers $`4\le D\le40`$. Nothing independent above 40. | MEDIUM |
| 5 | NR / critical collapse | novelty | "Prior numerics cover more than 'integer $`D = 4`$–$`14`$': Bland et al. did $`3.5\le D\le14`$, Bland's thesis $`3.02`$–$`3.9`$, Taves–Kunstatter (uncited) $`D\le7`$. Get the history right." | Intro sentence is inaccurate (REF_CHECK.md §4). | MEDIUM |
| 6 | Incumbent (Ecker et al.) | correctness | "Our Eq. (20) is derived in our expansion at the horizon from $`f\le1`$. Your rebuttal uses your scaling and a global bound $`b\ge1`$. Show the next term in *our* variables. Note also that a 2026 follow-up (Lai & Wen) uses the condition." | Sec. eeg gives the global bound, the discriminant and the data violating Eq. (20). Lai–Wen is not mentioned. | LOW (correctness answered); MEDIUM for the missing citation |
| 7 | Incumbent | correctness | "Our $`D = 5`$ value differs by $`6\times10^{-5}`$ because of resolution, you say, using our repository. Did you run our code?" | Sec. results: two independent codes agree to $`10^{-11}`$ at $`D=5`$, and the published resolution sequence moves toward this value. | LOW |
| 8 | Incumbent | novelty | "You attribute the $`\gamma\to1/2`$ conjecture to us; we cited Clark–Pimentel for it. And you do not cite Clark–Pimentel at all." | Not answered; the intro attribution is inflated (REF_CHECK.md §4). | MEDIUM |
| 9 | Large-$`D`$ gravity | novelty | "Your '$`n`$-free inner problem' at $`x\sim\sqrt n`$ is the leading-order large-$`D`$ system of Rozali and Way (2018, Eqs. (9)), in the regime where time is rescaled by $`\sqrt D`$. They wrote that 'the scalar field equation resembles a heat equation with spatial and temporal coordinates swapped', and their $`\gamma = 1/2`$ comes from the $`r^2`$ threshold. Prop. 5(i), 'the inner equations are the nonlinear heat equation', needs that credit. What is actually new is the DSS problem on those equations." | Not answered. RW are cited only for the $`\gamma=1/2`$ heuristic. | **HIGH** |
| 10 | Large-$`D`$ gravity | correctness | "Your all-orders statement holds within an ansatz of integer powers after an overall $`D^{-1/2}`$. Large-$`D`$ problems routinely have more than one scale (near zones of width $`1/D`$, $`\sqrt D`$ regions; you have one at $`x\sim\sqrt n`$ yourself). Does Prop. 1 claim anything about the true solution, or only about a formal family?" | Body: "this is the scaling measured above, and the statements below hold within this ansatz"; "Why the ansatz"; (H2) is a *hypothesis* of Conjecture 1. The abstract states the result without the ansatz. | MEDIUM |
| 11 | Large-$`D`$ gravity | novelty | "Non-perturbative, trans-series effects in $`1/D`$ are known in large-$`D`$ gravity (Emparan–Herzog review, Sec. III.F). Position the Stokes reading against that." | Not mentioned. | LOW |
| 12 | Exponential asymptotics | correctness | "'Selected beyond all orders' needs more than formal solvability for every $`\omega`$. You must know the outer series is the asymptotic expansion of actual solutions up to the light cone. That fails in your own inner region and at $`x = 1`$ (gaps (a)–(c)). Your abstract says 'so the period is selected beyond all orders' as a theorem." | Body: the gaps are listed carefully in Sec. inner; Conjecture 1 has (H2) as a hypothesis. Abstract: no qualifier. | MEDIUM |
| 13 | Exponential asymptotics | correctness | "$`\omega = (1.0000\pm0.0001)\,\omega_\infty`$ comes from twelve coefficients of a series whose terms peak near 240 at the zero, through Padé, ratio and 'factored-series' estimators. The $`\pm0.0001`$ is the spread of estimators, not an error bound. The $`\sigma_k`$ themselves come from late-order fits. State it that way where the number appears." | Body Sec. inner is explicit ("Whether the remaining $`10^{-4}`$ ... closes needs more orders"). The abstract presents "$`(1.0000 \pm 0.0001)`$" and "$`\Delta_\infty = 1.6439\pm0.0002`$ from the $`D=\infty`$ equations alone" with only a general "the rule itself is not yet proved". | MEDIUM |
| 14 | Exponential asymptotics | novelty | "Prop. 4 is a Watson's-lemma/Darboux late-terms computation for a Gauss function with one large parameter; that is textbook (Dingle; large-parameter $`{}_2F_1`$ asymptotics, e.g. Paris 2013). Do not call it new; call it an explicit instance. The new part is the identification with the light-cone obstruction of a gravitational problem." | The paper does not claim the technique is new. The README's "were not found in print" covers "the Stokes reading" broadly. | LOW |
| 15 | Exponential asymptotics | correctness | "The nonlinear multipliers $`S_\nu`$ need Borel summability of a system with a repeated fast eigenvalue across harmonics. Is $`S`$ even well defined?" | Sec. inner, gap (a); PROOF.md (Braaksma at fixed $`M`$). Answered in the body. | LOW |
| 16 | Computer-assisted proofs | correctness | "What is certified is one component of the jump, at one radius, for the one-harmonic truncation, whose zero is 22% from the physical frequency and does not survive harmonic 3. Two lemmas are missing. This certifies nothing about $`\Delta_\infty`$." | Sec. inner says all of this explicitly. | LOW |
| 17 | Computer-assisted proofs | novelty | "Choptuik's solution has a computer-assisted existence proof with $`\Delta`$ enclosed to 80 digits (Reiterer–Trubowitz, CMP 2019), and the Stokes constant of an inner equation has been certified (Baldomá–Capiński–Guardia–Seara, JNS 2023). Neither is cited in the paper. PROOF.md's 'the first rigorous derivation of a critical-collapse echo period' is wrong as worded." | Not in the paper; PROOF.md cites both but makes the overclaim (LIT_REVIEW.md C10). | MEDIUM |
| 18 | Constant recognition | correctness | "'The closest sits $`1.3\times10^{-5}`$ from the limit, more than 1000 times its uncertainty': with your stated $`\pm1.5\times10^{-8}`$ it is 835 times." | Contradicted by the paper's own numbers (§4). | **HIGH** (own numbers contradict) |
| 19 | Constant recognition | correctness | "Your 'nine candidate closed forms' are not candidates in any principled sense. Five are hits of a window search over about $`5.1\times10^5`$ expressions (about 9 expected by chance in that window); four are famous constants. Your 'search finds nothing' null has expected chance count 0.017 (0.46 in the wide window), so it has no power. State the family, height and expected count, or drop the search." | Partly: "Whether $`\Delta_\infty`$ has a closed form is open" is the right conclusion. The rest is not stated (§4). | MEDIUM |
| 20 | Constant recognition | correctness | "'Nine candidate closed forms for $`\Delta_\infty`$ are ruled out' (intro item 2) needs the conditional 'if $`\Delta`$ is analytic in $`1/D`$'." | Sec. limit's candidate subsection and the README carry the conditional; the intro item does not. | LOW |
| 21 | Practitioner | correctness | "Can I use $`a_0`$ of your rational formula as $`\Delta_\infty`$? It is $`1.6438975`$, $`4\times10^{-6}`$ off." | The README says so ("this formula's own limit is $`a_0 = 1.643898`$, $`4\times10^{-6}`$ high"); the paper does not. | LOW |
| 22 | General reader / abstract | correctness | "'A power-law approach to $`\Delta = 0`$ ... is excluded by the data.' Data to $`D = 61`$ cannot exclude a crossover at larger $`D`$ or exponents $`p\lt0.01`$." | Body: "$`p \ge 0.01`$", fitted from $`D=10`$; the abstract has no scope. | MEDIUM |
| 23 | General reader / abstract | correctness | "'$`\gamma_\infty = 1/2`$ ... which we explain: at $`D=\infty`$ the unstable mode is generated by an exact symmetry.' Explained only under Conjecture 1, uniqueness of the unstable mode and continuity of $`\lambda(D)`$." | The body italicises exactly these conditions; the abstract does not. | MEDIUM |

The sting test was applied to the HIGH rows: answering any of them requires an edit. Rows 7, 15 and 16 are honest
"answered" verdicts. They passed only after checking that the answer sits in the section where that referee's objection
forms.

## 3. The abstract, read cold

An abstract-only reader comes away believing:

1. $`\Delta(D)`$ is known to about 15 digits up to $`D = 61`$, with an independent check to $`D = 40`$. This is
   accurate.
2. $`\Delta_\infty = 1.64389375\pm1.5\times10^{-8}`$ if $`\Delta`$ is analytic in $`1/D`$. Accurate and well labelled.
3. $`\gamma_\infty = 1/2`$ is explained by a symmetry. Overstated: the conditions of row 23 and the convention of row 1
   are missing.
4. The $`\Delta\to0`$ conjecture is excluded. A power law is excluded only for $`p \ge 0.01`$ (row 22).
5. "We show ... the period is selected beyond all orders." This needs "within the scaling ansatz", and (H2) (rows 10
   and 12).
6. The Stokes multiplier and the selection rule are labelled conjectural. Good: "We conjecture", "the rule itself is
   not yet proved".
7. $`\Delta_\infty = 1.6439\pm0.0002`$ from the $`D=\infty`$ equations alone. The $`\pm0.0002`$ is an estimator
   bracket (row 13).
8. The AI-production disclosure is present. Good.

**Missing from the abstract:**
- the radius-exponent convention for $`\gamma`$;
- that the large-$`D`$ branch is assumed to remain the critical solution;
- credit to Rozali–Way for the $`D=\infty`$ inner equations;
- the scope of "excluded".

## 4. Constant recognition: the closed-form statements for $`\Delta_\infty`$

**The statements audited:**
- Paper intro item 2: "nine candidate closed forms for $`\Delta_\infty`$ are ruled out".
- Sec. "Candidate closed forms":
  - "The closest sits $`1.3\times10^{-5}`$ from the limit, more than 1000 times its uncertainty";
  - "A search over products of standard constants and Bessel zeros finds nothing in $`1.6438936`$–$`1.6438939`$, nor
    in a window about 27 times wider";
  - "Whether $`\Delta_\infty`$ has a closed form is open".
- README "Candidate closed forms, all ruled out": the same statements, plus the f64-era window search.
- Discussion: "(and possibly a closed form)".

**What was computed.** The script was
`/tmp/claude-0/-home-claude/5cec0889-b97c-529a-89c7-507003649638/scratchpad/cr/cr.py` (mpmath at 30 digits). It
re-enumerates exactly the product family of [solver/candidates.py](../solver/candidates.py):
- $`r\,b_1^{p_1}b_2^{p_2}`$, $`2\pi/(r b)`$ and $`r\ln b`$;
- $`r = i/j`$, with $`i, j \le 12`$ coprime;
- $`p\in\{\pm\tfrac12,\pm1,\pm2\}`$;
- 18 bases.

Results:

| Quantity | Value |
|---|---|
| Family size $`N`$ | 514 332 expressions ($`\log_{10}N = 5.71`$) |
| Local density near $`\Delta_\infty`$ (from $`\pm0.01`$) | about $`5.8\times10^4`$ per unit |
| f64-era window $`[1.64388, 1.64403]`$ (width $`1.5\times10^{-4}`$) | 5 found; about 8.7 expected by chance |
| Double-double window $`[1.6438936, 1.6438939]`$ (width $`3\times10^{-7}`$) | 0 found; 0.017 expected |
| "27 times wider" window $`[1.643890, 1.643898]`$ | 0 found; 0.46 expected |
| $`\pm1.5\times10^{-8}`$ around $`\Delta_\infty`$ | 0 found; 0.0017 expected |
| Nearest family members | $`+835`$ and $`-1806`$ error bars away |

Distances of the nine candidates from $`\Delta_\infty = 1.64389375`$, in units of the stated $`1.5\times10^{-8}`$
(and of the README's older $`10^{-8}`$):

| Candidate | value − $`\Delta_\infty`$ | ÷ $`1.5\times10^{-8}`$ | ÷ $`10^{-8}`$ |
|---|---|---|---|
| $`\tfrac{10}{9}\sqrt{j_{0,1}/\ln3}`$ | $`+1.253\times10^{-5}`$ | **835** | 1253 |
| $`\tfrac57\sqrt G\,j_{0,1}`$ | $`+8.09\times10^{-5}`$ | 5395 | 8093 |
| $`\tfrac75(\ln2)^{-2}\pi^{-1/2}`$ | $`+1.076\times10^{-4}`$ | 7175 | 10760 |
| $`\tfrac72\sqrt{\ln2/\pi}`$ | $`+1.215\times10^{-4}`$ | 8099 | 12150 |
| $`\tfrac98\sqrt{j_{2,1}/j_{0,1}}`$ | $`+1.269\times10^{-4}`$ | 8461 | 12690 |
| $`\pi^2/6`$ | $`+1.04\times10^{-3}`$ | $`6.9\times10^4`$ | $`1.0\times10^5`$ |
| $`2\pi/j_{1,1}`$ | $`-4.11\times10^{-3}`$ | $`2.7\times10^5`$ | $`4.1\times10^5`$ |
| $`\sqrt e`$ | $`+4.83\times10^{-3}`$ | $`3.2\times10^5`$ | $`4.8\times10^5`$ |
| $`(1+\sqrt5)/2`$ | $`-2.59\times10^{-2}`$ | $`1.7\times10^6`$ | $`2.6\times10^6`$ |

**Verdicts on the wording:**

- **"more than 1000 times its uncertainty": wrong.** It is true only for the README's older $`\pm10^{-8}`$ (1253×). With
  the paper's and the README headline's $`\pm1.5\times10^{-8}`$ it is 835×. The README still shows
  $`\pm0.00000001`$ in two places: the display equation in "The formula" and the "none (free fit)" row of the candidate
  table. Those disagree with $`\pm1.5\times10^{-8}`$ elsewhere. Proposed: "about 840 times its uncertainty", and use
  $`\pm1.5\times10^{-8}`$ throughout.
- **"nine candidate closed forms ruled out": justified as stated in the body, with two qualifications.**
  - *They were already excluded.* Every one of the nine lies at least 835 error bars from the free-fit limit. The
    forced-fit held-out test is therefore a second, independent exclusion, and it is the one that does not depend on
    the error bar's honesty. Both rest on analyticity in $`1/D`$, which the intro item omits.
  - *They have no standing as candidates.* The protocol calls this the "lookup-table shortcut". Five of the nine are
    window hits of a $`5\times10^5`$-expression search placed around the f64 estimate, where about 9 hits are expected
    by chance, and the other four are famous constants chosen by hand. Ruling them out is correct but tells us
    nothing about whether $`\Delta_\infty`$ has a closed form.
  - Proposed wording: "Five constants that a product search placed inside the f64-era window (about nine such
    coincidences are expected by chance), and four classical constants, are all excluded: each lies at least 800
    times the uncertainty from the limit, and forcing any of them as the limit worsens the held-out prediction at
    least 100-fold."
- **"A search ... finds nothing in $`1.6438936`$–$`1.6438939`$, nor in a window 27 times wider": true, but powerless
  as stated.**
  - *Expected chance counts.* At this family's density they are 0.017 and 0.46, so the null says only that
    $`\Delta_\infty`$ is not one of these 514 332 expressions.
  - *Digit budget.* The family consumes about 5.7 digits. $`\Delta_\infty`$ is known to about 8, so the surplus is
    about 2 digits, far below the protocol's "many tens". Even a hit in the $`\pm1.5\times10^{-8}`$ window (expected
    0.0017) would have been a weak result, not a recognition.
  - *No controls.* The search has no positive control (a known constant recovered through the same code) and no
    negative control (a random number). [solver/scratch/closed_form.py](../solver/scratch/closed_form.py) has a random
    control, but for $`D = 4`$ and with PSLQ.
  - Proposed wording: "The constant is not a product of two of the listed constants with rational prefactor ($`i/j`$,
    $`i,j\le12`$) and exponents $`\pm\tfrac12,\pm1,\pm2`$ (514 332 expressions, about 0.02 expected by chance in the
    window); at eight known digits no integer-relation search could support a closed form."
- **"Whether $`\Delta_\infty`$ has a closed form is open": justified.** It is the protocol's honourable null.
  Recognition should wait until the $`D=\infty`$ selection rule delivers $`\omega_\infty`$ to 100+ digits; then a
  declared ring, a height bound, controls and held-out digits apply. notes/BOOTLOOPS.md §5 makes the same point.
- **The README heading "Candidate closed forms, all ruled out".** It suggests the candidates were serious. Proposed:
  "Closed forms tested and excluded".

## 5. Cross-checks from the protocol

- **Headline-number units:**
  - $`\gamma`$: mass versus radius convention (row 1).
  - $`\Delta_\infty`$ uncertainty: two values in the README.
  - "1000 times": convention-dependent and wrong for the stated error bar (row 18).
- **Claims against the project's own files:**
  - "more than 1000 times" contradicts the paper's own $`\pm1.5\times10^{-8}`$.
  - The README's single-unstable-mode caveat (§"What the numbers mean physically") is absent from the paper.
  - Kummer (README) versus Weber/Hermite (paper) for the linear inner equation is not a contradiction: parabolic
    cylinder functions are Kummer functions.
- **Necessity language.**
  - "The non-analytic piece must come from a Stokes phenomenon" (after Prop. 3) rests on consequence (iii), which
    holds at every order of the WKB expansion. It is fine as an inference, but it is not a theorem.
  - "analyticity at the light cone forces $`C = 0`$" is fine.
- **Null hypothesis.**
  - The analyticity test has one: the half-integer series and the added non-analytic terms (Table 4, and "Adding one
    non-analytic term ... changes neither the prediction nor the limit"). Good.
  - The $`\Delta\to0`$ exclusion's null is "power law with $`p\ge0.01`$". State it in the abstract.
  - The closed-form search had none (§4).
- **Temporal honesty.** $`\lvert F_3/F_1\rvert = 0.0551`$ is called a prediction that "the data reproduce". If the
  extrapolated 0.055 from the solutions was known before the inner computation, say "agrees with" rather than
  "predicts". I could not determine the order of events from the files read.
- **Conjecture labelling at title and abstract level.**
  - The title's "evidence for a finite limit" is properly hedged.
  - "selection beyond all orders in $`1/D`$" in the title is stated as fact but holds within the ansatz (row 10).

## 6. Prioritised changes (proposals; nothing applied)

1. **Fix "more than 1000 times its uncertainty"** to "about 840 times".
   - Where: paper Sec. "Candidate closed forms"; README "Candidate closed forms".
   - Also make the README's $`\Delta_\infty`$ uncertainty $`\pm1.5\times10^{-8}`$ everywhere (two places still say
     $`\pm0.00000001`$).
   - Why: rows 18 and 3; the project's own number contradicts the sentence.
2. **Credit Rozali and Way for the inner equations and the heat-equation form.**
   - Where: Sec. inner, first paragraph; Prop. 5(i); README "What is new here"; GAMMA.md §2–3.
   - Write out and verify the variable map to their Eqs. (9a)–(9c).
   - Why: row 9; LIT_REVIEW.md C4–C5.
3. **State the $`\gamma`$ convention** in the intro: $`r_{\rm BH}\propto(p-p^*)^\gamma`$, equivalently
   $`M\propto(p-p^*)^{(D-3)\gamma}`$, with $`\gamma = 1/\lambda`$. Why: row 1.
4. **State that the large-$`D`$ branch is assumed to be the critical solution with one unstable mode.**
   - Only one growth rate is computed, and there are no evolutions above $`D = 14`$.
   - Cite Clark–Pimentel (JHEP 11 (2025) 017) for the large-$`D`$ CSS critical solutions as the alternative.
   - List it under "Open" in the Discussion.
   - Why: row 2.
5. **Scope the abstract.**
   - "Within the scaling measured on the solutions, the $`1/D`$ expansion ... can be carried to all orders" (rows 10
     and 12).
   - "A power-law approach to $`\Delta = 0`$ with exponent $`p\ge0.01`$ ... is excluded" (row 22).
   - "which, under Conjecture 1 and uniqueness of the unstable mode, we explain" (row 23).
   - "$`(1.0000\pm0.0001)\,\omega_\infty`$ (range of estimators)" (row 13).
6. **Fix the literature sentences in the intro.**
   - "for integer $`D = 4`$–$`14`$" becomes "$`D = 3.5`$–$`14`$ (mostly integer)", and add Taves–Kunstatter (row 5).
   - Attribute $`\gamma\to1/2`$ to Rozali–Way, as reported in EEG and EEGJ, and cite Clark–Pimentel (row 8).
   - Cite Lai & Wen (arXiv:2610.04960) in Sec. eeg (row 6).
7. **Cite Reiterer–Trubowitz (CMP 368, 143 (2019))** where the computer-assisted certificate is described, and
   Baldomá–Capiński–Guardia–Seara (JNS 33 (2023)) as the template. Correct PROOF.md's "first rigorous derivation of a
   critical-collapse echo period" (row 17; LIT_REVIEW.md C10).
8. **Reword the closed-form passage** as in §4: the family, height and expected chance count for the null; "excluded"
   with the analyticity conditional in intro item 2; drop "candidate" for the five window hits; README heading
   "Closed forms tested and excluded". Rows 19–20.
9. **Describe $`\pm1.5\times10^{-8}`$ as a spread over model choices,** and run (or list as owed) a planted-truth test
   of `hp/dinf_budget.py`. Row 3.
10. **Scope the README's novelty sentences** as proposed in LIT_REVIEW.md §2 ("not found in the critical-collapse or
    large-$`D`$ literature searched"; Prop. 4 as an explicit instance of standard late-order analysis; Prop. 5 as the
    DSS counterpart of RW's heuristic, conditional on Conjecture 1). Rows 11 and 14.
11. **Say in the paper that the formula's $`a_0`$ is not $`\Delta_\infty`$** (it is $`4\times10^{-6}`$ high). Row 21.
12. **Say whether $`\lvert F_3/F_1\rvert = 0.0551`$ was computed blind** to the extrapolated value; if not, write
    "agrees with" instead of "predicts". §5.

Already done in this audit (REF_CHECK.md): the DLMF citation for Weber's Stokes constant (§12.8 replaced by §12.9),
the Rozali–Way journal and title, the review's title, Donninger's journal, and the Bland thesis title.

## 7. Top unanswered objections

The five most serious objections the paper does not yet answer:

1. Row 2: whether the DSS branch is still the critical solution with a single unstable mode at large $`D`$ (only one
   growth rate is computed).
2. Row 9: the inner equations, and their heat form, are Rozali–Way's leading-order system, uncredited.
3. Row 1: the $`\gamma`$ convention is mass in the intro, radius in the result.
4. Row 18: "more than 1000 times its uncertainty" is 835× at the stated error bar.
5. Rows 10 and 12: the abstract's "selected beyond all orders" is unscoped (ansatz and (H2)).
