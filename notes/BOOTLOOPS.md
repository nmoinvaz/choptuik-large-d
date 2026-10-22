# BootLoops: what it is, and what it offers this project

An assessment of [github.com/BootLoops-ai/bootloops](https://github.com/BootLoops-ai/bootloops), written 2026-10-21 from a
read-only shallow clone of commit [`e16ba28`][commit] (2026-10-08, "cutbasis: … (ships with 1.0.1)"). Nothing from the
repository was run, built, imported or installed. The only code run was a python-flint check on our own dependency
(§3, item 4) and two small file-scanning scripts of my own. Everything below about BootLoops comes from its own files,
linked at that commit.

**In one line:** a large, MIT-licensed toolkit of certified-arithmetic and Feynman-integral instruments, written by
Claude under a physicist's direction and documented for agents to drive. It has nothing for our physics or our
multi-agent setup. What it does have is a written-down verification discipline, and three of its rules expose gaps in
our proof certificate, our $`\Delta_\infty`$ error bar and our README numbers.

## 1. What it is

- **Purpose.** "BootLoops 1.0 is a harness for large language models doing precision quantitative science": an
  extensive scientific software package "written, ported, and upgraded to be driven by an LLM", together with "the
  working protocols that make the model's results checkable" ([README][readme]). Each instrument is documented with
  what it does, when to use it, what its output means, and "what test its answer must pass before anyone believes it".
- **Authors.** "Created by Matthew D. Schwartz; code written by Claude (Anthropic) under his supervision". The author
  "set every problem, approved every plan, and checked the reported results against independent routes". Copyright
  Anthropic, PBC. It is "maintained by Matthew D. Schwartz, not by Anthropic. It is not an officially supported
  Anthropic product" ([README][readme], [LICENSE][license]). The results and papers are on bootloops.ai, not in the
  repository.
- **Organisation.** There are six sibling repositories under BootLoops-ai ([README][readme], "Related repositories"):
  - `bootloops`: this one, the toolkit;
  - `skills`: the protocols, as Agent Skills / Claude Code plugins;
  - `jackandjill`: Bayesian phylogenetics;
  - patched forks of Kira (GPL), Blade and AMFlow.cpp.
- **Stack and size.** About 250k lines of Python (973 files) and 43k lines of Julia (160 files), plus 25k lines of
  Markdown. The Python core is plain 3.12 with mpmath, sympy, numpy and python-flint (Arb). External engines are
  installed separately: Kira, FireFly, FORM, AMFlow, msolve, Singular, OSCAR, and SageMath with ore_algebra
  ([INSTALL.md][install], [toolkit/external/TOOLS.md][tools-ext]).
- **Contents.**
  - **50 packages under `tools/`** ([index][tools-readme]). Most of the machinery is for Feynman integrals:
    integration-by-parts reduction, Landau singularities, ε-graded differential-equation transport, elliptic and
    Calabi–Yau periods. A general layer comes with them:
    - ball-arithmetic certification ([BALLER][baller], [ERAS][eras], [Clinch][clinch]);
    - certified quadrature ([Nestor][nestor-dir]);
    - certified analytic continuation of linear ODEs ([Holonomic][holonomic], a layer over ore_algebra);
    - integer-relation detection ([Lockpick][lockpick]);
    - held-out cross-validation gates ([Gatekeeper][gatekeeper]);
    - a claims-integrity checker ([Emitall][emitall]).
  - **Three in-house engines** under `upgrades/`: SOFIA.jl, Eichler.jl and Leviathan.
  - **One operations package**, [Turnstile][turnstile] (admission control for long jobs on a shared machine).
  - **Recipes** ([RECIPES.md][recipes]).
- **Maturity.**
  - **Version:** 1.0.1, 2026-10-09 ([CHANGELOG][changelog]).
  - **History:** one squashed commit, about two weeks old; 4 stars when I read the GitHub page.
  - **Users:** the changelog credits outside users with bug reports and a fix (issues #2 and #3, and a separate
    feedback repository).
  - **Self-tests:** every package declares a verification class (selftest, partial, smoke, data-gated) and ships an
    acceptance battery, run by [`run_selftests.py`][selftests]. Data-gated packages refuse with exit code 4 and a printed
    reason.
  - **Tested on:** Linux x86_64 and arm64; not on macOS.
  - **Not done here:** I ran none of it, so none of these claims is checked.
- **Main ideas** (beyond the instruments):
  1. **Tools documented for an agent.** The GUIDEs follow a common layout of PURPOSE, USE-WHEN, NOT-FOR, INVOKE, GATES
     and FOOTGUNS. The FOOTGUNS sections record measured failures.
  2. **Fail-closed acceptance:**
     - an independent route, compared at points no fit saw;
     - planted positive controls that prove a check can fail;
     - refusals that name their reason;
     - "banked" reference values and a single "value of record";
     - receipts and sha-pinned files;
     - verdicts written by scripts, never typed ("Verdicts come from CLOSE_*.json, never from driver logs",
       [Lockpick][lockpick]).
  3. **"Every problem it meets leaves tools behind for the next one."**

## 2. Parallels, and where there are none

- **Same way of working, different shape.** Both projects are Claude working under one scientist's direction, and both
  disclose it: their "Status, provenance and responsible use" ([README][readme]) corresponds to our
  [How this work was done](../README.md#how-this-work-was-done).
  - **BootLoops** is one agent, a toolkit and protocols, with the human approving a plan before anything runs.
  - **We** coordinate cloud and local sessions through [`comms.txt`](../comms.txt), agents in worktrees, and
    [AGENTS.md](../AGENTS.md) conventions.
  - BootLoops has no inter-agent channel, no cloud/local split and no orchestration. Turnstile is the only overlap
    (item 6).
- **Shared verification habits.** Several of their protocols are things we already do:
  - **Held-out points:** the hollow points in the formula fit, and the 351 unseen points in the double-double candidate
    test.
  - **Independent routes:** two solvers; two machines.
  - **Fresh-agent checks:** re-derivation by agents that did not write the derivation, and a referee pass.
  - **Timing discipline:** pricing a run from a small one first, as for T28's `J^2.5` estimate.
- **No overlap in the mathematics or physics.** The repository has nothing on:
  - Stokes constants, Borel summation or resurgence;
  - validated integration of nonlinear ODEs (no Lohner method);
  - Newton–Kantorovich proofs for boundary-value problems;
  - double-double arithmetic.

  Its certified ODE transport is for linear operators only (holonomic, Picard–Fuchs, Fuchsian), with regular-singular
  landings ([Holonomic][holonomic], [Eichler.jl][eichler]). Our inner problem is nonlinear with an irregular singular
  point, so no instrument there computes or certifies any of our quantities.

## 3. Opportunities, ranked by value

Items 1, 2 and 4 are concrete and cheap. Item 3 is moderate. Items 5–7 are minor or depend on future work.

1. **Mutation controls and an independent second implementation for the part-A certificate.** Highest value: it is on
   the critical path of [PROOF.md](../PROOF.md) stage 2.
   - **Our current state.** [proof/A_certify.md](../proof/A_certify.md) §9 certifies the one-harmonic zero with a
     Krawczyk test over $`X = \hat x + [-10^{-9}, 10^{-9}]^2`$. Its checks are positive ones: the linear problem against
     Kummer U ([`lin_check.py`](../solver/proof/lin_check.py)), and agreement with double precision. It states that
     "the certificate rests on the correctness of the code, which no second implementation has checked yet".
   - **What BootLoops says.** The BALLER manual records a limit of the Krawczyk test ([MANUAL.md, lines 290–299][baller-manual]):
     the centred contraction test is blind to the sign of c, and "any invertible claimed-H self-preconditions to E ~ 0
     near a true zero". A mutation that changes only the Jacobian enclosure therefore proves nothing. "Meaningful
     mutation controls flip signs in the ORACLE MATH (zero moves -> Krawczyk refuses)".
   - **The independence rule.** Holonomic's "independence fence" ([GUIDE][holonomic]) says an oracle and the evaluator
     it checks must share no code. Otherwise every cross-check grades the code against itself.
   - **Adoption:**
     - **(a) Mutation controls (minutes).** Add two or three to [`validated_int.py`](../solver/proof/validated_int.py)'s
       `krawczyk` mode, each a change to the function rather than to J. Examples: shift ω in the right-hand side by a
       few h, flip the sign of one coupling term, or centre the box 3h off the zero. Record that each one makes K(X) leave
       int X. Each box run at h = 10⁻⁹ takes about 4 minutes.
     - **(b) A second validated integrator (agent-days).** It must share no code with `validated_int.py` or
       `series_arb.py`. A C⁰ Taylor-model or interval-Picard integrator over the same path is enough at h = 10⁻⁹.
2. **A planted-truth test of the $`\Delta_\infty`$ error bar.** High value, low cost.
   - **The gap.** $`\Delta_\infty = 1.64389375 \pm 1.5\times10^{-8}`$ is the spread of 48 fit choices
     ([`hp/dinf_budget.py`](../hp/dinf_budget.py); README, *Is the period analytic in 1/D?*), backed by held-out
     prediction. Neither `hp/` nor `solver/` has a planted-truth test: no script runs synthetic data with a known
     limit through the same pipeline.
   - **The BootLoops rule.** "Planted-truth controls that recover a known answer before any real data is touched"
     ([README][readme]).
   - **Adoption.** Build synthetic $`\Delta(D)`$ on the real D grid, with each point's estimated noise, from known
     functions. Use one that is analytic in 1/D, and variants with a small non-analytic term ($`D^{-1/2}`$,
     $`\ln D/D^2`$) or an exponentially small one. Run them through `dinf_budget.py` unchanged, and check that the
     planted limit falls inside the quoted bar as often as the bar claims. This tests whether ±1.5×10⁻⁸ is an honest
     uncertainty or only a spread of choices. It uses only fits, so it costs minutes.
3. **Automated checks of quoted numbers (the Emitall pattern).** Medium-high value as agents keep editing the README.
   - **How Emitall works** ([GUIDE][emitall]). A project writes one declarative spec that maps each quoted headline
     number to a stored receipt, a field and a comparison mode. Modes include printed precision and outward rounding of
     error-bar edges. The checker re-emits every value and fails on any drift.
   - **Why it fits us.** Our README and paper quote hundreds of numbers, and many agents edit them. Most of the numbers
     already sit in JSON:
     - `sigma_err` in the amplitude scans;
     - [`validated_int_krawczyk_1e-9.json`](../solver/proof/validated_int_krawczyk_1e-9.json);
     - the fit outputs.
   - **Adoption.** Write our own checker of about 100 lines on the same design, covering about 30 headline claims:
     - $`\Delta_\infty`$, $`\omega_\infty`$ and the $`\gamma_\infty`$ precision;
     - the fit residuals and held-out errors;
     - the Krawczyk margins;
     - F₃/F₁, and the M = 3 and M = 5 values of Δ.

     Writing our own keeps the untrusted code out. The tex comment-swallow gate (`paper_seams.py`) is not worth it here:
     [paper/main.tex](../paper/main.tex) has 7 comment lines.
4. **A python-flint footgun, verified here.** Medium value, trivial cost.
   - **The claim.** The BALLER manual ([lines 286–289][baller-manual]) says `x**n` on a zero-containing arb ball returns
     NaN.
   - **What I measured** with our python-flint 0.9.0 (the manual says 0.8.0):
     - `arb(0, r)**k` is `nan` for any r > 0 and k = 2 or 3;
     - `arb(0, r)*arb(0, r)` is finite;
     - with a nonzero midpoint, or for `acb`, it is fine.
   - **Our code is not affected.** I checked the `**` uses in [`solver/proof/`](../solver/proof/). They act on `acb`
     ([`series_arb.py`](../solver/proof/series_arb.py), [`krawczyk.py`](../solver/proof/krawczyk.py)) or on exact
     nonzero balls such as `arb(r0) ** 2` ([`certify_M1.py`](../solver/proof/certify_M1.py),
     [`validated_int.py`](../solver/proof/validated_int.py), [`fixed_point.py`](../solver/proof/fixed_point.py)).
   - **Adoption.** One line in AGENTS.md: in arb code, write `x*x` instead of `x**2` for real balls that may be centred
     at 0. The same manual's dps_lint rules make a useful review checklist for our mpmath scripts: constants frozen at
     import-time precision, and imports that reset `mp.dps`. I did not run the linter.
5. **Integer-relation discipline for closed-form searches.** Low value now; high later if the selection rule is
   proved.
   - **What we do.** [`solver/scratch/closed_form.py`](../solver/scratch/closed_form.py) uses raw `mp.pslq` on about
     9 digits, with a random-number control. The README's candidate search is a product search over a declared list of
     constants ([`solver/candidates.py`](../solver/candidates.py)).
   - **The BootLoops discipline** ([Lockpick GUIDE][lockpick], [RECIPES][recipes] "Weight-graded PSLQ closure
     discipline"):
     - declare and freeze the basket of constants before the target exists;
     - plant positive controls inside the basket;
     - require stability at two precisions;
     - refuse scans whose coefficient height exceeds what the digits support;
     - read a null as excluding "ONLY the frozen basket at the printed height";
     - the sealed runner refuses targets under 150 digits.
   - **What it means for us.** At 8 digits for $`\Delta_\infty`$, no integer-relation verdict is meaningful, and the
     README's "whether Δ∞ has a closed form at all is open" is the right wording.
   - **Adoption now.** State the expected number of chance hits in the search window, so the scope of the null is
     explicit.
   - **Later.** If the D = ∞ selection rule gives $`\omega_\infty`$ to 100+ digits ([PROOF.md](../PROOF.md): "from which
     Δ∞ would follow to any precision"), a sealed closure test in this style is the right instrument.
6. **A job ledger for the local machine (Turnstile).** Low to medium value.
   - **What it does** ([GUIDE][turnstile]). Every long job declares its estimated memory (`--est-rss`) and threads
     (`--est-threads`) and passes a priority token. An append-only ledger records estimated against measured peak
     memory, wall time and exit code.
   - **The parallel.** It mechanises rules our comms apply by hand:
     - run 3 ω at once only if free RAM is at least 3 GB;
     - the 5 h rule;
     - 4–6 threads at below-normal priority;
     - "workers used ~0.9 GB each, expect ~1.5 GB at J 1000".
   - **Why it does not fit as is.** Turnstile is bash with `flock` and `taskset` and is Linux-only. Our local machine
     runs Windows (AGENTS.md: `taskkill`, spawn).
   - **Adoption.** Take the idea: one ledger line per launch, giving the estimate and the measured peak. That could be a
     short Python wrapper, or a convention in comms.
7. **The `skills` repository.** Unassessed; possibly the most relevant part for our agent workflow.
   - **Contents** ([README][readme], [INSTALL.md][install]). The protocols themselves: acceptance gates, planted truth,
     provenance bookkeeping, "simulating referees, auditing the literature behind a novelty claim, verifying
     bibliographies".
   - **Where they would apply here:**
     - our referee pass;
     - [PROOF.md](../PROOF.md)'s citations marked *(from memory)*, which it says "should be checked against the
       published records";
     - its novelty claim ("the first rigorous derivation of a critical-collapse echo period").
   - **Not read.** The fetch of github.com/BootLoops-ai/skills was blocked by the tool's provenance rule. Reading it is
     the natural follow-up, with the same read-only care.

**The other direction (the user's call; nothing was done):** our QR-Lohner validated integrator and Krawczyk assembly in
python-flint fill a gap BootLoops lacks. BALLER lists "Taylor-form enclosures" under NOT-FOR, and the toolkit has no
validated integrator for nonlinear ODEs. The README invites contributions.

## 4. Risks

- **Licence.**
  - **Main licence:** MIT (Copyright Anthropic, PBC), so copied code must carry its notice. Prose is CC BY 4.0
    ([LICENSE-CONTENT][license-content]).
  - **Three GPL files:**
    - `tools/subtropica/src/lr_refine.jl`, GPL-3.0-or-later;
    - `tools/eichler/genus2/mestre_port.py`, GPL-2.0-or-later;
    - `tools/formglue/form_hyper.py`, GPL-3.0-only.

    ([THIRD_PARTY.md][third-party])
  - **GPL elsewhere:** the Kira fork is GPL, and Holonomic drives GPL SageMath.
  - **Our side:** this repository has no LICENSE file. That needs deciding before any code is vendored. Items 1–6 above
    need no code from BootLoops.
- **Security.** By its own account the toolkit evaluates its inputs: "Many of them evaluate the contents of their input
  files (JSON, YAML, `.m`, `.ms`, `.jl`, pickle and similar), so a file received from someone else can run arbitrary
  commands" ([README][readme], "Security considerations"). In the tree:
  - **Pickles:** four ship (`tools/baller/vendor/lgf_monodromy/BC_cache.pkl` and three 1 MB files in
    `tools/surd/gate/cells/`). Loading a pickle runs code.
  - **Subprocesses and dynamic code** (grep counts, not audited): about 135 Python files call `subprocess`, and about
    35 contain `eval(` or `exec(` calls.
  - **The self-test runner** executes every package's battery.
  - **Installs by the agent:** the README suggests letting the agent install whatever engine a self-test says is
    missing.
  - **Network:** the one network call I found is in the vendored SubTropica reference (`SubTropica.wl` fetches arXiv
    PDFs), not in BootLoops' own Python.
  - **No executables** ship (checked by file signature).

  If anything is ever run, it belongs in a container, never on the local machine that runs our long jobs.
- **Maturity and size.** It is two weeks public, with one squashed commit and a small user base. At about 300k lines it
  is too large for us to vet, and its verification is the author's own, by the README. Take ideas from it, not
  dependencies.
- **Domain mismatch.** Most of the code serves Feynman-integral work, so the instruments themselves offer little; the
  value is in the discipline.

**Text addressed to AI agents.** The whole repository is written for agents to read: every GUIDE is a set of
instructions to the agent driving the tool. I found no prompt injection, meaning no text that tries to redirect an
agent away from the user's task. I searched for override phrases, HTML comments in Markdown, and invisible or
bidirectional Unicode. The passages addressed to agents or their users, none of which I acted on:

- [README][readme], "Using it": the prompt a user is told to give their agent: "Read `tools/README.md` and the tool
  guides it points to, then propose a plan for this problem for me to approve."
- [README][readme], "External engines": "A practical route is to let the model handle it: *'the selftest says FORM is
  missing; install what it asks for.'*"
- [INSTALL.md][install], "The skills": "run `/bootloops-setup` to pick which skills to activate", or
  "`/plugin marketplace add BootLoops-ai/skills` in Claude Code". These are commands for the user to run, and the
  skills "arrive inert on every route".

## 5. Recommendation

Do not adopt BootLoops as a dependency. Its instruments serve Feynman-integral and statistics problems, not
critical collapse. It is large, new and self-verified, and it executes its inputs by design. It is worth borrowing from
as a written-down verification discipline from a project run the same way as ours, and three of its rules expose gaps
we should close cheaply.

1. **Part-A certificate.** Add mutation controls that change the function, not just J (minutes), and plan an
   independent second validated integrator. Our certificate currently rests on one unchecked code path.
2. **$`\Delta_\infty`$ error bar.** Run a planted-truth test through `hp/dinf_budget.py` to check that ±1.5×10⁻⁸ is an
   honest uncertainty.
3. **Quoted numbers.** Write a small claims-integrity checker that ties the README's headline numbers to their JSON
   receipts.

Also worth doing:

- Add the python-flint `arb(0, r)**k` NaN footgun to AGENTS.md.
- Once a closed-form search ever has 100+ digits, use the sealed integer-relation discipline for it.
- Read the sibling `skills` repository next, read-only. Its referee, literature-audit and bibliography protocols match
  open items in PROOF.md.

[commit]: https://github.com/BootLoops-ai/bootloops/tree/e16ba2820a6cdfffd0f0ce970163b867c554db41
[readme]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/README.md
[license]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/LICENSE
[license-content]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/LICENSE-CONTENT
[third-party]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/THIRD_PARTY.md
[install]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/INSTALL.md
[changelog]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/CHANGELOG.md
[selftests]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/run_selftests.py
[tools-readme]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/tools/README.md
[tools-ext]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/toolkit/external/TOOLS.md
[recipes]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/toolkit/ours/RECIPES.md
[baller]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/tools/baller/GUIDE.md
[baller-manual]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/tools/baller/MANUAL.md
[eras]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/tools/eras/GUIDE.md
[clinch]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/tools/clinch/GUIDE.md
[nestor-dir]: https://github.com/BootLoops-ai/bootloops/tree/e16ba2820a6cdfffd0f0ce970163b867c554db41/tools/nestor
[holonomic]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/tools/holonomic/GUIDE.md
[lockpick]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/tools/lockpick/GUIDE.md
[gatekeeper]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/tools/gatekeeper/GUIDE.md
[emitall]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/tools/emitall/GUIDE.md
[turnstile]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/ops/turnstile/GUIDE.md
[eichler]: https://github.com/BootLoops-ai/bootloops/blob/e16ba2820a6cdfffd0f0ce970163b867c554db41/upgrades/Eichler.jl/README.md
