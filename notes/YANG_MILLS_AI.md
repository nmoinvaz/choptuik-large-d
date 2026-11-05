# Yang–Mills results made with AI, 2024–2026, and what they mean for this project

Survey of 2026-11-05, branch `wip/yang-mills-ai`. Written by Claude (Anthropic) under Nathan Moinvaziri's direction and
not reviewed. No solver was run and no other file was changed. Status words follow AGENTS.md.

**How the sources were read.** Full texts were downloaded from arXiv and read as text extracted with pypdf. Abstracts
and journal references come from the arXiv API and INSPIRE, and web pages were read through WebFetch, which returns a
summary of the page. Each entry says which of these it rests on. Quotes are copied from the extracted text. No fetched
page contained text addressed to an AI agent.

## The answer

**Nothing found changes the Yang–Mills row of the README or the sentence "we know of no large-D study of critical
collapse for any matter other than the real scalar".** No AI-made or AI-assisted result concerns Einstein–Yang–Mills
collapse in any dimension. No study of Einstein–Yang–Mills collapse in D ≥ 5, with or without AI, was found.

Most AI results on Yang–Mills in this period concern the quantum theory. OpenAI claims an existence proof for quantum
Yang–Mills in four dimensions (October 2026, not released), Alpoge and Sussman post LLM-written constructive drafts, an
amplitude formula was conjectured by GPT-5.2, and machine learning is used for lattice gauge theory and amplitudes.
None of this touches classical critical collapse.

The one AI-assisted result on classical Yang–Mills blowup is a computation by Bizoń and Kluszczyński (October 2026).
They used "AI-assisted exact root-counting computations" to show that flat-space Yang–Mills has exactly three
self-similar profiles, up to reflection, in every odd space dimension d from 11 to 101. It belongs to a 2026 series by
Bizoń and coworkers on self-similar flat Yang–Mills blowup in high odd d, and the rest of the series uses no AI. The
series has no gravity and takes no limit d → ∞, so the README sentence, which is about critical collapse, still holds.
It affects the matter notes in three places.

1. **Out of date.** [matter/LITERATURE.md](../matter/LITERATURE.md) §5.3 and §9 say that no large-d study of Yang–Mills
   blowup was found, and §4.2 gives the flat-space landscape for d = 5, 7, 9 only. Bizoń, Glogić and Wasserman (Phys.
   Rev. D 2026) reduce the self-similar profiles in every odd d ≥ 11 to the roots of a polynomial, and a follow-up
   proves there are at least three, with exact counts of three up to d = 101. Their stated motivation is that threshold
   behaviour "may simplify as the dimension increases". None of them takes d → ∞.
2. **A sentence that is too strong.** [matter/STRUCTURE.md](../matter/STRUCTURE.md) §5(e) says that with gravity the
   flat blowup "ends in black holes with a mass gap; it is not a threshold phenomenon". That holds for the generic
   blowup. The flat critical solutions, which separate blowup from dispersal (W₁ in d = 5, and in odd d ≥ 11 whichever
   of u₋ and u* has one unstable mode), would by the same argument give a black-hole threshold with a mass gap. That is
   a further kind of threshold for some families, not a rival of the type II solution, so the row's type II prediction
   stands. *(Inferred, at the level of the notes' own argument. Already implicit in d = 5, made sharper by the 2026
   work.)*
3. **A precedent against "nothing suggests one".** The row's "not fixed by any symmetry" rests on a search for Lie point
   symmetries of bounded degree. STRUCTURE.md §10 notes that a non-point symmetry is "not excluded, though nothing
   suggests one". Bizoń (November 2026) explains an exact eigenvalue λ = 5, shared by every excited self-similar
   Yang–Mills profile in d = 5, by an explicit eigenfunction built from the profile, not by a Lie point symmetry. Such
   a search would not see that mode. The γ∞ conjecture is unchanged, but the words "nothing suggests one" should go.

On methods, the AI solvers for unstable self-similar singularities (Google DeepMind's 2025 work on the porous-media
and Boussinesq equations and its successors) and the 2026 physics-informed networks for Choptuik collapse reach double
precision at best. This project's Newton solver already finds its solutions in double-double and quad-double, so they
offer nothing here. The computer-assisted proofs are unaffected. The flat Yang–Mills family in high odd d may be a cheap
test of the D = ∞ inner selection (a lead, §4).

| # | Work | AI role | Status | Read | Bearing |
|---|---|---|---|---|---|
| A1 | OpenAI, quantum Yang–Mills existence (Oct 2026) | LLM proof, Lean formalization | announcement, unreleased | committee statement, full | none |
| A2 | Alpoge, Sussman, LLM constructive drafts | Claude writes, Codex condenses | unvetted drafts on GitHub | README | none |
| A3 | Guevara et al., single-minus gluon amplitudes | GPT-5.2 conjectured the formula | preprint | full text, parts | none |
| A4 | ML for lattice Yang–Mills, amplitudes, HYM connections | networks as solvers or samplers | journals 2024–2026 | abstracts or metadata | none |
| A5 | Bizoń, Kluszczyński, third profile in odd d ≥ 11 | AI-assisted exact root counting | preprint | full text | (b) notes, (c) lead |
| A6 | Wang et al. and successors, unstable singularities | networks with Gauss–Newton | preprints | abstracts | (c) none |
| A7 | Ferrer-Sánchez et al., networks for scalar collapse | networks as solvers | journal 2026 | most of full text | (c) none |
| N1 | Bizoń, Glogić, Wasserman, three profiles in odd d ≥ 11 | none | Phys. Rev. D 2026 | full text | (a) remark, (b) notes |
| N2 | Bizoń, exact unstable mode in d = 5 | none | preprint | full text | (a) caveat, (c) lead |
| N3 | Santos, Marouda, Hilditch, EYM in D = 4 | none | Phys. Rev. D 2026 | full text, parts | none |
| N4 | stability, heat-flow and other gauge-flow papers | none | preprints | abstracts | none |
| N5 | large-D collapse papers of 2024–2026 | none | various | abstracts | (b) unchanged |

## 1. Results made with AI

### A1. OpenAI: existence of quantum Yang–Mills in four dimensions (announced 26 October 2026)

**What is claimed.** The shepherding committee (M. R. Douglas, D. J. Gross (chair), A. Jaffe, I. R. Klebanov,
N. Nekrasov) writes of "the recent proof by OpenAI of the existence of quantum Yang-Mills theory, including coupling to
fermions". It credits Balaban's renormalization group as having "enabled OpenAI to follow the theory all the way to
infinite distances and prove the existence of its proper continuum limit", and adds "But confinement and the mass gap
remain very important open challenges". The committee holds "the full 1147-page LLM proofs" and a Lean formalization,
and will release the Lean proof after checking that its definitions match the standard ones.

**How.** An LLM wrote the natural-language proof, and OpenAI is providing a Lean formalization. Nothing is public.

**Status.** Announcement only, with no proof, theorem statement or Lean code released, and no review. The mass gap is
not claimed.

**Read.** The committee statement *Proof by OpenAI of the Existence of Quantum Yang-Mills theory* (KITP, October 2026,
[PDF](https://www.kitp.ucsb.edu/sites/default/files/YM_working_group.pdf)), full text, and its one-page *Rules and
Procedures*. OpenAI's own page
([link](https://openai.com/index/building-scientific-understanding-with-academic-communities/)) returned HTTP 403 and
was not read. The committee's second statement of 3 November returned HTTP 404. P. Woit's blog post of 26 October
([link](https://www.math.columbia.edu/~woit/wordpress/?p=15948)), a secondary source, was read as a summary.

**Bearing.** None. This is constructive quantum field theory in flat four-dimensional space. It has no classical
dynamics, gravity or self-similarity.

### A2. Alpoge and Sussman: LLM-written constructions of QED₃, YM₃ and YM₄ (GitHub, October 2026 onward)

**What.** The repository
[ethanwsussman/Yang--Mills-LLM-experiment](https://github.com/ethanwsussman/Yang--Mills-LLM-experiment)
(created 2026-10-06) holds "Documents related to an attempt with Levent Alpoge to get Claude to finish Balaban's program
to construct Yang--Mills". Its README reports a 242-page QED₃ manuscript claiming the Osterwalder–Schrader axioms, YM₃
manuscripts of 857 and 1201 pages claiming them for tr F², and an unfinished YM₄ argument on ℝ⁴. It opens with
"Its content has not been satisfactorily vetted. All claims should be understood as tentative, pending verification."

**How.** Claude writes long drafts, and the README says these are "fed to Codex to extract a minimal manuscript".

**Status.** Drafts in a public repository, unrefereed and, by the authors' account, unverified.

**Read.** The README only, downloaded through the GitHub API on 2026-11-05.

**Bearing.** None, for the reason given under A1.

### A3. Guevara, Lupsasca, Skinner, Strominger, Weil: single-minus gluon tree amplitudes

**What.** Tree-level n-gluon amplitudes with one negative helicity, usually taken to vanish, are nonzero on
"half-collinear" configurations in Klein space or for complex momenta. A closed form is given for the decay of one
minus-helicity gluon into n − 1 plus-helicity gluons.

**How.** "The key formula (39) for the amplitude in this region was first conjectured by GPT-5.2 Pro and then proved by
a new internal OpenAI model. The solution was checked by hand using the Berends–Giele recursion."

**Status.** Preprint, [arXiv:2602.12176](https://arxiv.org/abs/2602.12176) v2 (9 March 2026), no journal reference on
arXiv or INSPIRE. A supersymmetric follow-up, A. Brandhuber, P. Pichini, G. Travaglini, C. Wen, JHEP 10 (2026) 171,
[arXiv:2603.26609](https://arxiv.org/abs/2603.26609), was seen as INSPIRE metadata only.

**Read.** Abstract and the paragraph on the AI's role, from the full text.

**Bearing.** None. These are perturbative scattering amplitudes.

### A4. Machine learning for quantum Yang–Mills and related problems

| Work | What the network does | Read |
|---|---|---|
| R. Abbott et al., Phys. Rev. D 114, 014513 (2026), [arXiv:2603.02984](https://arxiv.org/abs/2603.02984) | normalizing flows cut the variance of glueball correlators in SU(3) Yang–Mills and QCD by 10–60 | abstract |
| Z. Li et al., JHEP 06 (2026) 207, [arXiv:2510.24494](https://arxiv.org/abs/2510.24494) | a transformer predicts Wilson loops in two-dimensional lattice Yang–Mills | metadata |
| T. Cai et al., Mach. Learn.: Sci. Technol. 5, 035073 (2024), [arXiv:2405.06107](https://arxiv.org/abs/2405.06107); T. Cai et al., JHEP 04 (2025) 143, [arXiv:2501.05743](https://arxiv.org/abs/2501.05743) | transformers compute planar N = 4 super-Yang–Mills amplitude coefficients | metadata |
| C. Mishra, J. Tan, JHEP 06 (2026) 093, [arXiv:2512.10907](https://arxiv.org/abs/2512.10907) | geometric machine learning solves the Hermitian Yang–Mills equations on Calabi–Yau bundles | abstract |
| N. Bodendorfer et al., Phys. Rev. D 112, 046010 (2025), [arXiv:2409.00398](https://arxiv.org/abs/2409.00398) | neural-network quantum states for a Yang–Mills matrix model | metadata |

**Bearing.** None. These are quantum, Euclidean or static problems. The list is a sample of an arXiv search (§6), not a
complete survey of this field.

### A5. Bizoń and Kluszczyński: a third self-similar profile in every odd d ≥ 11

**What.** In the notation of N1, the polynomial $`P_m`$ has a root in (0, 1) distinct from the two explicit roots for
every m ≥ 3. So in every odd d ≥ 11 there are at least three smooth self-similar profiles, up to reflection. Uniqueness
is open.

**How.** The proof is by hand. It takes the signs of $`P_m`$ at 0 and 1, an irreducibility argument over ℚ where
3(d − 2)(d − 4) is not a square, and the simplicity of the explicit roots for m ≥ 12, which covers the exceptional
dimensions d = 29, 365, 5045, … of a Pell equation. The inequalities in the appendix were checked in exact arithmetic
with Maple. AI enters in the open-problems section only: "Using AI-assisted exact root-counting computations, we have
verified this for odd dimensions 11≤d≤101". The paper does not say which AI was used or how. The paper also reports,
as numerical evidence, that u₋ and u* meet at d ≈ 17.22278 (d taken real) "in what appears to be a transcritical
bifurcation". It conjectures a third profile in every d ≥ 11, even d included. It restates the mode counts of N1
"apart from the time-translation mode".

**Status.** Preprint, P. Bizoń, R. Kluszczyński, [arXiv:2610.36784](https://arxiv.org/abs/2610.36784) v1 (27 October
2026), not yet on INSPIRE.

**Read.** Full text.

**Bearing.** (b) and (c), with N1.

### A6. Networks that find unstable self-similar singularities (not Yang–Mills)

- Y. Wang, …, T. Buckmaster, …, J. Gómez-Serrano, …, C.-Y. Lai (22 authors), *Discovery of Unstable Singularities*,
  [arXiv:2509.14185](https://arxiv.org/abs/2509.14185) v1 (September 2025), no journal reference, a Google DeepMind
  collaboration with Brown, NYU and Stanford (its blog post of 18 September 2025, read as a summary). Physics-informed
  networks with a high-precision Gauss–Newton optimizer find unstable self-similar profiles for the porous-media and
  Boussinesq equations and for 3D Euler with a boundary, some "near double-float machine precision", aimed at
  computer-assisted proofs. Abstract (arXiv API) and the abstract page (WebFetch).
- Y. Wang, T. Léger, C.-Y. Lai, T. Buckmaster, [arXiv:2511.22819](https://arxiv.org/abs/2511.22819) (November 2025),
  preprint. A residual re-weighting reaches round-off for more unstable profiles, and finds new porous-media and
  nonlinear Schrödinger solutions. Abstract.
- A. Ganeshram, V. Duruisseaux, A. Anandkumar, [arXiv:2609.10867](https://arxiv.org/abs/2609.10867) v2 (September
  2026), preprint. A physics-informed network finds an approximate self-similar profile of 3D Euler on ℝ³ at blowup rate
  0.5 and certifies it with splines. Abstract.

No paper of this kind on Yang–Mills, wave maps or Einstein's equations was found (§6).

**Bearing on (c).** None. These papers solve for a self-similar profile and its scaling exponent together, at the
precision a computer-assisted proof needs. That is what the project does for the echo period Δ, with Fourier modes in
time and Newton's method in double-double and quad-double (`hp/`). Its 1 + 1-dimensional solutions are analytic, so a
spectral basis converges fast, and the large-D wall comes from conditioning and the number of modes, not from how the
solution is represented. A network at double precision would be a step back. *(Inferred.)*

### A7. Physics-informed networks for Choptuik collapse (not Yang–Mills)

**What.** A. Ferrer-Sánchez, N. Villanueva-Espinosa, C. Hernani Morales, R. Ruiz de Austri-Bazan, J. A. Font,
J. D. Martín-Guerrero, M. W. Choptuik, Mach. Learn.: Sci. Technol. 7, 025038 (2026),
[arXiv:2511.15247](https://arxiv.org/abs/2511.15247). The spherical Einstein–massless-scalar system in D = 4, in
polar-areal coordinates, is solved with several network types (plain and high-precision networks, Kolmogorov–Arnold
networks, and a modified network, "ModPINN") and compared with finite differences. Near criticality the best summed
relative l² error is 0.588 ± 0.060. The critical amplitude is taken from earlier finite-difference work, and Δ and γ
are not computed.

**Read.** The HTML full text through WebFetch, its first 100 000 of 145 000 characters.

**Bearing on (c).** None. The accuracy is many orders of magnitude short of what the critical solution here needs.

## 2. Recent results without AI

### N1. Bizoń, Glogić, Wasserman: three self-similar solutions in high odd dimensions

**What.** SO(d) Yang–Mills in d + 1 Minkowski dimensions with the magnetic ansatz,
$`\phi_{tt} = \phi_{rr} + (d-3)\phi_r/r + (d-2)\phi(1-\phi^2)/r^2`$, the equation of LITERATURE.md §4.2 with w → φ.
- *Proved.* For odd d ≥ 11 the smooth self-similar profiles, up to reflection, correspond one to one to the zeros in
  (0, 1) of an explicit polynomial $`P_m(z)`$ of degree m = (d − 5)/2. This polynomial is the resonance condition for
  smoothness at the past light cone.
- *Computed.* There are three zeros for m = 3–15 (d = 11–35). A computer-algebra check of a coefficient sign pattern,
  credited to A. Soranzo, extends the count to m ≤ 40.
- *Explicit.* A second closed-form profile, $`u_- = 1 - \alpha_-y^2/(y^2+\beta_-)`$, the partner of the known
  $`u_+`$ with $`\alpha_\pm = 2 \pm 2\sqrt{(d-4)/(3(d-2))}`$ and
  $`\beta_\pm = \tfrac23(d-4) \pm \tfrac13\sqrt{3(d-2)(d-4)}`$. It is regular for d ≥ 11: "To the best of our
  knowledge, the solution u₋ has not previously appeared in the literature." The third profile u* is found by shooting.
- *Beyond the past light cone.* u± "obviously extend smoothly to the entire spacetime", and u* is only $`C^m`$ at the
  future light cone.
- *Numerical, "ongoing work".* u₋ has two unstable modes for 11 ≤ d ≤ 17 and one for d ≥ 19, and u* the reverse. "The
  solution with a single unstable mode (either u∗ or u−, depending on the dimension) is expected to be critical."
- The motivation is "the idea that critical phenomena at the threshold of blowup may simplify as the dimension
  increases", "conceptually" related to the large-D limit of gravity (they cite Emparan and Herzog). They write "We are
  not aware of any studies of solutions of Eq. (9) in higher dimensions."

**Status.** Phys. Rev. D 114, 064021 (2026), doi:10.1103/vhyp-lb9z (INSPIRE),
[arXiv:2602.00345](https://arxiv.org/abs/2602.00345) v3 (August 2026), whose comment reads "Error in Appendix
corrected; main results unchanged". No AI is acknowledged.

**Read.** Full text of v3.

**Bearing.** (b) LITERATURE.md §4.2, §5.3 and §9 need it (§5). (a) The threshold remark of the answer's item 2.
(c) A lead.
- *Lead (inferred).* The D = ∞ flat inner problem of STRUCTURE.md §5, $`W_{tt} = W_s + W(1-W^2)/(2s)`$, has the
  large-d limits of both explicit profiles as exact solutions,
  $`w = 1 - 2\alpha s/(2s + c\,t^2)`$ with $`(\alpha, c) = (2 \pm 2/\sqrt3,\ (2 \pm \sqrt3)/3)`$. This was checked
  with sympy in a scratch script (§7), with a mutated c that fails. LITERATURE.md §4.2 records the + case only.
- *Lead (inferred).* Both explicit values $`u_\pm(1)^2`$ tend to 1 as d grows, since β± grows like d. The selection
  data at the past light cone then fade, while the profiles vary on y ~ √d, outside the past light cone. That is the
  scalar's picture, where selection moves to an inner region at x ~ √n. Whether the third root $`z_*(d)`$ of
  $`P_m`$ has a limit fixed by the D = ∞ inner problem would test the inner-selection idea on a continuously
  self-similar family, with exact data for every odd d. It costs exact polynomial arithmetic only. No such computation
  was run.

### N2. Bizoń: an exact unstable mode in d = 5

**What.** In d = 5 there are infinitely many self-similar profiles $`u_n`$, and numerics had found that every excited
one (n ≥ 1) has least unstable eigenvalue λ = 5. Bizoń proves it. In x = 1/y and the Sturm–Liouville form of the
linearized problem, $`\varphi_n = (U_n/U_0)'`$ is an eigenfunction with E = −16, that is λ = 5. Here $`U_0`$ is the
explicit fundamental profile. The ansatz $`\varphi = A U' + B U`$ asks for coefficients that do not depend on the
profile. A node count shows that the mode is the least unstable one. Identity (24) holds for any smooth U, and it
makes the mode exact whenever U solves the profile equation. He also notes that the spectral problem in d dimensions is
invariant under λ ↦ d − 3 − λ. This gives exact modes λ = 3 and 5 in d = 7 and 9 for profiles that vanish at the
light cone.

**Status.** Preprint, [arXiv:2611.04861](https://arxiv.org/abs/2611.04861) v1 (3 November 2026). No AI is acknowledged.

**Read.** Full text.

**Bearing.** (a) The answer's item 3. *(Inferred.)* The identity holds on the self-similar (τ-independent) reduction,
not for every solution of the time-dependent equation, so it is a conditional identity and not a symmetry of the PDE.
For the ℓ = 1 inner problem, whose DSS solutions are periodic in time, the analogue would be an eigenfunction given by a
universal first-order expression in the periodic solution. A polynomial search for that, like `struct_symmetry.py`'s,
would be a cheap way to test whether γ∞ of the ℓ = 1 class has a structural value (c).

### N3. Santos, Marouda, Hilditch: Einstein–Yang–Mills collapse in D = 4 with null infinity

**What.** Characteristic evolution in compactified Bondi coordinates, fourth order, magnetic SU(2) ansatz in D = 4. The
critical solution is DSS, and the Bondi mass and news echo too. Black-hole masses scale with a universal exponent.

| quantity | Santos et al. 2026 | Gundlach 1997 [G97] | Maliborski–Rinne 2018 [MR18] |
|---|---|---|---|
| Δ | ≃ 0.7388 | 0.73784 ± 0.00002 | 0.7364 ± 0.0007 |
| γ | 0.1977 ± 0.0009 | 0.1964 ± 0.0007 | 0.20018 ± 0.00017 (horizon mass) and 0.19714 ± 0.00074 (curvature) |

MR18 gives both values (checked in its full text, the section on type II collapse). Santos et al. quote the
second, LITERATURE.md §4.1 the first, so the two do not conflict. The ± are as each paper states them, and none is a
rigorous bound.

**Status.** Phys. Rev. D 113, 024058 (2026) (INSPIRE), [arXiv:2510.25534](https://arxiv.org/abs/2510.25534). No AI.

**Read.** Abstract, introduction, the comparison passages and the conclusions, from the full text.

**Bearing.** None on the large-D claims. STRUCTURE.md's D = 4 contrast (γ = 0.196 for Yang–Mills against 0.119 for
ℓ = 1) stands, and the new value could be listed in LITERATURE.md §4.1.

### N4. Other recent Yang–Mills blowup and flow results (no AI, no bearing)

- R. Donninger, M. Ostermann, [arXiv:2409.14733](https://arxiv.org/abs/2409.14733) v2 (April 2025), preprint. The
  explicit self-similar blowup of corotational wave maps and equivariant Yang–Mills is stable in every
  energy-supercritical dimension, in regions that reach toward the future light cone. Abstract and contents.
- J. Kim, S. Lee, [arXiv:2411.19293](https://arxiv.org/abs/2411.19293), preprint. Type I blowup of the Yang–Mills
  heat flow on ℝⁿ for 5 ≤ n ≤ 9, near the equivariant shrinkers. Abstract.
- Y. Sire, J. Wei, Y. Zheng, Y. Zhou, [arXiv:2601.21017](https://arxiv.org/abs/2601.21017), preprint. Global
  oscillating solutions of the equivariant Yang–Mills heat flow in dimension 4. Abstract.
- A. Royo Abrego, [arXiv:2609.27644](https://arxiv.org/abs/2609.27644) (Ricci–Yang–Mills-type flows, with a shrinking
  soliton on an SU(2) bundle over S⁴), and A. M. Silva, [arXiv:2610.33874](https://arxiv.org/abs/2610.33874) (the
  Yang–Mills gradient flow of centre vortices). Both preprints. Abstracts.

### N5. Large-D collapse, 2024–2026 (for (b))

The large-D collapse papers found are all about the real scalar. Clark–Pimentel (JHEP 2025), Ecker–Ecker–Grumiller
(PRL 2026, and with Jechtl, SciPost 2026), Lai–Wen ([arXiv:2610.04960](https://arxiv.org/abs/2610.04960)) and Wu
([arXiv:2610.20705](https://arxiv.org/abs/2610.20705)) are already in notes/LIT_REVIEW.md or matter/LITERATURE.md. Two
more were found, with abstracts read. C.-H. Wu, *Quantum fate of the Choptuik naked singularity*, Phys. Rev. D 114,
044092 (2026), [arXiv:2606.17135](https://arxiv.org/abs/2606.17135), treats the real scalar in 2 + 1 and 3 + 1
dimensions. L.-J. Xin, X. Zhang, [arXiv:2611.00829](https://arxiv.org/abs/2611.00829), treat a scalar-perturbed
Reissner–Nordström interior for 4 ≤ D ≤ 9 and find γ ≃ 1/(D − 2) for the inner-horizon radius. That is a threshold
inside a black hole, not collapse at large D. L. Bridera, E. Gallo, T. Mädler, Gen. Relativ. Gravit. 58, 83 (2026),
[arXiv:2606.13636](https://arxiv.org/abs/2606.13636), set up the n-dimensional charged scalar but run no collapse. The
README sentence stands.

## 3. Bearing, by question

**(a) The Yang–Mills row.** Its three claims are that the type II solution has the ℓ = 1 multiplet's D = ∞ inner problem
(checked symbolically), that the s-translation is broken (checked symbolically), and that γ∞ is the ℓ = 1 value
(conjecture). No work found bears on the first two, since nothing treats Einstein–Yang–Mills above D = 4 or
Gundlach's small-scale system in any D. The conjecture is untouched, but its supporting caveat changes (item 3 of the
answer, N2). The premise that the type II threshold in D ≥ 6 is the small-scale system's, which STRUCTURE.md §10 marks
as argued, is unaffected. Its wording in §5(e) is too strong (item 2, N1).

**(b) "We know of no large-D study of critical collapse for any matter other than the real scalar."** It still holds.
The closest work, N1 and A5, is flat-space Yang–Mills at high fixed odd d, without gravity and without a limit d → ∞. It
does make matter/LITERATURE.md out of date (§5).

**(c) Methods.** The AI solvers (A6, A7) offer nothing the project lacks. The OpenAI Lean formalization (A1) is not
public, and it concerns quantum field theory, so it has nothing to offer `solver/proof/`. Two non-AI leads come out of
the survey. N2's ansatz, a profile-universal first-order eigenfunction, could be tried on the ℓ = 1 inner problem.
N1's family, with exact light-cone data in every odd d, is a cheap test of the D = ∞ inner selection on a self-similar
solution. Neither has been run.

## 4. Leads, not results

1. Search the ℓ = 1 inner system for an eigenfunction of the form A ∂Y + B Y (+ C ∂ₜY), with coefficients polynomial in
   s, t and the solution's own fields, that solves the linearized equations on every time-periodic solution (N2). A
   success would give γ∞ of the 1/s class structurally. A failure within a stated degree would narrow STRUCTURE.md §10.
2. Compute the roots of $`P_m`$ exactly for odd d up to a few hundred, and from them the large-d behaviour of u*. Then
   test whether its inner limit is a third self-similar solution of the D = ∞ flat inner equation, picked out by a
   condition of that equation alone, as the scalar's Δ∞ is in its inner problem (N1, A5).

## 5. Proposed text (a proposal for the coordinator, not applied)

**README, "Other matter at large D".** No change is needed. Optionally, after "we know of no large-D study of critical
collapse for any matter other than the real scalar", add

> Flat-space Yang–Mills, without gravity, has at least three self-similar blowup profiles in every odd space dimension
> d ≥ 11, exactly three up to d = 101, but they have not been followed to d → ∞[^ym-highd].

with the footnote

```markdown
[^ym-highd]: P. Bizoń, I. Glogić, A. Wasserman, Phys. Rev. D 114, 064021 (2026), arXiv:2602.00345, and
    P. Bizoń, R. Kluszczyński, arXiv:2610.36784 (2026).
```

**matter/LITERATURE.md §4.2, after the d = 5 bullet.**

> - *odd d ≥ 11:* the smooth self-similar profiles correspond to the zeros in (0, 1) of an explicit polynomial of degree
>   (d − 5)/2, the light-cone resonance condition (proved, [BGW26]). There are at least three in every odd d ≥ 11
>   [BK26]. There are exactly three for 11 ≤ d ≤ 35 by explicit computation [BGW26], and for 11 ≤ d ≤ 101 by
>   "AI-assisted exact root-counting computations" [BK26]. Two are explicit, u₊ (the ground state below) and
>   $`u_- = 1 - \alpha_-y^2/(y^2+\beta_-)`$ with the signs of the square roots reversed. The third, u*, is numerical.
>   Numerically u* is the threshold solution for 11 ≤ d ≤ 17 and u₋ for d ≥ 19 (one unstable mode besides the gauge
>   mode, [BGW26]), and the two cross at d ≈ 17.22 [BK26].
> - *d = 5, excited profiles:* every $`u_n`$, n ≥ 1, has the exact least unstable eigenvalue 5, with eigenfunction
>   $`(U_n/U_0)'`$ [B26]. It comes from a profile-universal identity, not from a Lie point symmetry.

**matter/LITERATURE.md §4.2, the "Large d" remark.** Add

> The explicit u₋ has the inner limit $`w = 1 - 2\alpha s/(2s + c\,t^2)`$ with α = 2 − 2/√3 and c = (2 − √3)/3. It
> also solves the leading-order equation exactly *(remark, checked symbolically, for lit_remarks_check.py)*.

**matter/LITERATURE.md §5.3, last sentence.** Replace "No large-d study of wave-map or Yang–Mills blowup was found."
with

> No study of the limit d → ∞ of wave-map or Yang–Mills blowup was found. Self-similar Yang–Mills blowup has been
> studied at every fixed odd d ≥ 11 [BGW26, BK26] (§4.2).

**matter/LITERATURE.md §9, "Not found".** After "Yang–Mills (with or without gravity)" add "(flat Yang–Mills at high
fixed odd d is in [BGW26, BK26], with no limit d → ∞)".

**matter/LITERATURE.md §4.1.** Add after [MR18]

> - [SMH26] evolved the magnetic system with characteristic methods out to null infinity. They found Δ ≃ 0.7388 and
>   γ = 0.1977 ± 0.0009, and the same echoing in the Bondi mass and news.

**matter/LITERATURE.md §11, Yang–Mills.** Add

> - [BGW26] P. Bizoń, I. Glogić, A. Wasserman, Phys. Rev. D 114, 064021 (2026), [arXiv:2602.00345](https://arxiv.org/abs/2602.00345).
> - [BK26] P. Bizoń, R. Kluszczyński, [arXiv:2610.36784](https://arxiv.org/abs/2610.36784) (2026).
> - [B26] P. Bizoń, [arXiv:2611.04861](https://arxiv.org/abs/2611.04861) (2026).
> - [SMH26] R. P. Santos, K. Marouda, D. Hilditch, Phys. Rev. D 113, 024058 (2026), [arXiv:2510.25534](https://arxiv.org/abs/2510.25534).

**matter/STRUCTURE.md §5(e).** Replace "So the flat blowup ends in black holes with a mass gap; it is not a threshold
phenomenon." with

> So the generic flat blowup ends in black holes with a mass gap. The flat critical solutions (W₁ in d = 5, and u* or
> u₋ in odd d ≥ 11, LITERATURE.md §4.2) would by the same argument give a black-hole threshold with a mass gap, not a
> type II one (argued).

**matter/STRUCTURE.md §10, symmetry bullet.** Replace "though nothing suggests one" with

> In flat Yang–Mills in d = 5 an exact growth rate (λ = 5 for every excited profile) comes from a profile-universal
> identity on the self-similar reduction, not from a point symmetry [B26 in LITERATURE.md]. A search of that kind has
> not been run for the inner systems.

## 6. Search ledger

- **Web search** (2026-11-05), one query each on
  - neural-network discovery of unstable self-similar Yang–Mills blowup,
  - physics-informed networks for the Yang–Mills heat flow,
  - machine learning and Choptuik collapse,
  - higher-dimensional Einstein–Yang–Mills collapse and large-D Yang–Mills collapse,
  - AI and the Yang–Mills mass gap, and OpenAI's Yang–Mills announcement,
  - "AI-assisted" with Yang–Mills self-similar solutions or blowup,
  - Gauss–Newton networks after *Discovery of unstable singularities*,
  - AlphaEvolve, Deep Think or AlphaProof with Yang–Mills, and normalizing flows for lattice Yang–Mills.
- **arXiv API**, submissions from 2024 on, abstract searches for
  - Yang–Mills with self-similar, with gravitational collapse, with large D, and the Yang–Mills heat flow,
  - Yang–Mills or Einstein–Yang–Mills with neural network, physics-informed or machine learning,
  - critical collapse with the same terms, with language model or with artificial intelligence,
  - self-similar with physics-informed and blowup, and wave maps with neural,
  - computer-assisted with Yang–Mills, and Einstein–Yang–Mills with higher dimensions,
  - large D with critical collapse and with self-similar.
- **INSPIRE**, citations since 2024 of Gundlach 1997, Choptuik–Chmaj–Bizoń 1996, Bizoń–Tabor 2001, Maliborski–Rinne
  2018, Biernat–Bizoń 2015, Rozali–Way 2018, Clark–Pimentel 2025 and the two Ecker et al. papers of 2026, and abstract
  searches for Yang–Mills with self-similar, blowup, blow-up and critical collapse.
- **Not found:** any AI-made or AI-assisted result on Einstein–Yang–Mills, on Yang–Mills critical collapse, or on the
  Yang–Mills heat flow, any network-based discovery of self-similar Yang–Mills or wave-map solutions, and any study of
  Einstein–Yang–Mills collapse in D ≥ 5. "Not found" means these searches found nothing. It does not prove that nothing
  exists.
- **Searched but not pursued:** the many unreviewed preprints and Zenodo records that claim to solve the mass-gap
  problem (seen as search results, not read), and OpenAI's September 2026 Navier–Stokes claim, which is not about
  Yang–Mills *(unverified, seen in secondary coverage only)*.

## 7. The scratch check behind the u₋ remark

Run with sympy 1.14.0 under Python 3.14 in a scratch directory and not committed. It printed residual 0 for both
branches, False for both mutations, the two limits, and residual 0 for the exact profile at d = 19.

```python
import sympy as sp
s, t, d = sp.symbols('s t d', positive=True)
r3 = sp.sqrt(3)
def resid(alpha, c):
    w = 1 - 2*alpha*s/(2*s + c*t**2)
    return sp.simplify(sp.diff(w, t, 2) - sp.diff(w, s) - w*(1 - w**2)/(2*s))
for sgn in (+1, -1):
    a = 2*(1 + sgn/r3); c = (2 + sgn*r3)/3
    print("branch", sgn, "residual", resid(a, c))
    print("  mutation c*1.01 residual zero?", sp.simplify(resid(a, c*sp.Rational(101, 100))) == 0)
am = 2 - 2*sp.sqrt((d - 4)/(3*(d - 2))); bm = sp.Rational(2, 3)*(d - 4) - sp.sqrt(3*(d - 2)*(d - 4))/3
print("lim alpha_-", sp.limit(am, d, sp.oo), "lim beta_-/(d-3)", sp.simplify(sp.limit(bm/(d - 3), d, sp.oo)))
y = sp.symbols('y', positive=True); dd = 19
a19 = 2 - 2*sp.sqrt(sp.Rational(dd - 4, 3*(dd - 2))); b19 = sp.Rational(2, 3)*(dd - 4) - sp.sqrt(3*(dd - 2)*(dd - 4))/3
u = 1 - a19*y**2/(y**2 + b19)
ode = y**2*(1 - y**2)*sp.diff(u, y, 2) + ((dd - 3)*y - 2*y**3)*sp.diff(u, y) + (dd - 2)*u*(1 - u**2)
print("u_- profile ODE residual at d = 19:", sp.simplify(ode))
```
