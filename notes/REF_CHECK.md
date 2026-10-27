# Reference check: paper, README, PROOF.md

Audit of 2026-10-21. It follows the `ref-check` protocol of the BootLoops skills repository (report, then fix), with the
`reading-contract` rules for claim checks. Scope:

- the 18 `\bibitem` entries of [paper/main.tex](../paper/main.tex);
- the 16 bullets (17 works) of the README's References list;
- the 20 works in [PROOF.md](../PROOF.md)'s References and Precedents table.

That is 55 entries and 38 distinct works. Every fact below comes from a page opened in this session; the URL is given
next to it. Where a page could not be opened, the entry says so.

## How the records were reached

- **Blocked channels.** `curl` to inspirehep.net, export.arxiv.org, arxiv.org, api.crossref.org,
  api.semanticscholar.org, api.openalex.org and dlmf.nist.gov was refused by the session's egress proxy (403, policy
  denial; not retried). INSPIRE's BibTeX API was therefore not used for any entry.
- **What was used instead.** The `WebFetch` tool, on URLs surfaced by `WebSearch`. These were arXiv abstract pages,
  publisher landing pages (APS, Springer, Cambridge, numdam, Project Euclid listings), library catalogues and
  institutional repositories.
- **Limits of that channel.**
  - Several arXiv `abs` URLs returned the paper's full text instead of the abstract page, so the journal-ref field was
    not visible there. The DOI field shown on the arXiv page, or the publisher page, was used instead.
  - Some publisher pages refused the fetch: IOPscience (robots), Project Euclid (bot protection), one APS abstract page
    (client error) and MathSciNet (robots).
  - The fetch tool only returns quotes under about 125 characters, so long titles were checked in pieces.

Verdicts: **OK** (every field compared against a fetched record); **MINOR** (formatting or completeness, no factual
error); **FIX** (factual discrepancy, with `field: ours -> authoritative` and the source); **PARTIAL** (some fields
verified, the rest stated); **UNVERIFIABLE** (what was tried is listed).

## Summary

| List | Entries | OK | MINOR | FIX (applied) | FIX (proposed only) | PARTIAL / UNVERIFIABLE |
|---|---|---|---|---|---|---|
| paper `\bibitem` | 18 | 11 | 1 (A&S) | 5 | 0 | 1 partial (Bland et al. 2005 page number) |
| README References | 17 works | 11 | 0 | 5 | 0 | 1 partial (same CQG page caveat as the paper) |
| PROOF.md | 20 | 11 | 3 | 0 | 2 (PROOF.md is outside the edit mandate) | 1 partial, 3 unverifiable |

Fixes applied: six distinct discrepancies, in ten entries across the two files.

- `paper/main.tex`:
  - Rozali & Way: title and journal;
  - Gundlach–Hilditch–Martín-García review: descriptive phrase replaced by the actual title, year added;
  - Bland thesis: title and handle added;
  - Donninger: journal added;
  - DLMF: wrong section §12.8 replaced by §12.9.
- README References:
  - Rozali & Way;
  - Bland thesis;
  - Donninger;
  - Emparan & Herzog and the 2025 review (descriptive phrases replaced by titles; RMP journal ref added).

The paper was rebuilt twice with `pdflatex`. The log has no undefined citations and no errors (15 pages), and the
rendered bibliography carries every fix (checked with `pdftotext`).

## 1. The paper's bibliography (`paper/main.tex`)

| Key | Verdict | Finding and source |
|---|---|---|
| Choptuik1993 | OK | PRL 70, 9: [link.aps.org/doi/10.1103/PhysRevLett.70.9](https://link.aps.org/doi/10.1103/PhysRevLett.70.9) (received 22 Sep 1992, ©1993 APS). The title, "Universality and scaling in gravitational collapse of a massless scalar field", is from the APS search listing of the same DOI. |
| KoikeHaraAdachi1995 | OK | [journals.aps.org/prl/abstract/10.1103/PhysRevLett.74.5170](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.74.5170): T. Koike, T. Hara, S. Adachi, PRL 74, issue 26, 5170, published 26 June 1995. arXiv gr-qc/9503007 is the same paper ([arxiv.org/abs/gr-qc/9503007](https://arxiv.org/abs/gr-qc/9503007), which returned the full text). |
| RozaliWay2018 | **FIX (applied)** | The published version has a different title. **title:** "Gravitating scalars and critical collapse in the large D limit" -> "Gravitating scalar stars in the large *D* limit". **journal:** (none) -> J. High Energy Phys. 11 (2018) 106, DOI 10.1007/JHEP11(2018)106, published 19 Nov 2018. Sources: [link.springer.com/doi/10.1007/JHEP11(2018)106](https://link.springer.com/doi/10.1007/JHEP11(2018)106) (lists arXiv 1807.10283) and [doaj.org/article/b6017331078c4c348cfbe97ed0243187](https://doaj.org/article/b6017331078c4c348cfbe97ed0243187). The arXiv page still shows only v1 (26 Jul 2018) with the old title and no journal-ref ([arxiv.org/abs/1807.10283](https://arxiv.org/abs/1807.10283)). Per the protocol, the published title wins. This confirms the coordinator's lead. |
| Gundlach1997 | OK | PRD 55, issue 2, 695, published 15 January 1997: [journals.aps.org/prd/abstract/10.1103/PhysRevD.55.695](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.55.695), [link.aps.org/doi/10.1103/PhysRevD.55.695](https://link.aps.org/doi/10.1103/PhysRevD.55.695). arXiv gr-qc/9604019 ✓ ([arxiv.org/abs/gr-qc/9604019](https://arxiv.org/abs/gr-qc/9604019)). Side note: the Southampton repository record ([eprints.soton.ac.uk/29173](https://eprints.soton.ac.uk/29173)) carries pages "7353-7360", which belong to a different paper (PhysRevD.54.7353). This is a folklore-type error in that repository, not in our entry. |
| MartinGarciaGundlach2003 | OK | PRD 68, issue 2, 024011 (2003), DOI 10.1103/PhysRevD.68.024011. Sources: [eprints.soton.ac.uk/29194](https://eprints.soton.ac.uk/29194/) and the DOI field of [arxiv.org/abs/gr-qc/0304070](https://arxiv.org/abs/gr-qc/0304070). The APS page itself returned a client error. |
| GundlachReview2025 | **FIX (applied)** | The entry used a descriptive phrase as the title. **title:** "review of critical phenomena in gravitational collapse" -> "Critical Phenomena in Gravitational Collapse". **year:** (none) -> 2025 (v1, 10 Jul 2025). Authors ✓ (Carsten Gundlach, David Hilditch, José M. Martín-García). There is no journal-ref; arXiv comments: "Major update to the living review". Source: [web3.arxiv.org/abs/2507.07636](https://web3.arxiv.org/abs/2507.07636). |
| GarfinkleCutlerDuncan1999 | OK | Authors ✓, arXiv gr-qc/9908044 ✓, DOI 10.1103/PhysRevD.60.104007 ([arxiv.org/abs/gr-qc/9908044](https://arxiv.org/abs/gr-qc/9908044)). [link.aps.org/doi/10.1103/PhysRevD.60.104007](https://link.aps.org/doi/10.1103/PhysRevD.60.104007): received 13 Aug 1999, ©1999 APS. The APS listing title is "Choptuik scaling in six dimensions". |
| SorkinOren2005 | OK | PRD 71, issue 12, 124005, published 3 June 2005: [journals.aps.org/prd/abstract/10.1103/PhysRevD.71.124005](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.71.124005). arXiv hep-th/0502034 ✓. The published title ("Choptuik's scaling in higher dimensions") differs from arXiv v2 ("On Choptuik's scaling..."); the entry gives no title, so nothing to fix. |
| BlandEtAl2005 | PARTIAL | Authors J. Bland, B. Preston, M. Becker, G. Kunstatter, V. Husain ✓ (arXiv text, [arxiv.org/abs/gr-qc/0507088](https://arxiv.org/abs/gr-qc/0507088)). CQG, volume 22, issue 24, DOI 10.1088/0264-9381/22/24/009, from the IOPscience search listing; the IOP page refused the fetch (robots). **Page 5355 not verified**: no fetched record states it. Published title per the IOP listing: "Dimension dependence of the critical exponent in spherically symmetric gravitational collapse". |
| BlandKunstatter2007 | OK | Jason Bland, Gabor Kunstatter. arXiv hep-th/0702226 v1, 28 Feb 2007. DOI 10.1103/PhysRevD.75.101501 ([arxiv.org/abs/hep-th/0702226](https://arxiv.org/abs/hep-th/0702226)). The APS listing gives the title "5-D Choptuik critical exponent and holography". The APS landing page returned a client error, so the year 2007 rests on the arXiv date and PRD volume 75. |
| Bland2007 | **FIX (applied, completeness)** | The thesis title was missing. Added "Dimension dependence of the critical phenomena in gravitational collapse of massless scalar field", PhD (Department of Physics and Astronomy), University of Manitoba, ©2007, handle 1993/20437. Sources: [mspace.lib.umanitoba.ca item](https://mspace.lib.umanitoba.ca/items/25c2bbc1-c14f-4a1f-a308-97c39ad0b4d5/full) and the PDF's title page ([bitstream](https://mspace.lib.umanitoba.ca/bitstreams/319e409d-2cfd-46d5-93ca-2bc913dbbbfb/download)). |
| EckerEtAl2026 | OK | Christian Ecker, Florian Ecker, Daniel Grumiller, Tobias Jechtl, "Critical spacetime crystals in continuous dimensions". DOI 10.21468/SciPostPhys.20.6.176, i.e. SciPost Phys. 20, 176 (2026). Sources: [arxiv.org/abs/2602.10185v1](https://arxiv.org/abs/2602.10185v1) (DOI field), [citedrive record](https://www.citedrive.com/en/discovery/critical-spacetime-crystals-in-continuous-dimensions/). A v2 exists ([arxiv.org/html/2602.10185v2](https://arxiv.org/html/2602.10185v2)); its abstract numbers are unchanged. |
| EckerEckerGrumiller2026 | OK | PRL 136, 191401 (2026), issue 19 (15 May 2026), published 12 May 2026, DOI 10.1103/qgl5-5l3t ([journals.aps.org/prl/issues/136/19](https://journals.aps.org/prl/issues/136/19)). arXiv 2601.14358 v1, 20 Jan 2026 ✓ ([www.arxiv.org/abs/2601.14358](https://www.arxiv.org/abs/2601.14358)). |
| EmparanHerzog2020 | OK | arXiv journal-ref "Rev. Mod. Phys. 92, 45005 (2020)" and DOI 10.1103/RevModPhys.92.045005 ([www.arxiv.org/abs/2003.11394](https://www.arxiv.org/abs/2003.11394)). The arXiv journal-ref drops the leading zero; the DOI is the arbiter, and ours (045005) matches it. |
| Donninger2008 | **FIX (applied)** | The entry cited only the preprint; the journal version is citable. **journal:** (none) -> J. Evol. Equ. 9 (3), 511–523 (2009), DOI 10.1007/s00028-009-0022-x. The title and author match arXiv:0809.5177 ("Asymptotics and analytic modes for the wave equation in similarity coordinates"). Sources: [link.springer.com/article/10.1007/s00028-009-0022-x](https://link.springer.com/article/10.1007/s00028-009-0022-x), [arxiv.org/pdf/0809.5177](https://arxiv.org/pdf/0809.5177). The key still says 2008; keys were not renamed. |
| AbramowitzStegun | MINOR / partly unverifiable | DLMF §15.8(i) gives "A&S Ref: 15.3" for the $`z\to 1-z`$ transformations ([dlmf.nist.gov/15.8](https://dlmf.nist.gov/15.8)). The CRAN `hypergeo` manual says its functions `f15.3.3`–`f15.3.9` are "Transformations of the hypergeometric function" named after A&S equation numbers, and `i15.3.6` returns "the factors at the beginning of equations 15.3.6-9" ([cran.nics.utk.edu hypergeo refman](https://cran.nics.utk.edu/cran/web/packages/hypergeo/refman/hypergeo.html), [i15.3.6](https://search.r-project.org/CRAN/refmans/hypergeo/html/i15.3.6.html)). That is consistent with 15.3.6 being the $`1-z`$ connection formula, but **the printed page of A&S was not opened**, so the equation number is not verified. The entry lacks publisher and year. A wiki ([specialfunctionswiki](https://specialfunctionswiki.org/index.php/Book:Milton_Abramowitz/Handbook_of_mathematical_functions)) gives NBS Applied Mathematics Series 55, 1964, but a wiki is not an authoritative source, so nothing was added. Suggest citing DLMF Eq. 15.8.4 instead, which was read. |
| Dingle1973 | OK | R. B. Dingle, *Asymptotic expansions: their derivation and interpretation*, Academic Press, London/New York, ©1973, XV+521 pp. ([theses.univ-nantes.fr catalogue](https://theses.univ-nantes.fr/vufind/Record/PPN015800474/Details)). |
| DLMF | **FIX (applied)** | The entry cited a section with the wrong content. **section:** §12.8 -> §12.9. §12.8 is "Recurrence Relations and Derivatives" and contains no connection formulas, Stokes constants or large-variable expansions ([dlmf.nist.gov/12.8](https://dlmf.nist.gov/12.8)). The Weber Stokes constant that the paper cites is the $`\pm i\sqrt{2\pi}/\Gamma(\tfrac12+a)`$ term of Eq. 12.9.3 in "§12.9 Asymptotic Expansions for Large Variable" ([dlmf.nist.gov/12.9](https://dlmf.nist.gov/12.9)). §12.2(v) "Connection Formulas" would also serve ([dlmf.nist.gov/12.2](https://dlmf.nist.gov/12.2)). §2.11 "Remainder Terms; Stokes Phenomenon" ✓ ([dlmf.nist.gov/2.11](https://dlmf.nist.gov/2.11)). §13.7 "Asymptotic Expansions for Large Argument" (Kummer functions, with (iii) "Exponentially-Improved Expansion") ✓ ([dlmf.nist.gov/13.7](https://dlmf.nist.gov/13.7)). |

Two MINOR items in the paper, no action taken:

- Citation style. Entries carry no titles (house style) except the four that now have one. The CQG entry gives "J. Bland, B. Preston, ..." in full, while the README gives "J. Bland et al.".
- The A&S entry: see the row above.

## 2. README References list

The same works as the paper, minus A&S (cited in the README text as "Abramowitz & Stegun 15.3.6" but absent from its
list). Field-by-field results are as in §1; the README-specific items are below.

| README bullet | Verdict | Action |
|---|---|---|
| "M. Rozali and B. Way, arXiv:1807.10283" | **FIX (applied)** | Now *Gravitating scalar stars in the large D limit*, JHEP 11 (2018) 106 (sources as in §1). |
| "J. Bland ..., PhD thesis, University of Manitoba (2007)" | **FIX (applied, completeness)** | Title and handle added (sources as in §1). |
| "R. Donninger, arXiv:0809.5177" | **FIX (applied)** | J. Evol. Equ. 9, 511 (2009) added (source as in §1). |
| "R. Emparan and C. P. Herzog, large-D review, arXiv:2003.11394" | **FIX (applied)** | The descriptive phrase is replaced by the title *The large D limit of Einstein's equations* and the journal ref RMP 92, 045005 (2020) is added. Source: [www.arxiv.org/abs/2003.11394](https://www.arxiv.org/abs/2003.11394). |
| "C. Gundlach, D. Hilditch and J. M. Martín-García, review of critical phenomena in gravitational collapse (2025)" | **FIX (applied)** | Title *Critical phenomena in gravitational collapse* (source as in §1). |
| "J. Bland et al., CQG 22, 5355 (2005)" | MINOR + PARTIAL | Truncated author list (style). The page number is unverified, as in §1. |
| "NIST DLMF, §2.11 and §13.7" | OK | Both sections exist with the stated content (links in §1). |
| A&S missing from the list | MINOR | Cited in the text (Proposition 2 proof) but not listed. |
| All other bullets | OK | As in §1. |

## 3. PROOF.md references (not edited: outside the edit mandate; proposed changes listed)

| Entry in PROOF.md | Marked | Verdict | Finding and source |
|---|---|---|---|
| Costin, Duke Math. J. 93 (1998) 289–344, arXiv:math/0608408 | "journal details from memory" | OK; drop the flag | The Project Euclid listing gives the title "On Borel summation and Stokes phenomena for rank-1 nonlinear systems of ordinary differential equations", volume 93, issue 2, DOI 10.1215/S0012-7094-98-09311-5; the page itself was blocked by bot protection, so only the search listing was seen. Pages 289–344 come from the reference list of [numdam AFST_1999_6_8_4_579_0](https://www.numdam.org/item/AFST_1999_6_8_4_579_0/), which gives "Duke Mathematical Journal, 93(2): 289-344, 1998". That is a citing record, not the publisher, but it agrees with the DOI and volume. The arXiv PDF title is "On Borel summation and Stokes phenomena for rank one nonlinear systems of ODE's" ([arxiv.org/pdf/math/0608408](https://arxiv.org/pdf/math/0608408)). |
| math/0608414 ("see also") | – | PARTIAL | The identifier resolves, per the search listing, to Costin, "Exponential asymptotics, transseries, and generalized Borel summation for analytic rank one systems of ODE's". Not opened. |
| Costin, *Asymptotics and Borel Summability*, Chapman & Hall/CRC (2008) | "from memory" | **UNVERIFIABLE year** | Title, author and publisher ✓. The year is disputed: the British Library record says ©2009 (Boca Raton; [eld.bl.uk/catalog/018426817](https://eld.bl.uk/catalog/018426817)), the CERN-derived record says 2008 ([ebusca.uv.mx/Record/cern-1999798](https://ebusca.uv.mx/Record/cern-1999798)), and the publisher page refused the fetch. Suggest "(2008; ©2009)" or checking the copyright page. |
| Costin & Tanveer, Ann. IHP Anal. Non Linéaire 24 (2007) 795–823 | "volume and pages from memory" | OK; drop the flag | 24 (5), 795–823 (2007), DOI 10.1016/j.anihpc.2006.07.002 ([numdam AIHPC_2007__24_5_795_0](https://www.numdam.org/item/AIHPC_2007__24_5_795_0)). Its arXiv number is math/0608290 (from the search listing), which could be added. |
| Costin & Tanveer, Navier–Stokes, arXiv:math/0612063 | – | OK (arXiv fields) | O. Costin, S. Tanveer; v1 2006-12-02, v2 2006-12-20 ([arxiv.org/abs/math/0612063](https://arxiv.org/abs/math/0612063)). No journal-ref on the page. |
| Kruskal & Segur, Stud. Appl. Math. 85 (1991) 129–181 | "details from memory" | OK; drop the flag | Volume 85, issue 2, pages 129–181, 1 Aug 1991 ([vivo.colorado.edu/display/pubid_44003](https://vivo.colorado.edu/display/pubid_44003)). |
| Combescot, Dombre, Hakim, Pomeau, Pumir, PRL 56 (1986) 2036 | "from memory" | OK; drop the flag | "Shape Selection of Saffman-Taylor Fingers", PRL 56, 2036, issue of 12 May 1986 ([journals.aps.org/prl/issues/56/19](https://journals.aps.org/prl/issues/56/19)). |
| Hong & Langer, PRL 56 (1986) 2032 | "from memory" | OK; drop the flag | "Analytic Theory of the Selection Mechanism in the Saffman-Taylor Problem", PRL 56, 2032, same issue (same source). |
| Tanveer, Phys. Fluids 30 (1987) 1589 | "from memory" | **UNVERIFIABLE** | Tried: three web searches (title words; "Phys. Fluids 30" 1589; Saffman–Taylor 1987), and the OSTI record [osti.gov/biblio/578102](https://www.osti.gov/biblio/578102). That record lists a 1987 Phys. Fluids paper "Analytic theory for the linear stability of the Saffman–Taylor finger" with no author or pages, which does not confirm the entry. Keep the flag. |
| Xie & Tanveer, ARMA 166 (2003) 219–286 | "from memory" | OK (secondary source); drop the flag after a publisher check | ARMA 166(3) (2003) 219–286, from the author's institutional page ([morgan.edu/.../xuming-xie](https://morgan.edu/mathematics/faculty-and-staff/xuming-xie)). The publisher page was not fetched. |
| Tanveer & Xie, CPAM 56 (2003) 353–402 | "from memory" | OK (secondary source) | CPAM 56(3) (2003) 353–402, same source. |
| Xie, needle crystals, J. Differential Equations 197 (2004) 349–426 | "from memory" | **UNVERIFIABLE** | Not on the author's institutional page. Two searches found no record (the closest hits are Xie's later needle-crystal papers, e.g. [arXiv:1108.2742](https://arxiv.org/pdf/1108.2742), whose reference list did not come through the fetch). Keep the flag. |
| Xie, arXiv:1610.00076 | – | **FIX (proposed)** | Published: *Eur. J. Appl. Math.* 30 (1), 63–116 (2019; online 10 Jan 2018), DOI 10.1017/S0956792517000390 ([cambridge.org record](https://www.cambridge.org/core/journals/european-journal-of-applied-mathematics/article/abs/rigorous-results-in-existence-and-selection-of-saffmantaylor-fingers-by-kinetic-undercooling/09CD9F0D6F61946A7A354651042CD2BB)). Claim check of PROOF.md's precedent row: in the arXiv text, Theorem 1.11(2) gives $`\mathrm{Im}\,\psi_0 \sim \tilde S(\alpha)e^{-\eta}`$, and the selection by $`\tilde S(\beta_0)=0`$, $`\tilde S'(\beta_0)\ne0`$ is Theorem 1.12 ([arxiv.org/pdf/1610.00076](https://arxiv.org/pdf/1610.00076)). So "Theorem 1.11 there gives ... and S̃(β₀) = 0 ... gives the selected branch" should read "Theorems 1.11–1.12". |
| Chapman, King & Adams, Proc. R. Soc. Lond. A 454 (1998) 2733–2755 | "details from memory" | OK; drop the flag | Proc. R. Soc. A 454 (1978), 2733–2755 (1998), DOI 10.1098/rspa.1998.0278 ([ora.ox.ac.uk record](https://ora.ox.ac.uk/objects/uuid:1ae58b22-555b-4d31-b1b8-89863e3e04a4)). |
| Baldomá, Fontich, Guardia & Seara, arXiv:1201.5152 | – | OK (arXiv fields) | Title and authors ✓ ([UPCommons record](https://upcommons.upc.edu/entities/publication/b06d778d-25e3-4482-a9cc-e25063438830/full), a 2011 preprint, 117 pp.). The journal version was not determined. |
| Baldomá, Capiński, Guardia & Seara, arXiv:2206.13840; J. Nonlinear Sci. (2023) | – | OK; add volume | J. Nonlinear Sci. 33 (2023), DOI 10.1007/s00332-022-09882-x ([recercat record](https://www.recercat.cat/handle/2445/195109?show=full)). PROOF.md has the DOI but not the volume (33). |
| Lombardi, LNM 1741, Springer 2000 | "from memory" | OK; drop the flag; MINOR | Title ✓, plus the subtitle "with Applications to Homoclinic Orbits in Reversible Systems"; LNM 1741; Springer 2000; DOI 10.1007/BFb0104102 ([link.springer.com/book/9783540677857](https://link.springer.com/book/9783540677857)). |
| Reiterer & Trubowitz, arXiv:1203.3766 | – | **FIX (proposed)** | Published: *Commun. Math. Phys.* 368, 143–186 (2019), DOI 10.1007/s00220-019-03413-8, same title and authors ([link.springer.com/article/10.1007/s00220-019-03413-8](https://link.springer.com/article/10.1007/s00220-019-03413-8)). Claim check ✓: the abstract says "The proof is computer assisted", with "a high precision data file (about 80 significant decimal digits, with rigorous error bounds)" ([arxiv.org/pdf/1203.3766](https://arxiv.org/pdf/1203.3766)). |
| Gundlach, PRL 75 (1995) 3214, gr-qc/9507054 | "journal details from memory" | OK; drop the flag; MINOR | DOI 10.1103/PhysRevLett.75.3214 on the arXiv page ([arxiv.org/abs/gr-qc/9507054](https://arxiv.org/abs/gr-qc/9507054)). The APS DOI page shows received 30 Sep 1994, ©1995 ([link.aps.org/doi/10.1103/PhysRevLett.75.3214](https://link.aps.org/doi/10.1103/PhysRevLett.75.3214)). The APS search listing gives the published title as "Choptuik Spacetime as an Eigenvalue Problem", without "The". |
| Chatzikaleas, Donninger & Glogić, arXiv:1701.05082 | – | OK (arXiv fields) | Title and authors ✓, v3 23 Jun 2017 ([arxiv.org/abs/1701.05082](https://arxiv.org/abs/1701.05082)). The journal version was not determined. |

Proposed edits to PROOF.md, for the integrator:

- drop "(from memory)" on the eight entries verified above;
- add the journal refs for Reiterer–Trubowitz (CMP 368, 143 (2019)) and Xie (EJAM 30, 63 (2019));
- add volume 33 for Baldomá et al. 2023;
- keep the flag on Tanveer 1987 and Xie JDE 2004, and on the year of Costin's book;
- change "Theorem 1.11" to "Theorems 1.11–1.12" in the Xie row.

## 4. Claim check (does the cited work say what the sentence says?)

Each check names the sentence's location, the source and the locator.

| Sentence (location) | Source and locator | Verdict |
|---|---|---|
| "$`\Delta = 3.445452402(3)`$" (paper intro; README) | Martín-García & Gundlach, Sec. IV.B, Eq. (58): `\Delta = 3.445452402(3)` ([arxiv.org/pdf/gr-qc/0304070](https://arxiv.org/pdf/gr-qc/0304070)) | SUPPORTED |
| "$`\gamma \simeq 0.374`$" (intro) | Gundlach 1997, abstract: "γ=0.374±0.001" ([ar5iv gr-qc/9604019](https://ar5iv.arxiv.org/html/gr-qc/9604019)) | SUPPORTED |
| "$`\gamma = 1/\lambda`$ for its growth rate" [KoikeHaraAdachi] | Abstract: "the critical exponent β and the largest Lyapunov exponent Reκ of the perturbation is related by" β = (Re κ)⁻¹, for a radiation fluid ([APS page](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.74.5170)) | SUPPORTED (the source's system is a radiation fluid) |
| "studied numerically ... for integer $`D = 4`$–$`14`$" [GCD, SO, Bland et al., BK, Bland 2007] (intro) | Bland et al. abstract: "for dimensions in the range `3.5 \leq D \leq 14`" (arXiv text; LaTeX quoted as code). EEGJ, Sec. 1: Bland's thesis has "nine data points from D = 3.02 to D = 3.9" ([arxiv.org/pdf/2602.10185](https://arxiv.org/pdf/2602.10185)). The paper itself later cites Bland et al. for $`\gamma(3.5)`$. | **DRIFTED/INFLATED**: not only integer D, and not only D ≥ 4. Proposed: "for $`D = 3.5`$–$`14`$ (mostly integer $`D`$)". Taves & Kunstatter (D = 4–7, reliable to 7; [arXiv:1105.0878](https://arxiv.org/pdf/1105.0878)) are uncited. |
| "who find a maximum of $`\Delta`$ near $`D = 3.756`$" [EEGJ] | Sec. 5.2, Eq. (70): `D_{\text{crit}} = 3.755726 \pm 10^{-6}`, `\Delta = 3.466772 \pm 10^{-6}` | SUPPORTED |
| $`\gamma(4) = 0.373961 \pm 10^{-6}`$ [EEGJ] | Sec. 5.3, Eq. (71): `\gamma = 0.373961 \pm 10^{-6}` | SUPPORTED |
| D = 5: "$`\Delta = 3.22176`$ and $`\gamma = 0.41322`$" [EEGJ] | Abstract: "Δ=3.445453 (3.22176) and γ=0.373961 (0.41322)" | SUPPORTED |
| "where they note that their light-cone expansion order must be increased" (D ≳ 4.8) | Sec. 4.1: "The expansion order around the SSH has to be increased above D=4.8" | SUPPORTED |
| "no points for $`4.79 \lt D \lt 5.49`$, where they report that their perturbation method fails" | Sec. 4.2: "in the range `4.81 \leq D \leq 5.38` it turns out to be violated", "the perturbations cannot be computed any more with the current method" | SUPPORTED in substance. The stated failure range is 4.81–5.38; 4.79–5.49 is the data gap in their repository, which was not opened here. |
| "Ecker, Ecker and Grumiller ... and, with [EEGJ], conjectured $`\Delta \to 0`$ and $`\gamma \to 1/2`$" (intro) | EEG, Summary: "Existing data [16, 17, 29] suggest `\Delta \to 0` as `D \to \infty`"; "Numerical and theoretical arguments suggest `\lim_{D\to\infty}\gamma=\frac{1}{2}`", with "hinting at a CSS limit [23]" (Clark & Pimentel) ([arxiv.org/html/2601.14358v1](https://arxiv.org/html/2601.14358v1)). EEGJ, Sec. 5.2: "compatible with the conjectures `\Delta \to 0`"; Sec. 5.3: "compatible with the conjecture `\lim_{D\to\infty} \gamma = \frac{1}{2}`, see for instance [12]". | **INFLATED attribution for $`\gamma`$**: EEG and EEGJ report the $`\gamma \to 1/2`$ conjecture and credit others (Clark–Pimentel in EEG; [12] in EEGJ, whose entry could not be read). The $`\Delta \to 0`$ part is SUPPORTED. See LIT_REVIEW.md for the proposed wording. |
| EEG Eq. (20) "$`\omega^2F'' + 3\omega F' = 0`$ at the zeros of $`F`$" (paper Sec. eeg; README) | EEG Eq. (20) gives Δ as the ratio of the moduli of β'' and 3β' at β = 0 (`\Delta = \frac{\lvert\beta''\rvert}{3\lvert\beta'\rvert}` evaluated at `\beta=0`, as extracted). Text: "If `\beta'' + 3\beta'` does not vanish simultaneously with `\beta`". | SUPPORTED, but only under a convention the paper does not state. *Inferred, not read:* Eq. (20) looks like $`\beta''+3\beta'=0`$ rewritten for a profile normalised to unit period, which turns the condition into a formula for $`\Delta`$. The README's "β'' + 3β' = 0 there (their Eq. (20))" quotes the condition that precedes Eq. (20), not Eq. (20) itself. Suggest "their condition leading to Eq. (20)". |
| EEG Eq. (22) | Eq. (22): `\max f(\tau, x) = 1 - \frac{\epsilon^2}{4} (1 - x^4) (\beta'(\tau_0))^2`; with $`\beta \approx \sqrt{2/n}F`$ and $`x = 0`$ this is $`1 - (\epsilon^3/2)\omega^2F'^2`$ | SUPPORTED (convention: EEG's $`\epsilon = 1/(D-1)`$, their Eq. (2)) |
| "the leading-order freedom ... is also noted in [EEG]" | EEG: "any periodic function `\beta(\tau)` with any echoing period `\Delta` solves our equations of motion" | SUPPORTED |
| "analysed in 3+1 dimensions by Donninger" | Abstract: "We consider the radial wave equation in similarity coordinates within the semigroup formalism" (3+1) ([arxiv.org/pdf/0809.5177](https://arxiv.org/pdf/0809.5177)) | SUPPORTED |
| "Bland's hypothesis $`\Delta_\infty = 5/2`$" [Bland2007] | Thesis abstract (OCR text): "the critical solution of the theory at the limit of large dimension is discretely self-similar with a period of 5/2" ([thesis PDF](https://mspace.lib.umanitoba.ca/bitstreams/319e409d-2cfd-46d5-93ca-2bc913dbbbfb/download)) | SUPPORTED |
| "Bland and Kunstatter's $`0.4131 \pm 0.0001`$" | Abstract: "Our current result is `\gamma_{5d} = 0.4131 \pm 0.0001`" ([web3.arxiv.org/abs/hep-th/0702226](https://web3.arxiv.org/abs/hep-th/0702226)) | SUPPORTED |
| "$`0.349 \pm 0.003`$ in [BlandEtAl2005]" for $`\gamma(3.5)`$ | not located (abstract read only) | UNCHECKED |
| "the heuristic of Rozali and Way" (Prop. 5 discussion) | Sec. 2.3: "While this simple result is suggestive, it remains unclear whether"; Sec. 1: "It is unclear though how this exponent is related to the finite" ([ar5iv 1807.10283](https://ar5iv.arxiv.org/html/1807.10283)) | SUPPORTED; see LIT_REVIEW.md for what else RW contains |
| "Weber's classical Stokes constant [DLMF]" | DLMF 12.9.3 has the $`\pm i\sqrt{2\pi}/\Gamma(\tfrac12+a)`$ term, valid for $`\tfrac14\pi+\delta \le \pm\mathrm{ph}\,z \le \tfrac54\pi-\delta`$ | SUPPORTED after the §12.9 fix (the entry previously pointed to §12.8) |
| "the connection formula [A&S]" | DLMF 15.8.4 (equivalent) read; A&S not opened | UNCHECKED at the equation-number level |
| "A divergent series records the exponentially small terms it switches on in its late orders [Dingle1973]" | the book was not opened | UNCHECKED (bibliographic fields only) |

Also outside the bibliography, GAMMA.md says of Clark & Pimentel: "A summary of the paper gives the exponent as
2(D − 3)/(D − 2)". The paper itself was read here: its Eq. (1.5), "γ = 2(D−3)/(D−2) ≈ 2 − 2/D", is for CSS-breaking
perturbations, with the mass critical exponent defined by $`M \sim (p-p^*)^\gamma`$, Eq. (2.1)
([arxiv.org/pdf/2504.10590](https://arxiv.org/pdf/2504.10590)). The GAMMA.md sentence relies on a secondary summary and
should cite the equation.

## 5. Exit checklist

- [x] Every entry has a verdict (55 entries, 38 works).
- [x] Every applied FIX cites its source URL. UNVERIFIABLE items are flagged: Tanveer 1987, Xie JDE 2004, the year of
      Costin's book, page 5355 of Bland et al., and A&S's equation number.
- [ ] Every DOI resolved and its landing page compared. Not everywhere: doi.org was not fetchable (provenance gate), so
      DOIs were compared on publisher landing pages where those opened (APS, Springer, Cambridge, numdam, ORA, SciPost
      via citedrive) and otherwise on the arXiv DOI field. IOP, one APS page, Project Euclid and MathSciNet refused the
      fetch.
- [x] Claim check on the load-bearing citations, with locators (§4); prose untouched.
- [x] No keys renamed; all `\cite` keys resolve (pdflatex log: no undefined citations).
- [x] Rebuild: two pdflatex passes, 0 errors, 0 undefined citations; `paper/dss_large_D.pdf` refreshed; the rendered
      bibliography shows the five fixes.

## Addendum (2026-10-21): method references

Added as README footnotes [^15]–[^19], in both papers (Morrison–Riley–Zancanaro, Wengert, Dekker) and in the
`amp_flint_par.py` docstring (Harvey, Mulders). Each checked against a source:

| Entry | Checked against |
|---|---|
| Morrison, Riley, Zancanaro, Commun. ACM 5(12), 613–614 (1962), doi:10.1145/355580.369128 | CACM article page (title, journal); ixdf.org author record (authors, volume, issue, pages, DOI) |
| Wengert, Commun. ACM 7(8), 463–464 (1964) | the article's own first page (cs.princeton.edu copy); CACM August 1964 issue listing |
| Dekker, Numer. Math. 18, 224–242 (1971), doi:10.1007/BF01397083 | the article's first page (volume, pages); Springer/Altmetric record (DOI, year) |
| Ogita, Rump, Oishi, SIAM J. Sci. Comput. 26(6), 1955–1988 (2005), doi:10.1137/030601818 | TUHH Open Research record |
| Anderson et al., LAPACK Users' Guide, 3rd ed., SIAM (1999), ISBN 0-89871-447-8 | netlib.org/lapack/lug |
| Mulders, AAECC 11, 69–88 (2000), doi:10.1007/s002000000037 | Unpaywall record |
| Harvey, arXiv:0712.4046 | the arXiv abstract (credits Kronecker 1882) |

Not added: Hanrot–Zimmermann (2004), whose HAL record refused the fetch.

## Addendum (2026-10-21): references of the mathematics paper

Checked by the agent that wrote `paper/math/` against publisher or catalogue pages (its report; not re-checked here):
Braaksma (numdam), Moore (SIAM), Krawczyk (researchr), Neumaier, Nedialkov–Jackson (pages 219–263, not 264), Olver,
Balser, Canalis-Durand–Mozo-Fernández–Schäfke, Nikolaev (Commun. Math. Phys. 400, 463–517), Padé (numdam), Aitken
(Cambridge, 1927). Marked *(unverified)* in the paper: the chapter title and pages of Lohner (1987) (the book,
Kaucher–Kulisch–Ullrich (eds.), Teubner 1987, was confirmed) and Canalis-Durand–Ramis–Schäfke–Sibuya (2000).

## Addendum (2026-10-22): references verified for proof/B3_borel.md

Checked by the agent that wrote [proof/B3_borel.md](../proof/B3_borel.md) (its §7 and §9; not re-checked here, except
that a search for a publisher record of Sokal 1980 found none). Used in proof/C_selection.md and, for Braaksma,
Canalis-Durand 1993, Ramis–Sibuya 1994 and Sokal 1980, in `paper/math/main.tex`.

| Entry | Checked against | Verdict |
|---|---|---|
| Braaksma, Ann. Inst. Fourier 42 (3) (1992) 517–540, doi:10.5802/aif.1301 | numdam item AIF_1992__42_3_517_0 (title, author, volume, issue, pages, DOI); full text (OCR) on aif.centre-mersenne.org | OK; claim check: no parameter dependence stated (OCR text) |
| Ramis, Sibuya, Ann. Inst. Fourier 44 (3) (1994) 811–848 | eudml record 75082 | OK (bibliographic); abstract read |
| Canalis-Durand, Ann. Inst. Fourier 43 (2) (1993) 469–483, doi:10.5802/aif.1341 | numdam item AIF_1993__43_2_469_0 and full text | OK |
| Ramis, Schäfke, Nonlinearity 9 (2) (1996) 353–384, doi:10.1088/0951-7715/9/2/004 | neicon archive record (arch.neicon.ru) | OK (bibliographic); not applicable to B3 |
| Canalis-Durand, Mozo-Fernández, Schäfke, J. Differential Equations 233 (2007) 485–511 | earlier addendum; statement from the abstract on Schäfke's RIMS talk page; HAL and ScienceDirect blocked | bibliographic as before; claim: ∂_y f(0,0,0) invertible |
| Canalis-Durand, Ramis, Schäfke, Sibuya, J. reine angew. Math. 518 (2000) 95–129 | academia.edu listing (journal, year only) | PARTIAL: keep *(unverified)* on volume/pages and on the statement |
| Sokal, J. Math. Phys. 21 (2) (1980) 261–263 | sugaku.net (title, journal, year); bibliography of Gillam–Gurarii, arXiv:math/0508425 (volume, issue, pages) | PARTIAL: citing record for volume/pages; flagged in C_selection.md and the paper |
| Sibuya, *Gevrey property of formal solutions in a parameter* (1990) | not found | UNVERIFIABLE |

Also corrected in proof/C_selection.md: the title of Costin, Duke Math. J. 93 (1998), to the one verified in §3 above.

## Addendum (2026-10-27): references for the README's "Why the solver stalls"

Checked by the coordinating session; pages opened in this session.

| Entry | Checked against | Verdict |
|---|---|---|
| Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM (2002) | the author's BibTeX page, nhigham.com/accuracy-and-stability-of-numerical-algorithms-bib-entry (author, title, publisher, address, edition "Second", 2002, xxx+680 pages, ISBN 0-89871-521-0; no DOI given there) | OK (bibliographic); cited for the standard statement that a backward-stable solve's relative error is about u κ, not for a numbered theorem |
| Dembo, Eisenstat, Steihaug, *Inexact Newton methods*, SIAM J. Numer. Anal. 19, 400–408 (1982), doi:10.1137/0719025 | the reference list of Cătinaş, J. Numer. Anal. Approx. Theory 29 (2) (2000), ictp.acad.ro/jnaat/journal/article/view/662 (authors, title, journal, volume, year, pages, DOI); the DOI resolves (302) to epubs.siam.org/doi/10.1137/0719025, whose page refused the fetch (403) | PARTIAL: citing record and DOI resolution, publisher page not read; the claim (local convergence when the forcing terms stay uniformly below 1) from search summaries of the abstract, not from a page opened |

## Addendum (2026-10-27): references of the quad-double solver (hp/python/hpkernel/driver.py newton_qd, multidouble QDLu)

Checked by the coordinating session.

| Entry | Checked against | Verdict |
|---|---|---|
| Deuflhard, *A modified Newton method for the solution of ill-conditioned systems of nonlinear equations with application to multiple shooting*, Numer. Math. 22, 289–315 (1974), doi:10.1007/BF01406969 | search records of the ZIB repository (opus4.kobv.de, docId 2037: volume 22, pages 289–315, 1974) and Springer (link.springer.com/doi/10.1007/BF01406969); some citing lists print the pages as "398–315", a typo | PARTIAL: bibliographic details from the records, the content (the damped Newton method for nearly singular Jacobians in multiple shooting) from the abstract only; cited for the natural monotonicity test, which newton_qd does not use |
