# Echo analogues in other fields: discrete self-similarity, log-periodicity and the Hopf route

Provenance. Written 2026-11-04 by a research subagent of the coordinator session, from web searches and fetches only
(no code run). Reading depth is marked on every entry of the reference list at the end: "abstract" (abstract page
only), "part" (sections of the text read through an HTML rendering or a saved PDF), "full" (whole text). No entry was
read in full. Bibliographic data were confirmed on arXiv, INSPIRE, Crossref (the publisher DOI registry) or the
publisher's page; entries that could not be confirmed there are marked (unverified). Project numbers quoted below are
"the project's values" and come from the coordinator's brief, not from the literature.

Search engine queries run (WebSearch, standard mode), in order, separated by "|":
Dallaston Fontelos Tseluiko Kalliadasis discrete self-similarity Hopf | Eggers Fontelos discrete self-similarity |
Lechner Thornburg Husa Aichelburg new transition DSS CSS | Hirschmann Eardley Hopf bifurcation complex scalar |
arXiv 2602.10185 Ecker Grumiller Jechtl large D | Kaplan Lee Son Stephanov conformality lost | Schmidt Keim Zhang
Nagel bubble vibrations | Glazek Wilson limit cycles | Derrida Itzykson Luck oscillatory amplitudes | Shytov Katsnelson
Levitov graphene; Ovdat | NLS blow-up discretely self-similar | Keller-Segel blow-up discretely self-similar | Chae
discretely self-similar Euler; Bradshaw Tsai | Hara Koike Adachi RG limit cycle | "Hopf bifurcation" "self-similar"
"critical collapse" | Zhou Sornette turbulence log-periodic | Shlesinger West bronchial tree | Sornette DSI Phys. Rep.
1998 | "Hopf bifurcation" self-similar blow-up "discretely self-similar" PDE | Feigenbaum renormalization Choptuik |
Turitsyn Lai Zhang bubble disconnection | porous medium focusing discrete self-similar | Ovdat Akkermans scale anomaly |
PRL 67 2106 | Sorkin Oren higher dimensions | Fibonacci quasicrystal log-periodic | optics photonics DSI Efimov-like |
Dallaston Eggers thin film two dimensions | Johansen Ledoit Sornette; Efimov 1970; Braaten Hammer | log-periodic
ecology epidemiology | Feigenbaum 1978 | Braaten Hammer; Efimov 1970 | Hirschmann Eardley PRD 56 4696 | Bulycheva.

## 0. Vocabulary used below

In similarity variables a blow-up is a dynamical system in logarithmic time. Continuous self-similarity (CSS) is a
fixed point of that flow, discrete self-similarity (DSS) a limit cycle; the echo is one turn of the cycle and the
echoing period is its period (Eggers and Fontelos [8] classify blow-ups this way; Hara, Koike and Adachi [21] did the
same for gravity). The literature knows two generic ways a CSS state becomes DSS as a parameter moves:

* Hopf route: a complex pair of eigenvalues of the CSS fixed point crosses the imaginary axis; a cycle is born with a
  finite period 2π/Im λ and an amplitude growing as the square root of the parameter distance; the second harmonic
  scales as the amplitude squared.
* Fixed-point annihilation (heteroclinic or homoclinic loop) route: two fixed points merge and leave the real axis
  ("conformality lost" [13]); the cycle that remains has finite amplitude and a period that diverges at the
  transition, logarithmically for a loop bifurcation [19, 20] and as a power −1/2 for the RG merger [13, 14].

The project's large-D limit (period 1.6439, amplitude falling, non-first-harmonic power tracking amplitude squared) has
the signature of the first route, with D = ∞ as the bifurcation point. The rest of this note sorts analogues by route.

## A. Where discrete self-similarity occurs outside gravity

**Thin-film rupture with a generalized disjoining pressure** (Dallaston, Fontelos, Tseluiko, Kalliadasis 2018 [1];
Dallaston et al. 2021 [2]). The 1D equation h_t + [h³(h_xx − 1/(n hⁿ))_x]_x = 0 with exponent n modelling the
intermolecular force (n = 3 van der Waals). The rupture profile repeats under rescaling at fixed intervals of
logarithmic time, leaving a geometric cascade of ridges and filaments (satellite and sub-satellite drops). Discrete.
The self-similar profile was continued in n with AUTO-07p; its eigenvalues cross the imaginary axis at n_s ≈ 1.567
(symmetric modes, ±0.912i) and n_a ≈ 1.545 (antisymmetric, ±0.885i), the period at birth is 2π/0.912 ≈ 6.9 and the
period observed by time-stepping at n = 1.5 is about 6.1. The amplitude law and the super/subcritical character are not
reported. In two dimensions [2] the irregular regime is DSS and "possibly chaotic". The authors cite Choptuik as the
precedent. Verdict: same mechanism, and the only non-gravitational case where the Hopf birth of a DSS blow-up has been
computed.

**Wave-kinetic blow-up (gravity-wave turbulence model)** (Schubring, Rosenhaus, Thalabard 2025 [3]). A differential
kinetic model for deep-water waves blows up in finite time; as the phase parameter φ passes φ* ≈ 2.7 the blow-up
changes from CSS to DSS, periodic in τ = −log(t* − t). The mechanism is not analysed (the split of the extrema
Ω₀± is reported, no eigenvalues, no amplitude or period law). Verdict: looks alike, probably the Hopf route, untested.

**Shell model of turbulence** (Campolina, Simonnet, Thalabard 2025 [4]). In the Sabra model's similarity-time
dynamics the reduced system follows, as the scaling exponent x decreases, "stable fixed point → Hopf bifurcation →
stable limit cycle → homoclinic orbit", and the self-similar blow-up profile is the homoclinic trajectory. Verdict:
same mathematics; it also shows the two routes joined, a cycle born at a Hopf point and dying at a loop with diverging
period.

**Incompressible Euler and Navier–Stokes.** Pumir, Shraiman and Siggia [5] and Pomeau and Sciamarella [6] proposed
log-periodic (DSS) blow-up; Eggers and Fontelos [8] record the proposal, with the buoyancy-driven porous-medium flow of
[5] as its model. Chae and Tsai [7] and Chae [7b] give integrability conditions that exclude DSS and asymptotically DSS
blow-up. No DSS Euler singularity is established. Verdict: not an analogue (proposal only).

**Bubble pinch-off** (Schmidt, Keim, Zhang, Nagel 2009 [9]; Turitsyn, Lai, Zhang 2009 [10]). The neck of a
disconnecting air bubble collapses with radius R̄ ∝ (t* − t)^(1/2); an azimuthal shape mode keeps a constant
amplitude A₂ while its phase advances as φ₂ = ln(R₀/R̄) + Ω₂ [10], that is harmonic in log time with angular frequency
1/2 in ln(t* − t) (the conversion is mine). Experimentally the amplitudes "freeze" and the frequencies chirp [9]. The
background is CSS and the oscillation is a linear, neutral mode whose amplitude is set by the initial asymmetry.
Verdict: looks alike. It is the degenerate end point of the Hopf route (a centre about a CSS state), not a limit cycle,
since nothing selects the amplitude.

**Fields with nothing discrete.** Viscous-thread pinch-off and van der Waals rupture (n = 3) are CSS with a countable
family of profiles [8]. Nonlinear Schrödinger blow-up is log-log (critical) or CSS (supercritical) [26], and log-log is
not log-periodic. Keller–Segel blow-up surfaced only as CSS (radial families for 3 ≤ d ≤ 9, stable CSS blow-up in
d = 3 [27]). Porous-medium hole filling has pitchfork bifurcations of the CSS focusing family as the exponent m varies
[28], a spatial symmetry breaking, not a temporal one. Verdict for all four: not analogues.

**Renormalization-group limit cycles, Efimov physics, "conformality lost"** (Efimov [11], Bedaque, Hammer and van
Kolck [12], Kaplan, Lee, Son, Stephanov [13], Mohapatra and Braaten [14], Glazek and Wilson [15], Braaten and Hammer
[16], LeClair, Román, Sierra [17], Bulycheva and Gorsky [18]). The three-body coupling runs on a limit cycle; the
Efimov spectrum repeats by e^(π/s₀) ≈ 22.7 (s₀ = 1.00624 in d = 3 [14]). Discrete. The transition is the fixed-point
merger of [13]: with β = (α − α*) − (g − g*)² the fixed points g* ± √(α − α*) collide, and past the merger the flow is
a cycle of RG period π/√(α* − α), so the log-period diverges and the scale ratio e^(−π/√(α*−α)) tends to zero. With the
dimension d as the parameter [14] the Efimov window is 2.30 < d < 3.76 and s₀ → 0 at both ends (period → ∞), while the
cycle's amplitude is finite (the coupling runs from one Bessel zero to the next). Verdict: same mechanism as the
gravitational sigma-model transition (B below), the opposite route from the project's large-D limit.

**Graphene atomic collapse** (Shytov, Katsnelson, Levitov 2007 [22]; Ovdat et al. 2017 [23]). For Coulomb strength
β = Zα > 1/2 an infinite geometric ladder of quasi-bound states appears, observed by STM around charged vacancies; the
ladder factor is universal. Discrete scale invariance switches on at β = 1/2 by fixed-point merger, with the log-period
diverging at threshold. Verdict: same mechanism as the Efimov case; infinite-period route.

**Hierarchical lattices and quasicrystals** (Derrida, Itzykson, Luck 1984 [24]; Kohmoto, Kadanoff, Tang 1983 [25];
Lifshitz and Even-Dar Mandel 2011 [25b]). Log-periodic critical amplitudes on the diamond hierarchical lattice follow
from the discreteness of the RG map (Julia set); the Fibonacci chain's trace map gives a Cantor spectrum and
log-periodic oscillations of the return probability. The scale factor is built into the lattice. Verdict: looks alike;
no bifurcation, nothing varies.

**Feigenbaum period doubling** (Feigenbaum 1978 [29]). Universal ratios δ = 4.6692 (parameter) and α = 2.5029
(phase space); a fixed point of a discrete RG (the doubling operator), as the Choptuik solution is a fixed point of the
Poincaré map of the RG flow [21]. The scale factors are fixed numbers; there is no small-amplitude limit. Verdict:
looks alike (same formal structure, a discrete RG fixed point), no Hopf content.

**Turbulence intermittency** (Zhou and Sornette 2002 [30]). Log-periodic corrections to the dissipation moments at
Re ≈ 2500 with scale ratio about 2 and up to five harmonics, extracted by a "canonical" averaging; a statistical
signature, contested. Verdict: looks alike.

**Finance** (Johansen, Ledoit, Sornette 2000 [31]; Ide and Sornette 2002 [32]). Log-periodic power laws in
ln(t_c − t) before crashes, with a fitted scale factor; [32] produces them from a 2D ODE whose origin is an unstable
spiral point (approximate log-periodicity, no limit cycle). Verdict: looks alike, nothing to borrow.

**Biology, combustion, optics.** Shlesinger and West [33] fit the bronchial tree's diameters by a power law with a
log-periodic modulation ("complex fractal dimension"), a static hierarchy. Banerjee, Pavithran and Sujith [34] report
log-periodic oscillations of a pressure mode before lean blowout in a combustor. No empirical ecological or epidemic
log-periodicity surfaced, only Sornette's proposals. In optics nothing dynamical surfaced: similaritons are CSS and the
only discrete scale invariance is engineered (log-periodic antennas, multilayer waveguides, sections of [35]).
Verdict: looks alike (hierarchies and precursors, no cycle).

## B. Weakly nonlinear limits and Hopf readings

| System | Parameter | Route | Period at transition | Amplitude at transition | Ref |
|---|---|---|---|---|---|
| Thin film, 1D | exponent n, n_s ≈ 1.567 | Hopf (computed) | finite, 2π/0.912 ≈ 6.9 | not reported | [1] |
| Shell model (Sabra) | scaling exponent x | Hopf then homoclinic | finite at birth, ∞ at death | small at birth | [4] |
| Wave-kinetic model | phase φ, φ* ≈ 2.7 | unanalysed | not reported | not reported | [3] |
| Bubble neck modes | none (linear) | centre about CSS | 4π in ln(t*−t) for n = 2 | free | [9, 10] |
| RG / Efimov in d | dimension d, 2.30 and 3.76 | fixed-point merger | diverges, s₀ → 0 | finite | [13, 14] |
| Graphene Coulomb | β = Zα, 1/2 | fixed-point merger | diverges | finite | [22, 23] |
| Brans–Dicke / 2d sigma model | κ_c ≈ 0.0754 | Hopf, conjectured | finite (conjecture) | shrinks to 0 (conjecture) | [19, 36, 37] |
| Einstein–SU(2) sigma model | η_c = 0.1701 | heteroclinic loop | Δ = −0.389 ln(η−η_c)+b | finite | [20, 38] |
| Choptuik, D → ∞ (project) | 1/D → 0 | Hopf (project's reading) | 1.6439 | 0.32 at D = 73, falling | this note |

Within gravity the Hopf reading has been used once. Hirschmann and Eardley [19] found that the leading mode of the
CSS solution of the self-coupled (sigma-model) scalar field changes stability at κ ≈ 0.0754 and wrote that this
"looks like a Hopf bifurcation"; Gundlach and Martín-García's review [37] restates their picture, "the DSS cycle
smoothly shrinks with growing κ" and disappears "with a finite value of the echoing period". Liebling and Choptuik [36]
saw the DSS-to-CSS change in Brans–Dicke collapse but did not follow the cycle to the transition. No paper found here
has verified the Hopf picture by tracking the DSS amplitude and period to κ_c, so within gravity it remains a
conjecture. The other gravitational transition, the SU(2) sigma model, is the opposite route: Lechner et al. [38]
called it a heteroclinic loop bifurcation (two CSS fixed points collide with the DSS cycle) and Aichelburg, Bizoń and
Tabor [20] measured Δ = a ln|η − η_c| + b with a = −0.389, matching −2/λ_CSS for λ_CSS ≈ 5.14, with a Shilnikov-type
loop conjectured and a fine structure (p_n − p*)/(p_{n+1} − p*) ≈ exp(Δ/2γ_DSS). Frolov [39] obtained an echoing period
from a CSS background alone, Δ = √2 π ≈ 4.44 from the growing perturbation of the Roberts solution, against the actual
3.44 at D = 4; the method is exact only at a bifurcation point. Yang and Zou [40] read DSS as RG limit cycles with the
CSS-to-DSS transition "as the stable and unstable fixed points collide and move to the complex plane" (the merger
route), give Δ = 2π/δ with δ² = 1/4 − K/2 for extremal-Kerr perturbations and a "gravitational Efimov effect" (a
geometric series of quasinormal frequencies); they do not mention Hopf bifurcations or large D.

**What the large-D papers say about the echo.** Ecker, Ecker and Grumiller [41] construct DSS solutions analytically
in 1/(D − 1): at leading order "any periodic function β(τ) with any echoing period Δ solves our equations of motion",
and the next order imposes one constraint on Δ. Their summary says existing data "suggest Δ → 0 as D → ∞, but the
precise rate matters", and "numerical and theoretical arguments suggest lim γ = 1/2, hinting at a CSS limit". The
companion paper [42] (numerics for 3.05 ≤ D ≤ 5.5; Δ = 3.445453 at D = 4, 3.22176 at D = 5, maximum near D ≈ 3.76)
states Conjecture 1, lim Δ = 0⁺ as D → ∞, with Δ falling no faster than D^(−1/3) and possibly like 1/ln D, Conjecture
2, lim γ = 1/2, and Δ ∝ (D − 3)^α near D = 3 with α ≳ 0.15; it also gives max f_c → 1 − O(1/D²). Neither paper uses
the words Hopf, bifurcation, weakly nonlinear, harmonic or sinusoidal, and neither discusses the oscillation amplitude.
Discrepancy to report: the project's Δ(73) = 1.78 with limit 1.6439 contradicts Conjecture 1 of [42]; the two agree
that the limit looks CSS (γ → 1/2 there, vanishing amplitude here). Earlier numerics [43, 44] reach D = 11 and 14 only.

## C. Same mechanism or look-alike, and what to borrow

Same mechanism (a limit cycle of the scaling or RG flow, the echo being the cycle): the thin-film DSS [1, 2], the shell
and wave-kinetic blow-ups [3, 4], the RG limit cycles of Efimov physics and graphene [11–18, 22, 23], and the
gravitational transitions [19, 20, 36–40]. Of these only [1, 4] and the conjecture of [19, 37] are on the Hopf route;
all the Efimov-type cases are on the merger route and are discriminated from the project's limit by their diverging
period. Look-alikes (discrete scale invariance without a selected cycle): bubble neck modes [9, 10] (a centre, not a
cycle), Feigenbaum [29], hierarchical lattices and quasicrystals [24, 25], turbulence and finance statistics [30–32],
anatomical and combustion log-periodicity [33, 34]. Not analogues: Euler DSS proposals [5–7], porous-medium pitchforks
[28], NLS and Keller–Segel (nothing discrete).

What the project can borrow.

1. The Hopf normal form as the test of its reading. With μ = 1/D and the bifurcation at μ = 0 the amplitude obeys
   A² ∝ μ, the period Δ(D) = Δ_∞ + c/D + O(1/D²), and the second-harmonic power ∝ A², which the project already sees
   from D = 16 up. On the project's own values A√D = 2.88, 2.81, 2.73 at D = 45, 61, 73 (drift −5 %) while A·D =
   19.4, 22.0, 23.4 (drift +21 %), so the square-root law fits better than 1/D; this is my arithmetic on quoted
   numbers, not a fit, and the project should fit A² against 1/D with the D ≥ 16 points and check the linearity of Δ
   in 1/D. A nonzero intercept of A² at 1/D = 0 would place the bifurcation at a finite D_c > 73 instead.
2. The bifurcation-point prediction of the period, as in [19, 37, 39]: at D = ∞ a CSS background should carry a pair
   of purely imaginary eigenvalues ±2πi/Δ_∞ = ±3.822i. The leading-order large-D equations of [41] admit any period,
   which is more degenerate than a Hopf point; their next-order constraint on Δ is the natural place to look for
   3.822 as the selected frequency. If the CSS background at D = ∞ exists and its spectrum has no such pair, the Hopf
   reading is wrong.
3. The numerical method of [1, 2]: continue the CSS profile in the parameter with a continuation package, locate the
   imaginary-axis crossing from the eigenvalue problem, then obtain the cycle by time-stepping in similarity variables
   with time-dependent origin shifts (x₀, t₀) and read the period off the orbit; [2] adds Floquet-type stability of the
   cycle by continuation. Near the bifurcation the cycle is better parametrized by its amplitude than by D.
4. The discriminator against the merger route: on that route Δ grows logarithmically [20] or as (α − α_c)^(−1/2)
   [13] and the amplitude stays finite [14]; the project's decreasing Δ and decreasing amplitude exclude it.

## Ranked closest analogues

1. Thin-film rupture with generalized disjoining pressure [1, 2]: a computed Hopf birth of a DSS blow-up from a CSS
   profile, with period 2π/Im λ at birth; the amplitude law is the one thing missing.
2. The Brans–Dicke / 2d sigma-model transition within gravity [19, 36, 37]: the same Hopf conjecture for the same
   kind of object, never verified; the project's large-D limit would be its first confirmed instance if the tests in
   C.1 and C.2 pass.
3. Bubble neck vibrations [9, 10]: a neutral log-periodic oscillation about a CSS collapse, the picture of the
   D = ∞ end point itself, with the amplitude left free because the problem is linear.

## The Hopf reading of the project's large-D limit

Untested in the literature, and neither supported nor contradicted by a direct result. Supported indirectly: the Hopf
route is the standard origin of DSS blow-up in the one non-gravitational case where the transition has been computed
[1], it is the route Hirschmann and Eardley conjectured for the κ transition in gravity [19, 37], and the project's
three signatures (finite period, falling amplitude, second-harmonic power ∝ amplitude²) are the normal form's. The
rough A ∝ D^(−1/2) of the quoted amplitudes (C.1) is consistent with μ = 1/D as the unfolding parameter. Contradicted
in one respect by Ecker et al. [41, 42], who conjecture Δ → 0⁺ as D → ∞ from data at D ≤ 14 and the structure of their
expansion; their γ → 1/2 "CSS limit" agrees with a vanishing-amplitude cycle but not with a finite period. The
Efimov-type literature [13, 14, 20, 22] is not a counter-example, since it concerns the other route, and it supplies the
test that rules that route out here (Δ would have to grow). What would settle it is C.2: a CSS solution of the D = ∞
equations with a purely imaginary eigenvalue pair at ±3.822i, or the next-order constraint of [41] selecting that
frequency.

## References

Depth: abstract | part | full. Verified on arXiv/INSPIRE/Crossref/publisher unless marked (unverified).

- [1] Dallaston, Fontelos, Tseluiko, Kalliadasis, Phys. Rev. Lett. 120, 034505 (2018); arXiv:1609.05938. part.
  https://arxiv.org/abs/1609.05938
- [2] Dallaston, Fontelos, Herrada, Lopez-Herrera, Eggers, J. Fluid Mech. 917, A20 (2021) (unverified, repository
  record only). abstract. https://doi.org/10.1017/jfm.2021.286
- [3] Schubring, Rosenhaus, Thalabard, arXiv:2509.23199 (2025). part. https://arxiv.org/abs/2509.23199
- [4] Campolina, Simonnet, Thalabard, arXiv:2501.07377 (2025). part. https://arxiv.org/abs/2501.07377
- [5] Pumir, Shraiman, Siggia, Phys. Rev. A 45, R5351 (1992); [6] Pomeau, Sciamarella, Physica D 205, 215 (2005).
  Both not read (cited through [8]), confirmed on Crossref.
- [7] Chae, Tsai, Math. Res. Lett. 21, 437 (2014); arXiv:1304.7414. abstract. https://arxiv.org/abs/1304.7414
  [7b] Chae, Nonlinear Anal. 125, 251 (2015); arXiv:1306.0305. abstract.
- [8] Eggers, Fontelos, Nonlinearity 22, R1 (2009) (journal pages unverified); arXiv:0812.1339. part.
  https://arxiv.org/abs/0812.1339 . Book: Eggers, Fontelos, Singularities (CUP 2015), ISBN 9781107098411, not read.
- [9] Schmidt, Keim, Zhang, Nagel, Nature Phys. 5, 343 (2009). abstract (Crossref, seminar abstracts).
  https://www.nature.com/articles/nphys1233
- [10] Turitsyn, Lai, Zhang, Phys. Rev. Lett. 103, 124501 (2009); arXiv:0902.0393. part. https://arxiv.org/abs/0902.0393
- [11] Efimov, Phys. Lett. B 33, 563 (1970), not read (Crossref); [12] Bedaque, Hammer, van Kolck, Phys. Rev. Lett.
  82, 463 (1999), not read (INSPIRE).
- [13] Kaplan, Lee, Son, Stephanov, Phys. Rev. D 80, 125005 (2009); arXiv:0905.4752. part.
  https://arxiv.org/abs/0905.4752
- [14] Mohapatra, Braaten, Phys. Rev. A 98, 013633 (2018); arXiv:1710.08447. part. https://arxiv.org/abs/1710.08447
- [15] Glazek, Wilson, Phys. Rev. Lett. 89, 230401 (2002); arXiv:hep-th/0203088. abstract. [16] Braaten, Hammer,
  Phys. Rep. 428, 259 (2006); arXiv:cond-mat/0410417. abstract.
- [17] LeClair, Román, Sierra, Phys. Rev. B 69, 020505 (2004); arXiv:cond-mat/0211338. not read (INSPIRE).
- [18] Bulycheva, Gorsky, Phys. Usp. 57, 171 (2014); arXiv:1402.2431. abstract. https://ufn.ru/en/articles/2014/2/g
- [19] Hirschmann, Eardley, Phys. Rev. D 56, 4696 (1997); arXiv:gr-qc/9511052. part. https://arxiv.org/abs/gr-qc/9511052
- [20] Aichelburg, Bizoń, Tabor, Class. Quantum Grav. 23, S299 (2006); arXiv:gr-qc/0512136. part.
  https://arxiv.org/abs/gr-qc/0512136
- [21] Hara, Koike, Adachi, arXiv:gr-qc/9607010 (1996, unpublished per INSPIRE). abstract; Koike, Hara, Adachi,
  Phys. Rev. Lett. 74, 5170 (1995). abstract.
- [22] Shytov, Katsnelson, Levitov, Phys. Rev. Lett. 99, 246802 (2007); arXiv:0708.0837. abstract (journal ref via ADS,
  unverified on publisher). https://arxiv.org/abs/0708.0837
- [23] Ovdat, Mao, Jiang, Andrei, Akkermans, Nature Commun. 8, 507 (2017) (journal ref unverified); arXiv:1701.04121.
  abstract. https://arxiv.org/abs/1701.04121
- [24] Derrida, Itzykson, Luck, Commun. Math. Phys. 94, 115 (1984). abstract (Project Euclid).
- [25] Kohmoto, Kadanoff, Tang, Phys. Rev. Lett. 50, 1870 (1983). not read (Crossref). [25b] Lifshitz, Even-Dar
  Mandel, Phil. Mag. 91 (2011) (pages unverified); arXiv:1006.0688. abstract.
- [26] Merle, Raphaël, Szeftel, arXiv:0907.4098. abstract (search snippet only).
- [27] arXiv:2209.11206 (stable Keller–Segel blow-up in d = 3; authors unverified). abstract (search snippet only).
- [28] Angenent, Aronson, J. Amer. Math. Soc. 14 (2001), "Non-axial self-similar hole filling for the porous medium
  equation". abstract (AMS page). https://www.ams.org/jams/2001-14-04/S0894-0347-01-00372-1/
- [29] Feigenbaum, J. Stat. Phys. 19, 25 (1978). abstract (INSPIRE 5488).
- [30] Zhou, Sornette, Physica D 165, 94 (2002) (journal ref unverified); arXiv:cond-mat/0110436. abstract.
- [31] Johansen, Ledoit, Sornette, Int. J. Theor. Appl. Finance 3, 219 (2000); arXiv:cond-mat/9810071. abstract.
- [32] Ide, Sornette, Physica A 307, 63 (2002); arXiv:cond-mat/0106047. abstract. https://arxiv.org/abs/cond-mat/0106047
- [33] Shlesinger, West, Phys. Rev. Lett. 67, 2106 (1991). abstract (Crossref, Scilit).
- [34] Banerjee, Pavithran, Sujith, arXiv:2205.05402; Phys. Rev. E 107, 024219 (2023) (unverified). abstract.
- [35] Sornette, Phys. Rep. 297, 239 (1998); arXiv:cond-mat/9707012. abstract.
- [36] Liebling, Choptuik, Phys. Rev. Lett. 77, 1424 (1996); arXiv:gr-qc/9606057. abstract.
- [37] Gundlach, Martín-García, Living Rev. Relativ. 10, 5 (2007); arXiv:0711.4620. part (first half).
  https://arxiv.org/abs/0711.4620
- [38] Lechner, Thornburg, Husa, Aichelburg, Phys. Rev. D 65, 081501 (2002); arXiv:gr-qc/0112008. abstract.
- [39] Frolov, Phys. Rev. D 61, 084006 (2000); arXiv:gr-qc/9908046. abstract.
- [40] Yang, Zou, Phys. Rev. D 109, 104034 (2024); arXiv:2207.04373. part. https://arxiv.org/abs/2207.04373
- [41] Ecker, Ecker, Grumiller, Phys. Rev. Lett. 136, 191401 (2026); arXiv:2601.14358. part.
  https://arxiv.org/abs/2601.14358
- [42] Ecker, Ecker, Grumiller, Jechtl, SciPost Phys. 20, 176 (2026); arXiv:2602.10185. part.
  https://arxiv.org/abs/2602.10185
- [43] Sorkin, Oren, Phys. Rev. D 71, 124005 (2005); arXiv:hep-th/0502034. abstract.
- [44] Bland, Preston, Becker, Kunstatter, Husain, Class. Quantum Grav. 22, 5355 (2005); arXiv:gr-qc/0507088. abstract.
