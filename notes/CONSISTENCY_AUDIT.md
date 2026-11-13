# Consistency audit of statements and numbers (#653)

Audited commit: **b73b80e87556f4c966fbe7373b8608401ced70ec** (main, 2026-11-13). Subagent of the coordinator, on the
Mac. Reading and comparison only. The computations were `check_numbers.py --selftest`, the link checker
`check_links.py`, and small scripts that read the data files and the section references.

Each finding gives its locations, the canonical source, a verdict (inconsistent, superseded, unsupported, broken
reference) and a diff. Every diff below applies to the audited commit on its own (`git apply --check`), and all of them
together apply in this order as one patch. With all of them applied, every text that `check_numbers.py` quotes is still
present and every check passes. Nothing here was applied to the README, the papers or the notes. The coordinator applies
them.

The new checks are a separate commit (the end of this note).

## Findings, most severe first

### 1. LARGE_D_WALL.md describes error bars that `delta_all.csv` no longer has (superseded)

| location | says |
|---|---|
| notes/LARGE_D_WALL.md:211–216 | from D 215 each chain_fit err is at least the projector shift, the largest err from 232.63 on, "2.6×10⁻⁹ at D 242.57", err_kind "projector first order"; values "unchanged until a chain on main's exact projector, from D 225 on, replaces them" |
| hp/data/delta_all.csv (9c34843c) | rows 225.13–245.64 come from `I603-win2` on the exact projector; err 3.8×10⁻¹¹–4.4×10⁻¹¹, err_kind "fits d6" or "fits extrapolated"; no row at 242.57; no err_kind "projector…" in any row |

Canonical: `hp/data/delta_all.csv`, written by `hp/export_delta_all.py`, whose docstring says the exact-projector runs
"carry no projector term". The replacement the note calls pending was made in 9c34843c.

```diff
--- a/notes/LARGE_D_WALL.md
+++ b/notes/LARGE_D_WALL.md
@@ -208,12 +208,11 @@
 1% wherever it was tested, though no state lay inside its chord window: the window measures the start, not Newton's
 basin. On the rounded rows, too, #562's homotopy traces stayed straight in Δ far past their chord windows.
 
-**The table's error bars include it.** The chain_fit rows of `hp/data/delta_all.csv` solve the rounded rows, so from
-D 215 on each row's err is at least this shift, log-linear in |shift| between the points of
-`hp/data/projector_shift.csv` (0b67ed9). It is the largest err from D 232.63 on: 4.3×10⁻¹³ there, 3.5×10⁻¹² at
-D 235.13 (where the runs' spread was 2.6×10⁻¹⁴), and 2.6×10⁻⁹ at D 242.57. Past D 235.13 the err rests on first-order
-figures (err_kind "projector first order"). The values are unchanged until a chain on main's exact projector, from
-D 225 on, replaces them (Nathan's decision, 2026-11-12).
+**The table's error bars.** From D 225.13 on, the chain_fit rows of `hp/data/delta_all.csv` come from the rerun on
+main's exact projector (#603; commit 9c34843c), so they carry no projector term. Below D 225.13 the rows solve the
+rounded rows, and the shift there, at most 7.8×10⁻¹⁶ (`hp/data/projector_shift.csv`), is below the fits term, which
+sets their err (`hp/export_delta_all.py`). Until 9c34843c the rounded rows ran to D 242.57, and the shift was their
+largest err from D 232.63 on (4.3×10⁻¹³ there, 2.6×10⁻⁹ at D 242.57, err_kind "projector first order").
 
 **The weak σ oscillate in D.** The equilibrated σ₁ and σ₂ oscillate with a period that shortens as D falls: 2.715 at
 D 232.6, 2.525 at 221.6 and 2.440 at 212.8 (#525; `I525-win2/` at 5037ec07). σ₃ and σ₄ have periods of 2.13–2.14
```

### 2. The README does not say that the top 21 chain error bars are extrapolated (unsupported bar)

| location | says |
|---|---|
| README.md:532–534 | "Its error column is the spread over the fit's degree where a rerun on the fits of degree 6 and 10 reached a D" |
| hp/data/delta_all.csv | err_kind "fits d10" 74–129, "fits d6" 130–230.13, "fits d6 interp" 191.5–231.13, **"fits extrapolated" 232.13–245.64 (21 rows, 4.0–4.4×10⁻¹¹)** |
| hp/export_delta_all.py:60–62 | above the comparison chains' frontier "the fits term is extrapolated log-linearly in D from the last six rows that have one (… an estimate)" |

The values at the frontier, the ones the README's headline reach rests on, carry an extrapolated estimate, not a
measured spread. The same paragraph's "to D = 215" is the extent of the series comparison (QUAD_DOUBLE.md 13.29), which
the chain now passes by 30 units of D. Canonical: the CSV and the script.

```diff
--- a/README.md
+++ b/README.md
@@ -529,9 +529,11 @@
 - the full system in double-double (D ≤ 67.77) or quad-double (68.5–73);
 - the pinned double-double runs after one full-system step (67.77–73.27);
 - the large-D chain from 74. The chain holds harmonics 5–13 of p1 on a fit of the full-system states (kind chain_fit).
-  It agrees with the 1/D series of the double-double points to within the series' own spread, to D = 215. Its error
-  column is the spread over the fit's degree where a rerun on the fits of degree 6 and 10 reached a D
-  ([notes/QUAD_DOUBLE.md](notes/QUAD_DOUBLE.md), 13.25).
+  It agrees with the 1/D series of the double-double points to within the series' own spread, to D = 215, where that
+  comparison ends. Its error column is the spread over the fit's degree where a rerun on the fits of degree 6 and 10
+  reached a D, directly or interpolated in 1/(D − 2), to D = 231.13. Above that, to 245.64, it is an estimate
+  extrapolated log-linearly in D from the last six measured rows (err_kind "fits extrapolated";
+  `hp/export_delta_all.py`, [notes/QUAD_DOUBLE.md](notes/QUAD_DOUBLE.md), 13.25).
 
 How the held harmonics' bias was found, removed and measured is in [notes/BIAS.md](notes/BIAS.md).
 `cd hp && python3 export_delta_all.py ../../choptuik-large-d-raw` rebuilds the file, and `--verify` checks every row
```

### 3. "Limits and open problems" still has the chain at D = 85.7 on a trend (superseded)

| location | says |
|---|---|
| README.md:1960–1962 | the Taylor continuation "carries the chain to D = 85.7", harmonics "held on a trend", values "5×10⁻⁸ to 1.4×10⁻⁷ below the 1/D series" |
| README.md:17, 76 | chain to D = 245.6, harmonics held on a fit (check `chain_reach`) |
| notes/BIAS.md:12; notes/QUAD_DOUBLE.md 13.29 | the fit removes the trend's bias; the chain_fit bias is +1.2e-14 at 80 and −3.1e-12 at 215 |

Canonical: README:17 and 76 (hp/data/delta_all.csv, highest chain_fit D 245.6366).

```diff
--- a/README.md
+++ b/README.md
@@ -1958,8 +1958,10 @@
   points supplies it ([notes/LARGE_D_WALL.md](notes/LARGE_D_WALL.md); [notes/QUAD_DOUBLE.md](notes/QUAD_DOUBLE.md),
   sections 12–13).
   - An exact Taylor continuation in D (the asymptotic numerical method, with the dimension entered as a jet through the
-    kernel) with a quad-double corrector carries the chain to D = 85.7. It does so only with harmonics 5–13 of the
-    centre profile held on a trend, and those values lie 5×10⁻⁸ to 1.4×10⁻⁷ below the 1/D series (section 13.19 there).
+    kernel) carries the chain past the wall, to D = 245.6 ([ALGORITHM.md](ALGORITHM.md), section 3). It does so only with
+    harmonics 5–13 of the centre profile held on a fit of the full solutions. Held on a trend instead, they put Δ
+    5×10⁻⁸ to 1.4×10⁻⁷ below the 1/D series by D = 85.7 (section 13.19 there), the bias that the fit removes
+    ([notes/BIAS.md](notes/BIAS.md)).
   - The wall is the digits the horizon block needs, which grow linearly with D, not the horizon series itself. In a toy
     of the horizon's two branches (`hp/horizon_toy.py`), the series keeps the full precision up to D ≈ 200. An integral
     equation from the horizon converges only after its first (D − 4)/2 Taylor terms are subtracted, and shooting outward
```

### 4. The physics paper quotes the extrapolated γ(3.5), which the direct value replaced (superseded)

| location | γ(3.5) | kind |
|---|---|---|
| paper/physics/main.tex:296 | 0.3235 ± 0.0030 | extrapolated from 3.64 ≤ D ≤ 4, an estimator range |
| notes/LOW_D.md:160–163 | 0.3235 ± 0.0030 | the same, with no pointer to the direct value |
| README.md:652 | 0.3223856, 3.8×10⁻⁹ from M = 96 to 128 | double-double, M = 128 |
| hp/data/delta_dd_lowD.csv, D 3.5 | 0.3223855828 (M 128), 0.3223855790 (M 96) | the source (check `gamma_35`) |

The direct value lies inside the old range, so nothing contradicts. The paper's "Open" list (main.tex:804) also still
names "γ just below D = 4", while delta_dd_lowD.csv has γ at 25 values of D from 3.5 to 4. That item is removed in
finding 10's diff, which edits the same sentence.

```diff
--- a/paper/physics/main.tex
+++ b/paper/physics/main.tex
@@ -293,8 +293,9 @@
 Kunstatter's $0.4131 \pm 0.0001$~\cite{BlandKunstatter2007}. Below $D = 4$ the double-double solver, with its matching nodes moved to $x = 0.005$
 and $0.8$, gives $\Delta(3.7) = 3.46520048 \pm 3\times10^{-8}$ and $\gamma(3.7) = 0.34865019 \pm 8\times10^{-8}$, converged in
 the $\tau$ resolution (48, 64 and 80 points); our 47-mode f64 points there are $2$--$5\times10^{-6}$ high in $\Delta$.
-Extrapolated from $3.64 \le D \le 4$, $\gamma(3.5) = 0.3235 \pm 0.0030$, in agreement with~\cite{EckerEtAl2026}
-($0.32239$) and not with $0.349 \pm 0.003$ in~\cite{BlandEtAl2005}. The modes needed grow from about 73 at $D = 4$ to
+At $D = 3.5$ the double-double solver gives $\gamma = 0.3223856$, which moves by $3.8\times10^{-9}$ from 96 to 128
+points in $\tau$, in agreement with~\cite{EckerEtAl2026} ($0.32239$) and not with $0.349 \pm 0.003$
+in~\cite{BlandEtAl2005}. The modes needed grow from about 73 at $D = 4$ to
 about 110 at $D = 3.5$ and several hundred near $D = 3.05$; at $D = 3$ the scalar field decouples from gravity in these
 equations, so the family cannot reach a regular $D = 3$ limit.
 
--- a/notes/LOW_D.md
+++ b/notes/LOW_D.md
@@ -162,6 +162,8 @@
   estimator range). The rule was first applied to Ecker et al.'s gamma curve at the same D and contained their
   gamma(3.5) = 0.32239 (error 1.1e-3); `--mutate` (a curve turning by 0.01) fails it. Our points sit within 8e-7 of
   their curve.
+- **gamma(3.5) directly** (`hp/data/delta_dd_lowD.csv`, #628): 0.3223855828 at M = 128, 3.8e-9 from M = 96. It
+  supersedes the extrapolation above, inside whose range it lies.
 - Against Bland et al. 2005 (Table I): 0.349 +- 0.003 at 3.5 is 0.0255 above, more than four times the two ranges added;
   Ecker et al.'s direct 0.32239 and Bland's 2007 thesis value 0.3235 +- 0.0018 *(unverified, from Ecker et al.'s data
   file)* agree with ours. So the 2005 value at 3.5 looks wrong; the README's "gamma(3.5) <~ 0.33, an open check" can be
```

### 5. BIAS.md has the chain stopping at 215 and an error-kind table that no longer matches (superseded)

| location | says | now |
|---|---|---|
| notes/BIAS.md:10–11 | chain_fit "reaches D = 215" | 245.6 (delta_all.csv) |
| notes/BIAS.md:39 | fits spread used for "chain_fit, 74–150" | 74–231.13, then "fits extrapolated" to 245.64 |
| notes/BIAS.md:75–76 | T148, T149 "will show whether the step misses a harmonic" | done: QUAD_DOUBLE.md 13.29 (harmonics 5, 7 to 1e-4; 9–13 missed) |
| notes/BIAS.md:84–85 | "The chain stops at 215 … T150 reruns it … toward D 400" | 245.6, exact projector from 225.13 (#603) |

The bias step itself stops at 215 (`hp/data/release_steps.csv`), so the diff says so rather than extend the claim.

```diff
--- a/notes/BIAS.md
+++ b/notes/BIAS.md
@@ -8,8 +8,9 @@
 Where it stands, as numerical evidence:
 
 - The chain (kind chain_fit, `hp/data/delta_all.csv`) holds the harmonics on a 1/n fit of the full-system states and
-  reaches D = 215.
-- Its bias, measured by a full-system step, is +1.2e-14 at D 80, at most +5.7e-13 up to 160 and −3.1e-12 at 215.
+  reaches D = 245.6 (#603).
+- Its bias, measured by a full-system step, is +1.2e-14 at D 80, at most +5.7e-13 up to 160 and −3.1e-12 at 215. The
+  step has not been taken above 215 (`hp/data/release_steps.csv`).
   The trend-held chain before it was biased by −5e-8 at 80 and −3.2e-6 at 200.
 - The step returns planted biases of harmonic 5 to 1e-5 of themselves, but between two held fits it removes only
   93–99% of the difference. So it is a measurement of the bias, not yet a correction.
@@ -36,7 +37,8 @@
 | starts | two starts with their own Taylor plans, a spread | quad-double full system, 68.5–73 |
 | pins | pin 7 against pin 5, two systems, a spread | pinned runs, 67.77–73.27 |
 | runs | two step paths (T139), a spread | chain_fit, where no fit spread exists |
-| fits | held fits of degree 6 and 10 against 8 (T140), a spread | chain_fit, 74–150 |
+| fits | held fits of degree 6 and 10 against 8 (T140, #608), a spread, direct or interpolated in 1/(D − 2) | chain_fit, 74–231.13 |
+| fits extrapolated | the fits term extrapolated log-linearly in D from the last six rows that have one, an estimate | chain_fit, 232.13–245.64 |
 | step | the junk-frozen full-system step's dΔ, a linear estimate of the bias | chain_fit, every 5 in D (not in err) |
 
 None is a bound. The largest available is the row's err (`hp/export_delta_all.py`).
@@ -72,8 +74,8 @@
 
 ## Open
 
-- **The remainder in 13.** Moves of single held coefficients (T148, T149) will show whether the step misses a
-  harmonic other than 5 or loses part of the response to the frozen junk.
+- **The remainder in 13.** Moves of single held coefficients (T148, T149; QUAD_DOUBLE 13.29) return harmonics 5 and 7
+  to 1e-4 and miss most of the response of harmonics 9–13, a likely source of the remainder.
 - **Above D 130.** No planted test reaches there. On the trend-held chain the step agreed with the exact shift to
   0.5–2% (13.29), at biases a million times larger than chain_fit's.
 - **Resolution.** The chain runs at M = 32 only. At D 80 its state, resampled to M = 48 in τ and solved again in
@@ -81,8 +83,8 @@
   `resolution-mac/scan_chainfit_D80.log`), so resolution is far below the bias there. Above D 80 the check needs the
   horizon block re-seeded at the new M: without it the start residual grows from 1.7e-9 at 80 to 2e15 at 200.
 
-- **Range.** The chain stops at 215, where the octo-double corrector's weakest pivot reaches 7e-16
-  (`T138-win2/chain_od.log`). T150 reruns it with the junk frozen in the corrector, toward D 400.
+- **Range.** The first chain stopped at 215, where the octo-double corrector's weakest pivot reaches 7e-16
+  (`T138-win2/chain_od.log`). The chain now reaches 245.6, on the exact projector from 225.13 (#603).
   The crawl past the wall (I19-win2, D = 214.6 to 215.03, 7 rows of delta_all.csv) accepted its steps with 4 junk
   directions frozen. Reloaded with the crawl's options and no step, 1 of its 7 saved states and 1 of 3 saved
   predictions converge again, and those two give the crawl's Δ and correction to the printed digits. The other 8 stop
```

### 6. The math paper's review status and outlook lag the proof notes (superseded, and inconsistent within the paper)

| location | says | canonical |
|---|---|---|
| paper/math/main.tex:167 (status table) | Theorem A(M = 3) "not independently reviewed" | main.tex:1876 and proof/A_M_review.md: reviewed in part |
| main.tex:136 | B(1): "a first independent review confirms it" | two reviews, P2_review.md and P2b_review.md (README:2006–2008) |
| main.tex:1319–1322 | P2's second implementations of LC's closure and J's secants "are not complete" | P2b_review.md §0: both certified by a second implementation |
| main.tex:1884–1885 | at M = 5 a certificate "must be posed on the Schur-corrected conditions, with multiprecision Lohner" | proof/A_M2.md: certified with a second-order chart on the D-jump conditions |
| main.tex:2204–2209 (Outlook 1–2) | B(1)'s missing items and reviews; Theorem A(M = 5) and the M = 3 review as future work | B(1) certified and reviewed twice; A(M = 5) certified and reviewed |

Canonical: proof/C_selection.md §13.1, proof/P2b_review.md, proof/A_M2.md, proof/A_M2_review.md. The paper does not
contain Theorem A(M = 5) at all (README:11–12 describes it correctly as A(M = 1) and A(M = 3)), so the diff only stops
the paper from calling it future work.

```diff
--- a/paper/math/main.tex
+++ b/paper/math/main.tex
@@ -133,7 +133,7 @@
 The two contraction arguments among them, the global continuation and the inner--outer matching, are proved for each
 fixed $M$ and carried out with computer-assisted constants for $M = 1$ and $n \ge e^{73.89}$. For $M = 1$ the theorem is
 certified, assuming Braaksma's theorem, for $n \ge e^{89.75}$, with the light cone handled in the exact light-cone gauge
-(Section~\ref{sec:B1status}); a first independent review confirms it with corrections that change no constant.
+(Section~\ref{sec:B1status}); two independent reviews confirm it, with corrections that change no constant.
 \item For the one-harmonic truncation, the selection condition has exactly one zero in an explicit box of half-width
 $10^{-9}$, and it is simple (Theorem~\ref{thm:A}). The proof is computer-assisted, assumes only Braaksma's summability
 theorem, and follows Baldom\'a, Capi\'nski, Guardia and Seara~\cite{BaldomaCapinskiGuardiaSeara2023}.
@@ -164,7 +164,7 @@
 Theorem~\ref{thm:B} & selection for the $M$-harmonic model & conditional on the lemmas of Table~\ref{tab:Bstatus}; for $M = 1$ certified, assuming Braaksma's theorem, $n \ge e^{89.75}$ (Section~\ref{sec:B1status}); reviewed in part \\
 Lemmas B1, B5.1, B6$'$, B8.1, B8.2; B2, B5.0 & ingredients of Theorem~\ref{thm:B} & proved; B2 and B5.0 checked symbolically \\
 Theorem~\ref{thm:A} & one simple zero of the $M = 1$ selection condition & certified (computer-assisted), assuming Braaksma's theorem \\
-Section~\ref{sec:A3} & one simple zero of the $M = 3$ selection condition & certified (computer-assisted), assuming Braaksma's theorem; not independently reviewed \\
+Section~\ref{sec:A3} & one simple zero of the $M = 3$ selection condition & certified (computer-assisted), assuming Braaksma's theorem; reviewed in part \\
 Lemma~\ref{lem:sigma1} & enclosure of the first amplitude coefficient $\sigma_1$ & certified (computer-assisted,\newline
 \repo{exact\_sigma.py}); not independently reviewed \\
 Section~\ref{sec:evidence} & $\omega^*/\winf = 0.999975$--$1.00005$ at fourteen to twenty-two orders (no zeros of $S_1$ nearer the origin than $x_0$ assumed) & numerical evidence \\
@@ -1318,8 +1318,9 @@
 Lemma~B6 and the factor $\kappa_{n,\mu}$ are not needed in this form. Every certificate is in ball arithmetic and has a mutation control that makes it
 fail. A first independent review (\repo{proof/P2\_review.md}) finds that the theorem stands as stated, with seven
 corrections that change no constant or threshold (three of them to what the class of Lemma~B8's continuation up to the
-light cone must contain); its second implementations of that continuation's closure and of the jump lemma's secant bounds
-are not complete. The threshold $n_0$ is far beyond any dimension a numerical solver reaches, so the theorem does not by
+light cone must contain). A second review (\repo{proof/P2b\_review.md}) completes the second implementations of that
+continuation's closure and of the jump lemma's secant bounds and confirms both. The threshold $n_0$ is far beyond any
+dimension a numerical solver reaches, so the theorem does not by
 itself identify the computed branch.
 
 \subsection{Gap (b): the harmonic tail}
@@ -1881,8 +1882,8 @@
 \subsection{What is not claimed}
 Theorem~\ref{thm:A} is a statement about the one-harmonic truncation ($M = 1$), not about the physical problem: its zero is
 $22\%$ below $\winf$ in frequency, and it does not survive the addition of harmonic $3$. Nothing is said about parameters outside
-$X$. The disc covering has no second implementation. At $M = 5$ the higher harmonics leak into the lower conditions with weights up
-to $10^{17}$, so a certificate there must be posed on the Schur-corrected conditions, with multiprecision Lohner integration.
+$X$. The disc covering has no second implementation. At $M = 5$ the first-order certificate fails, and a second-order chart
+with $C^2$ enclosures certifies the zero instead (\repo{proof/A\_M2.md}, reviewed in \repo{proof/A\_M2\_review.md}).
 
 % =====================================================================================================
 \section{The untruncated multiplier as an entire function}
@@ -2201,12 +2202,11 @@
 harmonics in place of a scalar parameter. The linear version of the whole chain (outer late orders, inner Stokes constant,
 light-cone coefficient) is proved. What would turn the selection rule into a theorem, in order of difficulty:
 \begin{enumerate}[leftmargin=2em]
-\item \emph{Theorem~\ref{thm:B} for $M = 1$}: the missing items of Section~\ref{sec:B1status}, chiefly a certified error bound
-in Lemma~B5(2) (or a matching domain of $n$-independent size, \repo{proof/B5\_inner.md}, \S5.4) and independent reviews of
-the certificate code; then the same for each fixed $M$. A proof of (H-$\sigma$) would identify the centre-regular solution
-with the Borel sum of the outer series, but the theorem does not need it.
-\item \emph{Theorem A($M = 5$)}, on the Schur-corrected conditions; an independent review of the $M = 3$ certificate
-(Section~\ref{sec:A3}); and a second implementation of the disc covering.
+\item \emph{Theorem~\ref{thm:B} for each fixed $M > 1$}, as for $M = 1$ (Section~\ref{sec:B1status}), and reviews of the
+certificate code not yet reviewed there (B5(1), B8 beyond the normalization run, (H8)). A proof of (H-$\sigma$) would
+identify the centre-regular solution with the Borel sum of the outer series, but the theorem does not need it.
+\item \emph{Theorem A($M = 5$)} in the paper (certified in \repo{proof/A\_M2.md}); a review of the $M = 3$ majorant code
+and disc certificates (Section~\ref{sec:A3}); and a second implementation of the disc covering.
 \item \emph{The untruncated zero}: the harmonic-tail lemma of Section~\ref{sec:tail} with $M = 7$ explicit, or rigorous
 $\sigma_k$, $k \le 14$, with a proof of the exponential type of $S_1$ (Section~\ref{sec:entire}).
 \item \emph{Uniformity in the harmonic index} (gap (b)) for the selection theorem itself, in a space of functions analytic in
```

### 7. MODE_COUNT.md's verdict and the physics paper keep the D = 30 pair open (superseded)

| location | says |
|---|---|
| notes/MODE_COUNT.md:21–25 (Verdict) | the extra pair at D = 30, M = 48 is "open", "a third resolution is needed to close it" |
| paper/physics/main.tex:795–796 | "which we take as a truncation artifact pending a third resolution" |
| notes/MODE_COUNT.md:447–452 (§5) | closed: on the fitted layout the pair is absent and the count is 0 in all 8 boxes (#250, #260) |
| README.md:2036–2043 | the same closure, plus open extra zeros at D = 54, 55 and 61 at M = 48, absent at M = 40 and 64 |

Canonical: MODE_COUNT.md §5. The physics paper also omits the D = 54, 55, 61 extras that the README reports.

```diff
--- a/notes/MODE_COUNT.md
+++ b/notes/MODE_COUNT.md
@@ -21,8 +21,8 @@
 One item is open. At a second resolution, M = 48 at D = 30, the count is the same except for an **extra conjugate
 pair at 4.267 ± 1.324i**. The pair is absent at M = 32, and absent at M = 48 when the centre data's τ-harmonics above
 13 are pinned (the cut the production runs use from D = 35 on). Without the cut, the M = 48 background converges only
-to |F| = 9×10⁻¹⁸. Everything points to an artifact of the solver's weakly coupled high-harmonic directions, but a third
-resolution is needed to close it (section 5). At D = 4 (f64 solver, real λ only) det M changes sign only at
+to |F| = 9×10⁻¹⁸. It was closed in section 5 (#250, #260): on a fitted layout the pair is absent and the count at
+M = 48 without the cut is 0 in every box, as at M = 32. At D = 4 (f64 solver, real λ only) det M changes sign only at
 λ_g = 0.6139 and λ_phys = 2.6741 on 0.02 < λ < 6.47.
 
 What is not covered: Re λ > 6.6, perturbations whose τ-harmonics lie above what the layout holds (above 31, and for the
--- a/paper/physics/main.tex
+++ b/paper/physics/main.tex
@@ -792,8 +792,9 @@
 determinant in $-0.08 < \mathrm{Re}\,\lambda < 6.18$, over a full period in $\mathrm{Im}\,\lambda$ (both symmetry sectors), finds
 besides $\lambda_{\rm phys}$ only the $\tau$ translation ($\lambda = 0$) and a gauge mode of the formulation at
 $\lambda = 1 + \langle c_1\rangle$; a planted-zero test of the same code counts correctly in all 8 cases. A second resolution
-shows one extra pair at $D = 30$, absent at the production resolution and with the high-harmonic cut, which we take as a
-truncation artifact pending a third resolution. Lower $D$ with complex $\lambda$, larger $\mathrm{Re}\,\lambda$ and
+shows one extra pair at $D = 30$, absent at the production resolution and with the high-harmonic cut; on a layout whose
+centre series converges it is absent at that resolution too. Extra zeros at $D = 54$, 55 and 61 at the second resolution
+are absent at a third, as truncation artifacts are. Lower $D$ with complex $\lambda$, larger $\mathrm{Re}\,\lambda$ and
 non-spherical perturbations are not covered, and no collapse has been evolved above $D = 14$, so a continuously
 self-similar solution~\cite{ClarkPimentel2025} taking over the threshold is not excluded; in that case $1/\lambda$ would
 not be the critical exponent. For the axion--dilaton the corresponding check was made at $D \le 40$
```

### 8. PROOF.md calls a certified theorem "proved" and lists finished work as targets (inconsistent status word, superseded)

| location | says | canonical |
|---|---|---|
| PROOF.md:208 | "**Theorem A(M = 1) is proved**" | certified (README:210; AGENTS.md: proved = by hand, certified = computer-assisted); PROOF.md:222–223 itself says "Theorem A(M = 3) is certified" |
| PROOF.md:214–215 | the M = 5 certificate "must use the Schur-corrected conditions … multiprecision Lohner" | the second-order chart did it (PROOF.md:225–229, proof/A_M2.md) |
| PROOF.md:230–231 | a stray "Theorem" at the end of line 230, before "Lemmas 6a(5)" | editing remnant |
| PROOF.md:235–237 | "Staged targets: **A(M=5)** …" | done (same section) |
| PROOF.md:42–43 (Summary) | target (A), the M = 5 zero, as future | done |

Lines 205–207 also say lemma 6a and 6b₁ are "proved", for certified results. The diff leaves those, since "proved by a
covering with 178 validated Taylor discs" names the method, but the coordinator may want "certified" there too.

```diff
--- a/PROOF.md
+++ b/PROOF.md
@@ -40,8 +40,8 @@
   $`e^{\pi\lvert\nu\rvert\omega/2}`$ that grow faster than the profile's harmonics decay (gap (b)). Neither template has
   that. Costin–Tanveer's Borel summability for PDEs with data analytic in a strip is the closest tool.
 - **Realistic targets, in order:** (A) a computer-assisted proof that the truncated (M = 5) inner multiplier has a
-  simple zero in an explicit small box around (A, ω/ω∞) = (2.7186, 0.9992), then a proof with a rigorous harmonic-tail
-  bound for the untruncated multiplier; (B) the selection theorem for large n, first for a model with finitely many
+  simple zero in an explicit small box around (A, ω/ω∞) = (2.7186, 0.9992), now certified (§4), then a proof with a
+  rigorous harmonic-tail bound for the untruncated multiplier; (B) the selection theorem for large n, first for a model with finitely many
   harmonics, then in a weighted space of functions analytic in a θ-strip; (C) then follows as a corollary of (B), with
   no further numerics. (A) is a substantial but standard computer-assisted proof; (B) is a research-level paper in
   exponential asymptotics; together they would derive a critical-collapse echo period, the large-D limit Δ∞, rigorously
@@ -205,14 +205,14 @@
 result and certifies the box ±10⁻⁸ as well, ten times larger, with the zero enclosed eight to ten times more tightly.
 Since then (§11) lemma 6a is proved, 6b is proved near ξ = 0, and the certificate is redone on a lower path that needs
 analyticity only for |ξ| ≤ 0.2. Lemma 6b₁, the analyticity of y⁺ on the annular sector 0.0624 ≤ |ξ| ≤ 0.2, is proved by
-a covering with 178 validated Taylor discs (§12). **Theorem A(M = 1) is proved**, assuming Braaksma's summability
+a covering with 178 validated Taylor discs (§12). **Theorem A(M = 1) is certified**, assuming Braaksma's summability
 theorem: the one-harmonic selection condition has exactly one zero in the box, it is simple, and it lies at A ∈
 [2.54214804761338, 2.54214804761975], ω/ω∞ ∈ [0.776994013607438, 0.776994013612010]. Not yet done: an independent
 re-implementation of the disc covering. Larger boxes fail because the interval Jacobian's excess grows like 7×10⁵·h (the
 solution is large where the ray meets the arc), so 10⁻³ boxes need a C² Lohner step; a lower path (ray to 0.2, then the
 real axis) should allow 10⁻⁸. The integration is stiff (fast modes shrink by e⁻³¹, so naive interval Taylor would
 overestimate by about e⁵⁰), which is why the Lohner wrapping is essential. At M = 5 the system is 38-dimensional, and
-the certificate must use the Schur-corrected conditions (leakage up to 10¹⁷), i.e. multiprecision Lohner.
+the certificate that works there is the second-order chart below.
 
 *Toward M = 3 and 5* ([proof/A_M.md](proof/A_M.md), [proof/A_lemmas_M.md](proof/A_lemmas_M.md)). The point certificate
 is **certified** at M = 3: the four real D-jump conditions at ξ₁ = 0.3 have exactly one zero, with a non-singular
@@ -227,14 +227,13 @@
 lemma and Braaksma's theorem, exactly one simple zero of the D-jump function in a curved parallelepiped of outer
 half-width 10⁻¹⁰, with A ∈ [2.7185791892117, 2.7185791892118] and ω/ω∞ ∈ [0.99922840320656, 0.99922840320657]. The
 same code reproduces Theorem A(M = 1) and the M = 3 certificate on boxes 10² to 2×10⁴ times larger. An independent
-review with its own integrator (proof/A_M2_review.md) finds the claim holds; its issues are in the controls. Theorem
+review with its own integrator (proof/A_M2_review.md) finds the claim holds; its issues are in the controls.
 Lemmas 6a(5) (at the certificate's p̂, #184, after the review #39 found the first run centred on x̂) and 6b₁(5) (independently reviewed, #39) are certified, so A(M = 5) holds under the same assumptions; a second validated integrator (CAPD, #189, #208) reproduces the flows of both lemmas, every row of 6b₁(5) included. An independent review (proof/A_M_review.md) confirms the
 M = 3 point certificate, H1–H3 and lemma 6a; the majorant code of lemmas S and 6b₀ and the 6b₁ disc certificates are
 not yet reviewed.
 
-Staged targets: **A(M=5)**, a certified simple zero of the five-harmonic multiplier (lemmas A.1, A.2, A.4 only; a
-standard computer-assisted proof); then **A(∞)**, the untruncated zero, adding A.3, or by the entire-function route
-below.
+Staged targets: **A(M=5)** is done (above); next is **A(∞)**, the untruncated zero, adding A.3, or by the
+entire-function route below.
 
 **Route A3: the entire-function certificate** ([proof/D_entire.md](proof/D_entire.md), `solver/proof/entire_cert.py`).
 The coefficients follow σ_k/σ_{k+1} ≈ (k + β)/κ⋆ with κ⋆ ≈ 1.28 + 1.08i and β ≈ 4.2 − 1.6i, the same at ω/ω∞ = 0.99,
```

### 9. PROOF_QUEUE.md's status column is behind (superseded)

| location | says | canonical |
|---|---|---|
| notes/PROOF_QUEUE.md:10 (P1) | B(1) certified, "not yet reviewed (P2)" | P2 and P2b reviewed it (same file, row P2) |
| notes/PROOF_QUEUE.md:13 (P3) | Theorem A(M = 5): "scoping note written", design decision pending | certified (proof/A_M2.md), reviewed (A_M2_review.md), lemmas 6a(5), 6b₁(5) (#184, #39) |

The file says "The coordinator keeps this file current". It is in `check_numbers.py`'s DOCS, but no quoted text is in
these cells.

```diff
--- a/notes/PROOF_QUEUE.md
+++ b/notes/PROOF_QUEUE.md
@@ -7,10 +7,10 @@
 
 | # | Item | Why it is needed | Where | Status |
 |---|---|---|---|---|
-| P1 | Close Theorem B(1), the selection theorem for the one-harmonic model | the whole chain from the inner zero to a solution exists for M = 1 except the links in C_selection.md §13.1 | cloud agent | **done** (2026-10-23; merged cba0f62 and 7d25496): with P1L, Theorem B(1) is certified for M = 1 for n ≥ e^{89.75}, assuming Braaksma's theorem; not yet reviewed (P2) |
+| P1 | Close Theorem B(1), the selection theorem for the one-harmonic model | the whole chain from the inner zero to a solution exists for M = 1 except the links in C_selection.md §13.1 | cloud agent | **done** (2026-10-23; merged cba0f62 and 7d25496): with P1L, Theorem B(1) is certified for M = 1 for n ≥ e^{89.75}, assuming Braaksma's theorem; reviewed by P2 and P2b |
 | P1L | The light cone: gaps (a)–(c) of Theorem B(1) | P1 found that the remaining links rest on three open problems at x = 1: (a) control of the nonlinear solution in the half-discs near the light cone, (b) the exact light-cone gauge c(1) = 1, (c) a proof that the outer jump is fast-type; without them Theorem B(1) is conditional | the P1 agent | **done** (2026-10-23, merged 7d25496): Lemma LC (B8 into the light-cone half-disc in the exact gauge c(1) = 1, a contraction, θ = 0.7355, gauge offset ≤ 3.59×10⁶n⁻⁵) gives (a) and (b); Lemma J (θ = 0.176) gives (c); proof/B_chain.md §§13–15 |
 | P2 | Independent review and second implementations of the certificate code | no certificate of B5, B8 or (H8) has been reviewed, nor P1's lemma L, B5(2), Lemma P, B7, Lemma LC and Lemma J (B_chain.md §10 lists the code and the hand arguments); lemma 6b₁'s disc covering has not been re-implemented; at M = 3 the majorant code of lemmas S and 6b₀ and the 6b₁ discs are not reviewed | cloud agent that has not seen P1 | **stopped by the user** (2026-10-23 17:28) after a first full draft, merged: Theorem B(1) stands as stated, with corrections F1–F7 that change no quoted constant or threshold (proof/P2_review.md §6; not yet applied to the notes and code); not done: second implementations of Lemma LC's closure bound algebra and of Lemma J's secant coefficients, and the older B5(1)/(H8) code and the M = 3 items; P2b (2026-11-05, [proof/P2b_review.md](../proof/P2b_review.md), merged): Lemma LC's closure algebra and Lemma J's secant coefficients re-implemented in mpmath interval arithmetic and confirmed, F1–F7 applied with every rerun's quoted constant unchanged; still not checked: B5(1)'s code, B8 beyond --d0, (H8), lemma 6b₁'s discs, the M = 3 items and Lemma J's θ₀ |
-| P3 | Theorem A(M = 5) | the five-harmonic zero, the first truncation close to the real answer (0.999228); the point certificate fails with first-order enclosures | cloud agent | scoping note written (2026-10-28, [proof/P3_scope.md](../proof/P3_scope.md)): the derivative radius grows 10⁸ on the real leg (fast directions), the harmonic-5 rows fail by 542 and the outer rows by 14; three designs. The M = 5 point run is done (T90, 2026-10-28, P3_scope.md §4): its derivative radius reaches 12 at ξ₁ against the box run's 3×10⁸, with the same growth along the path (10¹⁹) and a constant factor 10⁷ from the box, so the amplification is the enclosure's own and 12 already exceeds what Krawczyk tolerates; decision: design 2 (fast-direction-aware sets, cloud agent, two weeks, judged by the point run) first, design 3 beside it, design 1 only if the box runs then still fail with h |
+| P3 | Theorem A(M = 5) | the five-harmonic zero, the first truncation close to the real answer (0.999228); the point certificate fails with first-order enclosures | cloud agent | **done**: certified by the second-order chart ([proof/A_M2.md](../proof/A_M2.md)), reviewed ([proof/A_M2_review.md](../proof/A_M2_review.md)), lemmas 6a(5) and 6b₁(5) certified (#184, #39). Earlier: scoping note written (2026-10-28, [proof/P3_scope.md](../proof/P3_scope.md)): the derivative radius grows 10⁸ on the real leg (fast directions), the harmonic-5 rows fail by 542 and the outer rows by 14; three designs. The M = 5 point run is done (T90, 2026-10-28, P3_scope.md §4): its derivative radius reaches 12 at ξ₁ against the box run's 3×10⁸, with the same growth along the path (10¹⁹) and a constant factor 10⁷ from the box, so the amplification is the enclosure's own and 12 already exceeds what Krawczyk tolerates; decision: design 2 (fast-direction-aware sets, cloud agent, two weeks, judged by the point run) first, design 3 beside it, design 1 only if the box runs then still fail with h |
 | P4 | σ₁₅ at ω/ω∞ = 0.99, 1.00, 1.01 | numerical evidence: tests the ring of fifteen zeros that σ₁₄ left open, and narrows the estimator range | local machine (T44, after T43) | **done** (2026-10-25): σ₁₅ converged to 5×10⁻¹³ (T44, the Mac); the K = 15 analysis (notes/PLANTED_SELECTION.md §7) rejects every ring with zeros nearer than x₀, one ring with zeros just beyond x₀ survives (its zero inside the range), and the headline over K = 14, 15 is ω*/ω∞ = 0.999975–1.00005 (Δ∞ = 1.64387 ± 0.00006) |
 | P8 | σ₁₆ at ω/ω∞ = 0.99, 1.00, 1.01 | numerical evidence: tests the one ring of zeros that survived σ₁₅ (zeros at 7.3 < \|x\| < 12, notes/PLANTED_SELECTION.md §7.9) and narrows the range 0.999975–1.00005 | the Mac (T83) | done (2026-10-27, §8): that ring is rejected by σ₁₆, two rings built from σ₀…σ₁₅ survive in turn, the K = 16 range is 0.999980–1.000014, the headline unchanged |
 | P9 | σ₁₇ and σ₁₈ at ω/ω∞ = 0.99, 1.00, 1.01 | numerical evidence: tests the two rings that survived σ₁₆ (notes/PLANTED_SELECTION.md §8.4) and adds the K = 17, 18 ranges; the D-bracket question (§8.9) is fixed in §9.0 item 4 before the data (§5.3's rule as the primary reading, a second reading without D-bracket beside it) | the Mac (T86, T87) | σ₁₇ done (2026-10-28, §9.1–9.9): both rings rejected, no ring survives, K = 17 hull 0.999990–1.000009, headline unchanged. σ₁₈ done (2026-10-28, §10.1–10.9; §10.0 and its dry run were committed before the data): the two rings of the K = 17 generation rejected under both readings, no ring survives, K = 18 hull 0.999992–1.000004 inside the K = 17 range, headline unchanged, G below 1 for the first time (0.9999987), the projection favours ω* = 1 by 0.4×10⁻⁶ against a model error of 1.1×10⁻⁵ (not decisive); σ₁₉ (T88) tests C18-BP [3/15], [4/14], [5/13]: the nodes land 2026-10-28 about 23:00 UTC, but pilot (b) failed §11.0's agreement gate by 1.21 at σ₁₉ (20:55 UTC; a 1.3×10⁻¹¹ difference, nothing to the estimators), and T91 (2026-10-29 02:01 UTC) split the difference into a J effect of 1.85×10⁻²⁷ and a precision effect of 1.94×10⁻²⁷, each two thirds of the plateau error; item 1 was amended (dated 2026-10-29 02:12 UTC, §11.0) to add each node's pilot difference to its err19, which at wf 0.99 and 1.01 comes from T89's node runs (about 2026-10-30 02:00–12:00 UTC), so the K = 19 stages run then; §12.0, the K = 20 procedure, committed 2026-10-30 before T89's nodes landed; σ₁₉ done (2026-10-31, §11.1–11.9, four dated amendments at the end of §11.0): the ten K = 18 rings rejected, K = 19 range 0.999996–1.000004, headline unchanged; σ₂₀ next (§12.0); σ₂₀ done (2026-10-31, §12.1–12.10): the ten K = 19 rings rejected under the primary reading (C19-BP [4/15] by D-bracket alone), K = 20 range 0.999997–1.000002, headline unchanged; σ₂₁ (T94, the four K 43 files landed 2026-11-02): §13.0 committed 2026-10-31 before any σ₂₁ existed, its dry run passed (§13.1), the stages ran 2026-11-02 (§13.2–13.10): all ten K = 20 rings rejected under both readings, K = 21 range 0.999998–1.000002, headline unchanged; σ₂₂ (T99 and T104 on win2, landed 2026-11-03/04): §14.0 committed 2026-11-02 before any σ₂₂ existed, its dry run passed (§14.1), the stages ran 2026-11-04 (§14.2–14.10): all nine K = 21 rings rejected under both readings, K = 22 range 0.9999988–1.0000004, headline unchanged; no further order queued |
```

### 10. The physics paper lists an independent check above D = 40 as missing (superseded in part)

| location | says |
|---|---|
| paper/physics/main.tex:154–156 | "Above D = 40 these checks are internal to the double-double solver" |
| paper/physics/main.tex:803–804 (Open) | "an independent check of the double-double points above D = 40"; "γ just below D = 4" |
| notes/RT_FORMULATION.md §10.9, §11.7 | a collocation code in the Reiterer–Trubowitz formulation reproduces Δ to 3×10⁻¹⁵ at D = 40 and 50, 2×10⁻¹⁴ at 55, and 3.5×10⁻¹³ at 60 (32 × 99) |

The README does not mention the RT check either (its proposed text, §10.9 and §11.7, was not merged). What the two codes
share is the start of Newton, which at D = 60 decides between two nearby roots (§11.4), so "independent" needs that
caveat. The diff also drops "γ just below D = 4" (finding 4) and narrows "its 1/n coefficient" (finding 11), since the
three edits fall in one sentence.

```diff
--- a/paper/physics/main.tex
+++ b/paper/physics/main.tex
@@ -151,9 +151,10 @@
 $5.5\times10^{-10}$. The checks are
 listed in Table~\ref{tab:checks}. The 857 points fit smooth curves to about $4\times10^{-15}$; the estimated error per
 point is about $10^{-16}$ below $D = 27$ and up to a few $10^{-15}$ above (from the weak directions on $27$--$35$, the
-pinning above $D = 35$, and the relaxed tolerance above $D = 60.1$). Above $D = 40$ these
-checks are internal to the double-double solver (resolution, pinning, a second build on a second machine); the
-independent double-precision solver stops at $D = 40$.
+pinning above $D = 35$, and the relaxed tolerance above $D = 60.1$). The independent double-precision solver stops at
+$D = 40$. Above it, besides the double-double solver's own checks (resolution, pinning, a second build on a second
+machine), a collocation code in Reiterer and Trubowitz's formulation reproduces $\Delta$ to $3\times10^{-15}$ at
+$D = 40$ and $50$ and to $3.5\times10^{-13}$ at $D = 60$ (in the repository, \texttt{notes/RT\_FORMULATION.md}).
 
 Why the extra precision is needed is itself a physical result. The sensitivity of $\Delta$ to the residual,
 $\max|\partial\Delta/\partial F|$, grows about 1.8-fold per unit of $D$ below $D = 27$ and 1.5-fold at $D = 56$--$61$, to
@@ -800,8 +801,8 @@
 (Sec.~\ref{sec:matter}).
 
 \emph{Open.} Higher orders of the selection rule, which would give $\Dinf$ (and possibly a closed form) to more digits;
-a proof of the rule (Conjecture~1 of~\cite{math}); the uniqueness of the $\lambda = 2$ mode at $D = \infty$ and its $1/n$ coefficient; an
-independent check of the double-double points above $D = 40$; $\gamma$ just below $D = 4$; and the $\ell = 1$ multiplet
+a proof of the rule (Conjecture~1 of~\cite{math}); the uniqueness of the $\lambda = 2$ mode at $D = \infty$ and its $1/n$
+coefficient without truncation; and the $\ell = 1$ multiplet
 beyond $D = 9$ and the $\sigma$-model's DSS branch beyond $D = 11$, which would show whether a centrifugal term moves $\gamma_\infty$ away
 from $1/2$.
 
```

### 11. The physics paper says λ₁ would need the O(1/n) inner equations, which have been solved in truncation (superseded)

| location | says |
|---|---|
| paper/physics/main.tex:595–596 | 3.0324 is "a coefficient that would need the O(1/n) inner equations" |
| README.md:1823–1838; notes/FIRST_ORDER.md:30–34; GAMMA.md:184–188 | λ₁ = 7.76, 3.080, 3.023 at M = 1, 3, 5 from the O(1/n) inner problem, against the fitted 3.0324 |

Canonical: notes/FIRST_ORDER.md (table at lines 30–34).

```diff
--- a/paper/physics/main.tex
+++ b/paper/physics/main.tex
@@ -592,8 +592,9 @@
 $\lambda = 2$ and the gauge mode $\lambda = 1$, and $\lambda(D)$ is continuous at $1/n = 0$, then $\gamma_\infty = 1/2$.}
 The uniqueness of the unstable mode at $D = \infty$ is not shown. At finite $D$ the outer region and the $1/n$
 corrections to Eqs.~\eqref{eq:heat} depend on $r$ explicitly, so the symmetry is broken and the mode survives as an
-ordinary unstable mode with $\lambda = 2 + O(1/n)$; the data give $2 + 3.0324/n$ [Eq.~\eqref{eq:lambda}], a coefficient
-that would need the $O(1/n)$ inner equations.
+ordinary unstable mode with $\lambda = 2 + O(1/n)$; the data give $2 + 3.0324/n$ [Eq.~\eqref{eq:lambda}]. The $O(1/n)$
+inner equations truncated to $M = 1$, 3 and 5 harmonics give $7.76$, $3.080$ and $3.023$, approaching it (numerical
+evidence, conditional on the selection rule at that order); without truncation it is not computed.
 
 \subsection{The selection rule at $D = \infty$}
 \label{sec:selection}
```

### 12. The main-results table names the wrong bar for the 1199 double-double points (inconsistent)

| location | bar |
|---|---|
| README.md:91 | "change from the next resolution, per point" |
| hp/data/delta_all.csv, dd rows | err_kind "dM" for 36 rows at 3.04375–8.0; **"check" for all 1199 rows at 8.25–67.77** |
| hp/export_delta_all.py:37–39 | "check": "the size of the resolution and harmonic-freezing checks near this D" |
| README.md:238–239 | the figure's shading uses "the change from the next resolution, the nearby checks, or the difference between two runs" (correct) |

```diff
--- a/README.md
+++ b/README.md
@@ -88,7 +88,7 @@
 
 | Result | Value | Status | The bar | Where |
 |---|---|---|---|---|
-| Δ(D), γ(D) computed | 1199 double-double points, D = 8.25–67.77; 67 values of γ; points below D = 4 to 3.044 | numerical evidence | change from the next resolution, per point | [Values](#values), [Below D = 4](#below-d--4-and-the-limit-d--3) |
+| Δ(D), γ(D) computed | 1199 double-double points, D = 8.25–67.77; 67 values of γ; points below D = 4 to 3.044 | numerical evidence | below D = 8.25 the change from the next resolution, per point; above, the size of the resolution and freezing checks near each D | [Values](#values), [Below D = 4](#below-d--4-and-the-limit-d--3) |
 | One formula for all D ≥ 4 | a rational function of 1/(D − 2), 11 constants for Δ and 9 for γ | numerical evidence (an empirical fit) | its misfit on points it never saw | [The formula](#the-formula) |
 | The limit of Δ | Δ∞ = 1.643893746 ± 1.5×10⁻⁸ | conditional on Δ analytic in 1/D | the spread of 48 fit choices, corrected by a planted-truth test | [Is the period analytic in 1/D?](#is-the-period-analytic-in-1d) |
 | The limit of γ | γ∞ = 1/2 to about 5×10⁻⁷ | numerical evidence; explained given Conjecture 1 and one unstable mode | fit spread | [Proposition 5](#proposition-5-the-unstable-mode-at-infinite-d-is-exact), [GAMMA.md](GAMMA.md) |
```

### 13. The physics paper gives the double-double solver a range its own Values table exceeds (inconsistent)

| location | says |
|---|---|
| paper/physics/main.tex:145 | "Double-double solver (8.25 ≤ D ≤ 61.03)" |
| paper/physics/main.tex:243–244 (Table tab:values caption) | Δ and γ at D = 5–8 double-double (T79), D = 4 double-double at M = 192 |

The paper's data otherwise stop at 61.03, while the README has the full system to 67.77 and the chain to 245.6. That is
a narrower scope, not a contradiction, and is left to the coordinator.

```diff
--- a/paper/physics/main.tex
+++ b/paper/physics/main.tex
@@ -142,7 +142,7 @@
 extrapolated in the mode count. Above $D \approx 26$ rounding at $10^{-16}$ in the residual moves $\Delta$ by
 $10^{-7}$--$10^{-6}$ at $D = 30$--$40$.
 
-\emph{Double-double solver} ($8.25 \le D \le 61.03$): double-double arithmetic~\cite{Dekker1971} (about 32 digits) on SIMD vectors;
+\emph{Double-double solver} ($8.25 \le D \le 61.03$ for the fits, and the values at $D = 4$--$8$ of Table~\ref{tab:values}): double-double arithmetic~\cite{Dekker1971} (about 32 digits) on SIMD vectors;
 Taylor series in $x$ of order 36 with truncation $10^{-32}$; centre and horizon series to $10^{-33}$; collocation in
 $\tau$ on the half period (32 points); an exact Jacobian by forward-mode duals~\cite{Wengert1964} with a double-double LU; continuation in
 $D$ with deflated Newton steps. The Jacobian has a ladder of near-null directions (high $\tau$-harmonics of the centre
```

### 14. "Each row has … an error estimate" is false for 761 rows of `delta_all.csv` (inconsistent)

| location | says |
|---|---|
| README.md:521–522 | each row has "an error estimate and what that estimate is" |
| hp/data/delta_all.csv | err empty in 761 of 2339 rows: 733 pinned (only one pin reached that D), 24 dd (3.72–3.8 at M 80, and the M 48 "unresolved" rows at 4.25–8.2), 2 qd, 2 chain_fit (173.5, 190.5) |

```diff
--- a/README.md
+++ b/README.md
@@ -519,7 +519,7 @@
 
 **Every value of Δ** we computed, from D = 3.04375 to the large-D chain's frontier, is in
 [`hp/data/delta_all.csv`](hp/data/delta_all.csv), one row per D. Each row has Δ in up to four parts (their sum is the
-value), the kind of computation, an error estimate and what that estimate is, a published value at that D where one
+value), the kind of computation, an error estimate and what that estimate is where one exists, a published value at that D where one
 exists (with our Δ minus it), and the source run and state in the [raw data](#raw-data). Every published value,
 including Ecker, Ecker and Grumiller's table to D = 200[^23], is compared with ours in
 [`hp/data/delta_published_compare.csv`](hp/data/delta_published_compare.csv).
```

### 15. Bars in the README that do not say what they are (unsupported bar)

README.md:87 says "Each ± says what kind of bar it is". These do not:

| location | quoted | kind, from the source |
|---|---|---|
| README.md:462 | (1.6 ± 2.2)×10⁻⁶ | the least-squares standard error of the constant term (hp/gamma_compare.py:76–77) |
| README.md:1607 | limit 0.999985 ± 2×10⁻⁵ | from c₇'s scatter over repeated solves, conditional on the per-pair ratio (proof/B_tail_M7.md:211, 583) |
| README.md:1934 | 1.9226 ± 0.0003 | a spread over the fit's degree, m = 3–5 (matter/SIGMA.md:365) |
| README.md:474 | 1.64395 ± 0.00008 | no source found (solver/notes.md:405 quotes it without a kind) |
| README.md:639 | 3.221701945 ± 7×10⁻¹¹ extrapolated | no source found (not in hp/compare/dd_D5.log) |

The diff names the first three. The last two need their sources found first (recommended task 5).

```diff
--- a/README.md
+++ b/README.md
@@ -459,7 +459,8 @@
   10⁻⁸–10⁻¹². From M = 32 to 48, λ moves by less than the root's resolution, 10⁻¹⁴ at D = 50 and 5×10⁻¹³ at D = 58
   (`hp/gamma_m48_gap.py`, raw I266-mac).
 - In free fits in powers of 1/n, λ∞ − 2 (λ = 1/γ) falls by a factor of about 3.5 with each added power. On D ≥ 30 it is
-  2.4×10⁻⁴, 6.7×10⁻⁵, 1.9×10⁻⁵, 5.5×10⁻⁶ and (1.6 ± 2.2)×10⁻⁶ for four to eight powers, with no sign of levelling off
+  2.4×10⁻⁴, 6.7×10⁻⁵, 1.9×10⁻⁵, 5.5×10⁻⁶ and (1.6 ± 2.2)×10⁻⁶ (the fit's standard error) for four to eight powers, with
+  no sign of levelling off
   (`cd hp && python3 gamma_compare.py data/gamma_dd.jsonl data/local/gamma_dd.jsonl`). Fixing λ∞ = 2 fits better with
   the same number of parameters.
 
@@ -1604,7 +1605,8 @@
   third estimator the same: S₁ is numerically an entire function of A² of exponential type, and the series of
   G = e^{−κ̄A²}S₁ (same zeros, no cancellation) puts the real zero at 0.99988, A₀ = 2.7194
   ([proof/D_entire.md](proof/D_entire.md)). The truncation sequence alternates and converges (0.777, 1.0156, 0.999228,
-  1.000022; limit 0.999985 ± 2×10⁻⁵, [proof/B_tail_M7.md](proof/B_tail_M7.md)).
+  1.000022; limit 0.999985 ± 2×10⁻⁵, the scatter of M = 7's repeated solves, given that the per-pair ratio stays near
+  M = 7's, [proof/B_tail_M7.md](proof/B_tail_M7.md)).
 - *How far to trust those estimates.* A planted-truth test ([notes/PLANTED_SELECTION.md](notes/PLANTED_SELECTION.md))
   ran the same estimators, through the same code, on synthetic functions of the same class: Laplace integrals over a
   Borel arc whose coefficients match the real σ₀…σ₁₂ to 10⁻⁵, with the zero planted at ω/ω∞ = 0.996–1.004. The pair
@@ -1931,7 +1933,8 @@
 the σ-model's DSS branch.
 
 A by-product supports the selection rule in a second model. At η = 0.14 the zero of the D = ∞ Stokes multiplier for the
-CSS ground state, A₀ = 1.9225686, matches the finite-D ground state extrapolated to D = ∞, 1.9226 ± 0.0003.
+CSS ground state, A₀ = 1.9225686, matches the finite-D ground state extrapolated to D = ∞, 1.9226 ± 0.0003 (a spread
+over the fit's degree).
 
 **σ-model, echoing (DSS) branch** ([matter/SIGMA_DSS.md](matter/SIGMA_DSS.md)). Continued in 1/η from the ℓ = 1
 solution (the σ-model's limit as η → ∞), the branch reproduces Husa et al.'s D = 4 values: Δ to 0.6% or better for
```

### 16. The Theorem A(M = 3) details still say M = 5 needs a second-order design (superseded)

| location | says |
|---|---|
| README.md:1439–1440 | "At M = 5 the same certificate fails … It needs a second-order design ([proof/A_M.md])" |
| README.md:99, 212; proof/A_M2.md | the second-order chart certifies it |

```diff
--- a/README.md
+++ b/README.md
@@ -1436,8 +1436,8 @@
   intervals from the saved box, checked single Lohner steps and the start values against 60–75-digit references,
   confirmed hypotheses H1–H3 and reran lemma 6a. It found no error that changes a number or a verdict.
 - Not yet reviewed, here or at M = 1, are the majorant code of lemmas S and 6b₀ and the 6b₁ disc certificates.
-- At M = 5 the same certificate fails, because the interval enclosures grow too fast along the arc. It needs a
-  second-order design ([proof/A_M.md](proof/A_M.md)).
+- At M = 5 the same certificate fails, because the interval enclosures grow too fast along the arc. A second-order
+  design certifies it instead ([proof/A_M2.md](proof/A_M2.md)).
 - The first coefficient σ₁ of the untruncated condition is also enclosed rigorously (±3.5×10⁻⁵).
 
 Evidence beyond the linear case comes from the amplitude expansion ([next subsection](#the-inner-problem-at-large-x)).
```

### 17. Which machines ran the computations (inconsistent)

| location | says |
|---|---|
| README.md:2080–2081 | "his own machines (two Windows computers and two Macs)" |
| paper/physics/main.tex:812 | "on his own machine" |
| hp/data/delta_fits.csv, D 226.13–230.13 | sources `I608-dh610/…`, a Linux machine (AGENTS.md:46 names the Linux runner dh370); these differences are the err of five delta_all.csv rows |
| paper/ai/main.tex Table tab:machines | five machines at its commit (cloud, win, mac, win2, macm3) |

The date in the diff is when the Linux runners joined (the coordinator's records). The coordinator should confirm it.

```diff
--- a/README.md
+++ b/README.md
@@ -2078,7 +2078,7 @@
 ## How this work was done
 
 Nathan Moinvaziri chose the problem, set the direction at each step and ran the long computations on his own machines
-(two Windows computers and two Macs). The code, the computations, the derivations, the proofs and the text in this
+(two Windows computers, two Macs and, from 12 November 2026, two Linux computers). The code, the computations, the derivations, the proofs and the text in this
 repository were produced by Claude, Anthropic's AI model, working as a set of coordinated agents. These were cloud
 sessions at first, then a coordinating session on one Mac, runner sessions on the other machines, and subagents for
 separate tasks. The commits up to 5 November 2026 are marked as co-authored. He has not checked the mathematics himself.
--- a/paper/physics/main.tex
+++ b/paper/physics/main.tex
@@ -809,7 +809,7 @@
 \section{How this work was done}
 \label{sec:how}
 
-N.~M. chose the problem, set the direction at each step and ran the long computations on his own machine. The code, the
+N.~M. chose the problem, set the direction at each step and ran the long computations on his own machines. The code, the
 computations, the derivations, the proofs and the text, including this paper, were produced by Claude, Anthropic's AI
 model, working as a set of coordinated agents. N.~M. has not checked the mathematics himself, and no human expert has
 reviewed the work.
```

### 18. LOW_D.md's summary gives the lowest converged D as 3.7 (superseded, dated)

| location | says |
|---|---|
| notes/LOW_D.md:3, 9 | "Status of 2026-10-22"; "Lowest D with a demonstrated-converged Delta: D = 3.7"; D = 4 at M = 80 |
| hp/data/delta_all.csv; README.md:418, 520 | lowest 3.04375 (M = 384, err 8.8×10⁻¹¹, dM); D = 4 at M = 192 |

The status is dated, so this is history. The README sends readers there for "the converged values below D = 4"
(README:368), so a pointer at the top is proposed.

```diff
--- a/notes/LOW_D.md
+++ b/notes/LOW_D.md
@@ -6,6 +6,9 @@
 
 ## Summary
 
+- **Since this status:** the double-double points reach D = 3.04375 (M = 384, 8.8e-11 from M = 320;
+  `hp/data/delta_all.csv`, sections 11–15), and D = 4 is converged at M = 192 (1.1e-22 from M = 160). The items below are
+  the status of 2026-10-22.
 - **Lowest D with a demonstrated-converged Delta: D = 3.7** (double-double solver): Delta(3.7) = 3.46520048 +- 3e-8
   (M = 80: 3.4652004835; the bar is the last M difference, the M = 48 / 64 / 80 steps contracting 33-fold), unchanged to
   < 1e-16 by two node layouts with longer series and tighter Taylor steps (Sec. 3). gamma(3.7) = 0.34865019 +- 8e-8
```

### 19. ℓ = 1 runs past D = 9 are in the data but in no note (unsupported)

| location | says |
|---|---|
| README.md:1915–1916; matter/ELL.md:216–227 | the double-precision continuation stops at D ≈ 9; six points D = 4–9 |
| solver/matter/data/march_l1kc23hi.jsonl | a k_c 23 continuation with 16 records from D 9.05 to 10.5 (and `_reg`, 12 to 9.15) |
| solver/matter/data/ell_gamma.jsonl (listed in ELL.md:334) | a second row: D = 10, k_c 23, κ 0.7, Δ = 0.98497934, λ = 2.91496595 (γ = 0.34306), abs(F) 1.4×10⁻¹³ |

Only these files and `solver/matter/logs/t31_super.log` mention the run (`kc23hi`). No note says whether it is converged
or why it is left out. If it holds, it fits "Δ peaks near 0.988 at D ≈ 9 and then falls" (README:1922). The diff records
it in ELL.md as unused and unchecked.

```diff
--- a/matter/ELL.md
+++ b/matter/ELL.md
@@ -198,7 +198,9 @@
 
 Additional Δ points from the k_c = 23 continuation: Δ(8.15) = 0.9851763, Δ(8.375) = 0.9867067, Δ(8.7125) = 0.9879556,
 Δ(9.05) = 0.9881824. (D − 3)γ is the mass exponent. Data: `data/l1_kc23r.jsonl`, `data/l1_D4_kc*.jsonl`,
-`data/l1_kc23_gamma.jsonl`; table from [`ell_table.py`](../solver/matter/ell_table.py).
+`data/l1_kc23_gamma.jsonl`; table from [`ell_table.py`](../solver/matter/ell_table.py). A k_c = 23 continuation past
+D = 9 (`data/march_l1kc23hi.jsonl`, D 9.05 to 10.5) and its solution at D = 10 in `data/ell_gamma.jsonl` (κ = 0.7,
+Δ = 0.9849793, λ = 2.914966, abs(F) 1.4×10⁻¹³) are neither used nor checked here.
 
 **Accuracy.** τ truncation: k_c 23 → 31 at D = 6 changes Δ by 7×10⁻⁸ and λ by 2.6×10⁻⁷. RK steps: at D = 8, κ = 1 → 0.5
 moves Δ by 3.4×10⁻⁶ (∝ κ⁴), so κ = 0.7 carries ~1×10⁻⁶; horizon start u = 0.01 → 0.04 moves it by 4×10⁻⁷
```

### 20. Figure captions at D 232.6 and a 3×10⁻¹¹ chain bar (superseded, figures)

| location | says | now |
|---|---|---|
| README.md:943–948 | "The family to D 232.6" | the chain reaches 245.6 |
| ALGORITHM.md:20 | error bar "to 3×10⁻¹¹ on the chain" | up to 4.4×10⁻¹¹ (delta_all.csv) |
| ALGORITHM.md:54, 118, 125 | D 232.63 or 232.6 in three figures' alt texts | describes the figures as drawn |

The figures themselves need regenerating (recommended task 2). The diff only marks the README caption as a snapshot and
corrects the bar.

```diff
--- a/README.md
+++ b/README.md
@@ -942,7 +942,8 @@
 
 ![The solutions of all D as one family, animated: p1 over one echo period swept from D 38 to 232.6, the waves times √(D−2) nearly coinciding, each odd harmonic a nearly fixed factor smaller than the one before, and p over the radius and the echo phase with p1 its slope at the centre](hp/wave_family.svg)
 
-**The family to D 232.6** (`hp/plot_wave_family.py`). Times √(D−2), the first harmonic of p1 falls by 10% from D 38
+**The family to D 232.6**, where the chain stood when the figure was drawn (`hp/plot_wave_family.py`; the chain now
+reaches 245.6). Times √(D−2), the first harmonic of p1 falls by 10% from D 38
 to 232.6 (3.106 to 2.782), against 65% without the factor, on its way to the extrapolated 2.72. Within one wave each odd
 harmonic is a nearly fixed factor smaller than the one before: from harmonic 1 to 3 by 10.5 at D 38 and 16.3 at D
 232.6. Above D 73 harmonics 5–13 are held on a fit ([ALGORITHM.md](ALGORITHM.md), section 3), so there only harmonics
--- a/ALGORITHM.md
+++ b/ALGORITHM.md
@@ -17,7 +17,7 @@
    in D. Each number is therefore stored as a sum of 2, 4 or 8 ordinary doubles, and the larger D gets, the more parts
    each stage needs.
 
-![The large-D computation in six panels: Δ for every D; a solution as a wave of odd harmonics; the full equations' grip on harmonics 5 to 13 from D 79.8 to 232.6, up to thousands of times weaker per harmonic and weaker still as D grows; the smooth family that a fit of the full solutions carries past D 73; the size of each Taylor term of one chain step; and Δ's error bar, from about 10⁻¹⁵ for the full system to 3×10⁻¹¹ on the chain](hp/big_picture.svg)
+![The large-D computation in six panels: Δ for every D; a solution as a wave of odd harmonics; the full equations' grip on harmonics 5 to 13 from D 79.8 to 232.6, up to thousands of times weaker per harmonic and weaker still as D grows; the smooth family that a fit of the full solutions carries past D 73; the size of each Taylor term of one chain step; and Δ's error bar, from about 10⁻¹⁵ for the full system to 4×10⁻¹¹ on the chain](hp/big_picture.svg)
 
 **The large-D computation in one picture** (`hp/plot_big_picture.py`). The top row is the problem, and the bottom row
 is how a fit gets around it. Each panel is explained in detail here:
```

### 21. A cross-reference points at the wrong section (broken reference)

| location | says |
|---|---|
| README.md:1975 | "the lowest converged point so far is 3.04375 (The formula, the limit graph)", with The formula linked to the anchor `#the-formula` |
| README.md:355–435 | the D → 3 readings, the limit graph and 3.04375 are in "Below D = 4 and the limit D → 3" |

```diff
--- a/README.md
+++ b/README.md
@@ -1972,7 +1972,7 @@
   ([Is the period analytic in 1/D?](#is-the-period-analytic-in-1d), [notes/DINF_FITS.md](notes/DINF_FITS.md)).
 - **The limit D → 3.** Our polynomial in √(D − 3) keeps Δ(3) finite, between 1.72 and 1.80 over the fit degrees. Ecker
   et al. read a power law Δ → 0. The readings separate below D ≈ 3.04, where the lowest converged point so far is
-  3.04375 ([The formula](#the-formula), the limit graph).
+  3.04375 ([Below D = 4 and the limit D → 3](#below-d--4-and-the-limit-d--3), the limit graph).
 - **The proof status.** The selection rule is a conjecture with hypotheses H1–H2 and gaps (a)–(c). The one-harmonic
   theorem B(1) is certified assuming Braaksma's theorem, with a threshold n ≥ e^{89.75} far beyond any computed D, so it
   does not yet identify the computed branch ([PROOF.md](PROOF.md), [notes/PROOF_QUEUE.md](notes/PROOF_QUEUE.md)).
```

### 22. 99.4% of the power "from D ≈ 45 up" (inconsistent, minor)

| location | says |
|---|---|
| README.md:102 | "a single harmonic from D ≈ 45 up (99.4% of the power)" |
| README.md:259; hp/data/echo_shapes.csv | 99.21% at 45.05, 99.36% at 60.96, 99.42% at 73.27 (check `echo_shapes`) |

```diff
--- a/README.md
+++ b/README.md
@@ -99,7 +99,7 @@
 | The five-harmonic point certificate | one simple zero, ω/ω∞ ∈ [0.99922840320656, 0.99922840320657] | certified, assuming Braaksma's theorem and the start-value lemma; reviewed (proof/A_M2_review.md); lemmas 6a(5) and 6b₁(5) are certified at it (#184, #39), so Theorem A(M = 5) holds under the same assumptions; a second validated integrator (CAPD) reproduces its flows (#189, #208) | rigorous enclosures | [PROOF.md](PROOF.md) |
 | D = 4 and the maximum of Δ | agreement with the published values; the maximum at D = 3.755725 | numerical evidence | published uncertainty; fit spread | [Relation to published work](#relation-to-published-work) |
 | The limit D → 3 | a finite Δ(3) (1.72–1.80) or Δ → 0: open | open (two readings) | fit-degree spread | [Below D = 4 and the limit D → 3](#below-d--4-and-the-limit-d--3) |
-| The echo's shape | a single harmonic from D ≈ 45 up (99.4% of the power), a pulse near D = 3 | numerical evidence | — | [Results](#results) |
+| The echo's shape | a single harmonic from D ≈ 45 up (99.2–99.4% of the power), a pulse near D = 3 | numerical evidence | — | [Results](#results) |
 
 In detail:
 
```

### 23. Octo-double's digits: 126 or 127 (inconsistent, minor)

| location | says |
|---|---|
| ALGORITHM.md:176 | about 127 digits, unit roundoff about 10⁻¹²⁸ |
| notes/OCTO_DOUBLE.md:6; paper/ai/main.tex:58; paper/ai/SOURCES.md:43 | about 126 digits |

Eight 53-bit parts are 424 bits, 127.6 digits, the convention ALGORITHM.md's table uses for the other rows (106 bits,
"about 32"; 212 bits, "about 64"). So 127 is proposed.

```diff
--- a/notes/OCTO_DOUBLE.md
+++ b/notes/OCTO_DOUBLE.md
@@ -3,7 +3,7 @@
 The large-D chain builds its Taylor series in D with the quad-double jet (`anm_run.py --jet_q 1`,
 notes/QUAD_DOUBLE.md 13.20). That series errs by about the quad-double unit times G², where G ≈ 0.7^−(D−3) is the
 mass mode's gain, and near D 170–185 this reaches the chain's tolerance. This note adds the same series one level up,
-in octo-double (eight `f64` parts, about 126 digits), on multidouble 4.0.0's octo-double jet (`OJet<N>`), dual (`Ou`)
+in octo-double (eight `f64` parts, about 127 digits), on multidouble 4.0.0's octo-double jet (`OJet<N>`), dual (`Ou`)
 and LU (`ODLu`). It is not used by the chain until the quad-double series gives out.
 
 ## 1. What was added
--- a/paper/ai/main.tex
+++ b/paper/ai/main.tex
@@ -55,7 +55,7 @@
 periodic boundary-value problem solved by Newton's method from a prediction made at the last point. At large $D$ the
 mass mode amplifies errors at the horizon by a gain $G \approx 0.7^{-(D-3)}$, so the arithmetic must carry more digits as
 $D$ grows, from double-double~\cite{Dekker1971} (about 32 digits) through quad-double~\cite{HidaLiBailey2001} (about 64)
-to octo-double (about 126). At low $D$ one point costs 4 to 10 hours.
+to octo-double (about 127). At low $D$ one point costs 4 to 10 hours.
 
 The project began in a chat on claude.ai (Sec.~\ref{sec:origin}). Over the following three weeks I directed Claude
 sessions running in Claude Code~\cite{ClaudeCode} on five machines, and Claude wrote the code to my direction. The lowest
--- a/paper/ai/SOURCES.md
+++ b/paper/ai/SOURCES.md
@@ -40,7 +40,7 @@
 | item | source |
 |---|---|
 | lowest converged point D = 3.04375 | README.md line 361; `hp/data/delta_all.csv` first row |
-| G ≈ 0.7^−(D−3); about 32, 64 and 126 digits | `notes/OCTO_DOUBLE.md` introduction; `paper/physics/main.tex` line 145 |
+| G ≈ 0.7^−(D−3); about 32, 64 and 127 digits | `notes/OCTO_DOUBLE.md` introduction; `paper/physics/main.tex` line 145 |
 | 4 to 10 hours per low-D point | `notes/LOW_D.md` 15, "Cost" |
 | five machines; three weeks | comms headers; first commit 2026-10-19 03:53, last 2026-11-06 18:26 |
 | 31 stuck points | Appendix A, sourced below |
```

### 24. A misrounded Δ_min in the review notes (inconsistent, minor)

| location | says |
|---|---|
| notes/REVIEW_GHMG25.md:166 | Δ_min = 0.4598093 at k_c = 39 (parabola through η = 6, 7, 8.5) |
| matter/sigma_ms/data/delta_eta.csv | Δ at η = 7, k_c 39: 0.4598093636; the multishoot minimum 0.45980936174 (README:1942) |

Both round to 0.4598094 at seven decimals. The parabola's own minimum (raw I633-win) was not re-read.

```diff
--- a/notes/REVIEW_GHMG25.md
+++ b/notes/REVIEW_GHMG25.md
@@ -163,7 +163,7 @@
   two k_c. #631 narrows it to between η = 5 and 10: at k_c = 23 and 39, Δ(7) is the smallest of η = 5, 7, 10, 14, 20
   and ∞, every neighbouring difference resolved (raw I631-win 2d659f74); #633 narrows it to (6, 8.5), with Δ(7) still
   the smallest; a parabola through η = 6, 7 and 8.5 puts the minimum at η = 6.94 to 7.20 over the variable (η, ln η,
-  1/η) and k_c, Δ_min = 0.4598093 at k_c = 39 (raw I633-win 3046bc7d). Neither cited source says monotonic (both read
+  1/η) and k_c, Δ_min = 0.4598094 at k_c = 39 (raw I633-win 3046bc7d). Neither cited source says monotonic (both read
   in their arXiv sources on 2026-11-13). Husa et al. 2000 (gr-qc/0002067, Sec. IV): Δ/2 "varies strongly with the
   coupling constant", "At large η, Δ/2 asymptotes to 0.2300 ± 0.0003", and as η decreases towards 0.18 "Δ/2 rises
   sharply"; its Table I gives Δ/2 = 0.2295, 0.2304, 0.2293 and 0.2302 at η = 2, 5, 10 and 100, rounded to four figures
```

### 25. `chain_reach` cannot fail when the chain moves on (a check that cannot fail)

| location | says |
|---|---|
| check_numbers.py:1996–2003 (`chain_reach`) | takes the chain_fit row **nearest 245.6366** and checks that it rounds to 245.6 |

If the chain reached D 250, the check would still find the row at 245.64 and pass, while README:17 and 76 would be
stale. The fix is the new check `chain_frontier` (the highest chain_fit D), in the commit with the new checks (below).
`chain_reach` is left as it is.

## Counts by kind

| kind | findings |
|---|---|
| superseded | 1, 3, 4, 5, 6, 7, 9, 10, 11, 16, 18, 20 |
| inconsistent | 12, 13, 14, 17, 22, 23, 24 |
| inconsistent status word | 8 |
| unsupported (bar or value without source) | 2, 15, 19 |
| broken reference | 21 |
| a check that cannot fail | 25 |
| **total** | **25** |

## Recommended tasks

1. **Audit, part 2.** What: the parts this audit did not read (the math paper's proofs, the proof notes' certificate
   details, QUAD_DOUBLE.md §1–13.24, PLANTED_SELECTION.md, the matter notes), with #653's rules. Why: the coverage list
   below. Acceptance: as #653's (two locations or a source per finding, diffs that apply, `--selftest` passes). Depends
   on: #653.
2. **Figures at the frontier.** What: regenerate hp/wave_family.svg, hp/big_picture.svg, hp/echo_harmonics.svg,
   hp/fit_steps.svg and hp/harmonics_held.svg at the chain's current top, and add a check that each caption's D is
   delta_all.csv's highest chain_fit D. Why: finding 20. Acceptance: the check fails on the audited captions (232.6)
   and passes after. Depends on: #653.
3. **ℓ = 1 past D = 9.** What: say whether `march_l1kc23hi.jsonl` and the D = 10 solution are converged (k_c 23 → 31,
   the ψ-constraint, abs(dΔ/dF)), and add them to ELL.md's table or mark them unreliable. Why: finding 19. Acceptance:
   a k_c 31 re-solve at D = 10 agrees with k_c 23 within a bar stated before the run, or the note records that it does
   not. Depends on: nothing.
4. **check_links.py and GitHub's anchors.** What: drop subscript digits from anchors as GitHub does, and skip links in
   quoted proposal text. Why: 19 of its 20 flags here are its own artifacts. Acceptance: a planted heading
   "## 3. Lemma 6b₀" resolves from the anchor GitHub renders for it (checked on a rendered page), and a planted wrong
   anchor still fails. Depends on: nothing.
5. **Unnamed bars.** What: find the kind or the source of README:474's ± 0.00008 and README:639's ± 7×10⁻¹¹, and add a
   lint that lists each ± in the README whose sentence names no kind. Why: finding 15. Acceptance: on the audited
   commit the lint lists the five cases of finding 15, and after the fixes none. Depends on: #653.
6. **The papers at the repository's present state.** What: carry the physics and math papers to the present results
   (the full system to 67.77, the chain to 245.6 with its bias test, Theorem A(M = 5), the second B(1) review, the RT
   check), after the diffs of findings 4–13 are applied. Why: both papers stop at their October scope, and their
   "Open" lists name finished work. Acceptance: every number the papers share with the README equals the README's (a
   script that lists the differences fails on the audited commit), and `check_numbers.py --selftest` passes. Depends
   on: #653.

## New checks (the commit "check_numbers: check the chain's frontier and four quoted tables")

`check_numbers.py --selftest`: 1420 of 1420 checks pass, and 1420 of 1420 reject a value shifted by one unit (1362
before). `notes/HELD_HARMONICS.md` is added to DOCS.

| check | quoted numbers | source |
|---|---|---|
| `chain_frontier` | README "to D = 245.6", as the highest chain_fit D | hp/data/delta_all.csv |
| `profile_table` | the 25 entries of the README's and the physics paper's profile table, D = 20–60 | hp/data/largeD_F_all.csv |
| `ell_table` | Δ and γ of the ℓ = 1 table, D = 4–9, README and physics paper | solver/matter/data/l1_kc23r.jsonl, l1_kc23_gamma.jsonl, l1_D4_kc39.jsonl |
| `d4_ladder` | Δ(4) = 3.4454524022278213499715 (exact sum of the two parts), the four M changes, Values table Δ(4), γ(4) | hp/data/delta_dd_lowD.csv |
| `gamma_61` | γ(61) = 0.48935 in GAMMA.md and the physics paper | hp/data/gamma_dd.jsonl, local/gamma_dd.jsonl |
| `saved_states` | "53 states … from 8.25 to 60" | hp/data/states/*.npz |
| `held_release` | HELD_HARMONICS.md 4.4's `fits` spread column at D 74–79 and its range at 80–200 | hp/data/delta_all.csv |

## Checked and found consistent

- **Δ∞** 1.643893746 ± 1.5×10⁻⁸ and ω∞ 3.82213590 ± 4×10⁻⁸: every occurrence in the README, the physics paper,
  CLOSED_FORMS.md and the Values tables. Older quotes, 1.64389375 ± 1.5×10⁻⁸ in NUMERICS_REVIEW.md and LOADER_FIX.md,
  are dated records. "840 times its uncertainty" (1.255×10⁻⁵ / 1.5×10⁻⁸) and "2.4×10⁻⁹ relative" (2π/1.64389375 against
  2π/Δ∞) hold by arithmetic.
- **The truncation sequence** 0.777, 1.0156, 0.999228, 1.000022 against README:129–131's 29%, 1.5% and 0.077% in Δ and
  the certificates' boxes. The selection range 0.999975–1.00005 gives 1.64381–1.64393, and the K = 21 and 22 ranges
  agree between the README and PLANTED_SELECTION.md.
- **The A(M = 5) enclosure** in README:99, 212 and PROOF.md against A_M2.md and the centre of
  `solver/proof/data/A_M2_box_M5_1e-10.json`, 0.99922840320656778. The B(1) thresholds e^{89.75}, e^{67.84} and
  e^{73.89} agree in the README, PROOF.md, the math paper and C_selection.md §13.
- **γ**: γ(3.5) in the README, γ(61) in GAMMA.md and the physics paper, the λ∞ − 2 fit table (GAMMA.md, README, physics
  paper), and λ₁ (README, FIRST_ORDER.md, GAMMA.md). λ_g at D = 30, 45 and 61 (MODE_COUNT.md verdict) matches
  hp/data/lam_gauge.csv. μ(4) = −0.36512 agrees in the README, GHOST_ADMIXTURE.md, COMPLEX_CRITICAL.md and
  REVIEW_GHMG25.md (each layout's digits), and δ(5) = (3 − μ(5))γ(5) holds.
- **The σ-model's Δ(η) minimum** η = 6.98019, Δ = 0.45980936174: README, SIGMA_DSS.md §11 and REVIEW_GHMG25.md D9
  against fit_min.json and delta_eta.csv. The agreement with k_c 39 (1.2–2.1×10⁻⁹) and the M 48 → 64 change (about
  10⁻¹¹) match the CSV's columns.
- **The chain**: README:17, 76 and 520 against the CSV. ALGORITHM.md's "about 3×10⁻¹¹ near D 200–215" matches the rows
  (2.8–3.4×10⁻¹¹). HELD_HARMONICS.md 4.4's table matches the CSV (now checked). AI paper:62 and its Table tab:numbers
  are dated ("at the time of writing", "at commit 07eff89").
- **Low D**: the lowest point 3.04375 (README:418, 520, AI paper). The D = 4 ladder (now checked). The low-D maximum
  3.755725, 3.46677169 in the README and the physics paper.
- **The profile and ℓ = 1 tables** in the README, the physics paper and ELL.md (now checked).
- **Counts**: README "14 checks" of reproduce.sh (14 `record` calls without `--full`, the Proposition 4 one with it), 53
  saved states, 1199/857/342 points (check `data_points`), 67 γ points (check `gamma_points`).
- **Status words**: the README's two status tables against PROOF.md §5's table and C_selection.md §13. They agree, apart
  from finding 8.
- **Cross-file section references** ("FILE §N", "FILE section N.N") in README.md, GAMMA.md, PROOF.md, ALGORITHM.md,
  SOFTWARE.md, AGENTS.md, proof/*.md, matter/*.md and notes/*.md: every one names a heading that exists (a script that
  fires on a planted §99.9).
- **Markdown links**: `check_links.py` over README.md, GAMMA.md, PROOF.md, ALGORITHM.md, SOFTWARE.md, SETUP.md,
  AGENTS.md, proof/, notes/ and matter/: 986 links, 20 flagged. Thirteen are links inside text proposed for the README
  (proof/B_tail_M7.md, proof/A_M.md, proof/A_lemmas_M.md, proof/B5_inner.md, notes/RT_FORMULATION.md), written with
  root-relative paths that are right once pasted into the README. Six are table-of-contents anchors with subscript
  digits (6b₀, 6b₁, n₀), which check_links.py keeps (Python's `\w`) and GitHub very likely drops, so they are likely the
  checker's error, not verified on GitHub. One is a code fragment in A_M2_review.md:81 that the link regex reads as a
  link.

## Coverage: files read

- **In full**: README.md, paper/physics/main.tex, GAMMA.md, ALGORITHM.md, AGENTS.md, notes/BIAS.md,
  notes/PROOF_QUEUE.md, proof/P2b_progress.md, check_links.py.
- **In part**: paper/math/main.tex (abstract, introduction, status table, §B1status, §A3, "What is not claimed",
  Outlook), paper/ai/main.tex (abstract, introduction, tables), paper/ai/SOURCES.md (one row), PROOF.md (Summary, §4,
  §5), proof/C_selection.md (§13), proof/P2b_review.md (§0), proof/A_M2.md (header), notes/QUAD_DOUBLE.md (§13.25–13.29,
  headings), notes/HELD_HARMONICS.md (Recommendation, §1, §4.4–§6), notes/LARGE_D_WALL.md (opening, the octo-double
  wall), notes/MODE_COUNT.md (verdict, §5's closure), notes/LOW_D.md (Summary, §4, §6), notes/REVIEW_GHMG25.md (D8–D10),
  notes/RT_FORMULATION.md (verdict, §10.9, §11.7), notes/OCTO_DOUBLE.md (introduction), notes/GHOST_ADMIXTURE.md
  (opening), notes/CLOSED_FORMS.md (the tests and table), notes/LARGE_D_MIXED.md (§7), notes/FIRST_ORDER.md (tables),
  notes/INNER_STOKES.md and proof/B_tail_M7.md (the M = 7 lines), matter/SIGMA.md (η = 0.14), matter/ELL.md (the ℓ = 1
  table), check_numbers.py (structure and the checks named here), reproduce.sh, reproduce.py (the `record` calls),
  hp/export_delta_all.py (docstring), hp/gamma_compare.py (the error estimate), solver/notes.md (lines 380–405).
- **Searched, not read**: every other file of notes/, proof/ and matter/, and SOFTWARE.md and SETUP.md, for the section
  references, the Δ∞ variants, the γ(3.5) and σ-model values, the reach and frontier phrases and the point counts.
- **Data files read**: hp/data/delta_all.csv, delta_dd.csv, delta_dd_lowD.csv, delta_fits.csv, projector_shift.csv,
  release_steps.csv, chain_fit_vs_series.txt, gamma_dd.jsonl, local/gamma_dd.jsonl, lam_gauge.csv, echo_shapes.csv,
  largeD_F_all.csv, lowD_highM/lift_D4_points.jsonl, the file list of hp/data/states/;
  matter/sigma_ms/data/delta_eta.csv and fit_min.json; solver/proof/data/A_M2_box_M5_1e-10.json;
  solver/matter/data/ell_gamma.jsonl, l1_kc23r.jsonl, l1_kc23_gamma.jsonl and l1_D4_kc39.jsonl.

Not covered: most of the math paper's proofs, the proof notes' certificate details, and the long logs of
QUAD_DOUBLE.md §1–13.24, PLANTED_SELECTION.md, LIT_REVIEW.md and the matter notes. Recommended task 1 continues there.
