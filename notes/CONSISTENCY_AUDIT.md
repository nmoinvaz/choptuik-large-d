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
  link. #661 checked these on GitHub's rendered pages (the last section).

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

## check_links.py after #661

`check_links.py` now builds heading anchors with github-slugger 2.0.0's rule, which its docstring states with the
source. The rule reproduces the anchors GitHub rendered at 52433d0d for all 1724 headings of the 102 tracked .md files,
in order, and for a planted file of 28 edge cases that `--selftest` keeps. The old rule missed 66 of the 1724.

Over every tracked .md file it reads 998 links and flags 5. The old checker flagged 21 there, the 20 above and
notes/CORRECTOR_BASIN.md:207, which came later. The table-of-contents anchors with 6b₀, 6b₁ and n₀ now resolve. There
are five of them, not six. The links in quoted proposal text resolve from the repository root, which the docstring's
rule allows for such text. There are fourteen of them, not thirteen. A_M2_review.md:81 is no longer read as a link,
since GitHub does not render it as one.

The five that remain are broken on GitHub, as its rendered pages show. They are not fixed here.

| location | link | on GitHub |
|---|---|---|
| proof/A_certify.md:69 | `#103-results-at-ξ₁--03` | the heading's anchor is `#103-results-at-ξ--03` |
| proof/A_certify.md:76 | `#12-lemma-6b₁` | `#12-lemma-6b` |
| notes/SECOND_ORDER.md:18 | `#4-the-selection-at-second-order-δ₂-and-λ₂` | `#4-the-selection-at-second-order-δ-and-λ` |
| notes/SECOND_ORDER.md:19 | `#5-against-the-fits-and-δ-with-δ₁-and-δ₂-held-fixed` | `#5-against-the-fits-and-δ-with-δ-and-δ-held-fixed` |
| notes/CORRECTOR_BASIN.md:207 | `[ω](D)`, math in the formula for θ₀ | a link to a file D |

The first four keep the subscript digits, as the old rule did, so the old checker passed them. The count above of 19
artifacts in 20 flags holds, but the old checker also passed these four broken links.

## Follow-up #662

Finding 15's two unsourced bars are found, and the lint `readme_bars.py` lists each ± of the README whose sentence names
no kind. On the audited commit it lists the five cases of finding 15 and a sixth that the audit missed. The README lines
below are those of main at eaa4f460, where finding 15's diff is already applied (README:462, 1612, 1940).

### The bars and their sources

| README | bar | kind | source |
|---|---|---|---|
| 475 | 1.64395 ± 0.00008 | a spread over fit choices, the centre and half-width of the range 1.643882–1.644027 of the free limit c₀ over the 19 of 24 fits with χ²/dof ≤ 1 (1/D, 1/(D − 1), 1/(D − 2), degree 8–11, from D ≥ 10 or 12 to 30.2) | `solver/final_fit.py` (`step_converged`, `weighted_series`). README 15ceca38 stated the range and its fit choices. No committed script prints it |
| 643 | 3.221701945 ± 7×10⁻¹¹ | an estimator, the size of a geometric extrapolation step in the mode count, \|ext − kc 47\| = 6.89×10⁻¹¹ | `solver/data_all.py` `low_d()` at D = 5 (3.221701944781), Richardson in the RK step at kc 31, 39 and 47, from `solver/data/scan_ne6_lowD_kc*.jsonl` |
| 753 | ±3×10⁻¹¹ | a floor, `low_d()`'s σ = max(3×10⁻¹¹, \|kc 39 − kc 31\|/13), where the mode-count term is only 9×10⁻¹⁴ (D = 7) and 6×10⁻¹³ (D = 7.75) | `solver/data_all.py`. `hp/proto/val_lowD.py` prints it as the references' sigma |

Both are reproduced on main from `solver/data`:

```sh
PYTHONPATH=.. python3 -c "from final_fit import step_converged as sc, weighted_series as ws; D, y, s, _ = sc(); a = [c[0] for lo in (10, 12) for m in [(D >= lo) & (D <= 30.2)] for f in (lambda d: 1/d, lambda d: 1/(d-1), lambda d: 1/(d-2)) for k in (8, 9, 10, 11) for c, chi in [ws(f(D[m]), y[m], 1/s[m], k)] if chi <= 1]; print(len(a), round(min(a), 6), round(max(a), 6))"
PYTHONPATH=.. python3 -c "from data_all import low_d; lo = low_d(); print([(d, lo[d][0], lo[d][1]) for d in (5.0, 7.0, 7.75)])"
```

The first prints `19 1.643882 1.644027`, the same on solver/ as of 15ceca38. The second prints D = 5 as
(3.221701944781393, 6.89×10⁻¹¹) and D = 7, 7.75 with σ = 3×10⁻¹¹.

### The lint

`readme_bars.py` states its rule in its docstring. A sentence names a kind when it holds one of the words spread,
scatter, standard error, standard deviation, range, estimat, bound, enclos, interval, box, certif, rigorous,
resolution, fit choices or bar. A bar also passes when a later sentence of its paragraph starts with "The bar" (or a
like phrase), when its sentence cites a publication, or when the same value ± bar passes elsewhere in the file.
Extrapolated, converges, limit and fit alone do not count. A ± before a letter, or a lone ± before an integer, is a
sign.

| check | command | result | exit |
|---|---|---|---|
| audited commit | `python3 readme_bars.py --rev b73b80e8` | lists 462, 474, 639, 749, 1607, 1934, of 35 bars | 1 |
| planted bare ± | `python3 readme_bars.py --selftest` | the copy of the Main results row for Δ∞, its kind words dropped and its bar set to 7.654321 ± 0.000123, is listed | 1 |
| proposed text | `python3 readme_bars.py FILE`, FILE a copy of main's README.md with the diff below applied | lists none, of 35 bars | 0 |
| main, eaa4f460 | `python3 readme_bars.py` | lists 475, 643, 753 | 1 |

All 48 ± and \pm of main are read. 35 are bars and 13 are signs (±1/2, ±1, ±2; ±iπ three times; \pm\beta; \pm\pi;
±2, ±4, ±6; ±ω), and the "Each ±" of README:87.

### Proposed README text

It applies to main at eaa4f460 (`git apply --check`). With it applied, `check_numbers.py` quotes no changed line.

```diff
--- a/README.md
+++ b/README.md
@@ -472,9 +472,10 @@
 any finite order of the 1/n expansion ([Proposition 5](#proposition-5-the-unstable-mode-at-infinite-d-is-exact)).
 
 Δ∞ comes from fits of all the double-double points. The fits, their residuals and the error bar are in
-[Is the period analytic in 1/D?](#is-the-period-analytic-in-1d). The f64 points gave 1.64395 ± 0.00008. This formula's
-own limit is a₀ = 1.643898, 4×10⁻⁶ high, which its fitting range could not see. The coefficients of one fit are
-correlated, so they are not physical constants. Only the function is.
+[Is the period analytic in 1/D?](#is-the-period-analytic-in-1d). The f64 points gave 1.64395 ± 0.00008, the centre and
+half-width of the range of the limit over fit choices (variable 1/D, 1/(D − 1) or 1/(D − 2), degree 8–11, from D ≥ 10
+or 12 to 30.2, χ²/dof ≤ 1). This formula's own limit is a₀ = 1.643898, 4×10⁻⁶ high, which its fitting range could not
+see. The coefficients of one fit are correlated, so they are not physical constants. Only the function is.
 
 **The limit rests on an assumption, which the double-double points now support.** Δ∞ = 1.643893746 holds if Δ(D) is
 analytic in 1/D at D = ∞. That means no half-integer powers $`n^{-1/2}`$, no $`\log n`$, and no new scale far beyond the
@@ -640,8 +641,10 @@
 - **D = 5: a resolution effect in the published value.** Ecker et al. 2026 give Δ = 3.22176, γ = 0.41322, which are
   5.8×10⁻⁵ and 9.3×10⁻⁵ above the values here.
   - Our Δ(5) = 3.2217019446 comes from two independent codes, which agree to 10⁻¹¹ (`hp/compare/dd_D5.py`, log in
-    `hp/compare/dd_D5.log`). They are the f64 solver at 31, 39 and 47 Fourier modes (3.221701945 ± 7×10⁻¹¹ extrapolated)
-    and the double-double solver at M = 48 and 64 collocation points (3.22170194461806, 3.22170194462672).
+    `hp/compare/dd_D5.log`). They are the f64 solver at 31, 39 and 47 Fourier modes, extrapolated geometrically in the
+    mode count (3.221701945 ± 7×10⁻¹¹, the size of that extrapolation step, an estimate and not a bound,
+    `solver/data_all.py`), and the double-double solver at M = 48 and 64 collocation points (3.22170194461806,
+    3.22170194462672).
   - Their published data ([github.com/EckerChristian/CriticalSpacetimeCrystal](https://github.com/EckerChristian/CriticalSpacetimeCrystal), compared by `hp/compare/cmp_ecker.py`) show their D = 5 value still moving by −1.66×10⁻⁴ between 256 and 512 modes, toward ours. That is in the range above D ≈ 4.8 where their paper notes that the light-cone expansion order has to be increased.
   - Their offset from our values grows smoothly with D, from 2.7×10⁻⁷ at D = 4 to 5×10⁻⁴ at D = 5.5. The first is within
     their stated ±10⁻⁶, and ours matches Martín-García & Gundlach to 7×10⁻⁹ there.
@@ -750,7 +753,7 @@
 | horizon series against 40-digit mpmath | 2×10⁻³² |
 | Jacobian against double-double differences | agrees to the difference truncation |
 | Newton at D = 8 | quadratic, $`\lvert F\rvert = 5\times10^{-31}`$ in 4 steps |
-| Δ(7), Δ(7.75) against the 39-mode references (±3×10⁻¹¹) | +3.1×10⁻¹², −2.8×10⁻¹² |
+| Δ(7), Δ(7.75) against the 39-mode references (±3×10⁻¹¹, the floor of their error estimate in `solver/data_all.py`) | +3.1×10⁻¹², −2.8×10⁻¹² |
 | 32 / 48 / 64 collocation points in τ, D = 7–17 | Δ agrees to 10⁻¹⁶ |
 | first node at x = 0.02 / 0.05 / 0.1, D = 16.75 | Δ identical to 10⁻¹⁶ |
 | 32 / 48 collocation points in τ, D = 40 | Δ agrees to 9×10⁻²² |
```

The other three bars of finding 15 are its diff, in main since before eaa4f460. The lint passes them by "standard error"
(README:462), "scatter" (1612) and "spread" (1940).

### Discrepancies

- **D = 5, "two independent codes, which agree to 10⁻¹¹" (README:642).** The extrapolated f64 value, 3.221701944781,
  is 1.55×10⁻¹⁰ above the double-double M = 64 value 3.22170194462672 (1.63×10⁻¹⁰ above M = 48), 2.2 times its own bar
  of 6.9×10⁻¹¹. The f64 seed of `hp/compare/dd_D5.log` (kc 47, RK x2, 3.221701944776) is 1.49×10⁻¹⁰ above. Only the
  double-double M = 48 and 64 agree to 10⁻¹¹ (8.7×10⁻¹²). Quoting the f64 value as 3.221701945 rounds the gap away. Not
  changed in the text above. One resolution is "which agree to 2×10⁻¹⁰".
- **No script prints the range behind 1.64395 ± 0.00008.** `final_fit.py`'s own output fits every point to D = 38.2,
  including the six x_K 0.7 points above 30.2. Over its rows with χ²/dof ≤ 1 the range is 1.643896–1.644223 on main
  (1.643896–1.644168 at 15ceca38), not the quoted one, which needs the cut D ≤ 30.2.
- **Finding 15 counted five bars without a kind. There are six.** The sixth is the 39-mode references' ±3×10⁻¹¹
  (README:749 at b73b80e8).
- `solver/data_all.py`'s docstring says the kc 47 re-solves exist at D ≤ 4.25, but `low_d()` also extrapolates with them
  at D = 5.

### Recommended tasks

1. **The f64 Δ∞ range in a script.** What: an option of `solver/final_fit.py` that prints the range of c₀ over the fits
   with χ²/dof ≤ 1 on D ≤ 30.2, and a `check_numbers.py` check of the README's 1.64395 ± 0.00008 against it. Why: the
   second discrepancy above. Acceptance: the check passes on main, and `--selftest` rejects a one-unit shift. Depends
   on: #662.
2. **The D = 5 agreement.** What: decide the text of README:642 against the 1.55×10⁻¹⁰ gap, and add a check of the f64
   and double-double values of D = 5 (`low_d()[5.0]` against `hp/compare/dd_D5.log`). Why: the first discrepancy.
   Acceptance: the check fails on the text "agree to 10⁻¹¹" and passes on the text chosen. Depends on: #662.
3. **The lint in the routine.** What: name `python3 readme_bars.py` beside `check_numbers.py --selftest` in AGENTS.md's
   "Quoted numbers" rule, once the text above is applied. Why: a new ± without a kind would otherwise pass unseen.
   Acceptance: `readme_bars.py` exits 0 on main after the edit, and its `--selftest` exits 1. Depends on: #662.

## Follow-up #755

Two checks of `check_numbers.py` tie the bars of follow-up #662 to scripts. Both pass on main 22254bb7, and `--selftest`
rejects a one-unit shift of each of their 15 items (1573 of 1573 in all).

| check | README | items | source |
|---|---|---|---|
| `f64_dinf_range` | 475 | 24 fit choices, 19 kept, ends 1.643882 and 1.644027, centre 1.64395, half-width rounded up 0.00008, and 1.64395 ± 0.00008 contains the range | `solver/final_fit.py --range` (`limit_range`) |
| `d5_two_codes` | 643–648 | f64 3.221701945 and its estimate 7×10⁻¹¹, double-double M = 48 and 64 and their 10⁻¹¹, the codes' gap 2×10⁻¹⁰, the f64 value 1.5×10⁻¹⁰ above M = 64, twice its estimate | `solver/data_all.py` `low_d()[5.0]`, `hp/compare/dd_D5.log` |

The D = 5 check fails on the old text "which agree to 10⁻¹¹". From `solver/data`:

```sh
python3 ../final_fit.py --range               # 19 of 24 fits ... c0 1.643882 .. 1.644027, centre 1.643955, half-width 0.000073
python3 ../final_fit.py --plant               # every plant passes, exit 0
python3 ../final_fit.py --plant --mutate      # every plant fails, exit 1
python3 ../final_fit.py --plant --seeds 40    # each plant: 14 of 40 seeds pass, c0 inside 32
```

### The plant

`final_fit.py --plant` runs synthetic Δ(D) with a known limit through `limit_range`, on the real D grid with the real
weights. The base is the stored formula (P8/Q2 in 1/(D − 2), limit a₀ = 1.6438975180). P2 adds a pole at D = −8.3 and
P3 a log branch point at D = −20, as T2 and T5 of `hp/planted_dinf.py`. The noise is Gaussian at 0.8 times the weights'
σ, where the real fits' χ²/dof is 0.63. The script's docstring fixed the plants and the pass criteria before the first
run.

| plant | fits kept | range | centre − a₀ | verdict |
|---|---|---|---|---|
| P1, the formula | 19 | 1.643896–1.644031 | +6.6×10⁻⁵ | pass |
| P2, pole | 18 | 1.643821–1.643944 | −1.5×10⁻⁵ | pass |
| P3, log branch | 19 | 1.643714–1.643960 | −6.1×10⁻⁵ | pass |
| control `--mutate`, +0.00024 on every point | 18–19 | | +1.8 to +3.1×10⁻⁴ | all fail |

**The plants pass at their fixed seeds by luck of the noise.** Over seeds 1–40 each plant passes 14 times, a₀ lies
inside the range 32 times, and one seed keeps no fit. The centre moves by −2.85×10⁻⁴ to +3.43×10⁻⁴ about a₀ (rms
1.5×10⁻⁴), and the median half-width is 1.4×10⁻⁴. P2 and P3 give P1's numbers to three digits, since the fits absorb
their structure and the noise sets the spread. So ± 0.00008 is a spread over fit choices for one draw of the data, not a
bound. White noise at the weights' level would move the centre by about twice it. The real range is half as wide as the
plants' median, so the real errors are smaller or more correlated than this noise model. The double-double Δ∞ =
1.643893746 lies 6.1×10⁻⁵ below the f64 centre, inside the range.

### Discrepancies

- **README:475 calls 0.00008 the half-width of the range.** The half-width is 7.28×10⁻⁵, which rounds to 0.00007.
  0.00008 is that half-width rounded up, and only the rounded-up bar about 1.64395 contains the range, since 1.64395 +
  0.00007 = 1.64402 lies below 1.644027. The check tests the rounded-up half-width and the containment. One resolution
  is "the centre and half-width, rounded up, of the range".
- **README:647 says the f64 value lies 1.5×10⁻¹⁰ above the M = 48 and 64 values.** It is 1.55×10⁻¹⁰ above M = 64 and
  1.63×10⁻¹⁰ above M = 48. The check uses M = 64. One resolution is "lies 1.5×10⁻¹⁰ above the M = 64 value".

Either change moves a text that `check_numbers.py` quotes (`f64_dinf_range`, `d5_two_codes`), so the check's text
follows the README.

## Part 2 (#658)

Audited commit: **90a81f69** (main, 2026-11-14). Agent mac658, on the Mac, under #653's rules (the top of this note).
Scope: the math paper's proofs, the proof notes' certificate details (`proof/`, PROOF.md beyond its Summary, §4 and §5),
`notes/QUAD_DOUBLE.md` §1–13.24, `notes/PLANTED_SELECTION.md` and the matter notes (`matter/`). Reading and comparison,
with small scripts that read the data files. Each finding gives its locations, the canonical source, a verdict and a
diff. Every diff applies to 90a81f69 on its own (`git apply --check`), and all of them together in the order below,
which is also the order of severity. With all of them applied, every text `check_numbers.py` quotes is still present,
and `check_numbers.py --selftest` passes on both of its lines (1629 of 1629 pass, and 1629 of 1629 reject a one-unit
shift), on a scratch copy of 90a81f69 with this branch's `check_numbers.py`. Nothing here was applied to the files
audited. Lines a diff lengthens past 120 characters are for `wrap_prose.py --write`.

### Findings, most severe first

#### P2-1. PROOF.md §1 still has the selection range at fifteen orders, with one surviving ring (superseded)

| location | says |
|---|---|
| PROOF.md:82–88 | the twelve-order expansion "brackets" ω*/ω∞ in 0.99915–1.00011; 0.999975–1.00005 "over fourteen and fifteen orders"; "the fourteenth and fifteenth coefficients rejected every ring …, and the one surviving ring … puts the zero inside the range" |
| README.md:132–138, 1620–1621, 1670–1678 | the pair is "not a bracket"; the range is over nine orders, the fourteenth to twenty-second; the rings just beyond x₀ survived one coefficient each and were rejected through σ₂₂; M = 7 gives 1.000022 |

Canonical: the README and notes/PLANTED_SELECTION.md §7–14. PROOF.md:399 quotes the same range without the order count
and stays as it is. The diff also carries finding P2-13's change to Corollary C (PROOF.md:79), whose context overlaps.

```diff
--- a/PROOF.md
+++ b/PROOF.md
@@ -76,16 +76,16 @@
 profile and frequency satisfy F_n → F(A*, ω*) and ω_n = ω* + O(1/n). Conversely, any family of such solutions with
 F_n → F_∞, ω_n → ω_∞ in the sense of hypothesis (H1) has S(F_∞, ω_∞) ≡ 0.
 
-**Corollary C (identification).** If the critical solutions computed for D = 8–61 form the branch of Theorem B, then
-ω∞ = ω*, and Δ∞ = 2π/ω* is determined by the inner problem alone.
-
-The numerics so far are consistent with all three: the twelve-order amplitude expansion brackets ω*/ω∞ in
-0.99915–1.00011 with Padé estimates 0.9999, and the five-harmonic truncation gives 0.999228 (README, "Where that leaves
-it"). A planted-truth test of those estimators (notes/PLANTED_SELECTION.md) puts the zero at 0.999975–1.00005 over
-fourteen and fifteen orders (an estimator range, which contains the finite-D value) if S₁ has no zeros closer to the
-origin than x₀; the fourteenth and fifteenth coefficients rejected every ring of such zeros constructed so far, and the
-one surviving ring, with zeros just beyond x₀, puts the zero inside the range. They are evidence, not part of the
-proof.
+**Corollary C (identification).** If the critical solutions computed for D = 8.25–67.77 form the branch of Theorem B,
+then ω∞ = ω*, and Δ∞ = 2π/ω* is determined by the inner problem alone.
+
+The numerics so far are consistent with all three: at twelve orders the amplitude expansion gives ω*/ω∞ =
+0.99915–1.00011 (two estimators, not a bracket) with Padé estimates 0.9999, and the five- and seven-harmonic
+truncations give 0.999228 and 1.000022 (README, "Where that leaves it"). A planted-truth test of those estimators
+(notes/PLANTED_SELECTION.md) puts the zero at 0.999975–1.00005 over the fourteenth to twenty-second orders (an estimator
+range, which contains the finite-D value) if S₁ has no zeros closer to the origin than x₀. Those coefficients rejected
+every ring of such zeros constructed so far, and the rings with zeros just beyond x₀ each survived one coefficient and
+put the zero inside the range. They are evidence, not part of the proof.
 
 ## 2. What is already proved here
 
```

#### P2-2. QUAD_DOUBLE.md puts the frontier gap at a tenth of its own table's (inconsistent; an opening line superseded)

| location | says |
|---|---|
| notes/QUAD_DOUBLE.md:880 (§12.4) | at D 73.27 "the double-double one lies 1e-15 from them" |
| notes/QUAD_DOUBLE.md:840–841 (its table) | T75's double-double Δ_full1 1.7807570807362567, the quad-double ones 1.78075708073624606 − 1.5×10⁻¹⁷ and 1.78075708073624539 + 1.3×10⁻¹⁷ |
| hp/data/delta_all.csv, D 73.265722 | 1.7807570807362567 − 9.18×10⁻¹⁷ (kind pinned, no err) |
| notes/QUAD_DOUBLE.md:11 (opening) | "Nothing in the solver uses them yet." |
| notes/QUAD_DOUBLE.md §12.1; README.md:91, 529 | the quad-double solver, built 2026-10-27; the quad-double points 68.5–73 |

The double-double value lies 1.06×10⁻¹⁴ and 1.12×10⁻¹⁴ above the two quad-double ones (exact sums of the parts, mpmath),
ten times the text. The row of `delta_all.csv` at that D is the double-double value, so its Δ carries this 10⁻¹⁴, which
no err column shows (only one pin reached that D, so the rule of `hp/export_delta_all.py` gives none). Canonical: the
note's own table.

```diff
--- a/notes/QUAD_DOUBLE.md
+++ b/notes/QUAD_DOUBLE.md
@@ -877,7 +877,7 @@
 corrected pinned pairs above lie within 1e-17 (71.0) and 2e-17 (71.5) of it: that is the linearised correction's own
 error, and the pins' agreement to 1e-32 at 71.5 was a coincidence of the two pins. From 72 on no start of double-double
 quality reaches the full basin, as the table of basins said. At the frontier the two pins' quad-double values differ by
-7e-16 and the double-double one lies 1e-15 from them: the digits there are set by the pin's cost (the full step's |dz|
+7e-16 and the double-double one lies 1.1e-14 above them: the digits there are set by the pin's cost (the full step's |dz|
 is 7.9e-6 against 1.6e-9 at 72.4), not by the arithmetic. So the quad-double solver adds digits where the pin is cheap,
 up to about 72.5 in this layout, and past that the full system's junk, not precision, bounds Delta.
 
```

```diff
--- a/notes/QUAD_DOUBLE.md
+++ b/notes/QUAD_DOUBLE.md
@@ -8,7 +8,8 @@
 
 Quad-double numbers (an unevaluated sum of four f64, about 64 significant digits) are now in the `hpkernel` crate:
 a scalar type `QD` and an 8-lane SIMD type `QV` with bitwise the same results, on the AVX-512, AVX2 and portable
-backends of `v8.rs`. Nothing in the solver uses them yet. Every existing output is bit-identical (section 9). The
+backends of `v8.rs`. Sections 1–10 describe them as first built, when nothing in the solver used them yet; the
+quad-double solver built on them is in section 12.1 onward. Every existing output is bit-identical (section 9). The
 logs behind every number here are in `hp/data/quad_double/`, and the commands run in `hp/`.
 
 **Summary.**
```

#### P2-3. The math paper calls λ₁ open, which the truncations have computed (superseded)

| location | says |
|---|---|
| paper/math/main.tex:860–864 | "Two questions are open: whether λ = 2 is the only physical unstable mode at D = ∞, and that coefficient", −S_{1/n}/∂_λS |
| paper/math/main.tex:2219 (Outlook) | open: "the coefficient of 1/n in λ" |
| notes/FIRST_ORDER.md:30–34, README.md:1833–1848 | λ₁ = 7.76, 3.080, 3.023 at M = 1, 3, 5 from the O(1/n) inner problem, against the fitted 3.0324; "M = 7 and an untruncated computation are not done" |

This is #653's finding 11 for the math paper. Canonical: notes/FIRST_ORDER.md. The diff also carries finding P2-9's
change on line 864, since the two touch the same line.

```diff
--- a/paper/math/main.tex
+++ b/paper/math/main.tex
@@ -857,11 +857,13 @@
 
 At finite $D$ the outer region and the $1/n$ corrections to the inner equations depend on $r$ explicitly
 (Lemma~\ref{lem:B50}), so the symmetry is broken and the mode survives as an ordinary unstable mode with
-$\lambda = 2 + O(1/n)$, whose $1/n$ coefficient is measured in~\cite{companion}. Two questions are open: whether
-$\lambda = 2$ is the only physical unstable mode at $D = \infty$, and that coefficient, which by first-order perturbation
-theory is
+$\lambda = 2 + O(1/n)$, whose $1/n$ coefficient is measured in~\cite{companion}. Whether $\lambda = 2$ is the only
+physical unstable mode at $D = \infty$ is open. The coefficient is, by first-order perturbation theory,
 $-S_{1/n}/\partial_\lambda S$ at $\lambda = 2$, with $S$ the perturbation's fast-mode multiplier and $S_{1/n}$ the part induced
-by the $O(1/n)$ inner equations. The one-harmonic model of Section~\ref{sec:thmA} has a root $\lambda = 1.765$; this is not a
+by the $O(1/n)$ inner equations. In the $M$-harmonic truncations it is $7.76$, $3.080$ and $3.023$ for $M = 1$, $3$ and $5$,
+against $3.0324$ from the solutions (numerical evidence, conditional on Conjecture~\ref{conj} at that order;
+\repo{notes/FIRST\_ORDER.md}); without truncation it is not computed. The one-harmonic model of Section~\ref{sec:thmA}
+has a root $\lambda = 1.769$; this is not a
 counterexample, because its background is not the $D = \infty$ solution and because Galerkin truncation does not commute
 with the gauge step, which multiplies by $P(\theta)$.
 
@@ -2216,7 +2218,8 @@
 \end{enumerate}
 Together these would derive $\Dinf = 2\pi/\omega^*$ from the $D = \infty$ equations. Numerically the rule already gives $\Dinf$ to
 about $10^{-4}$; more orders, or $M = 9$, would sharpen that, and whether $\Dinf$ has a closed form is open. For the critical
-exponent, open are the uniqueness of the unstable mode at $D = \infty$ and the coefficient of $1/n$ in $\lambda$. A
+exponent, open are the uniqueness of the unstable mode at $D = \infty$ and the coefficient of $1/n$ in $\lambda$ without
+harmonic truncation. A
 centrifugal term of order $n/r^2$ (scalar multiplets with angular momentum, wave maps, Yang--Mills) becomes an explicit
 $1/s$ in the inner equations and removes the symmetry of Proposition~\ref{prop:5}. Other matter models test this, and the
 companion paper reports the tests~\cite{companion}: for the axion--dilaton, whose inner problem keeps the symmetry,
```

#### P2-4. CSS.md does not say which fits its λ∞ and γ∞ figures come from (unsupported bar)

| location | says |
|---|---|
| matter/CSS.md:349 (§7), 31 (summary); README.md:1877 | "λ∞ = 2 to about 10⁻⁵, γ∞ = 1/2 to about 3×10⁻⁶" |
| solver/matter/css_fit.py (docstring) | "The quoted limit is the median of the kept fits whose truncation is 'converged' …, with the full spread over (K, Dmin) as the error bar" |
| solver/matter/data/css_fit_k1.json | `lambda` = [2.0000955, 2.0000086, 2.000258], `gamma` = [0.4999677, 0.4999152, 0.4999969] (median, low, high) |
| matter/CSS.md:328–334 (table) | free fits: λ∞ − 2 = 8.64×10⁻⁶ (K = 7, D ≥ 20), 1.68–1.70×10⁻⁵ (K = 6–7, D ≥ 24), 8.18–8.27×10⁻⁶ (D ≥ 30) |

The quoted figures are the highest-order free fits of the note's table, which the note's argument (each added power
divides λ∞ − 2 by about 3) supports. The script that writes the data quotes the median instead, which is ten times
larger. Neither is named. The ω agreements of the same section (1.6×10⁻⁴, 1.3×10⁻⁴) hold with the unrounded medians
(1.579×10⁻⁴, 1.317×10⁻⁴; the new check `css_omega`), though the table's rounded 3.1994 and 3.9784 would give 1.5×10⁻⁴
and 1.4×10⁻⁴, as the physics paper's "before rounding" says.

```diff
--- a/matter/CSS.md
+++ b/matter/CSS.md
@@ -347,7 +347,10 @@
 The 1/n coefficient is 3.598–3.601 over the fits with 5–6 powers and D ≥ 20–24, rising slowly with the order, so
 c₁ = 3.60 ± 0.01. For the real scalar's DSS solution GAMMA.md has λ = 2 + 3.0324/n − 34.3/n², a similar series.
 **Conclusion: λ∞ = 2 to about 10⁻⁵, γ∞ = 1/2 to about 3×10⁻⁶** (an extrapolation from D ≤ 80; at D = 80,
-γ = 0.4896024).
+γ = 0.4896024). Both are read off the highest-order free fits of the table, K = 7 on D ≥ 20 and K = 6–7 on D ≥ 24
+and 30 (λ∞ − 2 = 0.8–1.7×10⁻⁵): an estimate, not a bound. `css_fit.py`'s own summary in `data/css_fit_k1.json`, the
+median of the best third of its converged fits over (K, D ≥), includes lower orders and gives λ∞ − 2 = 9.6×10⁻⁵
+(0.9×10⁻⁵ to 2.6×10⁻⁴) and γ∞ − 1/2 = −3.2×10⁻⁵ (−8.5×10⁻⁵ to −3.1×10⁻⁶).
 
 **ω.** The phase-rotation frequency extrapolates to
 
```

#### P2-5. C_selection.md's summary says the B(1) certificate code is unreviewed (superseded, inconsistent within the file)

| location | says |
|---|---|
| proof/C_selection.md:76–77 | "The certificate code has not been reviewed independently." |
| proof/C_selection.md:959–962 (§13.1) | B7, Lemma LC and Lemma J have second implementations (P2_review.md, P2b_review.md); lemma L, B5(2), Lemma P audited; B5(1), B8 beyond `--d0` and (H8) not reviewed |

Canonical: §13.1 of the same file, and proof/P2b_review.md.

```diff
--- a/proof/C_selection.md
+++ b/proof/C_selection.md
@@ -73,8 +73,9 @@
   ([B_chain.md](B_chain.md) §15; Section 13.1 lists every link with its status and threshold). The theorem is proved
   with the selection function "label of the outer jump at a fixed point of the Stokes line" rather than the
   light-cone coefficient C_{n,μ}; B6 and the factor κ_{n,μ} are not needed. The light cone is handled by Lemma LC (B8
-  continued into the light-cone disc in the exact gauge) and Lemma J (the jump vanishes with its label). The
-  certificate code has not been reviewed independently.
+  continued into the light-cone disc in the exact gauge) and Lemma J (the jump vanishes with its label). Two
+  independent reviews ([P2_review.md](P2_review.md), [P2b_review.md](P2b_review.md)) re-implement B7, Lemma LC and
+  Lemma J; the code of B5(1), of B8 beyond `--d0` and of (H8) is not yet reviewed (Section 13.1).
 
 | Step | Content | Status (details in Section 13) |
 |---|---|---|
```

#### P2-6. A_M.md and A_lemmas_M.md say their code is unreviewed, and A_lemmas_M.md has M = 5 waiting (superseded)

| location | says |
|---|---|
| proof/A_M.md:8, 17–18, 91, 333 | "not reviewed"; M = 5 "the new code not reviewed"; "the new files above (not reviewed)"; "the new files are not reviewed" |
| proof/A_M_review.md (Verdict) | the M = 3 point certificate is certified; the exact-rational re-assembly reproduces its margins |
| proof/A_M2.md:7, 191–192 | "reviewed independently in A_M2_review.md"; certify_M2*.py reviewed |
| proof/A_M2_review.md:181 | item 9: "A_M2.md and A_M.md say 'the new code not reviewed'; the coordinator decides how to change that" |
| proof/A_lemmas_M.md:12 | "Nothing here has been reviewed independently yet." |
| proof/A_lemmas_M.md:766 | "At M = 5 everything waits for its certificate except lemma S, 6b₀, the flat estimate and step (i)" |
| proof/A_lemmas_M.md:37–38 (its own summary), README.md:208–209 | 6a(3) reviewed (A_M_review.md); 6a(5) and 6b₁(5) certified at A_M2.md's p̂, the code reviewed by #39 |

A_M2.md was updated after its review, and A_M.md was not. #40 (open) is to write Theorem A(M = 5) into A_lemmas_M.md, so
the second diff may be overtaken by it. The lines the first diff lengthens past 120 characters are for `wrap_prose.py
--write`.

```diff
--- a/proof/A_M.md
+++ b/proof/A_M.md
@@ -5,7 +5,8 @@
 box". The analytic lemmas that turn the point certificate into the statement about the selection condition (the
 analogues of lemmas S, 6a and 6b of A_certify.md §11–12) are not part of this note; the M-harmonic start-value fixed
 point is being written separately (`solver/proof/fixed_point_M.py`, notes in `proof/A_lemmas_M.md`). Written by Claude
-(Anthropic) under Nathan Moinvaziri's direction; not reviewed.
+(Anthropic) under Nathan Moinvaziri's direction; its M = 3 certificate is reviewed independently in
+[A_M_review.md](A_M_review.md).
 
 **Status.** M = 1: this code path reproduces Theorem A(M = 1)'s box (§6.1). M = 3: **certified**, given the M-harmonic
 start-value lemma `fixed_point_M.start_data_M` and Braaksma's theorem, on a parallelepiped of outer half-width 10⁻¹⁰
@@ -13,8 +14,8 @@
 four box runs of §7 on the local machine all failed: at h = 3×10⁻¹¹ and 10⁻¹² (twice) the derivative radius grew past
 order one on the arc, and the one run that finished (h = 10⁻¹³) fails the Krawczyk test (margins −13.7, −11.8, +0.29,
 +0.33, −542, −511; `logs/A_M_krawczyk_M5_1e-13.log`). The design change of §7, a second-order chart with C² enclosures,
-is [A_M2.md](A_M2.md): with it M = 5 is **certified** (given the same lemma and Braaksma's theorem; the new code not
-reviewed) on a curved parallelepiped of outer half-width 10⁻¹⁰ (A_M2.md §6.4). An independent review
+is [A_M2.md](A_M2.md): with it M = 5 is **certified** (given the same lemma and Braaksma's theorem; the new code reviewed in
+[A_M2_review.md](A_M2_review.md)) on a curved parallelepiped of outer half-width 10⁻¹⁰ (A_M2.md §6.4). An independent review
 ([A_M_review.md](A_M_review.md)) confirms the M = 3 certificate. With lemma 6a run on this certificate's state box
 (ρ₆ₐ = 0.041) and the other lemmas of [A_lemmas_M.md](A_lemmas_M.md), **Theorem A(3) is certified** (A_lemmas_M.md §7).
 
@@ -88,7 +89,7 @@
   by the certificate's code path, not a certificate. `python3 certify_M.py startcheck M` checks that the lemma's
   enclosure contains this file's own series (independent code; at M = 3 they agree to 4×10⁻⁷⁶).
 - The correctness of the code: python-flint 0.9 (arb/acb), validated_int.py's Lohner step (reviewed,
-  [A_review.md](A_review.md)), and the new files above (not reviewed).
+  [A_review.md](A_review.md)), and the new files above (reviewed with the M = 3 certificate, [A_M_review.md](A_M_review.md)).
 
 ## 2. Design: the Schur-corrected Krawczyk test in a tilted box
 
@@ -330,7 +331,7 @@
 radius in Y Dg̃(X) 0.23.
 
 > **Certified** (given the start-value lemma `fixed_point_M.start_data_M` and Braaksma's theorem; trusted code
-> certify_M*.py, validated_int.py and python-flint 0.9; the new files are not reviewed): the D-jump function g of §1 at
+> certify_M*.py, validated_int.py and python-flint 0.9; the new files reviewed in [A_M_review.md](A_M_review.md)): the D-jump function g of §1 at
 > M = 3 has exactly one zero in the parallelepiped
 > P = p̂ + T·([−10⁻¹⁰, 10⁻¹⁰]² × [−4×10⁻¹⁶, 4×10⁻¹⁶]²), and its Jacobian is non-singular there. The zero lies in
 > (endpoints rounded outward):
```

```diff
--- a/proof/A_lemmas_M.md
+++ b/proof/A_lemmas_M.md
@@ -9,7 +9,9 @@
 Status words as in AGENTS.md: *proved* (by hand), *certified* (computer-assisted, the trusted code named),
 *conditional* (with the hypothesis named), *numerical evidence*. The one fact assumed throughout, as at M = 1, is
 Braaksma's summability theorem for the existence of the lateral solutions y± and their asymptotics
-([C_selection.md](C_selection.md) Lemma B4(M)). Nothing here has been reviewed independently yet.
+([C_selection.md](C_selection.md) Lemma B4(M)). Lemma 6a(3) and the M = 3 assembly are reviewed independently in
+[A_M_review.md](A_M_review.md), and the lemma 6a and 6b₁ code at M = 5 by #39 (the summary below); the majorant code
+of lemmas S and 6b₀ is not yet reviewed.
 
 Contents
 
@@ -763,8 +765,7 @@
 - a certificate of S₁ = 0 alone, at fixed higher harmonics: that would not satisfy 6a, which needs all (M + 1)/2
   complex J_{D,μ}(ξ₁), or an invertible linear image of them such as `hier.py`'s de-mixed residuals.
 
-At M = 5 everything waits for its certificate except lemma S, 6b₀, the flat estimate and step (i), which are certified
-here. 6b₁(5) is a local-machine task (§9).
+At M = 5 the point certificate is [A_M2.md](A_M2.md)'s, and the summary above gives the status of each lemma at it.
 
 ## 8. Mutation table
 
```

#### P2-7. B5_inner.md's status table still has (H8) open and B5(2) estimated (superseded)

| location | says |
|---|---|
| proof/B5_inner.md:586–587 (§8) | "(H8) itself: **open**"; "B5(2) error at n₀(1): … estimated in section 7 (ln n ≳ 105)" |
| proof/B5_inner.md:56 (summary) | "A discrepancy, to resolve before B(1) and A(1) can be combined": C_selection.md §1 and the paper truncate b, c to \|ν\| ≤ M − 1 |
| proof/B3_borel.md:796 (§8) | "B3 actual solution, fixed order: = B8 on the real segment (to be written with constants)" |
| proof/B8_constants.md §0, §8 | (H8) **certified** for n ≥ e^{73.89}, δ = (8.3×10⁻⁵, 79.04, 3.232, 6.464); B8(1) certified for n ≥ e^{62} |
| proof/B_chain.md §0, §3 | B5(2) certified, error ≤ e^{76.94}/n for n ≥ e^{73.89} |
| proof/C_selection.md:103–104; paper/math/main.tex:880–888 | b and c on \|ν\| ≤ M + 1 |

B5_inner.md's summary already marks its B5(2) estimate superseded, but its status table and the discrepancy bullet were
not updated. Canonical: B8_constants.md, B_chain.md and C_selection.md §1.

```diff
--- a/proof/B5_inner.md
+++ b/proof/B5_inner.md
@@ -53,7 +53,7 @@
   (with the best κ₁ ≈ 0.85; `b5_threshold.py`). That is an estimate, not part of this lemma (section 7).
   *Superseded:* the estimate uses the exponent ⟨F²⟩ where the fast mode of the Φ-component has
   Re 𝒢 = ⟨F²⟩ − 3/2, and omits the weight m_Φ = |z|; B5(2) is now certified ([B_chain.md](B_chain.md) §3).
-- **A discrepancy, to resolve before B(1) and A(1) can be combined.**
+- **A discrepancy, since resolved: C_selection.md §1 and the paper now use b and c on |ν| ≤ M + 1.**
   - C_selection.md §1 and the paper truncate b and c to even |ν| ≤ M − 1.
   - The certified inner problem (and every inner solver, `lateral_np.py`, `hier.py`, `first_order_jet.py`) keeps c on
     |ν| ≤ M + 1. For M = 1 that is c on {0} against {0, ±2}.
@@ -583,8 +583,8 @@
 | existence of a super-solution, hence some n₀(M), for each fixed M | **proved, conditional on (H8)**, with Braaksma's Gevrey bounds; no numerical n₀(M) |
 | M = 1: all constants, the super-solution, n₀(1) = max(e^{73.89}, n₈) (max(e^{69.78}, n₈) at Z = 0.009) | **certified** (`b5_bounds.py`, python-flint; with lemma S of A_certify.md rerun over the complex box); **conditional on (H8)** and on Braaksma's theorem as in Theorem A(M = 1) |
 | leading data δ⁰ of (H8) | **numerical evidence** (`b5_data.py`; cross-checked) |
-| (H8) itself | **open**: B8's remainder in these variables, plus the series comparison of section 7 |
-| B5(2) error at n₀(1) | not part of this lemma; estimated in section 7 (ln n ≳ 105, `b5_threshold.py`) |
+| (H8) itself | **certified** for M = 1, n ≥ e^{73.89}, δ = (8.3×10⁻⁵, 79.04, 3.232, 6.464) ≤ δ_ref ([B8_constants.md](B8_constants.md) §8) |
+| B5(2) error at n₀(1) | not part of this lemma; **certified** in label form, error ≤ e^{76.94}/n for n ≥ e^{73.89} ([B_chain.md](B_chain.md) §3); section 7's estimate (ln n ≳ 105) is superseded |
 
 The trusted code: `b5_bounds.py` (new, not reviewed independently), `b5_formulas.py` (new), `fixed_point.py` and
 `lemma6_sector.py` (A_certify.md; reviewed in A_review.md, rerun over a different box), python-flint 0.9.
--- a/proof/B3_borel.md
+++ b/proof/B3_borel.md
@@ -793,7 +793,7 @@
 | (H-σ), M = 1 | Borel transform analytic near [0, ∞) | **conjecture**; **numerical evidence** σ ≈ π/2; proof plan §4 with one open technical point (derivative loss) |
 | (H-σ), M ≥ 3 | | **conjecture**; numerical evidence that σ is set by an x-independent singularity near −1.2 (M = 3) |
 | B3 formal half | | **proved** (B3.7) |
-| B3 actual solution, fixed order | | = B8 on the real segment (to be written with constants) |
+| B3 actual solution, fixed order | | = B8 on the real segment, written with constants in [B8_constants.md](B8_constants.md) (**certified** for M = 1, n ≥ e^{62}) |
 | B3 strong form (1-sum) | | ⇔ (H-σ) (Nevanlinna–Sokal), **open** |
 | B4 summability | | **cited**, Braaksma 1992, hypotheses checked against the paper |
 | B4 parameter-analyticity | | **proved** given Braaksma (Lemma B4.P); not in Braaksma 1992 or the Ramis–Sibuya 1994 abstract |
```

#### P2-8. PROOF.md's staged plan still lists the M = 5 certificate and Theorem B(1) as future work (superseded)

| location | says |
|---|---|
| PROOF.md:407 (§7, stage 2) | difficulty "a computer-assisted-proof project of a few months", who "computer-assisted-proof community" |
| PROOF.md:410 (§7, stage 5) | Theorem B for a finite-harmonic model, "hard, a full paper" |
| PROOF.md:414–416 | "Stages 1 and 2 are achievable … would already be a publishable result: a computer-assisted certificate that the five-harmonic inner problem selects ω/ω∞ = 0.9992 ± 10⁻⁶" |
| PROOF.md:225–232 (§4), 321 (§5) | A(M = 5) certified on a box of half-width 10⁻¹⁰, ω/ω∞ ∈ [0.99922840320656, 0.99922840320657]; B(1) certified for n ≥ e^{89.75} |

#653's finding 8 fixed the Summary and §4 but not §7. Canonical: PROOF.md §4 and §5.

```diff
--- a/PROOF.md
+++ b/PROOF.md
@@ -404,16 +404,16 @@
 | Stage | Goal | Ingredients | Difficulty | Who |
 |---|---|---|---|---|
 | 1 | Write the inner problem as a first-order system at the irregular singular point ξ = ∞ (t = 1/ξ²), list the transseries, and identify S_ν with Costin's Stokes constants | Costin (1998); the existing formal series (`solver/inner_borel/`, `inner_np.py`) | low–moderate, weeks | here |
-| 2 | Certified M = 5 zero (Theorem A(M=5)) | A.1 by fixed point with explicit constants; A.2 validated Taylor integration; A.4 Krawczyk; tolerances from `feasibility.py` | moderate, a computer-assisted-proof project of a few months | computer-assisted-proof community |
+| 2 | Certified M = 5 zero (Theorem A(M=5)) | A.1 by fixed point with explicit constants; A.2 validated Taylor integration; A.4 Krawczyk; tolerances from `feasibility.py` | done (§4, [proof/A_M2.md](proof/A_M2.md)) | here |
 | 3 | Growth bound for σ_k and the entire-function structure; factor out the Γ dependence | the exact amplitude recursion; rigorous Darboux bounds for its late orders | moderate–hard | exponential asymptotics |
 | 4 | Harmonic tail bound A.3, then Theorem A(∞) | weighted dual norms on the tuned family; Schur complement estimates | hard, genuinely new | both communities |
-| 5 | Theorem B for a finite-harmonic model (θ-dependence truncated in the full equations, not only the inner problem) | Xie–Tanveer architecture; Costin; B.3–B.7 | hard, a full paper | exponential asymptotics |
+| 5 | Theorem B for a finite-harmonic model (θ-dependence truncated in the full equations, not only the inner problem) | Xie–Tanveer architecture; Costin; B.3–B.7 | hard, a full paper; done for M = 1, assuming Braaksma's theorem (§5, [proof/B_chain.md](proof/B_chain.md)) | exponential asymptotics; here for M = 1 |
 | 6 | Theorem B in the strip-analytic setting | Costin–Tanveer PDE Borel summability; gap (b) | research-level | exponential asymptotics |
 | 7 | Corollary C, with a certified large-D solution | Reiterer–Trubowitz method at one D | moderate given 2–6 | GR / computer-assisted proofs |
 
-Stages 1 and 2 are achievable with current tools and would already be a publishable result: a computer-assisted
-certificate that the five-harmonic inner problem selects ω/ω∞ = 0.9992 ± 10⁻⁶, together with the measured convergence
-in M and the untruncated bracket. Stage 4 is where a new idea is needed.
+Stage 2 is done (§4): the five-harmonic inner problem selects ω/ω∞ ∈ [0.99922840320656, 0.99922840320657], assuming
+Braaksma's theorem and the start-value lemma. With the measured convergence in M and the untruncated bracket, stages 1
+and 2 would already be a publishable result. Stage 4 is where a new idea is needed.
 
 What is being worked on now, and in what order: [notes/PROOF_QUEUE.md](notes/PROOF_QUEUE.md).
 
```

#### P2-9. The one-harmonic model's root: 1.765 or 1.769 (superseded, a bar dropped)

| location | says |
|---|---|
| README.md:1861, GAMMA.md:171, paper/math/main.tex:864 | the one-harmonic model has a root λ = 1.765 |
| README at 3498262d (where 1.765 entered) | "λ = 1.765 ± 0.005 (γ = 0.567; `pert_ratio.py`)", in the one-harmonic model at its zero, the bar's kind not stated |
| notes/FIRST_ORDER.md:33, 314 | λ* = 1.769122008 for M = 1, the model's growth rate near 2 at its zero (A = 2.5421480, ω*/ω∞ = 0.776994014) |

The two are the same quantity: the later one lies inside the first one's ±0.005, which the quotes dropped. Canonical:
notes/FIRST_ORDER.md (`data/first_order_hier_M1_xi0.2.json`). The math paper's line is in finding P2-3's diff.

```diff
--- a/README.md
+++ b/README.md
@@ -1858,7 +1858,7 @@
 - that λ = 2 is the only physical unstable mode at D = ∞;
 - the 1/n coefficients without truncation, and a proof of the O(1/n) matching.
 
-The one-harmonic model's root, λ = 1.765, is not a counterexample, for two reasons. Its background is not the D = ∞
+The one-harmonic model's root, λ = 1.769, is not a counterexample, for two reasons. Its background is not the D = ∞
 solution, and Galerkin truncation does not commute with the gauge step, which multiplies by P(θ). More in
 [GAMMA.md](GAMMA.md).
 
--- a/GAMMA.md
+++ b/GAMMA.md
@@ -168,7 +168,7 @@
 $`r_{\rm div} = \sqrt{2(\varepsilon-1)}`$, from an exact symmetry; γ∞ = 1/2 follows given Conjecture 1, the uniqueness
 of the unstable mode and continuity of λ(D).
 
-**The one-harmonic model's 1.765.** There λ = 2 is not a root, for two reasons:
+**The one-harmonic model's 1.769.** There λ = 2 is not a root, for two reasons:
 - the model's background is not the D = ∞ solution (its ω is 22% low);
 - Galerkin truncation does not commute with the gauge fix, which multiplies by P(θ), so the truncated model has no exact
   λ = 2 mode with h₀ = 0. (The s-translation alone does commute with the truncation.)
```

#### P2-10. B_tail.md still waits for M = 7 (superseded)

| location | says |
|---|---|
| proof/B_tail.md:52, 307–309 (§7) | "M = 7 should land at 0.99997–1.00017, a sharp test when it runs"; "M = 7 … is the next data point" |
| proof/B_tail_M7.md §2–3; README.md:1617 | M = 7 gives ω/ω∞ = 1.000022 (±2×10⁻⁵, the scatter of repeated solves), inside the prediction; last ratio −0.049 |

Canonical: proof/B_tail_M7.md.

```diff
--- a/proof/B_tail.md
+++ b/proof/B_tail.md
@@ -49,7 +49,8 @@
   A 10⁻⁴ box needs M = 9 explicit.
 - **The two routes are consistent.** The truncated sequence and the alternation sign put the limit at 0.99923–1.00011. The
   untruncated bracket at order 12 is 0.99915–1.00011, with the central Padé value 0.99989. They overlap, and both
-  contain 1. M = 7 should land at 0.99997–1.00017, a sharp test when it runs.
+  contain 1. M = 7 should land at 0.99997–1.00017, a sharp test when it runs. (It landed at 1.000022, inside the
+  prediction; [B_tail_M7.md](B_tail_M7.md) §2.)
 - **Secondary Stokes lines from F's complex singularities: no sign of them.** They are absent at every finite order in
   A, where each order is a trigonometric polynomial and the late orders contain only fast-mode families. S₁ is numerically
   entire in A², and the two routes agree. What is not excluded is a non-uniformity in the order of limits (sum in A,
@@ -305,8 +306,8 @@
 ## 7. What remains open
 
 1. **The jump-conditioned tail bound itself.** The per-pair ratio −0.045 to −0.057 is measured on three points (M = 1, 3,
-   5), with M = 1 pre-asymptotic. M = 7 (`hier.py 7 … --predict --floor --schur`) is the next data point. The prediction is
-   c₇ = 0.99997–1.00017.
+   5), with M = 1 pre-asymptotic. M = 7 (`hier.py 7 … --predict --floor --schur`) was the next data point, with the
+   prediction c₇ = 0.99997–1.00017. It gives 1.000022 and a last ratio of −0.049 ([B_tail_M7.md](B_tail_M7.md) §2–3).
 2. **Why the Galerkin projection explodes at σ_{M+1}.** Either its multiplier is genuinely large or its late orders have
    a family structure outside the fit basis (the fit residual rises). Deciding this is a few hours' work: compute the
    projected system's fast-mode exponents, refit, and compare with a lateral (`inner_np.py`) evaluation of the same
```

#### P2-11. Matter notes list as open what other matter notes have done (superseded)

| location | says |
|---|---|
| matter/SIGMA.md:9, §7 (408–415) | "The DSS branch was not computed (§7)"; "Not known. Whether the DSS branch exists at all D for fixed η, and where η_c(D) … goes" |
| matter/SIGMA_DSS.md §1 | the DSS branch computed at D = 4 (η_c = 0.1702) and continued to D = 10.5–11 at η = 1 and 0.3 |
| matter/STRUCTURE.md:456–458 (§10) | wave maps' D = ∞ inner problem "was not computed"; "the finite-D Hirschmann–Eardley solutions were not computed here; that … would test the D = ∞ selection rule cleanly" |
| matter/SIGMA.md §6 (363–366); matter/CSS.md §7 | the D = ∞ CSS ground state at η = 0.14, A₀ = 1.9225686; the finite-D Hirschmann–Eardley solutions to D = 80 and the selection test, which passes |

Neither SIGMA.md nor STRUCTURE.md points to the notes that did the work. Canonical: SIGMA_DSS.md and CSS.md. The diff
adds the pointers and changes no claim.

```diff
--- a/matter/SIGMA.md
+++ b/matter/SIGMA.md
@@ -6,7 +6,8 @@
 inner variables, breaks this. This page tests the prediction on the self-gravitating hedgehog σ-model (wave map into
 $`S^{D-1}`$, coupling η = 4πGf²). It covers: the D-dimensional equations (checked against Bizoń–Wasserman in D = 4); the
 continuously self-similar (CSS) critical solution and its growing mode, validated against every published D = 4 and
-flat-space number found; its continuation in D; and the D = ∞ inner problem. The DSS branch was not computed (§7).
+flat-space number found; its continuation in D; and the D = ∞ inner problem. The DSS branch was not computed here
+(§7); [SIGMA_DSS.md](SIGMA_DSS.md) computes it.
 
 Notation as in the README: D = n + 2, d = n + 1 space dimensions, x = r/(−t), τ = −ln(−t), s = r²/(2n), and γ = 1/λ is
 the radius exponent ($`r_{\rm BH} \propto (p-p_*)^\gamma`$; the mass exponent is (D − 3)γ). Statements are marked, as in
@@ -406,7 +407,9 @@
 
 ## 7. The DSS branch
 
-Not computed: the time went to the CSS branch, and the ℓ = 1 multiplet (the η → ∞ limit) is another agent's task. What is known about it here:
+Not computed here: the time went to the CSS branch, and the ℓ = 1 multiplet (the η → ∞ limit) is another agent's task.
+[SIGMA_DSS.md](SIGMA_DSS.md) has since computed the branch at D = 4 and continued it to D ≈ 10.5–11 at η = 1 and
+0.3. What was known about it here:
 - *Argued.* At fixed η its D = ∞ inner problem is the system of §6.1, n-free and without s-translation, so its λ∞ is not
   protected at 2. It depends on η and tends, as η → ∞, to the ℓ = 1 multiplet's value (STRUCTURE.md). In D = 4 the
   DSS exponent is about 0.11 for η ≳ 0.2 and tends to 0.1185 ± 0.0005 as η → ∞ [HLPTA00]. [LTHA02] give γ_DSS ≈ 0.11.
--- a/matter/STRUCTURE.md
+++ b/matter/STRUCTURE.md
@@ -453,9 +453,12 @@
 - **ℓ ∝ n** needs the turning-point layer with gravity written out, including how the time dependence enters.
 - **Yang–Mills:** that the type II threshold in D ≥ 6 is the U-system's critical solution is argued, not shown; no
   Einstein–Yang–Mills collapse in D ≥ 5 has been computed.
-- **Wave maps:** the D = ∞ inner problem as a function of η (its CSS solutions and the DSS branch) was not computed.
-- **The CSS check:** the finite-D Hirschmann–Eardley solutions were not computed here; that, and the next order in 1/n
-  of the CSS inner problem, would test the D = ∞ selection rule cleanly.
+- **Wave maps:** the D = ∞ inner problem as a function of η (its CSS solutions and the DSS branch) was not computed
+  here. [SIGMA.md](SIGMA.md) §6 solves its CSS ground state at η = 0.14, and [SIGMA_DSS.md](SIGMA_DSS.md) follows the
+  finite-D DSS branch to D ≈ 10.5–11.
+- **The CSS check:** the finite-D Hirschmann–Eardley solutions were not computed here. [CSS.md](CSS.md) §7 has since
+  computed them to D = 80 and made the test, which passes. The next order in 1/n of the CSS inner problem is not
+  computed.
 
 ## 11. Scripts
 
```

#### P2-12. The math paper's coefficient record stops at σ₁₂ and σ₁₇ (superseded, inconsistent within the paper)

| location | says |
|---|---|
| paper/math/main.tex:1916 | "σ₁–σ₁₂ are available at ω/ω∞ = 0.99, 1.00, 1.01" |
| paper/math/main.tex:2185 | "σ₁₄–σ₁₇ were fitted with one window, to plateau steps of 3×10⁻¹³ to 1.5×10⁻¹² relative" |
| paper/math/main.tex:2129–2180 | the evidence uses σ₁₃ … σ₂₂ |
| README.md:1641–1648 | σ₁₅ … σ₂₂ converged to 4.5×10⁻¹³ … 2.3×10⁻¹² relative (σ₁₈ the largest) |
| solver/inner_amp/scan45J1250/ | `sigF_w{0.9900,1.0000,1.0100}_K45_J1250_p1280.json` (σ to 22 at the three ω) |

Canonical: the README's table and the data files.

```diff
--- a/paper/math/main.tex
+++ b/paper/math/main.tex
@@ -1913,7 +1913,7 @@
 \sigma_5 = -0.00555413676 - 0.00280595708\iu,\\
 \sigma_6 = (-2.7853745790 - 10.776628405\iu)\times10^{-4},\quad \sigma_7 = (1.0223846436 - 1.4905053530\iu)\times10^{-4};
 \end{gather*}
-$\sigma_1$--$\sigma_{12}$ are available at $\omega/\winf = 0.99$, $1.00$, $1.01$ (\repo{solver/inner\_amp/}). A fit is accurate but not an enclosure; Lemma~\ref{lem:sigma1} below encloses $\sigma_1$
+$\sigma_1$--$\sigma_{22}$ are available at $\omega/\winf = 0.99$, $1.00$, $1.01$ (\repo{solver/inner\_amp/}). A fit is accurate but not an enclosure; Lemma~\ref{lem:sigma1} below encloses $\sigma_1$
 rigorously. Summing \eqref{eq:sigmak} directly at the zero is hopeless: the terms $|\sigma_k|A_0^{2k}$ grow to about $240$ at
 $k = 9$ and are still $160$ at $k = 12$, and would fall below $10^{-6}$ only near $k \approx 37$.
 
```

```diff
--- a/paper/math/main.tex
+++ b/paper/math/main.tex
@@ -2182,7 +2182,7 @@
 \subsection{What the evidence does not show}
 The $\pm0.00006$ above is the range of calibrated estimators, not an error bound, and it rests on the assumption about the
 other zeros of $S_1$.
-The coefficients $\sigma_k$ come from late-order fits; two fit windows agree to about $10^{-19}$ for $\sigma_1$--$\sigma_{12}$ ($\sigma_{13}$ to $2\times10^{-13}$ relative; $\sigma_{14}$--$\sigma_{17}$ were fitted with one window, to plateau steps of $3\times10^{-13}$ to $1.5\times10^{-12}$ relative), and the
+The coefficients $\sigma_k$ come from late-order fits; two fit windows agree to about $10^{-19}$ for $\sigma_1$--$\sigma_{12}$ ($\sigma_{13}$ to $2\times10^{-13}$ relative; $\sigma_{14}$--$\sigma_{22}$ to plateau steps of $3\times10^{-13}$ to $2.3\times10^{-12}$ relative), and the
 estimates need them to about $10^{-10}$, but only $\sigma_1$ is enclosed rigorously (Lemma~\ref{lem:sigma1}). At the zero the series \eqref{eq:sigmak} is far from summed, and every
 estimator (crossings, ratios, Pad\'e approximants, the factored series) is a resummation whose convergence is observed, not
 proved; the arc-shaped Borel singularity of Section~\ref{sec:entire} is a known source of slow convergence. Settling the other zeros needs more orders, starting with $\sigma_{23}$. Finally, agreement of the numbers is evidence that the selection rule is the right one, not a proof
```

#### P2-13. PROOF.md gives the computed range as D = 8–61 (superseded)

| location | says |
|---|---|
| PROOF.md:79 (Corollary C), 394 (§6) | "the critical solutions computed for D = 8–61"; "The double-double solutions at D = 8–61" |
| README.md:91, 109 | 1199 double-double points, D = 8.25–67.77 |

Canonical: the README's main-results table and `hp/data/delta_all.csv`. The diff below is §6's line; Corollary C's is in
finding P2-1's diff. The math paper's "computed to D = 61" (abstract) and "857 values 8.25 ≤ D ≤ 61.03" (§2.2) are the
same scope, left to #653's recommended task 6 (the papers at the repository's present state).

```diff
--- a/PROOF.md
+++ b/PROOF.md
@@ -391,8 +391,8 @@
 - **Uniqueness in the box.** Theorem A must certify that there is exactly one zero of S₁ in a box that also contains
   the numerically extrapolated ω∞ = 3.82213590 (±4×10⁻⁸) and A∞ ≈ 2.70–2.72. A box of radius 10⁻³ suffices for the
   identification; it does not need the numerics' digits.
-- **That the computed branch is the branch of Theorem B.** The double-double solutions at D = 8–61 are numerical. A
-  Reiterer–Trubowitz-style computer-assisted existence proof at one large D (say D = 40–60), with the certified solution
+- **That the computed branch is the branch of Theorem B.** The double-double solutions at D = 8.25–67.77 are numerical.
+  A Reiterer–Trubowitz-style computer-assisted existence proof at one large D (say D = 40–60), with the certified solution
   inside the basin of Theorem B's construction, would close this; alternatively, Theorem B's construction at n ≥ n₀ with
   explicit n₀ and a certified solution at some D ≥ n₀ + 2.
 
```

#### P2-14. A_certify.md calls the certified lemma 6b₁ "proved" (inconsistent status word)

| location | says |
|---|---|
| proof/A_certify.md:41, 1386 | "lemma 6b₁ is proved"; "Lemma 6b₁ is proved in §12" |
| proof/A_certify.md:37 (the paragraph before); PROOF.md:208; README.md:209 | lemma 6b₁ "is certified" by the covering with validated Taylor discs |

AGENTS.md: *proved* is by hand, *certified* computer-assisted. #653's finding 8 noted the same word in PROOF.md, which
main has since changed (71015044).

```diff
--- a/proof/A_certify.md
+++ b/proof/A_certify.md
@@ -38,7 +38,7 @@
 the box, and it is simple**, at A ∈ [2.54214804761338, 2.54214804761975], ω/ω∞ ∈ [0.776994013607438,
 0.776994013612010] (§11.7).
 
-**Since then (§12):** lemma 6b₁ is proved. The annulus is covered by 178 validated Taylor discs (radius 10–24 in
+**Since then (§12):** lemma 6b₁ is certified. The annulus is covered by 178 validated Taylor discs (radius 10–24 in
 u = 1/ξ², orders 40–70). Their centre values come from single-trajectory Lohner runs along the start ray and 12 arcs,
 and the disc functions glue to one analytic continuation of y⁺. With §11 this gives **Theorem A(M = 1)** without
 conditions, given Braaksma's theorem for the existence of y⁺ (§12.6): the M = 1 selection condition has exactly one
@@ -1383,7 +1383,7 @@
 has exactly one simple zero x* in X. By 11.5 (whose hypothesis now holds), J_D(0.3; x) = 0 ⇔ y⁺ ≡ y⁻. ∎
 
 **What remains:** lemma 6b₁ only (11.4 lists three routes and their cost). Without it, the honest statement is item 1
-above together with 6a as stated. (Lemma 6b₁ is proved in §12, and the theorem is restated there without the
+above together with 6a as stated. (Lemma 6b₁ is certified in §12, and the theorem is restated there without the
 condition, §12.6.)
 
 **Independence.**
```
#### P2-15. The σ-model's γ at its fold rounds to 0.514, not 0.513 (inconsistent, minor)

| location | says |
|---|---|
| README.md:1938 | γ rises "to 0.513 at a fold at D = 8.544" |
| matter/SIGMA.md:42, 252, 261; paper/physics/main.tex:752 | γ = 0.5135 (0.51355, λ = 1.94724) at the fold |
| solver/matter/data/sigma_fold_eta0.14.jsonl | γ = 0.513548 at the point of largest D, 8.544346; a parabola through it and its neighbours puts the fold at D = 8.54438 |

Canonical: the data file. The new check `sigma_fold` ties SIGMA.md's and the physics paper's figures to it.

```diff
--- a/README.md
+++ b/README.md
@@ -1935,7 +1935,7 @@
 
 **σ-model, self-similar (CSS) branch** ([matter/SIGMA.md](matter/SIGMA.md)). The D-dimensional equations are checked
 against Bizoń and Wasserman's at D = 4, and the D = 4 values are reproduced: λ ≈ 5.14, γ ≈ 0.18, and the horizon
-thresholds. Continued in D at fixed coupling η = 0.14, γ rises from 0.188 at D = 4 to 0.513 at a fold at D = 8.544,
+thresholds. Continued in D at fixed coupling η = 0.14, γ rises from 0.188 at D = 4 to 0.5135 at a fold at D = 8.544,
 where the branch ends. In flat space it collapses onto a singular cone (u ≡ π/2) as D → 8, with λ → 2. So this branch
 never reaches large D and cannot test γ∞. Its γ ends near 1/2 for a different reason. The test of the prediction needs
 the σ-model's DSS branch.
```

#### P2-16. B_tail_M7.md quotes one part of complex σ₄ and σ₆ as "true" (inconsistent, minor)

| location | says |
|---|---|
| proof/B_tail_M7.md:288–289 | σ₄ = −399 "(true −0.029)", σ₆ = −1371 "(true −0.0011)" |
| proof/B_tail.md:214–215, README.md:1411 | σ₄ = −0.0289 + 0.0108i, σ₆ = −0.00028 − 0.00108i ("−0.0003 − 0.0011i") |

−0.029 is Re σ₄, but −0.0011 is Im σ₆. Canonical: proof/B_tail.md §4.

```diff
--- a/proof/B_tail_M7.md
+++ b/proof/B_tail_M7.md
@@ -285,8 +285,8 @@
 | hier's model, M = 1 | odd ≤ 1, even ≤ 2 | σ₁ unchanged (4.5×10⁻¹⁰, fit level) | σ₂ changed by 0.0590 (14% of \|σ₂\|) | `trunc_hier_M1_K5_J500_w1.0000.json` |
 | hier's model, M = 3 | odd ≤ 3, even ≤ 4 | σ₁–σ₃ unchanged (≤ 2.7×10⁻¹⁰) | σ₄ changed by 3.74×10⁻⁵ (0.12% of \|σ₄\|) | `trunc_hier_M3_K9_J500_w1.0000.json` |
 | hier's model, M = 5 | odd ≤ 5, even ≤ 6 | σ₁–σ₅ unchanged (≤ 4.2×10⁻¹⁰) | σ₆ changed by 7.9×10⁻⁹ (7×10⁻⁶ of \|σ₆\|) | `trunc_hier_M5_K13_J500_w1.0000.json` (a first run, cut by a container restart, `trunc_hier_M5_cut.log`) |
-| `amp_trunc.py`, M = 3 | all ≤ 3 | σ₃ changed by 5.3×10⁻⁴ | σ₄ = −399 (true −0.029) | B_tail.md §4 |
-| `amp_trunc.py`, M = 5 | all ≤ 5 | σ₅ changed by 4.4×10⁻⁴ | σ₆ = −1371 (true −0.0011) | B_tail.md §4 |
+| `amp_trunc.py`, M = 3 | all ≤ 3 | σ₃ changed by 5.3×10⁻⁴ | σ₄ = −399 (true −0.0289 + 0.0108i) | B_tail.md §4 |
+| `amp_trunc.py`, M = 5 | all ≤ 5 | σ₅ changed by 4.4×10⁻⁴ | σ₆ = −1371 (true −0.00028 − 0.00108i) | B_tail.md §4 |
 
 (`python3 trunc_seq_M7.py --amp`.) Lemma O's prediction holds exactly in both truncations: hier's model first changes at
 σ_{M+1}, `amp_trunc.py`'s at σ_M. And the explosion is a property of dropping the even harmonic M + 1, not of Galerkin
```


### Counts by kind

| kind | findings |
|---|---|
| superseded | P2-1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13 |
| inconsistent | P2-2, 15, 16 |
| inconsistent status word | P2-14 |
| unsupported (a bar whose kind or source is not named) | P2-4 |
| broken reference | none (below) |
| **total** | **16** |

### Recommended tasks

1. **The frontier rows of `delta_all.csv`.** What: give the pinned rows that only one pin reached (D 73.1–73.27) an err,
   or replace the 73.2657 row by the quad-double Δ_full1 of QUAD_DOUBLE.md §12.4 (T84-mac). Why: finding P2-2; the row
   at 73.265722 lies 1.06×10⁻¹⁴ above both quad-double values and carries no err. Acceptance: `export_delta_all.py
   --verify` passes, and a check that compares the row with the quad-double value fails on the CSV at 90a81f69 and
   passes after. Depends on: #658.
2. **A lint for stale status claims in the proof notes.** What: a script that lists, per named result (the names of
   `check_proof_table.py`), the status and review phrases of proof/*.md ("not reviewed", "open", "waits", "next data
   point", "proved" for a computer-assisted step) and compares them with the README's table "What is proved, and how".
   Why: findings P2-5, 6, 7, 10 and 14 are each a note that kept an old status after the work moved on, and #653 found
   the same in PROOF.md and PROOF_QUEUE.md. Acceptance: at 90a81f69 it lists C_selection.md:77, A_M.md:8,
   A_lemmas_M.md:12, B5_inner.md:586 and A_certify.md:41; with the diffs above applied it lists none of them; a planted
   line "Theorem B(1) … not reviewed" is listed. Depends on: #658.
3. **Theorem A(M = 5) in A_lemmas_M.md.** #40, open, covers it. Finding P2-6's second diff is the minimum until then.

### New checks (the commit "check_numbers: check the A(3) and A(5) enclosures, …")

`check_numbers.py --selftest` on the branch: 1629 of 1629 pass, and 1629 of 1629 reject a value shifted by one unit
(1573 before). PROOF.md, notes/FIRST_ORDER.md, matter/CSS.md and matter/SIGMA.md are added to DOCS. Every check passes
on 90a81f69 and with every diff above applied, since none of them quotes a changed line.

| check | quoted numbers | source |
|---|---|---|
| `m5_enclosure` | A(M = 5)'s zero, ω/ω∞ ∈ [0.99922840320656, 0.99922840320657] (README, PROOF.md) and A ∈ [2.7185791892117, 2.7185791892118] (PROOF.md), as outward roundings | solver/proof/logs/A_M2_krawczyk_M5_1e-10.log |
| `m3_enclosure` | A(M = 3)'s zero, A and ω/ω∞ (README, math paper), from the Krawczyk log and, rounded outward, from the independent mpmath-interval recheck | solver/proof/logs/A_M_krawczyk_M3_1e-10.log, A_M_recheck_M3_1e-10.log |
| `first_order_lam` | λ* = 1.769122008, 2.006804459, 1.9996689 (M = 1, 3, 5) and the short row 1.76912 (FIRST_ORDER.md) | solver/inner_pert/data/first_order_hier_M*.json |
| `sigma_fold` | the fold D = 8.5444 (8.544 in the README) and γ = 0.5135 there (SIGMA.md, physics paper), the table's 1.94724 and 0.51355 | solver/matter/data/sigma_fold_eta0.14.jsonl |
| `css_omega` | ω∞ 3.1998939020 and 3.9789695074 (inner), 3.1994 and 3.9784 with their fit ranges (extrapolated), the agreements 1.6×10⁻⁴ and 1.3×10⁻⁴ (CSS.md, README, math paper) | solver/matter/data/css_inner_kpath.jsonl, css_fit_k0.json, css_fit_k1.json |
| `css_lambda_table` | the 25 entries of CSS.md §7's table of λ∞ − 2 | solver/matter/data/css_fit_k1.json |

The 1.769 of finding P2-9 is checked in FIRST_ORDER.md, its source note. A check of the README's and GAMMA.md's text
belongs with that diff, once it is applied.

### Checked and found consistent

- **Theorem B in the math paper**: n₀ = e^{89.75}, e^{67.84}/n, B5(2)'s e^{76.94}/n for n ≥ e^{73.89}, B8 for n ≥
  e^{62}, δ = (8.3×10⁻⁵, 79.04, 3.232, 6.464), x = (392, 9.2×10⁴, 21, 30), θ_LC = 0.7355, d₀ = 3.59×10⁶, θ_J = 0.176,
  against C_selection.md §13.1, B_chain.md, B8_constants.md, B5_inner.md and P2b_review.md. The light-cone exponents'
  gaps (|Im| ≥ 6.89, 7.26 for M ≥ 3; |Re| < 0.02) and the M = 5 singular values 6.8×10³, 3.9×10³ against C_selection.md
  and A_certify.md.
- **Theorem A(M = 1) in the math paper** against A_certify.md: the contraction table, the enclosure radii, the 743 steps
  and J(x̂), the Krawczyk margins and the mutation table's C0 and C7, the zero box, the 178 discs and 6073 Lohner steps,
  and CAPD's margins at 10⁻⁸ and its zero box on the ray–arc path. **A(M = 3) and A(M = 5)**: the enclosures in the
  README, PROOF.md, the math paper, A_M.md, A_M2.md and A_M_review.md against the Krawczyk logs (now checked).
- **The entire-function section** against D_entire.md: the ratio-law table, the G table, the Jacobian's singular values
  0.53 and 0.26, the margins 2.6×10⁻³ and 5.7×10⁻⁴, C₃ = 110.3 and the σ₁ enclosures. **Gap (b)** against B_tail.md: q∞,
  |q|A² = 0.0633, σ_F = 1.428, 0.0574, −399 and −1371, 1.380, 0.374.
- **The evidence for K = 13–22** in the math paper and the README against PLANTED_SELECTION.md §5.4, §6.9 and
  §7.8–14.10: every arg crossing, ring-free zero, estimator range, ring count and σ error. The README's planted-truth
  biases (1.3–5.4×10⁻³, −0.8 to +1.9×10⁻⁴, up to 8×10⁻⁴, 0.99992–1.00019) against §3.3–4.
- **QUAD_DOUBLE.md's Δ values**: all 37 occurrences of Δ to 12–19 digits in §1–13.24 against `delta_qd.csv` and
  `delta_all.csv`. Each equals a CSV value to the digits shown, apart from the frontier (finding P2-2), the pin ladder's
  uncorrected Δ at 73.27, which differ from the CSV by the tabulated correction 1.16×10⁻¹², and the three chain values
  of 13.19 at 79.77–80.27, which the later chain replaced.
- **Matter**: CSS.md's |f(0)| and A² agreements (3.1×10⁻⁵, 5.2×10⁻⁵; 1.8×10⁻⁵, 2.5×10⁻⁵) against `css_fit_k*.json` and
  `css_inner_kpath.jsonl`; SIGMA_DSS.md's offsets (−0.0028 to −0.0033 at η = 1, −0.0146 to −0.0165 at η = 0.3) against
  its own table and the README; SIGMA.md's A∞ = 1.9226 ± 0.0003 (a spread over the fit's degree, which covers the three
  fits 1.92247–1.92282) against A₀ = 1.9225686; ELL.md's D = 4 values against Olabarrieta et al.'s; CSS.md's κ = 1
  uniqueness against COMPLEX_CRITICAL.md's κ = 0 pair, which are different models.
- **Paths**: every repository path that the scope files cite exists on main, apart from two that the notes date
  themselves: `hp/full_step_check.py` (on wip/qd-horizon, QUAD_DOUBLE.md §12) and `hp/src/qd.rs` (moved to multidouble,
  the note's opening).

### Coverage: files read

- **In full**: paper/math/main.tex §1 Results and status table, §2, §3, §4, §6, §7 (Lemma S's proof read for its
  numbers), §8 and §9; PROOF.md §1–3 and §6–8 with its references; proof/D_entire.md Summary to §3 and §5; matter/ELL.md
  §5–9; matter/CSS.md summary, §6 and §7; matter/YM_FAMILY.md; matter/SIGMA_DSS.md §1; notes/PLANTED_SELECTION.md §1,
  §3.3, §5.4, §6.9 and §7.8–14.10 (each order's combination and verdict); notes/QUAD_DOUBLE.md's opening, §1, §9–12.7,
  §13.17–13.20 and §13.23–13.24.
- **In part**: paper/math/main.tex §5 (Proposition 5, read for its numbers); proof/A_certify.md (status, §4.4–4.5, §9.3,
  §10.3, §11.7, §12.6), A_lemmas_M.md (summary, §7), A_M.md (status, §1, §6.2), A_M2.md (status, §6.4), A_M_review.md
  and A_M2_review.md (verdicts, A_M2's item 9), A_review.md (the linear check), B_chain.md (§0), B8_constants.md (§0,
  §8, §9), B5_inner.md (summary, §7, §8), B3_borel.md (§8), B_tail.md (verdict, §4, §7), B_tail_M7.md (summary, §4),
  C_selection.md (§0, §1, §13.1), FORMAL.md, P3_scope.md and P5_scope.md (openings); matter/SIGMA.md (summary, §5, §7),
  STRUCTURE.md (§10), COMPLEX_CRITICAL.md (summary); notes/QUAD_DOUBLE.md §12.4's tables; notes/PLANTED_SELECTION.md
  §2.1.
- **Searched, not read**: the rest of proof/ (P2_review.md and P2b_review.md beyond their constants, the bodies of
  B_chain.md, B8_constants.md, B5_inner.md, B3_borel.md and C_selection.md §2–12), matter/LITERATURE.md, the bodies of
  SIGMA.md and STRUCTURE.md, QUAD_DOUBLE.md §2–8, §13.1–13.16 and §13.21–13.22, and PLANTED_SELECTION.md §2.2–3.2,
  §3.4–6.8 and each order's procedure (§7.0–14.7). They were searched for status phrases ("not reviewed", "open",
  "waits", "next data point", "proved"), for the numbers above, and for the paths they cite.
- **Data files read**: hp/data/delta_all.csv, delta_qd.csv, delta_dd.csv, delta_dd_lowD.csv;
  solver/proof/logs/A_M2_krawczyk_M5_1e-10.log, A_M_krawczyk_M3_1e-10.log, A_M_recheck_M3_1e-10.log;
  solver/inner_pert/data/first_order_hier_M{1,3,5}*.json; solver/matter/data/css_fit_k0.json, css_fit_k1.json,
  css_inner_kpath.jsonl, sigma_fold_eta0.14.jsonl; the file list of solver/inner_amp/.
