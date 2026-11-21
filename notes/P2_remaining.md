# P2's remaining items: a review plan, and Lemma J's θ₀ checked independently

Issue #988, queue item P2 of [PROOF_QUEUE.md](PROOF_QUEUE.md). P2 ([proof/P2_review.md](../proof/P2_review.md)) and
P2b ([proof/P2b_review.md](../proof/P2b_review.md)) left six certificate items without an independent check
(P2_review.md §5, P2b_review.md §8). This note plans a check of each (§1) and does the smallest (§2). Written
2026-11-19 by Claude (Anthropic) under Nathan Moinvaziri's direction. The reviewer, a helper session, had not worked
on Theorem B's certificates or read P1's transcripts. It read P2_review.md and P2b_review.md first. Not reviewed.
Status words as in [AGENTS.md](../AGENTS.md).

## 1. The six items

Sizes are line counts of the certificate scripts at main b081fbb2. Costs are estimates.

| # | Item | What it certifies, and where | The link that rests on it | Code | What an independent check needs | Cost |
|---|---|---|---|---|---|---|
| 1 | Lemma J's θ₀ | The a priori step of Lemma J. On (1, 1 + δ/2], with m = 1, the affine map with the endpoint value v_S(1 + δ/2) contracts (θ₀ = 5.8×10⁻⁵), so Ψ_out is a Neumann sum there and N(Ψ_out) < ∞. `b8m_lcj.py` (one block near its end), B_chain.md §14, last paragraph | (H-J), and through it Theorem B(1) (B_chain.md §15). Without it, T₀'s contraction does not reach Ψ_out | about 35 lines, on top of Lemma J's secant coefficients | the Γ⁰ entries re-derived from the statement, evaluated on secant coefficients that share no code with `b8m_lcj.py`. P2b's are such coefficients (P2b_review.md §4) | hours (done here, §2) |
| 2 | Lemma 6b₁'s discs, M = 1 | y⁺ analytic on the annular sector 0.0624 ≤ \|ξ\| ≤ 0.2, by 178 validated Taylor discs and their gluing. `lemma6b1_disc.py`, `lemma6b1_cover.py`, `lemma6b1_estimate.py`, A_certify.md §12 | Theorem A(M = 1): lemma 6b, that the continued y⁺ is the Borel sum. Everything in Theorem B(1) starts from Theorem A's box | 215 + 478 + 140 | (a) the covering's geometry and gluing in exact rationals, which is cheap; (b) a second disc certificate in mpmath intervals (Taylor coefficients and a remainder bound for the ODE on each disc), for every disc or a stated sample. A rerun on Windows (#889) reproduced the rows, but a rerun is not an independent check | (a) hours, (b) 2–4 days |
| 3 | (H8) | B8's continuation turned into B5(1)'s data on the inner arc \|x\| = n^γ, δ = (8.2×10⁻⁵, 79.03, 3.231, 6.463), and 122.31 for δ_Φ at d₀ = 10⁷. `b8m_h8.py`, B8_constants.md §8.1–8.3 | B5(1), so B5(2)'s C_E and B7, for n ≥ e^{73.89} (C_selection.md §13.1) | 339 | a reread of §8.2's three pieces by hand, then the conversion in mpmath intervals from B8's printed h. The input h is item 4's, so the check would share it; that is recorded, not avoided | 1–2 days |
| 4 | B8 beyond `--d0` | Lemma B8(1): the bootstrap P(h) < h on 𝒟_n (K = 4, κ = 0.3, κ₁ = 0.6; §7.4 item 1), closing for ln n₀ ≥ 62 and with the normalization offset for ln n₀ ≥ 73.89. `b8_alg.py`, `b8m_bounds.py`, `b8m_scan.py`, B8_constants.md §§3–7 | B8(1), (H8), B5(1) and the gauge range of Lemma LC (C_selection.md §13.1) | 227 + 402 + 84 | a second bootstrap in other arithmetic. P2b's LC majorant covers B8's estimates on the same bands in the light-cone gauge (P2b_review.md §8), and P2 reproduced the formal series and atoms (P2_review.md §5). What remains is B8's own closure with c(0) = c^K(0) + d. P2b's bound algebra could serve, with the sharing recorded; **done** (§7, #1061): an independent second bootstrap closes it | 3–5 days |
| 5 | B5(1) | The inner fixed point W = O(1/n) on the matching domain (κ = 0.3, Z = 0.0085), with the super-solutions x_Φ = (39784, 38776) that enter C_E. `b5_bounds.py`, `b5_formulas.py`, B5_inner.md §§3–5 | B5(1), and so B5(2) and B7. x_Φ is a factor of C_E, the dominant constant of n₀ (notes/P7_scope.md §4.1) | 692 + 288 | a reread of B5_inner.md §4's proof and a second super-solution certificate (b + Lx < x) in mpmath intervals. Its input is lemma S's enclosure over 𝒦 (A_certify.md §4.5), which would be shared | 4–7 days |
| 6 | The M = 3 items | Lemma S(3) and lemma 6b₀(3), which share the majorant code of `lemma6_sector_M.py` and the helpers of `fixed_point.py`, and the 183 discs of lemma 6b₁(3), in `lemma6b1_disc_M.py`, `lemma6b1_cover_M.py` and `lemma6b1_estimate_M.py` (A_lemmas_M.md §§3, 5). A_M_review.md checked the gluing, not the discs | Theorem A(M = 3). It is not on Theorem B(1)'s path. It anchors the truncation sequence that P6 uses | 189 + 520 (helpers), 141 + 598 + 207 | a second majorant implementation for lemma S (the start-value lemma's second implementation, #427, does not cover it), and the disc work of item 2 at M = 3 | 3–5 days |

**Order, by cost against the weight of the link.**
1. Item 1. It is the cheapest, and it is on Theorem B(1)'s path (done in §2).
2. Item 2(a), the covering's geometry. It costs hours and sits under the whole chain. Then item 2(b).
3. Item 3, (H8). It is moderate, and it sets the n ≥ e^{73.89} links.
4. Item 5, B5(1). It is the largest, and it carries the dominant constant.
5. Item 4, B8. It is large, and P2b's light-cone majorant already covers part of it.
6. Item 6. It is large and off Theorem B(1)'s path.

## 2. Item 1: Lemma J's θ₀, checked independently

### 2.1 Protocol, fixed before the run

- **Independence.** `b8m_lcj.py` is not opened before the numbers are committed. The Γ⁰ entries are derived here from
  B_chain.md §14's equations. The secant coefficients come from P2b's code (`review_p2b/p2b_lc.py` and
  `p2b_lcj.py`), which shares no code with the certificate (P2b_review.md §1). They are evaluated on P2b's L cell
  (1, 1.025] with every shape set to 1. One fact about the certificate was seen in a search before this protocol
  was written: a comment of `b8m_lcj.py` names the a priori norm "m = 1 on (1, 1 + delta/2], s_V = s_c = 1".
- **The derivation** (the docstring of `solver/proof/review_p2c/p2c_theta0.py`). Plain sup norms on I = (1, 1 + δ/2],
  with m = 1 and unit shapes.
  - S, backward, with B_chain.md §14's kernel bound: Γ⁰_{S,j} = (1 + h_H)/(1 − h_H) sup_I ℓ_{G,j}.
  - V: Γ⁰_{V,S} = |ω| ln(1 + δ/2).
  - c: Γ⁰_{c,j} = ln(1 + δ/2) sup_I ℓ_{F_c,j}.
  - β: Γ⁰_{β,j} = sup_I ℓ_{ĝ,j}/(n − 1).
  - The pseudo-component of δD/(x − 1) is added to ℓ_G and ℓ_ĝ, as P2b adds it.
- **Pass criteria.** θ₀ < 1 on the certificate's class constants (`--h=cert`) and on P2b's own
  (`--h=p2b_lc_cells.json`). The certificate's θ₀ = 5.8×10⁻⁵ is compared, not required. P2_review.md §3.3's
  θ₀ ≤ 1.8×10⁻⁵, from the saved Γ₀, is compared too. A difference larger than a factor 10 is located after the commit,
  by reading the certificate's block.
- **Controls.** `--planted`: the Perron certification on a rank-one matrix of known root, and the Γ⁰ assembly against
  hand values, must pass. `--planted --mutate=perron` must fail. `--mutate=nodecay`, the backward kernel times n₀
  (the certificate's own mutation in B_chain.md §14), must give θ₀ > 1 and exit 1.

### 2.2 Results

**θ₀ < 1 is confirmed. The certificate's Γ⁰ agrees with this one entry by entry, up to the differences explained
below, and its Perron root is 1.08×10⁻¹⁶ against 1.03×10⁻¹⁶ here. The printed θ₀ = 5.8×10⁻⁵ is a valid upper bound,
about 10¹¹ above the root.** The own numbers were committed (59591319) before `b8m_lcj.py` was opened. Logs are in
`solver/proof/review_p2c/`, run with `/Users/nathan/venv-314/bin/python` (mpmath 1.3.0), in about 3 s each.

| run | Perron root of Γ⁰ | certified θ₀ | exit |
|---|---|---|---|
| `--h=cert` (the certificate's class) | 1.029×10⁻¹⁶ | 1.000×10⁻¹² | 0 |
| `--h=p2b_lc_cells.json` (P2b's own class) | 1.029×10⁻¹⁶ | 1.000×10⁻¹² | 0 |
| `--h=cert --rate=half` (the weighted form's kernel factor) | 1.198×10⁻¹⁶ | 1.000×10⁻¹² | 0 |
| `--planted` | 0.2070 (planted 0.2070) | 0.2104 | 0 |
| `--planted --mutate=perron` | 2.054 | 2.069 | 1 (fails as it must) |
| `--h=cert --mutate=nodecay` (pre-set control) | 0.553 | 0.564 | 0 (**did not fire**) |
| `--h=cert --mutate=gaininv` (added after the run) | 9.08×10¹⁵ | 9.27×10¹⁵ | 1 (fails as it must) |

The certified θ₀ sits at 10⁻¹² because the certification's weight search adds 10⁻¹² to its trial root.

- **On L every loop of Γ⁰ is tiny.** The O(1) entries are Γ⁰_{V,S} = |ω| ln(1.025) = 0.0733 and Γ⁰_{c,β} = 0.0247,
  since ℓ_{F_c,β} = 1. Every entry that closes a loop with them is O(ρ) or O(1/n), and ρ = (1 + |x|)²/n ≈ 4/n on L.
  So θ₀ is about the square root of the largest two-cycle, Γ⁰_{c,β}Γ⁰_{β,c} = 0.0247 × 2.2×10⁻³¹.
- **The pre-set control did not fire, and that is reported, not hidden.** The backward kernel times n₀ multiplies the
  S row's O(ρ) coefficients by n₀ and leaves them O(1). The largest loop, S → V → S, then reaches 0.31, and θ₀ = 0.564.
  The certificate's own mutation fires in the weighted norm of the main contraction, where the S row is not O(ρ).
  Here it is not a test. The control added after the run, with that reason recorded in the script's docstring,
  inverts the β row's gain at n = n₀. It is the factor (n₀ − 1)² that a multiplication in place of a division would
  make, and it fails at θ₀ = 9.3×10¹⁵. The inversion for every n ≥ n₀ is unbounded, and P2b's algebra refuses it
  with an error.

### 2.3 The comparison with the certificate

`b8m_lcj.py`'s block (the comment "the a priori norm (m = 1 on (1, 1 + delta/2], s_V = s_c = 1)" and the lines after
it) and its saved Γ₀ (`b8m_lcj.json`) were read after the commit. The certificate's Perron root was recomputed with
this review's certification (`p2c_cert_gamma0.py`, `p2c_cert_gamma0.log`).

| entry | certificate | here | the difference |
|---|---|---|---|
| Γ⁰_{V,S} | 0.07424 | 0.07333 | the certificate integrates dy/y over (1, 1 + δ/2] as δ/2, here as ln(1 + δ/2) |
| Γ⁰_{S,V}, Γ⁰_{S,c} | 4.21×10⁻³², 6.58×10⁻³¹ | 3.39×10⁻³², 5.30×10⁻³¹ | the ratio, 1.242, is the kernel factor (next row). With it removed, the secant coefficients agree to four digits (4.2110×10⁻³² and 6.5832×10⁻³¹) |
| the β row and column | Γ⁰_{β,V} = 5.34, Γ⁰_{β,c} = 27.2; Γ⁰_{S,β} = 3.7×10⁻⁶³ | 4.3×10⁻³², 2.2×10⁻³¹; 3.6×10⁻³¹ | the certificate's shape is s_β = 1/n, here 1: a diagonal scaling, which leaves the Perron root unchanged |
| Γ⁰_{c,S} | 4.24×10⁻³³ | 5.17×10⁻³⁵ | the certificate's c row carries F_c's pseudo-component (fcpL × cpj), which P2b's F_c node does not have (§2.4) |
| the kernel factor | (1 + h_H)/(1 − h_H) = 1.242, with h_H = 0.108 over the whole segment | the same form, with h_H on L, 9.7×10⁻³², so 1.000 | both use the full decay rate, which m = 1 allows. The certificate's h_H is conservative on L |
| Perron root | 1.08×10⁻¹⁶ | 1.03×10⁻¹⁶ | agreement to 5% |
| printed θ₀ | 5.84×10⁻⁵ | 1.0×10⁻¹² | the certificate's power iteration floors each weight at 10⁻¹², and the tight weight needs a ratio of about 4×10²³ (`p2c_cert_gamma0.log`). Its θ₀ is a valid bound, only loose. P2_review.md §3.3's 1.8×10⁻⁵ is a bound of the same kind |

### 2.4 Discrepancies (reported, not fixed)

1. **F_c's pseudo-component on L.** The certificate's c row adds fcpL × cpj (`b8m_lcj.py`, the Gam0["c"] line): the
   derivative of δF_c with respect to sup_{[1,x]}|v_c′|. P2b's F_c node has no such entry (its pseudo-component
   derivative is 0, `p2b_lc.cell_eval`), so P2b's Lemma J and this check omit a term that the certificate includes.
   Which form is right needs the bounded form of F_c on L, which B_chain.md §13.3 does not write out. The term cannot
   change θ₀'s verdict, since the c row's loops are O(ρ). Whether it changes P2b's main θ (0.164, against the
   certificate's 0.176) was not checked.
2. **The printed θ₀ is loose by a factor of about 10¹¹.** That is harmless, because the lemma needs only θ₀ < 1.
   B_chain.md §14 quotes 5.8×10⁻⁵, and P2_review.md §3.3 quotes ≤ 1.8×10⁻⁵. A proposed wording for B_chain.md §14:
   "(θ₀ ≤ 6×10⁻⁵, a loose bound; the Perron root of Γ⁰ is about 10⁻¹⁶, #988)".
3. **This review's pre-set real-data control did not fire** (§2.2). That is a defect of the control, not of the
   certificate.

### 2.5 Reproduce

From `solver/proof/review_p2c/`, with python and mpmath (P2b's modules are imported from `../review_p2b/`):

```
python3 p2c_theta0.py --planted                       # PASS, exit 0
python3 p2c_theta0.py --planted --mutate=perron       # FAIL, exit 1
python3 p2c_theta0.py --h=cert --log=p2c_theta0_cert.log              # theta0 1.0e-12, exit 0
python3 p2c_theta0.py --h=p2b_lc_cells.json --log=p2c_theta0_own.log  # theta0 1.0e-12, exit 0
python3 p2c_theta0.py --h=cert --mutate=gaininv       # theta0 9.3e15, exit 1
python3 p2c_theta0.py --h=cert --mutate=nodecay       # theta0 0.564, exit 0 (the pre-set control; it did not fire)
python3 p2c_cert_gamma0.py                            # the certificate's Gamma0: Perron root 1.08e-16
```

## 3. Item 2(a): lemma 6b₁'s M = 1 covering, its geometry checked independently

Issue #991. Lemma 6b₁ (A_certify.md §12.3–12.4) glues 178 discs of lemma D into an analytic continuation of y⁺ over
A₁ = {25 ≤ |u| ≤ 260, −3π/4 ≤ arg u ≤ 0}. This section checks the geometry of that covering. The discs' own
certificates are item 2(b).

### 3.1 Protocol, fixed before the run

- **Independence.** `lemma6b1_cover.py` and the certificate's check output `lemma6b1_check_1e-9.json` are not opened
  before the numbers are committed. Only the top-level key names of that output were seen while finding the data. The
  inputs are the stored runs `lemma6b1_runs/row_00.json` to `row_11.json`. They give per row ρ_j (the decimal the run
  used), r_j and n_j, and per disc t_jk and its certificate flag. The definitions of the tiles, the gluing order and
  the agreement points are A_certify.md §12.3's statements: a_j = ρ_j − 0.6r_j, b_j = ρ_j + 0.6r_j,
  α_j = 0.6 · (3π/4)/K_j, φ_jk = −(3π/4)(1 − t_jk) and G₀ = {|u| > 257, −3π/4 − 0.4 < arg u < 0}. Here a_j, b_j and
  α_j are taken as exact rationals and π-multiples, not as the certificate's binary floats. A lemma G proof may
  choose its tiles inside the discs, so passing with these tiles proves the gluing on its own.
- **Arithmetic.** mpmath intervals (`iv`, 200 bits) for every quantity involving π, sin or cos, and exact rationals
  elsewhere. A comparison counts only when it is proven. A strict inequality needs the upper end of one side below the
  lower end of the other.
- **What is checked** (`solver/proof/review_p2c/p2c_cover.py`).
  - G1. The data: 12 rows, 178 discs, every disc flagged certified, t_j0 = 0 and t_jK = 1, and r_j < ρ_j.
  - G2. Each closed tile lies in its open disc. The distance from the centre is largest at a corner of the closed
    polar rectangle, so the four corners are checked.
  - G3. The open tiles cover the closed A₁. The bands cover [25, 260] with strict overlaps, and in every row the
    angular intervals cover [−3π/4, 0] with strict overlaps.
  - G4. A second, tile-free route: A₁ is split into closed polar cells, and each cell is proven to lie in one open
    disc. Cells are bisected until that holds, with a cap of 10⁵ cells.
  - G5. The gluing, tile by tile in §12.3's order. Each meeting set M_m (T_m against G₀ and every earlier tile) is
    built from the pairwise intersections not proven empty. M_m must be connected through proven-nonempty overlaps,
    which is undecided if a needed overlap is not proven. The agreement point q_m must lie in T_m and in the partner
    piece. The partner is G₀ for (0, 0), tile (j, k − 1) for k ≥ 1 with q at mid-angle on the arc, and tile
    (j − 1, 0) for k = 0 with q on the ray at radius (a_{j−1} + b_j)/2. The curve along which the two functions agree
    must lie in both discs: for k ≥ 1 the half-arcs, 2ρ_j sin(Δφ/4) < r_j; for k = 0 the ray segments,
    ρ_{j−1} − |q| < r_{j−1} and |q| − ρ_j < r_j.
  - G6. Reachability: every tile's partner comes earlier in the order, so every tile is reached from (0, 0) on the
    start ray, and s₀ = 262e^{−3iπ/4} lies in G₀.
- **Pass criteria.** G1 to G6 all hold. The counts of pieces and links are reported and compared afterwards with
  A_certify.md §12.4's 566 and 450, which are not required to match, since the meeting sets are counted here in a
  different way.
- **Planted failures, which must each fail.** (P1) Disc (5, 6) is removed. (P2) Row 3's α is set to 0.49 Δφ, so
  consecutive tiles of that row no longer overlap and the links between them break. Both run through the same code.

### 3.2 Results

**The geometry is confirmed. The open tiles cover A₁, each closed tile lies in its open disc, A₁ is also covered by
the discs directly, and every link of lemma G holds. Both planted coverings fail.** The own numbers were committed
(148a63c4) before `lemma6b1_cover.py` and `lemma6b1_check_1e-9.json` were opened. Logs and JSON are in
`solver/proof/review_p2c/` (`p2c_cover_real.log`, `_drop.log`, `_gap.log`), run with
`/Users/nathan/venv-314/bin/python` (mpmath 1.3.0) in 1 to 11 s each.

| check | real covering | P1: disc (5, 6) removed | P2: row 3's α = 0.49 Δφ |
|---|---|---|---|
| G1 data | ok: 12 rows, 178 discs, all certified | ok (177 tiles) | ok |
| G2 tiles in discs | ok, smallest margin r² − d² = 21.4 | ok | ok |
| G3 tiles cover A₁ | ok | **FAIL**: row 5's tiles 5 and 7 do not overlap | **FAIL**: row 3's consecutive tiles do not overlap |
| G4 cells in discs | ok: 8930 cells, none bisected | **FAIL**: a sliver near \|u\| = 133, arg u = −1.162 lies in no disc | ok (the discs are unchanged) |
| G5 gluing | ok: 566 pieces, 450 proven links (388 in spanning trees) | **FAIL**: tile (5, 7) has no partner; the meeting set of tile (6, 5) is not shown connected | **FAIL**, 32 failures: tiles (3, 1) to (3, 17) have no piece with their partner, and 15 tiles of row 4 have meeting sets not shown connected |
| G6 reach, s₀ ∈ G₀ | ok | ok | ok |
| exit | 0 | 1 | 1 |

- The tiles here are exact (a_j = ρ_j − 3r_j/5, α_j = (3/5)(3π/4)/K_j). Lemma G may use any tiles inside the discs,
  so these checks prove the gluing by themselves. Their agreement with the certificate's float tiles is a cross-check.
- G4 is the tile-free route. It covers A₁ with closed polar cells of 2.5 in |u| and 3π/380 in angle, and each is
  proven inside one open disc with no bisection needed. Under P1 it finds the uncovered sliver where disc (5, 6) was,
  which shows that the discs, and not only the tiles, would then fail to cover.
- P1 fails in the same three places the certificate's own control `check --drop=5,6` reports (A_certify.md §12.5):
  tile (5, 7) has no agreement point, tile (6, 5)'s meeting set is disconnected, and (5, 6)'s direction is not
  covered.

### 3.3 The comparison with the certificate

The certificate's check output (`lemma6b1_check_1e-9.json`) reports ok, no failures, 178 tiles, 566 pieces and 450
links. These are the same counts as here, from code that shares nothing with it but the statements of A_certify.md
§12.3 and the stored runs. Its tiles are the same formulas in binary floats (α_j = 0.6 Δφ_j, a_j = ρ_j − 0.6 r_j,
`lemma6b1_cover.py`), which differ from the exact ones here in the last bits. No discrepancy was found.

One wording point, not a discrepancy: PROOF.md §4 and issue #991 describe the annular sector as
0.0624 ≤ |ξ| ≤ 0.2. A₁ = {25 ≤ |u| ≤ 260} is 0.0620 ≤ |ξ| ≤ 0.2, which contains it. G₀ = {|u| > 257} covers the rest
down to |ξ| = 0.06238.

### 3.4 Reproduce

From `solver/proof/review_p2c/`, with python and mpmath:

```
python3 p2c_cover.py --log=p2c_cover_real.log        # PASS, exit 0 (about 2 s)
python3 p2c_cover.py --plant=drop --log=p2c_cover_drop.log   # FAIL, exit 1 (about 11 s)
python3 p2c_cover.py --plant=gap --log=p2c_cover_gap.log     # FAIL, exit 1
```

## 4. Item 2(b): lemma 6b₁'s M = 1 disc certificates, a second implementation

Issue #992. Lemma D (A_certify.md §12.2) certifies, for each of the 178 discs, that y⁺ continues analytically from the
disc's centre to the whole disc, with a bound M on |y − y₀|. This section re-implements that certificate.

### 4.1 Protocol, fixed before the run

- **Independence.** `lemma6b1_disc.py` and `lemma6b1_estimate.py` are not opened. The code
  (`solver/proof/review_p2c/p2c_disc.py`) is written from A_certify.md §12.2's statement of lemma D and from
  §4.4's equations in the u = 1/z variables of §11.4. The only shared things are the inputs: the stored runs, the
  equation's form and coefficients, κ = 100, the order n_j and the radius r_j. The equation is compared with
  `lemma6_usys.SysU.rhs` at random points, the only use of the certificate's code. Seen before this protocol: the
  stored Λ and sup bounds of discs (0, 0) to (0, 4) and (11, 0) to (11, 2) (the first lines of `rows.log` and
  `row_11.log`, and row_00's first record), and A_certify.md §12.4's per-row maxima.
- **Method, own.** The arithmetic is python-flint 0.9 balls (acb) at 192 bits, in code written here.
  - Taylor coefficients: Picard iteration on truncated power series (acb_series), started from the input box, n + 1
    times.
  - Defect: in the form with denominators cleared, d̃ = u^m (p′ − F(u, p)) with m = 1 for P and V and m = 2 for C. It
    is an exact polynomial in τ (acb_poly products), so no expansion of 1/u with a remainder is needed. Its coefficients
    below order n must contain 0, and |d(τ)| ≤ |τ|ⁿ Δ̃/(|c| − r)^m with Δ̃ = Σ_{k≥n} |d̃_k| r^{k−n}.
  - Lipschitz matrix: forward-mode derivatives (series of length 2) of F over the box B, the z-ball 1/(c + disc(r))
    and the ω-ball.
  - Check: lemma D's contraction with Perron weights, Λ = r/(n + 2) · max_i Σ_j ω_i L_ij/ω_j < 1,
    ρ = 2F₀/(1 − Λ), and b_j ≥ ρ r^{n+1}/ω_j, iterated with B = Y₀ + disc(M⁰ + b) until it holds. M_j = M⁰_j + b_j.
- **Inputs.**
  - The 12 discs on the start ray (k = 0) use the stored row-start boxes (`lemma6b1_runs/ray.json`, `starts`), exactly.
    They include the corners (0, 0) and (11, 0) and the disc with the largest stored Λ, (11, 0).
  - The 166 arc discs have no stored box. Their boxes are reconstructed by this review: its own high-precision,
    non-rigorous integration along the arc from the stored row-start box's midpoint, inflated to 4 times the stored
    box radius plus 10⁻¹⁵. A certificate there covers lemma D's computation on that box. It covers the certificate's
    box only if that box lies inside, which is assumed, not proven. The integrator is first checked against the stored
    ray boxes: from s₀ along the ray to s₁, its value must lie within the stored box at s₁.
- **Sample, then all.** First the 12 ray discs, all of row 11 (the smallest margins), and the real-axis corner discs
  (0, 24) and (11, 6). Then all 178 if a disc costs under 30 s.
- **Pass criteria.** Every disc certified (Λ < 1, the vanishing orders hold, b closes). The margins (Λ and max_j M_j)
  are recorded and committed before the stored ones are read in full, then compared.
- **Controls, which must each fail.**
  - (K1) A too-large disc: disc (11, 0)'s stored input with radius 16. The branch point u* = 9.6746e^{−2.1198i}
    (A_certify.md §12.1) lies at distance about 15.7 from its centre, so it is inside.
  - (K2) The sign mutation: the defect computed with the fast term's sign flipped (V_u = +V/2 + …), while the Taylor
    coefficients use the true equation, on the 12 ray discs. Every certificate must fail.
- **Planted checks, which must pass.** The equation against `SysU.rhs` (relative 10⁻⁴⁰ at 10 random points), and the
  integrator against the stored ray box at s₁.

### 4.2 Results

The own numbers were committed (ac394431) before the stored per-disc margins were read for the comparison (§4.3).
The logs and records named below are in the raw-data repository's folder `I992-mac/` (ccf0c978).
The work paused there at the coordinator's word on 2026-11-19 and resumed the same day.

- **Planted checks pass.** The equation matches `SysU.rhs` at 10 random points to a relative 3.5×10⁻⁵⁸ (`--checkeq`).
  The integrator lands inside each stored ray box from s_j to s_{j+1}, at deviations of 9.6×10⁻²⁴ to 1.3×10⁻²⁰
  against box radii of 2×10⁻⁹ to 8×10⁻⁸ (`p2c_disc_checkint.log`). The first run of this check, before the series-cap
  fix below, had deviations up to 1.7×10⁻⁹ and passed too (`p2c_disc_checkint_devel.log`).
- **All 178 discs, protocol inputs: 177 certified** (`p2c_disc_all.log`, 511 s).
  - All 12 start-ray discs, with exact stored inputs, are certified. Their Λ are 0.1699 for (0, 0) and 0.4397 for
    (11, 0).
  - **Disc (11, 1) is not certified (Λ = 1.494)**, on its reconstructed box (4 × the stored radius 1.0×10⁻⁶).
- **Located.** It is input size, not a certificate error.
  - The reconstructed input at (6, 7) is y⁺: three integration paths agree to 10⁻²².
  - This implementation's Picard iteration on ball series amplifies the input box's radius more than the certificate
    does. On (6, 7) a box of 10⁻¹² gives max M⁰ = 0.88, and the protocol's 4 × box radius gives 5.08.
  - Rerunning (11, 1) at 0, 1, 2 and 3 times the stored radius certifies it with Λ = 0.393, 0.453, 0.541 and 0.674
    (`p2c_disc_diag_11_1.log`, outside the protocol).
- **Controls fail as they must.** K1, disc (11, 0) at radius 16 with u* inside, gives Λ = 4.7×10⁸. K2, the sign
  mutation, fails all 12 ray discs at order 0 of the V defect.
- **Development fixes**, found on discs (0, 0) and (11, 0) before the full run, whose stored Λ and sup had been seen
  (§4.1):
  - python-flint's series cap, 10 by default, which truncated the Taylor series;
  - the weight orientation, ω_i = 1/v_i;
  - the z-ball, which now uses lemma D's own enclosure 1/c + disc(r/(|c|(|c| − r))). python-flint's inverse of the
    box was 8 times looser.
- **Disc (11, 1)** is finished in §4.4, and the comparison is in §4.3.

### 4.3 The comparison with the certificate

`p2c_disc_compare.py` reads the stored per-disc Λ, sup (the bound M) and box radius from `lemma6b1_runs/row_XX.json`
and sets them beside this review's (`p2c_disc_compare.log`).

- **Same inputs, same margins.** On the 12 start-ray discs the input is the stored box itself. Λ agrees to at most
  5.6×10⁻⁶ relative and M to at most 3.1×10⁻¹¹ on rows 0 to 10. On (11, 0) both are larger here by 1.6×10⁻³ (Λ 0.43968
  against 0.43897, M 15.489 against 15.469). That is the input-size effect of §4.2 on the stored box of 7.9×10⁻⁸
  (attributed, not proven).
- **Arc discs.** Their inputs here are larger than the certificate's: 4 times (protocol) or 2 times (§4.4) the stored
  radius, centred on this review's midpoint. Λ agrees to 0.2% on rows 0 to 10 at both sizes. M agrees to 0.3% on
  rows 0 to 3 and 7 to 10. On rows 4 to 6 and 11 it is larger, and more so at 4 times than at 2 times (row 4 agrees
  at 2 times):

| row | max Λ, stored | here, 4× | here, 2× | max M, stored | here, 4× | here, 2× |
|---|---|---|---|---|---|---|
| 4 | 0.1793 | 0.1793 | 0.1793 | 0.4023 | 0.4347 | 0.4023 |
| 5 | 0.1833 | 0.1833 | 0.1833 | 0.6173 | 1.19 | 0.6372 |
| 6 | 0.1909 | 0.1910 | 0.1909 | 1.067 | 5.081 | 2.614 |
| 11 | 0.4390 | 0.6482 (6 of 7) | 0.5410 | 15.47 | 84.45 (6 of 7) | 43.01 |

- **The stored files against A_certify.md §12.4.** The row maxima of Λ, box radius and sup in the stored files
  round to the table's entries (for example row 11: 0.4390, 1.77×10⁻⁵, 15.47 against 0.439, 1.8×10⁻⁵, 15.5).

### 4.4 Disc (11, 1)

- **At the protocol's input** (4 × its stored box radius 1.024×10⁻⁶, plus 10⁻¹⁵) it is not certified: Λ = 1.494.
- **Cause: the input box.** The bound M grows by 9.85×10⁶ per unit of input radius, between 0 and 1 times the stored
  radius (`p2c_disc_11_1_growth.log`). The growth allowed by the Lipschitz bound over the disc,
  exp((n + 2)Λ) at the thin input, is 1.49×10⁷.
  - So the growth comes from the Lipschitz-based bound, not from the ball series alone.
  - A centred (Gronwall) form would carry the same factor. On the 4.1×10⁻⁶ box it would allow a drift of about 60,
    so it was not pursued.
- **The input that is needed.** The certificate's box at (11, 1) contains y⁺(c) and has radius at most 1.024×10⁻⁶
  (A_certify.md §12.4: the Lohner box at the centre, the input of lemma D).
  - It therefore lies inside this review's midpoint ± (2 × 1.024×10⁻⁶ + |midpoint − y⁺(c)|) on each axis.
  - The midpoint agrees with y⁺(c) to 10⁻²² by three paths. That is the protocol's non-rigorous assumption, and its
    10⁻¹⁵ covers it.
  - So 2 times the stored radius plus 10⁻¹⁵ contains the certificate's box. The 4 times of §4.1 was a margin chosen
    beforehand, not a need.
- **At 2 times, all 178 discs are certified** (`p2c_disc_all_infl2.log`, 212 s). Disc (11, 1) has Λ = 0.541 and max
  M = 43.0, against the stored 0.418 and 14.6.

**Status.** The protocol's inputs certify 177 of 178 discs. The smallest inputs that contain the certificate's boxes,
under the same assumption, certify 178 of 178, and both controls fail.
- **The 12 start-ray discs** are confirmed outright, on the stored inputs.
- **The 166 arc discs:** the review covers lemma D's computation on boxes assumed to contain the certificate's. That
  assumption, the midpoint's accuracy, is not proven (§4.1).
- **Versions:** python-flint 0.9.0 and FLINT 3.6.0.

### 4.5 Reproduce

From `solver/proof/review_p2c/`, with python-flint 0.9.0, and the raw-data repository cloned beside this checkout
(`git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw` from the checkout's root). The
records go to a task folder there, given by `--out`; `I992-mac/` holds this review's.

```
RAW=../../../../choptuik-large-d-raw/I992-mac
python3 p2c_disc.py --checkeq                                  # PASS, exit 0
python3 p2c_disc.py --checkint > $RAW/p2c_disc_checkint.log    # PASS, exit 0
python3 p2c_disc.py --discs=all --out=$RAW > $RAW/p2c_disc_all.log    # 177 of 178, exit 1 (about 510 s)
python3 p2c_disc.py --discs=all --infl=2 --tag=all_infl2 --out=$RAW > $RAW/p2c_disc_all_infl2.log   # 178 of 178, exit 0
python3 p2c_disc.py --discs=11:1 --infl=0 --tag=diag_11_1_infl0 --out=$RAW   # and --infl=1, 2, 3 (tags ..._infl1, 2, 3)
python3 p2c_disc.py --discs=11:0 --radius=16 --out=$RAW > $RAW/p2c_disc_K1_toolarge.log   # K1: not certified, exit 1
python3 p2c_disc.py --discs=ray --mutate=sign --out=$RAW > $RAW/p2c_disc_K2_sign.log      # K2: 0 of 12, exit 1
python3 p2c_disc_compare.py $RAW/p2c_disc_all.jsonl $RAW/p2c_disc_all_infl2.jsonl > $RAW/p2c_disc_compare.log
python3 p2c_disc_11_1_growth.py $RAW > $RAW/p2c_disc_11_1_growth.log
```

## 5. Item 3: (H8), B8's continuation turned into B5(1)'s data, checked independently

Issue #1017. (H8)(ii) (B8_constants.md §8.2–8.3) bounds W = Y⁺_out − Z⁺_in on the inner arc |x| = n^γ in B5's
variables, |W_P| ≤ δ_P/n, |W_Φ| ≤ δ_Φn^{−2+2γ}, |W_C| ≤ δ_C/n, |W_ϑ| ≤ δ_ϑ/n, for every n ≥ e^{73.89} and every
parameter in the box. B8 gives δ = (8.3×10⁻⁵, 79.04, 3.232, 6.464) (§8.3, rounded up), B_chain.md §4 the same as
(8.2×10⁻⁵, 79.03, 3.231, 6.463), and δ_Φ = 122.31 for the normalization offset d₀ = 10⁷. This section recomputes them.

### 5.1 Protocol, fixed before the run

- **Independence.** `b8m_h8.py` and its logs are not opened until this review's numbers are committed. Nor are
  `b8m_formal.py`, `b8m_atoms.py`, `b8m_bounds.py` and `b8_alg.py`. The code (`solver/proof/review_p2c/p2c_h8.py`) is
  written from B8_constants.md §1.2's reduced system, §2's statement of the series and §8.2's formulas.
  - **Shared inputs:** the certified h for d₀ = 0 and 10⁷ (the `hyp` entries of `b8m_scan_K4_box.log` and
    `b8m_scan_K4_d0_1e7.log`, item 4's output), lemma S's η bound (5.5×10⁻⁷ for v at |z| = 0.0085, scaling as |z|¹²,
    from `b5_bounds_k0.3_Z0.0085.log`), K = 4, γ = 0.3, ln n₀ = 73.89, ω∞ = 2π/1.64389375, and the parameter box
    (A, ω/ω∞) ∈ (2.5421480475, 0.7769940136) + 2×10⁻⁶(1 + i)[−1, 1]² (B8_constants.md §8.4, item 3).
  - **Seen before this protocol:** B8_constants.md §0, §1–8 (the h, the atom table of §3, the (R), (T), η table of §8.3
    with its leading coefficients) and B_chain.md §4's d₀ table.
- **Method, own.** Arithmetic is mpmath complex intervals (iv) at 256 bits, not python-flint.
  - **Series.** The reduced system is solved as a double series in (ε, z) with z = x²/n. In these variables it is
    regular in ε: the ε⁰ part is the inner system, and Z^K is the part with a + j ≤ K for monomials ε^a z^j.
    The gauge is c^K = 1 at x = 1, that is at z = ε, and V(0) = [c(0)2F] at every order.
  - **T.** Z^K is converted exactly to (P, Φ, C, ϑ), with C⁻¹ and 1/(1 − ε) as series with bounded tails. The ε⁰ part
    of the full series converted the same way is Z_in. The difference T − Z_in is bounded monomial by monomial on the
    arc, where |ε^a z^j| = n^{−a−0.4j}.
    - The structural zeros are imposed. These are the ε⁰ terms up to z^K, by Lemma F.3, and P's z⁰ terms at ε ≥ 1,
      since P(0) = 2F at every order.
    - The inner tail K < j ≤ 30 comes from the same solver. η is added as stated.
  - **R.** W^R is bounded from h on the arc: X = 1 + n^γ, ρ = X²/n, w₁ = ρ^{K+1}, w = θρ^{K+1}. P, C and ϑ follow by
    §8.2's formulas.
    - Φ needs xR'. xR_V' = ΩR_S and xR_c' = [R_cβ] + [c^KR_β] − ρ₄ are used as stated.
    - xR_S' is this review's own route, not Lemma R's G₁ and W₂ decomposition. It is the mean-value bound of
      D⁻¹(AS − x²BV) between Z^K and Z^K + R over the ball of R, plus Z^K's own residual xS^K' − D^{K,−1}(…), each
      bounded on the arc. R_β = R̂_β + qR_S takes q's bound from its definition in Lemma R.
  - **All n ≥ n₀.** Every term is n^{−e} times a constant with e ≥ 0, so it is evaluated at n₀. A term with e < 0
    stops the run.
- **Pass criteria.**
  - **(P1)** At the box centre, the leading coefficients agree with B8's printed values to all their digits:
    Φ_{±1} at εz, −15.714 ± 74.930i; C at ε, −3.2313 and −1.6156; ϑ at ε, −6.4625 and −3.2313. The inner series also
    agrees with Theorem A's `series_arb.py` (not B8's code) to 10⁻⁴⁰ relative, and the series' residual in the
    reduced system has the orders of Lemma F.2.
  - **(P2)** This review's δ, as rigorous upper bounds over the box for all n ≥ n₀, is at most
    δ_ref = (1, 200, 10, 20).
  - **(P3)** After the own numbers are committed, they are compared with `b8m_h8`'s logs.
    - The T parts of δ_Φ, δ_C and δ_ϑ must agree to 10⁻³ relative.
    - The R parts of P, C and ϑ, which use the same formulas, must agree to 10⁻² relative.
    - The R part of Φ comes by another route. It is reported against B8's 2.46, and a difference of more than a factor
      10 either way is investigated.
    - δ_P is expected to be far below B8's 8.3×10⁻⁵, because B8's value is the width of a ball whose centre is
      exactly 0.
  - **(P4)** At d₀ = 10⁷, with its h, δ_Φ is reported against 122.31, under (P3)'s rule for Φ.
- **Controls, which must each fail.**
  - **(K1)** The planted perturbation: A scaled by 1 + 10⁻³ in the series. (P1) must fail by more than its rounding.
  - **(K2)** h_S scaled by 1 + 10⁻³. (P3)'s comparison of the R part of P must flag it. This is valid only if the
    unperturbed agreement is better than 10⁻³, which is recorded.
  - **(K3)** The sign control: the series solved with the sign of the x²Ω term of A flipped, while the residual check
    uses the true system. The residual orders of (P1) must fail.
- **Output.** The logs and records go to the raw-data repository as `I1017-mac/`. The code, and this section, stay in
  the branch.

### 5.2 Results (own numbers, committed before the comparison)

The logs and records are in the raw-data repository's folder `I1017-mac/` (b5deb210). The code is `p2c_h8.py` at
1506f53f, run with mpmath 256-bit intervals.

- **(P1) passes.**
  - The series Z^K solves the reduced system on a + j ≤ 4, with residual coefficients at most 3.8×10⁻⁶⁵ at the box
    centre. The lowest nonzero residual grade is 5 for S, β and c.
  - The leading coefficients at the centre are Φ_{±1} = −15.71416905 ± 74.93015179i at εz (modulus 76.56019), C at ε
    −3.231258348 and −1.615629174 (that is −A²/2 and −A²/4), and ϑ at ε −6.462516695 and −3.231258348. They agree with
    B8_constants.md §8.2's printed digits. P's ε z⁰ coefficient is 0, and its εz coefficient is half of Φ's.
  - The inner series, c and P to z³⁰, agrees with `series_arb.py` to 1.4×10⁻⁶⁵ relative, with every ball overlapping.
    The bar was 10⁻⁴⁰.
  - Over the box, Z^K's ε⁰ part equals the inner series to z⁴, as Lemma F.3 states. At the centre the structural zeros
    removed are at most 4×10⁻⁶⁷.
- **(P2) passes**, for every n ≥ e^{73.89} over the box:

| | series (T) | remainder (R) | η | δ | δ_ref |
|---|---|---|---|---|---|
| P | 5.59×10⁻¹² | 7.75×10⁻²⁶ | 4.4×10⁻¹⁰⁴ | 5.59×10⁻¹² | 1 |
| Φ | 76.5792 | 2.463 | 3×10⁻⁸⁹ | 79.0419 | 200 |
| C | 3.23126 | 1.35×10⁻²⁷ | 7.6×10⁻¹¹⁵ | 3.23126 | 10 |
| ϑ | 6.4634 | 3.44×10⁻¹⁴ | 0 | 6.4634 | 20 |

  - **Φ's remainder** (2.463) is almost all the main term D⁻¹C²R_S (2.395). The series' residual r_S adds 0.068, from
    its ε⁰z⁵ term (τ²⁰); ‖q‖ ≤ 7.4×10⁻¹³.
  - **δ_P** is the εz term φ₁/2 · n₀^{−0.4}, since the ε z⁰ coefficient is 0.
  - **At the box centre** (`--point`): δ_Φ = 79.0226 (T 76.5602, R 2.462), δ_C = 3.23126 and δ_ϑ = 6.46252. So the
    box widens Φ's T by 0.019 here.
- **(P4).** At d₀ = 10⁷, with that run's h, R_Φ = 45.74 and δ_Φ = 122.32. The other δ are unchanged, and the R parts
  are R_P = 2.29×10⁻²⁴, R_C = 1.04×10⁻²⁵ and R_ϑ = 6.29×10⁻¹³.
- **Controls.**
  - **K1** (A × (1 + 10⁻³)) fails (P1) on B8's digits, with exit 1.
  - **K3** (the x²Ω sign flipped in the solver) fails the residual orders (5.8×10³ on a + j ≤ 4) and the comparison
    with `series_arb.py`, with exit 1.
  - **K2** (h_S × (1 + 10⁻³)) moves R_P from 7.747×10⁻²⁶ to 7.753×10⁻²⁶, that is by 7.7×10⁻⁴, and δ_Φ by 2.4×10⁻³. Its
    reading is in §5.3.
- **Development fix** (before these runs): the S update first subtracted G's (1 − ε)C²S term after adding it, so the
  interval widths doubled at every Picard step. Over the box Φ's leading ball was 87 wide. G's rest is now computed
  directly.
- **Harness fix:** a first comparison with `series_arb.py` built its inputs at 53 bits, before raising flint's
  precision, and so differed at 10⁻¹⁰. At 256 bits the difference is 1.4×10⁻⁶⁵.

### 5.3 The comparison with `b8m_h8.py`'s logs

`p2c_h8_compare.py` reads this review's records and B8's logs (`b8m_h8_K4_L73.89.log`, `b8m_h8_K4_L73.89_d0_1e7.log` and
`b8m_h8_centre.log`), opened only after §5.2 was committed (`I1017-mac/logs/compare.log`).

| | here | `b8m_h8` | relative | bar |
|---|---|---|---|---|
| T of Φ | 76.5792 | 76.5684 | 1.4×10⁻⁴ | 10⁻³, pass |
| T of C | 3.2312634 | 3.2312634 | 1.4×10⁻⁹ | 10⁻³, pass |
| T of ϑ | 6.4634016 | 6.4631590 | 3.8×10⁻⁵ | 10⁻³, pass |
| R of P | 7.7474365×10⁻²⁶ | 7.7474365×10⁻²⁶ | 2.6×10⁻¹⁰ | 10⁻², pass |
| R of C | 1.3538021×10⁻²⁷ | 1.3538021×10⁻²⁷ | 2.9×10⁻¹⁷ | 10⁻², pass |
| R of ϑ | 3.4352230×10⁻¹⁴ | 3.4352230×10⁻¹⁴ | 2.4×10⁻¹⁰ | 10⁻², pass |
| R of Φ (another route) | 2.46266 | 2.46251 | 6.2×10⁻⁵ | factor 10, pass |
| δ_Φ at d₀ = 10⁷ (P4) | 122.3203 | 122.3093 | 9.0×10⁻⁵ | reported |
| R of Φ at d₀ = 10⁷ | 45.74109 | 45.74093 | 3.3×10⁻⁶ | factor 10, pass |

- **At the box centre**, this review's T parts equal `b8m_h8_centre.log`'s in all six printed digits: P 5.58426×10⁻¹²,
  Φ 76.5602, C 3.23126 and ϑ 6.46252.
  - Over the box, Φ's T differs by 1.4×10⁻⁴. That is the width of the coefficient balls over the box, 0.019 here
    against 0.008 in B8.
- **δ_P.** B8's 8.2255×10⁻⁵ is the width of the ball of P's ε z⁰ coefficient, which is exactly 0 (P(0) = 2F). With
  that zero imposed, δ_P = 5.59×10⁻¹², the εz term. B8's bound is valid, only loose.
- **R of Φ** comes by this review's route, the mean value of the S-equation plus Z^K's residual, not Lemma R's G₁ and
  W₂. It agrees with B8's to 6.2×10⁻⁵. Here it is almost all the main term D⁻¹C²R_S, 2.395 of 2.463 (§5.2).
- **K2 is not flagged at the protocol's bar.** h_S × (1 + 10⁻³) moves R_P by 6.6×10⁻⁴ relative to B8's, against an
  unperturbed agreement of 2.6×10⁻¹⁰. The bar of 10⁻² set in §5.1 is too loose to catch it. Any bar between the two
  would catch it. This is a defect of the protocol, recorded rather than corrected after the fact: as run, K2 did not
  fire.

### 5.4 Discrepancies (reported, not fixed)

- **B_chain.md rounds δ down.** §3.2 and §4's table quote δ = (8.2×10⁻⁵, 79.03, 3.231, 6.463) for d₀ = 0. The certified
  bounds are (8.2255×10⁻⁵, 79.0309, 3.231263, 6.463159), so every entry is below its bound. Upper bounds are rounded
  up, as B8_constants.md §8.3 does: (8.3×10⁻⁵, 79.04, 3.232, 6.464).
  - x_Φ = (39784, 38776) is the superposition for these data (B_chain.md §3.2). If it was computed from the
    rounded-down values, it belongs to slightly smaller data, and needs a rerun with the rounded-up δ.
  - The d₀ = 10⁶ and 10⁷ entries (82.18 and 122.31) are rounded up correctly.
- **η of C.** B8_constants.md §8.2 states that lemma S's η scales as |z|¹³ for c. `b8m_h8`'s η for C, 4.4×10⁻¹⁰⁴,
  equals its η for P, which is |z|¹². That is conservative and immaterial: 7.6×10⁻¹¹⁵ here with |z|¹³.

**Status.** (H8)(ii)'s δ, for every n ≥ e^{73.89} over the box, is confirmed by an independent computation. It shares
with B8 only the certified h, lemma S's η bound, the parameter box and the hand-proved lemmas (Lemma F, the formulas of
§1.2 and §8.2). The two agree, and both lie within δ_ref. Its series is checked independently against Theorem A's
`series_arb.py`. Versions: mpmath (iv) at 256 bits, and python-flint 0.9.0 only for `series_arb.py`.

### 5.5 Reproduce

From `solver/proof/review_p2c/`, with the raw-data repository cloned beside this checkout (§4.5):

```
RAW=../../../../choptuik-large-d-raw/I1017-mac
python3 p2c_h8.py --centre                       # (P1) PASS, exit 0
python3 p2c_h8.py --centre --mutate=A            # K1: FAIL, exit 1
python3 p2c_h8.py --centre --mutate=sign         # K3: FAIL, exit 1
python3 p2c_h8.py --out=$RAW                     # (P2) PASS, exit 0 (about 50 s)
python3 p2c_h8.py --point --out=$RAW             # the same at the box centre
python3 p2c_h8.py --d0=1e7 --out=$RAW            # (P4)
python3 p2c_h8.py --mutate=hS --out=$RAW         # K2
python3 p2c_h8_compare.py $RAW > $RAW/logs/compare.log   # (P3, P4) PASS, exit 0
```

`$RAW/run.sh` runs the first seven with their logs.

## 6. Item 5: B5(1), the inner fixed point, checked independently

Issue #1019. Lemma B5(1) (B5_inner.md §§2–5) bounds W = Y⁺_out − Z⁺_in on the working domain by |W_j| ≤ x_j m_j/n, given
(H8). The bound comes from a super-solution b + Lx < x of a 10 × 10 system whose constants are suprema over the sector
Σ_Z = {0 < |t| ≤ Z, 0 ≤ arg t ≤ π/2 + κ} (κ = 0.3, Z = 0.0085). This section first settles whether B5's data cover
(H8)'s certified δ (§6.0), then recomputes the lemma's constants and super-solutions.

### 6.0 The δ question (answered first, `I1019-mac/delta/`, d80a66c2)

- `b5_threshold.py`'s DELTA0 = (0, 76.56, 3.23125, 6.46251) is B5's own leading O(1/n) coefficients of W at the box
  centre, from `b5_data.log`. That is |φ₁|, |[F²]| and 2|[F²]|, without B8's remainder, the box spread or η, so it is
  not (H8)'s δ. Its label "upper values" is wrong: each entry is below the exact coefficient (76.56019, 3.2312583,
  6.4625167, §5.2). It feeds only that script's superseded estimate and B5_inner.md §5.2's illustration.
- The chain's x_Φ and C_E come from `b5_jump.py`, which reads the certified δ from `b8m_h8_K4_L73.89.log` at full
  precision. Rerun as stored, it reproduces `b5_jump.log`: x_Φ = (39784.4, 38775.8), C_E = 2.532366×10³³. With
  B8_constants.md §8.3's rounded-up δ (8.3×10⁻⁵, 79.04, 3.232, 6.464) the result is x_Φ = (39789.7, 38781.0), C_E =
  2.532706×10³³. So the stored values stand.

### 6.1 Protocol, fixed before the run

- **Independence.** `b5_bounds.py`, `b5_formulas.py`, `b5_numeric.py`, `b5_system.py`, `b5_data.py` and their logs are
  not opened until this review's numbers are committed. The code (`solver/proof/review_p2c/p2c_b5.py`) is written from
  B5_inner.md §§1–4.
  - **Shared inputs:** κ, Z, κ₁ = 0.6, γ = 0.3, ln n₀ = 73.89, δ_ref = (1, 200, 10, 20), the parameter box (§5.1:
    centre ± 2×10⁻⁶ real, ± 10⁻⁶ imaginary), and lemma S's certified η bound at |z| = 0.0085 (B5_inner.md §5.1: p
    2.7×10⁻⁸, v 5.5×10⁻⁷, c 6.5×10⁻¹¹, weight exponent k = 12).
  - **Seen before:** B5_inner.md in full, `b5_threshold.py`, `b5_jump.py` and their logs (§6.0), and the stored
    super-solutions of §5.2.
- **Method, own.** mpmath complex intervals at 256 bits, not python-flint.
  - **Inner solution.** The inner series to z³⁰ comes from #1017's solver (ε⁰ part of B8's reduced system, checked
    there against `series_arb.py` to 10⁻⁶⁵), converted to (P, Φ, C, Q). η enters as stated, through its weighted bound.
  - **Model.** Lemma 1's system and the Φ- and ϑ-rates of §1.3 are coded from the note. They are checked at random
    complex states against B8's reduced system of #1017 (another statement of the same M-model): the same solution,
    rates converted.
  - **G⁰ and the Jacobian blocks** by forward-mode derivatives of my own G⁰. The sup over Σ_Z is taken on my own cells,
    polar cells with the series evaluated in intervals plus η.
  - **R_Φ** from the ε-expansion of f^ε_Φ − f^0_Φ on the inner series, computed as a series in (ε/z, z), not from
    Lemma 3's formula. Lemma 3's claim, that the ε/z² term cancels, is then a check. The tail in τ = ε/t ≤ τ₀ is bounded
    separately. R_C, T_F and Λ_ϑ follow §4's definitions, and the ε-parts ℓ_j follow §4's Schwarz–Cauchy bound with my
    own G.
  - **L** is assembled per §4's table. Super-solutions are found by iterating x ← (b + Lx)(1 + margin) and certified by
    b + Lx < x in intervals. This is done for the forcing, unit data, δ_ref and (H8)'s rounded-up δ.
- **Pass criteria.**
  - **(P1)** Lemma 1 agrees with B8's reduced system to 10⁻⁴⁰ relative at 10 random states. G⁰ equals the ε → 0 limit
    of my exact Φ-rate minus Φ/(2z²) to 10⁻⁴⁰. The ε/z² part of r_Φ on the inner series vanishes to 10⁻⁴⁰ at the
    centre. The value at t → 0, R(0), agrees with B5_inner.md §2's 35.18 to its digits.
  - **(P2)** With my constants, the super-solution for δ_ref is certified (b + Lx < x strictly), max x ≤ 10¹², and the
    Perron-weighted norm of L is below 1.
  - **(P3)** After my numbers are committed, they are compared with `b5_bounds_k0.3_Z0.0085.log`: L's largest entries,
    the norm Λ, R_Φ, R_C, T_F and every super-solution. These are upper bounds by different cells, so the reading is a
    ratio. A ratio outside [1/3, 3] is investigated, and a ratio inside is reported as is.
- **Controls, each with a bar that its mutation crosses by a wide margin.**
  - **(K1) Off the solution.** The inner series is perturbed by A × (1 + 10⁻³) in Y_in only, the model keeping A. Lemma
    3's cancellation then fails, so |t r_Φ/ε| grows like 1/|t|. The check, max over |t| = 10⁻⁶ Z of |t r_Φ/ε| at most
    3 × its max at |t| = Z, must fail. On the solution the two are within a few tens of percent.
  - **(K2) Term control.** G⁰ without its 𝒞ℬ₀P/(2z) term. The (P1) check of G⁰ against the limit of the exact rate
    (bar 10⁻⁴⁰) must fail.
  - **(K3) The contraction must be able to fail.** The Φ ← C block of L is scaled by 1.5. The super-solution for δ_ref
    must then not be certified, because the norm is above 1.
- **Output.** The logs and records go to the raw-data repository as `I1019-mac/b5/`. The code, and this section, stay in
  the branch.

### 6.2 Results (own numbers, committed before the comparison)

The logs and records are in `I1019-mac/b5/` (09c7f920). The code is `p2c_b5.py` at 5f58c30b, and d3d9cdef for the box
rerun. It uses mpmath 256-bit intervals.

- **(P1) passes.**
  - Lemma 1 agrees with B8's reduced system to 2.8×10⁻⁷⁷. So B5_inner.md and B8_constants.md state the same M-model,
    each coded from its own note.
  - §1.3's Φ_z equals the total derivative along the flow to 4.0×10⁻⁴⁹, and G⁰ equals the ε → 0 limit to 6.0×10⁻⁶⁸.
  - On the inner series, z²(f^ε − f⁰) as a series in (τ, z) has vanishing τ⁰ part (8.8×10⁻⁶⁰ relative) and vanishing
    z⁰, z¹ parts at every τ order (3.5×10⁻⁷⁵): Lemma 3's cancellation.
  - R(0) = 35.1835.
  - Lemma 3's formula, coded from the note, agrees with that ε-series to 1.9×10⁻⁴⁹ at interior points. It is
    therefore used for R_Φ on the cells.
- **(P2) passes**, over the box, for every n ≥ e^{73.89}. The run uses 3073 polar cells: 32 uniform radial bins on
  [0.1Z, Z], 16 geometric bins below, 64 angular bins and an origin box. The bootstrap ball is included, with a priori
  radius x ≤ 10¹².

| constant | here (box) | B5_inner.md §5.2, certified | §5.2, double precision (not rigorous) |
|---|---|---|---|
| L, Φ₋₁ ← C₀ | 245.7 | 278 | 237 |
| L, Φ₊₁ ← C₀ | 199.5 | 235 | 196 |
| L, Φ₊₁ ← C₂ | 144.0 | 173 | 137 |
| L, Φ ← P | 98.0 | 101 | 97 |
| Perron-weighted norm of L | 0.9410 | 0.9663 | 0.937 |
| R_Φ | 49.58 | 91.5 | 32–40; 35.18 at z → 0 |
| R_C | 3.345 | 3.43 | 3.23 at z → 0 |
| T_F | 3.934 | 4.15 | |
| ε-parts ℓ_j | ≤ 5.0×10⁻¹⁴ | ≤ 7.6×10⁻¹⁴ | |

  Every bound here lies between B5's double-precision samples and its certified bounds.

  The super-solutions are all certified (b + Lx < x), with the maximum over harmonics shown:

| right-hand side | x_P | x_Φ | x_C | x_ϑ |
|---|---|---|---|---|
| forcing only | 16.07 | 3782 | 0.435 | 7.87 |
| unit data in P | 7.135 | 1444 | 0.163 | 10⁻⁸ |
| unit data in Φ | 0.0438 | 10.30 | 0.0010 | 10⁻⁸ |
| unit data in C | 17.70 | 4164 | 1.464 | 10⁻⁸ |
| unit data in ϑ | 0.0789 | 18.55 | 0.0065 | 1.000 |
| δ_ref = (1, 200, 10, 20) | 210.5 | 49297 | 15.57 | 27.87 |
| (H8)'s δ rounded up (8.3×10⁻⁵, 79.04, 3.232, 6.464) | 77.24 | 18174 | 5.288 | 14.33 |

  For (H8)'s δ, x_Φ = (18174.4, 17603.9) for μ = (−1, +1). `b5_jump.py` gives (39789.7, 38781.0) with the same δ
  (§6.0). The two are consistent bounds: this one is tighter by the smaller R_Φ and contraction.
- **Controls.**
  - **K2** (G⁰ without 𝒞ℬ₀P/(2z)) fails its check at 5.5×10⁻³ against 10⁻⁴⁰.
  - **K1 as written** (A × (1 + 10⁻³) in Y_in) is not off the solution. A enters only through P(0) = 2F, so it is
    another solution, and Lemma 3 still holds. It moved R(0) to 35.265 and so failed the R(0) check, not the growth
    check that it was meant to fail. That is a design error of the protocol. **K1b** (Φ + 10⁻³z, B5_inner.md §6's
    kind) is off the solution: Lemma 3's zeros fail at 5×10⁻⁴, and |t r_Φ/ε| reaches 58844 at |t| = 10⁻⁶Z against 43
    at Z.
  - **K3 as written** (Φ ← C × 1.5) did not fire: the norm is 0.955, and δ_ref's super-solution is still certified.
    So did a × 2 supplement (0.964). The Φ ← C loop closes through the small C rows (γZ ≈ 0.03), and the norm is set by
    the Φ ↔ P loop. That is a design error of the protocol: the block was chosen from B5's numbers, not from the loop
    structure. **K3c** (Φ ← P × 2, B5_inner.md §6's own mutation) fires: norm 1.243 (B5 reports 1.27), and no
    positive super-solution, with exit 1.
- **Development fixes**, made before these runs:
  - R_Φ by Lemma 3's formula rather than by the ε-series, whose rigorous tail bound lost the ε-cancellation;
  - the super-solution as (I − L)⁻¹(b + m·1), so that rows without data keep slack;
  - τ₀ = n^{−2γ}, where a first version used n^{−(1−2γ)}, which was only conservative;
  - the box run's centre-only checks.

### 6.3 The comparison with `b5_bounds`

`p2c_b5_compare.py` sets this review's box record beside `b5_bounds_k0.3_Z0.0085.json`, opened after §6.2 was committed
(`I1019-mac/b5/logs/compare.log`).

- **Constants**, as ratios here/b5_bounds: Perron norm 0.974, R_Φ 0.542, R_C 0.977, T_F 0.948, ε-part 0.66. L's largest
  entries are 0.88 (Φ₋₁ ← C₀) to 0.97 (Φ ← P). Every one is inside [1/3, 3].
- **Super-solutions.** Every case is certified on both sides. Inside [1/3, 3]:
  - δ_ref: P 0.54, Φ 0.53, C 0.75–0.81, ϑ 0.96;
  - unit data in P: 0.68–0.71;
  - unit data in C: 0.60–0.90;
  - unit data in ϑ: 0.46–1.00.
- **Ten ratios outside [1/3, 3]**, investigated as the protocol asks:
  - forcing only: P and Φ 0.24–0.25, C 0.17–0.19;
  - unit data in Φ: the C components, 0.22–0.29.

  They are the product of two factors measured separately. One is R_Φ (ratio 0.542). The other is the amplification
  of Φ data through the P ↔ Φ loop (unit-Φ x_Φ ratio 0.436). Their product, 0.237, matches the forcing-only Φ ratio of
  0.247. The C components follow through the C ← P coupling. Both computations are upper bounds, and this one is
  tighter. No discrepancy is found.
- **What the tighter bound would give B5(2).** C_E ∝ x_Φ,₋₁ + x_Φ,₊₁ (B_chain.md §3.2). With this review's x_Φ for
  (H8)'s rounded-up δ, C_E would be 1.1533×10³³ = e^{76.128}, against `b5_jump`'s 2.532706×10³³ = e^{76.915}. That
  is not certified by reviewed code, and it is reported for information only.

### 6.4 Discrepancies (reported, not fixed), and status

- **The δ question (§6.0).** `b5_threshold.py`'s DELTA0 is B5's leading data, rounded down despite its label. It is
  not (H8)'s δ. The chain's x_Φ and C_E use the certified δ, so they stand.
- **B5's certified constants are conservative.** They are larger than this review's by factors up to 1.8 (R_Φ), and
  its super-solutions by up to 4 (forcing only). Both are valid bounds.
- **The η bound** is quoted to two digits in B5_inner.md §5.1 (2.7×10⁻⁸, 5.5×10⁻⁷, 6.5×10⁻¹¹). This review inflated
  those values by 10 % rather than read them from `b5_bounds`'s record. Its effect on every constant is below the
  digits shown.
- **Protocol defects of this review**, recorded rather than corrected:
  - K1 was not off the solution.
  - K3 targeted a block that does not set the contraction.
  - Their supplements K1b and K3c, chosen as in B5_inner.md §6, fire.

**Status.** B5(1) for M = 1 is confirmed for every n ≥ e^{73.89} over the box (B8's ±2×10⁻⁶(1 + i), which contains
B5's) by an independent computation. Its constants come out tighter than the certificate's, and its super-solution for
δ_ref is certified (Perron norm 0.941).
- **What it shares** with `b5_bounds`: lemma S's η bound, the parameter box and the statements of B5_inner.md §§1–4.
  Lemma 3's formula is shared, and checked here against an independent ε-expansion to 1.9×10⁻⁴⁹.
- **The inner series** is #1017's, checked against `series_arb.py`.
- **Not done:** this review's code has not itself been reviewed. Versions: mpmath (iv) at 256 bits.

### 6.5 Reproduce

From `solver/proof/review_p2c/`, with the raw-data repository cloned beside this checkout (§4.5):

```
RAW=../../../../choptuik-large-d-raw/I1019-mac/b5
python3 p2c_b5.py --model --phiz                    # stage A, PASS, exit 0
python3 p2c_b5.py --phiz --mutate=G0                # K2: FAIL, exit 1
python3 p2c_b5.py --residual                        # stage B, PASS, exit 0 (about 20 s)
python3 p2c_b5.py --residual --mutate=offphi        # K1b: FAIL, exit 1
python3 p2c_b5.py --sector --box --nr=32 --na=64 --out=$RAW    # stage C over the box, (P2) PASS, exit 0 (about 2 min)
python3 p2c_b5.py --sector --nr=16 --na=32 --mutate=jacP2 --out=$RAW   # K3c: no super-solution, exit 1
python3 p2c_b5_compare.py $RAW > $RAW/logs/compare.log                  # (P3)
```

`$RAW/run_c.sh` runs the stage-C set and the K3 variants. The stage-A and stage-B logs are beside them.

## 7. Item 4: Lemma B8(1)'s bootstrap closure, a second implementation

Issue #1061. **Claim under review** (C_selection.md §13.1, B8_constants.md §§0, 7, 8.4): with K = 4, κ = 0.3, κ₁ = 0.6,
δ = 0.05, the bootstrap P(h) < h of Lemma B8.a closes for ln n₀ ≥ 62 in the gauge c(0) = c^K(0), and for ln n₀ ≥ 73.89
with the normalization offset |c(0) − c^K(0)|₁ ≤ d₀n^{−K−1}, d₀ = 10⁷. Written 2026-11-21 by Claude (Anthropic) under
Nathan Moinvaziri's direction, by a fresh agent that had not worked on B8 or read its derivation sessions. Not reviewed.

### 7.1 Protocol, fixed before the run

- **Independence.**
  - **Read:** B8_constants.md §§0–7 and §8.4 item 1 (the offset), the B8 rows and bullets of C_selection.md §13.1,
    P2b_review.md §§0–1 and 8, B5_inner.md §5.1's parameter box, and this file's §§1–2 and §6.1 for the format.
  - **Seen before the run:** B8_constants.md §3's atom table, §5's kernel numbers at ln n₀ = 73.89 (s_* = 0.1875,
    H_F = 14.04, N_F = 117.5, K_F = 1.763, γ_F = 0.4150), §7.1's table (h and the ratios at ln n₀ = 73.89, 70, 66, 62)
    and §7.2.
  - **Not opened** until this review's numbers are committed: `b8_alg.py`, `b8m_bounds.py`, `b8m_scan.py`,
    `b8m_formal.py`, `b8m_atoms.py`, `b8m_h8.py`, `b8m_identities.py`, their logs, the `review_p2b/` code and
    `review_p2c/p2c_h8.py`. The box centre and the series' gauge were taken by `grep` (A* = 2.5421480475,
    w* = 0.7769940136 in `fixed_point.py`).
- **Shared with the certificate.** The statements: the model and its reduced form (§§1.1–1.2), the description of the
  order-by-order recursion and its gauge (§2), Lemma W's norms, Lemma R's identities (§4), Lemma K (§5), the map P of
  Lemma B8.a with its regions and paths (§§1.3, 6), and the offset's P_V + 2|F|₁d₀, P_c + d₀ (§8.4). The settings:
  K, κ, κ₁, δ, x_F = 20, the box (A, ω/ω∞) ∈ (A*, w*) ± 2×10⁻⁶(1 + i), ω∞ = 2π/1.64389375, d₀ = 10⁷ and, for 𝒟′_n,
  Z_B = 0.0085.
- **Method, own** (`solver/proof/review_p2c/p2c_b8_series.py`, `p2c_b8_alg.py`, `p2c_b8_boot.py`; python-flint 0.9.0,
  FLINT 3.6.0, arb/acb at 256 bits).
  - **Series and atoms.** My own recursion for Z_k (k ≤ 4), derived from the reduced system, in acb balls over the box.
    The residuals τ_D, τ_v, τ_b, ρ₄ and the K-only combinations (ρ₂ det C, ρ₃ det C², η̃ det C, ℓ det C²/x², Γ as an
    exact quotient) are exact polynomials in (ε, x). Lemma F's structural zeros are dropped after a check.
  - **One set of formulas, two arithmetics.** G₁, W₂, ĝ, q and the derivative bounds of pass 1 are written once. A point
    arithmetic (acb jets of f, x f′, (x d/dx)²f) checks Lemma R and is the planted truth. A cell arithmetic gives norm
    bounds uniform in n ≥ n₀: radial cells for N0, N1 and the ray R (a cell [0, 10⁻⁶] with powers of r kept, 24
    log-spaced cells to 0.05, 160 to 20 with breaks at 1 ∓ δ/2, 1 ∓ 0.01 and 1), and F as one region in (ρ, X). Each
    bound is affine in h (a product of two remainders folded with h ≤ H_max), so P is built once per n₀.
  - **Gains.** Lemma K.1–K.3 evaluated per target cell (the arc term α e^{H α} where N_r is not bounded below), and
    K.3's γ_F on F.
  - **Closure.** Iterate h ← P(h) from 0 to its fixed point h*; then test P(h) < h strictly in balls at h = h* + t u,
    (I − J)u = h*, J the Jacobian of P at h*, for t = 10⁻⁴ … 0.3, with h ≤ H_max. Not the certificate's descent.
- **Pass criteria.**
  - **(P1) Structure.** Lemma F's dropped coefficients contain 0 over the box and are below 10⁻⁶⁰ at the centre. §3's
    atom table is compared, not required.
  - **(P2) Identities.** I1–I4, xV^K′ = ΩS^K and my factorizations of Λ, (A^K)⁻¹ and q, at 6 random complex points
    (x, n) at the box centre: relative residual below 10⁻⁴⁰.
  - **(P3) Planted instances.** On 11 cells at ln n₀ = 73.89, with h = (2.4×10⁶, 6.3×10⁶, 6.2×10⁵, 1.7×10⁵), random
    remainders of norm exactly h·w at sampled points, their derivatives from pass 1's equations: every exact quantity
    below its bound (ratio < 1).
  - **(P4) Kernel and gains.** Lemma K.1's margin on the domain's ray on a grid t ∈ [10⁻⁴, 10⁶]. The scalar kernel
    e^{(n−1)(Φ(x)−Φ(s))} (H′ = 0) integrated along P_x at n = 300 for 6 targets, below the gain formula.
  - **(P5) Closure search, planted.** An affine map whose spectral radius crosses 1 at L = 58.946: not closed at
    L − 0.2, closed at L + 0.2.
  - **(P6) The claim.** Strict closure on 𝒟_n at ln n₀ = 62 with d₀ = 0, and at 73.89 with d₀ = 10⁷. Then the
    smallest closing ln n₀ by bisection to 0.25, for d₀ = 0 and d₀ = 10⁷, on 𝒟_n and on 𝒟′_n. If either point of the
    claim does not close here, that is reported as not confirmed by this implementation, and located after the
    comparison; no setting is changed to make it close.
  - **(P7) Comparison,** after the commit of (P6): h, s_*, K_F, γ_F at 73.89 and 62 against §§5, 7.1. A ratio outside
    [1/3, 3] is investigated.
- **Controls.**
  - **(K1) Scaled atom.** S₂'s top coefficient × 1.001: (P1) must fail.
  - **(K2) Dropped term.** ρ₂ without its first term: I1 must fail. x W₂ dropped from the formulas: I2 must fail. In the
    cell arithmetic only, ρ₂ dropped from N₁, or W₂ dropped: (P3) must fail.
  - **(K3) Wrong κ.** The decay margin taken as sin 0.4 with the domain at κ = 0.3: the kernel check must fail. κ = 0.02
    throughout (consistent but too small): no closure at 73.89 (s_* < 0).
  - **(K4) Sensitivity,** reported, no pass bar: τ_D × 10.
  - **Found insensitive during development**, recorded here: dropping half of the product rule in every derivative
    (P3 still passes; the dropped half is small against the bounds' slack), and the gain check under the wrong κ margin
    (the gain bounds are about 8 times the planted integrals at n = 300).
- **Output.** Logs and JSON records go to the raw-data repository as `I1061-mac/`. The code and this section stay in the
  branch.

### 7.2 Results (own numbers, committed before the comparison)

Logs and JSON records are in the raw-data repository, `I1061-mac/` (`run_i1061.sh`, `run_i1061b.sh`,
`run_i1061c.sh`), run 2026-11-21 02:37–04:33 UTC with the code at 302a7197. Each n₀ takes one to four builds of
379 cells at 33–41 s each.

**The claim is confirmed: P(h) < h closes at ln n₀ = 62 with d₀ = 0, and at ln n₀ = 73.89 with d₀ = 10⁷, on 𝒟_n.**
Every component of P(h) is below h as balls (upper end of P(h) below the lower end of h), with h ≤ H_max.

| ln n₀ | d₀ | closes | h = (h_V, h_S, h_β, h_c) | ρ(J) at h* | s_* | K_F | γ_F |
|---|---|---|---|---|---|---|---|
| 62 | 0 | yes | (6.58e6, 1.77e7, 1.06e6, 3.21e5) | 0.880 | 0.1636 | 2.033 | 0.5479 |
| 73.89 | 0 | yes | (1.62e6, 4.37e6, 4.38e5, 1.12e5) | 0.647 | 0.1875 | 1.763 | 0.4151 |
| 73.89 | 10⁷ | yes | (1.02e8, 1.38e8, 1.18e7, 1.27e7) | 0.686 | 0.1875 | 1.763 | 0.4151 |
| 62 | 10⁷ | no: no fixed point below 10¹¹ | | | | | |

In every closing run both maxima, of P_S and of P_β, are attained on F. Doubling the cells (n0c = 48, n1c = 320)
changes no printed digit of h at 62 (d₀ = 0) or at 73.89 (d₀ = 10⁷) (`fine_*.log`).

**Smallest closing ln n₀** (bisection; "not at" is the largest tested ln n₀ that did not close):

| d₀ | domain | closes at | not at | ρ_max at the two ends | how it fails below |
|---|---|---|---|---|---|
| 0 | 𝒟_n | 59.9375 | 59.75 | 0.010478, 0.010511 | ρ(J) → 1 (0.961 at 59.9375) |
| 10⁷ | 𝒟_n | 70.4375 | 70.25 | 0.008916, 0.008940 | no self-consistent H_max; ρ(J) = 0.763 at 70.4375 |
| 0 | 𝒟′_n | ≤ 11 (the bracket's end) | | 0.00927 at 11 | |
| 10⁷ | 𝒟′_n | 12.4062 | 12.0546 | 0.008877 at 12.4062 | no self-consistent H_max; ρ(J) = 0.754 at 12.4062 |

- On 𝒟_n the ln n₀ dependence enters through ρ_max(n₀) = (1 + R′_{n₀})²/n₀ alone. The N0 and N1 cells contribute
  terms of relative size ε₀ ≤ e^{−54}, and F depends on n₀ only through ρ_max and n₀/(n₀ − 1).
- **On 𝒟′_n (|x|² ≤ Z_B n, Z_B = 0.0085) neither 62 nor 73.89 is a threshold of the bootstrap.** There
  ρ_max = (1 + (Z_B n₀)^{1/2})²/n₀ stays near 0.0085, and the closure gives the same h at ln n₀ = 54, 62 and 73.89,
  with and without the offset. A second pair of bisections (`run_i1061c.sh`) went down to ln n₀ = 11, below which F
  is empty ((Z_B n)^{1/2} < x_F for ln n < 10.8). Without the offset the closure holds at 11, with h = (2.59e6, 6.98e6,
  5.61e5, 1.53e5). With d₀ = 10⁷ it holds down to 12.4062, where ρ_max has risen to 0.008877.
- **So the closure is a function of ρ_max alone**, to the resolution of these bisections. Without the offset it
  holds at ρ_max = 0.010478 and fails at 0.010511 (𝒟_n). With d₀ = 10⁷ it holds at 0.008916 (𝒟_n) and 0.008877
  (𝒟′_n), and fails at 0.008940 (𝒟_n) and 0.008951 (𝒟′_n, ln n₀ = 12.0546).
- **The offset threshold 73.89 is not sharp.** With d₀ = 10⁷ the closure holds on 𝒟_n down to ln n₀ = 70.4375. The
  certificate ran the offset only at 73.89 (B5(1)'s threshold), so this is consistent with the claim.

**Checks** (§7.1's criteria):

| check | bar | result |
|---|---|---|
| (P1) structural zeros, centre / box | below 10⁻⁶⁰ / contain 0 | largest 1.14×10⁻⁶⁹ (τ_v) / all contain 0: PASS |
| (P1) §3's atom table | compared | reproduced to its digits: V 5.084, 48.3, 526, 6.0e3; S 15.9, 254, 4.0e3, 5.8e4; β 12.9, 254, 3.3e3, 4.9e4; c − 1 12.9, 177, 2.1e3, 2.5e4; Γ 9.69, 191, 2.8e3, 3.8e4; τ_D, τ_b 9.3e5, 7.9e5; τ_v, ρ₄ 1.3e6, 9.3e5 |
| (P2) Lemma R and factorizations | below 10⁻⁴⁰ | largest 9.6×10⁻⁶⁴ (I3): PASS |
| (P3) planted instances, 11 cells | below 1 | largest just below 1, the R_c integrand on the cell [0, 10⁻⁶]: PASS |
| (P4) Lemma K.1's margin on the ray | ≤ 0 | −9.1×10⁻¹³ (the margin is tight as t → ∞): PASS |
| (P4) scalar kernel integral / gain, n = 300 | below 1 | largest 0.106: PASS |
| (P5) planted closure search, threshold 58.946 | open at −0.2, closed at +0.2 | as required: PASS |

The (P3) ratio is tight by construction: the R_c integrand is dominated by |[c^K R̂_β]| ≈ |R̂_β|, and the instance
takes |R̂_β| = h_β w exactly.

**Controls:**

| control | required | result |
|---|---|---|
| K1: S₂'s top coefficient × 1.001 | (P1) fails | τ_D's dropped coefficients no longer contain 0 (0.232 at the centre): exit 1 |
| K2: ρ₂ without its first term | I1 fails | I1 relative residual 0.51: exit 1 |
| K2: x W₂ dropped from the formulas | I2 fails | I2 residual 19.9: exit 1 |
| K2: ρ₂ dropped in the cell arithmetic only | (P3) fails | exact/bound up to 1.2×10²⁹: exit 1 |
| K2: W₂ dropped in the cell arithmetic only | (P3) fails | exact/bound up to 398: exit 1 |
| K3: decay margin sin 0.4, domain κ = 0.3 | kernel check fails | margin +0.094: exit 1 |
| K3: κ = 0.02 throughout | no closure at 73.89 | s_* = −0.083: not closed, exit 2 |
| K4: τ_D × 10 at 62 (no bar) | reported | still closes, h = (5.03e7, 1.36e8, 6.11e6, 2.03e6), ρ(J) = 0.916 |

K4 shows that the forcing's size moves h, not the closure. ρ(J) depends on the forcing only through the folded
quadratic terms. So a wrong atom is caught by (P1), not by the closure. The two insensitive checks recorded in the
protocol behave as recorded: half of the product rule dropped leaves (P3) at its value, and the gain check under the
wrong margin gives 0.119.

**The search driver was changed once, after the first bisection** (069f22c0; the first driver's logs are in
`I1061-mac/first_driver/`). The first bisection gave 60.125. At 59.9375 its fixed point lay just under H_max, so the
test point h* + t u left the range where the bounds hold and was rejected. The driver now rebuilds the bounds until
h* ≤ H_max ≤ 1.2 h*, and tests only points with h ≤ H_max. It also has a polyhedral fallback: P is a maximum of
affine maps, so {h : P(h) < h} is a linear feasibility problem. No run above needed the fallback. The bounds, the
cells and the pass criteria did not change. Every certificate point is verified in balls whichever way it was found.

### 7.3 The comparison with the certificate

Opened after 700cd8c9: `b8m_bounds.py`, `b8_alg.py`, `b8m_scan.py`, the scan logs `b8m_scan_K4_box.log`,
`b8m_scan_K4_d0_1e7.log`, `b8m_scan_K4_d0.log`, `b8m_scan_K4_fixedZ.log` and `b8m_atoms_K4.log`.

**Fixed points of P** (the certificate's `b_fixed` and mine), and the certified h (the certificate's `hyp`):

| ln n₀ | d₀ | | V | S | β | c |
|---|---|---|---|---|---|---|
| 62 | 0 | fixed point, here / certificate | 1.016 | 1.016 | 1.056 | 0.986 |
| 62 | 0 | certified h, certificate / here | 7.16e6 / 6.58e6 | 1.91e7 / 1.77e7 | 1.09e6 / 1.06e6 | 3.57e5 / 3.21e5 |
| 73.89 | 0 | fixed point, here / certificate | 0.747 | 0.747 | 0.717 | 0.739 |
| 73.89 | 0 | certified h, certificate / here | 2.41e6 / 1.62e6 | 6.27e6 / 4.37e6 | 6.17e5 / 4.38e5 | 1.67e5 / 1.12e5 |
| 73.89 | 10⁷ | fixed point, here / certificate | 1.110 | 1.246 | 1.112 | 1.019 |
| 73.89 | 10⁷ | certified h, certificate / here | 9.71e7 / 1.02e8 | 1.20e8 / 1.38e8 | 1.13e7 / 1.18e7 | 1.28e7 / 1.27e7 |

Every ratio lies in [0.72, 1.25], inside (P7)'s [1/3, 3]. The kernel constants agree to 3–4 digits:

| ln n₀ | | s_* | H_F | N_F | K_F | γ_F |
|---|---|---|---|---|---|---|
| 62 | certificate | 0.16364 | 14.390 | 98.57 | 2.0327 | 0.54769 |
| 62 | here | 0.16360 | 14.393 | 98.57 | 2.0334 | 0.54790 |
| 73.89 | certificate | 0.18751 | 14.044 | 117.48 | 1.7630 | 0.41504 |
| 73.89 | here | 0.18749 | 14.047 | 117.48 | 1.7634 | 0.41514 |

Mine is larger by 2×10⁻⁴ in H_F because my F majorant bounds |x|^m by X^m, dropping θ^m. The thresholds agree:
the certificate closes at 62 and fails at 58 ("c lower bound"), and mine closes at 59.9375 and not at 59.75. Its
`fixedZ` run (ρ_max = 0.0085 at ln n₀ = 90) gives the 73.89 values to five digits, which is §7.2's observation that
the closure depends on ρ_max only.

**Where the two differ, located by reading the certificate's code:**
- *One box per region against radial cells.* The certificate bounds N0, N1, R and F each as one box. On N1 that box
  takes X²/|x² − 1| ≤ 230 (`ell["N1"]`) and on R ≤ 2√2. Its β forcing on N1 is then 5.1×10⁵ (`gl_N1`) and pins P_β at
  6.10×10⁵ for ln n₀ = 73.89, 70 and 66 (B8_constants.md §7.2). On my cells N1's β forcing is below F's, and both
  maxima sit on F. This is the 25–30 % at 73.89. At 62 F dominates in both, and the fixed points agree to 6 %.
- *Products of norms against exact cancellations.* The certificate bounds Λ G₁ as ‖(D^K)⁻¹‖‖A^K‖‖(A^K)⁻¹‖|N₁|,
  q(ΛG₁ + xW₂) as ‖q‖ times that, ρ₂ and ρ₃ term by term, and keeps (n − 1)δD S and δD D⁻¹ A₀ S apart. I use
  Λ G₁ = (D^K)⁻¹N₁, q(ΛG₁ + xW₂) = ℓ̂(I + e)⁻¹[N₁ + (I − δD D⁻¹)N₂]/(n − 1), ρ₂ det C and ρ₃ det C² as exact
  polynomials, and −(n − 1)x²δD D⁻¹S. These matter on N0 and near the light cone, not on F.
- *The weight w₁ on F.* My cells carry w = θρ^{K+1} exactly and bound w₁ = w/θ by w/θ_F, a factor 21/20 on the R_V
  and R_c terms. The certificate keeps θ in its monomials. With the offset, h_V and h_c are large and this factor
  shows. A diagnostic run with w₁ ≤ w on F (unsound, `diag_thF.py`) gives h* = (9.76e7, 1.26e8, 1.09e7, 1.25e7). That
  is about 40 % of the gap in h_S; the rest was not traced.
- *The far sup.* The certificate takes, per monomial, the smaller of X_lo^a n₀^e and X_lo^{a+2e}ρ_max^{−e}, using
  n ≥ n₀ at small X. Mine uses the (X, ρ) box only.
- *The closure test.* The certificate descends from 1.1 times its fixed point by h ← 1.01 P(h). Mine tests
  h* + t u with (I − J)u = h*, with H_max self-consistent.

**What is shared.** The statements listed in §7.1, the settings, the box centre and ω∞, d₀ and Z_B, and the ball
library (python-flint 0.9.0 on FLINT 3.6.0). No code is shared: the series, the atoms, the bound arithmetic, the
gains, the identities' check and the closure search are this review's own.

### 7.4 Discrepancies and observations (reported, not fixed)

1. **The threshold 62 belongs to 𝒟_n, not to 𝒟′_n.** Item 4's row in §1 and issue #1061 state the closure "on
   𝒟′_n, closing for ln n₀ ≥ 62". On 𝒟′_n (fixed Z_B = 0.0085) the bootstrap closes far lower: at ln n₀ = 11 here
   without the offset, and from 12.41 with d₀ = 10⁷. The value 62 is the threshold on 𝒟_n, whose radius
   (κ₁n/(cos κ ln n))^{1/2} gives ρ_max ≈ κ₁/(cos κ ln n₀) > Z_B for ln n₀ < 73.89. The certificate's own `fixedZ` run
   shows the same. C_selection.md §13.1's row (n ≥ e^{62}; with the offset, n ≥ e^{73.89}) is correct but
   conservative. No proved statement changes.
2. **The offset threshold 73.89 is not sharp.** With d₀ = 10⁷ the closure holds on 𝒟_n down to ln n₀ = 70.4375
   here. The certificate ran the offset only at 73.89, which is all B5(1) needs.
3. **§3's atom table omits V's ε⁴ entry** (7.04×10⁴). The certificate's atoms carry it (its x V′ at ε⁴ is
   1.727×10⁵, as mine); `b8m_atoms_K4.log` prints four entries per row, and the table copied them.
4. **B8_constants.md §7.2's "P_β is almost all forcing on N1"** describes the certificate's N1 box, not the
   problem. On finer cells the N1 forcing falls below F's.
5. **The scaled-atom control works through the structure check only.** τ_D × 10 still closes at 62 (§7.2, K4). A
   check that a wrong atom moves the closure would not fire; Lemma F's structural zeros do.

### 7.5 Reproduce

From `solver/proof/review_p2c/`, with the raw-data repository cloned beside this checkout:

```
RAW=../../../../choptuik-large-d-raw/I1061-mac
python3 p2c_b8_series.py --centre                      # (P1), STRUCTURE PASS, exit 0
python3 p2c_b8_series.py --centre --mutate=atom        # K1, STRUCTURE FAIL, exit 1
python3 p2c_b8_boot.py --check=identities              # (P2), largest residual 9.6e-64, exit 0
python3 p2c_b8_boot.py --check=identities --mutate=drop_rho2   # K2, I1 0.51, exit 1
python3 p2c_b8_boot.py --check=instances --L0=73.89    # (P3), exit 0 (about 4 s)
python3 p2c_b8_boot.py --check=instances --L0=73.89 --mutate=bound_W2   # K2, 398, exit 1
python3 p2c_b8_boot.py --check=kernel --mutate=kappa   # K3, margin +0.094, exit 1
python3 p2c_b8_boot.py --L0=62                         # closes, h = (6.58e6, 1.77e7, 1.06e6, 3.21e5) (about 80 s)
python3 p2c_b8_boot.py --L0=73.89 --d0=1e7             # closes, h = (1.02e8, 1.38e8, 1.18e7, 1.27e7)
python3 p2c_b8_boot.py --bisect=56,62                  # closes at 59.9375, not at 59.75 (about 25 min)
RAW=$RAW bash $RAW/run_i1061.sh; RAW=$RAW bash $RAW/run_i1061b.sh; RAW=$RAW bash $RAW/run_i1061c.sh   # everything
```

`run_i1061.sh` was stopped after its `run_L62_d1e7` step; `run_i1061b.sh` runs the steps after it, with the offset
brackets corrected. Times are on an M-series Mac at nice 10.

**Not done.** This review's code has not itself been reviewed. (H8)'s conversion was item 3 (§5) and is not redone.
The bootstrap was run only for M = 1, K = 4, κ = 0.3, κ₁ = 0.6, δ = 0.05 and x_F = 20.
