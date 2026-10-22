# Reproduction test

This is a pass over the repository as an outside reader would make it, with only the README and a fresh clone. For each
documented command it asks three things:

- does the command run as written, from the stated directory?
- do the files it names exist?
- does it print the number the text quotes, at the quoted precision?

Nothing in the repository was changed except this file. Every fix below is a proposal.

The test was made on 2026-10-21 from a fresh `git clone https://github.com/nmoinvaz/dss-residual-kernel` at **78de00b**
("README: 'Not found in the critical-collapse ...' as its own paragraph").

## Environment

| | |
|---|---|
| Machine | Linux 6.18.44 x86_64 (Ubuntu 24.04.5), 2 cores, Intel Xeon @ 2.80 GHz with AVX-512F, AVX2 and FMA |
| Load | Shared with several other agents. Load average was 7–13 on 2 cores during every run, so wall times are 2–5× the quoted ones; timing differences are not counted as failures. |
| Rust | rustc and cargo 1.97.0, maturin 1.15.0 |
| Python | 3.13.16 in a fresh venv. It started with only numpy 2.5.4, scipy 1.18.1, mpmath 1.3.0 and sympy 1.14.0 (the README's list). python-flint 0.9.0 was added next (needed, see below), then hpkernel (`hp/build.sh`) and rustkernel (`pip install .`), both built from the clone. |
| Runs | Every command ran under `nice -n 10`, one process at a time. The Rust builds used `CARGO_BUILD_JOBS=1`. |

The system Python already had `rustkernel` and `hpkernel` installed by other sessions, so a run against it could not
reveal a missing build step. All runs used the venv instead.

## Summary

- **75 runs of 71 distinct documented commands** from README.md, PROOF.md, proof/A_certify.md, proof/A_review.md,
  GAMMA.md and matter/*.md, including the build and test lines. Of the 71:
  - 59 are ok.
  - 3 fail: `hp/compare/cmp_ecker.py`, and `solver/matter/css_bg.py` for both κ = 1 and κ = 0.
  - 1 silently does nothing: `hp/gamma_compare.py` as quoted.
  - 1 needs a dependency the README does not list: `check_numbers.py`.
  - 7 run but disagree with the text: `candidates.py`, `plot_heldout.py`, `pert_check3.py` at its defaults,
    `largeD_expand.py`, `zero_M1.py`, one bound of `formula_all.py`, and 1 px in `plot_extrapolation.py`.
  - About 55 more were skipped as too long for this machine or out of scope; they are listed below with reasons.
  - The two helper scripts I wrote (data counts, and a test of the `css_bg.py` fix) are not counted.
- **The headline numbers reproduce.**
  - `./reproduce.sh --full`: 15 PASS, 0 FAIL, 0 SKIP (746 s).
  - `check_numbers.py --selftest`: 34 of 34 pass, and all 34 one-unit shifts are caught.
  - `cargo test --release`: 13 passed.
  - The held-out table (`hp/dd_fits_mp.py 10 45 --clean --dmax=60.1`) matches every quoted cell.
  - The M = 1 certificate re-check, the 8 mutation controls and lemmas S and 6a match proof/A_certify.md to every
    quoted digit.
  - The CSS table and fits of matter/CSS.md are exact.
- **Failures:**
  - `hp/compare/cmp_ecker.py` crashes with a NameError.
  - `solver/matter/css_bg.py`, the D = 4 validation that matter/CSS.md documents, converges to a wrong solution and
    still exits 0.
- **No broken links**: 368 relative links, 222 anchors and 11 footnotes all resolve. No figure is missing.

## Prioritised fixes

1. **`solver/matter/css_bg.py`: the D = 4 validation fails.**
   - What happens: run as documented (`python3 css_bg.py`, from `solver/matter`), Newton drifts from the published
     ω = 1.1769527 to ω = 3×10⁻⁵ (|F| = 4.5×10⁻⁵ after 40 iterations). It prints diffs of −1.18 in ω and 0.64 in
     |f(x_h)|, and exits 0. `css_bg.py 0` (κ = 0) ends at ω = 2.1755 against 1.9154446. Both happen with the venv and
     with the system Python.
   - Cause: `__main__` uses `cm.Grid(N)` and `guess_from_shooting(..., wcut=0.93)`. The callers that work
     (`css_momcheck.py`, `css_scan.py`) pass a grid stretch and `wcut=0.999`.
   - Fix, tested in a scratch copy: use `A = 5.0 if kap == 1 else 4.0`, `g = cm.Grid(N, A)`, `wcut=0.999` and
     `solve(..., A=A)`. That reproduces HHS96 to 1×10⁻⁹ (ω = 1.176952723) and HE95a to 2×10⁻⁷ (ω = 1.9154448).
   - Also: exit non-zero when a diff exceeds its tolerance, so a failing validation cannot pass silently.
2. **`check_numbers.py` needs python-flint, which the README does not say.**
   - What happens: with the README's four packages it dies with `ModuleNotFoundError: No module named 'flint'`, raised
     from the `certificate` check through `solver/proof/mutation_controls.py`.
   - Fix: list python-flint as required in "Build and test" and next to `check_numbers.py` ("How this work was done").
     Alternatively, make that one check SKIP without flint, as `reproduce.py` does.
   - The same gap applies more widely: every script in `solver/proof/` (and `review/`) and `solver/inner_amp/amp_flint*.py`
     needs flint, while the README names it only "for `solver/inner_borel/`".
3. **`hp/compare/cmp_ecker.py` crashes.**
   - What happens: `NameError: name 'sys' is not defined` at line 13. Line 3 imports `os, glob, json`; `sys` is first
     imported on line 35.
   - Fix: `import os, glob, json, sys` on line 3.
   - The script also needs a local clone of github.com/EckerChristian/CriticalSpacetimeCrystal as its first argument
     (default `ecc`). The README cites the script without saying so; say it next to the citation (README, "D = 5"
     bullet).
4. **`hp/gamma_compare.py`, quoted without arguments (README §The formula, GAMMA.md §1 and §7), does nothing.**
   - What happens: it prints a header line and exits 0.
   - Fix: write the command as
     `cd hp && python3 gamma_compare.py data/gamma_dd.jsonl data/local/gamma_dd.jsonl`. That form reproduces GAMMA.md's
     table exactly (3.8×10⁻⁴ … (1.6 ± 2.2)×10⁻⁶, a₁ = 3.03243). Alternatively, make those two files the default.
5. **`solver/candidates.py`: the held-out rms column of the README candidate table no longer reproduces.**
   - What happens: the χ²/dof column matches, but the held-out rms differs in 7 of 10 rows. For example:
     - free fit: 1.5×10⁻⁶ (quoted 1.4×10⁻⁶);
     - π²/6: 2.9×10⁻⁶ (quoted 1.9×10⁻⁶);
     - 2π/j₁,₁ and √e: 1.8×10⁻⁵ (quoted 1.3×10⁻⁵);
     - φ: 1.0×10⁻⁴ (quoted 7.7×10⁻⁵).
   - The README also says the held-out points are "D = 34–39.5", while the script uses every f64 point with D > 30.2.
   - Fix: rerun `cd solver/data && python3 ../candidates.py`, then update the column and the description of the
     held-out set (or restore the set the table was made with).
   - Separately, the value of 2π/j₁,₁ is 1.63978796, so the table's "1.6397878" should be **1.6397880**.
6. **`hp/plot_heldout.py` cannot regenerate the committed `hp/heldout.svg`.**
   - The README says the figure uses the points up to D = 60.1, and the Layout says it was "drawn before the points
     above D = 60.1 were added". The script has no `--dmax`, though, so `python3 plot_heldout.py` overwrites the
     committed figure with a different one (544 held-out points instead of 517).
   - Fix: add `--dmax` as in `dd_fits_mp.py` and document `python3 plot_heldout.py --dmax=60.1`. Alternatively,
     regenerate the figure on today's data and update the text.
7. **`solver/inner_pert/pert_check3.py`: the quoted number needs arguments that the docs do not give.**
   - What happens: README Proposition 5 and GAMMA.md §4 quote agreement "to 10⁻¹⁸⁰ … through j = 11 at 45 harmonics".
     The default run (M = 25, J = 10) prints "worst 5.9e-24".
   - `pert_check3.py 45 11` gives 2.1×10⁻¹⁷⁸ and the 7.5% controls, as quoted.
   - Fix: quote the command as `pert_check3.py 45 11`, or make 45 and 11 the defaults.
8. **Small text drifts** (each checked against the script output):
   - **Expansion order.** `solver/largeD_expand.py` runs to third order by default (`NORD = 3`; it prints orders 1–3,
     up to B₄, G₃, S₃, D₃). The README says "run to fourth order". Fix: say "third order (`largeD_expand.py 4` for
     the fourth)".
   - **`zero_M1.py` radii and digits.** `solver/inner_stokes/zero_M1.py` uses three matching radii (0.2, 0.225, 0.275)
     and prints five digits (A 2.54215, ω/ω∞ 0.77699). The README says "the same at four matching radii" and quotes
     A = 2.542148, ω = 2.969777. Fix: "three radii", or print more digits.
   - **Formula table bound.** `solver/formula_all.py` gives 4.3×10⁻⁹ for Δ on 4.3–5, where the README's "4.3 – 20" row
     says "≤ 4×10⁻⁹". Fix: write "≤ 4.3×10⁻⁹", or round the bound up.
   - **GAMMA.md §1.** It says the 1/n coefficient is "stable to 10⁻⁴ over fit choices: 3.0313–3.0324". That range is
     1.1×10⁻³ wide, so "stable to 10⁻³".
   - **Run time.** README §Reproduce says 4–6 min, while `reproduce.sh` and `reproduce.py` say about 6 min. Measured:
     730 s without hpkernel and 746 s with `--full`, under load 7–13. Fix: one figure in all three places, or "6–12
     minutes on a busy 2-core machine".
   - **Layout, γ points.** It says `hp/gamma_dd.py` has "results in hp/data/gamma_dd.jsonl, 28 points D = 10-61". That
     is true of that one file, but the 67 points quoted everywhere else need `hp/data/local/gamma_dd.jsonl` too. Name
     both files.
   - **Saved states.** `hp/data/states/` is described as "one per integer D from 8 to 60". The files are the nearest
     computed points (8.25, 16.76, 22.30, 53.79, …), so say "about one per unit of D".
   - **`plot_extrapolation.py`.** Regenerating `solver/extrapolation.svg` changes one shaded rectangle's width (107.3
     to 107.4 px). This is harmless, but the figure is not byte-identical to its script's output.
9. **Layout block, entries that are missing** (every listed entry exists):
   - `paper/` (main.tex, Makefile, figures/, dss_large_D.pdf);
   - `src/lib.rs`;
   - `hp/planted_dinf.py` (linked from the README text, but not in Layout);
   - `hp/dd_fits.py`, `hp/pinning.py`, `hp/compare_interp.py`, `hp/compare_old.py`.
10. **Build notes.**
    - **Environment.** "Build and test" says `pip install .` and `python tests/test_bindings.py` without saying that it
      assumes a venv. `build.sh` and `hp/build.sh` install with `pip install --break-system-packages`, so without a venv
      they write into the system site-packages. Both worked in the venv.
    - **"Use" snippet.** It imports `spec_long` and `spec_c`, which live in `solver/`; say "from `solver/`" (or add
      `sys.path`).

## Commands checked

The "From" column is relative to the repository root. "ok" means the command ran and matched the quoted numbers at
their precision.

### Build, test, reproduce (README)

| Command | From | Result | Quoted → got |
|---|---|---|---|
| `cargo test --release` | root | ok | 13 passed (avx512, avx2, baseline exercised); build 6 min with one job |
| `pip install .` | root | ok | rustkernel 0.1.0; `available_backends()` = avx512, avx2, auto:avx512, auto:avx2, baseline |
| `python tests/test_bindings.py` | root | ok | "all backends bitwise equal", "jacobian matches complex step" |
| `./build.sh main` | root | ok | wheel in `dist/`, installed |
| `hp/build.sh` | root | ok | `hpkernel avx512` |
| `./reproduce.sh` (no hpkernel) | root | ok | 12 PASS, 1 SKIP; 730 s (quoted 4–6 min) |
| `./reproduce.sh --full` (with hpkernel) | root | ok | 15 PASS; Δ∞ fits 1.643893745–1.643893752; held-out 7.4×10⁻¹³; K at x = 0.9 = 1.00063 − 0.00024i; \|F₃/F₁\|(60) = 0.08027; 746 s |
| `python3 check_numbers.py --selftest` (README deps only) | root | **failed** | ModuleNotFoundError: flint (fix 2) |
| `python3 check_numbers.py --selftest` (+ python-flint) | root | ok | 34 of 34 pass; 34 of 34 shifts caught |

### Results and Evidence (README)

| Command | From | Result | Quoted → got |
|---|---|---|---|
| `python3 ../formula_all.py` | solver/data | ok (one cell) | Coefficients and poles (−3.24, −11.0, −13.3, −20.4) exact. γ(4) = 0.3739610 and the Values-table γ column match. Δ on 4.3–5: 4.3×10⁻⁹ vs "≤ 4×10⁻⁹" (fix 8) |
| `python3 ../candidates.py` | solver/data | **different number** | The 5 window hits and the χ²/dof column match; held-out rms differs in 7 of 10 rows; held-out set is D > 30.2, not 34–39.5 (fix 5) |
| `python3 ../plot_delta.py` | solver/data | ok | delta.svg byte-identical |
| `python3 ../plot_candidates.py` | solver/data | ok | candidates.svg identical |
| `python3 ../plot_extrapolation.py` | solver/data | ok (1 px) | One rectangle 107.3 vs 107.4 px |
| `python3 ../plot_physics.py`, `../plot_echo.py`, `../plot_spectrum.py`, `../plot_selection.py`, `../plot_gamma.py` | solver/data | ok | All five SVGs byte-identical |
| `python3 plot_heldout.py` | hp | **different figure** | 544 vs 517 held-out points; no `--dmax` (fix 6) |
| `python3 dd_fits_mp.py 10 45 --clean --dmax=60.1` | hp | ok | Drops 26 points; fit on 351. 1/D deg 16: 4.1×10⁻¹⁵, 5.2×10⁻¹³, 1.64389375. 1/D deg 18: 1.8×10⁻¹², 1.64389377. 1/n deg 18: 2.8×10⁻¹², 1.64389377. n^-1/2 deg 20: 2.0×10⁻¹¹, 1.636. D^-1/2 deg 20: 1.6×10⁻¹⁰, 1.547. Δ(100) = 1.7450264845. 477 s |
| `python3 gamma_compare.py` (as quoted) | hp | **silent no-op** | Header only, exit 0 (fix 4) |
| `python3 gamma_compare.py data/gamma_dd.jsonl data/local/gamma_dd.jsonl` | hp | ok | D ≥ 30: 2.4×10⁻⁴, 6.7×10⁻⁵, 1.9×10⁻⁵, 5.5×10⁻⁶, (1.6 ± 2.2)×10⁻⁶; D ≥ 24 row exact; a₁ = 3.03243 |
| `python3 cmp_ecker.py` | hp/compare | **failed** | NameError: `sys` (fix 3) |
| data counts via `dd_fits_mp.load_points` / `clean` | root | ok | 857 points 8.25–61.03; 94 above 60.1; 32 in 60.95–61.03; cleaning drops 26 (≤ 60.1) and 27 (all), 823 kept; 352 cleaned points in 45–59.8 (quoted 351, "can differ by one or two") |
| `python3 largeD_scaling.py`, `python3 largeD_F.py` | hp | ok | Table D = 20–60: 3.480 … 2.962, 0.130 … 0.080, 3.021 … 3.421, 6.162 … 4.416 |

### Large D (README)

| Command | From | Result | Quoted → got |
|---|---|---|---|
| `python3 largeD_expand.py` | solver | ok (text) | P₀, B₂, G₁, S₁, D₁ as printed; runs to third order, README says fourth (fix 8) |
| `python3 largeD_formal_check.py` | solver | ok | residual × n³ constant for n = 50–800, both ω |
| `python3 largeD_fast_transport.py` | solver | ok | 5.6×10⁻², 2.9×10⁻², 1.5×10⁻², 7.5×10⁻³, 3.8×10⁻³ and 0.57, 0.26, 0.13, 0.062, 0.031; gravity factor 41 ("about 40") |
| `python3 largeD_exponent.py` | hp | ok | n⟨G(1)⟩/2 2.038 … 2.964; ⟨F²⟩ 6.162 … 4.416; shift 2.63 … 3.30; \|c_x − G\| ≤ 1.7×10⁻¹⁴ |
| `python3 largeD_linear_late.py` | solver | ok | χ at x = 1/√2: 0.0069 − 1.5703i; g −0.52 − 4.50i; x = 0.8, 0.9 rows match |
| `python3 largeD_dingle_K.py` (via `reproduce.sh --full`) | solver | ok | K/K_pred at x = 0.9 = 1.0006 − 0.0002i |
| `python3 largeD_inner.py` | solver | ok | ≤ 2.0×10⁻¹³ at orders 1–10 |
| `python3 largeD_stokes_C.py` | solver | ok | 1.00014 (μ = 1), 1.0013 (μ = 3) at n = 10⁵; n(ratio − 1) → 13.86 − 7.65i; 2.1–1.27 at n = 18–58 |
| `python3 lateral_np.py check` | solver/inner_stokes | ok | 1.000000000002 − 7×10⁻¹²i at two radii |
| `python3 inner_np.py checklin 9` | solver/inner_stokes | ok | 1.00000000 for harmonics 1–9 |
| `python3 inner_stokes.py linear` | solver/inner_borel | ok | 5.67×10⁻²⁷ ("6×10⁻²⁷") |
| `python3 zero_M1.py` | solver/inner_stokes | ok (text) | A 2.54215, ω/ω∞ 0.77699 at 3 radii; README says 4 radii (fix 8); 374 s |
| `python3 amp_series.py` | solver/inner_amp | ok | 2.3–4.0×10⁻¹⁶ at orders 0–3 |
| `python3 heat_form.py`, `python3 rw_map.py` | solver/inner_pert | ok | all residuals 0 |
| `python3 pert_check2.py` | solver/inner_pert | ok | 1.5×10⁻¹⁷⁷ |
| `python3 pert_check3.py` (defaults) | solver/inner_pert | **different number** | worst 5.9×10⁻²⁴ vs quoted 10⁻¹⁸⁰ (fix 7) |
| `python3 pert_check3.py 45 11` | solver/inner_pert | ok | worst 2.1×10⁻¹⁷⁸; controls λ = 1.9, 2.1 off by 7.55% |
| `python3 eeg20.py`, `tilt.py`, `maxc.py` | solver/eeg20 | ok | Eq. (20) ratio 0.98 at D = 60; tilt 0.69, 0.78, 0.84, 0.88; max-c ratio 0.36 → 0.86 |
| `python3 largeD_Fratios.py` | hp | ok | \|F₃/F₁\| → 0.0552, 0.0549 (degree 1: 0.058); D = 60 ratios 0.080 … 0.129; degree-3 limits 0.055 … 0.072 |

### Proof (PROOF.md, proof/A_certify.md, proof/A_review.md)

| Command | From | Result | Quoted → got |
|---|---|---|---|
| `python3 certify_M1.py` | solver/proof | ok | contraction 0.2918, radius 6.0×10⁻³⁴ (3 s, quoted ~1 min) |
| `python3 residual_poly.py 0.08` | solver/proof | ok | |
| `python3 fixed_point.py` | solver/proof | ok | 2.4×10⁻³⁴, 6.0×10⁻³⁴, 5.9×10⁻³⁶ |
| `python3 krawczyk.py --xi1=0.3` | solver/proof | ok | J_D radius ≤ 2.3×10⁻⁵, derivative ≤ 1.2×10⁻²; 86 s (quoted ~15 s) |
| `python3 validated_int.py krawczyk 1e-9 --point=validated_int_krawczyk_1e-9.json --box=validated_int_krawczyk_1e-9.json` | solver/proof | ok | margins 7.925×10⁻¹⁰, 9.247×10⁻¹⁰ |
| `python3 validated_int.py linear` | solver/proof | ok | R₁ contains 1 |
| `python3 mutation_controls.py` | solver/proof | ok | 8 of 8 as expected; C1 −59, −22; C2 −2.0, −0.35; C5 −1.2×10⁻⁹; C7 −4×10⁻⁸ |
| `python3 lemma6_sector.py --json=lemma6_sector_1e-9.json` | solver/proof | ok | a_eff 0.2036, contraction 0.4903, F 1.04×10⁴⁴, ρ 8.14×10⁴⁴, 5.03×10⁴⁴ < ρ |
| `python3 lemma6_sector.py --mutate=series` | solver/proof | ok | reports False |
| `python3 lemma6_path.py krawczyk --point=lemma6_path_point.json --box=lemma6_path_box_1e-9.json` | solver/proof | ok | margins 8.570×10⁻¹⁰, 9.712×10⁻¹⁰; enclosure [2.54214804761338, 2.54214804761975] |
| `python3 lemma6_jump.py --box=lemma6_path_box_1e-9.json --json=lemma6_jump_1e-9.json` | solver/proof | ok | ρ₆ₐ = 4.221×10⁻³; 291 steps; max radius of B 1.81×10⁻⁶ |
| `python3 lemma6_usys.py` | solver/proof | ok | 1.9×10⁻⁵⁸ |
| `python3 rv_residual_modp.py` (R1) | solver/proof/review | ok | kmins [53,53,53,53,52,52,52] |
| `python3 rv_kernel_bounds.py` (R2) | solver/proof/review | ok | |
| `python3 rv_krawczyk_recheck.py` (R4) | solver/proof/review | ok | margins 7.92×10⁻¹⁰, 9.25×10⁻¹⁰; x̂ ± 1.3×10⁻¹⁰ passes; F3 reported, as documented |
| `python3 rv_fixed_point_audit.py` (R7) | solver/proof/review | ok | |
| `python3 series_check.py` | solver/proof/capd_check | ok | PASS |

### GAMMA.md and matter/*.md

| Command | From | Result | Quoted → got |
|---|---|---|---|
| `heat_form.py`, `rw_map.py`, `pert_check2.py`, `pert_check3.py [45 11]`, `gamma_compare.py` | see above | | |
| `python3 css_bg.py` | solver/matter | **failed** | ω → 3×10⁻⁵ instead of 1.176952722; exit 0 (fix 1) |
| `python3 css_bg.py 0` | solver/matter | **failed** | ω = 2.1755 instead of 1.9154446 (fix 1) |
| `python3 css_table.py` | solver/matter | ok | CSS.md §5 table identical |
| `python3 css_fit.py` | solver/matter | ok | λ∞ − 2 table exact; c₁ 3.598–3.601; ω∞ = 3.19939 ("3.1994") |
| `python3 css_equations.py` | solver/matter | ok | ALL CHECKS PASSED |
| `python3 sigma_css.py flat` | solver/matter | ok | q, U(1), λ match references (518 s) |
| `python3 sigma_eqs.py` | solver/matter | ok | ALL PASS |
| `python3 sigma_equator.py` | solver/matter | ok | |
| `python3 struct_inner.py` | solver/matter | ok | 22 [ok], all ok |
| `python3 struct_css.py check` | solver/matter | ok | |

## Commands skipped

Each run of the commands below would take from tens of minutes to hours on this machine, or would break the
one-process rule.

| Command | Why skipped |
|---|---|
| `hp/dinf_budget.py`, `check_numbers.py --full` | 48 fits at 60 digits on up to 823 points; the 5-fit subset in `reproduce.sh` took 540–595 s here, so about 1.5 h. Its range is partly covered by `reproduce.sh` and `check_numbers.py` |
| `hp/dd_candidates.py 45 --dmax=59.8` | 9 candidates × degrees 14–26 at 60 digits, over an hour here |
| `hp/dd_fits_mp.py 10 37.5 …`, `… --mixed …`, `10 1e9 …` | Same kind as the `10 45` run (8 min); left out for CPU |
| `hp/planted_dinf.py` | Documented 4–6 min per plant × 6 |
| `hp/compare/dd_D5.py`, `hp/scan_hp.py`, `hp/gamma_dd.py`, `hp/gamma_batch.py` | Production double-double solves (30–50 s per point, many points) |
| `solver/inner_stokes/hier.py`, `nested.py`, `full_M.py`, `staged.py`, `continue_M.py`, `zero_M3.py` | Documented hours (M = 5 on 8 threads; M = 7 over 5 h) |
| `solver/inner_amp/amp_flint.py`, `amp_flint_par.py`, `scan_omega.py`, `zero_selection.py`, `arg_crossing.py`, `solver/inner_borel/inner_zero.py` | Documented up to about 10 h per ω; `inner_zero.py` needs a start point and 220 orders |
| `validated_int.py krawczyk 1e-9` (full), `--only=point/box`, `krawczyk 1e-8 --rho=0.2`; `lemma6_path.py point/box` | Point and box runs of about 5 min each on an idle machine, the first with 2 processes; their saved JSON was re-checked instead |
| `lemma6_jump.py … --mutate=swap/coords`, `lemma6_sweep.py` | 2–3 min each; the sweep is unfinished by design |
| `certify_M1.py --pm`, `fixed_point.py --box/--sector/0.08` | Variants of runs that were done |
| `capd_check/build.sh` and `build/capd_check …` | Clones CAPD over the network and builds it in C++ with MPFR; the certificate run alone is about 10 CPU minutes |
| review R3, R5, R6, R8 | 30 s to 4 min each; R1, R2, R4 and R7 were run |
| `struct_eqs.py` (8–9 min), `struct_outer.py`, `struct_symmetry.py`, `sigma_runs.py`, `sigma_checks.py`, `css_scan/lateral/gamma/count/locate/inner/scaling.py` | Minutes to hours per point or box |
| `./build.sh all` | Two more full LTO builds with `-C target-cpu=native` |
| `make` in `paper/` | Needs pdflatex with revtex4-2, which is not installed |
| `cmp_ecker.py` past its NameError | Needs a clone of the Ecker data repository, which this session cannot access |

## Links, figures, Layout

- **Links.** A GitHub-slug checker read README.md, PROOF.md, GAMMA.md, PARALLEL.md, AGENTS.md, matter/*.md, proof/*.md,
  notes/*.md, solver/README.md and solver/CONTEXT.md: **0 broken** among 368 relative links and 222 anchors. In the
  README, all 11 footnotes are both defined and used.
- **File references.** Every file name in backticks or code blocks exists (relative to the directory each command
  runs in). The unresolved names in proof/FORMAL.md and notes/BOOTLOOPS.md refer to external repositories and are
  linked as such.
- **Figures.** All 9 SVGs that README embeds exist, and all 6 PDFs that `paper/main.tex` includes exist. Seven SVGs
  regenerate byte-identically; `extrapolation.svg` differs by 0.1 px and `heldout.svg` differs in content (fix 6).
- **Layout block.** Every listed path exists. The unlisted ones are given in fix 9.

## Dependencies

- **Enough for the core:** numpy, scipy, mpmath and sympy (the README's list) are enough for `reproduce.sh`, which
  reports flint and hpkernel as optional SKIPs.
- **python-flint is needed but listed only for `solver/inner_borel/`.** It is required for `check_numbers.py` (which
  crashes without it), all of `solver/proof/`, `solver/proof/review/` and `solver/inner_amp/amp_flint*.py` (fix 2).
- **Rust and maturin** are needed for `hp/build.sh` and `pip install .`; the README says so.
- **Not needed: matplotlib.** No documented script imports it (the plots are written as SVG by hand); only
  `solver/plot_convergence.py` and `solver/scratch/*` do.
- **Not stated in the README:**
  - CMake, a C++ compiler, MPFR, git and network access, needed by `capd_check/build.sh`. This is stated in
    proof/A_certify.md §10.5, but not in the README.
  - The external Ecker data clone, needed by `cmp_ecker.py`.

## How to rerun this

1. Clone the repository into an empty directory.
2. Create a venv and run `pip install numpy scipy mpmath sympy python-flint`.
3. Put the venv first on PATH and run `hp/build.sh` and `pip install .`.
4. Run each command above from the directory given, under `nice -n 10`.

The machine's load roughly doubles to quintuples the quoted times.
