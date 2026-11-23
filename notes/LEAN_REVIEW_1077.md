# LEAN_REVIEW_1077: independent review of the Lean Krawczyk step (#1077)

A review of branch `feat/lean-krawczyk` at 88577a3d: the Lake project `lean/`, its Krawczyk theorems, the four
instances for Theorem A at M = 1, 3 and 5, their controls, and the exporter `lean/tools/export_krawczyk.py`. Written
2026-11-23 by Claude (Anthropic) under Nathan Moinvaziri's direction, as a separate agent that had not seen how the
work was made. It was judged from the code and the sources only. Status words as in AGENTS.md.

The review's own files are in `lean/review/` on branch `review/lean-1077`. No file under review was changed. Fixes
are proposed as diffs in §4.

## Verdict

**Sound, with changes needed before merging.** No error was found. The theorems are proved from the three standard
axioms, the four instances are the saved certificate data bit for bit, and their hypotheses can be met. The changes
needed are to what guards that faithfulness (it is the exporter alone, and one of its guards is weak), to one sentence
that overstates what Lean checks at M = 3 and 5, and to the record of versions.

| question | verdict |
|---|---|
| 1. Do the statements say what FORMAL.md §4 says? | Yes for 2 unknowns. For n unknowns the hypotheses are on Y g and Y Dg, which §4 does not yet say. No hypothesis is vacuous in the four instances (*proved* in Lean, findings 1–3). `check` itself does not guard against a vacuous instance (findings 2, 4) |
| 2. Is the exporter faithful? | Yes. Every rational of the four instances was reproduced without the exporter, by two routes (§2). The Lean boxes lie inside the integrated ones |
| 3. Are the controls meaningful? | The `*Wide` controls fail for the stated reason, and are weak. No check on the branch would catch a transposition in the exporter (findings 5, 6). One was written and run: 23 mutations fire, and a planted map goes through the exporter |
| 4. Build | `lake build` passes (8936 jobs), no `sorry`, and every theorem depends only on propext, Classical.choice, Quot.sound |
| 5. Anything overstated? | One sentence of the issue's closing comment (finding 7), and smaller points of record (findings 8–13) |

## 1. Findings

Severity is one of *error*, *needs change*, *minor*. There is no *error*.

1. **The statements match FORMAL.md §4 for 2 unknowns** (no change needed). `krawczyk_of_check` has the signature of
   §5's sketch. The box is `closedBall` in the sup norm of `Fin 2 → ℝ`. `EnclOK` asks for g(x̂) at the exact rational
   centre and for Dg(x) at every x of the box, entry (i, j) being ∂gᵢ/∂xⱼ, which is the order of `krawczyk.py`'s JX
   and of CAPD's `DF_over_X`. The hypothesis on g is `HasFDerivWithinAt` at each point of the box, which is weaker than
   C¹, so the theorem is stronger than §4's sentence. Since `check` requires h > 0, the box has interior and the
   derivative within it is unique: Dg cannot be chosen freely. The conclusion is existence, uniqueness in the box and
   det Dg ≠ 0 at the zero. The proof shows det Dg ≠ 0 on the whole box and states less.

2. **For n unknowns `check` does not read Y** (*needs change*: documentation and the exporter, §4.1). `DataN.check`
   and `BoxData.check` use only `Yg`, `YJ` and `w`. Y appears only in `EnclOK`, which says that `Yg` and `YJ` enclose
   the products with *this* Y. So Lean cannot tell whether the Y in the file is the Y the products were formed with.
   The kernel proves `BoxData.check { M3cert with Y := 0 } = true` and the same with Y transposed
   (`review/Axioms.lean`, `review/Mutations.lean`). With Y = 0 `EnclOK` is false for every map and the instance's
   theorem would be vacuous. This is a property of the design, which FORMAL.md §4 does not describe: its statement is
   on g(x̂) and J(X), with Y read by the test. The product form is needed (the hull of Dg is too wide at M ≥ 3), so
   the link has to be guarded outside Lean, and it has to be said.

3. **None of the four instances is vacuous** (*proved*, `review/NonVacuous.lean`; proposed for `DSS/`, §4.2). When no
   interval is empty, the affine map g(x) = g_lo + J_lo (x − x̂) meets the hypotheses of a 2-unknown instance. For n
   unknowns the same holds with g(x) = Y⁻¹(Yg_lo + YJ_lo (x − x̂)) when det Y ≠ 0. The kernel decides that the
   intervals of all four instances are non-empty and that det Y ≠ 0 for M3cert and M5cert, so
   `M1capd_nonvacuous`, `M1arb_nonvacuous`, `M3cert_nonvacuous` and `M5cert_nonvacuous` hold, each from the three
   standard axioms. The determinant check fires on a matrix of rank one (`detY_fires`).

4. **`check` does not see an empty interval** (*minor*). With g's first interval of M1capd inverted to ⟨hi, lo⟩ the
   kernel still proves `check = true` (`check_blind_to_empty`), and `EnclOK` is then false. The exporter asserts
   lo ≤ hi (`lean_I`), so its files are safe. The theorems of finding 3 close this in Lean.

5. **For 2 unknowns `check` passes on J and Y transposed together** (*needs change*: a check against the source,
   §4.3). `check` catches Y transposed, Y negated, J transposed and the rows of J swapped, each alone (decided false).
   It passes when J and Y are both transposed (`capd_J_and_Y_transposed`, `arb_J_and_Y_transposed`, decided true),
   and the exporter computes Y from the J it writes. So an exporter that transposed J would produce a file that
   builds, about a different map. Nothing on the branch compares the Lean data with the source except the exporter.

6. **The `*Wide` controls are valid and weak** (*minor*). Each fails for the stated reason, the widened entry
   [0][0]. They show that `check` is not constantly true. They do not test the g term, the other rows at M ≥ 3, or
   the transcription. The table gives each row's value, which must be below 1 (my exact mirror of `check`, which
   agrees with the kernel on the 8 files and on the mutations of `review/Mutations.lean`).

   | instance | rows of the clean instance | rows of its `Wide` control | [0][0] widened: fails from |
   |---|---|---|---|
   | M1capd | 0.1283, 0.0167 | 1293, 2502 | ×394 |
   | M1arb | 0.2075, 0.0753 | 12020, 24760 | ×38 |
   | M3cert | 0.0056, 0.0030, 0.2223, 0.2225 | 1843, then unchanged | ×541 |
   | M5cert | 0.0037, 0.0017, 0.1110, 0.1111, 0.0885, 0.0886 | 2.017, then unchanged | ×4.95×10⁵ |

   M5certWide passes the threshold by a factor 2. One minus the clean rows are the margins that A_certify.md, A_M.md
   and A_M2.md quote (0.994, 0.997, 0.778, 0.778 at M = 3), so the Lean test and the Python tests agree to the digits
   shown.

7. **"Its logic and its arithmetic no longer rest on Python" is overstated for M = 3 and 5** (*needs change*:
   wording, §4.4). For M = 1 Lean forms Y g and I − Y J itself from the enclosures of g and J. For M = 3 and 5 the
   products Y l (c + Q q) and Y l (cV + QV qV) are formed in Python (arb, 768 bits) and enter Lean as hypotheses. For
   M = 5 so is the second-order step: the enclosure G1 + Σₖ G2 Xₖ of Y Dgt over the box, and the mean value argument
   that it contains Y Dgt(x), are in `certify_M2.py` and A_M2.md, not in Lean. Lean checks the final row test on those
   enclosures and proves that the test implies the conclusion. The commit title "Theorem A(M = 5)'s second-order
   Krawczyk step, checked by the kernel" has the same overstatement.

8. **The exporter's guard on Y at M = 3 compares printed strings** (*needs change*, §4.1). `certM` recomputes Y and
   asserts that it reproduces `krawczyk_test`'s Y g by `.equal` if arb has it, else by `str`. python-flint 0.9.0's
   `arb` has no `equal`, and the printed Y g[0] is `[+/- 1.25e-20]`. `review/guard_probe.py` multiplies Y[0][0] by
   1 + 2⁻ᴷ: the string guard still passes at K = 30 and fails only from K = 20. The Y of record is right all the
   same: recomputed by the same statements it gives `krawczyk_test`'s Y g and Y Dgt back with midpoints and radii
   equal bit for bit, and at M = 5 `assemble` returns Y itself.

9. **The generated files do not record the python-flint and FLINT versions** (*needs change*, AGENTS.md "Record the
   python-flint and FLINT versions with every certificate run", §4.5). M1arb, M3cert and M5cert hold digits that arb
   computed (320 and 768 bits). On the Mac (python-flint 0.9.0, FLINT 3.6.0) the exporter regenerates all 8 files
   byte for byte. Another FLINT version may give other last digits, and the file would still be valid.

10. **The regeneration commands are not documented** (*minor*, §4.5). The exporter's docstring gives `capd` and
    `arb` and not `certM` and `certM2`. The header's path comes from `path.split('choptuik-large-d/')[-1]`, so it is
    the repository path only when the script is given a repository-relative path, or runs in a clone of that name
    that is not a worktree under it.

11. **The two M = 1 instances are about two maps** (*minor*). M1capd encloses F = (Re J_D, Im J_D) and M1arb
    encloses g = κ J_D. They have the same zeros, and A_certify.md §10 says so. The Lean headers and the README do
    not.

12. **The chart of M5cert is the second-order one** (*minor*). The M5cert header says "the chart around p^" and the
    issue's comment cites A_M.md §2 for both. At M = 5 it is `certify_M2.py`'s
    p(x) = p̂ + T x + ½(0, Γ[x_o, x_o]) of A_M2.md. In both, p̂ is the 256-bit binary midpoint of the file's decimals
    (`exact_p`), not the decimals. That a zero of gt in the chart box is a zero of g in p(X), and that Dg is
    invertible with Dgt, is outside Lean (the chart is a shear with unit triangular derivative).

13. **Open items that the last comment does not list** (*minor*). The first comment moved the Palomar layout
    (`Challenge.lean`, `Solution.lean`, `comparator.json`) and leanprover/comparator to step 1. Step 1 is done and they
    are not there, and the last comment's list does not name them. FORMAL.md §3 and §5 still name Mathlib v4.33.1.
    The last comment does list the planted control through the exporter and the update of FORMAL.md and A_certify.md.

14. **Repetition** (*minor*, no change asked). `krawczyk_of_check` and `krawczykN_of_check` share most of their
    proof. The 2-unknown theorem follows from the n-unknown one by forming Y g and Y J in Lean, which is also the way
    to make `check` read Y at M ≥ 3: take the Lohner sets (c, Q, q) and (cV, QV, qV) as the data and form the products
    in exact rationals in the kernel. `review/reproduce.py` does this in Python in well under a second, and its exact
    intervals are 10⁻⁷ narrower than arb's. Then finding 2 would go away and finding 7's sentence would be true at
    M = 3.

## 2. The reproduction

`lean/review/reproduce.py` reads the generated Lean files back with its own parser and compares them with the saved
data. It shares no code with the exporter.

| route | instances | what it uses | what it shows |
|---|---|---|---|
| exact | M1capd | the JSON, hex doubles parsed by hand | xhat, g, J equal the file's; 10⁻⁹ ≤ CAPD's h (smaller by 6.2×10⁻¹⁷, relative); x̂ inside CAPD's enclosure of it |
| exact | M3cert, M5cert | the JSON and the Y of the Lean file, in `fractions.Fraction`; no python-flint, no `certify_M*.py` | each of the 20 and 42 Lean intervals contains the exact interval value of the same expression, and is at most 8.1×10⁻⁸ and 1.2×10⁻⁷ wider (relative); w is (h, h, w₃, w₃, …) and at most the stored box; det Y ≠ 0 |
| arb | M1arb, M3cert, M5cert | the certificate's own functions (`krawczyk_assemble`, `krawczyk_test`, `assemble`), and my conversion of a ball to its ends | every rational of g, J, Yg, YJ and Y equals the ball's exact end |

Result: `reproduction: PASS`, exit 0.

What the routes share. The exact route shares with the exporter the JSON files and Python's `fractions`; the layout
of the state (D on the odd harmonics first) is taken from A_M.md, as `certify_M_recheck.py` does. The arb route shares
with the exporter the certificate code, python-flint and `man_exp`; its conversion is checked by embedding each end
back at 8192 bits. The mirror of `check` shares only the text of the Lean definitions.

Its limits. The exact route sees a relative change of Y[0][0] of 2⁻⁴⁰ and not of 2⁻⁶⁰, since arb's balls are about
10⁻⁷ wider than the exact intervals. The arb route compares Y bit for bit.

**Are the Lean boxes inside the integrated ones?** Yes, from the code. M1arb: the box run covers
arb(A) + arb(0, arb(h)), whose radius is at least the decimal h. M1capd: CAPD's set covers the decimal centre
± (h + the width of its enclosure of x̂), with h the double above 10⁻⁹. M3, M5: the run's `dx`, `pdx` are
arb(0, arb(w)) for the same decimal strings, and the stored radii are 0.25% to 1% above the Lean w.

## 3. What was run

On the Mac: Lean 4.34.1, Mathlib v4.34.1 (rev d13f23b7), Python 3.14.7, python-flint 0.9.0, FLINT 3.6.0,
numpy 2.5.4. Commands from the repository root unless `lean/` is named.

| command | result |
|---|---|
| `cd lean && lake exe cache get && lake build` | passes, 8936 jobs |
| `grep` for `sorry`, `admit`, `axiom`, `native_decide`, `unsafe`, `set_option` in `lean/DSS` | none |
| `lake env lean -DwarningAsError=true` on a file with `sorry` | exit 1, so the README's sentence on `sorry` holds |
| `cd lean && lake env lean review/Axioms.lean` | 16 `#print axioms`: the three standard axioms each; intervals non-empty; the two blindness theorems |
| `cd lean && lake env lean review/NonVacuous.lean` | compiles; 4 non-vacuity theorems, three standard axioms each |
| `cd lean && lake env lean review/Mutations.lean` | compiles: 22 mutated instances with the stated verdicts |
| `cd lean && lake env lean review/MustFail.lean` | exit 1: `decide` fails on each of three false verdicts, so a stated verdict can fail |
| `python lean/review/reproduce.py mirror` | agrees with the Lean file's theorem on all 8 files |
| `python lean/review/reproduce.py all` | PASS |
| `python lean/review/reproduce.py controls` | 23 mutations, all fire |
| `python lean/review/planted.py WORKDIR` | PASS: in compiles, out and transposed do not |
| `python lean/review/guard_probe.py` | the table of finding 8 |
| `bash lean/review/regen_check.sh WORKDIR python` (the exporter on the 4 sources, with and without `--widen=1000000`) | the 8 files of `lean/DSS`, byte for byte |

`python` is the repository's interpreter with python-flint (on the Mac `/Users/nathan/venv-314/bin/python`).

**The planted control** (`review/planted.py`). The affine map F(x) = A (x − x\*), A = [[1, 100], [0, 1]], goes
through the exporter's `capd` path unmodified, as a CAPD-format file, and the Lean file is compiled. The criteria
were written in the script before the first run. With x\* at (h/2, −h/4) from the centre the exporter's
`check = true` theorem compiles. With x\* at 2h it does not. With DF transposed in the file, which is what a
transposing exporter would hand to Lean, it does not. Only the `capd` path has such a control; `arb`, `certM` and
`certM2` need a synthetic saved run and have none.

## 4. Proposed changes

### 4.1 The exporter writes the products formed with the Y it writes (findings 2, 8)

```diff
--- a/lean/tools/export_krawczyk.py
+++ b/lean/tools/export_krawczyk.py
@@ def certM(path, name, out, widen):
     Y = arb_mat([[arb(Yb[i, j].mid()) for j in range(m)] for i in range(m)])
-    Yg2 = CM.apply_set(Y * ell, c, Q, q)
-    assert all(Yg2[i].equal(res["Yg"][i]) if hasattr(Yg2[i], "equal") else str(Yg2[i]) == str(res["Yg"][i])
-               for i in range(m)), "Y recomputed does not reproduce krawczyk_test's Y g"
+    Yg2, YJ2 = CM.apply_set(Y * ell, c, Q, q), CM.apply_var(Y * ell, varb)
+    same = lambda a, b: a.mid().man_exp() == b.mid().man_exp() and a.rad().man_exp() == b.rad().man_exp()
+    assert all(same(Yg2[i], res["Yg"][i]) and all(same(YJ2[i][j], res["YJ"][i][j]) for j in range(m))
+               for i in range(m)), "Y recomputed does not reproduce krawczyk_test's Y g and Y Dgt"
@@
-    Yg = [list(arbq(res["Yg"][i])) for i in range(m)]
-    YJ = [[list(arbq(res["YJ"][i][j])) for j in range(m)] for i in range(m)]
+    Yg = [list(arbq(Yg2[i])) for i in range(m)]             # the products with the Y that is written
+    YJ = [[list(arbq(YJ2[i][j])) for j in range(m)] for i in range(m)]
```

`reproduce.py`'s arb route makes this comparison and it passes. And in the header of `DSS/KrawczykN.lean` and in the
README: "`check` does not read Y. That `Yg` and `YJ` enclose the products with this Y is part of `EnclOK`."

### 4.2 Non-vacuity in `DSS/` (findings 3, 4)

Move `exists_map` and `exists_mapN` of `review/NonVacuous.lean` into `DSS/Krawczyk.lean` and `DSS/KrawczykN.lean`,
and let the exporter write, after each `_unique_simple_zero`, the theorem `<name>_nonvacuous` as in that file.

### 4.3 A check against the source in the acceptance (findings 5, 6)

Keep `review/reproduce.py` (`all`, `controls`), `review/Mutations.lean` and `review/planted.py` with the project,
and run them when an instance is regenerated. Add planted controls for `arb`, `certM` and `certM2`.

### 4.4 Wording (findings 7, 11, 12)

For the README, after "outside Lean.":

> For M = 3 and 5 the enclosures are those of the preconditioned products Y g(0) and Y Dg over the chart box, formed
> by `certify_M.py` and `certify_M2.py` in arb (at M = 5 with the second-order mean value form of A_M2.md), so that
> step is part of `EnclOK` too. Lean checks the final row test and proves that it implies the conclusion.

And in the headers: M1capd "the map is F = (Re J_D, Im J_D)"; M1arb "the map is g = κ J_D"; M5cert "the second-order
chart of A_M2.md, p(x) = p̂ + T x + ½(0, Γ[x_o, x_o])".

### 4.5 Versions and commands (findings 9, 10)

```diff
--- a/lean/tools/export_krawczyk.py
+++ b/lean/tools/export_krawczyk.py
@@ def certM(path, name, out, widen):
+    import flint
@@
-         f"    {[round(r, 3) for r in res['rel_margins']]}). Coordinates: the chart around p^ (A_M.md §2), centre 0. -/", "",
+         f"    {[round(r, 3) for r in res['rel_margins']]}). Coordinates: the chart around p^ (A_M.md §2), centre 0.",
+         f"    python-flint {flint.__version__}, FLINT {flint.__FLINT_VERSION__}. -/", "",
```

The same line in `arbrun` and `emit`. In the docstring, the two missing commands, run from the repository root:

```
python3 lean/tools/export_krawczyk.py certM solver/proof/data/A_M_box_M3_1e-10.json M3cert lean/DSS/M3cert.lean
python3 lean/tools/export_krawczyk.py certM2 solver/proof/data/A_M2_box_M5_1e-10.json M5cert lean/DSS/M5cert.lean
```

## 5. What each instance proves, as checked here

*Certified*, with the trusted code named: for each instance, Lean's kernel (Lean 4.34.1, Mathlib v4.34.1) checks that
every map on the box that is differentiable there and whose enclosures are the file's has exactly one zero in the
box, with an invertible Jacobian at it. That the enclosures hold for the selection map is `EnclOK`, which rests on
`fixed_point.py`, `validated_int.py`, `certify_M.py`, `certify_M2.py` and arb, or on CAPD, as A_certify.md, A_M.md and
A_M2.md state. At M = 3 and 5, `EnclOK` includes the products with Y and the chart (finding 7, finding 12).
