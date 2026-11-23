# Lean 4 lemmas of choptuik-large-d

The candidates are listed in [../proof/FORMAL.md](../proof/FORMAL.md) §3; the tasks are the repository's Lean issues
(the first is #1077). Lean and Mathlib v4.34.1 (`lean-toolchain`, `lakefile.toml`, `lake-manifest.json`).

    cd lean && lake exe cache get && lake build

`warningAsError = true`, so a proof left as `sorry` fails the build.

| file | statement | FORMAL.md row | axioms |
|---|---|---|---|
| `DSS/InnerConstant.lean` | 2^(-s-1/2) / (Γ((1+s)/2) Γ((2+s)/2)) = 1 / (√(2π) Γ(1+s)) for every s : ℂ | 1 | propext, Classical.choice, Quot.sound |
| `DSS/Krawczyk.lean` | Krawczyk's test in contraction form, 2 unknowns, equal half-widths, on enclosures of g(x̂) and Dg | 17 | same |
| `DSS/KrawczykN.lean` | the same in n unknowns on enclosures of Y g(x̂) and Y Dg; boxes with unequal half-widths (`krawczyk_box`) | 17, 24 | same |
| `DSS/M1capd.lean`, `DSS/M1arb.lean` | Theorem A(M = 1)'s step (iv) on CAPD's and on arb's enclosures, by `decide +kernel` | 17 | same |
| `DSS/M3cert.lean`, `DSS/M5cert.lean` | Theorem A(M = 3) and A(M = 5)'s Krawczyk tests of record, by `decide +kernel` | 24 | same |
| `*Wide.lean` | each instance with one Jacobian entry widened 10⁶ times: the check is decided false (controls) | | same |

Each instance proves: every C¹ map whose enclosures are the saved ones has exactly one zero in the box, and its
Jacobian is invertible there. That the enclosures belong to the selection map (`EnclOK`) is the validated integration
(arb, CAPD) outside Lean. `tools/export_krawczyk.py` writes the instances from the saved runs as exact rationals.

For M = 3 and 5 the enclosures are those of the preconditioned products Y g(0) and Y Dg over the chart box, formed by
`certify_M.py` and `certify_M2.py` in arb (at M = 5 with the second-order chart of A_M2.md), so that step is part of
`EnclOK` too. Lean checks the final row test and proves that it implies the conclusion. `check` does not read Y; that
`Yg` and `YJ` are the products with the file's Y is part of `EnclOK`, and the exporter asserts it exactly
(`DSS/NonVacuous.lean` proves det Y ≠ 0 and that each instance's hypotheses can be met).

Checks (the independent review of #1077, `notes/LEAN_REVIEW_1077.md`), from the repository root unless stated:

    cd lean && python3 review/reproduce.py all            # each instance from its source by two routes; PASS
    cd lean && python3 review/reproduce.py controls       # 23 mutations of Y, YJ, J, g, w: all fire
    bash lean/review/regen_check.sh WORKDIR [PYTHON]       # the exporter regenerates the 8 files byte for byte
    cd lean/review && python3 planted.py WORKDIR          # a planted affine map through the capd path
    cd lean && lake env lean review/Mutations.lean        # 22 mutated instances decided by the kernel (exit 0)
    cd lean && lake env lean review/MustFail.lean         # must fail
