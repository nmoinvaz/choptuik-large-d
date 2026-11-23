# Lean 4 lemmas of choptuik-large-d

The candidates are listed in [../proof/FORMAL.md](../proof/FORMAL.md) §3; the tasks are the repository's Lean issues
(the first is #1077). Lean and Mathlib v4.34.1 (`lean-toolchain`, `lakefile.toml`, `lake-manifest.json`).

    cd lean && lake exe cache get && lake build

`warningAsError = true`, so a proof left as `sorry` fails the build.

| file | statement | FORMAL.md row | axioms |
|---|---|---|---|
| `DSS/InnerConstant.lean` | 2^(-s-1/2) / (Γ((1+s)/2) Γ((2+s)/2)) = 1 / (√(2π) Γ(1+s)) for every s : ℂ | 1 | propext, Classical.choice, Quot.sound |
