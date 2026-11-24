# The mode determinant near D 3: where its sign is lost, and a matched determinant (#1083)

Status: design and pass criteria (written before the prototype's checks); prototype in progress. Branch
`wip/mode-horizon-1083`. The measurements this note starts from are #1078's (dh370); the status words are AGENTS.md's.

## Pass criteria (fixed before the prototype's checks run)

Notation: D_out(λ) is the present determinant, `Solver.mode_det` on the state's own layout (every segment integrated
outward, `match_node` = K − 1). D_m(λ) is the matched determinant of section 4, the same kernel call on the layout whose
segments past node m are integrated inward (`match_node` = m), m the last node before the sonic valley. Both are
evaluated on the state's plan `Solver.plan(z, 1e-32, 0.05)`, as `hp/lowD_gamma.py` does.

1. **The four λ of comment 7120204793** at D 3.044375, M′ 96 (state blob 6a0f83d2): λ = 5.1408197887857,
   5.140819788786, 5.1408197887863025 and 5.1408197887866. With f_i = sign_i · exp(ln|D_m(λ_i)| − ln|D_m(λ_1)|),
   the four f_i lie on one straight line in λ to within 10⁻⁶ of max |f_i| (a straight line over 9×10⁻¹³ in λ is what a
   smooth function with at most one simple root gives there), and the double-double LU of D_m has its pivot ratio
   min |u_kk| / max |u_kk| above 10⁻²⁰. The present D_out fails both there (ln|det| moves by 4.0, 3.1 and 4.7 between
   neighbours, and the sign flips twice).
2. **D 3.15**, M′ 96 (state blob 9e683a14, its own layout and map): the root of D_m near λ 4.3977 equals the root of
   D_out, each found by `lowD_gamma.find_lam` with xtol 10⁻¹¹ from λ₀ 4.397684106690, to within 10⁻⁸ in λ, and both
   searches see the same number of sign changes in their first bracket.
3. **A planted root.** At D 3.044375 (state 6a0f83d2), on the matched layout, one horizon column j of the mode matrix
   is replaced by its value at λ minus its value at λ* (the column of a_h at the τ point of the valley, λ* = 5.0), so
   that the matrix has a zero column, and its determinant an exact root, at λ*. The prototype's root search on the
   planted determinant returns λ* to within 10⁻⁹. Controls, each of which must fail the same test: (i) the column
   replaced by its value at λ minus its value at λ* + 10⁻⁶ (the root moves by about 10⁻⁶); (ii) the same plant on the
   outward layout, whose sign at λ* ± 10⁻⁶ ... ± 10⁻³ must not show one clean change.
4. **The identity of section 4.** Where both are resolved (D 3.15 at λ 4.3 and 4.5; D 3.044375 at λ 1.0 and 1.1), the
   signs of D_out and D_m agree, and the change of ln|D_out| − ln|D_m| between the two λ equals the Liouville trace
   integral of section 4 over the segments past m, computed from the background alone, to within 10⁻⁶ of that change.
