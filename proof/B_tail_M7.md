# Gap (b) continued: the M = 7 zero and the tail lemma

This note continues [B_tail.md](B_tail.md) (gap (b) of Conjecture 1, [PROOF.md](../PROOF.md) §4 A.3 and stage 4 of §7). It
computes the seven-harmonic zero, updates the truncation-sequence analysis with it, states the tail lemma for the
jump-conditioned truncation precisely, proves the parts of it that can be proved by hand, estimates every constant from
the data, and names the one inequality that needs a new idea. Written by Claude (Anthropic) under Nathan Moinvaziri's
direction; not reviewed by a mathematician. Status words as in AGENTS.md: *proved* (by hand), *numerical evidence*,
*conjecture*, *conditional* (hypothesis named).

Scripts (all new; nothing existing was changed):

- `solver/inner_stokes/inner_nb.py`: a compiled (numba) backend for `inner_np.Inner`: the same series, right-hand side
  and DOP853 integrator (scipy's algorithm transcribed), 15–20× faster at M = 7.
- `solver/inner_stokes/hier_nb.py`: `hier.py` unchanged, run on that backend, with a `--pretune` start and a lazy
  residual (only the radii whose entries `hier.py` reads are integrated while a level approaches).
- `solver/proof/trunc_seq_M7.py`: the truncation sequence, ratios, coupling and limit estimates.
- `solver/proof/amp_trunc_hiermodel.py`: the amplitude expansion of `hier.py`'s own M-harmonic model (§4).

PLACEHOLDER-REST
