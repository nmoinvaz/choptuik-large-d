# Notes for agents working in this repository

Conventions for the Claude sessions (cloud and local) that run computations here. The physics and software are in
[README.md](README.md); how the computations parallelize is in [PARALLEL.md](PARALLEL.md).

- **Inner amplitude series:** use `solver/inner_amp/amp_flint_par.py` (and `scan_omega.py`, which calls it), not
  `amp_flint.py`. It gives the same digits, 26× faster on one process, and `--procs N` adds worker processes (spawn,
  Windows-safe; threads do not help because python-flint holds the GIL). For anything past σ₄ use `--fit phys` at
  640 bits (late-order fits with every neighbour family |ν′| ≤ k; the default `legacy` basis, ±2, ±4, ±6 only, biases σ₅
  and wrecks σ₆–σ₈), e.g. `amp_flint_par.py 1.0 15 750 640 --procs 6 --fit phys --mf 6:12`, and compare two J.
  Check `sigma_err` in the JSON. `amp_flint.py` stays as the serial reference for checks.
- **Inner selection with M harmonics:** use `solver/inner_stokes/hier.py` (hierarchical solve; required for M ≥ 5,
  where `full_M.py`'s homotopy stalls), or `nested.py` for M = 3. Never a joint solve of all conditions: it is
  ill-conditioned (about 10¹⁰) and stops on false near-zeros. hier.py uses 2 processes at most (`--procs 2`), so fill
  the thread budget with several jobs rather than one job with more processes. For M ≥ 7 add `--schur` (without it one
  ulp of B₇ is a 0.3–3% floor on ω); `--verbose=2` prints every level's solves. Killing a hier.py main process leaves its
  spawn workers running: on Windows kill the tree, `taskkill /T /F /PID <pid>`.
- **Double-double solver (`hp/`):** rebuild with `hp/build.sh` after pulling changes to `hp/src`. γ(D) comes from
  `hp/gamma_dd.py` / `hp/gamma_batch.py --procs N`.
- **Thread budget on the local 8-thread machine:** 4–6 threads at below-normal priority; one process per thread. It has 4
  physical cores, so with 6 busy threads a job runs about 4× slower than on 2 cloud cores; scale cloud estimates.
- **python-flint pitfall (0.9.0):** `arb(0, r)**k` returns NaN for a ball that contains 0 with r > 0 and k = 2 or 3,
  while `x*x` is correct. Write powers of balls that may contain 0 as products. (`solver/proof/` is unaffected: its
  powers are of exact positive balls. Found by the BootLoops review, notes/BOOTLOOPS.md.)
- **Quoted numbers:** after changing a number in the README, the paper or the proof notes, run
  `python3 check_numbers.py --selftest`; when a new headline number has a machine-readable source, add a check for it.
- **Novelty claims:** no sentence saying a result is new, "not found in print", or "first" goes into the README, the
  paper or the proof notes until the closest prior work has actually been read (not recalled) and recorded in
  notes/LIT_REVIEW.md, with what it does and does not contain. Credit prior equations and observations where they are
  used, not only in the references.
- **Coordination:** the cloud and local sessions talk through `comms.txt` (append only; pull before writing; commit only
  `comms.txt` in comms commits; header `== <date time PDT> cloud|local`). Agents making code changes work in a git
  worktree, not the main checkout.
