# Notes for agents working in this repository

Conventions for the Claude sessions (cloud and local) that run computations here. The physics and software are in
[README.md](README.md); how the computations parallelize is in [PARALLEL.md](PARALLEL.md).

- **Inner amplitude series:** use `solver/inner_amp/amp_flint_par.py` (and `scan_omega.py`, which calls it), not
  `amp_flint.py`. It gives the same digits, 26× faster on one process, and `--procs N` adds worker processes (spawn,
  Windows-safe; threads do not help because python-flint holds the GIL). For K ≥ 13 use J ≥ 1000 and compare two J:
  at J = 500, σ₆ is wrong and σ₇, σ₈ blow up. `amp_flint.py` stays as the serial reference for checks.
- **Inner selection with M harmonics:** use the nested solve, `solver/inner_stokes/full_M.py` (or `nested.py` for
  M = 3), not a joint solve of all conditions: the joint solve is ill-conditioned (about 10¹⁰) and stops on false
  near-zeros.
- **Double-double solver (`hp/`):** rebuild with `hp/build.sh` after pulling changes to `hp/src`. γ(D) comes from
  `hp/gamma_dd.py` / `hp/gamma_batch.py --procs N`.
- **Thread budget on the local 8-thread machine:** 4–6 threads at below-normal priority; one process per thread.
- **Coordination:** the cloud and local sessions talk through `comms.txt` (append only; pull before writing; commit only
  `comms.txt` in comms commits; header `== <date time PDT> cloud|local`). Agents making code changes work in a git
  worktree, not the main checkout.
