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
  the thread budget with several jobs rather than one job with more processes.
- **Double-double solver (`hp/`):** rebuild with `hp/build.sh` after pulling changes to `hp/src`. γ(D) comes from
  `hp/gamma_dd.py` / `hp/gamma_batch.py --procs N`.
- **Thread budget on the local 8-thread machine:** 4–6 threads at below-normal priority; one process per thread.
- **Coordination:** the cloud and local sessions talk through `comms.txt` (append only; pull before writing; commit only
  `comms.txt` in comms commits; header `== <date time PDT> cloud|local`). Agents making code changes work in a git
  worktree, not the main checkout.
