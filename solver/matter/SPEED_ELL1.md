# SPEED_ELL1: faster l = 1 root readings for #1025

Three tools on branch `wip/ell1-speed`, all in `solver/matter/`. They need the venv with multidouble's `dd_logdet`
(`venv-1025dd` on the Mac; `venv-314`'s multidouble lacks it). Nothing in `dss_ell.py`, `ell_ddm.py` or `ddnp.py`
changed, so the runners' pins of those files still hold.

| tool | replaces | what it does |
|---|---|---|
| `ell_fastm.py` | `DSSEll.mode_matrix` (float64, blocks 0-12) | the same M(lambda) by the variational RK4 map |
| `ell_fastdd.py` | `ell_ddm.DDEll.mode_matrix(only=[12])` | the same double-double block 12 by stage matrices |
| `d18_secant.py` | a finer grid of `d18_hybsign.py` points | Brent refinement of a bracketed root on the hybrid sign |

## ell_fastm: the float64 mode matrix

`DSSEll.mode_matrix` differentiates each shooting block by a complex step. Each RK4 stage runs the nonlinear right
side in complex arithmetic on nf + 1 batch members (nf = 518 at k_c 103), with complex products against the real tau
operators. The horizon series factors one complex N x N matrix per member and per order.

`mode_matrix_fast` integrates each block's base trajectory once in real arithmetic. It carries the nf columns as real
tangents through the linearised right side, whose coefficients are (1, N) arrays of the base stage state. Forward mode
of an RK4 step is the RK4 step of the variational equation, so the result is the derivative of the same discrete map.

- Tangents are parity-restricted Fourier coefficients in the block layout. The default `basis=jac` forms each stage's
  linearised map as one (blk, blk) matrix and applies it as a single product. `coef` and `grid` are kept as variants.
- Blocks 0 and 12 keep the complex step through their Taylor series. The horizon series' periodic solves use one real
  LU for all members.
- lambda enters every tangent tau derivative as D f + lambda f, as in `dss.py`.

## ell_fastdd: the double-double block 12

`ell_ddm` carries block 12's tangents as dual numbers. Every double-double product of the linearised right side acts on
nf tangent rows, with 12 Ozaki products per stage.

`DDJac` keeps `ell_ddm`'s horizon series and its base value arithmetic. It forms each stage's map as a double-double
(blk, blk) matrix J, with the (g -> f) block (B_g * coef_fg) @ S_f, and applies it as one Ozaki product dc @ J. The
products with the base coefficients then act on the n_g rows of B_g, not on the nf tangent rows.

## d18_secant: Brent refinement on the hybrid sign

Each evaluation is `d18_hybsign.py`'s, with the same dd_logdet sign and the same 1e-15 noise test on blocks 0-11
(seeds 100 and 101). `--mm=cs --b12=ddm` is `d18_hybsign.py`'s evaluation exactly. The defaults use the two tools above.

- f(lambda) = sign exp(ln|det| - ref), and Brent's zero finder on it (secant or inverse quadratic steps, bisection as
  the safeguard).
- Both input ends are evaluated first. The run is refused (exit 2) unless both are robust with opposite signs.
- Robust means both noise signs equal the sign, and dln, the largest change of ln|det| under the noise, is at most
  `--dln` (0.1). `--dln=inf` is `d18_hybsign.py`'s test alone.
- An interior point that is not robust stops the run with its last bracket (exit 4). So does `--maxev` (exit 4).
  The bracket reaching `--tol` gives exit 0.
- Output: one record per evaluation, then a `root` record with the bracket, its width and signs, and the det-linear
  root inside it.

(measurements and checks follow)

## How a runner uses them

Set up once, in a worktree of this branch (it does not touch the runner's pinned worktrees):

```bash
git -C ~/Source/choptuik-large-d fetch origin wip/ell1-speed
git -C ~/Source/choptuik-large-d worktree add --detach ~/Source/choptuik-ell1-speed origin/wip/ell1-speed
cd ~/Source/choptuik-ell1-speed/solver/matter
PY=<the venv whose multidouble has dd_logdet>
$PY ell_fastm.py selftest                        # exit 0
$PY ell_fastm.py selftest --mutate=nolam         # exit 1 (lambda dropped from the variational map)
$PY ell_fastdd.py selftest                       # exit 0
$PY ell_fastdd.py selftest --mutate=jround       # exit 1 (J rounded to float64)
$PY d18_secant.py selftest                       # exit 0
for m in half norobust nosigncheck; do $PY d18_secant.py selftest --mutate=$m; done   # exit 1 each
```

Then one refinement per bracket, one process. The bracket is a robust sign change of a `d18_hybsign.py` scan of the
same state and stub. `--pin` takes the blobs (`git hash-object --no-filters`) of the state and the stub. The run
refuses another state, a state whose residual does not reproduce the record, or ends without opposite robust signs.

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY -u d18_secant.py run \
    <raw>/<folder>/in/<stub>.jsonl <raw>/<folder>/in/<state>.npy <D> <LO> <HI> <raw>/<folder>/secant/secant.jsonl \
    --tol=1e-6 --maxev=10 --pin=<state blob>,<stub blob> > <raw>/<folder>/secant/secant.log 2>&1
```

Exit 0 means the bracket reached the tolerance, and exit 4 means it stopped earlier with a valid bracket. Either way
the last line of the jsonl is the `root` record. Adding `--ends=<scan jsonl>` reuses the scan's two end records and
saves two evaluations, but those ends then carry only `d18_hybsign.py`'s robustness test.
