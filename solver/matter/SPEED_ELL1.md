# SPEED_ELL1: faster l = 1 root readings for #1025

Three tools on branch `wip/ell1-speed`, all in `solver/matter/`. The runs behind the numbers below are in the
raw-data repository, folder `I1025-mac/speed_ell1/`, on its branch `wip/i1025-mac-speed` (commit 9f0b2dc6, one commit on
main's 083e16ec, not yet in main). They need the venv with multidouble's `dd_logdet`
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
(seeds 100 and 101). `--mm=cs` is `d18_hybsign.py`'s evaluation exactly. The default takes the float64 matrix from
`ell_fastm` and keeps `ell_ddm`'s block 12; `--b12=fast` takes `ell_fastdd`'s, which pays only up to k_c of about 63.

- f(lambda) = sign exp(ln|det| - ref), and Brent's zero finder on it (secant or inverse quadratic steps, bisection as
  the safeguard).
- Both input ends are evaluated first. The run is refused (exit 2) unless both are robust with opposite signs.
- Robust means both noise signs equal the sign, and dln, the largest change of ln|det| under the noise, is at most
  `--dln` (0.1). `--dln=inf` is `d18_hybsign.py`'s test alone.
- An interior point that is not robust stops the run with its last bracket (exit 4). So does `--maxev` (exit 4).
  The bracket reaching `--tol` gives exit 0.
- Output: one record per evaluation, then a `root` record with the bracket, its width and signs, and the det-linear
  root inside it.

## Measured speed-ups

All on the Mac (Apple M5), one BLAS thread and one rayon thread per process, nice 10, beside other users' jobs (load
11 to 62 on 10 cores). Each ratio comes from one process timing both codes one after the other, since timings in
separate processes moved by 2x with the load.

| k_c, D, lambda | quantity | old | new | ratio |
|---|---|---|---|---|
| 55, 18.125, 1.775 | float64 M, all 13 blocks | 73.8 s (complex step) | 13.2 s (`jac`) | 5.6 |
| 55, 18.125, 1.775 | double-double block 12 | 254.1 s (`ell_ddm`) | 148.6 s (`ell_fastdd`) | 1.7 |
| 103, 22.125, 0.4875 | float64 blocks 1 and 12 | 138.3 s | 51.0 s (`coef`) | 2.7 |
| 103, 22.125, 0.4875 | double-double stage (one RK4 stage of block 12) | 487 ms | 291 ms | 1.7 |
| 103, 22.125, 0.4875 | float64 M, all 13 blocks | 958.9 s | 129.9 s (`jac`) | 7.4 |
| 103, 22.125, 0.4875 | double-double block 12 | 749.2 s | 944.2 s (`ell_fastdd`) | 0.79 |

`jac` beats `coef` by 2x at k_c 55 (blocks 3 and 11: 3.5 s against 7.3 s). The horizon series alone took 139 s by
complex step at k_c 103, most of it the per-member complex LUs, which the one-LU solve removes.

`ell_fastdd` is faster at k_c 55 (1.4x and 1.7x in two runs) but not at k_c 103. There its stage matrices cost more
float64 products (2.5e9 multiply-adds per stage against `ell_ddm`'s 1.8e9) than its smaller elementwise work saves:
0.79x in one process while the load fell from 12 to 4, 1.7x per stage in another. So `d18_secant.py` keeps `ell_ddm`'s
block 12 by default.

The double-double block 12 dominates an evaluation once the float64 matrix is fast. Combining the two k_c 103 checks, the old
evaluation's matrices cost about 959 + 749 s and the new one's 130 + 749 s, 1.9x. One full evaluation with `ell_fastdd`'s block
12 took 563 s and 1316 s at k_c 103 (two loads), of which block 12 took 505 s and 1163 s and the three `dd_logdet` 21 s
and 8 s.

The larger gain is in the number of evaluations. Both refinements below went from the scan's 0.01 bracket to a bracket
under 1e-6 in 6 evaluations, the two ends included. Bisection needs 14 interior points for the same width (16 with the
ends), and a grid at that spacing is out of reach.

## Checks

| check | result | its control |
|---|---|---|
| `ell_fastm.py selftest` (k_c 15 state): `rhs_lin` against a central difference of `DSSEll.rhs` | 8.0e-9 (bar 1e-7) | `--mutate=nolam`: 8.6e-4, exit 1 |
| same: `coef` against `grid` tangents, `jac` against `coef` | 2.2e-16, 4.5e-16 (bar 1e-13) | |
| same: M_fast against the complex step, every block, three bases, lambda 0 and 0.9 | at most 3.3e-15 per block, ln within 8.5e-14 | `--mutate=coef` (one coefficient times 1 + 1e-6): 1.7e-9, exit 1 |
| `ell_fastm.py check`, k_c 31, D 10.5, lambda 2.87 (`grid`) | blocks 0-11 at most 2.1e-14, block 12 1.1e-12, ln within 2.7e-10: PASS | `--mutate=nolam`: block 11 1.03, ln off by 98.5, exit 1 |
| same, `jac` | blocks 0-11 at most 2.0e-14, block 12 1.1e-12, ln within 2.8e-10, 4.8x faster: PASS | |
| `ell_fastm.py check`, k_c 55, D 18.125, lambda 1.775 (`jac`) | blocks 0-11 at most 9.9e-15; block 12 1.3e-10 and the float64 sign differ: FAIL | |
| `ell_fastm.py check`, k_c 103, D 22.125, lambda 0.4875 | blocks 0-11 at most 4.4e-15; block 12 2.2e-9, float64 ln|det| off by 0.34 (same sign): FAIL | |
| `ell_fastdd.py selftest`: d/dtau basis rows; block 12 against `ell_ddm`, lambda 0 and 0.9 | 4.6e-31; 4.1e-29, 2.6e-29 (bar 1e-20) | `nolam`: 1.3e-1; `jround`: 1.2e-17; both exit 1 |
| `ell_fastdd.py check`, k_c 55, lambda 1.775, with the hybrid | 8.0e-27; hybrid ln|det| equal to all digits: PASS | |
| `ell_fastdd.py check`, k_c 103, lambda 0.4875 | 4.2e-25: PASS | |
| `d18_secant.py selftest`, planted cases C1-C4 | all as written | `half`, `norobust`, `nosigncheck`: exit 1 each |
| `d18_hybsign.py` (blob e089207a) rerun here on the k_c 55 state at 1.77 | ln|det| -1617.4278109138, the scan record to all printed digits | |

The k_c 55 check fails as written, on block 12 and the float64 determinant only. The float64 block 12 is wrong by
1.5e-10 in both codes (against `ell_ddm`'s double-double block: 1.52e-10 fast, 1.56e-10 complex step, at 1.77). Two
such blocks differ by as much again, and the float64 determinant's sign follows that error. This is the hybrid's
reason for a double-double block 12, which neither code's float64 block enters. The bar of 1e-10 holds at k_c 31 and
fails from k_c 55 up (2.2e-9 at k_c 103).

### Where block 12's float64 error comes from (k_c 55, lambda 1.77 and 1.78)

The horizon series in float64 matches the double-double series to 1.5e-14 (either LU). Feeding the double-double series
into the float64 variational RK leaves the block wrong by 1.44e-10, with the wrong hybrid sign at 1.78. So the error is
the float64 RK of the horizon segment (1 - u0 = 0.96 to 0.9), and only a double-double integration fixes it.

A mixed variant, J formed in float64 from the double-double coefficients and applied to double-double tangents, failed
its bars set before the run: block 6.5e-15 (bar 1e-15) and hybrid ln|det| off by 3.3e-3 (bar 1e-6). It was dropped;
commit c8e7ee9f holds it.

## Real refinements and the float64 floor of the root

k_c 55, D 18.125, bracket (1.77, 1.78) from the Mac scan `I1025-mac/d18/d18125k55_scan_1025.jsonl` (state blob
88ccda5c, stub 0fb77df8; the reader's det-linear root 1.772947, estimator range [1.772947, 1.773343]):

| evaluation | root (det-linear in the final bracket) | final bracket | evaluations | wall |
|---|---|---|---|---|
| `--mm=cs --b12=ddm` (`d18_hybsign.py`'s) | 1.7732997423 | [1.7732990863, 1.7732999430] | 6 | 2378 s |
| fast M, `ell_ddm` block 12 | 1.7733011670 | [1.7733011508, 1.7733016508] | 6 | 1057 s |

Both roots lie in the reader's estimator range, 3.5e-4 above the grid's det-linear estimate. **They differ by
1.4e-6, and their brackets are disjoint.** The two codes' float64 blocks 0-11 agree to 1e-14 per block, yet that moves
the hybrid ln|det| by 4.5e-4 at 1.77 and 2.7e-4 at 1.78 (signs equal). Every evaluation passed the 1e-15 noise test
(dln at most 0.068). Its random perturbations moved ln|det| by 7.2e-7 and 2.5e-7 at the same ends, 600 to 1100 times
less than the systematic difference, so that test does not see this floor.

So the hybrid determinant fixes this root only to about 1.4e-6 at k_c 55. At k_c 103, D 22.125, the fast ends on the
Mac against dh610's scan records (complex step, Linux) differ by 2.7e-5 (0.485) and 1.5e-3 (0.490) in ln|det|, with
equal signs. A shift of ln|det| by d at distance r from a simple root moves it by about d r, which puts the floor there
near 1.5e-7. A tolerance of 1e-5 is safe at both. `--eps=1e-12` makes the noise test conservative. For the k_c step of
the root (1e-3 to 7e-3 between k_c 95 and 103) neither matters.

## Not done

- The k_c 103 refinement itself: only its two ends (0.485, 0.490) ran here, as the timing estimate (1879 s for both).
  Their signs equal dh610's scan, and the det-linear root 0.4898964 matches the scan reader's 0.489896.
- A faster double-double block 12 than 1.7x: it needs fewer Ozaki products per stage than the 15 of `ell_ddm.Op`, at
  double-double accuracy, since the float64 shortcuts above fail.

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
    --tol=1e-5 --maxev=10 --pin=<state blob>,<stub blob> > <raw>/<folder>/secant/secant.log 2>&1
```

`--tol=1e-5` stays above the float64 floor of the root measured below (1.4e-6 at k_c 55).

Exit 0 means the bracket reached the tolerance, and exit 4 means it stopped earlier with a valid bracket. Either way
the last line of the jsonl is the `root` record. Adding `--ends=<scan jsonl>` reuses the scan's two end records and
saves two evaluations, but those ends then carry only `d18_hybsign.py`'s robustness test.

## First real use: k_c 103 at D 22.875

When winb's k_c 103 scan at D 22.875 lands, take its robust sign change (LO, HI) and its state and stub from the
scan folder's `in/`, pinned by blob as the scan pinned them. Then, in the worktree above:

```bash
F=<raw clone>/<winb's folder of the D 22.875 k_c 103 scan>
mkdir -p $F/secant
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY -u d18_secant.py run \
    $F/in/<stub>.jsonl $F/in/<state>.npy 22.875 <LO> <HI> $F/secant/secant_kc103_d22875.jsonl \
    --tol=1e-5 --maxev=10 --pin=<state blob>,<stub blob> > $F/secant/secant_kc103_d22875.log 2>&1; echo "exit $?"
```

Expect about 6 evaluations. On the Mac one evaluation at k_c 103 costs 15 to 25 min (the float64 M 0.6 to 2.4 min,
`ell_ddm`'s block 12 12 to 19 min, the three `dd_logdet` under 0.5 min), so about 1.5 to 2.5 h for the refinement on
one process. The
records' `sec_parts` give the split per evaluation on the runner's machine.
