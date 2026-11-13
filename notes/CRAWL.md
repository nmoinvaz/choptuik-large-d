# The crawl at the chain's front: where the residual jumps along the Newton step (#700)

DRAFT, filled in as the runs end.

## 1. The question

Past D 251.9 the octo-double corrector of #603 crawls. At each Newton step the natural test refuses lam = 1 at a
simplified-correction ratio of 1e2 to 1e17 and accepts lam = 1/2 at a ratio of exactly 0.5000, so max |F| and |dz|
halve per step until the step cap (#674 at D 251.886, a = 0.7276; #603's chain_ns at D 253.88, a = 0.0543). A smooth
residual along z + lam dz cannot be affine at lam = 1/2 to four digits and 1e30 off at lam = 1. This note locates the
point lam* where it stops being affine, names the rows that jump there, and names the branch in the code that makes
them jump.

## 2. Method

`hp/crawl_probe.py` (raw folder `I700-mac/` of the raw-data repository).

- `locate_jump` evaluates F(lam) = F(z + lam dz) on a ladder of lam and measures its deviation from the affine model,
  e(lam) = max |F(lam) - (1 - lam) F(0)|. Along an exact Newton step a smooth map deviates by its quadratic
  remainder, which grows like lam². A rung whose e exceeds K = 1e3 times the previous rung's e carried by
  (lam_b / lam_a)² is a jump. Its bracket is bisected, with the threshold set at the geometric mean of the two sides,
  until it is narrower than 1e-6. The rows that carry the jump are the fewest whose squared entries hold 99% of
  |F(lam_b) - F(lam_a) + (lam_b - lam_a) F(0)|².
- `iterate` rebuilds one attempt of anm_run.py with #603's flags and runs `newton_od` to its first crawl iterate. It
  saves that iterate z and its Newton step dz exactly as `ElimO.newton_o` formed them inside `newton_od`.
- `probe` runs `locate_jump` at that iterate on three residuals. Arm E is the corrector's own (ElimO: the trial
  projected when its pinned quad-double max |F| exceeds the switch, 1e-15, as the natural test evaluates it). Arm U is
  never projected (PinnedO). Arm P is always projected. It also forms the natural test's ratio at lam = 1 and 1/2,
  with the trial as the corrector projects it and with the trial projected.

## 3. The planted check (fixed before the first real run)

A smooth map on 161 unknowns, F(x) = A x + b + eps x∘x with eps = 1e-6, at a random start with its exact Newton step,
so that F(lam) = (1 - lam) F(0) + eps lam² dz∘dz. A jump of amplitudes (0.7, 0.7, 0.1) on rows 140, 141 and 150 is
switched on where a random linear form passes its value at lam_true = 0.7316807. Pass: lam* within 1e-6 of lam_true,
the 99% rows equal to the planted rows, and the same map without the jump reported affine to lam = 1. The five
mutations must each make it fail.

| run | lam* | rows carrying 99% | control | verdict |
|---|---|---|---|---|
| as planted | 0.731680727 (2.7e-8 off) | 140, 141, 150 | affine to 1 (growth 1.00) | PASS |
| `--mutate shift` (planted 5e-6 later) | 0.731685686 | 140, 141, 150 | affine | FAIL, as it must |
| `--mutate comp` (planted one row over) | 0.731680727 | 141, 142, 151 | affine | FAIL, as it must |
| `--mutate smooth` (no jump planted) | none | none | affine | FAIL, as it must |
| `--mutate control` (a jump in the control) | 0.731680727 | 140, 141, 150 | jumped | FAIL, as it must |
| `--mutate coarse` (bisection stopped at 1e-2) | 0.734375000 | 140, 141, 150 | affine | FAIL, as it must |

The first plant had eps = 1e-3, so its quadratic remainder at lam = 1 was 9e-3, and the jump stood only 135 times
above it, under K. The detector reported it affine, as specified, and that plant was resized to eps = 1e-6 before
any real run. So `locate_jump` resolves a jump only when it is more than K = 1e3 times the remainder at its rung.
