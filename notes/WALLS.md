# Walls of the continuation in D

A wall is a dimension where the continuation in D stops. Its steps in D shrink toward zero, and the solver no longer
converges from the predictions it is given. This note lists the walls met so far, what caused each, and what they
share. The evidence for each is in the section it links.

## In plain words

The program finds the echo period Δ by solving a large set of equations, then raising D a little and solving again.
The equations hold Δ only loosely, and that hold weakens as D grows. Near D 250 it weakens by about a factor of four
for each unit of D. Every approximation the program makes leaves a small error: the digits it carries, the number of
terms in each series, the length of each step. The loose hold magnifies that error by the same growing factor. A wall
is where the magnified error becomes as large as the solver can tolerate.

So far every wall has been the program's, not the solution's. When the digits, the series or the stepping method
changed, the wall moved to a larger D, and Δ(D) stayed smooth through the D where it had been.

## The walls so far

| D | what stopped | cause | how it was passed | evidence |
|---|---|---|---|---|
| 35.2 | double-double steps collapsed | the residual floor of 10⁻¹⁴ to 10⁻¹⁶ set by undetermined directions, with every harmonic kept | runs that held the weakest harmonics on their trend (T48) | [LARGE_D_WALL.md](LARGE_D_WALL.md), first section |
| 60.96 | the same | precision: the unit roundoff times the Jacobian's condition number reached 1.7 | the same runs, to 67.78 | the same |
| 73.3, 78.1 | the pinned double-double runs | the horizon block: the horizon series' radius shrinks with the accuracy of the horizon data, so Newton needs a guess no stored point supplies | the Taylor continuation in D, with quad- and octo-double arithmetic and harmonics 5–13 held on a fit | [QUAD_DOUBLE.md](QUAD_DOUBLE.md) 12–13, [ALGORITHM.md](../ALGORITHM.md) 3 |
| 232–242 | octo-double steps refused | the corrector, in its step after the first move, not the prediction | #603's chain went on to 259.89 | [LARGE_D_WALL.md](LARGE_D_WALL.md), "The refusals are the corrector's" |
| 246–259.9 | the chain leaves the 1/n series, then folds at D* = 259.903 | a term of the discrete system that grows as its smallest pivot falls, as about e^(−1.44D) | open (below) | [LARGE_D_WALL.md](LARGE_D_WALL.md), #805 and "Where the series is singular" |

## What they share

- **One weakly held direction.** Δ and a few harmonics form the directions the equations fix least. The solver's
  smallest pivot measures that hold, and past D 245 it falls as about e^(−1.44D) (#805).
- **An error divided by that hold.** Each wall is where one approximation's error, divided by the hold, reaches what
  the solver can absorb. The error comes from the digits at 60.96, Newton's basin at 73–78 and the corrector's rise
  at 232–242. Past 246 it is a term of the discretization, |chain − series| ≤ 2.2×10⁻³⁷/p_min.
- **A wall moves when the method improves, but the mechanism stays.** The accuracy needed grows exponentially in D,
  so each improvement buys a finite stretch of D.

## Telling the method's wall from the solution's

A feature of the solution stays where it is when the method changes, and a wall of the method moves. Every wall above
was tested this way, by changing one setting: the digits, the held harmonics, the steps, or the horizon order.

For the fold at 259.903 the test is under way.
- The 1/n series fitted below D 245 shows no fold there, and would have seen a square root of the continuum's size
  (track C, #835).
- The leading candidate is the horizon series, which the chain sums to 130 terms at every D. At D 150, raising the
  order settles Δ to 10⁻¹⁵ by 160 terms ([LARGE_D_WALL.md](LARGE_D_WALL.md), "The horizon order raised at D 150").
- **Near the fold the cut weakens the hold most of all.** Raising the order from 130 lifts the hold on Δ by e^18.5,
  e^27.7 and e^34.8 at D 237, 249 and 254, complete by order 160–200 (#902). With the series summed in full, the hold
  still weakens with D, but about e^−0.28 per unit of D, against e^−1.26 at order 130. So the cut is most of the wall,
  and an intrinsic weakening remains.
- **The other settings are not the source.** An adjoint split of the departure past D 245 finds the plan, the centre
  and transition orders, and the held fit too small to explain it, while the order's response is 66 times it (#905).
  The held fit's degree spread keeps one sign while the departure changes sign twice (#904).
- **The order has to be climbed.** A jump from 130 to 250 diverges at D 225 and 245, while rungs of +10 or less
  converge (#903).
- **Under way:** chains at fixed orders 135, 142 and 160 run from D 200–225 to the fold, and #903's chain raises the
  order as D grows. A fold that stays within 0.01 of 259.903 would be the continuum's, and one that moves by more than
  0.05 the truncation's. The details are in [LARGE_D_WALL.md](LARGE_D_WALL.md), from "The grip on Δ against the
  horizon order near the wall" on.

## The same mechanism elsewhere

The σ-model at η = 0.3 near D 10.5 has a band of weakly held directions in its f64 solver. There the walks without the
smoothness prior do not settle, and two branches end 10⁻² apart ([matter/SIGMA_DSS.md](../matter/SIGMA_DSS.md) §11).
