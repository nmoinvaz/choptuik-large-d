# Step-size test (hp/step_test.py)

Reference: delta_dd.csv to D = 61.026449; Chebyshev fit rms 8.5e-15 (used above D = 61.0). MUTATED: reference shifted (ref).

## Start states

| start | D | floor | max dDelta/dF | floor x sens | Delta - saved state | its |
|---|---|---|---|---|---|---|
| start_D60.0020_x0.7_kf13 | 60.0020 | 1.6e-23 | 8.9e+13 | 1.4e-09 | +0.0e+00 | 10 |
| start_D60.0020_x0.6_kf13 | 60.0020 | 1.1e-19 | 4.9e+07 | 5.3e-12 | +0.0e+00 | 3 |

## Runs

| run | end | ok / failed | D covered | largest step | edge | Newton its | D per 100 its (steady) | D per hour (steady) | max abs dDelta_est | max abs(Delta - ref) | check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base_D60.00 | max_points 8 | 8 / 4 | 0.2600 | 0.0484 | 0.0385 | 202 | 0.1287 (0.1048) | 2.117 | 1.1e-17 | 1.0e-09 | FAIL |
| node06_D60.00 | 6 failed steps in a row | 0 / 6 | 0.0000 | 0.0000 | 0.0040 | 35 | 0.0000 (0.0000) | 0.000 | 0.0e+00 | nan | - |
