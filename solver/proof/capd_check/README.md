# capd_check: an independent CAPD check of the M = 1 certificate

A second implementation of steps (ii)–(iv) of [proof/A_certify.md](../../../proof/A_certify.md) (§9), in C++ with the
CAPD library, sharing no code with `validated_int.py` or `krawczyk.py`. Results and discussion: A_certify.md §10.

| file | what it is |
|---|---|
| `capd_check.cpp` | the vector field (written from A_certify §2), the path, the validated integration (CAPD Lohner sets), the Krawczyk test, mutation controls |
| `export_start.py` | the one link to the Python proof code: writes the step-(i) start data of `fixed_point.start_data` (and κ) to `data/` as outward-rounded double and double-double intervals |
| `series_check.py` | sanity check of the exported start values against an independent mpmath recursion of the formal series |
| `CMakeLists.txt`, `build.sh` | build: `build.sh` clones CAPD at a pinned commit and builds both |
| `data/` | the exported start data (point and boxes 10⁻⁹ … 10⁻⁵) |
| `results/` | the logs of the runs quoted in §10, the saved point runs, and the Krawczyk inputs as exact hex bounds (below) |

**CAPD** (Computer Assisted Proofs in Dynamics, <https://github.com/CAPDGroup/CAPD>) is free software under the GNU
General Public License v3. It is not vendored here; `build.sh` clones commit
`37057e5899bccaa6cd5701057b7e6ea5e232ef8b` (version 6.1.0) and builds the library with multiprecision (MPFR) support.

```
./build.sh                       # needs cmake, g++, git, libmpfr-dev, libgmp-dev; about 5 min on 2 cores
build/capd_check data/start_point.txt data/start_box_1e-9.txt data/start_box_1e-8.txt     # about 10 CPU minutes
python3 series_check.py          # seconds
python3 export_start.py          # regenerate data/ from fixed_point.py, seconds
```

**results/**

- `run_standard.txt`, `run_rho0.2.txt`: the full runs (multiprecision point run, then the box runs) on the standard
  path and on the lower path. They were made one build earlier than the final source. The changes since do not affect
  these runs: the control summary line, skipping control C6 when the clean test fails, and 17-digit printing of the
  zero.
- `point_standard.txt`, `point_rho0.2.txt`: the jump functionals at x̂ from those point runs (outward hex bounds).
- `boxes_standard.txt` (10⁻⁹, 10⁻⁸, 10⁻⁷), `boxes_rho0.2.txt` (10⁻⁹, 10⁻⁸): the box runs redone with the final build
  from the saved point runs (`--load-point`); the same numbers.
- `run_double.txt`: everything in double precision (`--prec=0`): fails at 10⁻⁹, passes at 10⁻⁸.
- `capd_enclosures_*.json`: F(x̂), DF over X, κ and the zero enclosure, as exact hex bounds, for an external checker.
