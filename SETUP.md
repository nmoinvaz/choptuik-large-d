# Setup

The scripts need Python 3.13 or 3.14 with the packages in [requirements.txt](requirements.txt). The double-double solver
and the f64 solver's residual are Rust kernels built on the machine that runs them. Everything else is optional.

## System requirements

| | needed | notes |
|---|---|---|
| OS | macOS, Windows 10/11 or Linux | the runners: macOS 27 on Apple M3 and M5, Windows 11 on Intel x86_64 |
| CPU | x86_64 or aarch64 | the kernels use AVX-512, AVX2+FMA or NEON when present, else a portable path; every path gives bitwise identical results ([SOFTWARE.md](SOFTWARE.md)) |
| Memory | 8 GB for the Python scripts | building `hp` takes more: rustc peaked at 15.2–15.8 GB with LTO off and 16 codegen units, and the default fat-LTO build needs more again. Long runs grow with M and the process count: #82's three amplitude nodes took 55–70 GB together |
| Disk | about 1 GB | the repository's history is about 400 MB, and the raw-data repository's about 220 MB |
| Network | crates.io and GitHub | for the Rust builds and multidouble's Python package |

## Software requirements

| | version | used by |
|---|---|---|
| Python | 3.13 or 3.14; 3.10 and later parse every script | everything |
| [uv](https://docs.astral.sh/uv/) | any recent | the steps below. pip and venv work too |
| numpy, scipy, mpmath, sympy, python-flint | pinned in [requirements.txt](requirements.txt) | everything; python-flint (FLINT 3.6.0) for `check_numbers.py`, `solver/proof/`, `solver/inner_amp/` and `solver/inner_borel/` |
| Rust (stable, via [rustup](https://rustup.rs)) and maturin | maturin 1.16 | `pip install .` (rustkernel) and `hp/build.sh` (hpkernel and multidouble's Python package) |
| git | any | cloning the two repositories |
| [gh](https://cli.github.com) | logged in with access to this repository | runners only: tasks are issues (AGENTS.md) |
| bash | any | `hp/build.sh`, `reproduce.sh`; on Windows, Git Bash |

Optional:

| | used by |
|---|---|
| MPFR and GMP development headers, a C++ compiler | CAPD, `solver/proof/capd_check/build.sh` |
| pdflatex with amsart (TeX Live) | the papers, `make` in `paper/math/` and `paper/physics/` |
| cairosvg | `paper/svg2pdf.py` |
| numba | `solver/inner_stokes/inner_nb.py`, `solver/matter/ell_collapse.py` |
| matplotlib | `solver/plot_convergence.py` |

## Steps

macOS and Linux:

```sh
git clone https://github.com/nmoinvaz/choptuik-large-d && cd choptuik-large-d
git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw   # raw output, read by some scripts
uv python install 3.14.10
uv venv --seed --python 3.14.10 .venv      # --seed puts pip in the venv, which hp/build.sh calls
source .venv/bin/activate
uv pip install -r requirements.txt
pip install .                              # rustkernel, the f64 solver's residual (optional)
hp/build.sh                                # hpkernel and multidouble's Python package (optional)
```

Windows, in PowerShell for the venv and Git Bash for `hp/build.sh`:

```powershell
uv python install 3.14.10
uv venv --seed --python 3.14.10 .venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

`uv python install` adds an interpreter beside any existing one and changes no system Python, PATH entry or launcher.

## Checking the setup

```sh
python3 check_numbers.py --selftest   # every quoted number against its source; exit 0
./reproduce.sh                        # the headline numbers, PASS / FAIL / SKIP per check
```

A new build of `hp` is checked against the stored reference files with `hp/highm_check.py ref`, `scaled` and `jets`, as
AGENTS.md describes.

## Known issues

- `hp/build.sh` keeps an installed multidouble Python package of the same version number, so it reinstalls it every
  time. Uninstall multidouble first when moving to a new venv.
- On macOS 27, a release-stripped Rust extension can fail to load with "mis-aligned LINKEDIT string pool". Build with
  `CARGO_PROFILE_RELEASE_STRIP=false`.
- `check_numbers.py --selftest`'s formula-refit item depends on the BLAS library under numpy, and fails on some
  machines (#183).
