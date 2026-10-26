"""Where the double-double solver states (.npz) are read from.

hp/data/states (in the repo) holds one state for every integer D from 8 to 60 (the first file, in sorted order, that
the scripts' pattern up*_D{t}.*.npz picks from the runs' directory) plus the files that hp/largeD_scaling.py and
solver/largeD_formal_check.py name explicitly, 53 files, about 1 MB. The runs' own directory (/home/claude/dss/hp_scan
on the cloud disk) and its other states are missing since 2026-10-26; these are the ones that remain.

    from states import H, path        # H: directory prefix for glob patterns; path("up13_D60.0020") -> full path

Reading a state needs the compiled double-double kernel: install Rust and maturin, then run hp/build.sh (it builds
for the local instruction set and pip-installs the hpkernel wheel); `python3 -c "import hpkernel"` checks it.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_STATES = os.path.join(HERE, "data", "states")
H = REPO_STATES + os.sep


def path(name):
    """Full path of a named state in hp/data/states."""
    fname = name if name.endswith(".npz") else name + ".npz"
    p = os.path.join(REPO_STATES, fname)
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"{fname}: not in hp/data/states (the cloud runs' other states are missing)")
