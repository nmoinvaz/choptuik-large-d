"""Where the double-double solver states (.npz) are read from.

hp/data/states (in the repo) holds one state for every integer D from 8 to 60 (the first file, in sorted order, that
the scripts' pattern up*_D{t}.*.npz picks from the runs' directory) plus the files that hp/largeD_scaling.py and
solver/largeD_formal_check.py name explicitly, 53 files, about 1 MB. When that directory is missing, the runs'
own directory /home/claude/dss/hp_scan (not in the repo) is used.

    from states import H, path        # H: directory prefix for glob patterns; path("up13_D60.0020") -> full path

Reading a state needs the compiled double-double kernel: install Rust and maturin, then run hp/build.sh (it builds
for the local instruction set and pip-installs the hpkernel wheel); `python3 -c "import hpkernel"` checks it.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_STATES = os.path.join(HERE, "data", "states")
RUNS = "/home/claude/dss/hp_scan"
H = (REPO_STATES if os.path.isdir(REPO_STATES) and os.listdir(REPO_STATES) else RUNS) + os.sep


def path(name):
    """Full path of a named state, looked up in hp/data/states first, then in the runs' directory."""
    fname = name if name.endswith(".npz") else name + ".npz"
    for d in (REPO_STATES, RUNS):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fname)
