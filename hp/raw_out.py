#!/usr/bin/env python3
"""Raw run output goes to a task folder of the raw-data repository, github.com/nmoinvaz/choptuik-large-d-raw (AGENTS.md,
"Raw output goes to the raw-data repository"). A long-running writer calls require_raw on its output paths before its
first write; it exits unless each path is inside a folder of a checkout of that repository (a git repository whose
root holds the file RAW_DATA_REPO), not at its root. --allow_unraw on the writer's command line writes anyway (tests,
scratch).

    python3 hp/raw_out.py --selftest [--mutate=accept_any]

Control (--selftest): in a planted data repository a path in a task folder passes, while one at its root, one in an
ordinary repository and one outside any repository are refused, and the refused ones pass with allow.
--mutate=accept_any makes the check accept every path; the selftest must then fail. Exit 1 on failure.
"""
import os
import subprocess
import sys

FLAG = "--allow_unraw"
HELP = "write output outside the raw-data repository's task folders (tests and scratch only; AGENTS.md)"
MARKER = "RAW_DATA_REPO"
DATA_URL = "https://github.com/nmoinvaz/choptuik-large-d-raw"


def pop_allow(argv):
    """remove --allow_unraw from argv (a list, in place) for writers that parse positionals by index"""
    found = FLAG in argv
    while FLAG in argv:
        argv.remove(FLAG)
    return found


def data_folder(path):
    """the task folder of the raw-data checkout that holds path ('' at the checkout's root), or None outside one"""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    r = subprocess.run(["git", "-C", d, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    root = os.path.realpath(r.stdout.strip())
    if not os.path.isfile(os.path.join(root, MARKER)):
        return None
    # both sides resolved: git reports the root through symlinks (macOS /var -> /private/var)
    parts = os.path.relpath(os.path.realpath(path), root).split(os.sep)
    if parts[0] == os.pardir:
        return None
    return parts[0] if len(parts) > 1 else ""


def require_raw(*paths, allow=None, _accept_any=False):
    """exit unless every path is in a task folder of the raw-data repository; allow defaults to --allow_unraw in
    sys.argv"""
    if allow is None:
        allow = FLAG in sys.argv
    for p in paths:
        if p is None:
            continue
        if not (_accept_any or data_folder(p) or allow):
            raise SystemExit(f"{p}: not in a task folder of the raw-data repository. Raw output goes there (AGENTS.md): "
                             f"git clone {DATA_URL} ../choptuik-large-d-raw, and write into "
                             f"../choptuik-large-d-raw/<task>-<machine>/; {FLAG} writes it anyway")


def selftest(mutate=""):
    import shutil
    import tempfile
    t = tempfile.mkdtemp(prefix="raw_out_")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    try:
        data, code, outside = (os.path.join(t, x) for x in ("raw", "code", os.path.join("plain", "x.jsonl")))
        for g, files in ((data, [MARKER]), (code, ["f"])):
            os.makedirs(g)
            run = lambda *a: subprocess.run(["git", "-C", g, *a], check=True, capture_output=True, env=env)  # noqa: E731
            run("init", "-q", "-b", "main")
            for f in files:
                open(os.path.join(g, f), "w").close()
            run("add", *files), run("commit", "-qm", "f")

        def refused(p, allow=False):
            try:
                require_raw(p, allow=allow, _accept_any=mutate == "accept_any")
                return False
            except SystemExit:
                return True
        ok = dict(folder_passes=not refused(os.path.join(data, "T0-mac", "sub", "a.jsonl")),
                  root_refused=refused(os.path.join(data, "a.jsonl")),
                  code_refused=refused(os.path.join(code, "data", "a.jsonl")),
                  outside_refused=refused(outside),
                  allow_passes=not refused(outside, allow=True) and not refused(os.path.join(code, "a.jsonl"), allow=True))
        for k, v in ok.items():
            print(f"  {k}: {'PASS' if v else 'FIRE'}")
        return all(ok.values())
    finally:
        shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        m = next((v.split("=", 1)[1] for v in sys.argv if v.startswith("--mutate=")), "")
        good = selftest(m)
        print(f"selftest: {'PASS' if good else 'FAIL'}")
        sys.exit(0 if good else 1)
    print(__doc__)
