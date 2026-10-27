#!/usr/bin/env python3
"""Raw run output goes to a raw/<task> worktree (AGENTS.md, "Raw output goes to its own branch"). A long-running writer
calls require_raw on its output paths before its first write; it exits unless each path's directory is in a git
worktree whose branch is raw/<anything>. --allow_unraw on the writer's command line writes anyway (tests, scratch).

    python3 hp/raw_out.py --selftest [--mutate=accept_any]

Control (--selftest): in a planted repository, a path on branch raw/T0 passes, one on main and one outside any
repository are refused, and the refused ones pass with allow. --mutate=accept_any makes the check accept every
path; the selftest must then fail. Exit 1 on failure.
"""
import os
import subprocess
import sys

FLAG = "--allow_unraw"
HELP = "write output outside a raw/<task> worktree (tests and scratch only; AGENTS.md)"


def pop_allow(argv):
    """remove --allow_unraw from argv (a list, in place) for writers that parse positionals by index"""
    found = FLAG in argv
    while FLAG in argv:
        argv.remove(FLAG)
    return found


def branch_of(path):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    r = subprocess.run(["git", "-C", d, "symbolic-ref", "--short", "-q", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def require_raw(*paths, allow=None, _accept_any=False):
    """exit unless every path is in a worktree on a raw/ branch; allow defaults to --allow_unraw in sys.argv"""
    if allow is None:
        allow = FLAG in sys.argv
    for p in paths:
        if p is None:
            continue
        b = branch_of(p)
        if not (_accept_any or b.startswith("raw/") or allow):
            raise SystemExit(f"{p}: not in a raw/<task> worktree (branch {b or 'none'}). Raw output goes to its own "
                             f"branch (AGENTS.md): git worktree add --orphan -b raw/<task> ../raw-<task>, and write "
                             f"there; {FLAG} writes it anyway")


def selftest(mutate=""):
    import shutil
    import tempfile
    t = tempfile.mkdtemp(prefix="raw_out_")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    try:
        g, w, outside = os.path.join(t, "code"), os.path.join(t, "raw-T0"), os.path.join(t, "plain", "x.jsonl")
        os.makedirs(g)
        run = lambda *a: subprocess.run(["git", "-C", g, *a], check=True, capture_output=True, env=env)  # noqa: E731
        run("init", "-q", "-b", "main")
        open(os.path.join(g, "f"), "w").close()
        run("add", "f"), run("commit", "-qm", "f")
        run("worktree", "add", "-q", "--orphan", "-b", "raw/T0", w)

        def refused(p, allow=False):
            try:
                require_raw(p, allow=allow, _accept_any=mutate == "accept_any")
                return False
            except SystemExit:
                return True
        ok = dict(raw_passes=not refused(os.path.join(w, "sub", "a.jsonl")),
                  main_refused=refused(os.path.join(g, "data", "a.jsonl")),
                  outside_refused=refused(outside),
                  allow_passes=not refused(outside, allow=True) and not refused(os.path.join(g, "a.jsonl"), allow=True))
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
