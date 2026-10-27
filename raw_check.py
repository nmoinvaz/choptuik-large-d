#!/usr/bin/env python3
"""Check that a runner's raw/<task> branch on GitHub holds the files its post names (AGENTS.md, raw output). The
coordinator runs it on each post that reports a run; exit 1 when the branch or a file is missing.

    python3 raw_check.py raw/<task> [PATTERN ...]          (patterns as in fnmatch, against the branch's paths)
    python3 raw_check.py --selftest [--mutate=no_ls]

It fetches the branch from origin, then prints its last commit (hash, time, age) and, per pattern, the number of
files that match. Control (--selftest): a planted origin with raw/T0 holding a.jsonl; the check passes for a.jsonl,
fails for b.jsonl and for a branch that was never pushed. --mutate=no_ls treats every pattern as present (the
selftest must then fail).
"""
import fnmatch
import os
import subprocess
import sys
import time


def git(repo, *a):
    r = subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)
    return r.returncode, r.stdout


def check(repo, branch, patterns, no_ls=False):
    rc, _ = git(repo, "fetch", "-q", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
    if rc:
        print(f"{branch}: not on origin")
        return False
    _, head = git(repo, "log", "-1", "--format=%h %ct %s", f"origin/{branch}")
    h, ct, subj = head.strip().split(" ", 2)
    age = (time.time() - int(ct)) / 60
    print(f"{branch}: {h} {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(int(ct)))} ({age:.0f} min ago) {subj}")
    _, ls = git(repo, "ls-tree", "-r", "--name-only", f"origin/{branch}")
    paths = ls.split()
    ok = True
    for p in patterns:
        n = len(paths) if no_ls else len(fnmatch.filter(paths, p))
        print(f"  {p}: {n} file(s)")
        ok &= n > 0
    return ok


def selftest(mutate=""):
    import shutil
    import tempfile
    t = tempfile.mkdtemp(prefix="raw_check_")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    try:
        o, w, c = (os.path.join(t, x) for x in ("origin.git", "runner", "coord"))
        sh = lambda *a: subprocess.run(list(a), check=True, capture_output=True, env=env)  # noqa: E731
        sh("git", "init", "-q", "--bare", o)
        sh("git", "init", "-q", "-b", "raw/T0", w)
        open(os.path.join(w, "a.jsonl"), "w").write("{}\n")
        sh("git", "-C", w, "add", "a.jsonl"), sh("git", "-C", w, "commit", "-qm", "T0: a")
        sh("git", "-C", w, "remote", "add", "origin", o), sh("git", "-C", w, "push", "-q", "origin", "raw/T0")
        sh("git", "init", "-q", c), sh("git", "-C", c, "remote", "add", "origin", o)
        nl = mutate == "no_ls"
        res = dict(present=check(c, "raw/T0", ["a.jsonl"], nl), missing_file=not check(c, "raw/T0", ["b.jsonl"], nl),
                   missing_branch=not check(c, "raw/T9", ["a.jsonl"], nl))
        for k, v in res.items():
            print(f"  {k}: {'PASS' if v else 'FIRE'}")
        return all(res.values())
    finally:
        shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        m = next((v.split("=", 1)[1] for v in sys.argv if v.startswith("--mutate=")), "")
        good = selftest(m)
        print(f"selftest: {'PASS' if good else 'FAIL'}")
        sys.exit(0 if good else 1)
    if len(sys.argv) < 2 or not sys.argv[1].startswith("raw/"):
        sys.exit(__doc__)
    sys.exit(0 if check(os.path.dirname(os.path.abspath(__file__)), sys.argv[1], sys.argv[2:]) else 1)
