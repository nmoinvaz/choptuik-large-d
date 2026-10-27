#!/usr/bin/env python3
"""Check that a runner's task folder in the raw-data repository on GitHub holds the files its post names (AGENTS.md,
raw output). The coordinator runs it on each post that reports a run; exit 1 when the folder or a file is missing.

    python3 raw_check.py <folder> [PATTERN ...] [--data DIR]   (patterns as in fnmatch, against the folder's paths)
    python3 raw_check.py --selftest [--mutate=no_ls]

DIR is a clone of github.com/nmoinvaz/choptuik-large-d-raw (default ../choptuik-large-d-raw beside this repository).
A folder given as raw/<name> means <name>, the folder imported from that branch. It fetches origin's main, then prints
the folder's last commit (hash, time, age) and, per pattern, the number of files that match. Control (--selftest): a
planted origin holding T0-x/a.jsonl; the check passes for a.jsonl, fails for b.jsonl and for a folder that was never
pushed. --mutate=no_ls treats every pattern as present (the selftest must then fail).
"""
import fnmatch
import os
import subprocess
import sys
import time


def git(repo, *a):
    r = subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)
    return r.returncode, r.stdout


def check(repo, folder, patterns, no_ls=False):
    folder = folder[4:] if folder.startswith("raw/") else folder
    rc, _ = git(repo, "fetch", "-q", "origin", "+refs/heads/main:refs/remotes/origin/main")
    if rc:
        print(f"{repo}: fetch from origin failed")
        return False
    _, ls = git(repo, "ls-tree", "-r", "--name-only", "origin/main", "--", folder + "/")
    paths = [p[len(folder) + 1:] for p in ls.split("\n") if p]
    if not paths:
        print(f"{folder}: not on origin")
        return False
    _, head = git(repo, "log", "-1", "--format=%h %ct %s", "origin/main", "--", folder + "/")
    h, ct, subj = head.strip().split(" ", 2)
    age = (time.time() - int(ct)) / 60
    print(f"{folder}: {h} {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(int(ct)))} ({age:.0f} min ago) {subj}")
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
        sh("git", "init", "-q", "--bare", "-b", "main", o)
        sh("git", "init", "-q", "-b", "main", w)
        os.makedirs(os.path.join(w, "T0-x"))
        open(os.path.join(w, "T0-x", "a.jsonl"), "w").write("{}\n")
        sh("git", "-C", w, "add", "T0-x"), sh("git", "-C", w, "commit", "-qm", "T0-x: a")
        sh("git", "-C", w, "remote", "add", "origin", o), sh("git", "-C", w, "push", "-q", "origin", "main")
        sh("git", "init", "-q", c), sh("git", "-C", c, "remote", "add", "origin", o)
        nl = mutate == "no_ls"
        res = dict(present=check(c, "T0-x", ["a.jsonl"], nl), missing_file=not check(c, "T0-x", ["b.jsonl"], nl),
                   missing_folder=not check(c, "T9-x", ["a.jsonl"], nl))
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
    args = [a for a in sys.argv[1:]]
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "choptuik-large-d-raw")
    if "--data" in args:
        i = args.index("--data")
        data = args[i + 1]
        del args[i:i + 2]
    if not args:
        sys.exit(__doc__)
    sys.exit(0 if check(data, args[0], args[1:]) else 1)
