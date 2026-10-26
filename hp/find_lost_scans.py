#!/usr/bin/env python3
"""Search a machine for the cloud continuation runs' raw output (runs up, up2-up13, up18, up20, up21; D 8.25-61.03),
written to /home/claude/dss/hp_scan and never pushed: records scan_up<N>.jsonl and states up<N>_D<D>.npz.

    python3 hp/find_lost_scans.py --selftest [--mutate=no_probes|no_fsck]
    python3 hp/find_lost_scans.py OUT.json [--roots DIR ...] [--collect DIR]

Probes: the 757 rows of hp/data/delta_dd.csv whose run is a cloud run; their Delta to 11 decimals.
1. Files: names scan_up<N>.jsonl or up<N>_D<D>.npz of a cloud run (N not 14-17, 22, 23, 70), and directories named
   hp_scan, under each root. States already in git (hp/data/states, hp/data/seeds) are listed apart.
2. Contents: text files (.jsonl .json .log .txt .csv .md .output .out, under 2 GB) under each root with a probe Delta;
   a file with at least 5 distinct probes is "likely". Copies of delta_dd.csv and files tracked in a repository found
   under the roots (already in git) are skipped; untracked files in a checkout are searched.
3. Git: every repository under the roots (one per object store), its refs, reflogs and unreachable commits, for paths
   named as in 1, leaving out objects that a remote-tracking branch holds (already on GitHub).
Default roots: the home directory and the temp directory; .git, node_modules, .cargo, .rustup, target, site-packages
and (macOS) Library are skipped (the Trash is searched), except Claude's temp folders, which the temp root covers.
--collect DIR copies every file of 1 and every likely file of 2 into DIR (git blobs by `git cat-file`), with
DIR/manifest.json naming each one's source.

Control (--selftest): a planted tree with a cloud-run record file, a state name, a git repository whose branch holding
scan_up6.jsonl was deleted, and a decoy with no probe; all three must be found and the decoy not. Exit 1 on failure.
--mutate=no_probes empties the probe set (record_file must fire), no_fsck skips the unreachable commits
(deleted_branch must fire).
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_RUNS = {"14", "15", "16", "17", "22", "23", "70"}
NAME = re.compile(r"^(?:scan_up(\d*)\.jsonl|up(\d*)_D\d+\.\d+\.npz)$")
NUM = re.compile(r"\d\.\d{11,}")
TEXT = (".jsonl", ".json", ".log", ".txt", ".csv", ".md", ".output", ".out")
SKIP = {".git", "node_modules", ".cargo", ".rustup", "target", "site-packages", "Library", "AppData"}


def probes(csv_path):
    rows = [r for r in csv.DictReader(open(csv_path)) if re.fullmatch(r"up\d*", r["run"])]
    return {r["Delta_hi"][:13] for r in rows}, len(rows)


def cloud_name(name):
    m = NAME.match(name)
    if not m:
        return False
    n = m.group(1) if m.group(1) is not None else m.group(2)
    return n not in LOCAL_RUNS


def walk(roots):
    seen = set()
    for root in roots:
        for d, dirs, files in os.walk(root, onerror=lambda e: None):
            key = os.path.realpath(d)
            if key in seen:
                dirs[:] = []
                continue
            seen.add(key)
            if ".git" in dirs or ".git" in files:
                yield "repo", d
            dirs[:] = [x for x in dirs if x not in SKIP]
            for x in dirs:
                if x == "hp_scan":
                    yield "dir", os.path.join(d, x)
            for f in files:
                yield "file", os.path.join(d, f)


def scan_text(path, P):
    hits = set()
    try:
        if os.path.getsize(path) > 2 << 30:
            return hits
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                for m in NUM.findall(line):
                    if m[:13] in P:
                        hits.add(m[:13])
    except OSError:
        pass
    return hits


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, errors="replace")
    return r.stdout if r.returncode == 0 else ""


def scan_git(repo, stores, fsck=True):
    common = git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    if not common or common in stores:
        return []
    stores.add(common)
    out = []
    pushed = {ln.split(" ")[0] for ln in git(repo, "rev-list", "--objects", "--remotes").splitlines()}
    for line in git(repo, "rev-list", "--all", "--reflog", "--objects").splitlines():
        sha, _, path = line.partition(" ")
        if path and sha not in pushed and (cloud_name(os.path.basename(path)) or "hp_scan" in path):
            out.append(dict(repo=repo, blob=sha, path=path, via="refs/reflog"))
    lost = [ln.split()[2] for ln in git(repo, "fsck", "--unreachable", "--no-reflogs", "--no-progress").splitlines()
            if fsck and ln.startswith("unreachable commit")]
    for c in lost[:5000]:
        for line in git(repo, "ls-tree", "-r", c).splitlines():
            meta, _, path = line.partition("\t")
            if (cloud_name(os.path.basename(path)) or "hp_scan" in path) and meta.split()[2] not in pushed:
                out.append(dict(repo=repo, blob=meta.split()[2], path=path, via=f"unreachable {c[:10]}"))
    return out


def search(roots, P, committed, fsck=True):
    rep = dict(roots=roots, names=[], committed_states=0, dirs=[], likely=[], weak=[], git=[], repos=0, files=0)
    stores, tracked = set(), set()
    for kind, p in walk(roots):
        if kind == "dir":
            rep["dirs"].append(p)
        elif kind == "repo":
            rep["repos"] += 1
            tracked.update(os.path.join(p, f) for f in git(p, "ls-files", "-z").split("\0") if f)
            rep["git"] += scan_git(p, stores, fsck)
        elif p not in tracked:
            rep["files"] += 1
            b = os.path.basename(p)
            if cloud_name(b):
                if b in committed and os.sep + os.path.join("hp", "data") + os.sep in p:
                    rep["committed_states"] += 1
                else:
                    rep["names"].append(p)
            if b.endswith(TEXT) and not b.startswith("delta_dd"):
                h = scan_text(p, P)
                if len(h) >= 5:
                    rep["likely"].append(dict(path=p, probes=len(h)))
                elif h:
                    rep["weak"].append(dict(path=p, probes=len(h)))
    seen, uniq = set(), []
    for g in rep["git"]:
        if g["blob"] not in seen:
            seen.add(g["blob"])
            uniq.append(g)
    rep["git"] = uniq
    return rep


def collect(rep, dest):
    os.makedirs(dest, exist_ok=True)
    man = []
    for i, p in enumerate(rep["names"] + [x["path"] for x in rep["likely"]]):
        q = os.path.join(dest, "files", f"{i:04d}_{os.path.basename(p)}")
        os.makedirs(os.path.dirname(q), exist_ok=True)
        shutil.copy2(p, q)
        man.append(dict(copy=os.path.relpath(q, dest), source=p))
    for i, g in enumerate(rep["git"]):
        q = os.path.join(dest, "git", f"{i:04d}_{os.path.basename(g['path'])}")
        os.makedirs(os.path.dirname(q), exist_ok=True)
        with open(q, "wb") as fh:
            fh.write(subprocess.run(["git", "-C", g["repo"], "cat-file", "-p", g["blob"]], capture_output=True).stdout)
        man.append(dict(copy=os.path.relpath(q, dest), source=f"{g['repo']} {g['via']} {g['path']} blob {g['blob']}"))
    with open(os.path.join(dest, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    return len(man)


def committed_names():
    out = set()
    for sub in ("states", "seeds"):
        for d, _, fs in os.walk(os.path.join(HERE, "data", sub)):
            out.update(f for f in fs if cloud_name(f))
    return out


def selftest(P, mutate=""):
    t = tempfile.mkdtemp(prefix="find_lost_scans_")
    try:
        some = sorted(P)[:6]
        os.makedirs(os.path.join(t, "a", "b"))
        with open(os.path.join(t, "a", "b", "copied_records.jsonl"), "w") as fh:
            fh.writelines(json.dumps({"D": 10.0 + i, "Delta": float(v + "37")}) + "\n" for i, v in enumerate(some))
        with open(os.path.join(t, "a", "decoy.jsonl"), "w") as fh:
            fh.write(json.dumps({"D": 9.0, "Delta": 1.23456789012345}) + "\n")
        open(os.path.join(t, "a", "up7_D41.0890.npz"), "wb").close()
        g = os.path.join(t, "repo")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        run = lambda *a: subprocess.run(["git", "-C", g, *a], check=True, capture_output=True, env=env)  # noqa: E731
        os.makedirs(g)
        run("init", "-q", "-b", "main")
        open(os.path.join(g, "x"), "w").close()
        run("add", "x"), run("commit", "-qm", "x")
        run("checkout", "-qb", "gone")
        open(os.path.join(g, "scan_up6.jsonl"), "w").write("{}\n")
        run("add", "scan_up6.jsonl"), run("commit", "-qm", "s")
        run("checkout", "-q", "main"), run("branch", "-qD", "gone")
        run("reflog", "expire", "--expire=now", "--all")
        rep = search([t], set() if mutate == "no_probes" else P, set(), fsck=mutate != "no_fsck")
        ok = dict(record_file=any(x["path"].endswith("copied_records.jsonl") for x in rep["likely"]),
                  state_name=any(p.endswith("up7_D41.0890.npz") for p in rep["names"]),
                  deleted_branch=any(x["path"] == "scan_up6.jsonl" and x["via"].startswith("unreachable") for x in rep["git"]),
                  decoy_clean=not any("decoy" in x["path"] for x in rep["likely"] + rep["weak"]))
        for k, v in ok.items():
            print(f"  {k}: {'PASS' if v else 'FIRE'}")
        return all(ok.values())
    finally:
        shutil.rmtree(t, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--roots", nargs="*", default=[os.path.expanduser("~"), tempfile.gettempdir()])
    ap.add_argument("--collect", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", default="", choices=["", "no_probes", "no_fsck"])
    a = ap.parse_args()
    P, n = probes(os.path.join(HERE, "data", "delta_dd.csv"))
    print(f"{n} cloud-run rows, {len(P)} probes")
    if a.selftest:
        ok = selftest(P, a.mutate)
        print(f"selftest: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    if not a.out:
        ap.error("OUT.json is required without --selftest")
    rep = search(a.roots, P, committed_names())
    if a.collect:
        rep["collected"] = collect(rep, a.collect)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"searched {rep['files']} files and {rep['repos']} repositories under {', '.join(a.roots)}")
    print(f"  cloud-run file names: {len(rep['names'])} (states already in git: {rep['committed_states']})")
    print(f"  hp_scan directories: {len(rep['dirs'])}")
    print(f"  files with a cloud-run Delta: likely (>= 5 probes) {len(rep['likely'])}, weak {len(rep['weak'])}")
    print(f"  git objects (refs, reflogs, unreachable commits): {len(rep['git'])}")
    for x in rep["names"][:20] + rep["dirs"][:20]:
        print(f"    {x}")
    for x in rep["likely"][:20]:
        print(f"    {x['probes']:4d} probes  {x['path']}")
    for x in rep["git"][:20]:
        print(f"    git {x['via']}: {x['path']} in {x['repo']}")
    if a.collect:
        print(f"collected {rep['collected']} files into {a.collect}")


if __name__ == "__main__":
    main()
