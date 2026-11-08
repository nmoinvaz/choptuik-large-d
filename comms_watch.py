#!/usr/bin/env python3
"""Wait for a message or a task addressed to one machine, then exit so its Claude Code session acts on it.

A session starts this as a background command; it costs nothing while it waits. Every --interval seconds it checks,
in this order:

- the Coordination issue of --repo (the open issue labelled "coordination", AGENTS.md "Coordination"): a comment
  whose first line is "<sender> to <recipients>" naming --name or "all", newer than the last one it printed. It prints
  those comments in full, records the newest as handled (in the state file, by default in the checkout's git
  directory), and exits 0;
- the task issues (AGENTS.md, "Tasks: the GitHub project"): the open issues labelled --name and not "running". It
  prints them and exits 0. A runner adds "running" when it starts a task, so a task wakes it once;
- the archive comms.txt on origin/main, for entries addressed to --name after those in HEAD. comms.txt takes no new
  entries since 2026-11-08; this check remains for an entry written by an old habit.

After --max-minutes with nothing for it, it exits 3 with "nothing new", and the session simply restarts it (Claude
Code stops a background command after 2 h, so keep the limit below that). The coordinator watches with
--name coordinator; a comment addressed to its old name, cloud, reaches it too.
If gh is missing or not logged in, it says so once and watches comms.txt only. Standard library, git and gh only.

    python3 -u comms_watch.py --name coordinator|win|win2|macm3|mac [--interval 60] [--max-minutes 110]
        [--repo nmoinvaz/choptuik-large-d] [--no-issues] [--state PATH]
    python3 comms_watch.py --selftest [--mutate=parse]     # the header parser on fixed cases; exits 1 on a failure
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = re.compile(r"^== \S+ \S+ (?:UTC|PDT) (.+?)\s*$")


def git(*args):
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, check=True).stdout


def headers(text):
    return [line.rstrip() for line in text.splitlines() if line.startswith("== ")]


def addressed(header, name):
    """True for a cloud entry whose recipient list names this runner."""
    m = HEADER.match(header)
    if not m or not m.group(1).startswith("cloud to "):
        return False
    return name in [r.strip() for r in m.group(1)[len("cloud to "):].split(",")]


COORD_HEADER = re.compile(r"^[\s*#>_]*([A-Za-z0-9]+)\s+to\s+([A-Za-z0-9,\s]+?)[\s*_:.]*$")


def coord_addressed(body, name, mutate=None):
    """True when the comment's first line is '<sender> to <recipients>' naming name or 'all' (and not sent by name)"""
    first = (body or "").strip().splitlines()[0] if (body or "").strip() else ""
    m = COORD_HEADER.match(first)
    if not m:
        return False
    sender, recips = m.group(1), [r.strip() for r in m.group(2).split(",") if r.strip()]
    if mutate == "parse":
        recips = recips[:1]
    names = {name, "cloud"} if name == "coordinator" else {name}  # cloud: the coordinator's name before 2026-11-08
    return sender not in names and (bool(names & set(recips)) or "all" in recips)


def selftest(mutate=None):
    cases = [("cloud to win2, win, macm3\nbody", "win", True), ("cloud to win2\nbody", "win", False),
             ("**coordinator to all**\nbody", "macm3", True), ("win2 to coordinator\n...", "coordinator", True),
             ("win2 to cloud\n...", "coordinator", True), ("coordinator to all\n", "coordinator", False),
             ("win2 to cloud\n...", "win2", False), ("no header here", "win", False),
             ("cloud to win, macm3:\n", "macm3", True), ("", "win", False)]
    bad = [(b, n, e) for b, n, e in cases if coord_addressed(b, n, mutate) != e]
    for b, n, e in bad:
        print(f"selftest: {b.splitlines()[0] if b else repr(b)!s} for {n}: expected {e}")
    print(f"selftest: {len(cases) - len(bad)} of {len(cases)} cases pass")
    return 1 if bad else 0


def coord_issue(repo):
    """the number of the open issue labelled coordination; None when gh cannot be used or there is none"""
    r = subprocess.run(["gh", "issue", "list", "--repo", repo, "--label", "coordination", "--state", "open",
                        "--limit", "5", "--json", "number"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    nums = sorted(i["number"] for i in json.loads(r.stdout or "[]"))
    return nums[0] if nums else None


def coord_comments(repo, num):
    """the comments of the Coordination issue as dicts (id, t, body), oldest first; None when gh fails"""
    r = subprocess.run(["gh", "api", "--paginate", f"repos/{repo}/issues/{num}/comments?per_page=100",
                        "--jq", ".[] | {id: .id, t: .created_at, body: .body} | @json"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return [json.loads(line) for line in r.stdout.splitlines() if line.strip()]


def ready_issues(repo, name):
    """the open issues labelled name and not running, as 'number title' lines; None when gh cannot be used"""
    r = subprocess.run(["gh", "issue", "list", "--repo", repo, "--label", name, "--state", "open", "--limit", "50",
                        "--json", "number,title,labels"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    out = []
    for i in json.loads(r.stdout or "[]"):
        if "running" not in [lb["name"] for lb in i.get("labels", [])]:
            out.append(f"#{i['number']} {i['title']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required="--selftest" not in sys.argv)
    ap.add_argument("--repo", default="nmoinvaz/choptuik-large-d")
    ap.add_argument("--no-issues", action="store_true")
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--max-minutes", type=float, default=110)
    ap.add_argument("--state")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(a.mutate))
    if not a.state:
        gd = git("rev-parse", "--git-dir").strip()
        a.state = os.path.join(gd if os.path.isabs(gd) else os.path.join(HERE, gd), f"comms_watch_{a.name}.json")
    try:
        with open(a.state) as fh:
            last = json.load(fh).get("last", "")
    except (OSError, ValueError):
        last = ""
    handled = len(headers(git("show", "HEAD:comms.txt")))
    end = time.time() + 60 * a.max_minutes
    coord = None if a.no_issues else coord_issue(a.repo)
    where = f"the Coordination issue #{coord} after {last or 'its start'}, the task issues, and " if coord else ""
    print(f"watching for {a.name}: {where}origin/main comms.txt after the {handled} entries in HEAD", flush=True)
    while True:
        if coord is not None:
            cs = coord_comments(a.repo, coord)
            mine = [c for c in (cs or []) if c["t"] > last and coord_addressed(c["body"], a.name)]
            if mine:
                print(f"new messages for {a.name} on the Coordination issue #{coord}:", flush=True)
                for c in mine:
                    print(f"--- {c['t']} (comment {c['id']})\n{c['body'][:6000]}", flush=True)
                with open(a.state, "w") as fh:
                    json.dump({"last": max(c["t"] for c in mine)}, fh)
                sys.exit(0)
        try:
            git("fetch", "-q", "origin", "main")
            new = headers(git("show", "origin/main:comms.txt"))[handled:]
        except subprocess.CalledProcessError as e:
            print(f"git failed ({e.stderr.strip()[:200]}); retrying", flush=True)
            new = []
        mine = [h for h in new if addressed(h, a.name)]
        if mine:
            print("new comms.txt entries for " + a.name + " (the archive; reply on the Coordination issue):\n"
                  + "\n".join(mine), flush=True)
            sys.exit(0)
        if not a.no_issues:
            tasks = ready_issues(a.repo, a.name)
            if tasks is None:
                print("gh cannot list issues (missing or not logged in); watching comms.txt only", flush=True)
                a.no_issues = True
            elif tasks:
                print(f"tasks labelled {a.name} and not running:\n" + "\n".join(tasks), flush=True)
                sys.exit(0)
        if time.time() >= end:
            print(f"nothing new for {a.name} in {a.max_minutes:g} min", flush=True)
            sys.exit(3)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
