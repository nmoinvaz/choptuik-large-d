#!/usr/bin/env python3
"""Wait for comms.txt entries addressed to one runner, then exit so its Claude Code session acts on them.

A local runner's session starts this as a background command; it costs nothing while it waits. Every --interval
seconds it fetches origin/main and compares origin's comms.txt with the one in the local checkout's HEAD (what the
session last pulled and handled). When origin has new entries addressed to --name ("cloud to win", or a list such as
"cloud to win, mac"), it prints their headers and exits 0; the session then pulls, handles them, and starts it again.
After --max-minutes with nothing addressed to it, it exits 3 with "nothing new", and the session simply restarts it
(Claude Code stops a background command after 2 h, so keep the limit below that). Standard library and git only.

    python3 -u comms_watch.py --name win|mac [--interval 60] [--max-minutes 110]
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--max-minutes", type=float, default=110)
    a = ap.parse_args()
    handled = len(headers(git("show", "HEAD:comms.txt")))
    end = time.time() + 60 * a.max_minutes
    print(f"watching origin/main comms.txt for entries to {a.name} after the {handled} in HEAD", flush=True)
    while True:
        try:
            git("fetch", "-q", "origin", "main")
            new = headers(git("show", "origin/main:comms.txt"))[handled:]
        except subprocess.CalledProcessError as e:
            print(f"git failed ({e.stderr.strip()[:200]}); retrying", flush=True)
            new = []
        mine = [h for h in new if addressed(h, a.name)]
        if mine:
            print("new entries for " + a.name + ":\n" + "\n".join(mine), flush=True)
            sys.exit(0)
        if time.time() >= end:
            print(f"nothing new for {a.name} in {a.max_minutes:g} min", flush=True)
            sys.exit(3)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
