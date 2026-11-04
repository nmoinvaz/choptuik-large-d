#!/usr/bin/env python3
"""Resolve every markdown link in the given files (default: README.md and every notes/*.md): a relative file must
exist, and a #anchor must match a heading of the target file (GitHub's rule: lower case, punctuation other than
hyphens and spaces dropped, spaces to hyphens, repeated anchors numbered). Exits 1 on a broken link.

    python3 check_links.py [FILE ...]
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def anchors(path):
    seen, out = {}, set()
    for line in open(path, encoding="utf-8"):
        if not line.startswith("#"):
            continue
        h = line.lstrip("#").strip().lower()
        h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", h)
        h = re.sub(r"<[^>]+>", "", h)
        h = re.sub(r"[^\w\s-]", "", h)
        a = re.sub(r"\s", "-", h)
        n = seen.get(a, 0)
        seen[a] = n + 1
        out.add(a if n == 0 else f"{a}-{n}")
    for m in re.finditer(r'<a (?:id|name)="([^"]+)"', open(path, encoding="utf-8").read()):
        out.add(m.group(1))
    return out


def main():
    files = sys.argv[1:] or ["README.md"] + sorted(glob.glob(os.path.join(ROOT, "notes", "*.md")))
    bad, n = [], 0
    cache = {}
    for f in files:
        f = f if os.path.isabs(f) else os.path.join(ROOT, f)
        text = open(f, encoding="utf-8").read()
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for m in re.finditer(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", text):
            t = m.group(1)
            if t.startswith(("http://", "https://", "mailto:")):
                continue
            n += 1
            path, _, frag = t.partition("#")
            target = f if path == "" else os.path.normpath(os.path.join(os.path.dirname(f), path))
            if not os.path.exists(target):
                bad.append((os.path.relpath(f, ROOT), t, "missing file")); continue
            if frag and target.endswith(".md"):
                if target not in cache:
                    cache[target] = anchors(target)
                if frag not in cache[target]:
                    bad.append((os.path.relpath(f, ROOT), t, "no such anchor"))
    for f, t, why in bad:
        print(f"{f}: {t}: {why}")
    print(f"{n} links checked, {len(bad)} broken")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
