#!/usr/bin/env python3
"""Re-wrap over-long prose lines of the markdown notes at 120 characters (AGENTS.md, "Writing").

    python3 wrap_prose.py [--write] [--width 120] FILE ...        default: notes/*.md

Without --write it lists every prose line over the width and exits 1 if there is one. With --write it re-wraps only
the paragraphs that hold such a line, leaving every other line byte for byte. Left alone: fenced blocks (``` and
```math), tables, images, HTML, headings, lines with URLs and footnote definitions. Inline math $`...`$, code
`...` spans and links are never split across lines. Phrases that check_numbers.py pins must stay findable: its single-line string
literals are kept on one line, and a paragraph that contains one of its multi-line literals (a pin that encodes the
present line break) is not touched."""
import ast
import glob
import os
import re
import sys
import textwrap

ROOT = os.path.dirname(os.path.abspath(__file__))
SPAN = re.compile(r"\$`.*?`\$|`[^`\n]*`|\[[^\]\n]*\]\([^)\n]*\)")   # math, code, links stay on one line
ITEM = re.compile(r"^(\s*)([-*]|\d+\.)\s+")
NB = "\x00"


def pins():
    """(single-line literals, multi-line literals) of check_numbers.py"""
    tree = ast.parse(open(os.path.join(ROOT, "check_numbers.py"), encoding="utf-8").read())
    lits = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and " " in n.value and len(n.value) >= 12]
    return [s for s in lits if "\n" not in s], [s for s in lits if "\n" in s]


def bare(line):
    """the line without its quote markers and indentation"""
    return re.sub(r"^(\s*>)+", "", line).lstrip()


def skip(line):
    s = bare(line)
    return (not s or s.startswith(("|", "![", "<", "#", "[^")) or "http" in line)


def paragraphs(lines):
    """(start, end) of each prose paragraph, fences and skipped lines excluded; a list item starts a paragraph"""
    out, fence, start = [], False, None
    for i, l in enumerate(lines + [""]):
        if bare(l).startswith("```"):
            fence = not fence
            if start is not None:
                out.append((start, i)); start = None
            continue
        if fence or skip(l):
            if start is not None:
                out.append((start, i)); start = None
            continue
        if start is None:
            start = i
        elif ITEM.match(l) or (l.startswith(">") != lines[start].startswith(">")) or (not l.startswith((" ", ">")) and lines[i - 1].startswith((" ", ">")) and ITEM.match(lines[start])):
            out.append((start, i)); start = i
    return out


def rewrap(block, width, protect):
    """block: the paragraph's lines; returns the re-wrapped lines"""
    first = block[0]
    quote = first.startswith(">")
    body = [l[1:].lstrip() if quote else l for l in block]
    m = ITEM.match(body[0])
    if m:
        head, cont = m.group(0), " " * len(m.group(0))
    else:
        ind = re.match(r"\s*", body[0]).group(0)
        head, cont = ind, ind
    text = " ".join([body[0][len(head):].strip()] + [l.strip() for l in body[1:]])
    for s in SPAN.findall(text):
        text = text.replace(s, s.replace(" ", NB))
    for p in protect:
        if p in text:
            text = text.replace(p, p.replace(" ", NB))
    pre = "> " if quote else ""
    w = width - len(pre)
    out = textwrap.wrap(text, width=w, initial_indent=head, subsequent_indent=cont, break_long_words=False, break_on_hyphens=False)
    return [pre + l.replace(NB, " ") for l in out]


def main():
    args = sys.argv[1:]
    write = "--write" in args
    width = 120
    if "--width" in args:
        i = args.index("--width"); width = int(args[i + 1]); del args[i:i + 2]
    files = [a for a in args if not a.startswith("--")] or sorted(glob.glob(os.path.join(ROOT, "notes", "*.md")))
    single, multi = pins()
    n_long, n_wrapped, n_kept = 0, 0, 0
    for f in files:
        text = open(f, encoding="utf-8").read()
        lines = text.split("\n")
        new = list(lines)
        shift = 0
        for a, b in paragraphs(lines):
            block = lines[a:b]
            if max(len(l) for l in block) <= width:
                continue
            n_long += sum(len(l) > width for l in block)
            if not write:
                for k, l in enumerate(block):
                    if len(l) > width:
                        print(f"{os.path.relpath(f, ROOT)}:{a + k + 1}: {len(l)}")
                continue
            joined = "\n".join(block)
            if any(p in joined or p in "\n".join(lines[max(0, a - 1):b + 1]) for p in multi):
                n_kept += 1
                continue
            rep = rewrap(block, width, single)
            new[a + shift:b + shift] = rep
            shift += len(rep) - (b - a)
            n_wrapped += 1
        if write and new != lines:
            open(f, "w", encoding="utf-8").write("\n".join(new))
    if write:
        print(f"{n_wrapped} paragraphs re-wrapped, {n_kept} left for their multi-line pins")
    else:
        print(f"{n_long} prose lines over {width}")
        sys.exit(1 if n_long else 0)


if __name__ == "__main__":
    main()
