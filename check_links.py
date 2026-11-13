#!/usr/bin/env python3
"""Resolve every markdown link in the given files, by default README.md and every notes/*.md. A relative file must
exist, and a #anchor must match a heading of the target file or an <a id> or <a name> in it. Exits 1 on a broken
link.

    python3 check_links.py [FILE ...]
    python3 check_links.py --selftest [--mutate=keep_no|no_proposal|proposal_everywhere]

Heading anchors follow github-slugger 2.0.0 (github.com/Flet/github-slugger at 3461c43, index.js and
script/generate-regex.js), which slugs headings as GitHub does. It lower-cases the heading's text, drops every
character of the general categories No, Pd, Ps, Pe, Pi, Pf, Po, S, Cc, Cf, Co, Cn, Zs, Zl and Zp other than ' ', '-'
and the Alphabetic ones, and turns each ' ' into '-'. A repeated anchor a becomes a-1, a-2 and so on, skipping any
anchor already taken. Subscript and superscript digits (No) are dropped, so "## 3. Lemma 6b₀" gives #3-lemma-6b.
Letters of any script, marks, decimal digits and '_' (Pc) are kept. GitHub's docs give the rule in words ("Basic
writing and formatting syntax", Section links, docs.github.com). The heading's text is its rendered text. Emphasis,
link targets, images and HTML tags are dropped, and code spans, entities and footnote labels give their text. The
rule reproduces the anchors GitHub rendered for all 1724 headings of this repository's .md files at 52433d0d, and
for a planted file of edge cases that the selftest keeps (#661).

A link is a CommonMark inline link or image, [text](dest "title"), outside fenced code, code spans and HTML comments.
Text that GitHub does not render as a link, such as "[0, h](F_y(Yt)Pt + B)", is not read as one.

Quoted proposal text is a blockquote line, or a double-quoted span of a paragraph, in a section whose heading says
"proposed text", "proposed wording" or "proposed entries" with at most two words between (as in "Proposed README
text"). It is meant to be pasted into another file, usually one at the repository root, and its links are written
relative to that file. A link in it also passes when it resolves from the repository root, and fails when it
resolves from neither place. Each --mutate control breaks one rule, and the selftest must then fail.
"""
import bisect
import glob
import html
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from urllib.parse import unquote

ROOT = os.path.dirname(os.path.abspath(__file__))
MUTATE = ""  # selftest controls: keep_no, no_proposal, proposal_everywhere

# General categories that github-slugger's script/generate-regex.js strips.
DROP = {"No", "Pd", "Ps", "Pe", "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So", "Cc", "Cf", "Co", "Cn", "Zs", "Zl", "Zp"}
# The Alphabetic code points among them, all So, which it keeps (circled and squared Latin letters).
# With these, slug_drops agrees with its regex.js on every code point but 10103 that its Unicode 13 tables strip
# and Python's Unicode 16 tables class as letters, marks or decimal digits (#661).
ALPHA_SO = ((0x24B6, 0x24E9), (0x1F130, 0x1F149), (0x1F150, 0x1F169), (0x1F170, 0x1F189))


def slug_drops(c):
    if c in " -":
        return False
    cat = unicodedata.category(c)
    if MUTATE == "keep_no" and cat == "No":
        return False
    if cat not in DROP:
        return False
    return not any(a <= ord(c) <= b for a, b in ALPHA_SO)


def slug(text):
    return "".join(c for c in text.lower() if not slug_drops(c)).replace(" ", "-")


CODE_SPAN = re.compile(r"(?<![`\\])(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.S)
ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")
FENCE = re.compile(r"^(?:[ ]{0,3}>[ ]?)*[ \t]*(`{3,}|~{3,})(.*)$")
ATX = re.compile(r"^(?:[ ]{0,3}>[ ]?)*(?:[ ]{0,3}(?:[-*+]|\d{1,9}[.)])[ \t]+)?[ ]{0,3}(#{1,6})(?=[ \t]|$)(.*)$")
SETEXT = re.compile(r"^[ ]{0,3}(=+|-+)[ \t]*$")
NOT_PARA = re.compile(r"^(?:[ ]{4}|\s*(?:[>|#<]|[-*+][ \t]|\d{1,9}[.)][ \t]|(?:[-*_][ \t]*){3,}$))")
BLOCKQUOTE = re.compile(r"^[ ]{0,3}>")
PROPOSAL = re.compile(r"\bproposed(?:\s+\S+){0,2}?\s+(?:text|wording|entries)\b", re.I)


def fenced(lines):
    """Indices of the lines inside fenced code blocks, fence lines included."""
    out, fence = set(), None
    for i, line in enumerate(lines):
        m = FENCE.match(line)
        if fence is None:
            if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
                fence = m.group(1)
                out.add(i)
        else:
            out.add(i)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not m.group(2).strip():
                fence = None
    return out


def code_text(s):
    """A code span's content as CommonMark renders it."""
    s = s.replace("\n", " ")
    return s[1:-1] if len(s) > 2 and s[0] == " " and s[-1] == " " and s.strip() else s


def heading_text(s):
    """Rendered text of a heading's inline markdown."""
    keep = []

    def hold(t):
        keep.append(t)
        return f"\x00{len(keep) - 1}\x00"

    s = CODE_SPAN.sub(lambda m: hold(code_text(m.group(2))), s)
    s = ESCAPE.sub(lambda m: hold(m.group(1)), s)
    s = re.sub(r"!\[[^\]]*\]\((?:[^()]|\([^()]*\))*\)", "", s)        # images: alt text is not rendered text
    s = re.sub(r"\[\^([^\]]+)\]", r"\1", s)                            # footnote reference
    s = re.sub(r"\[([^\]]*)\]\((?:[^()]|\([^()]*\))*\)", r"\1", s)     # inline link
    s = re.sub(r"<!--.*?-->|</?[A-Za-z][^>]*>", "", s)                 # HTML
    for _ in range(2):                                                 # _emphasis_ and __strong__
        s = re.sub(r"(?<!\w)(_+)(?=\S)(.+?)(?<=\S)\1(?!\w)", r"\2", s)
    s = html.unescape(s)
    return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)


def headings(lines, code):
    """(line index, level, text) of every ATX and setext heading."""
    out, para = [], None
    for i, line in enumerate(lines):
        if i in code or not line.strip():
            para = None
            continue
        m = ATX.match(line)
        if m:
            t = re.sub(r"(?:^|[ \t]+)#+[ \t]*$", "", m.group(2).strip()).strip()
            out.append((i, len(m.group(1)), t))
            para = None
            continue
        m = SETEXT.match(line)
        if m and para is not None:
            t = " ".join(x.strip() for x in lines[para:i])
            out.append((para, 1 if m.group(1)[0] == "=" else 2, t))
            para = None
            continue
        if para is None and not NOT_PARA.match(line):
            para = i
        elif NOT_PARA.match(line):
            para = None
    return out


def anchors(path):
    """The heading anchors of path in document order, and its explicit <a id> and <a name> anchors."""
    text = open(path, encoding="utf-8").read()
    lines = text.split("\n")
    code = fenced(lines)
    seen, heads = {}, []
    for _, _, t in headings(lines, code):
        base = a = slug(heading_text(t))
        while a in seen:
            seen[base] += 1
            a = f"{base}-{seen[base]}"
        seen[a] = 0
        heads.append(a)
    t, _ = mask(text, lines, code)
    return heads, re.findall(r"<a\s[^>]*?\b(?:id|name)=[\"']([^\"']+)[\"']", t)


def mask(text, lines, code):
    """text with fenced code, HTML comments and code spans blanked, keeping every offset, and the start offsets of
    its paragraphs (runs of lines that are not blank)."""
    t = "\n".join(" " * len(x) if i in code else x for i, x in enumerate(lines))
    blank = lambda m: re.sub(r"[^\n]", " ", m.group(0))
    t = re.sub(r"<!--.*?-->", blank, t, flags=re.S)
    out, pos, paras = [], 0, []
    for b in re.finditer(r"(?:[^\n]*\S[^\n]*(?:\n|$))+", t):                # code spans stay within a paragraph
        out.append(t[pos:b.start()])
        out.append(CODE_SPAN.sub(blank, b.group(0)))
        pos = b.end()
        paras.append(b.start())
    out.append(t[pos:])
    assert len("".join(out)) == len(text)
    return "".join(out), paras


def escaped(t, i):
    n = 0
    while i - 1 - n >= 0 and t[i - 1 - n] == "\\":
        n += 1
    return n % 2 == 1


def link_start(t, close, para):
    """Index of the '[' that opens the link text ending at t[close] == ']', searching back to para, or None."""
    depth, i = 0, close - 1
    while i >= para:
        c = t[i]
        if c in "[]" and not escaped(t, i):
            if c == "]":
                depth += 1
            elif depth == 0:
                return i
            else:
                depth -= 1
        i -= 1
    return None


def skip_space(t, i):
    nl = 0
    while i < len(t) and t[i] in " \t\n":
        if t[i] == "\n":
            nl += 1
            if nl > 1:
                return None
        i += 1
    return i


def destination(t, i):
    """Parse '(dest "title")' from t[i], just after the '('. Returns dest or None if it is not a link."""
    i = skip_space(t, i)
    if i is None or i >= len(t):
        return None
    if t[i] == "<":
        j = i + 1
        while j < len(t) and t[j] not in "<>\n":
            j += 2 if t[j] == "\\" else 1
        if j >= len(t) or t[j] != ">":
            return None
        dest, j = t[i + 1:j], j + 1
    else:
        depth, j = 0, i
        while j < len(t) and not t[j].isspace() and ord(t[j]) >= 0x20:
            if t[j] == "\\":
                j += 2
                continue
            if t[j] == "(":
                depth += 1
            elif t[j] == ")":
                if depth == 0:
                    break
                depth -= 1
            j += 1
        if depth:
            return None
        dest = t[i:j]
    k = skip_space(t, j)
    if k is None or k >= len(t):
        return None
    if t[k] != ")":
        if k == j or t[k] not in "\"'(":
            return None
        close = ")" if t[k] == "(" else t[k]
        k += 1
        while k < len(t) and t[k] != close:
            if t[k] == "\n" and t[k + 1:k + 2] == "\n":
                return None
            k += 2 if t[k] == "\\" else 1
        k = skip_space(t, k + 1)
        if k is None or k >= len(t) or t[k] != ")":
            return None
    return ESCAPE.sub(r"\1", dest)


def links(path):
    """(line number, dest, in quoted proposal text) of every inline link and image in path."""
    text = open(path, encoding="utf-8").read()
    lines = text.split("\n")
    code = fenced(lines)
    t, paras = mask(text, lines, code)
    starts = [0]
    for x in lines[:-1]:
        starts.append(starts[-1] + len(x) + 1)
    proposal, stack = [False] * len(lines), []
    hs = {i: (lv, h) for i, lv, h in headings(lines, code) if not BLOCKQUOTE.match(lines[i])}  # quoted ones are text
    for i in range(len(lines)):
        if i in hs:
            lv, h = hs[i]
            while stack and stack[-1][0] >= lv:
                stack.pop()
            stack.append((lv, bool(PROPOSAL.search(h))))
        proposal[i] = any(p for _, p in stack)
    out = []
    for m in re.finditer(r"\]\(", t):
        if escaped(t, m.start()):
            continue
        para = paras[bisect.bisect_right(paras, m.start()) - 1] if paras else 0
        s = link_start(t, m.start(), para)
        dest = None if s is None else destination(t, m.end())
        if dest is None:
            continue
        ln = bisect.bisect_right(starts, s) - 1
        before = t[para:s]                                                 # its paragraph up to the link
        quoted = (bool(BLOCKQUOTE.match(lines[ln])) or before.count('"') % 2 == 1
                  or before.count("“") > before.count("”"))
        prop = MUTATE == "proposal_everywhere" or (MUTATE != "no_proposal" and proposal[ln] and quoted)
        out.append((ln + 1, dest, prop))
    return out


def check(files, root):
    """Returns (number of links checked, number in quoted proposal text, [(file, line, target, why)])."""
    bad, n, nprop, cache = [], 0, 0, {}
    for f in files:
        f = f if os.path.isabs(f) else os.path.join(root, f)
        for ln, t, prop in links(f):
            if t.startswith(("http://", "https://", "mailto:")):
                continue
            n += 1
            nprop += prop
            path, _, frag = t.partition("#")
            path, frag = unquote(path.split("?")[0]), unquote(frag)
            bases = [os.path.dirname(f)] + ([root] if prop and path else [])
            why = None
            for base in bases:
                target = f if path == "" else os.path.normpath(os.path.join(base, path))
                if not os.path.exists(target):
                    why = why or "missing file"
                    continue
                if frag and target.endswith(".md"):
                    if target not in cache:
                        heads, named = anchors(target)
                        cache[target] = set(heads) | set(named)
                    if frag not in cache[target]:
                        why = "no such anchor"
                        continue
                why = None
                break
            if why:
                bad.append((os.path.relpath(f, root), ln, t, why))
    return n, nprop, bad


# Planted headings and the anchors GitHub rendered for them (notes/PLANT_661.md at c61e711, a temporary commit of
# wip/i661-subagent, through the contents API with Accept: application/vnd.github.html, 2026-11-13).
PLANT = """# Plant for check_links.py (#661), temporary

## 3. Lemma 6b₀

## 3. Lemma 6b₀

## Δ∞ and ω∞, ν′ ± 10⁻⁸ × 2

## Σ, Ω and ΔΓ upper case Ä

## Lemma 6b₁ — the `disc` *cover* [link](x.md) – end

## ℓ = 1 past D = 9: k_c 23 → 31 (§4.4) ≥ ½

## <a id="x"></a>Tag  two  spaces

## $`\\Delta_\\infty`$ math

## Ⅻ Ⓐ é ü İ ß 😀 n₀

## _emph_ and __strong__ and snake_case_name and *star*

## Escapes \\_x\\_ and \\* and &amp; &lt; and &#916;

## Footnote[^1]

## Image ![alt text](img.png) here

## <sub>sub</sub> and <code>tag</code> html

## a

## a-1

## a

## Closing hashes ##

#NoSpace

> ## Quoted heading

   ### Indented three

    #### Indented four

- ## In a list item

Setext heading
---

Setext one
===

```bash
# comment in code
```

~~~
## tilde fenced
~~~

## Tab\tinside and trailing\x20\x20\x20

## Code span `` a`b `` and `x_y_z`

## Strike ~~gone~~ and ~one~

## Emoji :smile: and :+1:

[^1]: A note.
"""
PLANT_GITHUB = """plant-for-check_linkspy-661-temporary 3-lemma-6b 3-lemma-6b-1 δ-and-ω-ν--10--2
σ-ω-and-δγ-upper-case-ä lemma-6b--the-disc-cover-link--end ℓ--1-past-d--9-k_c-23--31-44-- tag--two--spaces
delta_infty-math ⅻ-ⓐ-é-ü-i̇-ß--n emph-and-strong-and-snake_case_name-and-star escapes-_x_-and--and---and-δ footnote1
image--here sub-and-tag-html a a-1 a-2 closing-hashes quoted-heading indented-three in-a-list-item setext-heading
setext-one tabinside-and-trailing code-span-ab-and-x_y_z strike-gone-and-one emoji-smile-and-1""".split()

# Links planted in notes/B.md. Each line: expected verdict (ok, or the reason it must fail), then the text.
LINKS = """\
# B

## 1. Anchors

ok: [lemma](A.md#3-lemma-6b), the anchor GitHub renders for "## 3. Lemma 6b₀"
ok: [repeat](A.md#3-lemma-6b-1)
no such anchor: [old rule](A.md#3-lemma-6b₀)
no such anchor: [wrong](A.md#3-lemma-6c)
no such anchor: [in a fence](A.md#comment-in-code)
ok: [explicit](A.md#x)
missing file: [rooted](proof/X.md), outside quoted proposal text
missing file: [ω](D), math that GitHub renders as a link
not a link: [0, h](F_y(Yt)Pt + B) and `[code](nope.md)`

```
not a link: [fenced](nope.md)
```

> missing file: [quoted, no proposal](proof/X.md)

## 9. Proposed text for the README

missing file: [instructions, not quoted](proof/X.md)

> ok: [rooted](proof/X.md), quoted in a blockquote
> missing file: [broken from both](proof/Nope.md)

Replace "the old sentence" by "ok: [rooted](proof/X.md), in double quotes".

### 9.1 A subsection of the proposal

> ok: [rooted, in a subsection](proof/X.md)

## 10. After the proposal

> missing file: [quoted, after the section](proof/X.md)
"""


def selftest():
    d = tempfile.mkdtemp(prefix="check_links_selftest_")
    try:
        os.makedirs(os.path.join(d, "notes"))
        os.makedirs(os.path.join(d, "proof"))
        a = os.path.join(d, "notes", "A.md")
        with open(a, "w", encoding="utf-8") as fh:
            fh.write(PLANT)
        open(os.path.join(d, "proof", "X.md"), "w").write("# X\n")
        b = os.path.join(d, "notes", "B.md")
        with open(b, "w", encoding="utf-8") as fh:
            fh.write(LINKS)
        fails = 0
        heads, _ = anchors(a)
        for i in range(max(len(heads), len(PLANT_GITHUB))):
            g = heads[i] if i < len(heads) else None
            w = PLANT_GITHUB[i] if i < len(PLANT_GITHUB) else None
            if g != w:
                print(f"FAIL anchor {i}: GitHub {w!r}, check_links {g!r}")
                fails += 1
        expect = {}
        for i, line in enumerate(LINKS.split("\n")):
            m = re.match(r"^(?:> )?(ok|missing file|no such anchor): \[[^\]]*\]\(([^)\s]+)\)", line)
            if m and m.group(1) != "ok":
                expect[(i + 1, m.group(2))] = m.group(1)
        nok = sum(1 for x in LINKS.split("\n") if re.match(r"^(?:> )?ok: \[", x) or "\"ok: [" in x)
        n, nprop, bad = check([b], d)
        got = {(ln, t): why for _, ln, t, why in bad}
        for k in sorted(set(expect) | set(got)):
            if expect.get(k) != got.get(k):
                print(f"FAIL link line {k[0]} {k[1]}: expected {expect.get(k) or 'ok'}, got {got.get(k) or 'ok'}")
                fails += 1
        if n != nok + len(expect):
            print(f"FAIL links read: {n}, planted {nok + len(expect)}")
            fails += 1
        print(f"selftest: {len(PLANT_GITHUB)} GitHub anchors, {n} planted links ({len(expect)} must fail, "
              f"{nprop} in quoted proposal text): {'FAIL, ' + str(fails) + ' mismatches' if fails else 'pass'}"
              + (f" [mutate={MUTATE}]" if MUTATE else ""))
        return 1 if fails else 0
    finally:
        shutil.rmtree(d)


def main():
    global MUTATE
    args = sys.argv[1:]
    for x in [x for x in args if x.startswith("--mutate=")]:
        MUTATE = x.split("=", 1)[1]
        args.remove(x)
        if MUTATE not in ("keep_no", "no_proposal", "proposal_everywhere"):
            sys.exit(f"unknown mutation {MUTATE!r}")
    if "--selftest" in args:
        sys.exit(selftest())
    files = args or ["README.md"] + sorted(glob.glob(os.path.join(ROOT, "notes", "*.md")))
    n, nprop, bad = check(files, ROOT)
    for f, ln, t, why in bad:
        print(f"{f}:{ln}: {t}: {why}")
    print(f"{n} links checked, {len(bad)} broken ({nprop} in quoted proposal text, also resolved from the root)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
