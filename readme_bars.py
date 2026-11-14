#!/usr/bin/env python3
"""List each ± in README.md whose sentence names no kind of bar.

README.md says "Each ± says what kind of bar it is", and AGENTS.md asks that every quoted ± say whether it is a spread
over fit choices, a standard error, an estimator range or an enclosure. This lint lists the ones that do not. It exits 1
when it lists any, 0 when it lists none.

    python3 readme_bars.py [FILE]          # default README.md beside this script
    python3 readme_bars.py --rev REV       # README.md at a git revision, e.g. b73b80e8
    python3 readme_bars.py --selftest      # plants a bare ±, exits 1 when it is listed and 2 when not

The rule.

- A bar is a ± (or \\pm in math) with a number after it. Either a number or a closing bracket comes before it, as in
  "1.64395 ± 0.00008", "(1.6 ± 2.2)×10⁻⁶" and "(A, ω) ± 10⁻⁹", or it stands alone with a number that has a decimal
  point or a power of ten, as in "their ±10⁻⁶" and "(±3×10⁻¹¹)". A ± before a letter or a symbol (±iπ, ±ω, \\pm\\beta)
  is a sign, and so is a lone ± before an integer or a fraction (±2, ±1/2, ν′ − ν = ±2, ±4, ±6).
- The sentence of a bar is the sentence of prose that holds it. A table row is one sentence, and so is a ```math block.
  Fenced code other than math, inline code and link targets are not read.
- The sentence names a kind when it holds one of these words, case ignored: spread, scatter, standard error, standard
  deviation, range, estimat (estimator, estimate), bound, enclos (enclosure, enclosed), interval, box, certif,
  rigorous, resolution, fit choices, bar (the error bar, its bar, a pointer to where the bar is stated).
- A bar also passes when
  - a later sentence of the same paragraph, list item or table row starts with "The bar", "Its bar", "Their bar",
    "The error bar", "The bars", "Both bars" or "Each bar", which states the kind for the bars before it;
  - its sentence cites a publication ("et al", "published", "their", "cite", a footnote [^n], a year 1900–2099, or two
    capitalized names joined by &). The bar is then the publication's own, and the citation points to its kind;
  - the same value ± bar, equal as numbers, passes elsewhere in the file by one of the rules above. That is the
    canonical statement, and a repeat need not restate it.
- Words that say how a value was made, not what its bar is, do not count: extrapolated, converges, limit, fit alone.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹"
POW = rf"10[⁻⁺]?[{SUP}]+"
NUM = rf"(?:{POW}|\d+(?:\.\d+)?(?:\s*×\s*{POW})?)"
PM = re.compile(r"±|\\pm(?![a-zA-Z])")
RIGHT = re.compile(rf"\s*({NUM})")
LEFT = re.compile(rf"(\d+(?:\.\d+)?(?:\(\d+\))?)\s*$")
KIND = re.compile(r"spread|scatter|standard error|standard deviation|range|estimat|bound|enclos|interval|box|"
                  r"certif|rigorous|resolution|fit choices|\bbars?\b", re.I)
CITE = re.compile(r"et al|published|\btheir\b|\bcite|\[\^|(?<![\d.,])(?:19|20)\d\d(?!\d)|"
                  r"\b[A-Z][\w'’-]+ & [A-Z]")
BACKREF = re.compile(r"^\W*(?:The bar|Its bar|Their bar|The error bar|The bars|Both bars|Each bar)\b")
ABBR = {"al", "Eq", "Eqs", "Fig", "Figs", "Ref", "Refs", "Sec", "e.g", "i.e", "cf", "vs", "No", "Vol", "Phys", "Rev",
        "Lett", "Nucl", "Class", "Quant", "Grav", "Math", "Commun", "Proc", "Ann", "approx", "resp", "App", "Thm",
        "Prop", "pp", "eds", "Jr", "St", "Dr"}
SPLIT = re.compile(r"[.!?][)*\"”’]*\s+(?=[A-ZΑ-Ω*\[(`$\"“α-ω])")
PLANT = "7.654321 ± 0.000123"


def num(s):
    """A README number as a float: 1.5×10⁻⁸, 10⁻⁶, 0.000000015."""
    s = s.translate(str.maketrans(SUP + "⁻⁺", "0123456789-+")).replace(" ", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)?(?:×?10([-+]?\d+))?", s)
    if not m or not (m.group(1) or m.group(2)):
        return None
    return float(m.group(1) or 1) * (10.0 ** int(m.group(2)) if m.group(2) else 1.0)


def blocks(text):
    """(first line, [lines], kind) for each paragraph, list item, table row, heading or math block."""
    out, cur, fence = [], None, None
    for i, line in enumerate(text.split("\n"), 1):
        s = line.strip()
        if fence is not None:                                    # inside a fence
            if s.startswith("```"):
                if fence[0] == "math":
                    out.append((fence[1], fence[2], "math"))
                fence = None
            elif fence[0] == "math":
                fence[2].append(line)
            continue
        if s.startswith("```"):
            cur = None
            fence = ("math" if s[3:].strip() == "math" else "code", i, [])
            continue
        if not s:
            cur = None
            continue
        item = re.match(r"(?:[-*+]|\d+\.)\s", s) is not None
        if s.startswith("|") or s.startswith("#") or s.startswith("<") or item or cur is None:
            if re.fullmatch(r"\|[\s:|-]*\|?", s):                # table separator row
                cur = None
                continue
            cur = (i, [line], "row" if s.startswith("|") else "prose")
            out.append(cur)
            if s.startswith(("|", "#", "<")):
                cur = None                                        # a row, heading or tag ends at its line
        else:
            cur[1].append(line)
    return out


def clean(t):
    """Text for the word checks: no inline code, no link targets, math kept."""
    t = re.sub(r"(?<!\$)`[^`]*`(?!\$)", " ", t)
    return re.sub(r"\]\([^)]*\)", "]", t)


def sentences(joined, kind):
    if kind != "prose":
        return [(0, len(joined))]
    spans, start = [], 0
    for m in SPLIT.finditer(joined):
        word = re.search(r"(\S+)$", joined[start:m.start()])
        w = word.group(1).lstrip("([\"“*_") if word else ""
        if w in ABBR or re.fullmatch(r"[A-Z]", w):                # et al., Eq., an initial
            continue
        spans.append((start, m.start() + 1))
        start = m.end()
    spans.append((start, len(joined)))
    return spans


def bars(text):
    """Every bar with its line, text, sentence and the rule that passes it (None if none does)."""
    found = []
    for first, lines, kind in blocks(text):
        offs, parts, o = [], [], 0
        for k, line in enumerate(lines):                         # joined text, with each line's start offset
            offs.append((o, first + k + (1 if kind == "math" else 0)))
            parts.append(line.strip())
            o += len(line.strip()) + 1
        joined = " ".join(parts)
        code = [(m.start(), m.end()) for m in re.finditer(r"(?<!\$)`[^`]*`(?!\$)", joined)]
        sents = sentences(joined, kind)
        for m in PM.finditer(joined):
            if any(a <= m.start() < b for a, b in code):
                continue
            r = RIGHT.match(joined, m.end())
            if not r:
                continue                                          # a sign, as in ±iπ or \pm\beta
            left = joined[:m.start()].rstrip()
            lv = LEFT.search(left)
            b = r.group(1)
            if not lv and not left.endswith(")") and "." not in b and not re.search(POW, b):
                continue                                          # a sign, as in ±2 or ±1/2
            si = next(j for j, (a, e) in enumerate(sents) if a <= m.start() <= e)
            a, e = sents[si]
            sent = clean(joined[a:e])
            line = max(ln for oo, ln in offs if oo <= m.start())
            rule = ("kind" if KIND.search(sent) else
                    "later 'The bar'" if any(BACKREF.match(clean(joined[x:y])) for x, y in sents[si + 1:]) else
                    "citation" if CITE.search(sent) else None)
            pair = (num(lv.group(1)) if lv else None, num(b))
            found.append(dict(line=line, bar=joined[(lv.start() if lv else m.start()):r.end()].strip(),
                              pair=pair, sentence=joined[a:e].strip(), rule=rule))
    ok = [f["pair"] for f in found if f["rule"] and f["pair"][0] is not None]
    for f in found:
        v, b = f["pair"]
        if f["rule"] is None and v is not None and b is not None and any(
                p[0] is not None and p[1] is not None and abs(p[0] - v) <= 1e-12 * abs(v) and abs(p[1] - b) <= 1e-9 * b
                for p in ok):
            f["rule"] = "repeat"
    return found


def report(text, name):
    found = bars(text)
    bad = [f for f in found if f["rule"] is None]
    for f in bad:
        s = f["sentence"] if len(f["sentence"]) <= 160 else f["sentence"][:157] + "..."
        print(f"{name}:{f['line']}: {f['bar']}  |  {s}")
    by = {}
    for f in found:
        by[f["rule"]] = by.get(f["rule"], 0) + 1
    print(f"{len(found)} bars: " + ", ".join(f"{n} by {r}" for r, n in sorted(by.items(), key=lambda t: str(t[0]))
                                             if r) + f"; {len(bad)} name no kind")
    return bad


def selftest(text):
    """Copy the first sentence that passes by a kind word alone, drop its kind words, give its bar new numbers, and
    append the copy as its own paragraph. The lint must list it."""
    src = next(f for f in bars(text) if f["rule"] == "kind" and not CITE.search(clean(f["sentence"]))
               and f["pair"][0] is not None)
    assert PLANT not in text
    copy = KIND.sub("", src["sentence"].replace(src["bar"], PLANT, 1))
    print(f"selftest: source sentence (passes): {src['sentence'][:150]}")
    print(f"selftest: planted copy: {copy[:150]}")
    planted = text.rstrip("\n") + "\n\n" + copy + "\n"
    bad = report(planted, "planted")
    hit = any(f["bar"] == PLANT for f in bad)
    print("selftest: the planted bar is listed" if hit else "selftest: the planted bar is NOT listed (control failed)")
    return 1 if hit else 2


def main(argv):
    if "--selftest" in argv:
        return selftest((ROOT / "README.md").read_text(encoding="utf-8"))
    if "--rev" in argv:
        rev = argv[argv.index("--rev") + 1]
        text = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:README.md"], check=True,
                              capture_output=True, text=True).stdout
        name = f"{rev}:README.md"
    else:
        path = Path(argv[1]) if len(argv) > 1 else ROOT / "README.md"
        text, name = path.read_text(encoding="utf-8"), path.name
    return 1 if report(text, name) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
