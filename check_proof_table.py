#!/usr/bin/env python3
"""Check that the README's table "What is proved, and how" has a row for every named result.

    python3 check_proof_table.py [--mutate=drop_row]

The names are those of the form Proposition N, Theorem A(M = k), Theorem B(k), Theorem B(M), Lemma X (any case),
Conjecture N and Corollary C, in README.md (outside the table) and PROOF.md. A row covers a name when its first column
names it: "Lemmas S, 6b₀" covers lemma S and lemma 6b₀, and a lemma's harmonic count, as in 6a(5), is dropped.
Exits 1 when a name has no row. --mutate=drop_row removes the table's first row before the comparison (the control:
the check must then fail).
"""
import re
import sys

HEAD = "### What is proved, and how"
NAME = re.compile(r"\b(Proposition [1-9]|Theorem A\(M ?= ?\d\)|Theorem B\((?:\d|M)\)|[Ll]emmas? [A-Z0-9][A-Za-z0-9₀-₉]*(?:\.\d)?"
                  r"|Conjecture \d|Corollary C)")
NOT_NAMES = {"is", "of", "to", "by", "and", "must", "gives", "B"}
ALIASES = {"lemma 6b": {"lemma 6b₀", "lemma 6b₁"}}  # a name covered when all of its parts have rows


def norm(name):
    """one canonical spelling: Theorem A(M = 5), lemma 6b₁"""
    name = re.sub(r"\(M ?= ?(\d)\)", r"(M = \1)", name)
    if name.lower().startswith("lemma"):
        tail = name.split(None, 1)[1]
        tail = re.sub(r"\(\d\)$", "", tail)
        return None if tail in NOT_NAMES else "lemma " + tail
    return name


def table(text):
    """the first column of the table under HEAD, and the text without the table"""
    i = text.index(HEAD)
    j = text.find("\n#", i + len(HEAD))
    block = text[i:j if j > 0 else len(text)]
    rows = [ln for ln in block.splitlines() if ln.startswith("| ") and not ln.startswith("| Result") and not ln.startswith("|---")]
    return [r.split("|")[1].strip() for r in rows], text[:i] + text[i + len(block):]


def covered(first):
    """the names one row's first column covers"""
    out = set()
    for part in re.split(r";", first):
        part = part.strip()
        m = re.match(r"(Lemmas?|Theorems?) (.*)", part)
        if m and m.group(1).startswith("Lemma"):
            for item in m.group(2).split(","):
                n = norm("lemma " + item.strip())
                if n:
                    out.add(n)
            continue
        for item in re.split(r",\s*", part):
            n = norm(item.strip())
            if n:
                out.add(n)
    return out


def main():
    mutate = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
    readme = open("README.md").read()
    firsts, rest = table(readme)
    if mutate == "drop_row":
        firsts = firsts[1:]
    names = {norm(m.group(1)) for src in (rest, open("PROOF.md").read()) for m in NAME.finditer(src)} - {None}
    have = set().union(*(covered(f) for f in firsts))
    have |= {a for a, parts in ALIASES.items() if parts <= have}
    missing = sorted(names - have)
    print(f"{len(firsts)} rows, {len(names)} named results in README.md and PROOF.md")
    for n in missing:
        print(f"  no row: {n}")
    print("check:", "pass" if not missing else "FAIL")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
