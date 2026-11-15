#!/usr/bin/env python3
"""A lint for stale status claims in proof/ (#757): each proof note's status and review phrases about a named result,
against the status tables. notes/CONSISTENCY_AUDIT.md, Part 2 (findings P2-5 to P2-8 and P2-14), found such claims.

    python3 proof_status_lint.py [--rev REV] [--verbose]
    python3 proof_status_lint.py --selftest [--mutate=word]

The reference is the README's two status tables ("Main results", and "What is proved, and how" with its Reviews and
Where columns) and PROOF.md's "| Layer B item | Status |" table, which the README points to for results it does not list
itself. (H8) takes the status of PROOF.md's row for the global continuation B8, which states it. A result's status words
are AGENTS.md's (proved, checked symbolically, certified, conditional, conjecture, numerical evidence) and open, as its
Status cell has them; a Reviews cell records a review when it names a review note, an issue, a re-implementation or
CAPD.

A line of a proof note is listed when it makes one of three claims that the reference contradicts:
  open    "**open**", "is open", "remains open", "future work", "not yet certified", "(to be written", "estimated in",
          "is estimated" or "waits for its certificate", about a result named on the line that the reference calls
          certified or proved and not open (a line that names its own resolution, "proved in §", is not);
  word    "<result> is proved", for a result the reference calls certified and not proved;
  review  "not reviewed", "unreviewed", "not been reviewed", "no independent review", about a result with a recorded
          review. It is judged by its sentence: one that also records a review ("reviewed in", "audited", a review note)
          is a qualified statement and is not listed. The result is the one the sentence names; else, for a claim about
          the whole note ("nothing here", or a byline in its first 15 lines), any of the note's own results (the files
          of the Where column); else the subject of the paragraph or bullet, its first named result; else the note's own
          results, all of them reviewed. Where a Reviews cell is split by M, a line naming one M, or a note about one M
          (FILE_M), is judged by that part.
Review notes (*_review.md) record what was reviewed at the time and are not linted. Lines in code blocks, and lines that
say "superseded" or "since resolved", are skipped. Exit 1 when a line is listed.

--rev REV reads the README, PROOF.md and proof/ at a commit (git show). --selftest plants one stale line of each kind
into a copy of the current files ("Theorem A(M = 5) has not been reviewed independently." in proof/A_M2.md, "Theorem
B(1) is open, future work." in proof/B_chain.md, "With these, lemma 6b₁ is proved." in proof/A_lemmas_M.md): the lint
must list all three, and then exits 1 as on any listed line (2 when a plant is not listed). --selftest --mutate=word
plants a consistent line ("lemma 6b₁ is certified") in place of the status-word plant, which must go unlisted: that
run exits 2, the selftest's own control.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
WORDS = ("proved", "checked symbolically", "certified", "conditional", "conjecture", "numerical evidence", "open")


# ------------------------------------------------------------------ the files, from the tree or a commit
def reader(rev):
    if rev:
        def read(path):
            return subprocess.run(["git", "-C", ROOT, "show", f"{rev}:{path}"], capture_output=True, text=True,
                                  check=True).stdout

        def notes():
            out = subprocess.run(["git", "-C", ROOT, "ls-tree", "--name-only", rev, "proof/"], capture_output=True,
                                 text=True, check=True).stdout.split()
            return sorted(p for p in out if p.endswith(".md"))
    else:
        def read(path):
            return open(os.path.join(ROOT, path), encoding="utf-8").read()

        def notes():
            return sorted("proof/" + f for f in os.listdir(os.path.join(ROOT, "proof")) if f.endswith(".md"))
    return read, notes


# ------------------------------------------------------------------ result names
def key(name):
    """a result's name reduced to the key the notes are searched for"""
    n = name.strip().strip("*").strip()
    lead = re.match(r"(Theorems?|Lemmas?|Propositions?|Conjecture)\s+", n)
    if lead:                                      # "Lemma 6a" -> 6a, "Proposition 1" -> Proposition1
        n = n[lead.end():] if lead.group(1).startswith(("Lemma", "Theorem")) else lead.group(1) + " " + n[lead.end():]
    n = n.replace("₀", "0").replace("₁", "1").replace(" ", "")
    return n


def patterns(k):
    """regexes that find the result k in a line"""
    if k.startswith("A(M="):
        m = k[4:-1]
        return [rf"A\(M\s*=\s*{m}\)"]
    if k.startswith("B(") or k.startswith("B5(") or k.startswith("B8"):
        return [re.escape(k).replace(r"\(", r"\s*\(")]
    if k == "(H8)":
        return [r"\(H8\)"]
    if re.fullmatch(r"6b[01]|6a", k):
        if len(k) == 2:
            return [rf"\b{k}\b"]
        sub = {"0": "₀", "1": "₁"}[k[2]]
        return [rf"\b{k[:2]}[{k[2]}{sub}](?![0-9₀-₉])"]
    if k.startswith("Proposition") or k.startswith("Conjecture"):
        return [rf"\b{k[:-1]}\s*{k[-1]}\b"]
    if k in ("S", "L", "P", "J", "LC"):
        return [rf"\b[Ll]emma {k}\b"]
    return [rf"\b{re.escape(k)}\b"]


def names_of(cell):
    """the result names of a Result cell: 'Theorems A(M = 1), A(M = 3)' -> A(M=1), A(M=3)"""
    cell = re.sub(r"\*\*", "", cell)
    lead = re.match(r"(Theorems?|Lemmas?|Propositions?)\s+", cell)
    word = lead.group(1) if lead else ""
    parts = [p for p in re.split(r"[,;]\s*", cell[lead.end():] if lead else cell) if p.strip()]
    out = []
    for p in parts:
        p = p.strip()
        if word.startswith("Proposition"):
            p = "Proposition " + p
        out.append(key(p))
    return out


# ------------------------------------------------------------------ the reference
def table_rows(text, header_re):
    """the rows (cells) of the markdown tables whose header matches header_re"""
    rows, on = [], False
    for ln in text.splitlines():
        if ln.startswith("|") and re.search(header_re, ln):
            on, hdr = True, [c.strip() for c in ln.strip().strip("|").split("|")]
            continue
        if on and ln.startswith("|---"):
            continue
        if on and ln.startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows.append(dict(zip(hdr, cells)))
        elif on:
            on = False
    return rows


def reference(read):
    """{key: dict(words, reviews, where, source)}"""
    ref = {}
    readme = read("README.md")
    tables = (table_rows(readme, r"\|\s*Result\s*\|\s*What it says\s*\|\s*Status")
              + table_rows(readme, r"\|\s*Result\s*\|\s*Value\s*\|\s*Status"))
    for r in tables:
        where = re.findall(r"\((proof/[A-Za-z0-9_]+\.md)\)", r.get("Where", ""))
        for k in names_of(r["Result"]):
            e = ref.setdefault(k, dict(words=set(), reviews="", where=set(), source="README.md"))
            e["words"] |= {w for w in WORDS if w in r["Status"].lower()}
            e["reviews"] = e["reviews"] or r.get("Reviews", "")
            e["where"] |= set(where)
    proof = read("PROOF.md")
    for r in table_rows(proof, r"\|\s*Layer B item\s*\|\s*Status"):
        item, st = r.get("Layer B item", ""), r.get("Status", "")
        ks = [key(x) for x in re.findall(r"\(([^)]*)\)", item) if re.match(r"B\d", x)] or names_of(item)
        if "global continuation" in item:
            ks.append("(H8)")                    # PROOF.md states (H8)'s status in the B8 row
        for k in ks:
            k = k.replace("part1", "(1)").replace("B5part1", "B5(1)")
            e = ref.setdefault(k, dict(words=set(), reviews="", where=set(), source="PROOF.md"))
            e["words"] |= {w for w in WORDS if w in st.lower()}
            e["where"] |= set(re.findall(r"\((proof/[A-Za-z0-9_]+\.md)\)", st))
    return ref


POSITIVE_REVIEW = re.compile(r"_review\.md|#\d+|re-implement|CAPD|reviewed by|review notes?")
NEGATIVE_REVIEW = re.compile(r"not yet (reviewed|done)")


def reviewed(e, M=None):
    """whether the Reviews cell records a review (for M = k, the cell's part for that M where it is split by M)"""
    cell = e["reviews"]
    if not cell or cell in ("—", "-"):
        return False
    parts = [p.strip() for p in cell.split(";")]
    if M is not None:
        mine = [p for p in parts if re.match(rf"M\s*=\s*{M}\b", p)]
        if mine:
            parts = mine
    return any(POSITIVE_REVIEW.search(p) and not NEGATIVE_REVIEW.search(p) for p in parts)


# ------------------------------------------------------------------ the lint
R_REVIEW = re.compile(r"\b(not (yet )?(been )?reviewed|unreviewed|no independent review|has not been reviewed|"
                      r"been reviewed independently yet)", re.I)
R_OPEN = re.compile(r"\*\*open\*\*|\bis open\b|\bremains open\b|future work|not yet certified|\(to be written|"
                    r"\bestimated in\b|\bis estimated\b|waits for its certificate", re.I)
SKIP = re.compile(r"superseded|since resolved", re.I)
RESOLVED = re.compile(r"(proved|certified) in §")                 # a line that names its own resolution
PARTIAL = re.compile(r"_review\.md|re-implement|\baudited\b|\breviewed (in|with|by)\b", re.I)
WHOLE = re.compile(r"\bnothing here\b|\bnone of (this|it)\b", re.I)
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[*])")
# the number of harmonics each note is about, where a Reviews cell is split by M
FILE_M = {"proof/A_certify.md": 1, "proof/A_M.md": 3, "proof/A_M2.md": 5}


def sentence(block, start):
    """the sentence of block that holds position start"""
    cuts = [0] + [m.end() for m in SENT.finditer(block)] + [len(block)]
    for a, b in zip(cuts, cuts[1:]):
        if a <= start < b:
            return block[a:b]
    return block


def lint(read, notes, verbose=False):
    ref = reference(read)
    pats = {k: [re.compile(p) for p in patterns(k)] for k in ref}
    own = {}
    for k, e in ref.items():
        for f in e["where"]:
            own.setdefault(f, set()).add(k)

    def names(text):
        hits = []
        for k in ref:
            for p in pats[k]:
                m = p.search(text)
                if m:
                    hits.append((m.start(), k))
                    break
        return [k for _, k in sorted(hits)]
    found = []
    for f in notes():
        if f.endswith("_review.md"):
            continue                              # a review note records what was reviewed then
        lines = read(f).splitlines()
        fence, para = False, []
        for i, ln in enumerate(lines, 1):
            if ln.strip().startswith("```"):
                fence = not fence
                continue
            if fence:
                continue
            if not ln.strip() or re.match(r"\s*([-*]|\d+\.)\s", ln) or ln.startswith("#") or ln.startswith("|"):
                para = []                         # a paragraph, bullet or table row starts here
            para.append(ln)
            if SKIP.search(ln):
                continue
            here = names(ln)
            fm = FILE_M.get(f)
            Ms = sorted(set(re.findall(r"M\s*=\s*(\d+)", ln)))
            M = int(Ms[0]) if len(Ms) == 1 else fm
            # open: a certified or proved result called open or future work on its own line
            if R_OPEN.search(ln) and not RESOLVED.search(ln):
                for k in here:
                    e = ref[k]
                    if e["words"] & {"certified", "proved"} and "open" not in e["words"]:
                        why = f"{k}: {', '.join(sorted(e['words']))} ({e['source']})"
                        found.append((f, i, "open", why, ln.strip()))
                        break
            # word: "<result> is proved" for a result that is certified, not proved
            for k in here:
                e = ref[k]
                if "certified" in e["words"] and "proved" not in e["words"] and \
                        any(re.search(p.pattern + r"\s*(\([^)]*\)\s*)?(is|are) proved\b", ln) for p in pats[k]):
                    found.append((f, i, "word", f"{k}: certified, not proved ({e['source']})", ln.strip()))
                    break
            # review: a "not reviewed" sentence about a result with a recorded review
            m = R_REVIEW.search(ln)
            if m:
                block = " ".join(x.strip() for x in para)
                pos = len(block) - len(ln.strip()) + m.start()
                sent = sentence(block, pos)
                if PARTIAL.search(sent):
                    continue                      # a qualified statement: something here was reviewed, the rest not
                named = names(sent)
                if named:
                    hit = next((k for k in named if reviewed(ref[k], M)), None)
                elif WHOLE.search(sent) or (i <= 15 and own.get(f)):
                    hit = next((k for k in sorted(own.get(f, ())) if reviewed(ref[k], M)), None)
                else:
                    first = names(block)
                    if first:
                        hit = first[0] if reviewed(ref[first[0]], M) else None
                    else:
                        mine = sorted(own.get(f, ()))
                        hit = mine[0] if mine and all(reviewed(ref[k], M) for k in mine) else None
                if hit:
                    found.append((f, i, "review", f"{hit}: reviewed ({ref[hit]['reviews'][:60]})", ln.strip()))
    for f, i, rule, why, ln in found:
        print(f"{f}:{i}: {rule}: {why}: {ln[:120]}")
    print(f"proof_status_lint: {len(found)} line{'s' * (len(found) != 1)} listed, {len(ref)} results in the reference")
    if verbose:
        for k, e in sorted(ref.items()):
            print(f"  {k}: {sorted(e['words'])}, reviewed {reviewed(e)}, where {sorted(e['where'])} ({e['source']})")
    return found


PLANTS = [("proof/A_M2.md", "Theorem A(M = 5) has not been reviewed independently."),
          ("proof/B_chain.md", "Theorem B(1) is open, future work."),
          ("proof/A_lemmas_M.md", "With these, lemma 6b₁ is proved.")]


def selftest(mutate=""):
    """the plants must be listed; --mutate=word plants a consistent line ("lemma 6b₁ is certified") in place of the
    status-word plant, which must then go unlisted and fail the selftest (exit 2), the selftest's own control"""
    read0, notes0 = reader(None)
    plants = [(f, "With these, lemma 6b₁ is certified.") if mutate == "word" and "is proved" in line else (f, line)
              for f, line in PLANTS]
    planted = {f: read0(f) + "\n" + line + "\n" for f, line in plants}

    def read(path):
        return planted.get(path) or read0(path)
    out = lint(read, notes0)
    hits = {(f, ln) for f, _, _, _, ln in out}
    miss = [(f, line) for f, line in plants if (f, line) not in hits]
    if miss:
        print(f"selftest: FAILED, the plants {miss} are not listed")
        return 2
    print(f"selftest: all {len(PLANTS)} planted lines are listed (exit 1, as on any listed line)")
    return 1


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--selftest" in args:
        sys.exit(selftest("word" if "--mutate=word" in args else ""))
    rev = args[args.index("--rev") + 1] if "--rev" in args else None
    read, notes = reader(rev)
    sys.exit(1 if lint(read, notes, verbose="--verbose" in args) else 0)
