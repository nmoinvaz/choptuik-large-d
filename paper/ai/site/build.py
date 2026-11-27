#!/usr/bin/env python3
"""Data of the web page of the AI paper (paper/ai/site/index.html), from the paper's own files.

    python3 paper/ai/site/build.py          # rewrites the JSON block of index.html between its two markers
    python3 paper/ai/site/build.py --check  # exit 1 if index.html's block differs from what the files give

Reads paper/ai/walls.csv, walls_timeline.csv, prompt_times.csv, prompt_stats.csv, frontier.csv, graphs/index.csv,
wall_graphs.csv, the appendix tables of main.tex (what stuck, moved by, resolution, what changed), the list of 145
course-changing prompts in SOURCES.md and hp/data/delta_all.csv. The graphs beside the page are the SVGs as they were in the commits graphs/index.csv names
(git show <commit>:<file>).
"""
import csv, html, json, math, re, sys
from datetime import date
from decimal import Decimal, getcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
AI = HERE.parent
ROOT = AI.parent.parent
getcontext().prec = 60

# The large-D chain's early points, from the frontier figure of main.tex (days from 19 October 2026).
EARLY = [(0.30, 40.0), (1.32, 60.96), (6.20, 67.77), (8.04, 70.98), (8.24, 73.27), (8.73, 78.10), (16.06, 79.27),
         (16.93, 86.53), (17.14, 96.12), (17.33, 127.12), (18.00, 146.56), (18.12, 152.93), (18.21, 162.31),
         (18.34, 175.27), (18.42, 188.02), (18.56, 193.77), (18.67, 200.27), (18.78, 206.77)]
# The direct solves of #1088 on 2026-11-25 (UTC), by the issue comments SOURCES.md names (Section VIII).
DIRECT = [("04:04", 300, "300"), ("04:22", 250, "250"), ("04:22", 400, "400"), ("08:05", 1000, "1000"),
          ("10:54", 2000, "2000"), ("18:50", 2828, "2828"), ("23:53", 10000, "10⁴")]
DAY_25NOV = 38                                  # 2026-11-25 is day 38 from 18 October

SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def math_text(m):
    s = m.group(1)
    s = re.sub(r"\\times\s*10\^\{([^}]*)\}", lambda k: "×10" + k.group(1).translate(SUP), s)
    s = re.sub(r"10\^\{([^}]*)\}", lambda k: "10" + k.group(1).translate(SUP), s)
    s = re.sub(r"10\^(\d)", lambda k: "10" + k.group(1).translate(SUP), s)
    for a, b in [(r"\Delta", "Δ"), (r"\ell", "ℓ"), (r"\le", "≤"), (r"\approx", "≈"), (r"\infty", "∞"), (r"\to", "→"),
                 (r"\kappa", "κ")]:
        s = s.replace(a, b)
    return f"<i class=\"m\">{s}</i>" if re.fullmatch(r"[A-Za-zΔℓκ]", s) else s


def tex_html(s):
    s = s.strip()
    s = re.sub(r"~?\\cite\{Cochelin1994\}", " (Cochelin 1994)", s)
    s = re.sub(r"~?\\cite\{[^}]*\}", "", s)
    s = s.replace(r"M\"obius", "Möbius").replace(r"\dots\ ", "… ").replace(r"\dots", "…").replace(r"\ ", " ")
    s = s.replace("--", "–").replace(r"\%", "%").replace(r"\_", "_").replace("&", "&amp;")
    s = re.sub(r"\$([^$]*)\$", math_text, s)
    s = re.sub(r"\\file\{([^}]*)\}", r"<code>\1</code>", s)
    s = re.sub(r"\\prompt\{([^}]*)\}", r'<q class="p">\1</q>', s)
    for a, b in [("my requests", "Nathan's requests"), ("I suggested", "Nathan suggested"),
                 ("which I had chosen", "which Nathan had chosen")]:                 # the page speaks of him
        s = s.replace(a, b)
    return s


def note_html(s):
    """A walls.csv note: plain text whose double quotes hold his prompts."""
    s = s.replace("...", "…").replace("Moebius", "Möbius").replace("10^4", "10⁴")
    s = re.sub(r'"([^"]+)"', r'<q class="p">\1</q>', s)
    return tex_html(s)


def appendix(tex):
    """{row: {...}} from the two appendix tables: 6 columns in A, 5 in B."""
    out = {}
    for line in tex.splitlines():
        m = re.match(r"^(\d+) & (.*)\\\\$", line)
        if not m:
            continue
        cells = [c for c in re.split(r"(?<!\\) & ", m.group(2))]
        n = int(m.group(1))
        if len(cells) == 5:
            when, stuck, moved, res, changed = cells
            out[n] = dict(when=tex_html(when), stuck=tex_html(stuck), moved=tex_html(moved), resolution=tex_html(res),
                          changed=tex_html(changed))
        elif len(cells) == 4:
            when, prompt, followed, changed = cells
            out[n] = dict(when=tex_html(when), stuck="", moved=tex_html(prompt), resolution=tex_html(followed),
                          changed=tex_html(changed))
    return out


PUSH_WHERE = {"chat": "the chat", "43f19197": "coordinator", "cc3aa2b5": "Mac runner", "3bc27918": "Mac runner",
              "ebec4b64": "libraries"}


def pushes(src):
    """The 145 course-changing prompts of SOURCES.md: [day from 18 Oct (UTC), "HH:MM", where, kind letter, text]."""
    part = src.split("## The 145 prompts counted in the table of kinds", 1)[1].split("\n## ", 1)[0]
    out = []
    for line in part.splitlines():
        m = re.match(r"^\| (\d\d)-(\d\d) (\d\d):(\d\d) \| ([0-9a-z]+) \| ([PRWANMXSCU]) \| (.*) \|$", line)
        if not m:
            continue
        mo, d, hh, mm = (int(m.group(i)) for i in range(1, 5))
        day = (date(2026, mo, d) - date(2026, 10, 18)).days + (hh * 60 + mm) / 1440
        out.append([round(day, 4), f"{hh:02d}:{mm:02d}", PUSH_WHERE[m.group(5)], m.group(6),
                    html.escape(m.group(7), quote=False)])
    assert len(out) == 145, len(out)
    return out


def delta_str(r):
    """Δ as a decimal string to the digits the row's stated error supports, from its quad-double parts."""
    v = sum(Decimal(r[k]) for k in ("Delta_q0", "Delta_q1", "Delta_q2", "Delta_q3") if r[k])
    err = float(r["err"]) if r["err"] else 1e-10
    dec = max(4, min(30, int(math.floor(-math.log10(err)))))
    return f"{v:.{dec}f}", r["err"]


def delta_samples():
    rows = [r for r in csv.DictReader(open(ROOT / "hp/data/delta_all.csv")) if r["kind"] != "series"]
    rows = [r for r in rows if float(r["D"]) <= 8000]  # the paper's direct rows end at 8000; 10^4 is on its issue
    rows.sort(key=lambda r: float(r["D"]))
    lo, hi, n = math.log(float(rows[0]["D"])), math.log(8000.0), 90
    pick = []
    for i in range(n + 1):
        t = math.exp(lo + (hi - lo) * i / n)
        r = min(rows, key=lambda r: abs(math.log(float(r["D"])) - math.log(t)))
        if not pick or pick[-1] is not r:
            pick.append(r)
    for d in (4.0, 6.0, 7.0, 8.0):  # the trial's dimensions, exactly
        r = next(r for r in rows if float(r["D"]) == d)
        if r not in pick:
            pick.append(r)
    pick.sort(key=lambda r: float(r["D"]))
    out = []
    for r in pick:
        s, e = delta_str(r)
        out.append([float(r["D"]), float(r["Delta_q0"]), s, e, r["kind"]])
    return out


def main():
    tex = (AI / "main.tex").read_text()
    app = appendix(tex)
    tl = {int(r["row"]): r for r in csv.DictReader(open(AI / "walls_timeline.csv"))}
    names = {(r["wall"], r["file"]): r for r in csv.DictReader(open(AI / "wall_graphs.csv"))}
    graphs = []
    for r in csv.DictReader(open(AI / "graphs/index.csv")):
        g = names[(r["wall"], r["file"])]
        graphs.append(dict(wall=int(r["wall"]), file=r["file"], src="graphs/" + Path(r["file"]).name,
                           commit=r["commit"], date=r["date"], day=int(r["day"]), name=g["name"],
                           later=g["made"] == "after"))
    walls = []
    for r in csv.DictReader(open(AI / "walls.csv")):
        n = int(r["row"])
        t = tl[n]
        walls.append(dict(n=n, date=r["date"], day=float(t["day"]), y=float(t["y"]), kind=int(t["kind"]),
                          wall_kind=r["wall_kind"], moved_first_by=r["moved_first_by"], push=r["push_carried"],
                          idea_from=r["idea_from"], idea_kind=r["idea_kind"], held=r["held"], note=note_html(r["note"]),
                          graphs=[i for i, g in enumerate(graphs) if g["wall"] == n], **app[n]))
    assert len(walls) == 40 and all(w["when"] for w in walls)
    P = list(csv.DictReader(open(AI / "prompt_times.csv")))
    prompts = dict(day=[round(float(p["day"]), 4) for p in P], row=[int(p["row"]) for p in P],
                   form=[int(p["form"]) for p in P])
    stats = [[r["table"], r["row"], r["value"]] for r in csv.DictReader(open(AI / "prompt_stats.csv"))]
    front = [[round(d + 1, 3), D] for d, D in EARLY]
    front += [[round(float(r["day"]) + 1, 3), float(r["D"])] for r in csv.DictReader(open(AI / "frontier.csv"))][1:]
    direct = [[round(DAY_25NOV + (int(t[:2]) + int(t[3:]) / 60) / 24, 4), D, lab] for t, D, lab in DIRECT]
    data = dict(walls=walls, graphs=graphs, prompts=prompts, stats=stats, frontier=front, direct=direct,
                delta=delta_samples(), pushes=pushes((AI / "SOURCES.md").read_text()))
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = (HERE / "index.html").read_text()
    a, b = '<script id="data" type="application/json">', "</script>"
    i = page.index(a) + len(a)
    j = page.index(b, i)
    new = page[:i] + blob + page[j:]
    if "--check" in sys.argv:
        ok = new == page
        print("PASS" if ok else "FAIL: index.html's data block differs from the files")
        sys.exit(0 if ok else 1)
    (HERE / "index.html").write_text(new)
    print(f"wrote {len(blob)} bytes of data: {len(walls)} walls, {len(graphs)} graphs, {len(P)} prompts, "
          f"{len(front)} frontier points, {len(data['delta'])} values of Δ, {len(data['pushes'])} course-changing prompts")


if __name__ == "__main__":
    main()
