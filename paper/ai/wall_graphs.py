"""Vector PDFs of the repository's graphs that the AI paper shows beside its walls (Appendix C), for LaTeX.

    python3 paper/ai/wall_graphs.py               # writes paper/ai/graphs/<name>.pdf for every row of wall_graphs.csv
    python3 paper/ai/wall_graphs.py --check       # each graph's file exists, its wall is a row of walls.csv, and its
                                                  # source SVG is older than the next wall (exit 1 otherwise)
    python3 paper/ai/wall_graphs.py --check --mutate=late   # a graph dated after its next wall must fail the check

wall_graphs.csv lists each graph: the wall (a row of walls.csv), the SVG in the repository, and the name the paper gives
it. Each graph is printed as it was in the commit that added it, so it shows what was seen at the wall, not the redrawn
file of today. Headless Chrome prints it to a page of its own size, so the PDF keeps the vector drawing. An animated
SVG is printed with its animations paused at the time the `at` column gives, in seconds. Dates are UTC, as in walls.csv.
graphs/index.csv records each graph's commit and day (from 18 October, as in the timeline figure).
"""
import argparse, csv, html, os, re, shutil, subprocess, sys, tempfile
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def size(text):
    head = text[:3000]
    w, h = (re.search(rf'<svg[^>]*\b{k}="([\d.]+)', head, re.S) for k in ("width", "height"))
    return float(w.group(1)), float(h.group(1))


def added(path):
    """(date, hash) of the commit that added the file."""
    out = subprocess.run(["git", "-C", str(ROOT), "log", "--diff-filter=A", "--date=format-local:%Y-%m-%d",
                          "--format=%ad %h", "--", path], capture_output=True, text=True,
                         env={**os.environ, "TZ": "UTC"}).stdout.splitlines()
    d, h = out[-1].split()
    return date.fromisoformat(d), h


def render(path, commit, out, tmp, at=""):
    text = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], capture_output=True, text=True,
                          check=True).stdout
    w, h = size(text)
    body = re.sub(r"^\s*<\?xml[^>]*\?>", "", text)
    freeze = (f"<script>for (const s of document.querySelectorAll('svg')) {{ s.pauseAnimations(); "
              f"s.setCurrentTime({float(at)}); }}</script>") if at else ""
    page = Path(tmp) / f"{out.stem}.html"
    page.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="color-scheme" content="light">
<style>@page {{ size: {w}px {h}px; margin: 0 }} html, body {{ margin: 0; background: #fff }}
* {{ -webkit-print-color-adjust: exact; print-color-adjust: exact }} body > svg {{ display: block }}</style></head>
<body>{body}{freeze}</body></html>""")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--hide-scrollbars",
                    f"--user-data-dir={tmp}/profile", "--virtual-time-budget=30000", f"--print-to-pdf={out}",
                    page.as_uri()], check=True, capture_output=True, timeout=120)


def check(rows, mutate=""):
    walls = {r["row"]: date.fromisoformat(r["date"]) for r in csv.DictReader(open(HERE / "walls.csv"))}
    bad = []
    for g in rows:
        svg = ROOT / g["file"]
        if not svg.exists():
            bad.append(f"{g['file']}: missing")
            continue
        if g["wall"] not in walls:
            bad.append(f"{g['file']}: wall {g['wall']} is not a row of walls.csv")
            continue
        made = added(g["file"])[0] if mutate != "late" else date(2026, 12, 31)
        later = sorted(d for k, d in walls.items() if int(k) > int(g["wall"]) and d > walls[g["wall"]])
        if later and made > later[0] and g.get("made", "") != "after":
            bad.append(f"{g['file']}: made {made}, after the next wall ({later[0]}) and not marked 'after'")
    return bad


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--check", action="store_true"); a.add_argument("--mutate", default="")
    a = a.parse_args()
    rows = list(csv.DictReader(open(HERE / "wall_graphs.csv")))
    if a.check:
        bad = check(rows, a.mutate)
        print(f"check: {len(rows)} graphs, {len(bad)} problems: {'PASS' if not bad else 'FAIL'}")
        for b in bad:
            print("  ", b)
        return 1 if bad else 0
    outdir = HERE / "graphs"
    outdir.mkdir(exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="wall_graphs_")
    index = []
    try:
        for g in rows:
            made, commit = added(g["file"])
            out = outdir / (Path(g["file"]).stem + ".pdf")
            render(g["file"], commit, out, tmp, g.get("at", ""))
            index.append((g["wall"], g["file"], commit, made.isoformat(), (made - date(2026, 10, 18)).days, out.name))
            print(f"{g['wall']:>3}  {g['file']:32} {commit} {made} -> {out.relative_to(ROOT)}")
        with open(outdir / "index.csv", "w", newline="") as o:
            csv.writer(o, lineterminator="\n").writerows([("wall", "file", "commit", "date", "day", "pdf")] + index)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


sys.exit(main())
