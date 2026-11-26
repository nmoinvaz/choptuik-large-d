"""Counts of the forty walls of the AI paper (Appendices A and B) by who supplied the idea that passed each, its kind, and
what the prompt of mine that moved it first carried.

    python3 paper/ai/walls.py                  # prints the tables, writes walls_timeline.csv for the figure
    python3 paper/ai/walls.py --selftest [--mutate=date|count]

walls.csv is coded by hand from the appendix rows and the sources SOURCES.md names for them. The codes:
  wall_kind     what stalled: physics (a formula or the formulation), solver (a solve, a step or a continuation
                failed, stalled or hung), arithmetic (a residual floor or a loss of digits), cost (too slow), setup (data
                lost, machines idle), checks (a claim beyond its evidence, a result against the literature), direction
                (a turning point with no wall: a new goal or a review of the course)
  moved_first_by  prompt (mine), coordinator (its own diagnosis), other (another session or an outside protocol)
  push_carried  for a prompt: goal (a target, a raised limit, persistence or urgency), doubt (asks whether the method
                or the reading is wrong), choice (picks or approves an option the coordinator posed), outside look (a
                literature search, a review, a derivation from scratch), technique or resource (names a computing tool,
                a method or a machine), physics or math (names a form, a hypothesis or a mechanism), rule (a way of
                working)
  idea_from     who supplied the idea that passed the wall: me, coordinator, subagent, subagent literature, runner, protocol
  idea_kind     physics or math, numerical (a solver, a discretization, a continuation, a test design), computing
                (precision, speed, machines, software defects), process (rules, reviews, decisions about the work)
The check compares each row's date with the "when" column of the appendix tables in main.tex, so a row out of step
with the paper fails.
"""
import argparse, csv, re, sys
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANES = ["me", "coordinator", "subagent", "other"]          # the figure's rows; literature subagents share a lane
KINDS = ["physics or math", "numerical", "computing", "process"]
MONTHS = {"Oct": 10, "Nov": 11}


def lane(idea_from):
    return "subagent" if idea_from.startswith("subagent") else idea_from if idea_from in LANES else "other"


def appendix_dates(tex):
    """{row: (month, first day)} from the rows of the appendix tables, e.g. '27 & 6 Nov &' or '18 & 27 Oct--2 Nov &'."""
    out = {}
    for m in re.finditer(r"^(\d+) & (\d+)(?:--\d+)? (Oct|Nov)", tex, re.M):
        out[int(m.group(1))] = (MONTHS[m.group(3)], int(m.group(2)))
    for m in re.finditer(r"^(\d+) & (\d+) (Oct|Nov)--", tex, re.M):
        out[int(m.group(1))] = (MONTHS[m.group(3)], int(m.group(2)))
    return out


def tables(rows):
    n = len(rows)
    t = [("walls", "all", n)]
    for col in ("wall_kind", "moved_first_by", "idea_from", "idea_kind", "held"):
        for k, v in sorted(Counter(r[col] for r in rows).items(), key=lambda kv: -kv[1]):
            t.append((col, k, v))
    pushed = [r for r in rows if r["moved_first_by"] == "prompt"]
    for k, v in sorted(Counter(r["push_carried"] for r in pushed).items(), key=lambda kv: -kv[1]):
        t.append(("push_carried", k, v))
    for a in LANES:
        for b in KINDS:
            t.append(("lane x kind", f"{a} / {b}", sum(lane(r["idea_from"]) == a and r["idea_kind"] == b for r in rows)))
    for p in sorted({r["push_carried"] for r in pushed}):
        for a in LANES:
            v = sum(r["push_carried"] == p and lane(r["idea_from"]) == a for r in pushed)
            if v:
                t.append(("push x lane", f"{p} / {a}", v))
    phys = [int(r["row"]) for r in rows if r["idea_kind"] == "physics or math"]
    gaps = [b - a - 1 for a, b in zip(phys, phys[1:])]
    t.append(("physics or math", "rows", " ".join(map(str, phys))))
    t.append(("physics or math", "longest run of walls without one", max(gaps, default=len(rows) - len(phys))))
    return t


def check(rows, tex):
    """Problems: a row missing from the appendix, or a date that differs from its 'when' column."""
    app, bad = appendix_dates(tex), []
    for r in rows:
        k, d = int(r["row"]), date.fromisoformat(r["date"])
        if k not in app:
            bad.append(f"row {k}: not in the appendix tables")
        elif app[k] != (d.month, d.day):
            bad.append(f"row {k}: {r['date']} against the appendix's {app[k][1]}/{app[k][0]}")
    if sorted(int(r["row"]) for r in rows) != sorted(app):
        bad.append(f"rows {sorted(set(app) - {int(r['row']) for r in rows})} of the appendix are not coded")
    for r in rows:
        if r["idea_kind"] not in KINDS:
            bad.append(f"row {r['row']}: unknown idea_kind {r['idea_kind']!r}")
        if (r["moved_first_by"] == "prompt") != bool(r["push_carried"]):
            bad.append(f"row {r['row']}: push_carried must be set exactly when a prompt moved it first")
    return bad


def timeline(rows):
    """day (from 18 October, centred in its day), y (the lane, walls of one day stacked about it), kind index, row."""
    key = lambda r: (lane(r["idea_from"]), (date.fromisoformat(r["date"]) - date(2026, 10, 18)).days)
    size, seen, out = Counter(key(r) for r in rows), Counter(), []
    for r in rows:
        ln, d = key(r)
        k = seen[(ln, d)]
        seen[(ln, d)] += 1
        y = 3 - LANES.index(ln) + 0.24 * (k - (size[(ln, d)] - 1) / 2)
        out.append((d + 0.5, round(y, 2), KINDS.index(r["idea_kind"]), r["row"]))
    return out


def selftest(mutate):
    tex = "\n".join(["1 & 18 Oct & a", "2 & 19 Oct & b", "3 & 27 Oct--2 Nov & c", "4 & 7 Nov & d"])
    rows = [dict(row="1", date="2026-10-18", wall_kind="solver", moved_first_by="prompt", push_carried="goal",
                 idea_from="coordinator", idea_kind="numerical", held="yes", note=""),
            dict(row="2", date="2026-10-19", wall_kind="physics", moved_first_by="prompt", push_carried="physics or math",
                 idea_from="me", idea_kind="physics or math", held="no", note=""),
            dict(row="3", date="2026-10-27", wall_kind="cost", moved_first_by="coordinator", push_carried="",
                 idea_from="coordinator", idea_kind="computing", held="yes", note=""),
            dict(row="4", date="2026-11-07", wall_kind="solver", moved_first_by="other", push_carried="",
                 idea_from="subagent literature", idea_kind="physics or math", held="yes", note="")]
    if mutate == "date":
        rows[2]["date"] = "2026-11-02"
    if mutate == "count":
        rows[3]["idea_kind"] = "numerical"
    bad = check(rows, tex)
    t = {(a, b): c for a, b, c in tables(rows)}
    ok = not bad and t[("lane x kind", "subagent / physics or math")] == 1 and t[("idea_from", "coordinator")] == 2 \
        and t[("physics or math", "longest run of walls without one")] == 1
    print(f"selftest: {len(bad)} problems, counts {'as planted' if ok or bad else 'WRONG'}: {'PASS' if ok else 'FAIL'}")
    for b in bad:
        print("  ", b)
    return 0 if ok else 1


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--selftest", action="store_true"); a.add_argument("--mutate", default="")
    a = a.parse_args()
    if a.selftest:
        return selftest(a.mutate)
    rows = list(csv.DictReader(open(HERE / "walls.csv")))
    bad = check(rows, (HERE / "main.tex").read_text())
    for b in bad:
        print("PROBLEM:", b)
    for x in tables(rows):
        print(f"{x[0]:18} {x[1]:42} {x[2]}")
    with open(HERE / "walls_timeline.csv", "w", newline="") as o:
        w = csv.writer(o, lineterminator="\n")
        w.writerow(["day", "y", "kind", "row"])
        w.writerows(timeline(rows))
    return 1 if bad else 0


sys.exit(main())
