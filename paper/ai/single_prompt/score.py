#!/usr/bin/env python3
"""Score one trial of trial.sh: its table against the reference values, what it used, and what it reached outside.

    python3 score.py TRIAL_DIR REFERENCE_CSV --want 4,6,7,8 [--tol 1e-6]
    python3 score.py TRIAL_DIR REFERENCE_CSV --want 6:200          # every whole D from 6 to 200
    python3 score.py --selftest [--mutate=value|missing|net|copied]

REFERENCE_CSV is a file of a choptuik-large-d checkout, of one of two kinds (lines starting with # are comments):
- hp/data/published_delta.csv (columns D,Delta,uncertainty,ref): one reference, the published values;
- hp/data/delta_published_compare.csv (columns D,published_Delta,ours_Delta,...): two references for each D, the
  published value and our own. Our value decides the verdict, and the published one is shown beside it.
The trial's table is TRIAL_DIR/work/results.csv (columns D,Delta and any others). A value is OK when it lies within
--tol of the deciding reference at the same D, BAD otherwise, MISSING when the table has no row for a wanted D. The
transcript (stream-json) gives the turns, the tokens and every Bash command; a command that reaches the network is
listed. Exit 0 only when every wanted D is OK and no command used the network.

The report lists every row that is not OK, every tenth wanted row, the counts, and the reach: the highest wanted D up
to which every wanted row is OK. --selftest scores planted trials with known answers. Each --mutate breaks one in
one way and must exit 1.
"""
import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

NET = re.compile(r"\b(curl|wget|git\s+clone|pip3?\s+(install|download)|ssh|scp|rsync|nc|telnet)\b|https?://|urllib|requests\.|socket\.|aiohttp|httpx")


def read(text):
    return list(csv.DictReader(io.StringIO("".join(ln for ln in text.splitlines(True) if not ln.startswith("#")))))


def trial_table(text):
    rows = {}
    for r in read(text):
        try:
            rows[float(r["D"])] = float(r["Delta"])
        except (KeyError, TypeError, ValueError):
            pass
    return rows


def references(text):
    """{D: [(name, value), ...]}, the deciding reference first."""
    refs = {}
    for r in read(text):
        try:
            D = float(r["D"])
            if "published_Delta" in r:
                both = [("ours", r.get("ours_Delta")), ("published", r.get("published_Delta"))]
                refs.setdefault(D, []).extend((n, float(v)) for n, v in both if v)
            else:
                refs.setdefault(D, []).append((r.get("ref") or "published", float(r["Delta"])))
        except (KeyError, TypeError, ValueError):
            pass
    return refs


def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def transcript(text):
    """The Bash commands, the tools used, and the closing result of a stream-json transcript."""
    cmds, tools, result = [], {}, {}
    for line in text.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "result":
            result = ev
        if ev.get("type") != "assistant":
            continue
        for d in walk(ev):
            if d.get("type") == "tool_use":
                tools[d.get("name")] = tools.get(d.get("name"), 0) + 1
                if d.get("name") == "Bash":
                    cmds.append(str((d.get("input") or {}).get("command", "")))
    return cmds, tools, result


def score(results, reference, text, want, tol):
    got, refs = trial_table(results), references(reference)
    cmds, tools, result = transcript(text)
    two = any(n == "ours" for v in refs.values() for n, _ in v)
    count, rows, reach, unbroken, agree_pub = {"OK": 0, "BAD": 0, "MISSING": 0, "NO REFERENCE": 0}, [], None, True, 0
    for i, D in enumerate(want):
        r = refs.get(D, [])
        if D not in got:
            verdict, line = "MISSING", f"| {D:g} | | | | MISSING |"
        elif not r:
            verdict, line = "NO REFERENCE", f"| {D:g} | {got[D]:.7f} | | | NO REFERENCE |"
        else:
            val = got[D]
            name, ref = r[0] if two else min(r, key=lambda p: abs(p[1] - val))
            verdict = "OK" if abs(val - ref) <= tol else "BAD"
            line = f"| {D:g} | {val:.7f} | {ref:.7f} ({name}) | {val - ref:+.1e} | {verdict} |"
            pub = [v for n, v in r if n == "published"]
            if two and pub:
                agree_pub += abs(val - pub[0]) <= tol
                line += f" published {pub[0]:.6f}, {val - pub[0]:+.1e} |"
        count[verdict] += 1
        unbroken = unbroken and verdict == "OK"
        if unbroken:
            reach = D
        if verdict != "OK" or i % 10 == 0 or i == len(want) - 1:
            rows.append(line)
    net = [c for c in cmds if NET.search(c)] + [f"tool {t}" for t in tools if t in ("WebFetch", "WebSearch")]
    out = ["| D | trial | reference | difference | verdict |", "|---|---|---|---|---|"] + rows
    out.append(f"wanted {len(want)}: " + ", ".join(f"{k} {v}" for k, v in count.items() if v) + (f"; within {tol:g} of the published value: {agree_pub}" if two else ""))
    out.append(f"reach: every wanted D up to {reach:g} is OK" if reach is not None else "reach: the first wanted D is not OK")
    u = next(iter((result.get("modelUsage") or {}).values()), None) or {}
    out.append(f"use: {result.get('num_turns', '?')} turns, {result.get('duration_ms', 0) / 3.6e6:.2f} h wall, {result.get('duration_api_ms', 0) / 6e4:.0f} min of model time, "
               f"tokens out {u.get('outputTokens', '?')}, cache read {u.get('cacheReadInputTokens', '?')}, cache write {u.get('cacheCreationInputTokens', '?')}, "
               f"in {u.get('inputTokens', '?')}; ended as {result.get('subtype', 'no result event')}")
    out.append(f"the CLI's cost figure: ${result.get('total_cost_usd', float('nan')):.2f}, cost basis {u.get('costBasis', 'not stated')} (not a price unless the basis is known)")
    out.append(f"tools: {tools}")
    out.append(f"network: {len(net)} commands" + "".join(f"\n  {c[:200]!r}" for c in net))
    return count["OK"] == len(want) and not net, "\n".join(out)


def selftest(mutate):
    published = "# comment\nD,Delta,uncertainty,ref\n4.0,3.445452402,3e-9,A\n4.0,3.4454527,1e-6,B\n6.0,3.009483,5e-7,C\n"
    results = "D,Delta\n4,3.445452\n" + ("" if mutate == "missing" else f"6,{3.009493 if mutate == 'value' else 3.0094834}\n")
    cmd = "curl -s https://example.org/table" if mutate == "net" else "python3 solve.py --D 6"
    text = "\n".join(json.dumps(e) for e in [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]}},
        {"type": "result", "subtype": "success", "total_cost_usd": 1.5, "num_turns": 7, "duration_ms": 360000, "duration_api_ms": 60000,
         "modelUsage": {"m": {"outputTokens": 9000, "costBasis": "unknown"}}}])
    ok1, report = score(results, published, text, [4.0, 6.0], 1e-6)
    print(report)
    print(f"selftest: planted trial against one reference, {'PASS' if ok1 else 'FAIL'}")
    # two references, the published one wrong at D = 7: a right trial passes, one that copies the published value fails
    compare = "D,published_ref,published_Delta,published_unc,ours_Delta\n6.0,C,3.009483,5e-7,3.0094832\n7.0,C,2.842000,5e-7,2.8427532\n8.0,C,2.711790,5e-7,2.7117901\n"
    results = f"D,Delta,uncertainty\n6,3.009483,1e-9\n7,{2.842000 if mutate == 'copied' else 2.842753},1e-9\n8,2.711790,1e-9\n"
    ok2, report = score(results, compare, text, [6.0, 7.0, 8.0], 1e-6)
    print(report)
    fine = ok2 and "within 1e-06 of the published value: 2" in report and "up to 8 is OK" in report
    print(f"selftest: planted trial against two references, {'PASS' if fine else 'FAIL'}")
    return ok1 and fine


def wanted(spec):
    if ":" in spec:
        a, b = spec.split(":")
        return [float(d) for d in range(int(a), int(b) + 1)]
    return [float(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial", nargs="?")
    ap.add_argument("reference", nargs="?")
    ap.add_argument("--want", default="")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", default="")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.mutate) else 1)
    if not (a.trial and a.reference and a.want):
        sys.exit("give TRIAL_DIR, REFERENCE_CSV and --want (see --help)")
    t = Path(a.trial)
    res = t / "work" / "results.csv"
    ok, report = score(res.read_text() if res.exists() else "D,Delta\n", Path(a.reference).read_text(),
                       (t / "transcript.jsonl").read_text(errors="replace"), wanted(a.want), a.tol)
    print(report)
    print(f"verdict: {'OK' if ok else 'BAD'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
