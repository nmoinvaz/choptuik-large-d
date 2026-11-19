#!/usr/bin/env python3
"""#1021's merge of argcount points files computed in parts (one machine's cache, the others' from a copy of it).

    python3 argcount_merge.py OUT.jsonl BASE.jsonl PREFIX.jsonl PART.jsonl [PART.jsonl ...] [--mutate=prefix]

OUT is BASE unchanged, then each PART's lines after PREFIX. PREFIX is the input every PART started from. BASE and each
PART must begin with PREFIX byte for byte, or nothing is written and the exit is 1. A PART record is dropped when its
point already has a valued record (tuned below 1e-8, or by the continuation), so a later record never replaces a
value. --mutate=prefix changes one byte of the first PART's prefix in memory: the merge must then fail (exit 1).
"""
import json
import os
import sys


def valued(r):
    return (isinstance(r.get("info"), float) and r["info"] < 1e-8) or bool((r.get("fix") or {}).get("ok"))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mutate = "--mutate=prefix" in sys.argv
    out, base, prefix, parts = args[0], args[1], args[2], args[3:]
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    pre = open(prefix).read().splitlines(True)
    blines = open(base).read().splitlines(True)
    ok = blines[:len(pre)] == pre
    print(f"{base}: {len(blines)} lines, begins with {prefix}'s {len(pre)} lines: {ok}")
    merged = list(blines)
    have = {}
    for line in blines:
        r = json.loads(line)
        k = (r["A"], r["wf"])
        have[k] = have.get(k, False) or valued(r)
    for i, part in enumerate(parts):
        plines = open(part).read().splitlines(True)
        head = list(plines[:len(pre)])
        if mutate and i == 0:
            head[0] = head[0].replace("1", "2", 1)
        same = head == pre
        ok &= same
        added = dropped = over = 0
        for line in plines[len(pre):]:
            r = json.loads(line)
            k = (r["A"], r["wf"])
            if have.get(k):
                dropped += 1
                continue
            over += k in have
            merged.append(line)
            have[k] = valued(r)
            added += 1
        print(f"{part}: {len(plines)} lines, begins with the prefix: {same}; after it, {added} records added "
              f"({over} over a point without value), {dropped} dropped (point already valued)")
    if not ok:
        print("merge: FAIL (a file does not begin with the prefix), nothing written")
        return 1
    with open(out, "w") as fh:
        fh.writelines(merged)
    nv = sum(have.values())
    print(f"merge: PASS, {out}: {len(merged)} lines, {len(have)} points, {nv} valued, {len(have) - nv} without value")
    return 0


if __name__ == "__main__":
    sys.exit(main())
