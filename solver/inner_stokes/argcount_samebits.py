#!/usr/bin/env python3
"""#1021's bit check: every point that has a value in an earlier argcount points file must have the same record, bit
for bit, in a later one (a recount appends new records; a point computed again would append a second record).

    python3 argcount_samebits.py OLD.jsonl NEW.jsonl [--mutate=flip]

Prints the number of valued points, how many keep their record exactly, and how many old points without value got
one. --mutate=flip changes the last bit of one old value in memory: the check must then fail (exit 1).
"""
import json
import struct
import sys


def valued(r):
    return (isinstance(r.get("info"), float) and r["info"] < 1e-8) or bool((r.get("fix") or {}).get("ok"))


def last_records(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["A"], r["wf"])] = r
    return out


def main():
    old_path, new_path = sys.argv[1], sys.argv[2]
    old, new = last_records(old_path), last_records(new_path)
    if "--mutate=flip" in sys.argv:
        k = next(k for k, r in old.items() if valued(r))
        b = struct.unpack("<q", struct.pack("<d", old[k]["re"]))[0] ^ 1
        old[k] = dict(old[k], re=struct.unpack("<d", struct.pack("<q", b))[0])
    vk = [k for k, r in old.items() if valued(r)]
    same = sum(1 for k in vk if k in new and json.dumps(new[k], sort_keys=True) == json.dumps(old[k], sort_keys=True))
    gained = sum(1 for k, r in old.items() if not valued(r) and k in new and valued(new[k]))
    lost = len([k for k, r in old.items() if not valued(r)])
    lines_old = sum(1 for _ in open(old_path))
    prefix = all(a == b for a, b in zip(open(old_path), open(new_path)))
    print(f"valued points in {old_path.split('/')[-1]}: {len(vk)}; the same record in {new_path.split('/')[-1]}: "
          f"{same}; old points without value: {lost}, of which valued now: {gained}; the old file's {lines_old} lines "
          f"are the new file's first lines unchanged: {prefix}")
    ok = same == len(vk) and prefix
    print("bit check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
