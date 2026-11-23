#!/usr/bin/env python3
"""Review of #1079, check 5: the table of margins of the stored box files and of the perturbed reruns.

    python3 review_1079/perturb_table.py RAW          from solver/proof; prints Markdown

Reads the first line of each `certify_M.py krawczyk` log: macm3's and dh610's own (their box with their point file)
and the review's (`I1079-mac-review/logs/A_M_krawczyk_pt_*`: the rerun box with macm3's point file).
"""
import glob
import os
import re
import sys

PAT = re.compile(r"K\(X\) in int X: (True|False); relative margins ([-+0-9., ]+)")
ZERO = {"x1": "M3x1E2_h1e-12_w1e-13", "x2": "M3x2E2_h1e-12_w1e-12"}
WHAT = {"none": "unchanged code", "qr": "frames Q", "perron": "start lemma's weights"}


def first(path):
    with open(path) as fh:
        m = PAT.search(fh.readline())
    return m.group(1) == "True", [float(v) for v in m.group(2).split(",")]


def compact(raw):
    """one table, the two outer margins only: rows = box files, columns = the two zeros"""
    cells, order = {}, []
    for z, tag in ZERO.items():
        fs = [("macm3, stored", os.path.join(raw, "I1079-macm3/levers/logs", f"A_M_krawczyk_pt_{tag}.log")),
              ("dh610, stored", os.path.join(raw, "I1079-dh610/out/logs", f"A_M_krawczyk_pt_{tag}.log"))]
        for f in sorted(glob.glob(os.path.join(raw, "I1079-mac-review/logs", f"A_M_krawczyk_pt_{tag}_*.log"))):
            m = re.search(tag + r"_(none|qr|perron)_s(\d+)(?:_u(\d+))?\.log$", f)
            what, seed, ulps = m.group(1), m.group(2), m.group(3) or "1"
            if what == "none" and seed != "1079":
                continue
            name = {"none": "this Mac, unchanged code", "qr": f"this Mac, frames Q moved by 1 ulp, seed {seed}",
                    "perron": f"this Mac, start weights moved by {ulps} ulp"}[what]
            fs.append((name, f))
        for name, f in fs:
            ok, mg = first(f)
            if name not in order:
                order.append(name)
            cells[(name, z)] = f"{mg[0]:+.3f}, {mg[1]:+.3f}" + ("" if ok else " **fails**")
    key = lambda n: (0 if "stored" in n else 1 if "unchanged" in n else 2 if "start" in n else 3,  # noqa: E731
                     int(re.search(r"(\d+)$", n).group(1)) if re.search(r"seed (\d+)$", n) else 0)
    print("| box file | first zero | second zero |\n|---|---|---|")
    for name in sorted(order, key=key):
        print(f"| {name} | {cells.get((name, 'x1'), '')} | {cells.get((name, 'x2'), '')} |")


def main():
    raw = sys.argv[1]
    if "--compact" in sys.argv:
        return compact(raw)
    for z, tag in ZERO.items():
        rows = [("macm3, stored", os.path.join(raw, "I1079-macm3/levers/logs", f"A_M_krawczyk_pt_{tag}.log")),
                ("dh610, stored", os.path.join(raw, "I1079-dh610/out/logs", f"A_M_krawczyk_pt_{tag}.log"))]
        for f in sorted(glob.glob(os.path.join(raw, "I1079-mac-review/logs", f"A_M_krawczyk_pt_{tag}_*.log"))):
            m = re.search(tag + r"_(none|qr|perron)_s(\d+)(?:_u(\d+))?\.log$", f)
            what, seed, ulps = m.group(1), m.group(2), m.group(3) or "1"
            name = "this Mac, " + WHAT[what] + ("" if what == "none" else f" moved by {ulps} ulp") + \
                (f", seed {seed}" if what == "qr" else (", repeated" if (what, seed) == ("none", "2") else ""))
            rows.append((name, f))
        print(f"\n{'First' if z == 'x1' else 'Second'} zero, box (h, w) = (1e-12, {'1e-13' if z == 'x1' else '1e-12'}):\n")
        print("| box run | passes | margins (A, ω/ω∞, B₃, C₃) |\n|---|---|---|")
        lo = []
        for name, f in rows:
            ok, mg = first(f)
            lo.append((ok, mg))
            print(f"| {name} | {'yes' if ok else '**no**'} | " + ", ".join(f"{v:+.3f}" for v in mg) + " |")
        a = [mg[0] for _, mg in lo]
        w = [mg[1] for _, mg in lo]
        print(f"\n{len(lo)} box files, {sum(1 for ok, _ in lo if ok)} pass; A margin from {min(a):+.3f} to {max(a):+.3f}, "
              f"ω/ω∞ margin from {min(w):+.3f} to {max(w):+.3f}; smallest margin of any coordinate "
              f"{min(min(mg) for _, mg in lo):+.3f}.")


if __name__ == "__main__":
    main()
