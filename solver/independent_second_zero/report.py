"""Markdown tables from second_zero.py's JSON outputs (issue #203).

    python3 report.py <raw clone>/I203-mac/v2
"""
import json
import os
import sys

D = sys.argv[1]
WS = ["1.09", "1.10", "1.11"]
SETS = ["primary", "qhp23", "qhp22", "qhp21", "pade", "pade23", "pade22", "pade21"]


def node_rows(res, sets=SETS):
    print("| set | n per node | x1(1.09) | x1(1.10) | x1(1.11) | crossing | envelope |")
    print("|---|---|---|---|---|---|---|")
    for s in sets:
        r = res[s]
        if not r["n"]:
            print(f"| {s} | none at {r.get('empty_node')} | | | | | |")
            continue
        cells = []
        for w in WS:
            q = r["node"][w]
            cells.append(f"{q['re']:.5f} {q['im']:+.5f}i (±{q['re_half_range']:.1e}, ±{q['im_half_range']:.1e}i)")
        ns = "/".join(str(r["node"][w]["n"]) for w in WS)
        cr = f"{r['cross']:.5f}" if "cross" in r else "none"
        env = r["cross_envelope"]
        en = (f"{env[0]:.5f}–{env[1]:.5f}" + ("" if r["envelope_all_cross"] else " (some none)")) if env else "none"
        print(f"| {s} | {ns} | " + " | ".join(cells) + f" | {cr} | {en} |")


real = os.path.join(D, "real.json")
if os.path.exists(real):
    res = json.load(open(real))
    print("## real data\n")
    print("Domb–Sykes zeta, gamma per node:", [(d["zeta_ds"], d["gamma_ds"]) for d in res["diag"]], "\n")
    node_rows(res)
for name in ["P1_wp1.095.json", "P2_wp1.15.json", "stress_eps0.3_wp1.1.json"]:
    p = os.path.join(D, name)
    if not os.path.exists(p):
        continue
    a = json.load(open(p))
    print(f"\n## {name}: w_p {a['wp']}\n")
    print("true x1:", {w: a["truth"][w] for w in WS})
    print("\n| seed | crossing (primary) | envelope | Im x1 medians | max |x1 error| | n per node |")
    print("|---|---|---|---|---|---|")
    for seed, res in a["seeds"].items():
        r = res["primary"]
        if not r["n"]:
            print(f"| {seed} | empty at {r.get('empty_node')} | | | | |")
            continue
        err = max(abs(complex(r["node"][w]["re"], r["node"][w]["im"]) - complex(*a["truth"][w])) for w in WS)
        cr = f"{r['cross']:.5f}" if "cross" in r else "none"
        env = r["cross_envelope"]
        en = f"{env[0]:.5f}–{env[1]:.5f}" if env else "none"
        ims = ", ".join(f"{r['node'][w]['im']:+.4f}" for w in WS)
        ns = "/".join(str(r["node"][w]["n"]) for w in WS)
        print(f"| {seed} | {cr} | {en} | {ims} | {err:.1e} | {ns} |")
