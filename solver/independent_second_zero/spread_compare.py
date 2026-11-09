"""The approximants' spread on planted families against the real data's, and the planted accuracy (issue #214).

    python3 spread_compare.py REAL_JSON KUMMER_P1_JSON ARC_P1_JSON [MORE_PLANTED_JSON ...]

REAL_JSON is second_zero.py real's output (I203-mac/v2/real.json); the others are second_zero.py planted outputs.
The spread at a node is the half-range of x1 over a set's admitted approximants (Re and Im separately), the quantity
#203 quotes as +-.  For each planted file and set, its median over the seeds is compared with the real data's.
Verdict (I214-mac/README.md, fixed before the runs): the spread of ARC_P1_JSON matches when real/arc lies in 1/3-3 at
every node for both Re and Im (primary set).  Exits 0 when it matches, 2 when it does not.
Planted accuracy: |w_c - w_p| per seed, and |x1 error| per node (complex modulus), max over seeds.
"""
import json
import statistics
import sys

WS = ["1.09", "1.10", "1.11"]
SETS = ["primary", "qhp23", "qhp22", "qhp21"]


def node_spread(s, w):
    q = s["node"][w]
    return q["re_half_range"], q["im_half_range"]


def planted_table(name, a, real):
    seeds = list(a["seeds"])
    print(f"\n### {name}: family {a.get('family', 'kummer')}, w_p {a['wp']}, noise {a.get('noise', 1e-12)}, "
          f"seeds {','.join(seeds)}\n")
    print("| set | node | spread Re: median (min-max) | spread Im: median (min-max) | real Re, Im | real/planted Re, Im |")
    print("|---|---|---|---|---|---|")
    ratios = {}
    for st in SETS:
        for w in WS:
            sp = [node_spread(a["seeds"][s][st], w) for s in seeds if a["seeds"][s][st]["n"]]
            if not sp:
                print(f"| {st} | {w} | empty | empty | | |")
                continue
            re = [x[0] for x in sp]
            im = [x[1] for x in sp]
            rr, ri = node_spread(real[st], w)
            mre, mim = statistics.median(re), statistics.median(im)
            ratios[(st, w)] = (rr / mre if mre else float("inf"), ri / mim if mim else float("inf"))
            print(f"| {st} | {w} | {mre:.1e} ({min(re):.1e}-{max(re):.1e}) | {mim:.1e} ({min(im):.1e}-{max(im):.1e}) | "
                  f"{rr:.1e}, {ri:.1e} | {ratios[(st, w)][0]:.2g}, {ratios[(st, w)][1]:.2g} |")
    wp = a["wp"]
    errs = []
    xerr = {w: [] for w in WS}
    for s in seeds:
        p = a["seeds"][s]["primary"]
        if "cross" in p:
            errs.append(abs(p["cross"] - wp))
        for w in WS:
            if p["n"]:
                t = a["truth"][w]
                xerr[w].append(abs(complex(p["node"][w]["re"] - t[0], p["node"][w]["im"] - t[1])))
    if errs:
        print(f"\nprimary crossing error |w_c - w_p|: max {max(errs):.1e}, median {statistics.median(errs):.1e} "
              f"over {len(errs)} seeds with a crossing")
    else:
        print("\nprimary: no crossing in any seed")
    for w in WS:
        if xerr[w]:
            print(f"  |x1 error| at {w}: max {max(xerr[w]):.1e}, median {statistics.median(xerr[w]):.1e}")
    return ratios


def main(argv):
    real = json.load(open(argv[1]))
    print("## Real data (second_zero.py real)\n")
    for st in SETS:
        print(f"  {st}: " + "; ".join(f"{w} +-{node_spread(real[st], w)[0]:.1e}, +-{node_spread(real[st], w)[1]:.1e}i"
                                       for w in WS))
    out = {}
    for path in argv[2:]:
        out[path] = planted_table(path, json.load(open(path)), real)
    arc = out[argv[3]]
    prim = [arc[("primary", w)] for w in WS if ("primary", w) in arc]
    ok = len(prim) == 3 and all(1 / 3 <= r <= 3 for pair in prim for r in pair)
    print(f"\nverdict (primary, {argv[3]}): real/arc spread ratios "
          + ", ".join(f"{w}: {r[0]:.2g}, {r[1]:.2g}" for w, r in zip(WS, prim))
          + f" -> {'matches (all in 1/3-3)' if ok else 'does not match'}")
    kum = out[argv[2]]
    kp = [kum[("primary", w)] for w in WS if ("primary", w) in kum]
    print("for comparison, real/Kummer: " + ", ".join(f"{w}: {r[0]:.2g}, {r[1]:.2g}" for w, r in zip(WS, kp)))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
