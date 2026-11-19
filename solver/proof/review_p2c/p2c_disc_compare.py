#!/usr/bin/env python3
"""P2c review (#992): this review's lemma D margins (p2c_disc.py records) against the certificate's stored ones
(lemma6b1_runs/row_XX.json: Lam, sup = M, box_rad), disc by disc, after the own numbers were committed (ce097b2b).

    python3 p2c_disc_compare.py p2c_disc_all.jsonl [p2c_disc_all_infl2.jsonl ...]

Per row: the largest Lambda and M of each side, the range of the ratios mine/stored, and every disc that one side
certifies and the other does not. The row maxima of the stored files are also printed for A_certify.md 12.4's table.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "lemma6b1_runs")


def stored():
    out = {}
    for j in range(12):
        with open(os.path.join(RUNS, f"row_{j:02d}.json")) as fh:
            d = json.load(fh)
        for x in d["discs"]:
            out[(j, x["k"])] = dict(ok=x["ok"], Lam=x["Lam"], sup=x["sup"], box_rad=x["box_rad"])
    return out


def main():
    S = stored()
    for path in sys.argv[1:]:
        mine = {}
        for line in open(path):
            r = json.loads(line)
            mine[(r["j"], r["k"])] = r
        print(f"== {os.path.basename(path)}: {len(mine)} discs of {len(S)} stored")
        print(f"{'row':>3} {'discs':>5} {'ok mine/stored':>15} {'max Lam mine':>12} {'stored':>8} {'Lam ratio':>15} "
              f"{'max M mine':>10} {'stored':>8} {'M ratio':>15} {'max box_rad':>11}")
        odd = []
        for j in range(12):
            keys = sorted(k for k in S if k[0] == j and k in mine)
            if not keys:
                continue
            okm = sum(bool(mine[k]["ok"]) for k in keys)
            oks = sum(bool(S[k]["ok"]) for k in keys)
            both = [k for k in keys if mine[k]["ok"] and S[k]["ok"]]
            lr = [mine[k]["Lam"] / S[k]["Lam"] for k in both]
            mr = [mine[k]["Mmax"] / S[k]["sup"] for k in both]
            lmax = max((mine[k]["Lam"] for k in both), default=float("nan"))
            mmax = max((mine[k]["Mmax"] for k in both), default=float("nan"))
            print(f"{j:>3} {len(keys):>5} {f'{okm}/{oks}':>15} {lmax:>12.4f} {max(S[k]['Lam'] for k in keys):>8.4f} "
                  f"{f'{min(lr):.3f}-{max(lr):.3f}':>15} {mmax:>10.4g} {max(S[k]['sup'] for k in keys):>8.4g} "
                  f"{f'{min(mr):.3f}-{max(mr):.3f}':>15} {max(S[k]['box_rad'] for k in keys):>11.2e}")
            odd += [k for k in keys if bool(mine[k]["ok"]) != bool(S[k]["ok"])]
        for k in odd:
            m = mine[k]
            print(f"   disc {k}: mine {'ok' if m['ok'] else 'NOT certified (' + m['why'] + ')'}, stored "
                  f"{'ok' if S[k]['ok'] else 'not'} (Lam {S[k]['Lam']:.4f}, sup {S[k]['sup']:.4g}, "
                  f"box_rad {S[k]['box_rad']:.3g})")
        allb = [k for k in mine if k in S and mine[k]["ok"] and S[k]["ok"]]
        print(f"   all rows: {sum(bool(mine[k]['ok']) for k in mine)} of {len(mine)} certified here; "
              f"Lam ratio {min(mine[k]['Lam'] / S[k]['Lam'] for k in allb):.3f}-"
              f"{max(mine[k]['Lam'] / S[k]['Lam'] for k in allb):.3f}, "
              f"M ratio {min(mine[k]['Mmax'] / S[k]['sup'] for k in allb):.3f}-"
              f"{max(mine[k]['Mmax'] / S[k]['sup'] for k in allb):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
