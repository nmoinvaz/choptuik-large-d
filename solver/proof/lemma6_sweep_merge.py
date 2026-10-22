#!/usr/bin/env python3
"""Merge and check the chunks of the lemma 6b_1 tube sweep (lemma6_sweep.py --chunk=k/K --out=...).

Lemma 6b_1 is proved by the sweep if, and only if:
  - every chunk was made with the same h, dfrac, ratio and node list, and no mutation;
  - ratio >= (1 - dfrac)/(1 + dfrac), so consecutive arc tubes overlap and cover 25 <= |u| <= 260 (coverage_ok);
  - the arcs j = 0 .. N-1 all appear, and every arc tube and every ray tube finished.

    python3 lemma6_sweep_merge.py chunk_0.json chunk_1.json ...      # exit status 0 only if all of the above hold
"""
import json
import sys


def main(paths):
    runs = [json.load(open(p)) for p in paths]
    keys = [(r["h"], r["dfrac"], r.get("ratio"), tuple(r["rho"]), r["mutate"]) for r in runs]
    same = len(set(keys)) == 1
    r0 = runs[0]
    N = len(r0["rho"])
    arcs = {}
    for r in runs:
        for a in r["arcs"]:
            arcs.setdefault(a["j"], []).append(a["ok"])
    missing = [j for j in range(N) if j not in arcs]
    failed = sorted(j for j, oks in arcs.items() if not all(oks))
    ray_ok = all(all(a["ok"] for a in r["ray"]) for r in runs)
    secs = sum(r["sec"] for r in runs)
    ok = same and r0["mutate"] is None and bool(r0.get("coverage_ok")) and not missing and not failed and ray_ok
    print(f"{len(paths)} chunks, same parameters {same} (h {r0['h']}, dfrac {r0['dfrac']}, ratio {r0.get('ratio')}), "
          f"coverage {r0.get('coverage_ok')}; arcs present {N - len(missing)} of {N}, failed {len(failed)}"
          f"{' ' + str(failed[:10]) if failed else ''}; ray tubes ok {ray_ok}; {secs / 3600:.1f} CPU hours")
    print(f"lemma 6b_1 {'PROVED by the sweep' if ok else 'NOT established'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
