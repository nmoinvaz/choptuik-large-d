#!/usr/bin/env python3
"""A new leg of a finished lowD_highM.py run: its progress file copied to OUT.progress.json for --resume, with the
predictor histories' states looked up by file name in --states (a run's paths are those of the machine that wrote
them), the stop cleared, new D targets, and the wall-time count restarted. The pairs, nodes, resolutions, helper and
series orders carry over, so the leg continues the run's predictors and its move-up rule.

    python3 lowD_highM_leg.py PROGRESS.json OUT.jsonl --targets 3.04,3.03,3.02,3.01 [--states data/lowD_states]

then lowD_highM.py with --out OUT.jsonl --resume and the run's other arguments. Exit 1 if a history state is missing
or OUT.progress.json exists. OUT must be in a raw-data folder (raw_out.py; --allow_unraw writes anyway).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from raw_out import FLAG, HELP, require_raw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("progress")
    ap.add_argument("out")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--states", default=os.path.join(HERE, "data", "lowD_states"))
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    prog = a.out + ".progress.json"
    require_raw(prog, allow=a.allow_unraw)
    if os.path.exists(prog):
        raise SystemExit(f"{prog} exists: resume that leg instead")
    st = json.load(open(a.progress))
    if not st.get("done"):
        raise SystemExit(f"{a.progress} has not stopped: resume that run instead")
    missing = []

    def relink(hist):
        out = []
        for D, p in hist:
            q = os.path.abspath(os.path.join(a.states, p.replace("\\", "/").split("/")[-1]))
            if not os.path.exists(q):
                missing.append(q)
            out.append([D, q])
        return out

    for k in ("hist_c", "hist_w", "hist_n"):
        st[k] = relink(st.get(k) or [])
    if missing:
        print("missing states:\n  " + "\n  ".join(missing))
        sys.exit(1)
    targets = sorted((float(x) for x in a.targets.split(",")), reverse=True)
    last = st["hist_n"][-1][0]
    if targets[0] >= last:
        raise SystemExit(f"the targets must lie below the last point, D = {last}")
    # the finished run's stop and hours, for the record; post and the current point's start do not carry over
    st["leg_from"] = dict(progress=os.path.abspath(a.progress), done=st["done"], t_used=st["t_used"])
    st.update(done=None, targets=targets, t_used=0.0, post=None, point_start=None)
    os.makedirs(os.path.dirname(os.path.abspath(prog)), exist_ok=True)
    json.dump(st, open(prog, "w"), indent=1)
    print(f"{prog}: M {st['Mw']}/{st['Mn']} (helper {st.get('Mc')}), last D {last}, targets {targets}, "
          f"{len(st['hist_w'])} / {len(st['hist_n'])} history states, nodes {st['nodes'][0]}..{st['nodes'][-1]}")


if __name__ == "__main__":
    main()
