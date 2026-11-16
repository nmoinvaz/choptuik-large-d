#!/usr/bin/env python3
"""The chain in D with the horizon series' order tied to D (#903): anm_run.py in legs of fixed order ks, each leg
started from the last state of the one before with the order raised for its end.

The order follows the singular exponent of the horizon series. Re s grows linearly in D over the stored states of
hp/data/horizon_radius.csv (#693), Re s = 114.16 at D 225.131 to 128.509 at 253.804, so Re s(D) is the least-squares
line through them (about D/2 + 1.6). A leg from D0 to D1 = D0 + --leg runs at

    ks = ceil(--mult * Re s(D1)) + --pad          (defaults 1.95 and 10: ks 172 for D 152-162, 404 at D 400)

so the series is summed past about twice its exponent everywhere in the leg. At ks 130 the series is cut below that
from about D 150 on, and the cut grows to e^-30 of the matching determinant at D 225 (notes/FACTORED_HORIZON.md 4.2).

Each leg is one anm_run.py with the given flags plus --ks K --keep_plan 0 --D_max D1, into OUT/leg<i>_D<D0>_ks<K>.
--keep_plan 0 re-plans the base for the new order, and the base is polished at that order before the first step.
The driver stops when a leg exits non-zero, when a leg moves D by less than --min_progress (0.05), or at --D_stop.
It resumes from the highest-D state of its legs when started again with the same OUT. Every leg is recorded in
OUT/ks_chain.jsonl.

    python3 ks_chain.py RAW OUT --state STATE --D_stop 400 [--leg 10] [--mult 1.95] [--pad 10] [--dry] \
        -- <anm_run flags>

STATE is relative to RAW, as for anm_run.py; the anm_run flags include --steps. --dry prints the schedule and the first
leg's command and runs nothing. --anm replaces anm_run.py (for the driver's own test with a stub).
"""
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_RE = re.compile(r"anm_pin\d+_D(\d+\.\d+)\.npz$")


def re_s_line(path=os.path.join(HERE, "data", "horizon_radius.csv")):
    """the least-squares line Re s = a + b D through the distinct (D, Re s) pairs of horizon_radius.csv"""
    pts = sorted({(float(r["D"]), float(r["Re_s"])) for r in csv.DictReader(open(path)) if r["Re_s"]})
    n = len(pts)
    mx = sum(d for d, _ in pts) / n
    my = sum(s for _, s in pts) / n
    b = sum((d - mx) * (s - my) for d, s in pts) / sum((d - mx) ** 2 for d, _ in pts)
    return my - b * mx, b, pts


def order_for(D, a, b, mult, pad):
    return int(math.ceil(mult * (a + b * D))) + pad


def last_state(raw, out, first):
    """(D, path relative to raw) of the highest-D state among the legs in out, else of the first state"""
    best = None
    for p in glob.glob(os.path.join(out, "leg*", "anm_pin*_D*.npz")):
        m = STATE_RE.search(os.path.basename(p))
        if m and (best is None or float(m.group(1)) > best[0]):
            best = (float(m.group(1)), os.path.relpath(p, raw).replace(os.sep, "/"))
    if best is None:
        m = STATE_RE.search(os.path.basename(first)) or re.search(r"_D(\d+\.\d+)\.npz$", first)
        if not m:
            raise SystemExit(f"ks_chain: cannot read D from the state name {first}")
        best = (float(m.group(1)), first)
    return best


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    opt = dict(state=None, D_stop=400.0, leg=10.0, mult=1.95, pad=10, min_progress=0.05, dry=False,
               py=sys.executable, anm=os.path.join(HERE, "anm_run.py"))
    pos = []
    it = iter(argv)
    for a in it:
        if a == "--dry":
            opt["dry"] = True
        elif a.startswith("--"):
            k = a[2:]
            if k not in opt:
                raise SystemExit(f"ks_chain: unknown option {a}")
            v = next(it)
            opt[k] = type(opt[k])(v) if opt[k] is not None and not isinstance(opt[k], bool) else v
        else:
            pos.append(a)
    if len(pos) != 2 or not opt["state"]:
        raise SystemExit(__doc__)
    for bad in ("--ks", "--keep_plan", "--D_max", "--state"):
        if bad in extra:
            raise SystemExit(f"ks_chain: {bad} is set by the driver, not in the anm_run flags")
    raw, out = pos
    a, b, pts = re_s_line()
    print(f"Re s = {a:.4f} + {b:.6f} D from {len(pts)} states of horizon_radius.csv (D {pts[0][0]} to {pts[-1][0]}); "
          f"ks = ceil({opt['mult']} Re s(leg end)) + {opt['pad']}", flush=True)
    D0, st = last_state(raw, out, opt["state"])
    sched, d = [], D0
    while d < opt["D_stop"] - 1e-9:
        d1 = min(d + opt["leg"], opt["D_stop"])
        sched.append((d, d1, order_for(d1, a, b, opt["mult"], opt["pad"])))
        d = d1
    for d, d1, k in sched:
        print(f"  leg D {d:.4f} -> {d1:.4f}: ks {k}", flush=True)
    os.makedirs(out, exist_ok=True)
    rec_path = os.path.join(out, "ks_chain.jsonl")
    i = len(glob.glob(os.path.join(out, "leg*")))
    while D0 < opt["D_stop"] - 1e-9:
        D1 = min(D0 + opt["leg"], opt["D_stop"])
        K = order_for(D1, a, b, opt["mult"], opt["pad"])
        leg = os.path.join(out, f"leg{i:02d}_D{D0:.4f}_ks{K}")
        cmd = [opt["py"], "-u", opt["anm"], raw, leg, "--state", st] + extra + \
              ["--ks", str(K), "--keep_plan", "0", "--D_max", repr(D1)]
        print(f"leg {i}: D {D0:.4f} -> {D1:.4f} at ks {K} from {st}", flush=True)
        if opt["dry"]:
            print("  " + " ".join(cmd), flush=True)
            return 0
        os.makedirs(leg, exist_ok=True)
        t = time.time()
        with open(os.path.join(leg, "anm_run.log"), "a") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        Dn, stn = last_state(raw, out, opt["state"])
        rec = dict(leg=i, D_from=D0, D_to=D1, ks=K, state=st, rc=rc, wall_s=round(time.time() - t), D_end=Dn,
                   state_end=stn, utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        with open(rec_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  exit {rc}, wall {rec['wall_s']} s, last state D {Dn:.4f}", flush=True)
        if rc != 0:
            print("ks_chain: the leg exited non-zero; stop", flush=True)
            return rc
        if Dn < D0 + opt["min_progress"]:
            print(f"ks_chain: the leg moved D by less than {opt['min_progress']}; stop", flush=True)
            return 3
        D0, st, i = Dn, stn, i + 1
    print(f"ks_chain: reached D {D0:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
