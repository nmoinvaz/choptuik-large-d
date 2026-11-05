#!/usr/bin/env python3
"""The --tau_map option of lowD_highM.py (notes/LOW_D_SPEEDUP.md, step 5) against the uniform path: one continuation
point (D 4 -> 3.95, M 48 / 64, from hp/data/lowD_states/lowDn_D4.0000_M{48,64}.npz) run three times through
lowD_highM.py, (U) without the option, (I) with the identity map --tau_map S0,1 (the start states copied with that
tau_map in their meta), (X) the control, --tau_map S0,1+1e-9.
Pass: (I) equals (U) bit for bit (every Delta, |F|, iteration and step count of the pair record, and both words of
every saved state), and (X) differs from (U) in Delta (the comparison can fail).

    python3 tau_map_leg_check.py [--threads 1] [--s0 1.83] [--keep DIR]

Exit 0 when both hold. Minutes at one thread.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from hpkernel.driver import load, save  # noqa: E402

KEYS = ("D", "Mw", "Mn", "Delta", "Delta_lo", "Delta_w", "Delta_w_lo", "diff", "F_w", "F_n", "its_w", "its_n",
        "steps_w", "steps_n")


def run(work, name, threads, tau_map=None):
    d = os.path.join(work, name)
    os.makedirs(os.path.join(d, "states"))
    starts = []
    for M in (48, 64):
        src = os.path.join(HERE, "data", "lowD_states", f"lowDn_D4.0000_M{M}.npz")
        if tau_map:
            zh, zl, m = load(src)
            dst = os.path.join(d, f"start_M{M}.npz")
            save(dst, zh, zl, dict(m, tau_map=[tau_map[0], tau_map[1], None]))
            src = dst
        starts.append(src)
    cmd = [sys.executable, "-u", os.path.join(HERE, "lowD_highM.py"), "--start_w", starts[0], "--start_n", starts[1],
           "--Ms", "48,64", "--dD", "0.05", "--D_end", "3.9", "--diff_max", "1", "--hours", "2", "--point_hours", "2",
           "--threads", str(threads), "--out", os.path.join(d, "out.jsonl"), "--points", os.path.join(d, "points.jsonl"),
           "--states", os.path.join(d, "states"), "--allow_unraw"]
    if tau_map:
        cmd += ["--tau_map", f"{tau_map[0]!r},{tau_map[1]!r}"]
    with open(os.path.join(d, "log.txt"), "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
    pairs = [json.loads(ln) for ln in open(os.path.join(d, "out.jsonl")) if ln.strip()]
    pairs = [p for p in pairs if "Delta" in p]
    states = {}
    for fn in sorted(os.listdir(os.path.join(d, "states"))):
        if fn.endswith(".npz") and "_ckpt" not in fn:
            zh, zl, m = load(os.path.join(d, "states", fn))
            states[fn] = (np.asarray(zh), np.asarray(zl), m)
    return r.returncode, pairs, states


def same(u, v):
    """differences between two runs' pair records (KEYS) and states (both words); [] when bit for bit equal"""
    out = []
    if len(u[1]) != len(v[1]) or not u[1]:
        out.append(f"pair records {len(u[1])} against {len(v[1])}")
    for p, q in zip(u[1], v[1]):
        out += [f"pair {k}: {p.get(k)!r} against {q.get(k)!r}" for k in KEYS if p.get(k) != q.get(k)]
    if sorted(u[2]) != sorted(v[2]):
        out.append(f"states {sorted(u[2])} against {sorted(v[2])}")
    for fn in sorted(set(u[2]) & set(v[2])):
        (h1, l1, m1), (h2, l2, m2) = u[2][fn], v[2][fn]
        nd = int(np.sum(h1 != h2) + np.sum(l1 != l2)) if h1.shape == h2.shape else -1
        if nd:
            out.append(f"{fn}: {nd} differing words")
        if m1.get("segs") != m2.get("segs"):
            out.append(f"{fn}: plans differ")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--s0", type=float, default=1.83)
    ap.add_argument("--keep", default="", help="keep the runs' files in this directory")
    a = ap.parse_args()
    work = a.keep or tempfile.mkdtemp(prefix="tau_map_leg_")
    os.makedirs(work, exist_ok=True)
    U = run(work, "U", a.threads)
    I_ = run(work, "I", a.threads, (a.s0, 1.0))
    X = run(work, "X", a.threads, (a.s0, 1.0 + 1e-9))
    for name, r in (("U", U), ("I", I_), ("X", X)):
        p = r[1][-1] if r[1] else {}
        print(f"({name}) exit {r[0]}, {len(r[1])} pair record(s), Delta {p.get('Delta')!r} (lo {p.get('Delta_lo')!r}), "
              f"Delta_w {p.get('Delta_w')!r}, {len(r[2])} states")
    d_ui = same(U, I_)
    d_ux = same(U, X)
    ok_i = U[0] == 0 and I_[0] == 0 and not d_ui
    fired = bool(U[1]) and bool(X[1]) and U[1][-1].get("Delta") != X[1][-1].get("Delta")
    print("(I) against (U): " + ("bit for bit equal, PASS" if ok_i else "FAIL\n  " + "\n  ".join(d_ui or ["a run failed"])))
    print(f"(X) against (U): {len(d_ux)} differences" + (", Delta differs: the control fires" if fired else
          ", Delta equal: the control does NOT fire (FAIL)"))
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(0 if ok_i and fired else 1)


if __name__ == "__main__":
    main()
