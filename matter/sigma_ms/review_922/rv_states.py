#!/usr/bin/env python3
"""Review of smdriver.newton_band (#922), R6: every stored state of the band solver's runs is a root of the kernel's
system. Each state file of the named raw folders is loaded with its stored plan, and Solver.residual is evaluated
there by this script, which calls none of newton_band, newton_solve or run_eta.py.

    PYTHONPATH=<site> python3 rv_states.py RAW_CLONE --folders I845-mac,I913-mac,... [--mutate state] [--out FILE.json]
      Criteria, written before the run. For every state whose run record says converged:
        (1) max |F| is below 1e-28, the runs' tolerance;
        (2) max |F| equals the F of the run's record to 1e-3 of it.
      States without a record, or whose record says not converged, are listed and not judged.
      --mutate state   1e-20 is added to the first unknown of each state, in its low part: (1) must fail, and the
                       exit is 1. (Added to the high part it is rounded away and nothing changes: the first run of
                       this control did that and exited 0, raw I922-mac/states/mut_state_noop.txt.)
      Exit 0 only if every judged state meets (1) and (2).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import smdriver as sd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--folders", required=True)
    ap.add_argument("--mutate", default="", choices=["", "state"])
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    rows, bad, solvers = [], 0, {}
    for folder in a.folders.split(","):
        out = os.path.join(a.raw, folder, "out")
        recs = {}
        rp = os.path.join(out, "sm_records.jsonl")
        if os.path.exists(rp):
            for line in open(rp):
                r = json.loads(line)
                recs[r["tag"]] = r                      # the last record of a tag is the one that wrote the state
        for path in sorted(glob.glob(os.path.join(out, "sm_*.npz"))):
            z, meta = sd.load(path)
            if len(z) != 2:
                rows.append(dict(folder=folder, file=os.path.basename(path), judged=False, why=f"{len(z)} parts"))
                continue
            zh, zl = np.asarray(z[0], float), np.asarray(z[1], float).copy()
            if a.mutate == "state":
                zl[0] += 1e-20                          # into the low part: the high part would round it away
            key = (str(meta["eta"]), meta["M"], tuple(meta["nodes"]), meta["kt"], meta["kc"], meta["ks"],
                   float(meta.get("D", 4.0)))
            if key not in solvers:
                eta = None if meta["eta"] == "inf" else float(meta["eta"])
                solvers[key] = sd.make_solver(eta, meta["M"], nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"],
                                              ks=meta["ks"], threads=a.threads, D=float(meta.get("D", 4.0)))
            s = solvers[key]
            s.set_segs(meta["segs"])
            Fp = s.residual([zh, zl])
            F = sd.fnorm(Fp)
            F2 = float(np.linalg.norm(sum(np.asarray(f) for f in Fp)))
            rec = recs.get(meta.get("tag", ""))
            row = dict(folder=folder, file=os.path.basename(path), eta=str(meta["eta"]), D=float(meta.get("D", 4.0)),
                       M=meta["M"], K=len(meta["nodes"]), n=len(zh), F=F, F2=F2, Delta=repr(float(zh[-1])))
            if rec is None or not rec.get("converged"):
                row.update(judged=False, why="no record" if rec is None else "record: not converged")
            else:
                ok1 = bool(F < 1e-28)
                ok2 = bool(abs(F - rec["F"]) <= 1e-3 * rec["F"])
                row.update(judged=True, F_record=rec["F"], solver=rec.get("solver"), ok=ok1 and ok2, ok1=ok1, ok2=ok2)
                bad += not (ok1 and ok2)
            rows.append(row)
            j = row.get("judged")
            print(f"  {folder}/{row['file']}: eta {row['eta']}, D {row['D']}, M {row['M']}, K {row['K']}: max |F| "
                  f"{F:.3e}" + (f", record {row['F_record']:.3e} ({row['solver']}): {'OK' if row['ok'] else 'FAIL'}"
                                if j else f" (not judged: {row['why']})"), flush=True)
    judged = [r for r in rows if r.get("judged")]
    Fm = max((r["F"] for r in judged), default=float("nan"))
    ok = bad == 0 and len(judged) > 0
    print(f"R6{' (mutation ' + a.mutate + ')' if a.mutate else ''}: {len(rows)} state files, {len(judged)} judged, "
          f"{bad} fail; largest max |F| of the judged {Fm:.3e}; by solver "
          f"{ {k: sum(r.get('solver') == k for r in judged) for k in sorted({str(r.get('solver')) for r in judged})} }: "
          f"{'PASS' if ok else 'FAIL'}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(mutate=a.mutate, rows=rows, judged=len(judged), bad=bad, F_max=Fm, ok=ok), fh, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
