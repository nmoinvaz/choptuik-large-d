"""Bisection on the amplitude A of the l-multiplet collapse (ell_collapse.py) for the black-hole threshold.
usage: python ell_bisect.py n l A_lo A_hi [rel_tol=1e-12] [dx=4e-4]
Logs every run to data/bisect_n{n}_l{l}.jsonl and the final bracket to data/pstar_n{n}_l{l}.json."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ell_collapse import run

n, l = float(sys.argv[1]), int(sys.argv[2])
lo, hi = float(sys.argv[3]), float(sys.argv[4])
tol = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-12
dx = float(sys.argv[6]) if len(sys.argv) > 6 else 4e-4
RMAX = float(os.environ.get("ELL_RMAX", "30"))
tag = f"n{n:g}_l{l}" + (f"_dx{dx:g}" if dx != 4e-4 else "") + (f"_r{RMAX:g}" if RMAX != 30 else "")
OUT = os.path.join(HERE, "data", f"bisect_{tag}.jsonl")
T_END = float(os.environ.get("ELL_TEND", "15"))
KW = dict(n=n, l=l, dx=dx, t_end=T_END, disp_tmin=float(os.environ.get("ELL_DISP", "7")), disp_frac=0.1,
          central_dt=0.01, rmax=RMAX)


def go(A):
    t0 = time.time()
    o = run(A, **KW)
    rec = dict(n=n, l=l, A=A, status=o["status"], t=o["t"], peak=o["peak"], dx=dx, wall=round(time.time() - t0, 1),
               central_time=True)
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    if o["status"] == 0 and o["t"] >= T_END:
        print("  warning: undecided at t_end (counted as no black hole)", flush=True)
    return 1 if o["status"] > 0 else 0


assert go(lo) == 0, "lower bracket must disperse"
assert go(hi) == 1, "upper bracket must collapse"
while (hi - lo) / hi > tol:
    mid = 0.5 * (lo + hi)
    if mid in (lo, hi):
        break
    if go(mid):
        hi = mid
    else:
        lo = mid
json.dump(dict(n=n, l=l, A_lo=lo, A_hi=hi, dx=dx), open(os.path.join(HERE, "data", f"pstar_{tag}.json"), "w"), indent=1)
print("DONE", lo, hi, flush=True)
