#!/usr/bin/env python3
"""The step history of the re-seed's quad-double u0 polish on a stored prediction (#278).

    python3 reseed_hist.py PRED.npz [--threads 4] [--hp HP_DIR]

PRED.npz is a prediction saved by anm_run.py --save_pred 1, re-seeded at its own D as the chain does (reseed.Reseeder
with qd, the call of reseed.py checkq). After the rungs it prints every iterate of Reseeder.invert_q: the residual of
the u0 rung rows, the block steps in a_h and m_h ("pre") and the Newton steps with their refactorings. --hp imports
reseed.py and the kernel's driver from another checkout's hp directory, so a chain's own code and build can be tested.
It writes no file. Exit 0 when the re-seed converged, 1 otherwise.
"""
import os
import sys
import time


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    opt = dict(threads=4, hp=os.path.dirname(os.path.abspath(__file__)))
    for k in opt:
        if "--" + k in argv:
            opt[k] = type(opt[k])(argv[argv.index("--" + k) + 1])
            args.remove(argv[argv.index("--" + k) + 1])
    sys.path.insert(0, opt["hp"])
    import mpmath as mp
    from hpkernel.driver import load_q
    from reseed import Reseeder
    z4, meta = load_q(args[0])
    rs = Reseeder(meta, mp.mpf(meta["D_str"]), threads=opt["threads"], qd=True)
    polish, kept = rs.invert_q, []

    def recorded(z, d, **kw):
        kept.append(polish(z, d, **kw))
        return kept[-1]

    rs.invert_q = recorded
    t0 = time.time()
    _, diag = rs.reseed(z4, verbose=True)
    print(f"reseed_hist {args[0]}: D {meta['D_str']}, threads {opt['threads']}, re-seed "
          f"{'ok' if diag['ok'] else 'FAILED'}, {time.time() - t0:.0f} s", flush=True)
    for h in (kept[-1][1] if kept else []):
        print("    it {it:2d}  |R| {R:.3e}".format(**h) + (f"  pre {h['pre']}" if "pre" in h else "")
              + (f"  refactors {h['refactors']}" if "refactors" in h else ""), flush=True)
    return 0 if diag["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
