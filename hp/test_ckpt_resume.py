#!/usr/bin/env python3
"""The resume rule of lowD_dd.solve_point (notes/LOW_D.md 12): a checkpoint is resumed only by the solve that wrote it.

    python3 test_ckpt_resume.py              exit 0: all cases pass
    python3 test_ckpt_resume.py --mutate     the former rule (any checkpoint resumed); exit 1 is the expected result

Cases, on temporary files: (1) a checkpoint written from this start is resumed; (2) one from another start is set
aside as *_stale.npz and not resumed; (3) one without a fingerprint (written before the rule) is set aside; (4) a start
with the same state but another step plan does not match; (5) the fingerprint of a state is the same when recomputed
from a copy (a resume after a restart recomputes the start)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from hpkernel.driver import save  # noqa: E402
from lowD_dd import resume_checkpoint, start_fingerprint  # noqa: E402


def main():
    mutate = "--mutate" in sys.argv
    rng = np.random.default_rng(1)
    zh, zl = rng.standard_normal(97), rng.standard_normal(97) * 1e-17
    other = zh.copy(); other[5] += 1e-12
    segs = [[0.0, 0.1, 0.2], [0.2, 0.5]]
    fp = start_fingerprint(zh, zl)
    log = lambda m: None  # noqa: E731
    fails = []
    with tempfile.TemporaryDirectory() as d:
        def ck(name, start):
            p = os.path.join(d, name)
            save(p, zh * 2, zl, dict(D=3.0, M=32, it=4, **({"start": start} if start is not None else {})))
            return p
        p = ck("a_ckpt.npz", fp)
        r = resume_checkpoint(p, fp, log, mutate)
        if r is None or not np.array_equal(r[0], zh * 2):
            fails.append("(1) own checkpoint not resumed")
        p = ck("b_ckpt.npz", start_fingerprint(other, zl))
        r = resume_checkpoint(p, fp, log, mutate)
        if r is not None or not os.path.exists(p[:-4] + "_stale.npz") or os.path.exists(p):
            fails.append("(2) another start's checkpoint resumed or not set aside")
        p = ck("c_ckpt.npz", None)
        if resume_checkpoint(p, fp, log, mutate) is not None:
            fails.append("(3) a checkpoint without a fingerprint resumed")
        p = ck("d_ckpt.npz", start_fingerprint(zh, zl, segs))
        if resume_checkpoint(p, fp, log, mutate) is not None:
            fails.append("(4) the same state with another plan resumed")
    if start_fingerprint(zh.copy(), np.array(zl, copy=True), None) != fp or start_fingerprint(zh, zl, segs) == fp:
        fails.append("(5) the fingerprint is not a function of the state and the plan")
    print("mutated (the former rule): " if mutate else "", "; ".join(fails) if fails else "all 5 cases pass", sep="")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
