#!/usr/bin/env python3
"""#784: replay a refusal rule at Newton iteration 1 over anm_run.py logs of the octo-double corrector's natural test.

    python3 abort_replay.py LOG [LOG ...] [--R 1e7]
    python3 abort_replay.py --selftest

An attempt is the run of trial lines ("on k trial lam L: max|F| ...; simplified correction / |dz| = theta ...") that
ends at its "octo-double Newton: N steps, max|F| ..., converged" or "not converged" line. Its theta1 is the ratio of the
full-step trial at iteration 1 ("on 1 trial lam 1:"); an attempt without that line (converged before iteration 1, or
stopped at step 0) has none and is not judged. Each attempt is accepted (converged), refused at the step limit (not
converged after the most steps any attempt in the log took, 20 for #603's and #608's runs), or stopped earlier (the
stall rule or a failed projection). The rule refuses an attempt at iteration 1 when theta1 > R. Printed: every attempt
with theta1, the outcome and its wall time (the sum of its trials' seconds), then for R the accepted attempts the rule
would refuse (each one a lost step), the refused-at-the-limit ones it would not catch, and the time it would save (the
caught attempts' time past iteration 1). Exit 0 when no accepted attempt lies above R.

--selftest: a planted log with two accepted attempts (theta1 1e3 and 6e4) and two refused at the limit (8e9 and 1e14),
read at R 1e7 (no accepted attempt above R, both refused ones caught), and the same log with the 6e4 moved to 6e8 read at
R 1e7 (one accepted attempt above R): the first must pass and the second fail.
"""
import re
import sys

TRIAL = re.compile(r"on (\d+) trial lam ([0-9.e+-]+): max\|F\| ([0-9.e+-]+|nan); simplified correction / \|dz\| = "
                   r"([0-9.e+-]+|inf|nan) .*?(accepted|refused).*?, ([0-9.]+) s")
END = re.compile(r"octo-double Newton: (\d+) steps, max\|F\| ([0-9.e+-]+|nan), (not converged|converged)")


def attempts(text):
    """the attempts of one log: dicts with theta1, steps, converged, seconds and seconds_after1"""
    out, cur = [], []
    for line in text.splitlines():
        m = TRIAL.search(line)
        if m:
            cur.append(dict(it=int(m.group(1)), lam=float(m.group(2)), theta=float(m.group(4)), s=float(m.group(6))))
            continue
        e = END.search(line)
        if e:
            th1 = [t["theta"] for t in cur if t["it"] == 1 and t["lam"] == 1.0]
            out.append(dict(theta1=th1[0] if th1 else None, steps=int(e.group(1)), converged=e.group(3) == "converged",
                            seconds=sum(t["s"] for t in cur), seconds_after1=sum(t["s"] for t in cur if t["it"] >= 1)))
            cur = []
    return out


def judge(atts, R, limit):
    acc = [a for a in atts if a["theta1"] is not None and a["converged"]]
    lim = [a for a in atts if a["theta1"] is not None and not a["converged"] and a["steps"] >= limit]
    lost = [a for a in acc if a["theta1"] > R]
    missed = [a for a in lim if a["theta1"] <= R]
    saved = sum(a["seconds_after1"] for a in lim if a["theta1"] > R)
    return acc, lim, lost, missed, saved


def main(paths, R):
    atts = []
    for p in paths:
        a = attempts(open(p, encoding="utf-8", errors="replace").read())
        for x in a:
            x["log"] = p.rsplit("/", 1)[-1]
        atts += a
    limit = max((a["steps"] for a in atts if not a["converged"]), default=20)
    print(f"{len(atts)} attempts in {len(paths)} logs; the step limit read as {limit}")
    print("| log | theta1 | outcome | steps | seconds |\n|---|---|---|---|---|")
    for a in atts:
        if a["theta1"] is None:
            continue
        oc = "accepted" if a["converged"] else ("refused at the limit" if a["steps"] >= limit else "stopped earlier")
        print(f"| {a['log']} | {a['theta1']:.3e} | {oc} | {a['steps']} | {a['seconds']:.0f} |")
    acc, lim, lost, missed, saved = judge(atts, R, limit)
    hi_acc = max((a["theta1"] for a in acc), default=float("nan"))
    lo_lim = min((a["theta1"] for a in lim), default=float("nan"))
    print(f"\naccepted: {len(acc)}, largest theta1 {hi_acc:.3e}; refused at the limit: {len(lim)}, smallest theta1 {lo_lim:.3e}")
    print(f"R {R:.0e}: accepted attempts refused by the rule {len(lost)}; refused ones not caught {len(missed)}; "
          f"time saved {saved:.0f} s ({saved / 3600:.1f} h)")
    print("PASS" if not lost else "FAIL")
    return 0 if not lost else 1


def selftest():
    def log(ths):
        lines = []
        for th, conv in ths:
            lines += ["    on 0 trial lam 1: max|F| 1.0e-30; simplified correction / |dz| = 1.0e-01 (bound 0.750); accepted, 100.0 s",
                      f"    on 1 trial lam 1: max|F| 1.0e-40; simplified correction / |dz| = {th:.3e} (bound 0.750); refused, 1.0 s",
                      "    on 1 trial lam 0.5: max|F| 1.0e-41; simplified correction / |dz| = 5.000e-01 (bound 0.875); accepted, 150.0 s",
                      f"      octo-double Newton: {5 if conv else 20} steps, max|F| 1.0e-60, {'converged' if conv else 'not converged'}; x"]
        return "\n".join(lines)
    a = attempts(log([(1e3, True), (6e4, True), (8e9, False), (1e14, False)]))
    b = attempts(log([(1e3, True), (6e8, True), (8e9, False), (1e14, False)]))
    ja, jb = judge(a, 1e7, 20), judge(b, 1e7, 20)
    ok_a = len(a) == 4 and not ja[2] and not ja[3] and ja[4] == 302.0
    ok_b = len(jb[2]) == 1
    print(f"  planted log at R 1e7: lost {len(ja[2])}, missed {len(ja[3])}, saved {ja[4]:.0f} s -> {'ok' if ok_a else 'WRONG'}")
    print(f"  the 6e4 moved to 6e8: lost {len(jb[2])} -> {'ok (fails as it must)' if ok_b else 'WRONG'}")
    print("selftest", "passed" if ok_a and ok_b else "FAILED")
    return 0 if ok_a and ok_b else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    R = float(sys.argv[sys.argv.index("--R") + 1]) if "--R" in sys.argv else 1e7
    paths = [a for a in sys.argv[1:] if not a.startswith("--") and (sys.argv.index(a) == 1 or sys.argv[sys.argv.index(a) - 1] != "--R")]
    if not paths:
        sys.exit(__doc__)
    sys.exit(main(paths, R))
