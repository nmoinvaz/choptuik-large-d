#!/usr/bin/env python3
"""The A/B of anm_run.py --od_broyden 1 (newton_od's Broyden updates) against the same run without it: from the two
logs, each octo-double corrector run's corrections, factorizations (LUs) and seconds, and from the two saved states
Delta's agreement, all eight parts summed exactly.

    python3 od_broyden_ab.py LOG_A LOG_B STATE_A STATE_B [--kind chain|arc] [--dmax 1e-30] [--mutate=dsame]

A is the run without the updates, B the run with them, from the same stored state and step. Pass (exit 0), the criteria
fixed before the runs: (1) the last corrector run of each log converged; (2) the states are at the same D (D_str) and
|Delta_B - Delta_A| <= dmax; (3) B's last corrector run formed no more factorizations than A's, counting with --kind
chain (the continuation in D) the one B defers: a run that ends at tol without factors there (deferred) leaves the LU
of the next base's series to anm_run's base_lu_o, which keeps A's. The gain is A's seconds against B's in the last run
(the attempt), B's with a deferred LU charged at A's seconds per factorization (lu_seconds); where A's log gives
no such seconds, B / A is printed unadjusted and says so, since it then understates B's time; the error is
|Delta_B - Delta_A|. With --kind arc nothing is charged: arclength.py's series forms its own LU at every base.
--mutate=dsame compares STATE_A with itself and moves B's Delta by 2 dmax, so it must fail (2).

Run both arms with --od_times 1 to see each run's seconds split into factorizations, the trials' projections, their
residuals and solves, and the updates (times_of). A's factorizations are counted from its lines (count_lus): one at each run's start, one at each accepted natural trial
(newton_od forms the factors there) and at each accepted strong chord, and one more where the chord's factors were not
finite and the start's were formed again. B's are its "od Broyden: L factorizations" lines. od_broyden_check.py case F
checks both counts against a toy's newton_o calls. A log of the own test (--od_accept own) is refused."""
import argparse
import json
import re
import sys

import numpy as np

# od_broyden_check.py --mutate=count: the starts' factorizations left out; --mutate=border: border lines not skipped
MUT = {"nostart": False, "border": False}

RE_END = re.compile(r"^\s+octo-double Newton: \d+ steps, max\|F\| \S+, (converged|not converged)|^base polished at .*?, "
                    r"(converged|NOT converged)|^\s+corrector: \d+ steps, .*?, (converged|NOT converged)")
RE_ACC_TRIAL = re.compile(r"^\s+on (\d+) trial lam \S+: max\|F\| .*; simplified correction / \|dz\| = .*; accepted")
RE_OWN_TRIAL = re.compile(r"^\s+on \d+ trial lam \S+: max\|F\| .*; raw \|dz_t\|")
RE_CHORD = re.compile(r"^\s+on 0: strong chord .*: (accepted|refused), ")
RE_STEP = re.compile(r"^\s+on (\d+): \|F\| \S+ -> \S+, \|dz\| ")
RE_BROYDEN = re.compile(r"^\s+od Broyden: (\d+) factorizations, (\d+) update steps, (\d+) refused")
RE_TIMES = re.compile(r"^\s+od times: (\d+) factorizations ([0-9.]+) s, (\d+) trial projections ([0-9.]+) s, (\d+) trial "
                      r"residuals and solves ([0-9.]+) s, updates ([0-9.]+) s")
RE_NEW = re.compile(r"^(base |step \d+: )")            # anm_run's base line and a new base's series line
RE_BORDER = re.compile(r"^\s+border at D ")         # arclength.py's line per bordered factorization
RE_CONV = re.compile(r"^\s+on \d+: \|F\| \S+ converged")
RE_TRIAL_S = re.compile(r"^\s+on (\d+) trial lam 1: .*; accepted.*, ([0-9.]+) s$")
RE_STEP_S = re.compile(r"^\s+on (\d+): \|F\| \S+ -> .*, ([0-9.]+) s$")
RE_SECONDS = re.compile(r"(?:^\s+a \S+ -> D .*, (\d+) s$)|(?:^\s+corrector: .*?, (?:converged|NOT converged), ([0-9.]+) s)")


def count_lus(log, runs=None):
    """the factorizations a log of newton_od runs without the updates formed (natural test): the starts (runs, or
    the number of runs' end lines), the accepted natural trials, the accepted strong chords, and the chords accepted
    but not followed by their step line (their factors were not finite, and the start's were formed again). The step
    line is looked for past arclength's border lines, which BorderedCore.newton_o prints between the two."""
    lines = log.splitlines()
    if any(RE_OWN_TRIAL.match(x) for x in lines):
        raise SystemExit("od_broyden_ab: a log of the own test, whose trials at verbose 1 are not all printed")
    n = 0 if MUT["nostart"] else (runs if runs is not None else sum(1 for x in lines if RE_END.match(x)))
    chord_step = set()                  # the step lines that follow an accepted strong chord
    for i, x in enumerate(lines):
        if RE_ACC_TRIAL.match(x):
            n += 1
        m = RE_CHORD.match(x)
        if m and m.group(1) == "accepted":
            n += 1
            nxt = next((y for y in lines[i + 1:] if y.strip() and (MUT["border"] or not RE_BORDER.match(y))), "")
            if not RE_STEP.match(nxt):
                n += 1
            j = next((j for j in range(i + 1, len(lines)) if lines[j].strip() and not RE_BORDER.match(lines[j])), None)
            if j is not None and RE_STEP.match(lines[j]):
                chord_step.add(j)
    # an own-test run at verbose 1 prints no line for a trial accepted as usual, so a step that follows neither an
    # accepted natural trial of its own iteration nor an accepted chord marks such a log
    accepted = set()
    for i, x in enumerate(lines):
        m = RE_ACC_TRIAL.match(x)
        if m:
            accepted.add(m.group(1))
        st = RE_STEP.match(x)
        if st:
            if i not in chord_step and st.group(1) not in accepted:
                raise SystemExit(f"od_broyden_ab: step {st.group(1)} (line {i + 1}) follows no accepted natural trial "
                                 "line, as in a log of the own test")
            accepted.clear()
    return n


def deferred(log):
    """whether the last newton_od run of a log ended converged at a point where it formed no factors: its converged
    line follows a step line that says so ("no factors there") or an update step that formed none"""
    lines = [x for x in log.splitlines() if x.strip()]
    for i in range(len(lines) - 1, 0, -1):
        if RE_CONV.match(lines[i]):
            prev = lines[i - 1]
            return bool(RE_STEP.match(prev) and ("no factors there" in prev or
                                                 ("quasi-Newton" in prev and "factors formed again" not in prev)))
    return False


def lu_seconds(log):
    """the seconds of a factorization in a log without the updates, and where they come from: (seconds, source), or
    (None, "") when the log gives none. From its "od times" lines (--od_times 1) where it has them: their factorization
    seconds over their factorizations. Otherwise the median of a natural step's seconds less those of its one trial
    (accepted at lam 1, the step line right after it but for arclength's border lines), over the steps whose iteration
    ran no strong chord: such a step's seconds include the chord's, and its trial's do not"""
    ts = [t for t in (times_of(x) for x in split_times(log)) if t and t["n_lu"]]
    if ts:
        return sum(t["lu"] for t in ts) / sum(t["n_lu"] for t in ts), "A's od times, mean per factorization"
    lines = [x for x in log.splitlines() if not RE_BORDER.match(x)]
    out, chord = [], False
    for a, b in zip(lines, lines[1:]):
        if RE_CHORD.match(a):
            chord = True
        m, n = RE_TRIAL_S.match(a), RE_STEP_S.match(b)
        if m and n and m.group(1) == n.group(1) and not chord:
            out.append(float(n.group(2)) - float(m.group(2)))
        if RE_STEP_S.match(b) or RE_END.match(b):
            chord = False                       # the chord belongs to the iteration this line ends
    return (float(np.median(out)), "A's median natural step less its trial") if out else (None, "")


def split_times(log):
    """the "od times" lines of a log, one string each, for times_of"""
    return [x for x in log.splitlines() if RE_TIMES.match(x)]


def times_of(log):
    """the "od times" line of a newton_od run (anm_run.py --od_times 1) as a dict, or None"""
    for x in log.splitlines():
        m = RE_TIMES.match(x)
        if m:
            g = m.groups()
            return dict(n_lu=int(g[0]), lu=float(g[1]), n_proj=int(g[2]), proj=float(g[3]), n_rs=int(g[4]),
                        rs=float(g[5]), upd=float(g[6]))
    return None


def printed_counts(log):
    """the "od Broyden" lines of a log, one per newton_od run with the updates"""
    return [dict(lus=int(m.group(1)), quasi=int(m.group(2)), refused=int(m.group(3)))
            for m in (RE_BROYDEN.match(x) for x in log.splitlines()) if m]


def runs_of(log):
    """the log split into newton_od runs, each ending at its caller's end line; per run: the text, converged, the
    corrections (step lines) and the seconds of an attempt (anm_run's attempt line or arclength's corrector line)"""
    out, cur, pending = [], [], None
    for x in log.splitlines():
        cur.append(x)
        m = RE_END.match(x)
        if m:
            conv = next(g for g in m.groups() if g) == "converged"
            out.append(dict(text="\n".join(cur), converged=conv, steps=sum(1 for y in cur if RE_STEP.match(y))))
            cur = []
            pending = len(out) - 1
        if "attempt stopped" in x or RE_NEW.match(x):
            pending = None              # a failed start (no end line) or a new base: its seconds are not the run's
        s = RE_SECONDS.match(x)
        if s and pending is not None:
            out[pending]["seconds"] = float(next(g for g in s.groups() if g))
            pending = None
    return out


def delta_and_D(path):
    """(Delta as the exact sum of its eight parts, every entry so summed, D_str) of a saved state"""
    import mpmath as mp
    d = np.load(path, allow_pickle=False)
    keys = [k for k in ("zh", "zl", "z2", "z3", "z4", "z5", "z6", "z7") if k in d.files]
    meta = json.loads(str(d["meta"]))
    with mp.workprec(64 * 53):
        z = [mp.fsum(mp.mpf(float(d[k][i])) for k in keys) for i in range(len(d["zh"]))]
    return z[-1], z, meta.get("D_str")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_a")
    ap.add_argument("log_b")
    ap.add_argument("state_a")
    ap.add_argument("state_b")
    ap.add_argument("--kind", default="chain", choices=["chain", "arc"])
    ap.add_argument("--dmax", type=float, default=1e-30)
    ap.add_argument("--mutate", default="", choices=["", "dsame"])
    a = ap.parse_args()
    import mpmath as mp
    A, B = open(a.log_a).read(), open(a.log_b).read()
    ra, rb = runs_of(A), runs_of(B)
    pb = printed_counts(B)
    if len(pb) != len(rb):
        raise SystemExit(f"od_broyden_ab: {len(rb)} corrector runs in B's log, {len(pb)} od Broyden lines")
    print(f"{'run':>4} {'A steps':>8} {'A LUs':>6} {'A s':>8} {'B steps':>8} {'B LUs':>6} {'B upd':>6} {'B ref':>6} {'B s':>8}")
    for k in range(max(len(ra), len(rb))):
        x = ra[k] if k < len(ra) else None
        y = rb[k] if k < len(rb) else None
        la = count_lus(x["text"], runs=1) if x else None
        print(f"{k:>4} {x['steps'] if x else '':>8} {la if x else '':>6} {x.get('seconds', '') if x else '':>8} "
              f"{y['steps'] if y else '':>8} {pb[k]['lus'] if y else '':>6} {pb[k]['quasi'] if y else '':>6} "
              f"{pb[k]['refused'] if y else '':>6} {y.get('seconds', '') if y else '':>8}")
    if any(times_of(r["text"]) for r in ra + rb):
        print("seconds by part (--od_times 1): factorizations, trial projections, trial residuals and solves, updates")
        for name, rr in (("A", ra), ("B", rb)):
            for k, r in enumerate(rr):
                x = times_of(r["text"])
                if x:
                    print(f"  {name} run {k}: {x['n_lu']} LUs {x['lu']:.0f} s, {x['n_proj']} projections {x['proj']:.0f} s, "
                          f"{x['n_rs']} residuals and solves {x['rs']:.0f} s, updates {x['upd']:.0f} s")
    if not ra or not rb:
        print(f"FAIL: no completed corrector run in {'A' if not ra else 'B'}'s log ({a.log_a if not ra else a.log_b})")
        return 1
    conv = bool(ra[-1]["converged"] and rb[-1]["converged"])
    lu_a, lu_b = count_lus(ra[-1]["text"], runs=1), pb[-1]["lus"]
    dfr = int(a.kind == "chain" and deferred(rb[-1]["text"]))
    t_lu, t_src = lu_seconds(A)
    sa, sb = ra[-1].get("seconds"), rb[-1].get("seconds")
    sb_eff = None if sb is None else sb + (dfr * t_lu if dfr and t_lu is not None else 0.0)
    Da, za, Dsa = delta_and_D(a.state_a)
    Db, zb, Dsb = delta_and_D(a.state_a if a.mutate else a.state_b)
    with mp.workprec(64 * 53):
        if a.mutate == "dsame":
            Db += 2 * mp.mpf(a.dmax)
        dD = abs(Db - Da)
        dz = max(abs(p - q) for p, q in zip(za, zb))
    sameD = Dsa == Dsb
    ok = conv and sameD and dD <= a.dmax and lu_b + dfr <= lu_a
    print(f"(1) last runs converged: A {ra[-1]['converged'] if ra else None}, B {rb[-1]['converged'] if rb else None}")
    print(f"(2) D_str A {Dsa}, B {Dsb} ({'same' if sameD else 'DIFFERENT'}); |Delta_B - Delta_A| {mp.nstr(dD, 5)} (bound "
          f"{a.dmax:.0e}); max entry difference {mp.nstr(dz, 5)}")
    print(f"(3) factorizations in the last run: A {lu_a}, B {lu_b}" + (" + 1 deferred to the next series" if dfr else ""))
    gain = f"gain: corrector seconds A {sa}, B {sb}"
    if dfr and t_lu is not None:
        gain += f" + {t_lu:.0f} for the deferred LU ({t_src}) = {sb_eff:.0f}"
    if sa and sb_eff is not None:
        gain += f"; B / A {sb_eff / sa:.3f}"
        if dfr and t_lu is None:
            # A's log gives no factorization's seconds (lu_seconds), so the LU that B defers is not charged, and the
            # ratio understates B's time
            gain += (" unadjusted: B defers an LU whose seconds A's log does not give (no od times, and no natural "
                     "step without a chord accepted at lam 1), so the ratio understates B's time")
    print(gain)
    print(("PASS" if ok else "FAIL") + (f" (--mutate={a.mutate})" if a.mutate else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
