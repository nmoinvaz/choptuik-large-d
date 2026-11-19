#!/usr/bin/env python3
"""lowD_dd.floor_stop on #945's M 384 pass-2 sequence (planted from its log): with chord phases the residual-floor
rule must stop after the fourth full step (3.10e-24 -> 1.97e-24), not after the fifth, and must not stop after the third
(1.98e-21 -> 3.10e-24); without chord records it keeps the starts-only rule.

    python3 floor_stop_check.py            exit 0 iff all four cases come out as stated
    python3 floor_stop_check.py --mutate   the rule's ratio test reversed: the fourth-step case must fail (exit 1)
"""
import sys

import lowD_dd as L


def main(mutate):
    step = lambda F: dict(F=F, lam=1.0)
    chord = lambda F: dict(chords=[], F=F)
    ratio = 2.0 if mutate else 0.5          # the mutation: a ratio above 1 never reads a fall as 'no longer falling'
    h = [step(1.17e-16), chord(2.05e-17), step(2.05e-17), chord(1.98e-21), step(1.98e-21), chord(3.10e-24)]
    cases = [("after step 3 with chords", L.floor_stop(h, 1e-20, ratio), False)]
    h4 = h + [step(3.10e-24), chord(1.97e-24)]
    cases.append(("after step 4 with chords", L.floor_stop(h4, 1e-20, ratio), True))
    s2 = [step(1.98e-21), step(3.10e-24)]
    cases.append(("starts only, two steps", L.floor_stop(s2, 1e-20, ratio), False))
    cases.append(("starts only, three steps", L.floor_stop(s2 + [step(1.97e-24)], 1e-20, ratio), True))
    ok = True
    for name, got, want in cases:
        good = bool(got) == want
        ok &= good
        print(f"{name}: stop {bool(got)} (want {want}) {'ok' if good else 'WRONG'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main("--mutate" in sys.argv))
