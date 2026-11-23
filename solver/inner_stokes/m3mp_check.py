"""#1076: argcount's --m3mp on two cached points without value (one guard-blocked, one not tuned below 1e-8)."""
import json, math, os, shutil, sys
sys.path.insert(0, "/Users/nathan/Source/choptuik-inner-mp/solver/inner_stokes")
import argcount as ac


def main():
    T = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(T, "cache_pass2.jsonl")
    P = [(3.3185956, 0.2250978975), (3.2248456, 1.08154321)]
    ok = True
    c0 = os.path.join(T, "c_off.jsonl"); shutil.copy(src, c0)
    F0 = ac.Multiplier(3, c0, 1, r1=0.17, m3fix=True)
    F0.raw(P)
    off = [F0.vals[F0.key(p)] for p in P]
    n0 = sum(1 for _ in open(c0))
    ok &= all(not math.isfinite(abs(v)) for v in off) and n0 == sum(1 for _ in open(src))
    print(f"without --m3mp: values {off}, cache lines unchanged: {n0 == sum(1 for _ in open(src))}")
    c1 = os.path.join(T, "c_on.jsonl"); shutil.copy(src, c1)
    F1 = ac.Multiplier(3, c1, 2, r1=0.17, m3fix=True, m3mp=True)
    print("wanted:", [F1.mp_wanted(F1.key(p)) for p in P])
    F1.raw(P)
    on = [F1.vals[F1.key(p)] for p in P]
    print(f"with --m3mp: values {on}")
    F2 = ac.Multiplier(3, c1, 1, r1=0.17, m3fix=True, m3mp=True)
    again = [F2.mp_wanted(F2.key(p)) for p in P]
    same = all((math.isfinite(abs(F2.vals[F2.key(p)])) == math.isfinite(abs(v))) for p, v in zip(P, on))
    print(f"reloaded: wanted again {again}, the same values valued: {same}")
    ok &= any(math.isfinite(abs(v)) for v in on) and not any(again) and same
    print("m3mp test:", "PASS" if ok else "FAIL")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
