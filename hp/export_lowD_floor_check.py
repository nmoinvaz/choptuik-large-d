"""Check of export_lowD.floor_ok, the rule that admits low-D records accepted at the residual floor (#945).

    python3 export_lowD_floor_check.py                  every case as expected: exit 0
    python3 export_lowD_floor_check.py --mutate=nodelta the rule without its dDelta_est test: a case fails, exit 1

Cases are #945's D 3.04315 records (raw I945-win2/leg/leg_points.jsonl at e32ba4a4) and planted variants of them."""
import sys

import export_lowD as E

CASES = [  # name, F, dDelta_est, expected
    ("#945 M 320 at the floor", 7.348594570004412e-21, -1.01e-17, True),
    ("#945 M 384 at the floor", 9.614115883598831e-25, -2.9e-22, True),
    ("#945 M 384 before the floor (F 1.2e-16)", 1.1741493007772565e-16, None, False),
    ("planted: F at the floor, dDelta_est 1e-12", 7.3e-21, 1e-12, False),
    ("planted: F 1e-19, above the acceptance level", 1e-19, 1e-18, False),
    ("planted: no dDelta_est recorded", 7.3e-21, None, False),
]


def main():
    if "--mutate=nodelta" in sys.argv[1:]:
        E.floor_ok = lambda r: r.get("F") is not None and r["F"] <= E.FLOOR_F
    ok = True
    for name, F, e, want in CASES:
        got = E.floor_ok(dict(F=F, dDelta_est=e))
        ok &= got == want
        print(f"{name}: {got} (want {want}) {'ok' if got == want else 'WRONG'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
