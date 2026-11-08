"""The acceptance of issue #105 from rt_run.py --solver mp records: Delta_RT - Delta_hp at the check D values on two
grids, and the two grids against each other.

    python3 solver/rt/rt_grids.py A.jsonl B.jsonl [--D 15,20.043653,29.984746] [--tol 1e-10] [--mutate=shift]
        [--table]

For each D the last converged record of each file is used (labels start, cont, end). Exits 1 when a D is missing from
either file, when |Delta_RT - Delta_hp| >= tol on either grid, or when |Delta_A - Delta_B| >= tol. Delta differences
are taken from the records' 30-digit Delta_str in arb, so they are not limited by f64. --table prints every D both
files have. --mutate=shift adds 2 tol to grid B's Delta at the first D; the check must then exit 1.
"""
import sys
import json
import argparse
import flint

flint.ctx.prec = 192
arb = flint.arb


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("solver") == "mp" and r.get("ok"):
                out[round(r["D"], 9)] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--D", default="15,20.043653,29.984746")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--mutate", default=None, choices=["shift"])
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    Ds = [round(float(v), 9) for v in a.D.split(",")]
    ga = next(iter(A.values()))["grid"] if A else None
    gb = next(iter(B.values()))["grid"] if B else None

    def row(D, shift=0.0):
        ra, rb = A[D], B[D]
        da, db = arb(ra["Delta_str"]), arb(rb["Delta_str"]) + shift
        hp = ra["Delta_hp"]
        ea = rb_e = None
        if hp is not None:
            ea = ra["diff"]
            rb_e = rb["diff"] + shift
        return da, db, ea, rb_e, float((da - db).mid())

    failed = False
    print(f"grids A {ga} ({a.a}), B {gb} ({a.b}), tol {a.tol:g}, mutate {a.mutate}")
    print(f"| D | Delta_RT (A) | A - hp | B - hp | A - B | verdict |")
    print("|---|---|---|---|---|---|")
    for i, D in enumerate(Ds):
        if D not in A or D not in B:
            print(f"| {D} | missing in {'A' if D not in A else 'B'} | | | | FAIL |")
            failed = True
            continue
        da, db, ea, eb, dab = row(D, 2 * a.tol if (a.mutate == "shift" and i == 0) else 0.0)
        bad = ea is None or eb is None or abs(ea) >= a.tol or abs(eb) >= a.tol or abs(dab) >= a.tol
        failed |= bad
        fmt = lambda v: "n/a" if v is None else f"{v:+.2e}"
        print(f"| {D} | {da.mid().str(16, radius=False)} | {fmt(ea)} | {fmt(eb)} | {dab:+.2e} | "
              f"{'FAIL' if bad else 'ok'} |")
    if a.table:
        print("\nevery D in both files:")
        print("| D | A - hp | B - hp | A - B | Newton steps A, B | products A | products B |")
        print("|---|---|---|---|---|---|---|")
        for D in sorted(set(A) & set(B)):
            da, db, ea, eb, dab = row(D)
            fmt = lambda v: "n/a" if v is None else f"{v:+.2e}"
            print(f"| {D} | {fmt(ea)} | {fmt(eb)} | {dab:+.2e} | {len(A[D]['products'])}, {len(B[D]['products'])} | "
                  f"{A[D]['products']} | {B[D]['products']} |")
    print(f"exit {1 if failed else 0}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
