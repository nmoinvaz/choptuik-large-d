"""The acceptance of issue #105 from rt_run.py --solver mp records: Delta_RT - Delta_hp at the check D values on two
grids, and the two grids against each other.

    python3 solver/rt/rt_grids.py A.jsonl B.jsonl [--D 15,20.043653,29.984746] [--tol 1e-10] [--mutate=shift]
        [--table]

For each D the last converged record of each file is used (labels start, cont, end). Exits 1 when a D is missing from
either file, when |Delta_RT - Delta_hp| >= tol on either grid, or when |Delta_A - Delta_B| >= tol. Delta differences
are taken from the records' 30-digit Delta_str in arb, so they are not limited by f64. --table prints every D both
files have. --extra F1,F2,... prints, for the check D, the records of further files (other grids) against hp and
against B, outside the exit criteria. --mutate=shift adds 2 tol to grid B's Delta at the first D; the check must then
exit 1. --tol_ab (issue #163) sets a separate tolerance on |Delta_A - Delta_B| (default --tol).
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
    ap.add_argument("--tol_ab", type=float, default=None,
                    help="tolerance on |Delta_A - Delta_B| (default: --tol); issue #163's acceptance uses 1e-18")
    ap.add_argument("--mutate", default=None, choices=["shift"])
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--extra", default="")
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
    tol_ab = a.tol if a.tol_ab is None else a.tol_ab
    print(f"grids A {ga} ({a.a}), B {gb} ({a.b}), tol {a.tol:g}, tol_ab {tol_ab:g}, mutate {a.mutate}")
    print(f"| D | Delta_RT (A) | A - hp | B - hp | A - B | verdict |")
    print("|---|---|---|---|---|---|")
    for i, D in enumerate(Ds):
        if D not in A or D not in B:
            print(f"| {D} | missing in {'A' if D not in A else 'B'} | | | | FAIL |")
            failed = True
            continue
        da, db, ea, eb, dab = row(D, 2 * a.tol if (a.mutate == "shift" and i == 0) else 0.0)
        bad = ea is None or eb is None or abs(ea) >= a.tol or abs(eb) >= a.tol or abs(dab) >= tol_ab
        failed |= bad
        fmt = lambda v: "n/a" if v is None else f"{v:+.2e}"
        print(f"| {D} | {da.mid().str(16, radius=False)} | {fmt(ea)} | {fmt(eb)} | {dab:+.2e} | "
              f"{'FAIL' if bad else 'ok'} |")
    if a.extra:
        rows = [(A, a.a), (B, a.b)]
        for path in a.extra.split(","):
            rows.append((load(path), path))
        print("\nthe check D on every grid (extra files outside the exit criteria):")
        print("| D | grid | Delta_RT | - hp | - B | Newton steps | products | file |")
        print("|---|---|---|---|---|---|---|---|")
        for D in Ds:
            for R, path in rows:
                if D not in R:
                    continue
                r = R[D]
                d = arb(r["Delta_str"])
                vb = float((d - arb(B[D]["Delta_str"])).mid()) if D in B else None
                fmt = lambda v: "n/a" if v is None else f"{v:+.2e}"
                print(f"| {D} | {r['grid'][0]}x{r['grid'][1]} | {d.mid().str(20, radius=False)} | {fmt(r['diff'])} | "
                      f"{fmt(vb)} | {len(r['products'])} | {r['products']} | {path.rsplit('/', 1)[-1]} |")
    if a.table:
        print("\ntotals per file:")
        for R, path in ((A, a.a), (B, a.b)):
            rs = list(R.values())
            piv = [r["pivot_ratio"] for r in rs]
            print(f"  {path.rsplit('/', 1)[-1]}: {len(rs)} D, {sum(r['secs'] for r in rs) / 60:.0f} min, "
                  f"{sum(len(r['products']) for r in rs)} Newton steps, {sum(sum(r['products']) for r in rs)} products, "
                  f"{sum(len(r['refresh']) for r in rs)} rebuilds ({sum(r['t_refresh'] for r in rs) / 60:.0f} min), "
                  f"LU pivot ratio {min(piv):.1e} to {max(piv):.1e}, final max|F| at most "
                  f"{max(r['newton'][-1] for r in rs):.1e}")
        print("\nevery D in both files:")
        print("| D | A - hp | B - hp | A - B | Newton steps A, B | products A | products B | s per Newton step A, B |"
              " s per D A, B |")
        print("|---|---|---|---|---|---|---|---|---|")
        for D in sorted(set(A) & set(B)):
            da, db, ea, eb, dab = row(D)
            fmt = lambda v: "n/a" if v is None else f"{v:+.2e}"
            sp = lambda r: r["t_newton"] / max(1, len(r["products"]))
            print(f"| {D} | {fmt(ea)} | {fmt(eb)} | {dab:+.2e} | {len(A[D]['products'])}, {len(B[D]['products'])} | "
                  f"{A[D]['products']} | {B[D]['products']} | {sp(A[D]):.1f}, {sp(B[D]):.1f} | "
                  f"{A[D]['secs']:.0f}, {B[D]['secs']:.0f} |")
    print(f"exit {1 if failed else 0}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
