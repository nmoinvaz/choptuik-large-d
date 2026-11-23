"""all_values.csv: one row per value the project has computed, from the canonical files (a derived index, not a source).

    python3 make_all_values.py [--out=all_values.csv]

Each row: quantity, model, D, params, value, lo, hi, err, err_kind, status, method, source. The canonical place of
each value stays its own file (AGENTS.md, one place per fact); this index gathers them so they can be read together.
Sources:
- hp/data/delta_all.csv: Delta of the l = 0 scalar field, one best row per D (value = the sum of its parts);
- hp/data/delta_dd_lowD.csv: gamma at low D, per M (rows with a gamma);
- hp/data/gamma_published_compare.csv: our gamma (and other quantities) beside published ones (the 'ours' column);
- matter/published_compare.csv: other matter models beside published values (the 'ours' column);
- matter/ell1_best.csv: the l = 1 multiplet's lambda_1 and Delta per D;
- named_values.csv: zeros of the inner selection problem (Theorem A's certified enclosures; the excited M = 3 zeros).
Status words are AGENTS.md's: certified (computer-assisted, the trusted code named in source), numerical evidence."""
import csv
import math
import os
import sys

from mpmath import mp, mpf

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["quantity", "model", "D", "params", "value", "lo", "hi", "err", "err_kind", "status", "method", "source"]


def rd(p):
    return list(csv.DictReader(open(os.path.join(HERE, p))))


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def delta_l0():
    mp.dps = 80
    for r in rd("hp/data/delta_all.csv"):
        parts = [r.get(f"Delta_q{i}", "") for i in range(4)]
        s = sum((mpf(p) for p in parts if p not in ("", None)), mpf(0))
        err = num(r.get("err"))
        dig = 17 if not err else max(17, min(70, int(-math.log10(err)) + 4))
        yield dict(quantity="Delta", model="l0 scalar field", D=r["D"], params="", value=mp.nstr(s, dig),
                   err=r.get("err", ""), err_kind=r.get("err_kind", ""), status="numerical evidence",
                   method=r.get("kind", ""), source=f"hp/data/delta_all.csv ({r.get('source', '')})")


def gamma_lowd():
    for r in rd("hp/data/delta_dd_lowD.csv"):
        if num(r.get("gamma")) is None:
            continue
        yield dict(quantity="gamma", model="l0 scalar field", D=r["D"], params=f"M={r['M']}", value=r["gamma"],
                   status="numerical evidence", method=f"lowD dd ({r.get('status', '')})",
                   source=f"hp/data/delta_dd_lowD.csv ({r.get('source', '')})")


def gamma_published():
    for r in rd("hp/data/gamma_published_compare.csv"):
        if num(r.get("ours")) is None:
            continue
        yield dict(quantity=r.get("quantity", "gamma"), model="l0 scalar field", D=r["D"],
                   params=f"beside {r.get('published_ref', '')}", value=r["ours"], status="numerical evidence",
                   method="ours beside a published value", source=f"hp/data/gamma_published_compare.csv ({r.get('ours_source', '')})")


def matter_published():
    for r in rd("matter/published_compare.csv"):
        if num(r.get("ours")) is None:
            continue
        yield dict(quantity=r["quantity"], model=r["model"], D=r["D"], params=f"kappa={r.get('kappa', '')}",
                   value=r["ours"], status="numerical evidence", method=f"beside {r.get('published_ref', '')}",
                   source=f"matter/published_compare.csv ({r.get('ours_source', '')})")


def ell1():
    for r in rd("matter/ell1_best.csv"):
        common = dict(model="l = 1 multiplet", D=r["D"], status="numerical evidence",
                      source=f"matter/ell1_best.csv ({r.get('source', '')})")
        yield dict(quantity="lambda_1", params=f"k_c={r['kc']}, {r['build']}", value=r["lambda1"], err=r["err"],
                   err_kind=r["err_kind"], method="hybrid det sign scan" if num(r["D"]) and float(r["D"]) >= 18.9 else "brentq on det M",
                   **common)
        if num(r.get("Delta")) is not None:
            yield dict(quantity="Delta", params=f"k_c={r['kc']}, {r['build']}", value=r["Delta"],
                       method="ell_refine record", **common)


def named():
    for r in rd("named_values.csv"):
        lo, hi = r["lo"], r["hi"]
        mid = lo if lo == hi else repr((float(lo) + float(hi)) / 2)
        yield dict(quantity=r["quantity"], model=r["model"], D=r["D"], params=r["params"], value=mid, lo=lo, hi=hi,
                   err="" if lo == hi else repr((float(hi) - float(lo)) / 2),
                   err_kind="" if lo == hi else "half-width of the certified box",
                   status=r["status"], method="", source=f"named_values.csv ({r['source']})")


def main():
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), os.path.join(HERE, "all_values.csv"))
    rows = []
    n = {}
    for name, gen in (("Delta l0", delta_l0), ("gamma lowD", gamma_lowd), ("gamma published", gamma_published),
                      ("matter", matter_published), ("l = 1", ell1), ("zeros", named)):
        k = 0
        for r in gen():
            rows.append({c: r.get(c, "") for c in COLS})
            k += 1
        n[name] = k
    bad = [r for r in rows if num(r["value"]) is None]          # every value a number, or the index is not written
    if bad:
        print(f"{len(bad)} rows without a numeric value, e.g. {bad[0]}: not written")
        return 1
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows: " + ", ".join(f"{k} {v}" for k, v in n.items()) + f" -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
