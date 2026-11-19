#!/usr/bin/env python3
"""The tables of notes/QUADRATIC_FORMULA.md from solver/quadform/data/*.json (the outputs of algfit.py and formula.py).

    cd solver/quadform && python3 summarize.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = ["real", "T_cut", "T_pow", "T_pole", "T_pole7"]


def load(name):
    p = os.path.join(HERE, "data", name)
    return json.load(open(p)) if os.path.exists(p) else None


def table_A():
    print("A. rational against quadratic at equal constants, fitted on 4 <= D <= 30.2; score = largest |miss| on "
          "30.2 < D <= 245.64\n")
    print("| points | weights | N | rational | score | quadratic | score | ratio | pass A |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in SETS + ["real_before_amendment", "real_mutate_root", "real_mutate_shift"]:
        d = load(f"A_{s}.json")
        if not d:
            continue
        for wt in ("sigma", "uniform"):
            for N in (9, 11, 13):
                v = d["verdicts"][f"{wt}:N{N}"]
                last = ("yes" if d["pass_A"] else "no") if (wt, N) == ("uniform", 13) else ""
                print(f"| {s} | {wt} | {N} | {v['rational']} | {v['rational_score']:.1e} | {v['quadratic']} | "
                      f"{v['quadratic_score']:.1e} | {v['ratio']:.3g} | {last} |")
    print()


def table_B():
    print("B. quadratic forms on all points 3.04 <= D <= 245.64 (18 forms, N = 11 to 25)\n")
    print("| points | accepted | with branch points at 3 and -7 (frozen rule) | upper branch point, accepted | "
          "post hoc: forms with a real branch point in 2.9 < D < 3.04 | its range | Delta there, range | "
          "Delta_inf, accepted |")
    print("|---|---|---|---|---|---|---|---|")
    for s in SETS:
        d = load(f"B_{s}.json")
        if not d:
            continue
        acc = [r for r in d["rows"] if r["accepted"]]
        up = [r for r in d["rows"] if r["bp_upper"] is not None and 2.9 < r["bp_upper"] < 3.04]
        rng = f"{min(r['bp_upper'] for r in up):.4f} to {max(r['bp_upper'] for r in up):.4f}" if up else ""
        dl = [r["delta_at_upper"] for r in up if r.get("delta_at_upper") is not None]
        drng = f"{min(dl):.3f} to {max(dl):.3f}" if dl else ""
        print(f"| {s} | {len(acc)} of {len(d['rows'])} | {d['hit_cut']} | "
              f"{', '.join('%.4f' % r['bp_upper'] for r in acc if r['bp_upper'] is not None)} | {len(up)} of {len(d['rows'])} | "
              f"{rng} | {drng} | {', '.join(r['dinf'][:14] for r in acc)} |")
    print()
    d = load("B_real.json")
    if d:
        print("B, real points, every form:\n")
        print("| N | form | chi2/dof | max miss/sigma | defect | upper branch point | Delta there | Delta_inf |")
        print("|---|---|---|---|---|---|---|---|")
        for r in d["rows"]:
            up = f"{r['bp_upper']:.5f}" if r["bp_upper"] is not None else ""
            du = f"{r['delta_at_upper']:.4f}" if r.get("delta_at_upper") is not None else ""
            print(f"| {r['N']} | {r['form']} | {r['chi2dof']:.3g} | {r['max_over_sigma']:.3g} | "
                  f"{'yes' if r['defect'] else 'no'} | {up} | {du} | {r['dinf'][:14]} |")
        print()


def table_C():
    print("C. all points 3.04 <= D <= 67.77: the smallest N within 3.47 sigma at every point, and (post hoc) the "
          "smallest N with chi2/dof <= 4 and no defect\n")
    print("| points | class | first N within 3.47 sigma | first N with chi2/dof <= 4 | its form, chi2/dof, max miss/sigma |")
    print("|---|---|---|---|---|")
    for s in SETS:
        d = load(f"C_{s}.json")
        if not d:
            continue
        for cls in ("quadratic", "rational"):
            f = d["first"].get(cls)
            good = [r for r in d["rows"] if r["cls"] == cls and r["ok"] and not r["defect"] and r["chi2dof"] <= 4]
            g = min(good, key=lambda r: (r["N"], r["chi2dof"])) if good else None
            nmax = max((r["N"] for r in d["rows"] if r["cls"] == cls), default=0)
            print(f"| {s} | {cls} | {f['N'] if f else 'none to N = %d' % nmax} | {g['N'] if g else 'none to N = %d' % nmax} | "
                  f"{(g['form'] + ', %.3g, %.3g' % (g['chi2dof'], g['max_over_sigma'])) if g else ''} |")
    print()


def table_F():
    for name in sorted(f for f in os.listdir(os.path.join(HERE, "data")) if f.startswith("formula_q")):
        F = load(name)
        print(f"{name}: degrees (p, q, r) = {tuple(F['degrees_pqr'])}, {F['n_const']} constants, fitted on D "
              f"{F['fitted_on'][0]:g} to {F['fitted_on'][1]:g} ({F['weights']} weights), Delta_inf {F['Delta_inf']}, "
              f"real branch points at D = {', '.join('%.4f' % v for v in F['real_branch_points_D'])}\n")
        keys = list(F["max_miss_by_range"])
        rngs = list(F["max_miss_f64"])
        print("| D | " + " | ".join(keys) + " | the same quadratic in float64 |")
        print("|---|" + "---|" * (len(keys) + 1))
        for r in rngs:
            print(f"| {r} | " + " | ".join(f"{F['max_miss_by_range'][k].get(r, float('nan')):.1e}" for k in keys) +
                  f" | {F['max_miss_f64'][r]:.1e} |")
        print()
        for k in ("P", "Q", "R"):
            print(f"  {k}: " + ", ".join(F[k]))
        print(f"  sign: {F['sign']}\n")


if __name__ == "__main__":
    table_A()
    table_B()
    table_C()
    table_F()
