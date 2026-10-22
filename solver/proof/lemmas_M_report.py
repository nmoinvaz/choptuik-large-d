#!/usr/bin/env python3
"""Markdown tables for proof/A_lemmas_M.md from the saved runs in lemmas_M_runs/ and lemma6b1_runs_M<M>/ (no
computation). Every number the notes quote in their result tables is printed here from a file.

    python3 lemmas_M_report.py [fp | data | sector | jump | cover | mut | all]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "lemmas_M_runs")


def load(name):
    p = os.path.join(RUNS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def e(x, d=1):
    if x is None:
        return "—"
    s = f"{x:.{d}e}"
    m, ex = s.split("e")
    ex = int(ex)
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m}×10{str(ex).translate(sup)}"


def fp():
    rows = []
    for M in (3, 5):
        t = load(f"fixed_point_M_table_M{M}.json")
        if t:
            rows += t
    print("| M | kind | configuration | start ray, r₀ | splitting | parameters | N | contraction | sup\\|η\\| at \\|z\\| = R | holds |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        par = "point" if not (r.get("box") or r.get("cbox")) else (f"real ±{r['box']}" if r.get("box") else
                                                                    f"complex ±{r['cbox']}(1+i)")
        ray = f"{r['arg']}{' ± ' + str(r['sector']) if r.get('sector') else ''}, {r['r0']}"
        if "N" in r:
            print(f"| {r['M']} | {r['kind']} | {r['tag']} | {ray} | {'diag' if r['diag'] else 'plain'} | {par} | {r['N']} | "
                  f"{r['lip']:.4f} | {e(r['eta']) if r['ok'] else '—'} | {'yes' if r['ok'] else '**no**'} |")
        else:
            print(f"| {r['M']} | {r['kind']} | {r['tag']} | {ray} | {'diag' if r['diag'] else 'plain'} | {par} | — | — | — | "
                  f"**no** ({r['err']}) |")


def data():
    print("| M | start ray, r₀ | real box | value radius | derivative radius | Cauchy part (p, v, c) | sup\\|η\\| over X |")
    print("|---|---|---|---|---|---|---|")
    for M in (3, 5):
        t = load(f"fixed_point_M_data_M{M}.json")
        for r in t or []:
            print(f"| {r['M']} | {r['arg']}, {r['r0']} | ±{r['h']} | {e(r['val_rad'])} | {e(r['der_rad'])} | "
                  f"{e(r['deta'])} | {e(r['eta'])} |")


def sector():
    print("| M | R₊ (r₊²) | k | parameters | lemma S contraction | old sector lemma (diag) | Z | flat estimate Λ plain / diag | all |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in sorted(os.listdir(RUNS)):
        if name.startswith("lemma6_sector_M") and name.endswith(".json"):
            d = load(name)
            print(f"| {d['M']} | {d['rp']}² | {d['k']} | x̂ ± {d['box']} | {float(d['lip_S'].split()[0].strip('[')):.4f} | "
                  f"{float(d['lip_old'].split()[0].strip('[')):.4f} | {d['Z']} | "
                  f"{float(d['Lambda_plain'].split()[0].strip('[')):.4f} / {float(d['Lambda_diag'].split()[0].strip('[')):.4f} | "
                  f"{'yes' if d['ok'] else 'no'} |")
    for M in (3, 5):
        s = load(f"lemmaS_scan_M{M}.json")
        if s:
            print(f"\nlemma S scan, M = {M} (point x̂): " + "; ".join(
                f"r₊ {r['rp']} k {r['k']}: {r['lip']:.3f}" for r in s if "lip" in r))


def jump():
    for name in sorted(os.listdir(RUNS)):
        if name.startswith("lemma6a_") and name.endswith(".json") and "_B" not in name:
            d = load(name)
            print(f"{name}: M {d['M']}, xi1 {d['xi1']}, xi_a {d['xia']}, Lambda {d['Lambda']}, rho_6a {d['rho6a']}, ok {d['ok']}, "
                  f"steps {d['steps']}, {d['sec']:.0f} s, |H| <= {d['H_max']:.2e}, |G_slow^-1| <= {d['Gslow_inv_max']:.2e}, "
                  f"pilot {d.get('pilot')}, mutate {d.get('mutate')}")


def cover():
    for M in (3, 5):
        d = os.path.join(HERE, f"lemma6b1_runs_M{M}")
        if not os.path.isdir(d):
            continue
        ray = json.load(open(os.path.join(d, "ray.json"))) if os.path.exists(os.path.join(d, "ray.json")) else None
        if ray:
            print(f"M = {M}: ray run: start box radius {e(ray['start_rad'])}, {sum(l['steps'] for l in ray['legs'])} steps, "
                  f"{ray['sec']:.0f} s; " + ", ".join(f"|u| {l['to']}: {e(l['rad'])}" for l in ray["legs"]))
        print("| row | ρ_j | r, n | discs | certified | Lohner steps | max Λ | max box radius | max sup\\|y − y(c)\\| | time |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for f in sorted(os.listdir(d)):
            if f.startswith("row_") and f.endswith(".json"):
                r = json.load(open(os.path.join(d, f)))
                ds = r["discs"]
                sups = [x["sup"] for x in ds if x["sup"] is not None]
                print(f"| {r['j']} | {r['rho']} | {r['r']}, {r['n']} | {len(ds)} | {sum(x['ok'] for x in ds)} | {r['steps']} | "
                      f"{max(x['Lam'] for x in ds):.3f} | {e(max(x['box_rad'] for x in ds))} | "
                      f"{(max(sups) if sups else float('nan')):.3g} | {r['sec']:.0f} s |")


def mut():
    t = load("mutations_M.json")
    print("| name | what is broken | expected | exit | as expected |")
    print("|---|---|---|---|---|")
    for r in t or []:
        print(f"| `{r['name']}` | {r['what']} | {r['expected']} | {r['exit']} | {'yes' if r['matches'] else '**no**'} |")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, f in (("fp", fp), ("data", data), ("sector", sector), ("jump", jump), ("cover", cover), ("mut", mut)):
        if what in (name, "all"):
            print(f"\n### {name}\n")
            f()
