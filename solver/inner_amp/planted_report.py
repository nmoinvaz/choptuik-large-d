#!/usr/bin/env python3
"""Grades planted_results.jsonl against the frozen criteria of notes/PLANTED_SELECTION.md (section 2) and prints the
markdown tables of section 3. Every count is printed with its denominator; a planned plant that has no record is
listed as missing.

    python3 planted_report.py
"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import planted_run as R  # noqa: E402
import planted_plants as PP  # noqa: E402

EST = ("arg", "ratio", "pade", "G", "aitken")
LABEL = {"arg": "arg τ = π (above)", "ratio": "ratio (below)", "pade": "Padé central mid", "G": "factored G",
         "aitken": "Aitken (quad.)"}


def planned():
    names = [f"N0 {s}" for s in ("scan25phys", "scan23phys")] + [f"N1 lone zero wt={w}" for w in PP.OMEGA_T] + \
        ["N2 S=1", "N2 S=exp(bx)"]
    for tag in ("P-ARC", "P-ARC-B chord", "P-ARC-B mirror", "P-ARC-B beta2",
                "P-Z2 x2=9.0-1.0j", "P-Z2 x2=8.6+1.2j", "P-Z2 x2=9.3+0j"):
        names += [f"{tag} wt={w}" for w in PP.OMEGA_T]
    names += ["C-W1", "C-W2", "C-W3", "C-EXP"]
    for wt in ("1.0000", "0.9985"):
        names += [f"P-NOISE sigma_err wt={wt} seed={s}" for s in range(10)]
        names += [f"P-NOISE rel={e} wt={wt} seed={s}" for e in ("1e-12", "1e-9", "1e-6", "1e-4") for s in range(5)]
    for base in ("CORR P-ARC wt=1.0000", "CORR C-W1"):
        names += [f"{base} clean twin"] + [f"{base} X1 flip sigma_{k}" for k in (12, 9, 5)] + \
            [f"{base} X2 scale sigma_{k}" for k in (12, 9, 5, 2)] + [f"{base} X2 scale sigma_12 all nodes"] + \
            [f"{base} {t}" for t in ("X3a all +0.0005", "X3b 1.01->1.0105", "X3c 1.00->1.0005")]
    return names


def f4(v, scale=1e4, d=2):
    return "—" if v is None else f"{v * scale:+.{d}f}"


def group(by, fam_of):
    out = {}
    for r in by.values():
        f = fam_of(r)
        if f:
            out.setdefault(f, []).append(r)
    return out


def fam_label(r):
    n = r["name"]
    for tag in ("P-ARC-B chord", "P-ARC-B mirror", "P-ARC-B beta2", "P-Z2 x2=9.0-1.0j", "P-Z2 x2=8.6+1.2j",
                "P-Z2 x2=9.3+0j"):
        if n.startswith(tag + " "):
            return tag
    if n.startswith("P-ARC wt"):
        return "P-ARC"
    if r["family"] == "C":
        return "C (completions)"
    if r["family"] in ("REFIT",) or r["family"].startswith("P-NOISE"):
        return r["family"]
    return None


def main():
    by = {r["name"]: r for r in R.records()}
    plan = planned()
    missing = [n for n in plan if n not in by]
    extra = [n for n in by if n not in plan and not n.startswith("REFIT")]
    print(f"records {len(by)}; planned (excluding the post-freeze REFIT) {len(plan)}; missing {len(missing)}: {missing}")
    if extra:
        print("unplanned records:", extra)

    # ---- N0
    print("\n### N0\n")
    n0 = by.get("N0 scan25phys")
    if n0:
        e11, e12 = n0["est"]["11"], n0["est"]["12"]
        cen = [v for v in e12["pade_central_vals"]]
        got = {"arg_12": round(e12["arg"]["12"], 5), "ratio_12": round(e12["ratio"]["12"], 5),
               "arg_11": round(e11["arg"]["11"], 5), "ratio_11": round(e11["ratio"]["11"], 5),
               "Padé central (K = 12)": (round(min(cen), 5), round(max(cen), 5)) if None not in cen else None,
               "G_11": round(e11["G"], 5), "G_12": round(e12["G"], 5)}
        exp = {"arg_12": 1.00011, "ratio_12": 0.99915, "arg_11": 1.0002, "ratio_11": 0.99919,
               "Padé central (K = 12)": (0.99987, 0.99991), "G_11": 0.99987, "G_12": 0.99988}
        print("| quantity | expected (README, comms) | harness | |\n|---|---|---|---|")
        for k in exp:
            print(f"| {k} | {exp[k]} | {got[k]} | {'pass' if got[k] == exp[k] else 'FAIL'} |")
        txt = open(os.path.join(HERE, "scan25phys", "crossing.txt")).read().replace("\r\n", "\n")
        same = e12.get("linear_stdout", "") == txt
        print(f"| arg_crossing.py stdout vs crossing.txt | identical | {'identical' if same else 'differs'} | "
              f"{'pass' if same else 'FAIL'} |")
        ait = {int(k): v for k, v in e12["aitken_quad"].items()}
        print("\nAitken (quadratic crossings, triples ending at k = 7..12): "
              + ", ".join(f"{ait[k]:.6f}" for k in range(7, 13) if k in ait))
    n23 = by.get("N0 scan23phys")
    if n23:
        has12 = "12" in n23["est"]
        print(f"\nSelf-test: N0 on scan23phys against the K = 12 expectations: "
              f"{'FAIL (no K = 12 data), as required' if not has12 else 'ran at K = 12: control did not fire'}")

    # ---- N1, N2
    print("\n### N1, N2\n")
    worst = 0.0
    n1 = [r for r in by.values() if r["family"] == "N1"]
    for r in n1:
        for K in ("11", "12", "13"):
            g = R.grade(r, int(K))
            for k, v in g["err"].items():
                if k != "aitken":
                    worst = max(worst, abs(v) if v is not None else float("inf"))
    print(f"N1: {len(n1)}/7 plants; worst |estimate - wt| over arg, ratio, Padé, G and K = 11-13: {worst:.1e}")
    for r in by.values():
        if r["family"] == "N2":
            for K, e in r["est"].items():
                nums = {k: v for k, v in {"arg": [v for v in e["arg"].values() if v is not None],
                                          "ratio": [v for v in e["ratio"].values() if v is not None],
                                          "pade": [v for v in e["pade"].values() if v is not None],
                                          "G": e.get("G")}.items() if v}
                print(f"N2 {r['name']} K{K}: numbers reported: {nums or 'none'}; errors: {e.get('errors')}; "
                      f"G_err: {e.get('G_err', '')[:80]}")

    # ---- coverage
    fams = group(by, fam_label)
    order = ["P-ARC", "P-ARC-B chord", "P-ARC-B mirror", "P-ARC-B beta2", "C (completions)", "P-NOISE-a",
             "P-NOISE-b 1e-12", "P-NOISE-b 1e-9", "P-NOISE-b 1e-6", "P-NOISE-b 1e-4",
             "P-Z2 x2=9.0-1.0j", "P-Z2 x2=8.6+1.2j", "P-Z2 x2=9.3+0j", "REFIT"]
    print("\n### Coverage (passes / plants)\n")
    print("| family | K | P1 bracket covers | P2 directions | P3 Padé ±1.2e-4 | P3 G ±1.2e-4 | P4 Aitken ±2e-4 |")
    print("|---|---|---|---|---|---|---|")
    for f in order:
        rs = fams.get(f, [])
        for K in (11, 12, 13):
            rk = [r for r in rs if str(K) in r["est"]]
            if not rk:
                continue
            gs = [R.grade(r, K) for r in rk]
            c = {p: sum(g[p] for g in gs) for p in ("P1", "P2", "P3_pade", "P3_G", "P4")}
            n = len(gs)
            print(f"| {f} | {K} | {c['P1']}/{n} | {c['P2']}/{n} | {c['P3_pade']}/{n} | {c['P3_G']}/{n} | {c['P4']}/{n} |")

    # ---- signed errors
    print("\n### Signed errors at K = 12, (estimate − ω_t) × 10⁴: median [min, max]\n")
    print("| family | n | " + " | ".join(LABEL[e] for e in EST) + " |")
    print("|---|---|" + "---|" * len(EST))
    pooled = {e: [] for e in EST}
    for f in order:
        rs = fams.get(f, [])
        if not rs:
            continue
        errs = {e: [R.grade(r, 12)["err"][e] for r in rs] for e in EST}
        cells = []
        for e in EST:
            v = [x for x in errs[e] if x is not None]
            if f in ("P-ARC", "P-ARC-B chord", "P-ARC-B mirror", "P-ARC-B beta2", "C (completions)"):
                pooled[e] += v
            nn = len(errs[e]) - len(v)
            cells.append("—" if not v else f"{statistics.median(v) * 1e4:+.2f} [{min(v) * 1e4:+.2f}, {max(v) * 1e4:+.2f}]"
                         + (f" ({nn} none)" if nn else ""))
        print(f"| {f} | {len(rs)} | " + " | ".join(cells) + " |")
    print("\nBias rule (P-ARC, P-ARC-B, C pooled): ")
    for e in EST:
        v = pooled[e]
        if not v:
            continue
        pos = sum(x > 0 for x in v) / len(v)
        med = statistics.median(v)
        biased = (pos >= 0.8 or pos <= 0.2) and abs(med) > 3e-5
        print(f"- {LABEL[e]}: n = {len(v)}, positive {pos * 100:.0f}%, median {med * 1e4:+.2f}e-4 -> "
              f"{'BIASED' if biased else 'not biased by the rule'}")

    # ---- per plant
    print("\n### Per plant, K = 12: ω_t and errors × 10⁴\n")
    print("| plant | ω_t | arg₁₂ | ratio₁₂ | Padé | G₁₂ | Aitken | P1 | detectors |")
    print("|---|---|---|---|---|---|---|---|---|")
    for f in order[:5] + order[10:]:
        for r in fams.get(f, []):
            g = R.grade(r, 12)
            d = R.detectors(r, 12)
            print(f"| {r['name']} | {r['truth']:.6f} | " + " | ".join(f4(g['err'][e]) for e in EST)
                  + f" | {'yes' if g['P1'] else 'NO'} | {', '.join(d) or '—'} |")

    # ---- completions
    chk = json.load(open(R.CHECKS)) if os.path.exists(R.CHECKS) else {}
    print("\n### Completions: truth\n")
    for k in ("C-W1", "C-W2", "C-W3", "C-EXP"):
        if k in chk:
            c = chk[k]
            xr = {w: c["info"][w]["x_R"] for w in c["info"]}
            extra = {w: {kk: vv for kk, vv in c["info"][w].items() if kk in ("kappa", "beta", "max_pole",
                                                                              "reproduce_max_rel")}
                     for w in c["info"]}
            print(f"- {k}: ω_R = {c['truth']['im_x']:.6f} (Im x), {c['truth']['im_sqrt']:.6f} (Im √x); "
                  f"x_R = {xr}; {extra}")

    # ---- refit
    if "REFIT" in chk:
        print("\n### REFIT (post-freeze): max relative fit residual per node vs planted ω_t\n")
        print("| ω_t | 0.99 | 1.00 | 1.01 | arg₁₂ | ratio₁₂ | Padé | G₁₂ | P1 |\n|---|---|---|---|---|---|---|---|---|")
        for wt, res in chk["REFIT"].items():
            r = by.get(f"REFIT P-ARC wt={wt}")
            g = R.grade(r, 12) if r else None
            est = (" | ".join(f"{g['est'][e]:.6f}" if g['est'][e] is not None else "—" for e in ("arg", "ratio", "pade", "G"))
                   + f" | {'yes' if g['P1'] else 'NO'}") if g else "— | — | — | — | —"
            print(f"| {wt} | {res['0.99']:.1e} | {res['1.0']:.1e} | {res['1.01']:.1e} | {est} |")

    # ---- corruptions
    print("\n### Corruptions (K = 12): change vs the clean twin × 10⁴, and detectors\n")
    print("| plant | corruption | " + " | ".join(LABEL[e] for e in EST) + " | moved (>1.2e-4) | detectors fired |")
    print("|---|---|" + "---|" * len(EST) + "---|---|")
    for base in ("CORR P-ARC wt=1.0000", "CORR C-W1"):
        tw = by.get(f"{base} clean twin")
        if not tw:
            continue
        gt = R.grade(tw, 12)
        dt = R.detectors(tw, 12)
        print(f"| {base} | clean twin (errors vs truth) | " + " | ".join(f4(gt["err"][e]) for e in EST)
              + f" | — | {', '.join(dt) or 'none'} |")
        for n, r in by.items():
            if not n.startswith(base) or n.endswith("clean twin"):
                continue
            g = R.grade(r, 12)
            d = [x for x in R.detectors(r, 12) if x not in dt]
            dl = []
            for e in EST:
                a, b = g["est"][e], gt["est"][e]
                dl.append(None if a is None or b is None else a - b)
            moved = [e for e, x in zip(EST, dl) if x is not None and abs(x) > 1.2e-4] + \
                [e for e, x, a in zip(EST, dl, [g["est"][e] for e in EST]) if a is None and gt["est"][e] is not None]
            gerr = r["est"]["12"].get("G_production")
            note = f" (production G: {gerr[:40]})" if isinstance(gerr, str) and gerr != "ran" else ""
            print(f"| {base} | {n[len(base) + 1:]}{note} | " + " | ".join(f4(x) if x is not None else "none" for x in dl)
                  + f" | {', '.join(moved) or 'no'} | {', '.join(d) or 'none'} |")

    # ---- self-test of the coverage check
    p = by.get("P-ARC wt=1.0000")
    if p:
        g = R.grade(p, 12, truth=1.002)
        print(f"\nSelf-test: P-ARC wt = 1.0000 graded against 1.002: P1 {g['P1']}, P3 {g['P3']} "
              f"({'fires' if not g['P1'] else 'DID NOT FIRE'})")


if __name__ == "__main__":
    main()
