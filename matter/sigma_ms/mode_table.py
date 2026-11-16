#!/usr/bin/env python3
"""The growth rate lambda(D) of the sigma-model's DSS solution on the multishoot kernel (#918), from mode_lam.py's
records, against sigdss's values, into a csv and a markdown table.

    git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw     (once, beside the checkout)
    python3 matter/sigma_ms/mode_table.py ../choptuik-large-d-raw [--folder I918-mac] [--Dmin 9]
        [--csv matter/sigma_ms/data/lam_D.csv] [--mutate gauge|sigdss]

Input: <folder>/out/sm_lam.jsonl, for each (eta, D, M) its last record with a growth rate. sigdss's rates are read
from solver/matter/data.

Checks, each printed with its verdict; the exit status is 1 when one fails. The bounds were written here before the
kernel's first growth rate was computed.
  (1) The gauge root: at every state with D >= --Dmin, the two rates that enclose the sign change of det M nearest
      lambda_g = mean(1 - c_x(tau, 1)) both lie within 1e-10 of it. The states below --Dmin are the validation
      states: their distance is printed with the same bound's verdict, outside the exit status.
  (2) Against sigdss, where it needs no smoothness prior or has two resolutions:
        D 4, eta 1     |kernel - sigdss (k_c 15)| <= B4, B4 = sigdss's k_c 15 against the l = 1 solver's k_c 23 at
                       eta = inf, D 4 (0.0086)
        D 4, eta inf   |kernel - the l = 1 solver's k_c 23 value| <= B4
        D 6, eta 1     |kernel - sigdss (k_c 23)| <= sigdss's |k_c 15 - k_c 23| there (2.6e-6)
        D 9, eta 1     |kernel - sigdss (k_c 23)| <= sigdss's |k_c 15 - k_c 23| there (4.2e-4)
        D 9, eta 0.3   |kernel - sigdss (k_c 15)| <= the same 4.2e-4 (sigdss has one value at this point)
      A comparison whose kernel state is absent is listed as absent and fails.
Then, without a verdict, lambda(D) at eta = 0.3 from D 10 to 11 at M = 32: its differences, where it is smallest, and
the largest |lambda(M) - lambda(32)| over the other M as the resolution's spread.
--mutate gauge adds 1e-9 to one gauge distance in the range, which check (1) must catch; --mutate sigdss shifts the
kernel's rate at D 9, eta 1 by 1e-3, which check (2) must catch.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "solver", "matter", "data")


def rows_of(name):
    return [json.loads(line) for line in open(os.path.join(DATA, name))]


def sig_lam(name, D, eta, kc):
    """sigdss's growth rate in a data file at (D, eta, k_c); eta None is eta = infinity"""
    for r in rows_of(name):
        e = r.get("eta")
        e_ok = (eta is None and (e is None or r.get("eps") == 0.0)) or (eta is not None and e is not None
                                                                         and abs(e - eta) < 1e-5)
        if abs(r["D"] - D) < 1e-9 and e_ok and r.get("kc") == kc and r.get("lam") is not None:
            return r["lam"]
    raise KeyError((name, D, eta, kc))


def references():
    """[(eta, D, sigdss's rate, its source, the bound, the bound's source)]"""
    l1 = sig_lam("sigdss_l1_reference.jsonl", 4.0, None, 23)
    b4 = abs(sig_lam("sigdss_lam_d4.jsonl", 4.0, None, 15) - l1)
    d6 = sig_lam("sigdss_refine_kc23.jsonl", 6.0, 1.0, 23)
    b6 = abs(sig_lam("sigdss_m1up.jsonl", 6.0, 1.0, 15) - d6)
    d9 = sig_lam("sigdss_refine_kc23.jsonl", 9.0, 1.0, 23)
    b9 = abs(sig_lam("sigdss_m1up.jsonl", 9.0, 1.0, 15) - d9)
    return [("1", 4.0, sig_lam("sigdss_lam_d4b.jsonl", 4.0, 1.0, 15), "sigdss k_c 15", b4, "k_c 15 - 23 at eta inf, D 4"),
            ("inf", 4.0, l1, "l = 1 solver, k_c 23", b4, "k_c 15 - 23 at eta inf, D 4"),
            ("1", 6.0, d6, "sigdss k_c 23", b6, "k_c 15 - 23 there"),
            ("1", 9.0, d9, "sigdss k_c 23", b9, "k_c 15 - 23 there"),
            ("0.3", 9.0, sig_lam("sigdss_m03up.jsonl", 9.0, 0.3, 15), "sigdss k_c 15", b9, "k_c 15 - 23 at eta 1, D 9")]


def sigdss_all():
    """{(eta, D): [(source, rate)]} of sigdss's marches and refinements, for the table's last columns"""
    out = {}
    for name in ("sigdss_m03up.jsonl", "sigdss_m03hi.jsonl", "sigdss_m1up.jsonl", "sigdss_m1hi.jsonl",
                 "sigdss_refine_kc23.jsonl", "sigdss_lam_d4b.jsonl"):
        for r in rows_of(name):
            if r.get("lam") is None or r.get("eta") is None:
                continue
            eta = "1" if abs(r["eta"] - 1) < 1e-5 else "0.3" if abs(r["eta"] - 0.3) < 1e-5 else None
            if eta:
                src = name.replace("sigdss_", "").replace(".jsonl", "") + f", k_c {r['kc']}"
                out.setdefault((eta, r["D"]), []).append((src, r["lam"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--folder", default="I918-mac")
    ap.add_argument("--Dmin", type=float, default=9.0)
    ap.add_argument("--csv", default="")
    ap.add_argument("--mutate", default="", choices=["", "gauge", "sigdss"])
    a = ap.parse_args()
    recs = {}
    for line in open(os.path.join(a.raw, a.folder, "out", "sm_lam.jsonl")):
        r = json.loads(line)
        if r.get("lam") is None and r.get("gauge_diff") is None:
            continue
        key = (str(r["eta"]), float(r["D"]), int(r["M"]))
        recs[key] = dict(recs.get(key, {}), **{k: v for k, v in r.items() if v is not None})
    keys = sorted(recs, key=lambda k: (-float("inf") if k[0] == "inf" else -float(k[0]), k[1], k[2]))
    ok = True
    # (1)
    inr = [k for k in keys if k[1] >= a.Dmin and "gauge_diff" in recs[k]]
    gd = {k: recs[k]["gauge_diff"] for k in keys if "gauge_diff" in recs[k]}
    if a.mutate == "gauge":
        gd[inr[len(inr) // 2]] += 1e-9
    worst = max(gd[k] for k in inr) if inr else float("inf")
    v1 = bool(inr) and worst <= 1e-10
    ok &= v1
    print(f"(1) the gauge root at the {len(inr)} states with D >= {a.Dmin:g}: largest distance from lambda_g "
          f"{worst:.1e} (<= 1e-10): {'PASS' if v1 else 'FAIL'}")
    for k in keys:
        if k[1] < a.Dmin and k in gd:
            print(f"      validation state eta {k[0]}, D {k[1]:g}, M {k[2]}: {gd[k]:.1e} "
                  f"({'within' if gd[k] <= 1e-10 else 'outside'} 1e-10)")
    # (2)
    lam = {k: recs[k]["lam"] for k in keys if "lam" in recs[k]}
    if a.mutate == "sigdss":
        lam[("1", 9.0, 32)] += 1e-3
    print("(2) against sigdss:")
    for eta, D, ref, src, bound, bsrc in references():
        k = (eta, D, 32)
        if k not in lam:
            ok = False
            print(f"      eta {eta}, D {D:g}: no kernel state: FAIL")
            continue
        d = lam[k] - ref
        v = abs(d) <= bound
        ok &= v
        print(f"      eta {eta}, D {D:g}: kernel {lam[k]:.9f}, {src} {ref:.9f}, difference {d:+.2e} "
              f"(<= {bound:.1e} in size, {bsrc}): {'PASS' if v else 'FAIL'}")
    # the turn
    seq = [(k[1], lam[k]) for k in keys if k[0] == "0.3" and k[2] == 32 and 10.0 <= k[1] <= 11.0 and k in lam]
    if len(seq) >= 3:
        dl = [(q[0], (q[1] - p[1]) / (q[0] - p[0])) for p, q in zip(seq, seq[1:])]
        lo = min(seq, key=lambda x: x[1])
        spread = [(k, abs(lam[k] - lam[(k[0], k[1], 32)])) for k in keys
                  if k[0] == "0.3" and k[2] != 32 and k in lam and (k[0], k[1], 32) in lam]
        sp = max((x[1] for x in spread), default=float("nan"))
        print(f"lambda(D) at eta 0.3, M 32, D 10 to 11: {', '.join(f'{D:g}: {v:.9f}' for D, v in seq)}")
        print(f"    slopes d lambda / dD up to each D: {', '.join(f'{D:g}: {v:+.5f}' for D, v in dl)}")
        print(f"    smallest at D {lo[0]:g}; lambda(11) minus the smallest {seq[-1][1] - lo[1]:.3e}; resolution spread "
              f"(largest |lambda(M) - lambda(32)| over {[(k[1], k[2]) for k, _ in spread]}) {sp:.1e}")
    sig = sigdss_all()
    rows = []
    for k in keys:
        r = recs[k]
        gs, ls = r.get("gauge_sign_change") or ["", ""], r.get("lam_sign_change") or ["", ""]
        for src, sl in sig.get((k[0], k[1]), [("", "")]) if k[2] == 32 else [("", "")]:
            rows.append(dict(eta=k[0], D=k[1], M=k[2], F=r.get("F", ""), lam_g=r.get("lam_g", ""), gauge_lo=gs[0],
                             gauge_hi=gs[1], gauge_diff=r.get("gauge_diff", ""), lam=r.get("lam", ""), lam_lo=ls[0],
                             lam_hi=ls[1], gamma=1 / r["lam"] if "lam" in r else "", sigdss=src, sigdss_lam=sl,
                             kernel_minus_sigdss=(r["lam"] - sl) if "lam" in r and sl != "" else ""))
    print("\n| η | D | M | λ_g | gauge root's distance | λ | γ = 1/λ | sigdss's λ | kernel minus sigdss |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        g = f"{r['gauge_diff']:.1e}" if r["gauge_diff"] != "" else ""
        la = f"{r['lam']:.10f}" if r["lam"] != "" else ""
        ga = f"{r['gamma']:.10f}" if r["gamma"] != "" else ""
        s_ = f"{r['sigdss_lam']:.6f} ({r['sigdss']})" if r["sigdss"] else ""
        d = f"{r['kernel_minus_sigdss']:+.1e}" if r["kernel_minus_sigdss"] != "" else ""
        print(f"| {r['eta']} | {r['D']:g} | {r['M']} | {r['lam_g']:.10f} | {g} | {la} | {ga} | {s_} | {d} |")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{len(rows)} rows -> {a.csv}")
    print(f"\nMODE TABLE{' (mutation ' + a.mutate + ')' if a.mutate else ''}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
