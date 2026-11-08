"""Print the tables of notes/RT_FORMULATION.md from the records in a raw-data folder.

    python3 solver/rt/rt_summary.py ../choptuik-large-d-raw/I91-mac

Reads accept/accept.jsonl and accept_mutcoef.jsonl (rt_run.py), D4/res4.jsonl, cont/cont10.jsonl, rt_kappa*.jsonl
(rt_cond.py), kappa_mp.jsonl (rt_cond_mp.py), ours_kappa.jsonl (hp/stall_margin.py) and lu_cost.jsonl (lu_cost.py),
whichever exist. Exits 1 if the folder has none of them.
"""
import os
import sys
import json
import glob


def rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    d = sys.argv[1]
    seen = 0
    print("Delta against ours (rt_run.py):")
    print("| D | grid | L | residual | Delta_RT - Delta_hp | mu | run |")
    print("|---|---|---|---|---|---|---|")
    for f in ("D4/res4.jsonl", "cont/cont10.jsonl", "accept/accept.jsonl", "accept/accept_mutcoef.jsonl"):
        for r in rows(os.path.join(d, f)):
            seen += 1
            if r["label"] == "cont":
                continue
            how = "arb residual" if r.get("mixed") else "f64"
            print(f"| {r['D']:g} | {r['grid'][0]}x{r['grid'][1]} | {r['L']} | {how} | {r['diff']:+.2e} | "
                  f"{r['mu']:.12f} | {f.split('/')[-1].replace('.jsonl', '')}{' (mutate coef)' if r.get('mutate') else ''} |")
    ks = []
    for f in sorted(glob.glob(os.path.join(d, "rt_kappa*.jsonl"))):
        ks += rows(f)
    if ks:
        seen += 1
        print("\nRT Jacobian, f64 (rt_cond.py):")
        print("| D | grid | L | sigma_max | sigma_min | kappa | kappa_eq | weakest | share xi<-0.8 / |xi|<0.2 / xi>0.8 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(ks, key=lambda r: (r["D"], r["grid"])):
            loc = r.get("loc")
            ls = f"{loc['hor_minus']:.2f} / {loc['centre']:.2f} / {loc['hor_plus']:.2f}" if loc else ""
            w = r["weakest"]
            print(f"| {r['D']:g} | {r['grid'][0]}x{r['grid'][1]}{' (resampled)' if r.get('resampled') else ''} | "
                  f"{r['L']} | {r['sigma_max']:.2e} | {r['sigma_min']:.2e} | {r['kappa']:.2e} | {r['kappa_eq']:.2e} | "
                  f"{w['field']} h{w['harmonic_of_Delta']:g} | {ls} |")
    kn = rows(os.path.join(d, "rt_kappa_N.jsonl"))
    if kn:
        import numpy as np
        print("\nRT sigma_min and sigma_max against N at N_t = 16 (rt_kappa_N.jsonl): exponents of a least-squares fit "
              "in log N")
        print("| D | N | sigma_min | sigma_max | slope of ln sigma_min | slope of ln sigma_max |")
        print("|---|---|---|---|---|---|")
        for D in sorted({r["D"] for r in kn}):
            rr = sorted([r for r in kn if r["D"] == D], key=lambda r: r["grid"][1])
            lnN = np.log([r["grid"][1] for r in rr])
            a = np.polyfit(lnN, np.log([r["sigma_min"] for r in rr]), 1)[0]
            b = np.polyfit(lnN, np.log([r["sigma_max"] for r in rr]), 1)[0]
            Ns = ", ".join(str(r["grid"][1]) for r in rr)
            sm = ", ".join("%.2e" % r["sigma_min"] for r in rr)
            print(f"| {D:g} | {Ns} | {sm} | {rr[0]['sigma_max']:.0f} .. {rr[-1]['sigma_max']:.0f} | {a:+.2f} | "
                  f"{b:+.2f} |")
    km = rows(os.path.join(d, "kappa_mp.jsonl"))
    if km:
        seen += 1
        print("\nRT Jacobian in arb (rt_cond_mp.py):")
        print("| D | grid | L | state | bits | sigma_max | sigma_min | kappa | resolved | weakest | share xi<-0.8 / "
              "|xi|<0.2 / xi>0.8 | seconds |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(km, key=lambda r: (r["D"], r["grid"], r["src"])):
            loc = r.get("loc")
            ls = f"{loc['hor_minus']:.2f} / {loc['centre']:.2f} / {loc['hor_plus']:.2f}" if loc else ""
            w = r.get("weakest", {})
            src = "mapped" if "profiles" in r["src"] else "RT solution"
            print(f"| {r['D']:g} | {r['grid'][0]}x{r['grid'][1]} | {r['L']} | {src} | {r['prec']} | "
                  f"{r['sigma_max']:.2e} | {r['sigma_min']:.2e} | {r['kappa']:.2e} | {r['resolved']} | "
                  f"{w.get('field', '')} h{w.get('harmonic_of_Delta', '')} | {ls} | {r['secs']} |")
    ok = rows(os.path.join(d, "ours_kappa.jsonl"))
    if ok:
        seen += 1
        print("\nOurs (hp/stall_margin.py; kappa = u_kappa / 2^-104):")
        print("| D | state | kfreeze | sigma_min | sigma_max | kappa | weakest harmonic |")
        print("|---|---|---|---|---|---|---|")
        for r in ok:
            dd = r["dirs"][0]
            print(f"| {r['D']:g} | {os.path.basename(r['state'])} | {r['kfreeze']} | {dd['sigma']:.2e} | "
                  f"{dd['sigma_max']:.2e} | {r['u_kappa'] / 2.0 ** -104:.2e} | {dd['harm']} |")
    lc = rows(os.path.join(d, "lu_cost.jsonl"))
    if lc:
        seen += 1
        print("\nDense arb LU solve (lu_cost.py):")
        print("| L | bits | seconds | seconds / L^3 |")
        print("|---|---|---|---|")
        for r in lc:
            print(f"| {r['L']} | {r['prec']} | {r['seconds']:.2f} | {r['per_L3']:.2e} |")
    sys.exit(0 if seen else 1)


if __name__ == "__main__":
    main()
