"""Best available exact values: kc63/kc47 at D=4,5; old kc31 scan for D<7.75 (precise there); long-series kc23 for D>=7.75."""
import json, numpy as np
def load():
    rows = {}
    for l in open("scan_up_kc31.jsonl"):
        r = json.loads(l)
        if r.get("gamma") and r["D"] < 7.7: rows[round(r["D"], 4)] = dict(D=r["D"], Delta=r["Delta"], gamma=r["gamma"], lam_g=r["lam_gauge"], c0=r.get("c0_min"))
    for l in open("scan_long_kc23.jsonl"):
        r = json.loads(l)
        rows[round(r["D"], 4)] = dict(D=r["D"], Delta=r["Delta"], gamma=r.get("gamma") or rows.get(round(r["D"], 4), {}).get("gamma"), lam_g=r["lam_gauge"], c0=r["c0_min"], s0=r["s0"])
    import os
    if os.path.exists("gamma_long.jsonl"):
        for l in open("gamma_long.jsonl"):
            r = json.loads(l); k = round(r["D"], 4)
            if r.get("gamma"):
                if k in rows: rows[k]["gamma"] = r["gamma"]
    for l in open("scan_long_kc23.jsonl"):
        r = json.loads(l); k = round(r["D"], 4)
        if not r.get("gamma") and k not in rows: rows[k] = dict(D=r["D"], Delta=r["Delta"], gamma=None, lam_g=r["lam_gauge"], c0=r["c0_min"], s0=r["s0"])
    # re-converged values (exact horizon Jacobian): override Delta; drop old unpolished points above D=17.2
    pol = {}
    for fn in ("scan_long_pol.jsonl", "scan_longB.jsonl"):
        if os.path.exists(fn):
            for l in open(fn):
                r = json.loads(l); pol[round(r["D"], 4)] = r
    for k, r in pol.items():
        if k in rows: rows[k]["Delta"] = r["Delta"]; rows[k]["polished"] = True
        else: rows[k] = dict(D=r["D"], Delta=r["Delta"], gamma=None, lam_g=r.get("lam_gauge"), c0=r.get("c0_min"), s0=r.get("s0"), polished=True)
    for k in [k for k, v in rows.items() if k > 17.2 and not v.get("polished")]:
        del rows[k]
    rows[4.0] = dict(D=4.0, Delta=3.4454523972, gamma=0.3739609, lam_g=0.61385, c0=None)
    rows[5.0] = dict(D=5.0, Delta=3.2217019415, gamma=0.4131269359, lam_g=0.6159, c0=None)
    return rows
if __name__ == "__main__":
    rows = load(); rows = {k: v for k, v in rows.items() if v.get("gamma")}; D = np.array(sorted(rows)); n = D - 2
    Dl = np.array([rows[d]["Delta"] for d in D]); g = np.array([rows[d]["gamma"] for d in D]); lg = np.array([rows[d]["lam_g"] for d in D])
    ln = np.log(n)
    q = -np.gradient(np.log(Dl), ln); p = -np.gradient(np.log(0.5 - g), ln); pl = -np.gradient(np.log(2 / g - 4) if False else np.log(1/g - 2), ln)
    print(f"{'D':>7} {'Delta':>11} {'gamma':>10} {'q(Delta)':>8} {'p(1/2-g)':>8} {'p(lam-2)':>8} {'Delta*n^.25':>11} {'(1/2-g)sqrt n':>13} {'lam_g':>7}")
    for i in range(len(D)):
        print(f"{D[i]:7.3f} {Dl[i]:11.8f} {g[i]:10.7f} {q[i]:8.4f} {p[i]:8.4f} {pl[i]:8.4f} {Dl[i]*n[i]**.25:11.5f} {(0.5-g[i])*np.sqrt(n[i]):13.6f} {lg[i]:7.4f}")
