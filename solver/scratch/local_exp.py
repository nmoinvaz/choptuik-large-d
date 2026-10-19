"""Local power-law exponents: if gamma -> 1/2 - b n^-p and Delta ~ A n^-q, the local slopes reveal p and q."""
import json, numpy as np
rows = {}
for l in open("scan_up_kc31.jsonl"):
    r = json.loads(l)
    if r.get("gamma"): rows[round(r["D"], 4)] = r
rows[4.0] = dict(D=4.0, gamma=0.3739609, Delta=3.4454523972)
rows[5.0] = dict(D=5.0, gamma=0.4131269359, Delta=3.2217019415)
D = np.array(sorted(rows)); n = D - 2
g = np.array([rows[d]["gamma"] for d in D]); Dl = np.array([rows[d]["Delta"] for d in D]); lam = 1 / g
ln = np.log(n)
p_g = -np.gradient(np.log(0.5 - g), ln)
p_l = -np.gradient(np.log(lam - 2), ln)
q = -np.gradient(np.log(Dl), ln)
print(f"{'D':>7} {'gamma':>10} {'Delta':>10} {'p(1/2-g)':>9} {'p(lam-2)':>9} {'q(Delta)':>9} {'(1/2-g)sqrt n':>13} {'Delta n^.25':>11}")
for i in range(len(D)):
    print(f"{D[i]:7.3f} {g[i]:10.7f} {Dl[i]:10.6f} {p_g[i]:9.3f} {p_l[i]:9.3f} {q[i]:9.3f} {(0.5-g[i])*np.sqrt(n[i]):13.5f} {Dl[i]*n[i]**.25:11.5f}")
