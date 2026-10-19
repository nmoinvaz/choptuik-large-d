"""Best values: formula_zoo.load() base; Delta overridden by re-converged points (exact horizon Jacobian);
gamma overridden by complex-step mode-matrix roots; stale values above D=17.2 (Delta) / 16 (gamma) dropped."""
import json, os
from formula_zoo import load as _load
def load():
    rows = _load()
    pol = {}
    for fn in ("scan_long_pol.jsonl", "scan_longB.jsonl"):
        if os.path.exists(fn):
            for l in open(fn):
                r = json.loads(l); pol[round(r["D"], 4)] = r
    for k in [k for k in rows if k > 17.2 and k not in pol]: del rows[k]
    for k, r in pol.items():
        rows.setdefault(k, dict(D=r["D"], gamma=None))["Delta"] = r["Delta"]
    for k in rows:
        if k > 16.0: rows[k]["gamma"] = None
    if os.path.exists("gamma_cs.jsonl"):
        for l in open("gamma_cs.jsonl"):
            r = json.loads(l); k = round(r["D"], 4)
            if r.get("gamma") and k in rows: rows[k]["gamma"] = r["gamma"]; rows[k]["gamma_cs"] = True
    return rows
