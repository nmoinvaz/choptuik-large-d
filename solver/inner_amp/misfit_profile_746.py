#!/usr/bin/env python3
"""The misfit's profile over M on #718's ring records (issue #746; notes/PADE_FAR_END.md section 17).

#718 judged its plants by chi at 22 terms only, (c') held and (c'') relocated. Its records also keep chi at 20 and 21
terms. Here the profile chi_M, M = 20, 21, 22, of each plant is compared with the real series' in zk/chord, on the
records, with no new plant. The criterion (section 17.2, committed before any profile was read):

    rho_M = chi_M / chi_M(real) at the same node;  (d') 1/5 <= rho_M <= 5 for M = 20, 21, 22, held (twin kstar/chord);
    (d'') the same with the plant's own located zero (reloc).  P' = (a), (c'), (d') at the three nodes; P'' = (a), (c''),
    (d'').  The location rule is #718's, and a plant status the records cannot decide is 'unknown'.

Standard library only; it imports none of the plants' machinery, so it shares only the records with #718's script.

    git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw      (once)
    python3 solver/inner_amp/misfit_profile_746.py ../choptuik-large-d-raw/I718-mac [--out DIR] [--mutate=MUT]

MUT: scale_chi[:M] (chi_M of every plant profile and of J+ and the edge check x 10, M = 21 by default) must make J+ and
the edge check fail; drop_node (every record at 1.01 dropped) must make R-a fail. Exit 0 when every control and check
passes, 1 when one fails (a mutation that fires also exits 1), 2 when a mutation does not fire. The verdict is not an
exit code. Numerical evidence on plants, not a bound.
"""
import argparse
import json
import math
import os
import sys
import time

F = 5.0                                  # the profile window 1/F..F (frozen)
MS = (20, 21, 22)
NODES = (0.99, 1.0, 1.01)
WF0 = 1.0
N23 = 23
CHI_FACTOR = 10.0                        # (c')'s window 1/10..10 on chi_22
NEED_DA = 5                              # (a): m_DA <= 5
GROUPS = (("chord2", 0.3, 4), ("arc", 0.3, 4), ("chord2", 0.15, 4), ("chord2", 0.6, 4), ("chord2", 0.3, 2),
          ("chord2", 0.3, 6))
LAT_A = tuple(range(-1, 5))
LAT_B = tuple(range(-2, 5))
ORDER = {"pass": 2, "unknown": 1, "fail": 0}
PUB_REAL_WF0 = {20: 4745.0, 21: 4359.0, 22: 402.0}         # section 15.5, to its digits
PUB_COUNTS = {"a": 3477, "c": 5064, "ac": 3401, "ac2": 3433}  # section 16.4, at wf 1.00 of 5392 plants
PUB_RECORDS = {0.99: 303, 1.0: 5392, 1.01: 272}


def gkey(kind, L, P):
    return f"{kind}/{L}/{P}"


class Say:
    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        print(s, flush=True)
        self.lines.append(s)


say = Say()


# ---------------------------------------------------------------- the mutation hook, the criterion
class Mut:
    def __init__(self, spec):
        self.kind, self.M = None, None
        if spec:
            name, _, arg = spec.partition(":")
            if name not in ("scale_chi", "drop_node"):
                raise SystemExit(f"unknown mutation {spec!r}")
            self.kind = name
            self.M = int(arg) if (name == "scale_chi" and arg) else 21
            if self.kind == "scale_chi" and self.M not in MS:
                raise SystemExit("scale_chi:M needs M in 20, 21, 22")

    def prof(self, chi):
        """a plant-side profile as the criterion reads it; scale_chi scales one M's chi by 10"""
        if self.kind == "scale_chi":
            chi = dict(chi)
            chi[self.M] = chi[self.M] * 10.0
        return chi


def profile_ok(chi, real, f=F):
    """1/f <= chi_M / chi_M(real) <= f for every M, inclusive"""
    for M in MS:
        rho = chi[M] / real[M]
        if not (rho > 0 and 1.0 / f <= rho <= f):
            return False
    return True


def held_ok(r, rw, mut):
    return r["chi_M_min"] == N23 and profile_ok(mut.prof(r["chi"]), rw["chi"])


def reloc_ok(r, rw, mut):
    rl = r["reloc"]
    return rl is not None and rl["M_min"] == N23 and profile_ok(mut.prof(rl["chi"]), rw["chi"])


def plant_test(r, crit, profile_on, real, mut):
    """(a), (c') or (c'') as stored, and (d') or (d'') when the profile is on"""
    if not r["a"]:
        return False
    if crit == "c":
        return bool(r["c"]) and (held_ok(r, real[r["wf"]], mut) if profile_on else True)
    return bool(r["c2"]) and (reloc_ok(r, real[r["wf"]], mut) if profile_on else True)


# ---------------------------------------------------------------- records
def slim(r):
    tw = r["twin"]["kstar/chord"]
    j, rl = r["judge"], r["reloc"]
    reloc = None
    if "zero" in rl:
        reloc = {"M_min": rl["M_min"], "chi": {int(k): float(v) for k, v in rl["chi"].items()}}
    state = "skipped" if "skipped" in rl else "not_evaluated" if "not_evaluated" in rl else "evaluated"
    return {"id": r["id"], "wf": r["wf"], "pos": r["pos"], "ab": tuple(r["a_b_k"][:2]), "k": r["a_b_k"][2],
            "kind": r["kind"], "L": r["L"], "P": r["P"], "alpha": r["alpha"], "eta": r["eta"], "phase": r["phase"],
            "Q": tuple(r["Q"]), "a": bool(j["a"]), "c": bool(j["c"]), "c2": j["c2"], "m_DA": j["m_DA"],
            "ratio": j["ratio"], "ratio2": j["ratio2"], "chi": {int(k): float(v) for k, v in tw["chi"].items()},
            "chi_M_min": tw["M_min"], "reloc_state": state, "reloc": reloc}


class Data:
    def __init__(self, path, mut):
        self.rec = {}                      # (id, wf) -> slim record
        self.dup = 0
        self.n_lines = 0
        t0 = time.time()
        with open(os.path.join(path, "ring.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                self.n_lines += 1
                if r.get("type") == "g":
                    continue
                if mut.kind == "drop_node" and r["wf"] == 1.01:
                    continue
                key = (r["id"], r["wf"])
                if key in self.rec:
                    self.dup += 1
                self.rec[key] = slim(r)
        self.groups = {}                   # (pos, group key) -> [plant ids at wf 1.00]
        self.pos_Q = {}                    # pos -> (a, b), |Q| at wf 1.00, arg Q
        for (pid, wf), r in self.rec.items():
            if wf == WF0:
                self.groups.setdefault((r["pos"], gkey(r["kind"], r["L"], r["P"])), []).append(pid)
                if r["pos"] not in self.pos_Q:
                    q = complex(*r["Q"])
                    self.pos_Q[r["pos"]] = (r["ab"], abs(q), math.degrees(math.atan2(q.imag, q.real)))
        self.seconds = time.time() - t0


def load_real(path):
    ctl = json.load(open(os.path.join(path, "control.json")))
    real = {}
    for wf in NODES:
        n = ctl["nodes"][str(wf)]
        tw = n["real_twin"]["kstar/chord"]
        jp = n["real"]["twin"]["kstar/chord"]
        real[wf] = {"chi": {int(k): float(v) for k, v in tw["chi"].items()}, "M_min": tw["M_min"],
                    "ref22": float(n["real_ref"]["kstar/chord"][0]),
                    "jplus": {int(k): float(v) for k, v in jp["chi"].items()}, "jplus_M_min": jp["M_min"],
                    "R5": bool(n["R5"]["pass"]), "R5_dev": n["R5"]["dev"]}
    return ctl, real


# ---------------------------------------------------------------- the location rule (#718's), with unknown
def plant_status(D, pid, crit, profile_on, real, mut):
    """None when not a candidate at wf 1.00, else pass, fail or unknown from the records at 0.99 and 1.01"""
    if not plant_test(D.rec[(pid, WF0)], crit, profile_on, real, mut):
        return None
    outs = []
    for wf in (0.99, 1.01):
        r = D.rec.get((pid, wf))
        outs.append(None if r is None else plant_test(r, crit, profile_on, real, mut))
    if any(o is False for o in outs):
        return "fail"
    return "unknown" if any(o is None for o in outs) else "pass"


def best(sts):
    return max(sts, key=lambda s: ORDER[s]) if sts else "fail"


def analyze(D, crit, profile_on, real, mut):
    pos = {}
    for (p, g), pids in sorted(D.groups.items()):
        n_base = sum(1 for pid in pids if plant_test(D.rec[(pid, WF0)], crit, False, real, Mut(None)))
        sts = {pid: plant_status(D, pid, crit, profile_on, real, mut) for pid in pids}
        cand = {pid: s for pid, s in sts.items() if s is not None}
        gs = best(list(cand.values()))
        pos.setdefault(p, {"groups": {}})["groups"][g] = {
            "status": gs, "n_base": n_base, "n_cand": len(cand), "n_pass": sum(1 for s in cand.values() if s == "pass"),
            "n_unknown": sum(1 for s in cand.values() if s == "unknown"),
            "pass_ids": sorted(pid for pid, s in cand.items() if s == "pass")}
    for p, v in pos.items():
        v["status"] = best([g["status"] for g in v["groups"].values()])
        v["n_groups_pass"] = sum(1 for g in v["groups"].values() if g["status"] == "pass")
        v["n_groups_unknown"] = sum(1 for g in v["groups"].values() if g["status"] == "unknown")
    return pos


def lattice(pos_res, D):
    lat = {D.pos_Q[p][0]: v["status"] for p, v in pos_res.items()}
    nb = lambda ab: [(ab[0] + 1, ab[1]), (ab[0] - 1, ab[1]), (ab[0], ab[1] + 1), (ab[0], ab[1] - 1)]
    passing = sorted(ab for ab, s in lat.items() if s == "pass")
    unknown = sorted(ab for ab, s in lat.items() if s == "unknown")
    edge = [ab for ab in passing if any(n not in lat for n in nb(ab))]
    cand = [ab for ab in passing if ab not in edge and all(lat[n] == "fail" for n in nb(ab))]
    cand_undec = [ab for ab in passing if ab not in edge and ab not in cand and not any(lat[n] == "pass" for n in nb(ab))]
    return {"passing": passing, "unknown": unknown, "edge": edge, "candidates": cand,
            "candidates_undecided": cand_undec}


def pid_str(ab):
    return f"L{ab[0]:+d}{ab[1]:+d}"


# ---------------------------------------------------------------- the controls
def check_records(D, real, mut):
    """R-b, the counts, and the shape of the record set"""
    out = {}
    n = {wf: sum(1 for (_, w) in D.rec if w == wf) for wf in NODES}
    out["records"] = n
    out["records_ok"] = mut.kind == "drop_node" or (n == PUB_RECORDS and D.dup == 0 and len(D.groups) == 252
                                                    and len(D.pos_Q) == 42)
    bad = []
    for (pid, wf), r in D.rec.items():
        ref22 = real[wf]["ref22"]
        a = r["m_DA"] <= NEED_DA
        ratio = r["chi"][22] / ref22
        c = r["chi_M_min"] == N23 and 1 / CHI_FACTOR <= ratio <= CHI_FACTOR
        if r["reloc"] is not None:
            ratio2 = r["reloc"]["chi"][22] / ref22
            c2 = r["reloc"]["M_min"] == N23 and 1 / CHI_FACTOR <= ratio2 <= CHI_FACTOR
        else:
            ratio2, c2 = None, False
        ok = (a == r["a"] and c == r["c"] and c2 == bool(r["c2"]) and abs(ratio / r["ratio"] - 1) <= 1e-12
              and ((ratio2 is None and r["ratio2"] is None)
                   or (ratio2 is not None and r["ratio2"] is not None and abs(ratio2 / r["ratio2"] - 1) <= 1e-12)))
        if not ok:
            bad.append(f"{pid} wf {wf}")
    out["Rb_bad"] = bad
    out["Rb"] = not bad
    w0 = [r for (_, wf), r in D.rec.items() if wf == WF0]
    cnt = {"a": sum(r["a"] for r in w0), "c": sum(r["c"] for r in w0), "ac": sum(r["a"] and r["c"] for r in w0),
           "ac2": sum(r["a"] and bool(r["c2"]) for r in w0)}
    out["counts"] = cnt
    out["counts_ok"] = mut.kind == "drop_node" or cnt == PUB_COUNTS
    return out


def check_Ra(D, ring, real, mut):
    """with (d) off, the loader reproduces #718's ring.json"""
    bad = []
    unknown = 0
    res = {}
    for crit, tag in (("c", "c"), ("c2", "c2")):
        pr = analyze(D, crit, False, real, mut)
        res[crit] = pr
        for p, v in pr.items():
            sv = ring["positions"][p]
            if (v["status"] == "pass") != bool(sv["pass"][crit]):
                bad.append(f"{p} {crit}: position pass {v['status']} vs stored {sv['pass'][crit]}")
            for g, gv in v["groups"].items():
                st = sv["groups"][g][crit]
                unknown += gv["status"] == "unknown"      # a plant after the hit was never run, a group is decided
                if (gv["status"] == "pass") != st["pass"]:
                    bad.append(f"{p} {g} {crit}: group {gv['status']} vs stored pass {st['pass']}")
                if gv["n_base"] != st["n_wf1"]:
                    bad.append(f"{p} {g} {crit}: candidates {gv['n_base']} vs stored n_wf1 {st['n_wf1']}")
                if st["plant"] is not None and st["plant"] not in gv["pass_ids"]:
                    bad.append(f"{p} {g} {crit}: stored hit {st['plant']} is not a passing plant here")
        lat = lattice(pr, D)
        sv = ring["verdict"][crit]
        for key, mine in (("passing", lat["passing"]), ("edge", lat["edge"]), ("candidates", lat["candidates"]),
                          ("located", [])):
            if sorted(map(tuple, sv[key])) != sorted(mine):
                bad.append(f"{crit} {key}: {sorted(mine)} vs stored {sorted(map(tuple, sv[key]))}")
        if lat["unknown"] or lat["candidates_undecided"]:
            bad.append(f"{crit}: unknown positions with the profile off")
    if unknown:
        bad.append(f"{unknown} unknown group statuses with the profile off")
    return {"pass": not bad, "bad": bad[:20], "n_bad": len(bad)}, res


def check_jplus(real, mut):
    out = {}
    ok = True
    for wf in NODES:
        rw = real[wf]
        pl = mut.prof(rw["jplus"])
        rho = {M: pl[M] / rw["chi"][M] for M in MS}
        held = rw["jplus_M_min"] == N23 and profile_ok(pl, rw["chi"])
        rel = rw["R5"] and rw["jplus_M_min"] == N23 and profile_ok(pl, rw["chi"])   # the real zero is its located zero
        one = all(abs(raw / rw["chi"][M] - 1) <= 1e-9 for M, raw in rw["jplus"].items())
        anchor = wf != WF0 or all(abs(rw["chi"][M] - PUB_REAL_WF0[M]) <= 0.5 for M in MS)
        out[str(wf)] = {"rho": rho, "held": held, "reloc": rel, "ratio_one": one, "anchor": anchor}
        ok = ok and held and rel and one and anchor
    return {"pass": ok, "nodes": out}


def check_edge(real, mut):
    base = real[WF0]["chi"]
    cases = [(F * (1 - 1e-9), True), (F * (1 + 1e-9), False), ((1 / F) * (1 + 1e-9), True),
             ((1 / F) * (1 - 1e-9), False), (10.0, False), (0.1, False), (1.0, True)]
    bad = []
    for M in MS:
        for rho, inside in cases:
            chi = dict(base)
            chi[M] = base[M] * rho
            if profile_ok(mut.prof(chi), base) != inside:
                bad.append(f"M {M} rho {rho:.6g}: expected {'inside' if inside else 'outside'}")
    return {"pass": not bad, "bad": bad[:12], "n_bad": len(bad)}


# ---------------------------------------------------------------- the verdict, the tables
def pos_cell(v):
    ng, nu = v["n_groups_pass"], v["n_groups_unknown"]
    if v["status"] == "pass":
        return f"P{ng}" + (f"+{nu}?" if nu else "")
    return f"?{nu}" if v["status"] == "unknown" else "-"


def print_map(title, pr, D):
    say(f"\n{title}  (Pn: n of 6 groups pass; +k?: k groups unknown; ?k: unknown only; -: fail; |Q| at wf 1.00)")
    say("| b \\ a | " + " | ".join(f"{a:+d}" for a in LAT_A) + " |")
    say("|---|" + "---|" * len(LAT_A))
    by_ab = {D.pos_Q[p][0]: (p, v) for p, v in pr.items()}
    for b in reversed(LAT_B):
        cells = []
        for a in LAT_A:
            p, v = by_ab[(a, b)]
            cells.append(f"{pos_cell(v)} ({D.pos_Q[p][1]:.2f})")
        say(f"| {b:+d} | " + " | ".join(cells) + " |")


def summarize(name, base, pr, D, real, mut):
    lat0, lat = lattice(base, D), lattice(pr, D)
    st = {ab: v["status"] for ab, v in ((D.pos_Q[p][0], v) for p, v in pr.items())}
    was = lat0["passing"]
    out = {"variant": name, "n_718_pass": len(was), "pass": len(lat["passing"]), "unknown": len(lat["unknown"]),
           "718_pass_now_pass": sum(1 for ab in was if st[ab] == "pass"),
           "718_pass_now_unknown": sorted(ab for ab in was if st[ab] == "unknown"),
           "718_pass_now_fail": sorted(ab for ab in was if st[ab] == "fail"),
           "edge": lat["edge"], "candidates": lat["candidates"], "candidates_undecided": lat["candidates_undecided"]}
    absq = {D.pos_Q[p][0]: D.pos_Q[p][1] for p in pr}
    out["fail_at_absQ_ge_2.48"] = sorted(ab for ab in was if st[ab] == "fail" and absq[ab] >= 2.48)
    out["unknown_at_absQ_ge_2.48"] = sorted(ab for ab in was if st[ab] == "unknown" and absq[ab] >= 2.48)
    out["narrows"] = bool(out["718_pass_now_fail"])
    out["brackets"] = bool(out["718_pass_now_unknown"])
    return out, lat


def log10(x):
    return math.log10(x)


def median(x):
    s = sorted(x)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def ranks(x):
    idx = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            r[idx[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def context(D, real, mut):
    """reported, not judged: the profile of the plants that pass (a) and (c') or (c'') at wf 1.00"""
    rw = real[WF0]["chi"]
    out = {}
    for name, crit in (("held", "c"), ("relocated", "c2")):
        rows = []
        for (pid, wf), r in D.rec.items():
            if wf != WF0 or not r["a"]:
                continue
            if crit == "c" and r["c"]:
                chi = r["chi"]
            elif crit == "c2" and r["c2"] and r["reloc"] is not None:
                chi = r["reloc"]["chi"]
            else:
                continue
            rho = {M: chi[M] / rw[M] for M in MS}
            rows.append({"pos": r["pos"], "key": (r["pos"], gkey(r["kind"], r["L"], r["P"]), r["alpha"], r["phase"]),
                         "eta": r["eta"], "absQ": D.pos_Q[r["pos"]][1], "rho": rho,
                         "s20": rho[20] / rho[22], "s21": rho[21] / rho[22]})
        per = {}
        for x in rows:
            per.setdefault(x["pos"], []).append(x)
        table = {}
        for p, xs in per.items():
            table[p] = {"n": len(xs), "absQ": xs[0]["absQ"],
                        **{f"rho{M}": [min(x["rho"][M] for x in xs), max(x["rho"][M] for x in xs)] for M in MS},
                        "s20": [min(x["s20"] for x in xs), median([x["s20"] for x in xs]), max(x["s20"] for x in xs)],
                        "s21": [min(x["s21"] for x in xs), median([x["s21"] for x in xs]), max(x["s21"] for x in xs)]}
        # shape against eta at a fixed position, group, alpha and phase
        sets = {}
        for x in rows:
            sets.setdefault(x["key"], []).append(x)
        dd = {"s20": [], "s21": []}
        for xs in sets.values():
            if len(xs) >= 2:
                for s in dd:
                    v = [log10(x[s]) for x in xs]
                    dd[s].append(max(v) - min(v))
        # shape against |Q|, over every plant, and within the plants near the real chi_22 (0.5 <= rho_22 <= 2)
        sp = {}
        for lab, sel in (("all", rows), ("rho22_0.5_2", [x for x in rows if 0.5 <= x["rho"][22] <= 2])):
            if len(sel) > 2:
                sp[lab] = {"n": len(sel), "rho_s20_absQ": spearman([x["absQ"] for x in sel], [log10(x["s20"]) for x in sel]),
                           "rho_s21_absQ": spearman([x["absQ"] for x in sel], [log10(x["s21"]) for x in sel]),
                           "log10_s20": [min(log10(x["s20"]) for x in sel), max(log10(x["s20"]) for x in sel)],
                           "log10_s21": [min(log10(x["s21"]) for x in sel), max(log10(x["s21"]) for x in sel)]}
        out[name] = {"n_plants": len(rows), "per_position": table,
                     "eta_spread": {s: ({"n_sets": len(v), "median": median(v), "max": max(v)} if v else None)
                                    for s, v in dd.items()}, "vs_absQ": sp}
    return out


def post_freeze(D, real, mut, verdict_pos):
    """written after the verdict was read, so labeled so, and not frozen: how close each position's plants come to the real
    profile, how far the profile drifts across nodes, and what it would take to decide the unknown statuses"""
    out = {}
    lf = math.log10(F)
    for name, crit in (("held", "c"), ("relocated", "c2")):
        pos_best, per_pos = {}, {}
        for (pid, wf), r in D.rec.items():
            if wf != WF0 or not r["a"]:
                continue
            chi = r["chi"] if crit == "c" else (r["reloc"]["chi"] if r["reloc"] is not None else None)
            if chi is None or not (r["c"] if crit == "c" else r["c2"]):
                continue
            d = max(abs(log10(chi[M] / real[wf]["chi"][M])) for M in MS)
            pos_best[r["pos"]] = min(pos_best.get(r["pos"], math.inf), d)
            pp = per_pos.setdefault(r["pos"], {"n_ac": 0, "n_d": 0})
            pp["n_ac"] += 1
            pp["n_d"] += d <= lf
        # drift of rho_M between a node and wf 1.00, over the plants run at both
        drift = []
        for (pid, wf), r in D.rec.items():
            if wf == WF0:
                continue
            r0 = D.rec.get((pid, WF0))
            if r0 is None:
                continue
            c1 = r["chi"] if crit == "c" else (r["reloc"]["chi"] if r["reloc"] is not None else None)
            c0 = r0["chi"] if crit == "c" else (r0["reloc"]["chi"] if r0["reloc"] is not None else None)
            if c1 is None or c0 is None:
                continue
            drift.append(max(abs(log10((c1[M] / real[wf]["chi"][M]) / (c0[M] / real[WF0]["chi"][M]))) for M in MS))
        drift.sort()
        # the unknown plants: candidates at wf 1.00 with a missing node and no failing record
        unk_plants, unk_runs, margin = 0, 0, {}
        for (p, g), pids in D.groups.items():
            for pid in pids:
                if plant_status(D, pid, crit, True, real, mut) == "unknown":
                    unk_plants += 1
                    unk_runs += sum(1 for wf in (0.99, 1.01) if (pid, wf) not in D.rec)
                    r = D.rec[(pid, WF0)]
                    chi = r["chi"] if crit == "c" else r["reloc"]["chi"]
                    d = max(abs(log10(chi[M] / real[WF0]["chi"][M])) for M in MS)
                    margin[p] = max(margin.get(p, -math.inf), lf - d)
        unk_pos = [p for p, v in verdict_pos[crit].items() if v["status"] == "unknown"]
        dmax = drift[-1] if drift else math.nan
        out[name] = {"best_dist_by_pos": pos_best, "n_by_pos": per_pos,
                     "drift": {"n": len(drift), "median": median(drift) if drift else None,
                               "p90": drift[int(0.9 * (len(drift) - 1))] if drift else None, "max": dmax},
                     "unknown_plants": unk_plants, "unknown_runs_upper": unk_runs,
                     "unknown_positions": len(unk_pos),
                     "unknown_positions_margin_gt_maxdrift": sorted(p for p in unk_pos if margin.get(p, -1) > dmax),
                     "unknown_positions_margin": {p: margin.get(p) for p in unk_pos}}
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("raw_dir", help="I718-mac/ of the raw-data repository")
    ap.add_argument("--out", help="directory for profile.json")
    ap.add_argument("--mutate", default=None)
    args = ap.parse_args()
    mut = Mut(args.mutate)
    say(f"misfit_profile_746: raw {args.raw_dir}, window F = {F}, mutate {args.mutate}")
    ring = json.load(open(os.path.join(args.raw_dir, "ring.json")))
    ctl, real = load_real(args.raw_dir)
    D = Data(args.raw_dir, mut)
    say(f"read {D.n_lines} lines, {len(D.rec)} plant records, {len(D.groups)} position-groups in {D.seconds:.1f} s")

    checks = {}
    rec = check_records(D, real, mut)
    checks["records"] = rec["records_ok"]
    checks["R-b"] = rec["Rb"]
    checks["counts"] = rec["counts_ok"]
    say(f"records per node {rec['records']}, duplicates {D.dup}; counts at wf 1.00 {rec['counts']} "
        f"(published {PUB_COUNTS}); R-b mismatches {len(rec['Rb_bad'])}")
    ra, base = check_Ra(D, ring, real, mut)
    checks["R-a"] = ra["pass"]
    say(f"R-a (loader with (d) off against ring.json): {'OK' if ra['pass'] else 'FAIL'}"
        + ("" if ra["pass"] else f", {ra['n_bad']} differences, first: {ra['bad'][:3]}"))
    jp = check_jplus(real, mut)
    checks["J+"] = jp["pass"]
    say("J+ (the real profile against itself): " + ("OK" if jp["pass"] else "FAIL")
        + "; " + "; ".join(f"wf {wf}: held {v['held']}, reloc {v['reloc']}, rho {[round(v['rho'][M], 6) for M in MS]}, "
                           f"anchor {v['anchor']}" for wf, v in jp["nodes"].items()))
    ed = check_edge(real, mut)
    checks["edge"] = ed["pass"]
    say(f"edge check (planted rho at F, 1/F, 10, 1/10, 1 for each M): {'OK' if ed['pass'] else 'FAIL'}"
        + ("" if ed["pass"] else f", {ed['n_bad']} wrong, first: {ed['bad'][:3]}"))

    result = {"F": F, "mutate": args.mutate, "checks": checks, "records": rec["records"], "counts": rec["counts"],
              "R-a": ra, "J+": jp, "edge": ed}
    failed = [k for k, v in checks.items() if not v]
    expect = {"scale_chi": {"J+", "edge"}, "drop_node": {"R-a"}}.get(mut.kind)
    if mut.kind:
        fired = bool(expect) and expect <= set(failed)
        say(f"mutation {args.mutate}: failed checks {failed}; expected {sorted(expect)} -> "
            f"{'FIRES' if fired else 'DOES NOT FIRE'}")
        result["mutation_fires"] = fired
        code = 1 if fired else 2
    elif failed:
        say(f"controls FAILED: {failed}; no verdict (the stage is gated)")
        code = 1
    else:
        code = 0
    if code == 0:
        res = {}
        for name, crit, label in (("P'", "c", "(a)+(c')+(d')"), ("P''", "c2", "(a)+(c'')+(d'')")):
            pr = analyze(D, crit, True, real, mut)
            summ, lat = summarize(name, base[crit], pr, D, real, mut)
            print_map(f"{name} = {label}, the profile on", pr, D)
            print_map(f"#718's {'(a)+(c)' if crit == 'c' else '(a)+(c2)'} for comparison (profile off)", base[crit], D)
            say(f"\n{name}: positions pass {summ['pass']}, unknown {summ['unknown']}, of 42; of #718's "
                f"{summ['n_718_pass']} passing: still pass {summ['718_pass_now_pass']}, unknown "
                f"{len(summ['718_pass_now_unknown'])}, fail {len(summ['718_pass_now_fail'])}")
            say(f"  fail: {[pid_str(ab) for ab in summ['718_pass_now_fail']]}")
            say(f"  unknown: {[pid_str(ab) for ab in summ['718_pass_now_unknown']]}")
            say(f"  at |Q| >= 2.48: fail {len(summ['fail_at_absQ_ge_2.48'])}, unknown {len(summ['unknown_at_absQ_ge_2.48'])}; "
                f"edge-touching passing {len(summ['edge'])}; candidates {[pid_str(ab) for ab in summ['candidates']]}; "
                f"undecided {[pid_str(ab) for ab in summ['candidates_undecided']]}")
            res[name] = {"summary": summ, "positions": {p: {k: v for k, v in x.items()} for p, x in pr.items()}}
        result["verdict"] = res
        say("\n--- post-freeze (after the verdict was read; not frozen, not judged) ---")
        pf = post_freeze(D, real, mut, {"c": res["P'"]["positions"], "c2": res["P''"]["positions"]})
        result["post_freeze"] = pf
        for name, v in pf.items():
            say(f"\n{name}: the closest a plant at each position comes to the real profile, max_M |log10 rho_M| over the "
                f"plants passing (a) and (c{'′' if name == 'held' else '″'}) at wf 1.00 (window edge {math.log10(F):.3f})")
            say("| b \\ a | " + " | ".join(f"{a:+d}" for a in LAT_A) + " |")
            say("|---|" + "---|" * len(LAT_A))
            by_ab = {D.pos_Q[p][0]: p for p in D.pos_Q}
            for b in reversed(LAT_B):
                cells = []
                for a in LAT_A:
                    p = by_ab[(a, b)]
                    d = v["best_dist_by_pos"].get(p)
                    cells.append("-" if d is None else f"{d:.2f} ({v['n_by_pos'][p]['n_d']}/{v['n_by_pos'][p]['n_ac']})")
                say(f"| {b:+d} | " + " | ".join(cells) + " |")
            dr = v["drift"]
            say(f"drift of rho_M between wf 1.00 and the other nodes over {dr['n']} plants run at both: median "
                f"{dr['median']:.3f}, 90th percentile {dr['p90']:.3f}, max {dr['max']:.3f} decades")
            say(f"unknown: {v['unknown_plants']} plants at {v['unknown_positions']} positions, up to "
                f"{v['unknown_runs_upper']} plant-node runs to decide them; positions whose best candidate's margin "
                f"exceeds the largest drift: {len(v['unknown_positions_margin_gt_maxdrift'])} of {v['unknown_positions']}")
        ctx = context(D, real, mut)
        result["context"] = ctx
        for name, c in ctx.items():
            say(f"\ncontext, {name}: {c['n_plants']} plants pass (a) and (c{'′' if name == 'held' else '″'}) at wf 1.00; "
                f"shape at fixed position, group, alpha and phase over eta: {c['eta_spread']}")
            for lab, s in c["vs_absQ"].items():
                say(f"  vs |Q| ({lab}, n {s['n']}): Spearman log10(chi20/chi22 / real's) {s['rho_s20_absQ']:+.3f}, "
                    f"log10(chi21/chi22 / real's) {s['rho_s21_absQ']:+.3f}; log10 ranges {s['log10_s20']}, {s['log10_s21']}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "profile.json"), "w") as f:
            json.dump(result, f, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    say(f"exit {code}")
    sys.exit(code)


if __name__ == "__main__":
    main()
