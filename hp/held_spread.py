#!/usr/bin/env python3
"""The held fit's degree spread past D 248 against the chain's departure (#904): Delta(d6) - Delta(d10) from #608's
two chains, each evaluated from its own series (the stored jets) at matched D, against #603's chain - series there.

    cd hp && python3 held_spread.py run RAW [--out FILE.json] [--mutate base|fit]

Each chain is its states (anm_pin3_D*.npz) and jets (jet_pin3_D*.npz, saved at a base state) in its folders, in order
of D. Delta(D) is the jet whose step covers D (largest base <= D, up to the next state's D): sum_k z_k[Delta] u^k,
u = D - D_base, each coefficient its eight parts summed exactly; the truncation estimate is the last two terms. The
targets are #603's state D from 248 to the d6 chain's end, where #805's table (I805-mac/table.json) has chain - series.

Checks (exit 1 on failure): C1, each jet at its step's end against the next state's stored Delta, |diff| <= 1e-20;
C2, every state's held values (band_coeffs, exact, all eight parts) follow its chain's own fit from the chain's first
state, held(state) - held(first) = fit(D) - fit(D_first), to 1e-20. Mutations, each must fail: base (every jet's base
D moved by 1e-6, which C1 sees), fit (each chain checked against the other degree's fit, which C2 sees).
Acceptance of #904: A1, r = (d6 - d10)/(chain - series) at the targets in 251-253.43 all within 30% of their median;
A2, d6 - d10 at 250.1676 within a factor 2 of #905's adjoint, -1.683e-9 (Delta(d6) - Delta(d10); its shares
psi^T dF were +4.33e-10 and -1.25e-9, the negatives of the Delta shifts).
"""
import glob
import json
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHAINS = {
    "d6": dict(fit="held_fit_d6.json", other="held_fit_d10.json",
               folders=["I608-dh610/restart2_d6", "I836-dh370/drop", "I608-dh610/restart4_d6"]),
    "d10": dict(fit="held_fit_d10.json", other="held_fit_d6.json",
                folders=["I608-dh610/restart2_d10", "I608-dh610/restart3_d10"]),
}
ADJOINT = mp.mpf("-1.683e-9")
D_A2 = "250.16758165359497"
C1_BAR, C2_BAR = mp.mpf("1e-20"), mp.mpf("1e-20")


def say(s):
    print(s, flush=True)


def d_of(meta):
    return mp.mpf(meta["D_str"]) if "D_str" in meta else mp.mpf(float(meta["D"]))


def state_delta(path):
    from octo_corrector import load_o
    z = load_o(path)[0]
    return mp.fsum(mp.mpf(float(np.asarray(p)[-1])) for p in z)


def jet_coeffs(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    P = int(meta.get("parts", 8))
    N = d["z0"].shape[0] - 1
    c = [mp.fsum(mp.mpf(float(d[f"z{q}"][k][-1])) for q in range(P)) for k in range(N + 1)]
    return d_of(meta), c, meta


def horner(c, u):
    v = mp.mpf(0)
    for x in reversed(c):
        v = v * u + x
    return v


def held_of(path):
    from anm_run import load_state
    from octo_corrector import load_o
    from pinning import band_coeffs
    zh, zl, m = load_state(path)
    z8 = load_o(path)[0]
    M, kf = int(m["M"]), int(m["kfreeze"])
    ks = [k for k in range(kf + 1, 14) if k % 2 == 1]
    return ks, band_coeffs(zh[:M], zl[:M], M, ks, exact=True, rest=[np.asarray(p)[:M] for p in z8[2:]])


class Chain:
    def __init__(self, raw, name, mutate=""):
        cfg = CHAINS[name]
        self.name = name
        self.fit = cfg["other"] if mutate == "fit" else cfg["fit"]
        self.states, self.jets = [], []
        for f in cfg["folders"]:
            for p in sorted(glob.glob(os.path.join(raw, f, "anm_pin3_D*.npz"))):
                from anm_run import load_state
                D = d_of(load_state(p)[2])
                self.states.append((D, p))
            for p in sorted(glob.glob(os.path.join(raw, f, "jet_pin3_D*.npz"))):
                Db, c, meta = jet_coeffs(p)
                if meta.get("held_fit") and os.path.basename(meta["held_fit"]) != cfg["fit"]:
                    raise SystemExit(f"{p}: its held fit {meta['held_fit']} is not the {name} chain's")
                if mutate == "base":
                    Db += mp.mpf("1e-6")
                self.jets.append((Db, c, p))
        self.states.sort(key=lambda t: t[0])
        self.jets.sort(key=lambda t: t[0])
        seen, uniq = set(), []
        for D, p in self.states:
            key = mp.nstr(D, 25)
            if key not in seen:
                seen.add(key)
                uniq.append((D, p))
        self.states = uniq
        self.end = self.states[-1][0]

    def next_state(self, D):
        for Ds, p in self.states:
            if Ds > D + mp.mpf("1e-12"):
                return Ds, p
        return None

    def delta(self, D):
        """(Delta, truncation estimate, base) from the jet whose step covers D"""
        best = None
        for Db, c, p in self.jets:
            if Db <= D + mp.mpf("1e-30"):
                best = (Db, c, p)
        if best is None:
            raise ValueError(f"{self.name}: no jet at or below D {mp.nstr(D, 12)}")
        Db, c, p = best
        nx = self.next_state(Db)
        if nx is not None and D > nx[0] + mp.mpf("1e-12"):
            raise ValueError(f"{self.name}: D {mp.nstr(D, 12)} past the step of the jet at {mp.nstr(Db, 12)}")
        u = D - Db
        tr = abs(c[-1] * u ** (len(c) - 1)) + abs(c[-2] * u ** (len(c) - 2))
        return horner(c, u), tr, Db

    def c1(self):
        """each jet at the next state's D against that state's Delta"""
        out = []
        for Db, c, p in self.jets:
            nx = self.next_state(Db)
            if nx is None:
                continue
            Dn, pn = nx
            v = horner(c, Dn - Db)
            out.append(dict(base=mp.nstr(Db, 15), next=mp.nstr(Dn, 15), diff=float(v - state_delta(pn)), u=float(Dn - Db)))
        return out

    def c2(self):
        """every state's held values against its chain's fit increment from the first state"""
        from pinning import HeldFit
        f = HeldFit(os.path.join(HERE, "data", self.fit))
        D0, p0 = self.states[0]
        ks, h0 = held_of(p0)
        worst = []
        for D, p in self.states[1:]:
            _, h = held_of(p)
            inc = f.increment(D0, D, ks)
            e = max(max(abs(mp.mpf(a) - mp.mpf(a0) - da), abs(mp.mpf(b) - mp.mpf(b0) - db))
                    for (a, b), (a0, b0), (da, db) in zip(h, h0, inc))
            worst.append(dict(D=mp.nstr(D, 15), err=float(e), path=os.path.relpath(p)))
        return worst


def cmd_run(argv):
    pos = [a for a in argv if not a.startswith("--")]
    opt = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in argv if a.startswith("--"))
    raw = pos[0]
    mut = opt.get("mutate", "")
    mp.mp.dps = 60
    ch = {n: Chain(raw, n, mut) for n in CHAINS}
    ok = True
    out = dict(mutate=mut, chains={}, targets=[])
    for n, c in ch.items():
        r1 = c.c1()
        m1 = max(abs(x["diff"]) for x in r1)
        r2 = c.c2()
        m2 = max(x["err"] for x in r2)
        g1, g2 = m1 <= C1_BAR, m2 <= C2_BAR
        ok &= bool(g1 and g2)
        out["chains"][n] = dict(states=len(c.states), jets=len(c.jets), end=mp.nstr(c.end, 15), fit=c.fit, c1=r1, c2=r2,
                                c1_max=m1, c2_max=m2)
        say(f"{n}: {len(c.states)} states, {len(c.jets)} jets, to D {mp.nstr(c.end, 12)}; C1 max|jet - next state| {m1:.2e} "
            f"over {len(r1)} steps ({'pass' if g1 else 'FAIL'}); C2 held values against {c.fit}, max {m2:.2e} over "
            f"{len(r2)} states ({'pass' if g2 else 'FAIL'})")
    if mut:
        say(f"mutation {mut}: checks {'pass (the mutation is NOT caught)' if ok else 'fail, as they must'}")
        return 1 if ok else 0
    tab = json.load(open(os.path.join(raw, "I805-mac", "table.json")))
    end = min(c.end for c in ch.values())
    rows = [r for r in tab if r.get("chain_minus_series") is not None and 248 <= r["D"] <= float(end)]
    say("   D             d6 - d10        trunc6   trunc10   chain - series   r = (d6 - d10)/dep")
    for r in rows:
        D = mp.mpf(repr(r["D"]))
        v6, t6, _ = ch["d6"].delta(D)
        v10, t10, _ = ch["d10"].delta(D)
        dd = v6 - v10
        dep = r["chain_minus_series"]
        ratio = float(dd) / dep
        out["targets"].append(dict(D=r["D"], d6_minus_d10=float(dd), trunc6=float(t6), trunc10=float(t10), departure=dep,
                                   ratio=ratio, min_pivot=r.get("min_pivot")))
        say(f"  {r['D']:<13.8f} {float(dd):+.6e}  {float(t6):.1e}  {float(t10):.1e}  {dep:+.6e}   {ratio:+.4f}")
    win = [t for t in out["targets"] if 251 <= t["D"] <= float(end)]
    rs = sorted(t["ratio"] for t in win)
    med = rs[len(rs) // 2] if len(rs) % 2 else 0.5 * (rs[len(rs) // 2 - 1] + rs[len(rs) // 2])
    a1 = all(abs(x - med) <= 0.3 * abs(med) for x in rs)
    spread = max(abs(x) for x in rs) / min(abs(x) for x in rs) if all(x != 0 for x in rs) else float("inf")
    same = all(x * med > 0 for x in rs)
    out["A1"] = dict(n=len(rs), median=med, ratios=rs, within_30=a1, max_over_min=spread, same_sign=same)
    say(f"A1: {len(rs)} targets in 251-{mp.nstr(end, 8)}, median r {med:+.4f}, max|r|/min|r| {spread:.2f}, signs "
        f"{'all equal' if same else 'MIXED'} -> {'constant to 30%' if a1 else 'NOT constant to 30%'}"
        + ("" if a1 else (", drift above a factor 3" if spread > 3 or not same else ", drift within a factor 3")))
    v6, t6, _ = ch["d6"].delta(mp.mpf(D_A2))
    v10, t10, _ = ch["d10"].delta(mp.mpf(D_A2))
    dd = v6 - v10
    q = dd / ADJOINT
    a2 = bool(0.5 <= q <= 2)
    out["A2"] = dict(D=D_A2, d6_minus_d10=float(dd), adjoint=float(ADJOINT), ratio=float(q), within_2=a2)
    say(f"A2: at D {D_A2} d6 - d10 {float(dd):+.6e} against the adjoint {float(ADJOINT):+.3e}: ratio {float(q):+.3f} -> "
        f"{'within a factor 2' if a2 else 'NOT within a factor 2'}")
    if "out" in opt:
        json.dump(out, open(opt["out"], "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        sys.exit(cmd_run(sys.argv[2:]))
    raise SystemExit(__doc__)
