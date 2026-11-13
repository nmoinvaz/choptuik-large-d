#!/usr/bin/env python3
"""The residual gauge rate lam_g = 1 + <c1> = 1 - (n - 1)<a_h^2 - 1> of a stored state, in mpmath from the state's
parts, without f64 rounding (notes/MODE_COUNT.md sections 1 and 10). By the map between the gauges (notes/
RT_FORMULATION.md section 2) lam_g = 4 pi mu / Delta, with mu Reiterer and Trubowitz's echoing parameter; at D = 4 it
is the surface gravity nu_SH of the self-similarity horizon of the 2025 review (notes/REVIEW_GHMG25.md row 3).

a_h = a(tau, 1) is the block at off_ah = 3M + 4MK of the state (hp/src/model.rs, Ctx::off_ah), Delta the last entry,
n = D - 2 from meta D_str when the state has it (the chain's D in mpmath precision), else from the float D, which is
then exact. Every part of the state (zh, zl and z2..z7 where present) is summed exactly at PREC bits. <.> is the mean
over the M points of the half tau grid. It is the exact average of a_h^2 when a_h^2 has no harmonic at or above M on
that grid, so it differs from the continuum average by aliasing at the state's truncation level.

    python3 hp/lam_gauge.py show STATE.npz ...           lam_g, Delta and mu = lam_g Delta / (4 pi) of each state
    python3 hp/lam_gauge.py rt STATE_D4.npz              acceptance 1: lam_g against 4 pi mu_RT / (2 K_RT)
    python3 hp/lam_gauge.py identity RAW                 acceptance 2: lam_g against 4 pi mu / Delta of RT states
    python3 hp/lam_gauge.py f64 [--live]                 modecount_common's f64 lam_g against this one
    python3 hp/lam_gauge.py table RAW [--out CSV]        lam_g(D) on the stored states (hp/data/lam_gauge.csv)

RAW is a checkout of the raw-data repository (git clone https://github.com/nmoinvaz/choptuik-large-d-raw).
--mutate=n uses n in place of n - 1 in lam_g. rt, identity and f64 must then exit 1. Each check exits 1 when it fails.

rt (pass criterion fixed before the run): |lam_g - 4 pi mu_RT / (2 K_RT)| <= --tol (1e-20, the acceptance of #622), on
mu_check.py's state RAW/T45-win/hp/data/lowD_states/d4hm15_D4.0000_M192.npz. mu_RT and K_RT are Reiterer and
Trubowitz's enclosures (arXiv:1203.3766v1, radius 1e-80), from solver/rt/mu_check.py.

identity (pass criterion fixed before the run): at D = 15, 20.043653 and 40.020065 our double-double state and the RT
prototype's 32 x 121 state (solver/rt/, notes/RT_FORMULATION.md sections 9 and 10) give lam_g and lam_RT = 4 pi mu /
Delta, mu from the RT state's two parts and Delta from its run record (Delta_str, 30 digits, arb at 192 bits). The
states' precision is taken as their measured agreement in Delta, e_D = |Delta_RT - Delta_hp| / Delta_hp, and the check
passes when |lam_g - lam_RT| / lam_g <= 10 max(e_D, 1e-28) at all three D. The two codes share no equations,
unknowns or discretization (RT_FORMULATION.md 9.4 lists what they share), so this checks the identity, not the formula.
The second RT grid at each D (48 x 101, or 32 x 141 at D = 40) is printed as the RT side's own spread.

f64: hp/modecount_common.py forms st.lam_g = 1 + float(np.mean(c1)) in f64. The recorded values (hp/data/
modecount_check.jsonl, modecount_results.jsonl, on the unresampled states) and, with --live, setup() run on three states
(its complex-lambda solver stubbed, so only hpkernel is needed) must agree with lam_g here within the f64 bound
u (8 (n - 1) + M max|c1| + 4), u = 2^-52.
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PARTS = ("zh", "zl", "z2", "z3", "z4", "z5", "z6", "z7")
PREC = 1400          # bits: eight float parts span at most ~1100 bits, so their sum is exact
DIG = 40             # digits printed and written
U = 2.0 ** -52


def mutation(argv):
    for a in argv:
        if a.startswith("--mutate="):
            return a.split("=", 1)[1]
    return None


def opt(argv, name, default):
    for a in argv:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


# ---------------------------------------------------------------------------------------------------- the state


def load(path):
    """(parts, meta): the state's float parts, highest first, and its meta"""
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    parts = [np.asarray(d[k], dtype=float) for k in PARTS if k in d.files]
    return parts, meta


def layout(parts, meta):
    """(M, K, off_ah); refuses a vector whose length is not hp's 4M(K + 1) + 1"""
    M, K = int(meta["M"]), len(meta["nodes"])
    L = len(parts[0])
    if parts[0].ndim != 1 or L != 4 * M * (K + 1) + 1:
        raise ValueError(f"not hp's layout: length {L}, M {M}, K {K}")
    return M, K, 3 * M + 4 * M * K


def n_of(meta):
    with mp.workprec(PREC):
        D = mp.mpf(meta["D_str"]) if "D_str" in meta else mp.mpf(meta["D"])
        return D - 2


def exact(parts, i):
    with mp.workprec(PREC):
        s = mp.mpf(0)
        for p in parts:
            s += mp.mpf(float(p[i]))
        return s


def lam_g(parts, meta, mutate=None):
    """1 - k <a_h^2 - 1>, k = n - 1 (n with mutate = 'n'), at PREC bits"""
    M, K, off = layout(parts, meta)
    n = n_of(meta)
    with mp.workprec(PREC):
        ah = [exact(parts, off + j) for j in range(M)]
        avg = mp.fsum([(a - 1) * (a + 1) for a in ah]) / M
        k = n if mutate == "n" else n - 1
        return 1 - k * avg


def delta(parts):
    return exact(parts, len(parts[0]) - 1)


def lam_g_f64(parts, meta):
    """modecount_common.setup's formula, in f64 as it is there (zh + zl, numpy mean)"""
    M, K, off = layout(parts, meta)
    zl = parts[1] if len(parts) > 1 else np.zeros_like(parts[0])
    ah = np.asarray(parts[0][off:off + M]) + np.asarray(zl[off:off + M])
    n = meta["D"] - 2
    c1 = (n - 1) * (1 - ah ** 2)
    return 1 + float(np.mean(c1)), float(np.abs(c1).max())


def f64_bound(meta, c1max):
    return U * (8 * (meta["D"] - 3) + int(meta["M"]) * c1max + 4)


def s(x, d=DIG):
    return mp.nstr(x, d, strip_zeros=False)


def info(path, mutate=None):
    parts, meta = load(path)
    M, K, _ = layout(parts, meta)
    with mp.workprec(PREC):
        lg = lam_g(parts, meta, mutate)
        De = delta(parts)
        mu = lg * De / (4 * mp.pi)
    return dict(parts=parts, meta=meta, M=M, K=K, lam=lg, Delta=De, mu=mu)


def cmd_show(argv):
    mut = mutation(argv)
    for p in [a for a in argv if not a.startswith("--")]:
        r = info(p, mut)
        m = r["meta"]
        print(f"{p}: D {m.get('D_str', m['D'])} M {r['M']} K {r['K']} parts {len(r['parts'])}")
        print(f"  lam_g {s(r['lam'])}\n  Delta {s(r['Delta'])}\n  mu    {s(r['mu'])}"
              f"{'  (mutation n)' if mut else ''}")
    return 0


# ---------------------------------------------------------------------------------------------------- acceptance 1


def rt_constants():
    sys.path.insert(0, os.path.join(REPO, "solver", "rt"))
    import mu_check   # sets mp.mp.dps = 40 and parses MU_RT, K_RT at 40 digits
    return mu_check.MU_RT, mu_check.K_RT


def cmd_rt(argv):
    mut = mutation(argv)
    tol = mp.mpf(opt(argv, "tol", "1e-20"))
    paths = [a for a in argv if not a.startswith("--")]
    if len(paths) != 1:
        sys.exit("usage: lam_gauge.py rt STATE_D4.npz [--tol=1e-20] [--mutate=n]")
    MU, K = rt_constants()
    mp.mp.dps = 50
    r = info(paths[0], mut)
    if r["meta"]["D"] != 4.0:
        sys.exit(f"rt: the state is at D {r['meta']['D']}, not 4")
    lam_rt = 4 * mp.pi * MU / (2 * K)
    d = r["lam"] - lam_rt
    ok = abs(d) <= tol
    print(f"{paths[0]}: D 4 M {r['M']} parts {len(r['parts'])}")
    print(f"  lam_g              {s(r['lam'])}{'  (mutation n)' if mut else ''}")
    print(f"  4 pi mu_RT/(2 K_RT) {s(lam_rt)}")
    print(f"  difference {mp.nstr(d, 3)}  {'ok' if ok else 'FAIL'} (tol {mp.nstr(tol, 3)})")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------- acceptance 2

# (D, our state (repo path), RT state and its record (raw paths), a second RT grid and its record)
IDENTITY = [
    (15.0, "hp/data/states/up_D15.0000.npz",
     "I105-mac/controls/res_32x121_D15.0000_D15.0000_32x121.npz", "I105-mac/controls/res_32x121_D15.0000.jsonl",
     "I105-mac/controls/res_48x101_D15.0000_D15.0000_48x101.npz", "I105-mac/controls/res_48x101_D15.0000.jsonl"),
    (20.043653, "hp/data/states/up2_D20.0437.npz",
     "I105-mac/controls/res_32x121_D20.0437_D20.0437_32x121.npz", "I105-mac/controls/res_32x121_D20.0437.jsonl",
     "I105-mac/controls/res_48x101_D20.0437_D20.0437_48x101.npz", "I105-mac/controls/res_48x101_D20.0437.jsonl"),
    (40.020065, "hp/data/states/up7_D40.0201.npz",
     "I106-mac/controls/C_32x121_D40.0201_D40.0201_32x121.npz", "I106-mac/controls/C_32x121_D40.0201.jsonl",
     "I145-mac/controls/from101_32x141_D40.0201_32x141.npz", "I145-mac/controls/from101_32x141.jsonl"),
]
FACTOR = 10
FLOOR = mp.mpf("1e-28")


def rt_lam(raw, npz, rec):
    """(lam_RT, mu, Delta, grid) of an RT prototype state: mu = z[-1] in two parts, Delta from the record's Delta_str"""
    d = np.load(os.path.join(raw, npz))
    D, grid = float(d["D"]), [int(d["Nt"]), int(d["N"])]
    rows = [json.loads(x) for x in open(os.path.join(raw, rec)) if x.strip()]
    rows = [x for x in rows if x["D"] == D and x["grid"] == grid and "Delta_str" in x]
    if len(rows) != 1:
        raise RuntimeError(f"{rec}: {len(rows)} records at D {D} grid {grid}")
    with mp.workprec(PREC):
        mu = mp.mpf(float(d["zh"][-1])) + mp.mpf(float(d["zl"][-1]))
        De = mp.mpf(rows[0]["Delta_str"])
        return 4 * mp.pi * mu / De, mu, De, grid, D


def cmd_identity(argv):
    mut = mutation(argv)
    paths = [a for a in argv if not a.startswith("--")]
    if len(paths) != 1:
        sys.exit("usage: lam_gauge.py identity RAW [--mutate=n]")
    raw = paths[0]
    mp.mp.dps = 50
    ok_all = True
    print(f"criterion: |lam_g - lam_RT| / lam_g <= {FACTOR} max(|Delta_RT - Delta_hp| / Delta_hp, {mp.nstr(FLOOR, 2)})"
          f"{'  (mutation n)' if mut else ''}")
    for D, ours, npz, rec, npz2, rec2 in IDENTITY:
        r = info(os.path.join(REPO, ours), mut)
        if r["meta"]["D"] != D:
            raise RuntimeError(f"{ours}: D {r['meta']['D']} != {D}")
        lr, mu, De, grid, Drt = rt_lam(raw, npz, rec)
        lr2, mu2, De2, grid2, _ = rt_lam(raw, npz2, rec2)
        if Drt != D:
            raise RuntimeError(f"{npz}: D {Drt} != {D}")
        eD = abs(De - r["Delta"]) / r["Delta"]
        rel = abs(r["lam"] - lr) / r["lam"]
        bound = FACTOR * max(eD, FLOOR)
        ok = rel <= bound
        ok_all &= ok
        print(f"D {D}: ours {ours} (M {r['M']}, parts {len(r['parts'])}), RT {grid[0]}x{grid[1]} {npz}")
        print(f"  lam_g  {s(r['lam'], 30)}\n  lam_RT {s(lr, 30)}  (4 pi mu/Delta, RT state)")
        print(f"  4 pi mu_RT / Delta_hp {s(4 * mp.pi * mu / r['Delta'], 30)}")
        print(f"  mu_RT {s(mu, 30)}  mu_hp = lam_g Delta_hp/(4 pi) {s(r['mu'], 30)}")
        print(f"  |Delta_RT - Delta_hp|/Delta {mp.nstr(eD, 3)}  |lam_g - lam_RT|/lam_g {mp.nstr(rel, 3)}  "
              f"bound {mp.nstr(bound, 3)}  {'ok' if ok else 'FAIL'}")
        print(f"  RT spread: {grid2[0]}x{grid2[1]} lam_RT {s(lr2, 30)}, differs by "
              f"{mp.nstr(abs(lr2 - lr) / lr, 3)} (relative), Delta by {mp.nstr(abs(De2 - De) / De, 3)}")
    print("identity:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------------------------------- f64 check

F64_RECORDS = ["hp/data/modecount_check.jsonl", "hp/data/modecount_results.jsonl"]
F64_LIVE = ["hp/data/states/up_D8.2500.npz", "hp/data/states/up6_D30.0688.npz", "hp/data/states/up7_D45.0507.npz"]


def find_state(name, raw=None):
    for base in [os.path.join(REPO, "hp", "data", "states")] + ([raw] if raw else []):
        hits = glob.glob(os.path.join(base, "**", name), recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def cmd_f64(argv):
    mut = mutation(argv)
    raw = opt(argv, "raw", None)
    ok_all, seen = True, set()
    for f in F64_RECORDS:
        for line in open(os.path.join(REPO, f)):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("state") or "lam_g" not in rec:
                continue
            path = find_state(rec["state"], raw)
            if path is None:
                print(f"  {rec['state']}: not found (give --raw=RAW for the raw-data states)")
                continue
            parts, meta = load(path)
            if rec.get("M") != meta["M"] or (rec["state"], rec["lam_g"]) in seen:
                continue          # resampled to another M: not this state's a_h
            seen.add((rec["state"], rec["lam_g"]))
            lg = lam_g(parts, meta, mut)
            _, c1max = lam_g_f64(parts, meta)
            b = f64_bound(meta, c1max)
            d = float(lg - mp.mpf(rec["lam_g"]))
            ok = abs(d) <= b
            ok_all &= ok
            print(f"  {os.path.basename(f)} {rec['state']} (D {meta['D']}, M {meta['M']}): recorded "
                  f"{rec['lam_g']!r}, here {s(lg, 20)}, diff {d:.2e}, bound {b:.1e} {'ok' if ok else 'FAIL'}")
    if "--live" in argv:
        sys.path.insert(0, HERE)
        import modecount_common as mc
        mc.build_pair = lambda meta, zh, zl, M, kf, threads=1: (mc.build_solver(meta, zh, zl, M, kf, threads), None)
        for p in F64_LIVE:
            st = mc.setup(os.path.join(REPO, p), log=lambda *a: None)
            parts = [np.asarray(st.zh), np.asarray(st.zl)]
            lg = lam_g(parts, st.meta, mut)
            _, c1max = lam_g_f64(parts, st.meta)
            b = f64_bound(st.meta, c1max)
            d = float(lg - mp.mpf(st.lam_g))
            ok = abs(d) <= b
            ok_all &= ok
            print(f"  live setup({p}) (D {st.D}, |F| {st.F:.1e}): st.lam_g {st.lam_g!r}, here {s(lg, 20)}, "
                  f"diff {d:.2e}, bound {b:.1e} {'ok' if ok else 'FAIL'}")
    print("f64:", "PASS" if ok_all else "FAIL", "(mutation n)" if mut else "")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------------------------------- the table

# Stored states, D >= 4, in hp's layout. ("repo", glob) is relative to this repository, ("raw", glob) to RAW. kind:
# dd, pinned (held harmonics, dd or four parts), qd, chain (anm_run.py, four or eight parts).
SOURCES = [
    ("raw", "T45-win/hp/data/lowD_states/d4hm15_D4.0000_M*.npz", "dd"),
    ("repo", "hp/data/lowD_states/*_D4.0000_M*.npz", "dd"),
    ("raw", "T79-win/down_D*.npz", "dd"),
    ("raw", "T79-win/m64_D*.npz", "dd"),
    ("raw", "I621-macm3/i621_D*.npz", "dd"),
    ("raw", "win-misc/scratch/hp_val/hp_D7.75_M*.npz", "dd"),
    ("repo", "hp/data/states/*.npz", "dd"),
    ("raw", "I254-mac/states/*.npz", "dd"),
    ("raw", "I609-macm3/states/modecount_state_D30.0688_M*.npz", "dd"),
    ("raw", "I624-macm3/states/*.npz", "dd"),
    ("raw", "I252-mac/states/*.npz", "dd"),
    ("raw", "T12-win/scratch/t12/up1*_D*.npz", "dd"),
    ("raw", "T13-win/scratch/t13/up*_D*.npz", "dd"),
    ("raw", "T73/t73/pin*_D*.npz", "pinned"),
    ("raw", "T75/pin*_D*.npz", "pinned"),
    ("csv", "hp/data/delta_all.csv", None),
    ("raw", "I603-win2/chain/anm_pin3_D*.npz", "chain"),
]
# not solutions: pass-1 states, failed or stopped runs, checkpoints, predictions
SKIP = re.compile(r"_p1_|fail|ckpt|astray|polluted|stalled|pred_|_f64check|aside")
# The backbone (one state per D, the one the fit uses): delta_all.csv's state where it names one; else the first of
# these present at that D; else the first state at that D.
PRIORITY = ["T45-win/hp/data/lowD_states/d4hm15_D4.0000_M192.npz", "repo:hp/data/states/", "raw:T79-win/down_",
            "raw:T12-win/", "raw:T13-win/", "raw:I603-win2/"]


def raw_commit(raw):
    r = subprocess.run(["git", "-C", raw, "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def inventory(raw):
    """[(label, path, kind, named)]: named = delta_all.csv names this state for its D"""
    out, named = [], set()
    for where, pat, kind in SOURCES:
        if where == "csv":
            for r in csv.DictReader(open(os.path.join(REPO, pat))):
                if r["state"].strip():
                    k = {"chain_fit": "chain"}.get(r["kind"], r["kind"])
                    out.append(("raw:" + r["state"], os.path.join(raw, r["state"]), k, True))
                    named.add("raw:" + r["state"])
            continue
        base = REPO if where == "repo" else raw
        for p in sorted(glob.glob(os.path.join(base, pat))):
            rel = os.path.relpath(p, base)
            if SKIP.search(os.path.basename(rel)):
                continue
            out.append((f"{where}:{rel}", p, kind, False))
    seen, uniq = set(), []
    for lab, p, k, nm in out:
        if lab in seen:
            continue
        seen.add(lab)
        uniq.append((lab, p, k, lab in named))
    return uniq


def cmd_table(argv):
    paths = [a for a in argv if not a.startswith("--")]
    if len(paths) != 1:
        sys.exit("usage: lam_gauge.py table RAW [--out=hp/data/lam_gauge.csv]")
    raw = paths[0]
    out = opt(argv, "out", os.path.join(REPO, "hp", "data", "lam_gauge.csv"))
    commit = raw_commit(raw)
    rows, digests = [], {}
    for lab, p, kind, named in inventory(raw):
        parts, meta = load(p)
        if meta.get("prediction") or meta["D"] < 4:
            continue
        try:
            M, K, off = layout(parts, meta)
        except ValueError as e:
            print(f"  skip {lab}: {e}")
            continue
        key = np.concatenate([q[off:off + M] for q in parts] + [q[-1:] for q in parts]).tobytes()
        if key in digests:                 # the same state stored twice (a copy in another folder)
            digests[key]["copies"] += 1
            continue
        with mp.workprec(PREC):
            lg, De = lam_g(parts, meta), delta(parts)
            mu = lg * De / (4 * mp.pi)
        lf, _ = lam_g_f64(parts, meta)
        row = dict(D=repr(float(meta["D"])), D_str=meta.get("D_str", ""), lam_g=s(lg), Delta=s(De), mu=s(mu),
                   lam_g_f64=repr(lf), M=M, K=K, parts=len(parts), kfreeze=meta.get("kfreeze", "") or "",
                   pin_held=";".join(str(x) for x in meta.get("pin_held", []) or []), F=meta.get("F", ""),
                   kind=kind, role="", named=int(named), state=lab, copies=0,
                   _lam=lg, _Dkey=round(float(meta["D"]), 6))
        digests[key] = row
        rows.append(row)
    rows.sort(key=lambda r: (r["_Dkey"], r["state"]))
    groups = {}
    for r in rows:
        groups.setdefault(r["_Dkey"], []).append(r)
    for Dk, g in groups.items():
        best = [r for r in g if r["named"]]
        if not best:
            for pre in PRIORITY:
                best = [r for r in g if r["state"].startswith(pre) or r["state"] == "raw:" + pre]
                if best:
                    break
        best = (best or g)[0]
        for r in g:
            r["role"] = "backbone" if r is best else "alt"
        lams = [r["_lam"] for r in g]
        sp = max(lams) - min(lams) if len(g) > 1 else mp.mpf(0)
        best["n_same_D"] = len(g)
        best["spread_same_D"] = mp.nstr(sp, 3) if len(g) > 1 else ""
        for r in g:
            r.setdefault("n_same_D", "")
            r.setdefault("spread_same_D", "")
    cols = ["D", "D_str", "lam_g", "Delta", "mu", "lam_g_f64", "M", "K", "parts", "kfreeze", "pin_held", "F", "kind",
            "role", "n_same_D", "spread_same_D", "named", "copies", "state"]
    with open(out, "w", newline="") as f:
        f.write(f"# lam_g = 1 - (n - 1)<a_h^2 - 1> (hp/lam_gauge.py table), mu = lam_g Delta / (4 pi); raw-data "
                f"repository at {commit}; state 'raw:' paths are in it, 'repo:' in this repository\n")
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    bb = [r for r in rows if r["role"] == "backbone"]
    print(f"{len(rows)} states ({len(bb)} backbone D) from D {bb[0]['D']} to {bb[-1]['D']}; raw {commit}; -> {out}")
    for Dk, g in groups.items():
        if len(g) > 1:
            b = [r for r in g if r["role"] == "backbone"][0]
            desc = ", ".join(f"{r['kind']} M{r['M']} p{r['parts']}" + (f" pin{r['pin_held'].count(';') + 1}"
                                                                         if r["pin_held"] else "") for r in g)
            print(f"  D {Dk}: {len(g)} states, spread {b['spread_same_D']} ({desc})")
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd, argv = sys.argv[1], sys.argv[2:]
    fn = dict(show=cmd_show, rt=cmd_rt, identity=cmd_identity, f64=cmd_f64, table=cmd_table).get(cmd)
    if fn is None:
        print(__doc__)
        return 2
    return fn(argv)


if __name__ == "__main__":
    sys.exit(main())
