#!/usr/bin/env python3
"""Bit identity of every double-double entry point of hp's Solver across two builds (#523), beyond xsolve_bits.py and
highm_check.py: the hashes of every returned array at a low-D state (no kfreeze, M = 48) and at a large-D state
(kfreeze 3, M = 32).

    PYTHONPATH=<site> python3 api_bits.py dump HPDIR RAWDIR OUT.json [--perturb] [--only=lowd|large]
    python3 api_bits.py compare A.json B.json      # exit 1 on any difference

--perturb adds one ulp to the first entry of each state (the control: compare must report differences).

Every NaN is hashed as 0x7FF8000000000000, arm64's default NaN: x86 writes 0xFFF8000000000000, so without it the one
control output full of NaN (jacobian_f64/mut1) differs between platforms in the sign bit alone (#600, #604). The Mac's
stored references (I523-mac/c3/api.json) hash the same either way. From I523-mac/api_bits.py (#523), moved here.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np


def digest(x):
    if isinstance(x, (list, tuple)):
        return [digest(v) for v in x]
    if isinstance(x, float):
        return np.float64(x).tobytes().hex()
    a = np.array(x, dtype=np.float64, copy=True)
    # one NaN pattern on every platform: the sign bit of a NaN is not a number this repository uses
    a[np.isnan(a)] = np.frombuffer(bytes.fromhex("000000000000f87f"), dtype="<f8")[0]
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()[:20] + f":{a.shape}"


def vec(n, seed):
    rng = np.random.default_rng(seed)
    h = rng.uniform(-1, 1, n)
    return h, np.zeros(n)


def run_all(s, zh, zl, r, block, t0):
    from hpkernel.driver import dd_add
    n = s.size
    e = np.zeros(n)
    e[-1] = 1.0
    z0 = np.zeros(n)

    def put(k, v):
        r[k] = digest(v)
        print(f"  {k} ({time.time() - t0:.0f} s)", flush=True)

    put("residual", s.residual(zh, zl))
    put("plan", np.concatenate([np.asarray(g) for g in s.plan(zh, zl, 1e-30, 0.05)]))
    for w in (0, 1):
        put(f"series{w}", s.series(zh, zl, w))
        put(f"series{w}_scaled", s.series(zh, zl, w, True))
        put(f"series_scale{w}", s.series_scale(w))
    segs = s.get_segs()
    m = s.m
    off = 2 * m
    put("integrate", s.integrate(zh, zl, zh[off:off + 4 * m].copy(), zl[off:off + 4 * m].copy(), list(segs[0])))
    put("jacobian", s.jacobian(zh, zl))
    for t in (1, 2, 4, 8):
        put(f"jacobian_f64/t{t}", s.jacobian_f64(zh, zl, tangents=t))
        put(f"jacobian_f64/dd{t}", s.jacobian_f64(zh, zl, tangents=t, dd_base=True))
    for hdd in (1, 2, 4, 8, 16, 17, 5, 12):
        put(f"jacobian_f64/hdd{hdd}", s.jacobian_f64(zh, zl, horizon_dd=hdd))
    put("jacobian_f64/t1hdd4", s.jacobian_f64(zh, zl, tangents=1, horizon_dd=4))
    for mut in (1, 2, 3, 6):
        put(f"jacobian_f64/mut{mut}", s.jacobian_f64(zh, zl, mutate=mut))
    nw = s.newton(zh, zl)
    put("newton", list(nw[:4]) + [float(nw[4]), float(nw[5])])
    th, tl = dd_add(zh, zl, np.asarray(nw[0]), np.asarray(nw[1]))
    put("newton_chord", s.chord(th, tl))
    put("newton_lu_solve", s.lu_solve(e, z0))
    put("newton_lu_solve_t", s.lu_solve_t(e, z0))
    R = np.vstack([e, np.asarray(zh)])
    put("newton_lu_solve_many", s.lu_solve_many(R, np.zeros_like(R)))
    put("newton_lu_solve_many_t", s.lu_solve_many(R, np.zeros_like(R), transpose=True))
    for name, kw in (("default", {}), ("ddlu", dict(lu="dd")), ("t1", dict(tangents=1)), ("dd4", dict(dd_base=True)),
                     ("hdd4", dict(horizon_dd=4)), ("hdd17", dict(horizon_dd=17)), ("dense", dict(block=False))):
        o = s.newton_mixed(zh, zl, **kw)
        put(f"newton_mixed/{name}", list(o[:4]) + [float(o[4]), float(o[5])])
        put(f"newton_mixed/{name}/solve_t", s.lu_solve_t(e, z0))
    if block:
        for name, kw in (("f64x4", {}), ("f64x1", dict(tangents=1)), ("dd4", dict(dd_base=True)), ("jmut1", dict(jac_mutate=1)),
                         ("pivot", dict(block_mutate=1)), ("link", dict(block_mutate=2))):
            o = s.newton_mixed(zh, zl, block=True, **kw)
            put(f"block/{name}", list(o[:4]) + [float(o[4]), float(o[5])])
            put(f"block/{name}/solve_t", s.lu_solve_t(e, z0))
            put(f"block/{name}/chord", s.chord(th, tl))
            put(f"block/{name}/lu_info", [float(s.lu_info()[0])])
    vh, vl = vec(n, 7)
    for mut in (0, 1, 2, 3):
        put(f"jvp/{mut}", s.jvp(zh, zl, vh, vl, mutate=mut, with_residual=True))
    put("jvp/cut", s.jvp(zh, zl, vh, vl, with_residual=True, cut=True))
    for mut in (0, 21, 22):
        put(f"residual_jet/{mut}", s.residual_jet([zh, zh / 1000], [zl, zl / 1000], 3, mut))
    put("residual_jet/nout10", s.residual_jet([zh], [zl], 10))
    for lam in (0.5, 1.7):
        put(f"mode_matrix/{lam}", s.mode_matrix(zh, zl, lam, 0.0))
        put(f"mode_det/{lam}", [float(v) for v in s.mode_det(zh, zl, lam, 0.0)])
    put("mode_matrix/0", s.mode_matrix(zh, zl, 0.0, 0.0))
    st = s.lm_steps(zh, zl, [0.1, 1.0])
    put("lm_steps", [list(p) for p in st[0]] + [st[1], st[2]])
    for mode in (1, 2):
        s.set_horizon_alpha(mode)
        put(f"halpha{mode}/residual", s.residual(zh, zl))
        put(f"halpha{mode}/jacobian", s.jacobian(zh, zl))
        put(f"halpha{mode}/jacobian_f64", s.jacobian_f64(zh, zl))
        put(f"halpha{mode}/jvp", s.jvp(zh, zl, vh, vl, with_residual=True))
        put(f"halpha{mode}/mode_matrix", s.mode_matrix(zh, zl, 1.7, 0.0))
    s.set_horizon_alpha(0)


def dump(hp, raw, out, perturb, only):
    import hpkernel
    from hpkernel.driver import load, make_solver
    sys.path.insert(0, hp)
    from octo_corrector import load_o
    rec = {"hpkernel": hpkernel.__file__, "states": {}}
    t0 = time.time()
    if only in ("", "lowd"):
        zh, zl, m = load(os.path.join(hp, "data", "lowD_states", "lowDv_D3.7000_M48.npz"))
        zh, zl = np.array(zh, dtype=float), np.array(zl, dtype=float)
        if perturb:
            zh[0] = np.nextafter(zh[0], np.inf)
        s = make_solver(m["D"], 48, nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=2)
        s.set_segs(m["segs"] if m.get("segs") else s.plan(zh, zl, 1e-30, 0.05))
        r = {}
        run_all(s, zh, zl, r, True, t0)
        rec["states"]["lowD3.7_M48"] = r
    if only in ("", "large"):
        st = "anm-mac/run_trend_pin3_qd3/anm_pin3_D79.7657.npz"
        z8, m = load_o(os.path.join(raw, st))
        zh, zl = np.array(z8[0], dtype=float), np.array(z8[1], dtype=float)
        if perturb:
            zh[0] = np.nextafter(zh[0], np.inf)
        D = m.get("D_str", repr(float(m["D"])))
        s = make_solver(D, int(m["M"]), nodes=list(m["nodes"]), kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130),
                        threads=2, kfreeze=m.get("kfreeze", 3))
        s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
        r = {}
        run_all(s, zh, zl, r, False, t0)
        rec["states"][st] = r
    json.dump(rec, open(out, "w"), indent=1)
    print("wrote", out, f"{time.time() - t0:.0f} s")


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    bad = 0
    n = 0
    for st in A["states"]:
        for k, v in A["states"][st].items():
            n += 1
            w = B["states"].get(st, {}).get(k)
            if v != w:
                bad += 1
                print(f"DIFFERS  {st}  {k}")
    print(f"{n - bad} of {n} outputs byte-identical")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), "")
        dump(sys.argv[2], sys.argv[3], sys.argv[4], "--perturb" in sys.argv, only)
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
