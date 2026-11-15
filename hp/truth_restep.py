"""Re-converge a stored truth (#874).

store: N plain pinned Newton steps z_k = z_(k-1) + dz_k from a state in its own system (#837's truth_system, taken as
step_spread.py takes it), then the step at z_N for the record. z_N is saved as an eight-part state
(octo_corrector.save_o) with the source's meta (its stored plan and the rest) and meta["restep"] naming the source, so
that truth_system reads it in the source's system. The saved parts are read back and must equal z_N's.

check: a state read through truth_system and one plain step. Passes when max|dz| / max|z_h| <= --dz (1e-97) and the
pinned max|F| at the state <= --F (1e-92). Its control is the source truth, which must fail both bounds. --held_ref
STATE builds STATE's system too and reports the largest difference of the two held offsets (eight parts), 0 when the
state keeps the source's held values.

--match JSON compares each step's max|dz| and max|F| before it, as doubles, with the unperturbed path of a
step_spread.py --steps JSON (#852's I852-mac/steps_D240.json): store from its step 1 on, check against its step
--match_k. A state other than that path's iterate (the control) must not match.

  python3 truth_restep.py store RAW STATE OUT [--steps 2] [--threads 3] [--match JSON]
  python3 truth_restep.py check RAW STATE [--dz 1e-97] [--F 1e-92] [--threads 3] [--held_ref STATE]
                                [--match JSON --match_k K] [--json OUT.json]

STATE and OUT are relative to RAW, a checkout of the raw-data repository; OUT must be in a task folder of it. Exit 1
when a check fails.
"""
import json
import os
import sys
import time

import numpy as np


def _opts(argv, opt):
    args, i = [], 0
    while i < len(argv):
        if argv[i].startswith("--"):
            k = argv[i][2:]
            if k not in opt:
                raise SystemExit(f"truth_restep.py: unknown option {argv[i]}; known: {sorted(opt)}")
            opt[k] = argv[i + 1]
            i += 2
        else:
            args.append(argv[i])
            i += 1
    return args, opt


def _load(raw, path, threads):
    """the state as step_spread.py takes it: truth_system, then planted's start at rel 0; and max|z_h|"""
    from octo_corrector import o8_any, plant_start, truth_system
    from proj_shift import blocks
    from reseed import offsets
    z4, meta, sysm, info = truth_system(raw, path, threads=threads)
    M = int(meta["M"])
    K = (len(z4[0]) - 1) // (4 * M) - 1

    class RS:           # plant_start reads the layout only
        off = offsets(M, K)
    RS.K = K
    zs, _ = plant_start(z4, RS, 0.0, 1, 0)
    z = o8_any(zs)
    bl = {n: (lo, hi) for n, lo, hi in blocks(M, K)}
    zh = float(np.abs(np.asarray(z[0][bl["m_h"][0]:bl["a_h"][1]])).max())
    return z, meta, sysm, info, zh


def _step(so, z, zh):
    """the plain pinned Newton step at z: (dz in eight parts, its record); dDelta summed as step_spread.py sums it"""
    import mpmath as mp
    from octo_corrector import o8, omax
    t = time.time()
    st = so.newton_o(z)
    d = o8(st[0])
    with mp.workdps(150):
        dD = float(sum((mp.mpf(float(p[-1])) for p in d), mp.mpf(0)))
    dz = float(np.abs(d[0]).max())
    return d, dict(dz=dz, dz_rel=dz / zh, F=omax(st[1:3]), dDelta=dD, dDelta_str=repr(dD), piv=st[3],
                   s=round(time.time() - t, 1))


def _ref(path, state):
    """the unperturbed path's steps of a step_spread.py --steps JSON from the same state"""
    js = json.load(open(path))
    if js["state"] != state:
        raise SystemExit(f"truth_restep.py: {path} is from {js['state']}, not {state}")
    return next(r for r in js["runs"] if r["run"] == "unperturbed")["hist"]


def _same(h, r, name):
    same = h["dz"] == r["dz"] and h["F"] == r["F_before"]
    return same, (f"; against {name} step {r['k']}: max|dz| {r['dz']!r}, max|F| {r['F_before']!r}: "
                  f"{'equal' if same else 'NOT equal'}")


def _line(k, h):
    return (f"  step {k}: max|dz| {h['dz']!r} (/ max|z_h| {h['dz_rel']:.4e}), dDelta {h['dDelta_str']}, smallest "
            f"pivot {h['piv']:.4e}, max|F| before {h['F']!r} ({h['s']:.0f} s)")


def store(argv):
    from multidouble import od_axpy
    from octo_corrector import load_o, o8, save_o
    from raw_out import require_raw
    args, opt = _opts(argv, dict(steps="2", threads="3", match=""))
    raw, path, out = args
    fout = os.path.join(raw, out)
    require_raw(fout)
    n = int(opt["steps"])
    t0 = time.time()
    z, meta, (s, sp, sq, so), info, zh = _load(raw, path, int(opt["threads"]))
    print(f"truth_restep store {path}: D {meta['D_str'][:16]}, {n} steps, then the step at z_{n}; max|z_h| {zh!r}; "
          f"read in {info['read']} of {info['file_parts']} parts, held values {info['held']}, plan {info['plan']} "
          f"({time.time() - t0:.0f} s)", flush=True)
    ref = _ref(opt["match"], path) if opt["match"] else None
    ok, zk, hist = True, z, []
    for k in range(1, n + 2):
        d, h = _step(so, zk, zh)
        h["k"] = k
        line = _line(k, h)
        if ref and k <= len(ref):
            same, txt = _same(h, ref[k - 1], os.path.basename(opt["match"]))
            ok &= same
            line += txt
        print(line, flush=True)
        hist.append(h)
        if k <= n:
            zk = o8(od_axpy(1.0, *d, *zk))
    m2 = dict(meta)
    m2["restep"] = dict(source=path, steps=n, code="hp/truth_restep.py", hist=hist)
    save_o(fout, zk, m2)
    zb, mb = load_o(fout)
    back = (all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(zb, zk))
            and mb.get("segs") == meta.get("segs"))
    ok &= back
    print(f"  saved {out}: z_{n} in eight parts with the source's meta and plan; read back "
          f"{'equal' if back else 'NOT equal'}", flush=True)
    print(f"truth_restep store: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f} s)")
    return 0 if ok else 1


def check(argv):
    from multidouble import od_axpy
    from octo_corrector import o8, omax
    args, opt = _opts(argv, dict(dz="1e-97", F="1e-92", threads="3", held_ref="", match="", match_k="0", json=""))
    raw, path = args
    t0 = time.time()
    z, meta, (s, sp, sq, so), info, zh = _load(raw, path, int(opt["threads"]))
    print(f"truth_restep check {path}: D {meta['D_str'][:16]}; max|z_h| {zh!r}; read in {info['read']} of "
          f"{info['file_parts']} parts, held values {info['held']}, plan {info['plan']}"
          + (f"; restep of {meta['restep']['source']} by {meta['restep']['steps']} steps" if "restep" in meta else "")
          + f" ({time.time() - t0:.0f} s)", flush=True)
    rec = dict(state=path, zh=zh, info=info)
    if opt["held_ref"]:
        from octo_corrector import truth_system
        _, _, (_, sp0, _, _), _ = truth_system(raw, opt["held_ref"], threads=int(opt["threads"]))
        if sp.c8 is None or sp0.c8 is None:
            raise SystemExit("truth_restep.py: --held_ref needs both held offsets in eight parts")
        dh = float(np.abs(np.asarray(o8(od_axpy(-1.0, *sp0.c8, *sp.c8))[0])).max())
        hmax = float(np.abs(np.asarray(sp0.c8[0])).max())
        rec["held_diff"] = dh
        print(f"  held offset against {opt['held_ref']}'s: max difference {dh!r} (max|offset| {hmax:.4e})", flush=True)
    F0 = omax(so.residual_o(z))
    d, h = _step(so, z, zh)
    line = _line(1, h)
    ok_m = True
    if opt["match"]:
        same, txt = _same(h, _ref(opt["match"], meta["restep"]["source"] if "restep" in meta else path)
                          [int(opt["match_k"]) - 1], os.path.basename(opt["match"]))
        ok_m, line = same, line + txt
    print(line, flush=True)
    b_dz, b_F = float(opt["dz"]), float(opt["F"])
    checks = [("step", h["dz_rel"] <= b_dz, f"max|dz| / max|z_h| {h['dz_rel']:.4e} (<= {b_dz:g})"),
              ("residual", F0 <= b_F, f"pinned max|F| at the state {F0:.4e} (<= {b_F:g})")]
    if opt["match"]:
        checks.append(("match", ok_m, f"the step against step {opt['match_k']} of {os.path.basename(opt['match'])}"))
    for name, ok, what in checks:
        print(f"  check {name}: {what}: {'pass' if ok else 'FAIL'}")
    ok = all(c[1] for c in checks)
    rec.update(step=h, F=F0, checks={c[0]: c[1] for c in checks}, passed=ok)
    if opt["json"]:
        with open(opt["json"], "w") as fh:
            json.dump(rec, fh, indent=1)
    print(f"truth_restep check: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("store", "check"):
        raise SystemExit(__doc__)
    sys.exit((store if a[0] == "store" else check)(a[1:]))
