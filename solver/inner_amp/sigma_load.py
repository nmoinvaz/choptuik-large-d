#!/usr/bin/env python3
"""The real S_1 series sigma_k of amp_flint_par.py's files, read from their strings at full precision (issue #683).

A sigF_*.json file holds sigma_k twice, as doubles ("sigma") and as decimal strings ("sigma_str", mp.nstr(z, 20), so 20
significant digits). The loaders here parse the strings inside mp.workdps(DPS_LOAD) and never pass through a float.
Each loaded component then equals its string to 2^-997 relative, whatever precision the caller works at afterwards.
DPS_LOAD = 300 lies above every working precision of the Pade and twin scripts (60, 100, 120, 150 and 250 digits).

The strings are the floor. Rounding to 20 digits moves sigma_k by up to half a unit in the 20th digit (rounding_rel),
which is coarser than the measured errors (sigma_err_fit.py) at k = 1..8.

    python3 sigma_load.py --check [--mutate=53bit]
    python3 sigma_load.py --rounding [OUTDIR] [--allow_unraw]

--check loads every file the Pade and twin scripts read through each caller's own entry point (pade_geometry.load_scan,
pade_twin_count.real_sig, planted_plants.load_real), each called at mpmath's default 53 bits. Every nonzero component
must equal its string, parsed again at CHECK_DPS digits, within TOL = 1e-100 relative. --mutate=53bit makes the loaders
parse at 53 bits, the precision real_sig() held before #683, and the check must then fail. Exit 1 when it fails.
--rounding prints, per order, the strings' rounding against the measured error, and writes rounding.json to OUTDIR.
"""
import glob
import json
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
NODES = (0.99, 1.0, 1.01)
DPS_LOAD = 300
CHECK_DPS = 600
TOL = 1e-100
MUTATE = None                       # "53bit": parse at 53 bits (the control of --check)


def _ctx(prec):
    if MUTATE == "53bit":
        return mp.workprec(53)
    return mp.workprec(prec) if prec else mp.workdps(DPS_LOAD)


def parse(pair, prec=None):
    """one complex sigma_k from its two decimal strings, at DPS_LOAD digits (or prec bits)"""
    with _ctx(prec):
        return mp.mpc(mp.mpf(pair[0]), mp.mpf(pair[1]))


def load_file(fn, prec=None):
    """([sigma_0, sigma_1, ...], the file's record)"""
    r = json.load(open(fn))
    return [parse(p, prec) for p in r["sigma_str"]], r


def node_file(d, wf):
    fn = sorted(f for f in os.listdir(d) if f.startswith(f"sigF_w{wf:.4f}") and f.endswith(".json"))
    return os.path.join(d, fn[0]) if fn else None


def load_scan(d, nodes=NODES, prec=None):
    """{wf: [sigma_k]} from the files sigF_w<wf>*.json of directory d, for the nodes that have one"""
    out = {}
    for wf in nodes:
        fn = node_file(d, wf)
        if fn:
            out[wf] = load_file(fn, prec)[0]
    return out


def load_dir(d, prec=None):
    """{wf: ([sigma_k], sigma_err)} from every *.json of directory d"""
    out = {}
    for fn in sorted(glob.glob(os.path.join(d, "*.json"))):
        sig, r = load_file(fn, prec)
        out[float(r["wf"])] = (sig, list(r.get("sigma_err", [])))
    return out


# ---------------------------------------------------------------- the strings' rounding
def digits(s):
    """significant digits written in a decimal string"""
    m = s.lower().lstrip("+-").split("e")[0].replace(".", "").lstrip("0")
    return len(m)


def half_ulp(s, nd):
    """half a unit in the nd-th significant digit of the value written in s (0 for an exact zero)"""
    with mp.workdps(CHECK_DPS):
        x = mp.mpf(s)
        if x == 0:
            return mp.mpf(0)
        e = int(mp.floor(mp.log10(abs(x))))
        return mp.mpf(10) ** (e - nd + 1) / 2


def rounding(r):
    """per order k, the half units (h_re, h_im) of the strings' last digit and |dsigma_k| / |sigma_k|. nstr drops
    trailing zeros, so the digit count is the file's largest (20 for amp_flint_par.py)."""
    nd = max(digits(s) for p in r["sigma_str"] for s in p)
    out = []
    with mp.workdps(CHECK_DPS):
        for a, b in r["sigma_str"]:
            hr, hi = half_ulp(a, nd), half_ulp(b, nd)
            out.append((hr, hi, mp.sqrt(hr * hr + hi * hi) / abs(mp.mpc(mp.mpf(a), mp.mpf(b)))))
    return nd, out


# ---------------------------------------------------------------- check
def worst_rel(sig, strs):
    """largest |loaded - string| / |string| over the nonzero components, the string parsed at CHECK_DPS"""
    if len(sig) != len(strs):
        return mp.inf
    w = mp.mpf(0)
    with mp.workdps(CHECK_DPS):
        for z, (a, b) in zip(sig, strs):
            for x, s in ((z.real, a), (z.imag, b)):
                t = mp.mpf(s)
                if t != 0:
                    w = max(w, abs(x - t) / abs(t))
    return w


def check(log=print):
    sys.path.insert(0, HERE)
    import pade_geometry as pg
    import pade_twin_count as tc
    import planted_plants as pp
    strs = {}

    def strings(fn):
        if fn not in strs:
            strs[fn] = json.load(open(fn))["sigma_str"]
        return strs[fn]

    def fresh(call):                # mpmath's default precision, as at a script's first load
        mp.mp.prec = 53
        tc._REAL.clear()
        return call()
    paths = [("pade_geometry.load_scan(J1250)", pg.REAL["J1250"], lambda: fresh(lambda: pg.load_scan(pg.REAL["J1250"]))),
             ("pade_geometry.load_scan(J1500)", pg.REAL["J1500"], lambda: fresh(lambda: pg.load_scan(pg.REAL["J1500"]))),
             ("pade_twin_count.real_sig()", pg.REAL["J1250"], lambda: fresh(tc.real_sig)),
             ("planted_plants.load_real()", pp.REAL,
              lambda: {wf: v[0] for wf, v in fresh(pp.load_real).items()})]
    ok, rec = True, {}
    for name, d, call in paths:
        got = call()
        n = 0
        for wf, sig in got.items():
            fn = node_file(d, wf) if d != pp.REAL else next(f for f in sorted(glob.glob(os.path.join(d, "*.json")))
                                                             if float(json.load(open(f))["wf"]) == wf)
            w = worst_rel(sig, strings(fn))
            good = w <= TOL
            ok &= good
            n += 1
            rec[f"{name} wf {wf}"] = {"file": os.path.relpath(fn, HERE), "orders": len(sig), "worst_rel": float(w),
                                     "pass": good}
            log(f"{name:32s} wf {wf:<5}: {len(sig)} orders from {os.path.relpath(fn, HERE)}, largest |loaded - string| "
                f"/ |string| {float(w):.1e} (pass <= {TOL:.0e}) {'OK' if good else 'BAD'}")
        if n == 0:
            ok = False
            log(f"{name}: no file loaded, BAD")
    mp.mp.prec = 53
    log(f"check{' --mutate=' + MUTATE if MUTATE else ''}: {'PASS' if ok else 'FAIL'}")
    return ok, rec


def rounding_report(outdir=None, log=print):
    sys.path.insert(0, HERE)
    import pade_geometry as pg
    pg.NOISE_MODE = "measured"
    rec = {}
    files = [(lab, wf, node_file(pg.REAL[lab], wf)) for lab in ("J1250", "J1500") for wf in NODES]
    for lab, wf, fn in files:
        if fn is None:
            continue
        r = json.load(open(fn))
        nd, rows = rounding(r)
        rel = pg.noise_eps(wf)[:len(rows)] if lab == "J1250" else None
        out = []
        for k, (hr, hi, q) in enumerate(rows):
            e = rel[k] if rel else None
            out.append({"k": k, "half_ulp_re": float(hr), "half_ulp_im": float(hi), "rounding_rel": float(q),
                        "measured_rel": e, "ratio": (float(q) / e) if e else None})
        rec[f"{lab} {wf}"] = {"file": os.path.relpath(fn, HERE), "digits": nd, "orders": out}
        log(f"{lab} wf {wf} ({os.path.relpath(fn, HERE)}, {nd} digits): k  rounding_rel  measured_rel  ratio")
        for o in out[:23]:
            m = "-" if o["measured_rel"] is None else f"{o['measured_rel']:.1e}"
            q = "-" if o["ratio"] is None else f"{o['ratio']:.1e}"
            log(f"   {o['k']:2d}  {o['rounding_rel']:.1e}  {m}  {q}")
        above = [o["k"] for o in out[1:23] if o["ratio"] and o["ratio"] > 1]
        if rel:
            log(f"   rounding above the measured error at k = {above}")
            rec[f"{lab} {wf}"]["rounding_above_error_k"] = above
    if outdir:
        json.dump(rec, open(os.path.join(outdir, "rounding.json"), "w"), indent=1)
    return rec


def main():
    global MUTATE
    sys.modules["sigma_load"] = sys.modules[__name__]      # the callers' `import sigma_load` must see MUTATE
    argv = sys.argv[1:]
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "hp"))
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    MUTATE = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    if MUTATE not in (None, "53bit"):
        raise SystemExit(f"unknown mutation {MUTATE}")
    argv = [a for a in argv if not a.startswith("--mutate=")]
    if "--check" in argv:
        sys.exit(0 if check()[0] else 1)
    if "--rounding" in argv:
        rest = [a for a in argv if not a.startswith("--")]
        if rest:
            require_raw(os.path.join(rest[0], "x.json"), allow=allow)
            os.makedirs(rest[0], exist_ok=True)
        rounding_report(rest[0] if rest else None)
        sys.exit(0)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
