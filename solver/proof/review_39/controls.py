#!/usr/bin/env python3
"""Controls of the #39 review: the transcription test, and the mutations run through both the review's checks and
the original code. Every control states what must happen; the exit status is 0 only if every one behaves.

    python3 controls.py transcription [--mutate=sign]
        my rhs_generic (check_6b1.py) and its forward-mode Jacobian against lemma6_usys_M.SysUM.rhs and .lin (the
        original, python-flint) at random complex states, M = 5. This compares two transcriptions of the same
        equations; with --mutate=sign (mine with V_u = +V/2) the comparison must fail.
    python3 controls.py orig_discs --inputs=<dir> --rows=17 --discs=0,9 [--mutate=sign] [--r_scale=1.01]
        the original lemma6b1_disc_M.certify on the regenerated inputs (same inputs as check_6b1.py discs)
    python3 controls.py orig_cover --runs=<dir> [--mutate=drop:9,5 | radius:9 | g0]
        the original covering check (lemma6b1_cover_M.py check, through lemma6b1_cover.check) on a copy of the stored
        runs, with the same mutation as check_6b1.py cover --mutate=...
"""
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.dirname(HERE)


def transcription(mutate=None, trials=4, seed=39):
    sys.path.insert(0, PROOF)
    sys.path.insert(0, HERE)
    from flint import acb, arb, ctx
    import lemma6_usys_M as UM
    import check_6b1 as K
    import civ as C
    from mpmath.libmp import from_float, to_float
    ctx.prec = 200
    C.setprec(200)
    M = 5
    S = UM.SysUM(M)
    nd = S.nd
    random.seed(seed)
    worst_f = worst_j = 0.0
    for _ in range(trials):
        Y = [complex(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(nd)]
        w = complex(random.uniform(3, 4), 0.0)
        u = complex(random.uniform(-40, 40), random.uniform(-40, 40))
        z = 1 / u
        Ya = [acb(y.real, y.imag) for y in Y] + [acb(w.real, w.imag), acb(2.5)]
        za = 1 / acb(u.real, u.imag)
        F1, cache = S.rhs(Ya, acb(u.real, u.imag), za, acb(0))

        def civc(x):
            return ((from_float(x.real),) * 2, (from_float(x.imag),) * 2)
        zc = C.cinv(civc(u))
        z2c = C.cmul(zc, zc)
        wc = civc(w)
        DR = K.DRing(nd)
        Yd = [(civc(Y[i]), [K.ONE if j == i else K.CZ for j in range(nd)]) for i in range(nd)]
        Fd = K.rhs_generic(M, Yd, (zc, [K.CZ] * nd), (z2c, [K.CZ] * nd), wc, DR, mutate=mutate)
        for i in range(nd):
            a = complex(float(F1[i].real.mid()), float(F1[i].imag.mid()))
            b = C.cmid_float(Fd[i][0])
            worst_f = max(worst_f, abs(a - b) / (1 + abs(a)))
        for j in range(nd):
            e = [acb(1) if i == j else acb(0) for i in range(nd + 2)]
            col = S.lin(cache, e, acb(u.real, u.imag), za, acb(0))
            for i in range(nd):
                a = complex(float(col[i].real.mid()), float(col[i].imag.mid()))
                b = C.cmid_float(Fd[i][1][j])
                worst_j = max(worst_j, abs(a - b) / (1 + abs(a)))
    print(f"transcription (M = 5, {trials} random states): rhs max relative mismatch {worst_f:.1e}, Jacobian "
          f"{worst_j:.1e}" + (f"   [MUTATION {mutate}: must exceed 1e-3]" if mutate else "   (must be below 1e-12)"))
    if mutate:
        return 0 if max(worst_f, worst_j) > 1e-3 else 3
    return 0 if max(worst_f, worst_j) < 1e-12 else 3


def me2acb(p):
    from flint import arb, acb

    def a(q):
        return arb(q[0][0]) * arb(2) ** q[0][1] + arb(0, arb(q[1][0]) * arb(2) ** q[1][1])
    return acb(a(p[0]), a(p[1]))


def orig_discs(opts):
    """the original certify on the dumped inputs (exactly the arguments row_run passed)"""
    sys.path.insert(0, PROOF)
    from flint import arb, acb, ctx
    import lemma6b1_disc_M as LDM
    ctx.prec = 256
    mutate = opts.get("mutate")
    rsc = opts.get("r_scale")
    allok, anyok = True, False
    for j in [int(v) for v in opts["rows"].split(",")]:
        with open(os.path.join(opts["inputs"], f"row_{j:02d}.json")) as fh:
            inp = json.load(fh)
        sel = [int(v) for v in opts["discs"].split(",")] if "discs" in opts else range(len(inp["discs"]))
        for k in sel:
            d = inp["discs"][k]
            ctx.prec = 256
            c = me2acb(d["c"])
            Y0 = [me2acb(y) for y in d["Y0"]]
            w = me2acb(d["w"])
            r = d["r"] if rsc is None else repr(round(float(d["r"]) * float(rsc), 10))
            ctx.prec = 128
            res = LDM.certify(5, c, Y0, w, r, d["n"], mutate=mutate)
            allok = allok and res["ok"]
            anyok = anyok or res["ok"]
            print(f"ORIGINAL certify row {j} disc {k}: ok {res['ok']}, Lam {float(res['Lam'].upper()):.6f}, "
                  f"low coefficients 0: {res.get('low_ok')}" + (f"  [MUTATION {mutate}]" if mutate else "")
                  + (f"  [r x {rsc}]" if rsc else ""), flush=True)
    if mutate:
        return 0 if not anyok else 3
    return 0 if allok else 3


def orig_cover(opts):
    """the original covering check on a copy of the stored runs, mutated as check_6b1.py cover --mutate"""
    sys.path.insert(0, PROOF)
    import lemma6b1_cover_M as CM
    import lemma6b1_cover as LC
    mut = opts.get("mutate", "")
    tmp = tempfile.mkdtemp(prefix="i39_cover_")
    for f in os.listdir(opts["runs"]):
        if f.startswith("row_") or f == "ray.json":
            shutil.copy(os.path.join(opts["runs"], f), tmp)
    drop = None
    if mut.startswith("drop:"):
        drop = tuple(int(v) for v in mut.split(":")[1].split(","))
    elif mut.startswith("radius:"):
        j = int(mut.split(":")[1])
        f = os.path.join(tmp, f"row_{j:02d}.json")
        with open(f) as fh:
            d = json.load(fh)
        d["r"] = d["r"] * 1.01                                     # the stored row's disc radius x 1.01
        with open(f, "w") as fh:
            json.dump(d, fh)
    elif mut == "g0":
        CM.CONF[5]["g0"] = str(float(CM.CONF[5]["g0"]) * 1.01)     # G0's radius x 1.01
    CM.runs_dir = lambda _M: tmp
    with CM.patched_check(5):
        res = LC.check("1e-18", drop, verbose=True)
    print(f"ORIGINAL cover check: {'PASS' if res['ok'] else 'FAIL'}" + (f"  [MUTATION {mut}]" if mut else ""))
    if mut:
        return 0 if not res["ok"] else 3
    return 0 if res["ok"] else 3


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    cmd = args[0] if args else "transcription"
    if cmd == "transcription":
        return transcription(opts.get("mutate"))
    if cmd == "orig_discs":
        return orig_discs(opts)
    if cmd == "orig_cover":
        return orig_cover(opts)
    raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    sys.exit(main())
