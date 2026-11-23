#!/usr/bin/env python3
"""Independent reproduction of the Lean Krawczyk instances of #1077 (notes/LEAN_REVIEW_1077.md).

It shares no code with lean/tools/export_krawczyk.py. It reads the generated Lean files back (its own parser), and
compares them with the saved certificate data by two routes:

  exact   (m3, m5, capd) the JSON alone, in exact rational arithmetic (fractions.Fraction): the decimal strings of the
          saved Lohner sets, the hex floats of Q, and the Y of the Lean file give the exact interval value of
          Y l (c + Q q), Y l (cV + QV qV) and, for M = 5, G1 + sum_k G2 [-w_k, w_k]. arb evaluates the same
          expressions with outward rounding, so each Lean interval must contain the exact one, and be barely wider.
          This uses neither python-flint nor certify_M*.py. For capd the hex doubles are parsed by hand and must
          equal the Lean rationals.
  arb     (arb, m3, m5) the certificate's own code (validated_int.krawczyk_assemble, certify_M.krawczyk_test,
          certify_M2.assemble), not the exporter, with this file's conversion of an arb ball to its exact ends: the
          Lean rationals must equal them exactly.

    python3 reproduce.py all            every instance by every route that applies; exit 0 iff all agree
    python3 reproduce.py mirror         the Lean `check` in Fractions on the 8 instance files: the verdict must be the
                                        one the Lean file proves; prints each row's value and the widening of entry
                                        [0][0] at which the verdict flips
    python3 reproduce.py controls       every mutation below must make its comparison fail; exit 0 iff all fire
    python3 reproduce.py capd|arb|m3|m5 [--mutate=NAME] [--route=exact|arb]

Mutations (applied to the data read from the Lean file, i.e. what a wrong exporter would have written):
  transposeJ, swapg, negg (2 unknowns); transposeY, negY, swapY (rows 0 and 1 of Y), transposeYJ, swapYJ (rows 2
  and 3 of YJ), swapw (w_1 and w_2), perturbY=K (Y[0][0] times 1 + 2^-K).
Run with the repository's Python (python-flint is needed for the arb route only).
"""
import json
import os
import re
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROOF = os.path.join(ROOT, "solver", "proof")
SRC = {"capd": ("M1capd", os.path.join(PROOF, "capd_check", "results", "capd_enclosures_1e-9.json")),
       "arb": ("M1arb", os.path.join(PROOF, "validated_int_krawczyk_1e-9.json")),
       "m3": ("M3cert", os.path.join(PROOF, "data", "A_M_box_M3_1e-10.json")),
       "m5": ("M5cert", os.path.join(PROOF, "data", "A_M2_box_M5_1e-10.json"))}


# ---------------------------------------------------------------- the Lean files, read back

def read_lean(name):
    """the fields of the structure in lean/DSS/<name>.lean as Fractions; intervals as (lo, hi)"""
    txt = open(os.path.join(ROOT, "lean", "DSS", name + ".lean"), encoding="utf-8").read()
    body = txt.split(" where\n", 1)[1].split("\n\ntheorem", 1)[0]
    fields = {}
    for mt in re.finditer(r"^  (\w+) := (.*?)(?=^  \w+ := |\Z)", body, re.S | re.M):
        s = mt.group(2)
        s = re.sub(r"\(\((-?\d+) : ℤ\) : ℚ\) / (\d+)", r"Q(\1,\2)", s)
        s = s.replace("![", "[").replace("⟨", "(").replace("⟩", ")")
        assert re.fullmatch(r"[Q()\[\],\-\d\s]*", s), f"unexpected text in field {mt.group(1)} of {name}"
        fields[mt.group(1)] = eval(s, {"__builtins__": {}}, {"Q": Fr})
    fields = {k: _fr(v) for k, v in fields.items()}
    fields["proves"] = "true" if f"{name}.check = true" in txt else ("false" if f"{name}.check = false" in txt else "?")
    fields["sha256"] = re.search(r"sha256 ([0-9a-f]{64})", txt).group(1)
    return fields


def _fr(v):
    if isinstance(v, list):
        return [_fr(x) for x in v]
    if isinstance(v, tuple):
        return tuple(Fr(x) for x in v)
    return Fr(v)


# ---------------------------------------------------------------- Lean's `check`, mirrored in Fractions

def smul(y, a):
    return (min(y * a[0], y * a[1]), max(y * a[0], y * a[1]))


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    return (a[0] - b[1], a[1] - b[0])


def mag(a):
    return max(abs(a[0]), abs(a[1]))


def check2(d):
    """DSS.Krawczyk.Data.check: returns (verdict, [(gmag_i + h C_i) / h])"""
    Y, J, g, h = d["Y"], d["J"], d["g"], d["h"]

    def rowY(i, a):
        return add(smul(Y[i][0], a[0]), smul(Y[i][1], a[1]))
    rows = []
    for i in range(2):
        C = sum(mag(sub((Fr(int(i == j)),) * 2, rowY(i, [J[0][j], J[1][j]]))) for j in range(2))
        rows.append((mag(rowY(i, g)) + h * C) / h if h > 0 else None)
    return h > 0 and all(r < 1 for r in rows), rows


def checkN(b):
    """DSS.Krawczyk.BoxData.check (the rescaling to the unit box, then DataN.check): (verdict, row values)"""
    w, n = b["w"], len(b["w"])
    if not all(x > 0 for x in w):
        return False, None
    rows = []
    for i in range(n):
        yg = smul(1 / w[i], b["Yg"][i])
        C = sum(mag(sub((Fr(int(i == j)),) * 2, smul(w[j] / w[i], b["YJ"][i][j]))) for j in range(n))
        rows.append(mag(yg) + C)
    return all(r < 1 for r in rows), rows


def widen00(d, f, key):
    e = dict(d)
    M = [list(r) for r in d[key]]
    lo, hi = M[0][0]
    mid, rad = (lo + hi) / 2, (hi - lo) / 2
    M[0][0] = (mid - f * rad, mid + f * rad)
    e[key] = M
    return e


def flip_factor(d, chk, key):
    """the widening factor of entry [0][0] at which the verdict becomes false, to 3 digits (bisection in Fractions)"""
    lo, hi = Fr(1), Fr(2)
    while chk(widen00(d, hi, key))[0]:
        lo, hi = hi, hi * 2
        if hi > 10**12:
            return None
    for _ in range(40):
        mid = (lo + hi) / 2
        if chk(widen00(d, mid, key))[0]:
            lo = mid
        else:
            hi = mid
    return float(hi)


def mirror():
    ok = True
    print("Lean `check` mirrored in exact rationals (row value: must be < 1 for every row)")
    for name in ("M1capd", "M1capdWide", "M1arb", "M1arbWide", "M3cert", "M3certWide", "M5cert", "M5certWide"):
        d = read_lean(name)
        two = "J" in d
        v, rows = (check2 if two else checkN)(d)
        agree = (str(v).lower() == d["proves"])
        ok &= agree
        line = f"  {name:11s} mirror {str(v):5s} Lean proves check = {d['proves']:5s} {'agree' if agree else 'DISAGREE'}; rows " + \
            ", ".join(f"{float(r):.4g}" for r in rows)
        if not name.endswith("Wide"):
            ff = flip_factor(d, check2 if two else checkN, "J" if two else "YJ")
            line += f"; [0][0] widened: fails from x{ff:.4g}"
        print(line)
    return ok


# ---------------------------------------------------------------- mutations of the data read from Lean

def mutate(d, mut):
    if not mut:
        return d
    e = {k: ([list(r) if isinstance(r, list) else r for r in v] if isinstance(v, list) else v) for k, v in d.items()}
    T = lambda A: [list(r) for r in zip(*A)]
    if mut == "transposeJ":
        e["J"] = T(d["J"])
    elif mut == "swapg":
        e["g"] = d["g"][::-1]
    elif mut == "negg":
        e["g"] = [(-b, -a) for a, b in d["g"]]
    elif mut == "transposeY":
        e["Y"] = T(d["Y"])
    elif mut == "negY":
        e["Y"] = [[-x for x in r] for r in d["Y"]]
    elif mut == "swapY":
        e["Y"] = [d["Y"][1], d["Y"][0]] + d["Y"][2:]
    elif mut == "transposeYJ":
        e["YJ"] = T(d["YJ"])
    elif mut == "swapYJ":
        e["YJ"] = d["YJ"][:2] + [d["YJ"][3], d["YJ"][2]] + d["YJ"][4:]
    elif mut == "swapw":
        e["w"] = [d["w"][0], d["w"][2], d["w"][1]] + d["w"][3:]
    elif mut.startswith("perturbY="):
        e["Y"][0][0] = d["Y"][0][0] * (1 + Fr(1, 2 ** int(mut.split("=")[1])))
    else:
        raise SystemExit(f"unknown mutation {mut}")
    return e


# ---------------------------------------------------------------- sources, exact route

def hexq(s):
    """a C99 hex double '[-]0x1.hhhp[+-]e' as an exact rational, parsed by hand"""
    mt = re.fullmatch(r"(-?)0x([0-9a-f]+)(?:\.([0-9a-f]*))?p([+-]?\d+)", s.strip().lower())
    sign, ip, fp, ex = mt.group(1), mt.group(2), mt.group(3) or "", int(mt.group(4))
    v = Fr(int(ip + fp, 16)) * Fr(2) ** (ex - 4 * len(fp))
    return -v if sign else v


def ballq(s):
    """an arb string '[m +/- r]', '[+/- r]' or 'm' as the exact decimal interval (lo, hi)"""
    s = s.strip()
    if s.startswith("["):
        s = s[1:-1]
    if "+/-" in s:
        m, r = s.split("+/-")
        m = Fr(m.strip()) if m.strip() else Fr(0)
        r = Fr(r.strip())
        return (m - r, m + r)
    return (Fr(s), Fr(s))


def same(a, b, what, bad):
    if a != b:
        bad.append(what)


def run_capd(mut=None):
    name, path = SRC["capd"]
    L = mutate(read_lean(name), mut)
    d = json.load(open(path))
    bad = []
    xhat = [Fr(x) for x in d["xhat"]]
    assert all(isinstance(x, str) for x in d["xhat"])
    same(L["xhat"], xhat, "xhat is not the file's decimal centre", bad)
    for i in range(2):
        lo, hi = (hexq(t) for t in d["xhat_enclosure"][i])
        if not lo <= L["xhat"][i] <= hi:
            bad.append(f"xhat[{i}] outside CAPD's enclosure of the centre")
        same(L["g"][i], tuple(hexq(t) for t in d["F_at_xhat"][i]), f"g[{i}]", bad)
        for j in range(2):
            same(L["J"][i][j], tuple(hexq(t) for t in d["DF_over_X"][i][j]), f"J[{i}][{j}]", bad)
    hfile = hexq(d["h"])
    if not 0 < L["h"] <= hfile:
        bad.append("the Lean half-width exceeds CAPD's")
    v, rows = check2(L)
    print(f"capd exact: h_Lean / h_CAPD = 1 - {float(1 - L['h'] / hfile):.3e}; g, J, xhat equal the file's hex doubles: "
          f"{not bad}; mirror check {v}, rows {[round(float(r), 4) for r in rows]}")
    return bad


def layout(M):
    """the state layout of certify_M_sys.SysM, as proof/A_M.md states it: D on the odd harmonics -M..M first"""
    odd = [mu for mu in range(-M, M + 1) if mu % 2]
    even = [nu for nu in range(-(M + 1), M + 2) if nu % 2 == 0]
    n = 2 * len(odd) + len(even) + 1
    return odd, n


def exact_products(bx, Y, w):
    """exact interval values of Y g(0) and of the enclosure of Y Dgt over the box, from the JSON and Y alone"""
    M = bx["M"]
    m = M + 1
    odd, n = layout(M)
    S = bx["sets"]
    assert len(S["c"]) == n and len(S["Q"]) == 2 * n
    ell = []
    for nu in range(1, M + 1, 2):
        ip, im = odd.index(nu), odd.index(-nu)
        re_, im_ = [0] * (2 * n), [0] * (2 * n)
        re_[ip], re_[im] = 1, -1                    # Re J_nu = Re D_nu - Re D_-nu
        im_[n + ip], im_[n + im] = 1, 1             # Im J_nu = Im D_nu + Im D_-nu
        ell += [re_, im_]
    Yl = [[sum(Y[i][r] * ell[r][t] for r in range(m)) for t in range(2 * n)] for i in range(m)]

    def fl(x):
        return Fr(float.fromhex(x)) if "p" in x else hexq(x)

    def times(A, B):                                # exact rational matrix product, A m x 2n, B 2n x 2n
        return [[sum(A[i][t] * B[t][j] for t in range(2 * n) if A[i][t]) for j in range(2 * n)] for i in range(m)]

    def dot(row, col):                              # sum of rational times interval
        lo = hi = Fr(0)
        for y, a in zip(row, col):
            if y:
                s = smul(y, a)
                lo, hi = lo + s[0], hi + s[1]
        return (lo, hi)

    def stack(col):                                 # a column of n [re, im] strings -> 2n intervals
        return [ballq(v[0]) for v in col] + [ballq(v[1]) for v in col]

    def setval(cs, Qs, qs, ncol):                   # Y l (c_k + Q q_k) for every column k
        YlQ = times(Yl, [[fl(x) for x in row] for row in Qs])
        out = []
        for k in range(ncol):
            ck = stack([cs[i][k] for i in range(n)])
            qk = [ballq(qs[t][k]) for t in range(2 * n)]
            out.append([add(dot(Yl[i], ck), dot(YlQ[i], qk)) for i in range(m)])
        return [[out[k][i] for k in range(ncol)] for i in range(m)]        # [i][k]

    Yg = [r[0] for r in setval([[v] for v in S["c"]], S["Q"], [[x] for x in S["q"]], 1)]
    G1 = setval(S["cV"], S["QV"], S["qV"], m)
    if "cW" not in S:
        return Yg, G1
    pairs = [(j, k) for j in range(m) for k in range(j, m)]
    pidx = {}
    for p, (j, k) in enumerate(pairs):
        pidx[(j, k)] = pidx[(k, j)] = p
    G2 = setval(S["cW"], S["QW"], S["qW"], len(pairs))
    YJ = [[G1[i][j] for j in range(m)] for i in range(m)]
    for i in range(m):
        for j in range(m):
            for k in range(m):
                t = mag(G2[i][pidx[(j, k)]]) * w[k]              # G2 [-w_k, w_k], exactly
                YJ[i][j] = (YJ[i][j][0] - t, YJ[i][j][1] + t)
    return Yg, YJ


def widths_of(bx):
    M = bx["M"]
    assert isinstance(bx["h"], str) and all(isinstance(x, str) for x in bx["w"])
    return [Fr(bx["h"]), Fr(bx["h"])] + [Fr(bx["w"][(i // 2) % len(bx["w"])]) for i in range(M - 1)]


def contain(Lv, Ev, what, bad, stats):
    """the Lean interval must contain the exact one and be barely wider"""
    if not (Lv[0] <= Ev[0] and Ev[1] <= Lv[1]):
        bad.append(f"{what}: the Lean interval does not contain the exact value")
        return
    wl, we = Lv[1] - Lv[0], Ev[1] - Ev[0]
    ex = float((wl - we) / we) if we else float(wl)
    stats.append(ex)
    if ex > 1e-6:
        bad.append(f"{what}: the Lean interval is {ex:.2e} wider than the exact one (relative)")


def run_box_exact(key, mut=None):
    name, path = SRC[key]
    L = mutate(read_lean(name), mut)
    bx = json.load(open(path))
    m = bx["M"] + 1
    bad, stats = [], []
    ws = widths_of(bx)
    same(L["w"], ws, "w is not (h, h, w3, w3, ...) of the run", bad)
    same(L["xhat"], [Fr(0)] * m, "the chart centre is not 0", bad)
    # the stored parameter box of the run (decimal print of the binary balls 0 +- w) against the Lean half-widths
    dx = bx["sets"].get("dx") or bx["sets"].get("pdx")
    cov = []
    for j in range(m):
        lo, hi = ballq(dx[j][0])
        cov.append(float(hi / L["w"][j] - 1))
        if not (lo <= -L["w"][j] and L["w"][j] <= hi):
            bad.append(f"w[{j}] exceeds the stored half-width of the run")
    Yg, YJ = exact_products(bx, L["Y"], L["w"])
    for i in range(m):
        contain(L["Yg"][i], Yg[i], f"Yg[{i}]", bad, stats)
        for j in range(m):
            contain(L["YJ"][i][j], YJ[i][j], f"YJ[{i}][{j}]", bad, stats)
    detY = det(L["Y"])
    if detY == 0:
        bad.append("det Y = 0")
    v, rows = checkN(L)
    print(f"{key} exact: {m + m * m} intervals, Lean contains the exact value in "
          f"{m + m * m - sum(1 for b in bad if 'contain' in b)}; largest relative excess width "
          f"{max(stats) if stats else float('nan'):.2e}; stored box / Lean box - 1 <= {max(cov):.2e}, >= {min(cov):.2e}; "
          f"det Y != 0: {detY != 0}; mirror check {v}, rows {[round(float(r), 4) for r in rows]}")
    return bad


def det(A):
    A = [list(r) for r in A]
    n, d = len(A), Fr(1)
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] != 0), None)
        if p is None:
            return Fr(0)
        if p != c:
            A[c], A[p] = A[p], A[c]
            d = -d
        d *= A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            for k in range(c, n):
                A[r][k] -= f * A[c][k]
    return d


# ---------------------------------------------------------------- sources, arb route (the certificate's own code)

def ends(x):
    """the exact ends of an arb ball, checked by embedding them back at a precision where they are exact"""
    from flint import arb, fmpq, ctx
    m, e = x.mid().man_exp()
    rm, re_ = x.rad().man_exp()
    mid = Fr(int(m)) * Fr(2) ** int(e)
    rad = Fr(int(rm)) * Fr(2) ** int(re_)
    lo, hi = mid - rad, mid + rad
    old = ctx.prec
    ctx.prec = 8192
    try:
        for q, s in ((lo, -1), (hi, 1)):
            back = arb(fmpq(q.numerator, q.denominator))
            ref = arb(x.mid()) + s * arb(x.rad())
            assert back.rad() == 0 and ref.rad() == 0 and bool(back == ref), "an end is not exact"
    finally:
        ctx.prec = old
    return (lo, hi)


def run_arb(mut=None):
    name, path = SRC["arb"]
    L = mutate(read_lean(name), mut)
    sys.path.insert(0, PROOF)
    import validated_int as VI
    d = json.load(open(path))
    assert all(isinstance(d[k], str) for k in ("A", "wf", "h"))
    ok, K, margins, g_hat, JX, JD, dJD = VI.krawczyk_assemble(d["A"], d["wf"], d["h"], d["xi1"], d["point"], d["box"])
    bad = []
    same(L["xhat"], [Fr(d["A"]), Fr(d["wf"])], "xhat is not the file's decimal centre", bad)
    same(L["h"], Fr(d["h"]), "h is not the file's decimal half-width", bad)
    for i in range(2):
        same(L["g"][i], ends(g_hat[i]), f"g[{i}]", bad)
        for j in range(2):
            same(L["J"][i][j], ends(JX[i][j]), f"J[{i}][{j}]", bad)
    v, rows = check2(L)
    print(f"arb (M = 1) arb route: krawczyk_assemble's own test ok={ok}, margins {[f'{x:.3e}' for x in margins]}; "
          f"g, J equal its balls' exact ends: {not bad}; mirror check {v}, rows {[round(float(r), 4) for r in rows]}")
    return bad


def run_box_arb(key, mut=None):
    name, path = SRC[key]
    L = mutate(read_lean(name), mut)
    sys.path.insert(0, PROOF)
    from flint import arb, arb_mat, ctx
    bx = json.load(open(path))
    m = bx["M"] + 1
    bad = []
    if key == "m3":
        import certify_M as CM
        res = CM.krawczyk_test(bx, verbose=False)
        Yg, YJ = res["Yg"], res["YJ"]
        # Y is not returned by krawczyk_test: the same statements as its pt = None branch
        ctx.prec = CM.KPREC
        ell = CM.jump_rows(CM.CS.SysM(bx["M"]), None)
        c, Q, q, varb = CM.CL.load_lohner(bx["sets"])
        D0 = arb_mat(CM.apply_var(ell, varb))
        Yb = arb_mat([[arb(D0[i, j].mid()) for j in range(m)] for i in range(m)]).inv()
        Y = [[arb(Yb[i, j].mid()) for j in range(m)] for i in range(m)]
        # this Y must give krawczyk_test's products back, midpoint and radius bit for bit
        Yl = arb_mat(Y) * ell
        Yg2, YJ2 = CM.apply_set(Yl, c, Q, q), CM.apply_var(Yl, varb)
        eq = lambda a, b: a.mid().man_exp() == b.mid().man_exp() and a.rad().man_exp() == b.rad().man_exp()
        if not all(eq(Yg2[i], Yg[i]) and all(eq(YJ2[i][j], YJ[i][j]) for j in range(m)) for i in range(m)):
            bad.append("Y recomputed does not give krawczyk_test's Y g and Y Dgt back")
        own = f"ok={res['ok']}, relative margins {[round(r, 3) for r in res['rel_margins']]}"
    else:
        import certify_M2 as C2M
        res = C2M.krawczyk2_test(bx, verbose=False)
        A = C2M.assemble(bx)
        ctx.prec = C2M.KPREC
        Yg, G1, G2 = A["Yg"], A["G1"], A["G2"]
        Y = [[A["Y"][i, j] for j in range(m)] for i in range(m)]
        pidx = C2M.C2.pair_index(m)
        wd = [arb(bx["h"])] * 2 + [arb(bx["w"][(i // 2) % len(bx["w"])]) for i in range(bx["M"] - 1)]
        X = [arb(0, x) for x in wd]
        YJ = [[G1[i][j] + sum((G2[i][pidx[(j, k)]] * X[k] for k in range(m)), arb(0)) for j in range(m)] for i in range(m)]
        own = f"ok={res['ok']}, relative margins {[round(r, 3) for r in res['rel_margins']]}"
    for i in range(m):
        same(L["Yg"][i], ends(Yg[i]), f"Yg[{i}]", bad)
        for j in range(m):
            same(L["YJ"][i][j], ends(YJ[i][j]), f"YJ[{i}][{j}]", bad)
            e = ends(Y[i][j])
            if e[0] != e[1]:
                bad.append(f"Y[{i}][{j}] is not exact")
            same(L["Y"][i][j], e[0], f"Y[{i}][{j}]", bad)
    print(f"{key} arb route: the certificate's own test {own}; Yg, YJ, Y equal its balls' exact ends: {not bad}")
    return bad


# ---------------------------------------------------------------- drivers

ROUTES = {"capd": [("exact", run_capd)], "arb": [("arb", run_arb)],
          "m3": [("exact", lambda mut=None: run_box_exact("m3", mut)), ("arb", lambda mut=None: run_box_arb("m3", mut))],
          "m5": [("exact", lambda mut=None: run_box_exact("m5", mut)), ("arb", lambda mut=None: run_box_arb("m5", mut))]}

CONTROLS = [("capd", "exact", "transposeJ"), ("capd", "exact", "swapg"), ("capd", "exact", "negg"),
            ("arb", "arb", "transposeJ"), ("arb", "arb", "swapg"),
            ("m3", "exact", "transposeY"), ("m3", "exact", "negY"), ("m3", "exact", "swapY"),
            ("m3", "exact", "transposeYJ"), ("m3", "exact", "swapYJ"), ("m3", "exact", "swapw"),
            ("m3", "exact", "perturbY=30"), ("m3", "arb", "transposeY"), ("m3", "arb", "perturbY=760"),
            ("m5", "exact", "transposeY"), ("m5", "exact", "negY"), ("m5", "exact", "swapY"),
            ("m5", "exact", "transposeYJ"), ("m5", "exact", "swapYJ"), ("m5", "exact", "swapw"),
            ("m5", "exact", "perturbY=30"), ("m5", "arb", "transposeY"), ("m5", "arb", "perturbY=760")]
# The exact route sees a relative change of Y[0][0] down to 2^-40 and not 2^-60 (arb's balls are about 1e-7 wider
# than the exact intervals, relatively); the arb route compares Y bit for bit.


def run(key, mut=None, route=None, quiet=False):
    bad = []
    for rname, f in ROUTES[key]:
        if route and rname != route:
            continue
        if quiet:
            old, sys.stdout = sys.stdout, open(os.devnull, "w")
            try:
                b = f(mut)
            finally:
                sys.stdout.close()
                sys.stdout = old
        else:
            b = f(mut)
        bad += [f"{key} {rname}: {x}" for x in b]
    return bad


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    o = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
    cmd = a[0] if a else ""
    if cmd == "mirror":
        return 0 if mirror() else 1
    if cmd == "controls":
        allok = True
        for key, route, mut in CONTROLS:
            bad = run(key, mut, route, quiet=True)
            fired = bool(bad)
            allok &= fired
            print(f"  control {key:4s} {route:5s} {mut:14s}: {'fires' if fired else 'DOES NOT FIRE'}"
                  f" ({len(bad)} differences; first: {bad[0] if bad else '-'})")
        print("controls:", "all fire" if allok else "SOME DO NOT FIRE")
        return 0 if allok else 1
    if cmd in ("all",) + tuple(SRC):
        bad = []
        for key in (SRC if cmd == "all" else [cmd]):
            bad += run(key, o.get("mutate"), o.get("route"))
        for b in bad:
            print("  DIFFERENCE:", b)
        print("reproduction:", "PASS" if not bad else f"FAIL ({len(bad)} differences)")
        return 0 if not bad else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
