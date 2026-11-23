#!/usr/bin/env python3
"""Review of #1079, check 2: do the pieces cited for Theorem A(3) at an excited zero belong to one centre and fit
into one another? Read from the raw files only; nothing is integrated.

    python3 review_1079/sets_check.py RAW 1|2 [--mutate=centre|sixa|g0|path|hull]        from solver/proof
    python3 review_1079/sets_check.py RAW 1|2 --selftest       the clean check and the five mutations

Tests (every one must hold; the theorem is proof/A_lemmas_M.md section 6, with x^ read as this zero's centre):
  centre   the box file's p, the point file's p, the lemma 6a input's x and p, the covering's ray.json x and the centre
           printed by the H1 run are the same decimal strings
  path     box and point files: the same path, start ray 5pi/16 in [3pi/8 - 0.2, 3pi/8], r0 <= 0.1 and inside the
           covering's annulus (|xi| >= 0.062), arc 0.2, xi1 0.3; both start records rigorous
  hull     X_c's hull (formed here from p, the tilt C, h and w) lies in X_b = c +- h_b as arb rounds h_b, and X_b lies
           in X_6 = c +- 1e-6 (H1's box), with the slacks printed
  H1       the stored lemma S run: rp 0.1 (R+ = 0.01), box 1e-6, lemma S, the old sector lemma and the run all ok
  g0       257 >= 1/(R+ sin 0.4) and the covering's first row 262 > 257 (lemma 6b_1's use of lemma S, #864's use 2)
  H2       ray.json is a C^1 run at h_b; the 12 rows have its h and design, 183 discs, all ok; the stored check ok
  sixa     the lemma 6a input is the export of this box file: certify_M.export_6a run here on the box file gives the
           stored state box string for string; its x +- h lies in centre +- 1e-6; xi_a = 0.067 within lemma S's
           real reach; the stored result has the input's xi1 and ok
Mutations (each must end FAIL): centre (the covering of the other zero), sixa (the other zero's lemma 6a input),
g0 (R+ = 0.0099), path (the point file's xi1 relabelled 0.2), hull (h_b = 1e-12).
"""
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import flint  # noqa: E402
from flint import arb, ctx  # noqa: E402
import certify_M as CM  # noqa: E402
import lemma6b1_cover_M as LCM  # noqa: E402

TAGS = {1: ("M3x1E2_h1e-12_w1e-13", "x1E2"), 2: ("M3x2E2_h1e-12_w1e-12", "x2E2")}


def load(path):
    with open(path) as fh:
        return json.load(fh)


def check(raw, z, mutate=None, quiet=False):
    say = (lambda *a: None) if quiet else print
    tag, t = TAGS[z]
    zo = 3 - z
    L = os.path.join(raw, "I1079-macm3", "levers")
    H2 = os.path.join(raw, "I1079-macm3", "h2")
    bx = load(os.path.join(L, "data", f"A_M_box_{tag}.json"))
    pt = load(os.path.join(L, "data", f"A_M_point_M3{t}.json"))
    s6tag = TAGS[zo][0] if mutate == "sixa" else tag
    s6 = load(os.path.join(L, "data", f"A_M_6a_{s6tag}_in.json"))
    r6 = load(os.path.join(L, "lemmas_M_runs", f"lemma6a_{tag}.json"))
    h1 = load(os.path.join(L, "h1", f"lemma6_sector_M3_{tag}.json"))
    with open(os.path.join(L, "h1", f"I1079_sector_{tag}.log")) as fh:
        h1log = fh.readline()
    rdir = os.path.join(H2, f"lemma6b1_runs_M3_exc{zo if mutate == 'centre' else z}")
    ray = load(os.path.join(rdir, "ray.json"))
    chk = load(os.path.join(H2, f"lemma6b1_check_M3_exc{z}.json"))
    if mutate == "path":
        pt = dict(pt, path=pt["path"][:3] + ["0.2"])
    fails = []

    def need(name, cond, msg):
        say(f"  {name}: {msg}: {'ok' if cond else 'FAIL'}")
        if not cond:
            fails.append(name)

    say(f"zero {z} ({tag})" + (f", MUTATION {mutate}" if mutate else ""))
    # centre
    p = list(bx["p"])
    h1x = re.findall(r"'([^']+)'", h1log)
    need("centre", p == pt["p"] == s6["x"] == s6["p"] == ray["x"] == h1x,
         f"p of the box, the point run, the 6a input, ray.json and the H1 log equal ({p[0][:18]}.., {p[1][:18]}..)")
    # path
    ctx.prec = 320
    phi = arb(5) * arb.pi() / 16
    lo, hi = 3 * arb.pi() / 8 - arb("0.2"), 3 * arb.pi() / 8
    r0 = arb(bx["path"][0])
    geo = bx["path"][1] == "5pi/16" and bool(phi >= lo) and bool(phi <= hi) and bool(r0 <= arb("0.1")) and \
        bool(r0 * r0 * 260 >= 1) and bx["path"][2] == "0.2" and bx["path"][3] == "0.3"
    need("path", bx["path"] == pt["path"] and geo and bx["start"]["rigorous"] is True and pt["start"]["rigorous"] is True
         and bx["start"]["fpm_sha256"] == pt["start"]["fpm_sha256"],
         f"box path {bx['path']}, point path {pt['path']}; the ray's angle exceeds 3pi/8 - 0.2 by "
         f"{float((phi - lo).lower()):.4f}; 1/r0^2 = {float((1 / (r0 * r0)).mid()):.1f} <= 260")
    # hull
    ctx.prec = CM.KPREC
    pe = CM.exact_p(p)
    Cm = [[arb(x) for x in r] for r in bx["C"]]
    w = [arb(bx["h"]), arb(bx["h"])] + [arb(bx["w"][(i // 2) % len(bx["w"])]) for i in range(2)]
    hull = [pe[0] + arb(0, w[0]), pe[1] + arb(0, w[1])] + \
           [pe[i] + Cm[i - 2][0] * arb(0, w[0]) + Cm[i - 2][1] * arb(0, w[1]) + arb(0, w[i]) for i in (2, 3)]
    ctx.prec = 320
    c = [arb(v) for v in ray["x"]]
    hb = "1e-12" if mutate == "hull" else ray["h"]
    rb = arb(arb(0, arb(hb)).rad())
    sl = [rb - abs(arb(hull[i].mid()) - c[i]) - arb(hull[i].rad()) for i in range(4)]
    sl6 = arb(h1["box"]) - rb
    need("hull", all(bool(s >= 0) for s in sl) and bool(sl6 >= 0),
         "X_c's hull half-widths " + ", ".join(f"{float(x.rad()):.3e}" for x in hull) + f" in X_b = c +- {hb}: slack "
         + ", ".join(f"{float(s.lower()):+.3e}" for s in sl) + f"; X_b in X_6 = c +- {h1['box']}: slack "
         f"{float(sl6.lower()):.6e}")
    # H1
    need("H1", h1["M"] == 3 and h1["rp"] == "0.1" and h1["box"] == "1e-6" and h1["ok_S"] is True and h1["ok_old"] is True
         and h1["ok"] is True and h1["Z"] == "0.004489",
         f"lemma S {h1['lip_S']}, old sector lemma {h1['lip_old']}, flat estimate {h1['Lambda_plain']}, rp {h1['rp']}, "
         f"box {h1['box']}")
    # g0
    Rp = arb("0.0099") if mutate == "g0" else arb(h1["rp"]) ** 2
    g0 = arb(LCM.CONF[3]["g0"])
    need("g0", bool(g0 >= 1 / (Rp * arb("0.4").sin())) and bool(arb(LCM.CONF[3]["rho0"]) > g0)
         and float(ray["rho"][0]) == float(LCM.CONF[3]["rho0"]),
         f"g0 = {LCM.CONF[3]['g0']} >= 1/(R+ sin 0.4) = {float((1 / (Rp * arb('0.4').sin())).mid()):.2f}; first row "
         f"{ray['rho'][0]} > g0")
    # H2
    design = LCM.design(3)
    nd, bad, same = 0, 0, True
    for rw in design:
        row = load(os.path.join(rdir, f"row_{rw['j']:02d}.json"))
        nd += len(row["discs"])
        bad += sum(1 for d in row["discs"] if not d["ok"])
        same &= row["h"] == ray["h"] and row["rho"] == rw["rho"] and row["r"] == rw["r"] and row["n"] == rw["n"] and \
            [d["t"] for d in row["discs"]] == rw["t"] and row["mutate"] is None
    need("H2", ray.get("c1") is True and ray["M"] == 3 and ray["rho"] == [rw["rho"] for rw in design] and same
         and nd == 183 and bad == 0 and chk["ok"] is True and not chk["fails"] and chk["tiles"] == 183,
         f"C^1 ray at h = {ray['h']}; {len(design)} rows of the design, {nd} discs, {bad} not ok; stored check ok "
         f"{chk['ok']}, tiles {chk['tiles']}, pieces {chk['pieces']}, links {chk['links']}")
    # sixa
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        CM.export_6a(bx, tmp)
    finally:
        sys.stdout.close()
        sys.stdout = stdout
    ex = load(tmp)
    os.rename(tmp, tmp + ".done")
    ctx.prec = 320
    inside = all(bool(abs(arb(a) - arb(b)) + arb(s6["h"]) <= arb(r6["box"])) for a, b in zip(s6["x"], p))
    reach = bool(arb(r6["xia"]) ** 2 <= arb(h1["rp"]) ** 2 / arb(2).sqrt()) and r6["Z"] == h1["Z"]
    need("sixa", ex["yX_xi1"] == s6["yX_xi1"] and ex["h"] == s6["h"] and ex["xi1"] == s6["xi1"] == r6["xi1"] == bx["path"][3]
         and inside and reach and r6["ok"] is True and r6["mutate"] is None and r6["pilot"] is None,
         f"the 6a input's state box equals export_6a of the box file: {ex['yX_xi1'] == s6['yX_xi1']}; x +- {s6['h']} in "
         f"centre +- {r6['box']}: {inside}; xi_a {r6['xia']} in lemma S's real reach: {reach}; rho_6a {r6['rho6a']}, "
         f"ok {r6['ok']}")
    ok = not fails
    print(f"SETS CHECK (zero {z}, mutation {mutate}): " + ("HOLD" if ok else "FAIL: " + ", ".join(fails)))
    return ok, fails


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    raw, z = args[0], int(args[1])
    print(f"python-flint {flint.__version__}, FLINT {flint.__FLINT_VERSION__}")
    if "selftest" in opts:
        want = {None: [], "centre": ["centre", "hull"], "sixa": ["centre", "sixa"], "g0": ["g0"], "path": ["path"],
                "hull": ["hull"]}
        res = []
        for mu, fl in want.items():
            ok, fails = check(raw, z, mu, quiet=True)
            res.append(sorted(fails) == sorted(fl))
            print(f"  mutation {mu}: fails {fails} (must be {fl}): {'ok' if res[-1] else 'WRONG'}")
        print(f"SETS CHECK SELFTEST (zero {z}): {'PASS' if all(res) else 'FAIL'}")
        return 0 if all(res) else 1
    ok, _ = check(raw, z, opts.get("mutate"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
