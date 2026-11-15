#!/usr/bin/env python3
"""The start sets of lemmas 6a(M) and 6b_1(M) against the start-value lemma (#864): where do the lemma runs that
Theorem A(M) cites use step (i) (proof/A_lemmas_M.md 2), and is each use proved on a set that contains what the run
needs? The counterpart, for the lemmas, of start_cover_M2.py (the point certificate's start data, hypothesis (H0)).

The uses, read off the two lemma scripts (M = 3 and 5 alike):
  [1] lemma 6b_1, the start set at s0 = rho0 e^{-3 i pi/4} (lemma6b1_cover_M.start_data_u, C^1 mode): the old sector
      lemma (fixed_point_M.run_M on the rays |arg xi - 3pi/8| <= 0.2, |xi| <= START_R0[M], diag splitting) at the
      covering's centre c and on the complex box c +- (h + 1e-3), imaginary parts +- 1e-3. start_data_u raises unless
      both hold (lemma6b1_cover_M.py: "old sector lemma failed"), and unless the start point lies inside the lemma's
      radius, so this use is PROVED IN THE RUN. The values come from the point run, the derivative columns over
      X_b = c +- h from jets and Cauchy's estimate on the complex box.
  [2] lemma 6b_1, the check (lemma6b1_cover.check through patched_check): the radius g0 of G0, beyond which y+ is
      analytic by lemma 6b_0(M), i.e. lemma S(M) with R+ = rp^2 (g0 >= 1/(R+ sin 0.4)). The covering does not run
      lemma S; it takes g0 as a constant (CONF). So this use rests on a STORED RUN of lemma6_sector_M.py over
      x^ +- box, which must contain X_b.
  [3] lemma 6a (lemma6_jump_M.py): lemma S(M) and the flat estimate over centre +- hbox, the centre being --x or the
      stored zero x^. The script runs both and exits unless they hold and unless its input box x +- h lies inside
      (its "inside" test), so this use is PROVED IN THE RUN. It does not use the old sector lemma.

The stored runs (RUNS below) are those proof/A_lemmas_M.md cites for Theorem A(3) and A(5). Every set is rebuilt in
arb at 320 bits by the expressions of the code that ran it. Tests, per use:
  [1] (a) X_b lies in the complex box (ball in ball); (b) the diag side condition there, M W0 sup|Im w/w_inf| <= 3/2
      (A_M_review.md F3); (c) the start point: 1/rho0 < START_R0[M]^2.
  [2] (a) g0 >= 1/(R+ sin 0.4) and rho0 > g0; (b) the stored lemma S run holds (ok_S) and X_b lies in its box
      x^ +- box (ball in ball).
  [3] (a) the 6a input box x +- h lies in centre +- hbox: |x_i - centre_i| + h <= hbox, the script's own test.
Verdict COVERED (exit 0) when every test holds, NOT COVERED (exit 1) with the failing parts otherwise.

    python3 start_cover_lemmas_M.py --M=3
    python3 start_cover_lemmas_M.py --M=5 --raw=../../../choptuik-large-d-raw     (the M = 5 runs are raw output)
    python3 start_cover_lemmas_M.py --M=.. --rerun
        also reruns each use on its own sets (one process, under a minute per M):
        [1] start_data_u at the stored centre: its two fixed points must hold, and its values and derivative columns
            must overlap the stored C^1 start set (sets["0"] of ray.json) in every component;
        [2] lemma S over x^ +- box: must hold and reproduce the stored contraction;
        [3] lemma S and the flat estimate at the 6a run's centre: must hold and reproduce the stored Lambda.
    python3 start_cover_lemmas_M.py --M=.. --mutate=widen|edge|centre|g0|rho0
        widen   the covering's h and the 6a box's h set to 2e-6, past x^ +- 1e-6 and past centre +- hbox
        edge    those h set 9e-11 past the edge
        centre  the covering's centre and the 6a box moved by 2e-6 in w/w_inf
        g0      the covering's g0 lowered below 1/(R+ sin 0.4)
        rho0    the start radius rho0 lowered to 0.9 / START_R0[M]^2: the start point leaves the lemma's radius
      Each must give NOT COVERED (exit 1).
    python3 start_cover_lemmas_M.py --M=.. --mutate=edge0     null mutation, h 2e-10 inside the edge: stays COVERED
    python3 start_cover_lemmas_M.py --M=.. --selftest          the clean check, edge0 and the five mutations; exit 0
                                                               only if each gives what it must
"""
import json
import os
import sys
import time

import flint
from flint import acb, arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fixed_point_M as FPM  # noqa: E402
import lemma6_sector_M as L6SM  # noqa: E402
import lemma6b1_cover_M as LCM  # noqa: E402

PREC = 320
CAUCHY = "1e-3"               # start_data_u's default Cauchy radius (ray_run_c1 passes none)
SECTOR = "0.2"                # start_data_u's sector half-width (its literal argument)
MUTATIONS = ("widen", "edge", "centre", "g0", "rho0")
# the stored runs Theorem A(M) cites (proof/A_lemmas_M.md, the summary table and section 7). RAW/ is the raw-data
# clone (--raw). six_a_x: the file holding the --x of the 6a run (None: the run had no --x and used the stored zero).
RUNS = {
    3: dict(ray="lemma6b1_runs_M3_cert/ray.json",
            sector="lemmas_M_runs/lemma6_sector_M3_rp0.1_1e-6.json",
            six_a="lemmas_M_runs/lemma6a_M3_cert.json", six_a_box="data/A_M_6a_M3_1e-10_in.json", six_a_x=None,
            six_a_rp="0.1", six_a_k=20),
    5: dict(ray="RAW/I38-macm3/lemma6b1_runs_M5_cert/ray.json",
            sector="lemmas_M_runs/lemma6_sector_M5_rp0.08_1e-6.json",
            six_a="RAW/I184-macm3/lemma6a_M5_sub1e-18_phat.json", six_a_box="RAW/I103-macm3/A_M2_6a_M5_sub1e-18.json",
            six_a_x="RAW/I184-macm3/centre_passed.txt", six_a_rp="0.08", six_a_k=20),
}
ZEROS0 = {M: list(v) for M, v in FPM.ZEROS.items()}      # the stored zeros x^, before any run changes FPM.ZEROS


def pnames(M):
    return ["A", "w/w_inf"] + [f"{t}{nu}" for nu in range(3, M + 1, 2) for t in ("B", "C")]


def path_of(name, raw):
    if name.startswith("RAW/"):
        if raw is None:
            sys.exit(f"{name} is raw output: pass --raw=<root of the choptuik-large-d-raw clone>")
        return os.path.join(raw, name[4:])
    return os.path.join(HERE, name)


def load(name, raw):
    with open(path_of(name, raw)) as fh:
        return json.load(fh)


def slack_in(inner_c, inner_h, outer_c, outer_h):
    """min_i (outer_h - |inner_c_i - outer_c_i| - inner_h) as a float, and the index of the minimum"""
    s = [float((arb(outer_h) - abs(inner_c[i] - outer_c[i]) - arb(inner_h)).mid()) for i in range(len(inner_c))]
    i = min(range(len(s)), key=lambda j: s[j])
    return s[i], i


def gather(M, raw, mutate=None):
    """the configuration and the sets of the three uses, from the stored runs"""
    ctx.prec = PREC
    rn = RUNS[M]
    ray, sec, six = load(rn["ray"], raw), load(rn["sector"], raw), load(rn["six_a"], raw)
    sbox = load(rn["six_a_box"], raw)
    assert ray["M"] == M and ray.get("c1"), "ray.json is not a C^1 run for this M"
    xhat = [arb(v) for v in ZEROS0[M]]
    c = [arb(v) for v in ray["x"]]
    h = ray["h"]
    g0, rho0 = LCM.CONF[M]["g0"], str(LCM.CONF[M]["rho0"])
    assert float(ray["rho"][0]) == float(rho0), "ray.json was made with another rho0"
    r0 = LCM.START_R0[M]
    # the 6a run: its centre (--x or the stored zero) and its input box (x, or the centre when the file has no x)
    if rn["six_a_x"]:
        with open(path_of(rn["six_a_x"], raw)) as fh:
            cen6s = fh.read().strip().split(",")
    else:
        cen6s = list(ZEROS0[M])
    x6s = sbox.get("x", cen6s)
    cen6, x6, h6, hbox = [arb(v) for v in cen6s], [arb(v) for v in x6s], sbox["h"], six["box"]
    # mutations of the consumers' side
    lim_b = arb(sec["box"]) - max((abs(c[i] - xhat[i]) for i in range(M + 1)), key=lambda v: float(v.mid()))
    lim_6 = arb(hbox) - max((abs(x6[i] - cen6[i]) for i in range(M + 1)), key=lambda v: float(v.mid()))
    if mutate == "widen":
        h = h6 = "2e-6"
    elif mutate in ("edge", "edge0"):
        d = arb("9e-11") if mutate == "edge" else -arb("2e-10")
        h, h6 = (lim_b + d).mid().str(30, radius=False), (lim_6 + d).mid().str(30, radius=False)
    elif mutate == "centre":
        c[1] = c[1] + arb("2e-6")
        x6[1] = x6[1] + arb("2e-6")
    elif mutate == "g0":
        g0 = str(int(0.75 / (float(sec["rp"]) ** 2 * 0.389418342)))
    elif mutate == "rho0":
        rho0 = str(int(0.9 / float(r0) ** 2))
    cc = arb(CAUCHY)
    return dict(M=M, mutate=mutate, runs=rn, xhat=xhat, c=c, cs=ray["x"], h=h, rho0=rho0, g0=g0, r0=r0,
                Xb=[acb(arb(v.mid(), arb(h))) for v in c],
                SC=[acb(arb(v.mid(), arb(h) + cc), arb(0, cc)) for v in c],
                sec=sec, six=six, cen6=cen6, cen6s=cen6s, x6=x6, h6=h6, hbox=hbox, sbox_has_x="x" in sbox)


def check(g, quiet=False):
    """the tests of the three uses; returns (ok, list of failing parts)"""
    ctx.prec = PREC
    M, m = g["M"], g["M"] + 1
    names = pnames(M)
    fails = []
    say = (lambda *a: None) if quiet else print
    say(f"M = {M}" + (f", MUTATION {g['mutate']}" if g["mutate"] else ""))
    # ---- [1]
    say(f"[1] lemma 6b_1, start set at s0 = {g['rho0']} e^(-3 i pi/4)  [{g['runs']['ray']}]")
    say(f"    old sector lemma: rays |arg xi - 3pi/8| <= {SECTOR}, r0 = {g['r0']}, diag; at the centre c = "
        f"({', '.join(s[:16] for s in g['cs'])}) and on c +- ({g['h']} + {CAUCHY}), imaginary +- {CAUCHY}")
    a1 = all(g["SC"][i].contains(g["Xb"][i]) for i in range(m))
    imw = float((M * FPM.W0() * g["SC"][1].imag.abs_upper()).upper())
    b1 = bool(M * FPM.W0() * g["SC"][1].imag.abs_upper() <= arb(1.5))
    c1 = bool(1 / arb(g["rho0"]) < arb(g["r0"]) ** 2)
    say(f"    (a) X_b = c +- {g['h']} inside the complex box: {a1}")
    say(f"    (b) diag side condition M W0 sup|Im w/w_inf| = {imw:.4f} <= 3/2: {b1}")
    say(f"    (c) start point |z0| = 1/{g['rho0']} = {1 / float(g['rho0']):.6f} < r0^2 = {float(g['r0']) ** 2:.6f}: {c1}")
    say(f"    -> {'proved in the run (start_data_u raises otherwise)' if a1 and b1 and c1 else 'NOT COVERED'}")
    fails += [p for p, okp in (("[1](a) X_b outside the complex box", a1), ("[1](b) diag side condition", b1),
                               ("[1](c) start point outside the lemma's radius", c1)) if not okp]
    # ---- [2]
    sec = g["sec"]
    Rp = arb(sec["rp"]) ** 2
    need = 1 / (Rp * arb("0.4").sin())
    a2 = bool(arb(g["g0"]) >= need) and bool(arb(g["rho0"]) > arb(g["g0"]))
    X6 = [arb(v.mid(), arb(sec["box"])) for v in g["xhat"]]
    b2in = all(X6[i].contains(g["Xb"][i].real) for i in range(m))
    sl, si = slack_in(g["c"], g["h"], g["xhat"], sec["box"])
    b2 = bool(sec["ok_S"]) and b2in
    say(f"[2] lemma 6b_1, the check's G0 = {{|u| > {g['g0']}}} from lemma S({M})  [{g['runs']['sector']}]")
    say(f"    stored run: R+ = {sec['rp']}^2, k = {sec['k']}, N = {sec['N']}, over x^ +- {sec['box']}, holds "
        f"{sec['ok_S']}, contraction {float(arb(sec['lip_S']).mid()):.4f}")
    say(f"    (a) g0 = {g['g0']} >= 1/(R+ sin 0.4) = {float(need.mid()):.2f} and rho0 = {g['rho0']} > g0: {a2}")
    say(f"    (b) X_b inside x^ +- {sec['box']}: {b2in} (slack {sl:+.3e} at {names[si]})")
    say(f"    -> {'covered by the stored run' if a2 and b2 else 'NOT COVERED'}")
    fails += [p for p, okp in (("[2](a) g0 below 1/(R+ sin 0.4), or rho0 not beyond g0", a2),
                               ("[2](b) X_b outside the stored lemma S run's box", b2)) if not okp]
    # ---- [3]
    six = g["six"]
    a3 = all(bool(abs(g["x6"][i] - g["cen6"][i]) + arb(g["h6"]) <= arb(g["hbox"])) for i in range(m))
    sl6, si6 = slack_in(g["x6"], g["h6"], g["cen6"], g["hbox"])
    where = "the run's --x" if g["runs"]["six_a_x"] else "the stored zero x^ (no --x)"
    say(f"[3] lemma 6a, lemma S and the flat estimate  [{g['runs']['six_a']}]")
    say(f"    in the run: R+ = {g['runs']['six_a_rp']}^2, k = {g['runs']['six_a_k']}, over centre +- {g['hbox']}, the "
        f"centre being {where}: ({', '.join(s[:16] for s in g['cen6s'])}); Z = {six['Z']}, stored Lambda "
        f"{float(arb(six['Lambda']).mid()):.4f}, rho_6a {float(arb(six['rho6a']).mid()):.4f}, ok {six['ok']}")
    say(f"    (a) the input box x +- {g['h6']} ({'its own x' if g['sbox_has_x'] else 'x = the centre: the file has no x'}) "
        f"inside centre +- {g['hbox']}: {a3} (slack {sl6:+.3e} at {names[si6]})")
    say(f"    -> {'proved in the run (lemma6_jump_M.py exits otherwise)' if a3 and six['ok'] else 'NOT COVERED'}")
    fails += [p for p, okp in (("[3](a) the 6a input box outside centre +- hbox", a3),
                               ("[3] the stored 6a run did not hold", bool(six["ok"]))) if not okp]
    # ---- for reference (hypothesis H2 of section 6; not part of the verdict): the 6a input box, as the decimal box
    # x +- h, against X_b as the covering computed it. arb rounds a ball's radius up to a 30-bit number, so the
    # covering's parameter ball is slightly wider than c +- h.
    off = [float(abs(g["x6"][i] - g["c"][i]).mid()) for i in range(m)]
    io = max(range(m), key=lambda i: off[i])
    excess = float((arb(g["Xb"][0].real.rad()) - arb(g["h"])).mid())
    slr = [float((arb(g["Xb"][i].real.rad()) - abs(g["x6"][i] - g["c"][i]) - arb(g["h6"])).mid()) for i in range(m)]
    ir = min(range(m), key=lambda i: slr[i])
    say(f"    for reference (H2): the 6a input box x +- {g['h6']} against X_b: the centres differ by up to "
        f"{off[io]:.3e} (at {names[io]}); the covering's ball has radius {g['h']} + {excess:.3e}; inside: "
        f"{slr[ir] >= 0} (slack {slr[ir]:+.3e} at {names[ir]})")
    return not fails, fails


def verdict(M, raw, mutate=None, quiet=False):
    g = gather(M, raw, mutate)
    ok, fails = check(g, quiet)
    print(f"START COVER LEMMAS (M = {M}, mutation {mutate}): " + ("COVERED" if ok else "NOT COVERED: " + "; ".join(fails)))
    return ok, g


def rerun(g, raw):
    """each use rerun on its own sets, by the calls of the lemma scripts"""
    import fixed_point_M_b as B                          # only its reader of the stored Lohner set
    M, m = g["M"], g["M"] + 1
    ok = True
    t0 = time.time()
    # [1] start_data_u at the stored centre
    caught = []
    orig = FPM.run_M

    def spy(*a, **kw):
        o = orig(*a, **kw)
        caught.append(o)
        return o
    FPM.ZEROS[M] = list(g["cs"])
    FPM.run_M = spy
    try:
        ctx.prec = PREC
        u0 = acb(arb(g["rho0"])) * (acb(0, -1) * LCM.UM.PHI_U).exp()
        vals, cols = LCM.start_data_u(M, g["h"], u0)
        held = True
    except (RuntimeError, ValueError) as e:
        held = False
        print(f"rerun [1]: start_data_u raised ({e})")
    finally:
        FPM.run_M = orig
        FPM.ZEROS[M] = list(ZEROS0[M])
    if held:
        P, C = caught[0], caught[1]
        _, svals, scols = B.stored_set(path_of(g["runs"]["ray"], raw))
        nd = len(vals)
        ctx.prec = PREC
        ov = sum(bool(vals[i].overlaps(svals[i])) for i in range(nd))
        ins = sum(bool(svals[i].contains(vals[i])) for i in range(nd))
        ovc = sum(bool(cols[j][i].overlaps(scols[j][i])) for j in range(m) for i in range(nd))
        good = P["ok"] and C["ok"] and ov == nd and ovc == nd * m
        print(f"rerun [1]: old sector lemma at c: N {P['N']}, k {P['k']}, contraction {float(P['lip'].upper()):.4f}, "
              f"holds {P['ok']}; on the complex box: contraction {float(C['lip'].upper()):.4f}, holds {C['ok']}; "
              f"start set against the stored one: {ov} of {nd} values overlap ({ins} inside the stored balls), "
              f"{ovc} of {nd * m} derivative entries overlap: {good}  [{time.time() - t0:.0f} s]", flush=True)
        ok = ok and good
    else:
        ok = False
    # [2] lemma S over x^ +- box
    sec = g["sec"]
    S = L6SM.lemma_S_M(M, sec["rp"], int(sec["k"]), sec["box"], verbose=False)
    lip, stored = float(S["lip"].upper()), float(arb(sec["lip_S"]).mid())
    good = bool(S["ok"]) and abs(lip - stored) <= 1e-8 * stored and S["N"] == sec["N"]
    print(f"rerun [2]: lemma S({M}) over x^ +- {sec['box']} (R+ = {sec['rp']}^2, k = {sec['k']}): N {S['N']}, "
          f"contraction {lip:.10f} (stored {stored:.10f}), holds {S['ok']}: {good}  [{time.time() - t0:.0f} s]",
          flush=True)
    ok = ok and good
    # [3] lemma S and the flat estimate at the 6a run's centre
    six, rn = g["six"], g["runs"]
    FPM.ZEROS[M] = list(g["cen6s"])
    try:
        S6 = L6SM.lemma_S_M(M, rn["six_a_rp"], rn["six_a_k"], g["hbox"], verbose=False)
        R6 = L6SM.flat_estimate_M(S6, six["Z"], diag=False, verbose=False)
    finally:
        FPM.ZEROS[M] = list(ZEROS0[M])
    lam, stored = float(R6["lam"].upper()), float(arb(six["Lambda"]).mid())
    good = bool(S6["ok"] and R6["ok"]) and abs(lam - stored) <= 1e-8 * stored
    print(f"rerun [3]: lemma S({M}) over the 6a centre +- {g['hbox']}: contraction {float(S6['lip'].upper()):.4f}, holds "
          f"{S6['ok']}; flat estimate at Z = {six['Z']}: Lambda {lam:.10f} (stored {stored:.10f}), holds {R6['ok']}: "
          f"{good}  [{time.time() - t0:.0f} s]", flush=True)
    ok = ok and good
    print(f"RERUN (M = {M}): {'PASS' if ok else 'FAIL'}")
    return ok


def selftest(M, raw):
    res = []
    for mut, want in [(None, True), ("edge0", True)] + [(mu, False) for mu in MUTATIONS]:
        ok, _ = verdict(M, raw, mut, quiet=True)
        res.append(ok == want)
        print(f"  mutation {mut}: covered {ok} (must be {want}): {'ok' if ok == want else 'WRONG'}")
    print(f"SELFTEST (M = {M}): {'PASS' if all(res) else 'FAIL'}")
    return all(res)


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    M = int(opts.get("M", "5"))
    raw = opts.get("raw")
    print(f"python-flint {flint.__version__}, FLINT {flint.__FLINT_VERSION__}, {PREC} bits")
    if "selftest" in opts:
        sys.exit(0 if selftest(M, raw) else 1)
    ok, g = verdict(M, raw, opts.get("mutate"))
    if "rerun" in opts and ok and not opts.get("mutate"):
        ok = rerun(g, raw)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
