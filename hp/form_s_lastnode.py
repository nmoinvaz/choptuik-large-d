"""The old layout with its last node at the fast mode's anti-Stokes point (#1088): nodes 0.3, 0.4, 0.5, 0.6 and x_l, the
Moebius horizon series summed at x_l. The fast scalar mode F = x^-n (1 - x^2)^((n + kappa)/2) is O(1) only near
x_c = 1/sqrt 2 - (0.64-0.94)/n and 10^(0.00869 n - 0.146 kappa) larger at 0.7 (the wall's literature note,
scratchpad wall2_literature.md): at 0.7 the selection makes the horizon map exponentially nonlinear (the jumps' wall),
past x_c it hides it (the fine mesh's drift); at x_c neither.

    python3 form_s_lastnode.py ratio OUTDIR --D 2828.4271247461903 --states_s S1,S2,S3 --xl 0.700,0.703,0.705,0.70677,0.709
    python3 form_s_lastnode.py solve OUTDIR --D 2420 --states_s S1,S2,S3 --xl xc    (xc: 1/sqrt 2 - 0.79/n)
    python3 form_s_lastnode.py guess OUTDIR --D 10000 --states_s ... --xl xc --mob 25   (saves the guess state)
    python3 form_s_lastnode.py solvestate OUTDIR --state GUESS.npz                     (solves a saved guess)

ratio: the held system's guess residual per block at each x_l and the horizon block over the largest other block (the
basin measure). solve: held Newton at one x_l (the double-double Jacobian, the octo-double residual, Delta's first-order
bound 1e-15), the quad-double check, the Hamiltonian check, and a direct_s.jsonl record.
The guess: each source state's solution carried from its node 0.7 to x_l by Taylor steps (Solver.path_s), then every
unknown fitted in 1/n as form_s.py jump does.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

from form_s import (HeldS, blocks_s, direct_line, disc_report, ham_check, held_field, ks_mob,  # noqa: E402
                    make_solver_s, newton_ladder, opts, s_to_a, scale_s, unscale_s)
from form_s_fine import inner_z, outer_values  # noqa: E402
from hpkernel.driver import make_solver  # noqa: E402

M = 32


def xc_of(D, c=0.79):
    return 1.0 / np.sqrt(2.0) - c / (float(D) - 2.0)


def sources(globs, D_target, xl, sub, thr):
    """per source: (D, the inner unknowns p1, c0, nodes 0-3 (0.3-0.6), m_h, Delta as float64, its solution at x_l)"""
    from octo_corrector import load_o
    rows = []
    for p in sorted(set(q for g in globs.split(",") if g for q in glob.glob(g))):
        z, m = load_o(p)
        with mp.workdps(60):
            Dm = mp.mpf(m["D_str"])
        zi = inner_z(z, m)  # p1, c0, nodes 0-4, m_h, Delta
        v = np.asarray(zi[0]) + np.asarray(zi[1])
        n4 = v[2 * M + 3 * M * 4:2 * M + 3 * M * 5]
        x4 = float(m["nodes"][4])
        if xl < x4 - 1e-15:
            raise SystemExit(f"x_l {xl} lies inside the source's fifth node {x4}: the path runs outward only")
        nx = n4 if abs(xl - x4) < 1e-15 else outer_values(Dm, z, m, [xl], sub, thr)[0]
        head = np.concatenate([v[:2 * M + 3 * M * 4], nx, v[2 * M + 3 * M * 5:]])
        rows.append((float(m["D"]), head, m))
    rows.sort(key=lambda r: r[0])
    return rows


def guess(rows, D, deg):
    Ds = np.array([r[0] for r in rows])
    Y = np.array([scale_s(r[1], r[0], M, 5) for r in rows])
    c = np.polyfit(1.0 / (Ds - 2.0), Y, deg)
    y = np.array([np.polyval(c[:, j], 1.0 / (float(D) - 2.0)) for j in range(Y.shape[1])])
    return unscale_s(y, float(D), M, 5)


def setup(o, xl, thr):
    """the guess at x_l, its planned layout and the held solver"""
    from octo_corrector import o8
    with mp.workdps(60):
        Dt = mp.mpf(o["D"])
    rows = sources(o["states_s"], float(Dt), xl, int(o["sub"]), thr)
    zs = guess(rows, Dt, int(o["deg"]))
    nodes = list(rows[-1][2]["nodes"])[:4] + [xl]
    L = len(zs)
    z8 = [zs] + [np.zeros(L) for _ in range(7)]
    if o["delta"] != "fit":
        with mp.workdps(60):
            v = mp.mpf(o["delta"])
            for q in range(8):
                f = float(v)
                z8[q][-1] = f
                v -= mp.mpf(f)
    z8 = o8(z8)
    ks = ks_mob(float(Dt)) if o["ks"] == "auto" else int(o["ks"])
    zA = s_to_a(zs, float(Dt), M, 5, nodes)
    m0 = rows[-1][2]
    sa = make_solver(Dt, M, nodes=nodes, kt=int(m0.get("kt", 36)), kc=int(o["kc"]), ks=ks, threads=thr)
    segs = [list(map(float, g)) for g in sa.plan(np.ascontiguousarray(zA), np.zeros(len(zA)), float(o["plan_tol"]),
                                                    float(o["hmax"]))]
    kw = dict(kt=int(m0.get("kt", 36)), kc=int(o["kc"]), ks=ks, threads=thr, segs=segs, mob=float(o["mob"]))
    s = make_solver_s(Dt, M, nodes, kfreeze=3, **kw)
    sh = HeldS(s, M, 5, held_field(z8, M))
    meta = dict(D=float(Dt), D_str=mp.nstr(Dt, 50), M=M, nodes=nodes, kt=kw["kt"], kc=kw["kc"], ks=ks, kfreeze=3, form="S",
                hmob=float(o["mob"]), parts=8, segs=segs, plan_tol=float(o["plan_tol"]), deg=int(o["deg"]),
                held="5-13 at the guess's values", source="form_s_lastnode.py", xl=xl,
                states=[r[0] for r in rows])
    return Dt, rows, z8, sh, meta


def main(argv):
    from octo_corrector import delta_digits, save_o
    from raw_out import pop_allow, require_raw
    cmd = argv[0]
    argv = list(argv[1:])
    allow = pop_allow(argv)
    pos, o = opts(argv, dict(D="2828.4271247461903", states_s="", xl="xc", deg="2", mob="8", ks="auto", kc="250",
                             threads="2", sub="4", plan_tol="1e-40", hmax="0.05", delta="fit", tc="", it_max="10",
                             halvings="8", delta_tol="1e-15", commit="", ham="1", tol="1e-300", parts="8", ladder="dd,qd",
                             it_hi="2", check="auto", state="", disc="1"))
    outdir = pos[0]
    os.makedirs(outdir, exist_ok=True)
    log = os.path.join(outdir, f"lastnode_{cmd}.jsonl")
    require_raw(log, allow=allow)
    thr = int(o["threads"])
    xls = [xc_of(o["D"]) if x == "xc" else float(x) for x in o["xl"].split(",")]
    if cmd == "ratio":
        for xl in xls:
            t0 = time.time()
            Dt, rows, z8, sh, meta = setup(o, xl, thr)
            F = sh.residual_s([np.ascontiguousarray(p) for p in z8[:int(o["parts"])]])
            b = blocks_s(F, M, 5)
            other = max(b[:5])
            r = b[5] / other
            print(f"D {float(Dt):g}, x_l {xl:.6f}: guess's max |F| per block " + " ".join(f"{v:.1e}" for v in b)
                  + f"; horizon / largest other {r:.3e} (log10 {np.log10(r):.2f}); ks {meta['ks']}, "
                  f"{sum(len(g) - 1 for g in meta['segs'])} Taylor steps ({time.time() - t0:.0f} s)", flush=True)
            with open(log, "a") as fh:
                fh.write(json.dumps(dict(D_str=meta["D_str"], xl=xl, blocks=b, ratio=r, ks=meta["ks"], states=meta["states"],
                                         utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))) + "\n")
        return 0
    xl = xls[0]
    t_all = time.time()
    if cmd in ("solvestate", "checkstate"):
        # a guess saved by "guess" (this machine need not have path_s), or for checkstate a solved state: its nodes,
        # plan and held values
        from octo_corrector import load_o
        z8, meta = load_o(o["state"])
        with mp.workdps(60):
            Dt = mp.mpf(meta["D_str"])
        if o["ks"] != "auto":
            meta["ks"] = int(o["ks"])
        kw = dict(kt=int(meta["kt"]), kc=int(meta["kc"]), ks=int(meta["ks"]), threads=thr, segs=meta["segs"],
                  mob=float(meta["hmob"]))
        s = make_solver_s(Dt, M, list(meta["nodes"]), kfreeze=3, **kw)
        sh = HeldS(s, M, 5, held_field(z8, M))
        xl = float(meta["nodes"][4])
    else:
        Dt, rows, z8, sh, meta = setup(o, xl, thr)
    h8 = held_field(z8, M)  # the held field sh holds (HeldS of the start state's own values)
    kwd = dict(kt=int(meta["kt"]), kc=int(meta["kc"]), ks=int(meta["ks"]), threads=thr, kfreeze=3)

    def disc(z, Fq, gain):
        # the discretization tails (kc + 20, ks + 200, halved Taylor steps) against the quad-double residual
        F4 = Fq if len(Fq) == 4 else sh.residual_s([np.ascontiguousarray(z[q]) for q in range(4)])
        return disc_report(z, meta, kwd, float(meta["hmob"]), h8, F4, gain)
    if cmd == "checkstate":
        # the end checks of a saved solve made before they were part of solvestate: the check residual (--check,
        # auto octo-double), the discretization tails, the Hamiltonian check
        check = o["check"] if o["check"] != "auto" else "od"
        t0 = time.time()
        Fq = sh.residual_s([np.ascontiguousarray(z8[q]) for q in range({"qd": 4, "od": 8}[check])])
        Fc = float(np.abs(np.asarray(Fq[0]) + np.asarray(Fq[1])).max())
        gain = float(meta.get("delta_gain") or "nan")
        print(f"D {float(Dt):g} (ks {meta['ks']}): {check} check max |F| {Fc:.3e} ({time.time() - t0:.0f} s); Delta's "
              f"first-order bound {gain * Fc:.2e} (gain {gain:.2e})", flush=True)
        dsc = disc(z8, Fq, gain) if int(o["disc"]) else None
        hc = ham_check(z8, meta, threads=thr) if int(o["ham"]) else None
        if hc:
            print(f"  Hamiltonian code check: a rows up to {max(r['a'] for r in hc.values()):.1e}", flush=True)
        with open(os.path.join(outdir, "lastnode_checkstate.jsonl"), "a") as fh:
            fh.write(json.dumps(dict(state=os.path.basename(o["state"]), D_str=meta["D_str"], check=check, F=Fc,
                                     gain=gain, bound=gain * Fc, disc=dsc,
                                     ham_check={str(k): r for k, r in hc.items()} if hc else None,
                                     utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))) + "\n")
        return 0
    if cmd == "guess":
        F = sh.residual_s([np.ascontiguousarray(p) for p in z8[:2]])
        b = blocks_s(F, M, 5)
        out = os.path.join(outdir, f"guess_D{float(Dt):.4f}.npz")
        save_o(out, z8, dict(meta, guess_blocks=b))
        print(f"D {float(Dt):g}, x_l {xl:.6f} (ks {meta['ks']}, {sum(len(g) - 1 for g in meta['segs'])} Taylor steps): the "
              f"guess's max |F| per block (double-double) " + " ".join(f"{v:.1e}" for v in b)
              + f"; horizon / largest other {b[5] / max(b[:5]):.3e}; saved {out} ({time.time() - t_all:.0f} s)", flush=True)
        return 0
    F = sh.residual_s([np.ascontiguousarray(p) for p in z8[:2]])  # the guess's blocks in double-double (octo-double: the serial horizon at ks 1.3 n costs hours at D 10^4)
    b = blocks_s(F, M, 5)
    print(f"D {float(Dt):g}, x_l {xl:.6f} (ks {meta['ks']}): the guess's max |F| per block " + " ".join(f"{v:.1e}" for v in b)
          + f"; horizon / largest other {b[5] / max(b[:5]):.3e}", flush=True)
    ladder = o["ladder"].split(",")
    if float(Dt) >= 5000 and not set(ladder) & {"qd", "od"}:
        # the double-double residual's floor (about 5e-21) leaves Delta's bound near 1e-12 past D 5000
        raise SystemExit("D >= 5000 needs a quad- or octo-double stage in --ladder")
    step_out = os.path.join(outdir, f"lastnode_D{float(Dt):.4f}_step.npz")

    def on_step(z, rec):
        save_o(step_out, z, dict(meta, step_it=rec["it"], step_res=rec["res"], step_F=rec["F_new"]))
    z8, conv, recs = newton_ladder(sh, z8, float(o["tol"]), ladder, int(o["it_max"]), int(o["it_hi"]),
                                   int(o["halvings"]), log, delta_tol=float(o["delta_tol"]), jac="dd", on_step=on_step)
    res_last = recs[-1].get("res", ladder[0]) if recs else ladder[0]
    check = o["check"] if o["check"] != "auto" else {"dd": "qd", "qd": "od", "od": "od"}[res_last]
    Fq = sh.residual_s([np.ascontiguousarray(z8[q]) for q in range({"qd": 4, "od": 8}[check])])
    Fc = float(np.abs(np.asarray(Fq[0]) + np.asarray(Fq[1])).max())
    gains = [r["gain"] for r in recs if "gain" in r]
    gain = gains[-1] if gains else float("nan")
    meta.update(solved=bool(conv), F=Fc, check=check, delta_gain=gain, delta_bound_first_order=gain * Fc,
                floor_stop=bool(recs and recs[-1].get("floor")))
    print(f"  end state: {check} max |F| {Fc:.3e}; Delta's first-order bound {gain * Fc:.2e} (gain {gain:.2e})", flush=True)
    if int(o["disc"]):
        meta.update(disc=disc(z8, Fq, gain))
    if int(o["ham"]):
        hc = ham_check(z8, meta, threads=thr)
        meta.update(ham_check={str(k): r for k, r in hc.items()})
        print(f"  Hamiltonian code check: a rows up to {max(r['a'] for r in hc.values()):.1e}", flush=True)
    out = os.path.join(outdir, f"lastnode_D{float(Dt):.4f}.npz")
    save_o(out, z8, meta)
    Dl = delta_digits(z8, 40)
    print(f"  {'converged' if conv else 'NOT converged'}; Delta {Dl}; saved {out} ({time.time() - t_all:.0f} s in all)",
          flush=True)
    tc = None
    if o["tc"]:
        tc = dict(value=o["tc"].split(":")[0], err=float(o["tc"].split(":")[1]), source="given")
        with mp.workdps(60):
            print(f"  Delta - reference {mp.nstr(mp.mpf(Dl) - mp.mpf(tc['value']), 6)} (+- {tc['err']:.2e})", flush=True)
    if conv:
        rec = direct_line(z8, meta, out, "held", recs[-1].get("F_new"), res_last, Fc, check, bound=gain * Fc,
                          disc=meta.get("disc"), ham=meta.get("ham_check"), tc=tc, commit=o["commit"],
                          extra=dict(method="lastnode at x_c", valid=True, xl=xl, newton_steps=len(recs),
                                     jac="dd"))
        with open(os.path.join(outdir, "direct_s.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return 0 if conv else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
