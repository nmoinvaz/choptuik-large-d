"""The two-stage solve past the old layout's basin (#1088): a fine-mesh state (form_s_fine.py jump, which converges but
does not impose the selection) gives m_h, Delta, p1, c0 and the inner nodes; its node 0.7 is replaced by the Moebius
series' sum there from that m_h (hp's octo-double horizon, read as the horizon block of a residual whose node 0.7
holds zeros), and the old layout (nodes 0.3 ... 0.7, the series summed at 0.7, which carries the selection) is solved by
held Newton from it, p1's harmonics 5-13 held at the fine state's values; the lean defaults (double-double Jacobian,
octo-double residual, Delta's first-order bound), then the quad-double check and the Hamiltonian code check.

    python3 form_s_hybrid.py FINE_STATE.npz OUTDIR [--mob 8 --ks auto --kc 250 --threads 2 --tc VALUE:ERR]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

from form_s import (HeldS, blocks_s, direct_line, ham_check, held_field, ks_mob, make_solver_s, newton_o_s,  # noqa: E402
                    opts, s_to_a)
from hpkernel.driver import make_solver  # noqa: E402

M = 32


def main(argv):
    from octo_corrector import delta_digits, load_o, o8, save_o
    from raw_out import pop_allow, require_raw
    argv = list(argv)
    allow = pop_allow(argv)
    pos, o = opts(argv, dict(mob="8", ks="auto", kc="250", threads="2", tol="1e-300", it_max="10", halvings="8",
                             delta_tol="1e-15", plan_tol="1e-40", hmax="0.05", tc="", commit="", ham="1"))
    path, outdir = pos[0], pos[1]
    os.makedirs(outdir, exist_ok=True)
    log = os.path.join(outdir, "hybrid.jsonl")
    require_raw(log, allow=allow)
    t_all = time.time()
    z8f, mf = load_o(path)
    Kf = len(mf["nodes"])
    nodes = list(mf["nodes"])[:5]
    K = 5
    with mp.workdps(60):
        D = mp.mpf(mf["D_str"])
    mob = float(o["mob"])
    ks = ks_mob(float(D)) if o["ks"] == "auto" else int(o["ks"])
    thr = int(o["threads"])
    head = 2 * M + 3 * M * 4  # p1, c0, nodes 0-3
    # the old layout's vector from the fine state: node 4 (0.7) zero for now
    zc = []
    for q in range(8):
        v = np.asarray(z8f[q])
        mh = v[2 * M + 3 * M * Kf:2 * M + 3 * M * Kf + M]
        zc.append(np.ascontiguousarray(np.concatenate([v[:head], np.zeros(3 * M), mh, v[-1:]])))
    # the inner Taylor steps: the fine state's own plan for segments 1-4 (nodes 0.3 ... 0.7)
    segs = [list(map(float, g)) for g in mf["segs"][:4]]
    kw = dict(kt=int(mf.get("kt", 36)), kc=int(o["kc"]), ks=ks, threads=thr, segs=segs, mob=mob)
    s0 = make_solver_s(D, M, nodes, **kw)
    t0 = time.time()
    F = s0.residual_s(zc)
    r0 = K * 3 * M
    for q in range(8):
        zc[q][2 * M + 3 * M * 4:2 * M + 3 * M * 5] = np.asarray(F[q])[r0:r0 + 3 * M]
    zc = o8(zc)
    node4_fine = (np.asarray(z8f[0]) + np.asarray(z8f[1]))[2 * M + 3 * M * 4:2 * M + 3 * M * 5]
    node4_ser = np.asarray(zc[0])[2 * M + 3 * M * 4:2 * M + 3 * M * 5]
    print(f"D {float(D):g}: node 0.7 from the series at the fine state's m_h ({time.time() - t0:.0f} s): it differs from "
          f"the fine state's node 0.7 by {np.abs(node4_ser - node4_fine).max():.3e} (max |p| {np.abs(node4_ser[:M]).max():.4e})",
          flush=True)
    s = make_solver_s(D, M, nodes, kfreeze=3, **kw)
    h8 = held_field(z8f, M)
    sh = HeldS(s, M, K, h8)
    F = sh.residual_s([np.ascontiguousarray(zc[q]) for q in range(8)])
    b = blocks_s(F, M, K)
    print(f"  the hybrid guess on the old layout (ks {ks}): max |F| per block " + " ".join(f"{x:.1e}" for x in b), flush=True)
    meta = dict(mf, nodes=nodes, segs=segs, ks=ks, kc=kw["kc"], kfreeze=3, hmob=mob, held="5-13 at the fine state's values",
                source="form_s_hybrid.py", hybrid_from=os.path.basename(path))
    meta.pop("mesh", None)
    with open(log, "a") as fh:
        fh.write(json.dumps(dict(start="hybrid guess", D_str=mf["D_str"], blocks=b, node07_diff=float(np.abs(node4_ser - node4_fine).max()),
                                 utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))) + "\n")
    z8, conv, recs = newton_o_s(sh, zc, float(o["tol"]), int(o["it_max"]), int(o["halvings"]), log, jac="dd", res="od",
                                delta_tol=float(o["delta_tol"]))
    F4 = sh.residual_s([np.ascontiguousarray(z8[q]) for q in range(4)])
    Fc = float(np.abs(np.asarray(F4[0]) + np.asarray(F4[1])).max())
    gains = [r["gain"] for r in recs if "gain" in r]
    gain = gains[-1] if gains else float("nan")
    meta.update(solved=bool(conv), F=Fc, check="qd", delta_gain=gain, delta_bound_first_order=gain * Fc)
    print(f"  end state: quad-double max |F| {Fc:.3e}; Delta's first-order bound {gain * Fc:.2e} (gain {gain:.2e})", flush=True)
    if int(o["ham"]):
        hc = ham_check(z8, meta, threads=thr)
        meta.update(ham_check={str(k): r for k, r in hc.items()})
        print(f"  Hamiltonian code check: a rows up to {max(r['a'] for r in hc.values()):.1e}", flush=True)
    out = os.path.join(outdir, f"hybrid_D{float(D):.4f}.npz")
    save_o(out, z8, meta)
    Dl = delta_digits(z8, 40)
    with mp.workdps(60):
        dfine = float(mp.mpf(Dl) - mp.mpf(delta_digits(z8f, 40)))
    print(f"  {'converged' if conv else 'NOT converged'}; Delta {Dl} ({dfine:+.3e} from the fine state's); saved {out} "
          f"({time.time() - t_all:.0f} s in all)", flush=True)
    tc = None
    if o["tc"]:
        tc = dict(value=o["tc"].split(":")[0], err=float(o["tc"].split(":")[1]), source="given")
        with mp.workdps(60):
            print(f"  Delta - Track C {mp.nstr(mp.mpf(Dl) - mp.mpf(tc['value']), 6)} (+- {tc['err']:.2e})", flush=True)
    if conv:
        rec = direct_line(z8, meta, out, "held", recs[-1].get("F_new"), "od", Fc, "qd", bound=gain * Fc,
                          ham=meta.get("ham_check"), tc=tc, commit=o["commit"],
                          extra=dict(method="hybrid (fine m_h, old-layout Newton)", valid=True, newton_steps=len(recs),
                                     hybrid_from=os.path.basename(path)))
        with open(os.path.join(outdir, "direct_s.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return 0 if conv else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
