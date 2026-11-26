"""One ladder point's record (#1098): Delta of a full fastmul6 solve with each error term as its own field, from the
release logs of its stages (release_ls.jsonl, in the order run) and its final state.

Fields: Delta (40 digits) and Delta_parts; per_harmonic (each released harmonic's change, summed over its accepted
outer steps, and how it was released); and the error terms, each a separate field with its basis:
  err_tail     the harmonics past kmax: the last two released changes c_(k-2), c_k fitted as a geometric tail,
               |c_k| r/(1 - r) with r = |c_k/c_(k-2)| (an estimator; r >= 1 is reported as no bound, null), plus the
               last pending first-order dDelta at release
  err_resid    gain x the final target residual, when --gain is given (first-order bound)
  err_M        |Delta - Delta at --M_ref| (a measured change), when --M_ref is given
  err_steps, err_ks, err_kc, err_kt, err_arith, err_horizon: measured at D 300 and D 1000 (I1098 raw folders) and given
               as values with their basis; --steps etc. override them
  err_sum      the sum of the terms that have values (a conservative total, not a standard deviation)

    python3 hp/ladder_record.py STATE OUT.jsonl LOG.jsonl [LOG.jsonl ...] --start START.npz [--gain G] [--M_ref DELTA]
        [--note TEXT]

START is the first log's input state (its Delta is the base of the first harmonic's change).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpmath import mp  # noqa: E402

from octo_corrector import delta_digits, load_o  # noqa: E402

MEASURED = dict(
    err_steps=(0.0, "every Taylor step halved: 0.0 at D 300 and to 40 digits at D 1000 (I1098-mm5/fix, I1098-dh610/fix)"),
    err_ks=(0.0, "Moebius ks 800 and a 4 at ks 600/800: 0.0 at D 300 (I1098-mm5/horizon300); ks + 200: 0 at D 1000"),
    err_kc=(0.0, "kc + 20: no change at 1e-48 at D 300 and D 1000 (I1098-dh610/budget300, budget1000)"),
    err_kt=(0.0, "kt + 8: no change at 1e-48 at D 300 and D 1000"),
    err_arith=(1e-30, "d/dtheta remainders and lane-precision 1/k: the terms they removed were -2.2e-32 (D 300) and "
                      "-1.2e-24 (D 1000); the remainders' own rounding is 1e-16 of those"),
    err_horizon=(1e-27, "u against Moebius on fastmul6 at D 300: -2.4e-28 at kfree 13 (I1098-mm5/fix6)"),
)


def opts(argv):
    o = dict(gain="", M_ref="", note="", start="")
    pos = []
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            o[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    return pos, o


def main():
    pos, o = opts(sys.argv[1:])
    state, out, logs = pos[0], pos[1], pos[2:]
    z8, meta = load_o(state)
    mp.dps = 50
    recs = []
    for lg in logs:
        with open(lg) as fh:
            recs += [json.loads(line) for line in fh if line.strip()]
    per = {}
    prev = mp.mpf(delta_digits(load_o(o["start"])[0], 40)) if o["start"] else None
    pending, released_by, F_last = None, {}, None
    for r in recs:
        k = r["harm"]
        if r.get("accepted") and "Delta" in r:
            d = mp.mpf(r["Delta"])
            if prev is not None:
                per[k] = per.get(k, mp.mpf(0)) + d - prev
            prev = d
            F_last = r.get("F_new", F_last)
        if r.get("done"):
            released_by[k] = r.get("released_by", "")
            pending = r.get("dDelta_full")
            F_last = r.get("F", F_last)
    ks = sorted(per)
    err_tail = None
    basis_tail = "no two released harmonics with changes"
    if len(ks) >= 2:
        c1, c2 = abs(per[ks[-2]]), abs(per[ks[-1]])
        if c1 > 0 and c2 < c1:
            r = c2 / c1
            err_tail = float(c2 * r / (1 - r)) + abs(pending or 0.0)
            basis_tail = f"geometric from h{ks[-2]} ({mp.nstr(per[ks[-2]], 4)}) and h{ks[-1]} ({mp.nstr(per[ks[-1]], 4)}), " \
                         f"r {mp.nstr(r, 3)}, plus the last pending dDelta {pending}"
        else:
            basis_tail = f"the changes do not fall from h{ks[-2]} to h{ks[-1]}: no tail bound"
    rec = dict(D_str=meta["D_str"], M=int(meta["M"]), kfreeze=meta.get("kfreeze"), horizon=f"mob a={meta.get('hmob')}"
               if meta.get("hmob") else "u", ks=int(meta["ks"]), kc=int(meta["kc"]), kt=int(meta.get("kt", 36)),
               taylor_steps=sum(len(g) - 1 for g in meta["segs"]), Delta=delta_digits(z8, 40),
               Delta_parts=[float(z8[q][-1]) for q in range(8)], F_final=F_last, state=os.path.basename(state),
               per_harmonic={f"h{k}": dict(change=float(per[k]), released_by=released_by.get(k, "")) for k in ks},
               err_tail=err_tail, err_tail_basis=basis_tail, note=o["note"])
    if o["gain"]:
        rec.update(err_resid=float(o["gain"]) * float(F_last or 0.0), err_resid_basis=f"gain {o['gain']} x |F| {F_last}")
    else:
        rec.update(err_resid=None, err_resid_basis="no gain given")
    if o["M_ref"]:
        dm = abs(mp.mpf(rec["Delta"]) - mp.mpf(o["M_ref"]))
        rec.update(err_M=float(dm), err_M_basis=f"|Delta - {o['M_ref']}| (the other M)")
    else:
        rec.update(err_M=None, err_M_basis="no other M given")
    for k, (v, b) in MEASURED.items():
        rec[k] = float(o[k[4:]]) if o.get(k[4:]) else v
        rec[k + "_basis"] = b
    terms = [rec[k] for k in ("err_tail", "err_resid", "err_M") + tuple(MEASURED) if rec.get(k) is not None]
    rec["err_sum"] = float(sum(terms))
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("D_str", "Delta", "err_tail", "err_resid", "err_M", "err_sum")}), flush=True)


if __name__ == "__main__":
    main()
