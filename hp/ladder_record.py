"""One ladder point's record (#1098): Delta of a full fastmul6 solve with each error term as its own field, from the
release logs of its stages (release_ls.jsonl, in the order run) and its final state.

Fields: Delta (40 digits) and Delta_parts; per_harmonic (each released harmonic's change, summed over its accepted
outer steps, and how it was released); and the error terms, each a separate field with its basis:
  err_resid    a first-order bound at kfree k* (--kstar): gain (gainq.py at kfreeze k*, --gain) x that system's |F| at
               harmonic k*'s release (from the logs, or --F_kstar). Past some kfree the gain is so large (1.5e47 at D 300,
               M 32, kfree 31) that gain x |F| bounds nothing, so k* is the last kfree whose bound is useful
  err_tail     an estimator: the last pending first-order dDelta at release and the geometric remainder past the last
               released harmonic (from the last two changes c_(k-2), c_k: |c_k| r/(1 - r), r = |c_k/c_(k-2)|; none when
               r >= 1). The released changes past k* (changes_beyond_kstar) are measured values with no first-order bound
               of their own (the gain at kfree 17 is 6.4e30 at D 300, M 32); the err_xcheck fields test them
  err_M        |Delta - Delta at --M_ref| (a measured change), or --err_M with --M_basis
  err_steps, err_ks, err_kc, err_kt, err_arith, err_horizon: measured at D 300 and D 1000 (I1098 raw folders) and given
               as values with their basis; --steps etc. and --err_horizon/--horizon_basis override them
  err_xcheck_<name>  cross-checks (--xcheck "name|value|basis;..."), recorded and not added
  err_sum      the sum of err_resid, err_tail, err_M and the measured terms: it mixes a bound with estimators

    python3 hp/ladder_record.py STATE OUT.jsonl LOG.jsonl [LOG.jsonl ...] --start START.npz [--kstar K --gain G
        [--F_kstar F]] [--M_ref DELTA | --err_M E --M_basis TEXT] [--err_horizon E --horizon_basis TEXT]
        [--xcheck "name|value|basis;..."] [--note TEXT]

START is the first log's input state (its Delta is the base of the first harmonic's change). The record also carries the
fields export_delta_all.py's direct rows read: system "full", Delta_parts, F_check and its precision,
delta_bound_first_order (= err_resid), state and state_blob.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpmath import mp  # noqa: E402

from form_s import git_blob  # noqa: E402
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
    o = dict(gain="", M_ref="", note="", start="", err_M="", M_basis="", err_horizon="", horizon_basis="", kstar="",
             F_kstar="", xcheck="")
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
    pending, released_by, F_last, F_rel = None, {}, None, {}
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
            F_rel[k] = r.get("F")
    ks = sorted(set(per) | set(released_by))
    kstar = int(o["kstar"]) if o["kstar"] else None
    # the geometric remainder past the last released harmonic, from the last two with changes
    kc_ = [k for k in ks if k in per]
    rem, rem_basis = None, "fewer than two released harmonics with changes: no remainder estimate"
    if len(kc_) >= 2:
        c1, c2 = abs(per[kc_[-2]]), abs(per[kc_[-1]])
        if c1 > 0 and c2 < c1:
            r = c2 / c1
            rem = float(c2 * r / (1 - r))
            rem_basis = (f"the geometric remainder from h{kc_[-2]} ({mp.nstr(per[kc_[-2]], 4)}) and h{kc_[-1]} "
                         f"({mp.nstr(per[kc_[-1]], 4)}), r {mp.nstr(r, 3)}: {rem:.2e}")
        else:
            rem_basis = f"the changes do not fall from h{kc_[-2]} to h{kc_[-1]}: no remainder estimate"
    beyond = [k for k in ks if kstar is not None and k > kstar]
    s_beyond = mp.fsum(per.get(k, 0) for k in beyond)
    err_tail = abs(pending or 0.0) + (rem or 0.0)
    basis_tail = (f"an estimator: the last pending dDelta at release ({pending}) and {rem_basis}. The released changes "
                  f"beyond the bound's kfree {kstar} ({', '.join('h' + str(k) for k in beyond) or 'none'}, "
                  f"{mp.nstr(s_beyond, 6)} in all) are measured values with no first-order bound of their own; the "
                  f"err_xcheck fields test them")
    rec = dict(D_str=meta["D_str"], form="S", system="full", method="ladder", M=int(meta["M"]), kfreeze=meta.get("kfreeze"),
               horizon=f"mob a={meta.get('hmob')}" if meta.get("hmob") else "u", ks=int(meta["ks"]), kc=int(meta["kc"]),
               kt=int(meta.get("kt", 36)), taylor_steps=sum(len(g) - 1 for g in meta["segs"]), Delta=delta_digits(z8, 40),
               Delta_parts=[float(z8[q][-1]) for q in range(8)], F_final=F_last, F_check=F_last, F_check_precision="od",
               state=os.path.basename(state), state_blob=git_blob(state),
               per_harmonic={f"h{k}": dict(change=float(per.get(k, 0)), released_by=released_by.get(k, "")) for k in ks},
               kstar=kstar, changes_beyond_kstar=float(s_beyond), err_tail=err_tail, err_tail_basis=basis_tail,
               note=o["note"])
    if o["gain"] and kstar is not None:
        Fk = float(o["F_kstar"]) if o["F_kstar"] else F_rel.get(kstar)
        rec.update(err_resid=float(o["gain"]) * float(Fk) if Fk is not None else None,
                   err_resid_basis=f"a first-order bound at kfree {kstar}: gain {o['gain']} (gainq.py at kfreeze {kstar}) "
                                   f"x that system's |F| {Fk} at h{kstar}'s release")
    else:
        rec.update(err_resid=None, err_resid_basis="no gain or kstar given")
    rec["delta_bound_first_order"] = rec["err_resid"]
    if o["M_ref"]:
        dm = abs(mp.mpf(rec["Delta"]) - mp.mpf(o["M_ref"]))
        rec.update(err_M=float(dm), err_M_basis=f"|Delta - {o['M_ref']}| (the other M)")
    elif o["err_M"]:
        rec.update(err_M=float(o["err_M"]), err_M_basis=o["M_basis"])
    else:
        rec.update(err_M=None, err_M_basis="no other M given")
    for k, (v, b) in MEASURED.items():
        rec[k] = float(o[k[4:]]) if o.get(k[4:]) else v
        rec[k + "_basis"] = b
    if o["err_horizon"]:
        rec.update(err_horizon=float(o["err_horizon"]), err_horizon_basis=o["horizon_basis"])
    for x in [v for v in o["xcheck"].split(";") if v.strip()]:
        name, val, basis = x.split("|", 2)
        rec[f"err_xcheck_{name.strip()}"] = float(val)
        rec[f"err_xcheck_{name.strip()}_basis"] = basis.strip()
    names = ("err_resid", "err_tail", "err_M") + tuple(MEASURED)
    have = [k for k in names if rec.get(k) is not None]
    rec["err_sum"] = float(sum(rec[k] for k in have))
    rec["err_sum_basis"] = ("the sum of " + ", ".join(have) + ": it mixes a first-order bound (err_resid) with estimators "
                            "(err_tail, err_M, err_horizon) and measured discretization changes; the err_xcheck fields "
                            "are cross-checks, not added")
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("D_str", "Delta", "kstar", "err_resid", "err_tail", "err_M", "err_sum")}),
          flush=True)


if __name__ == "__main__":
    main()
