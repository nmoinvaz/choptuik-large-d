"""One ladder point's record (#1098): Delta of a full fastmul6 solve with each error term as its own field, from the
release logs of its stages (release_ls.jsonl, in the order run) and its final state.

Fields: Delta (40 digits) and Delta_parts; per_harmonic (each released harmonic's change, summed over its accepted
outer steps, and how it was released); changes_beyond_kstar (the released changes past k*, a measured value); and the
error terms, each a separate field with a basis that names its kind (bound, estimator, empirical or measured):
  err_resid    bound: first order at kfree k* (--kstar): gain (gainq.py at kfreeze k*, --gain) x that system's |F| at
               harmonic k*'s release (from the logs, or --F_kstar). Past some kfree the gain is so large (6.4e30 at
               kfree 17, 1.5e47 at kfree 31, D 300, M 32) that gain x |F| bounds nothing, so k* is the last useful one.
               --kstar_json KSTAR.json (kstar_stage.sh's, when it re-converged the k* system): gain x the re-converged
               |F| plus |the re-converge's move of Delta|, the bound at release kept as err_resid_before
  err_tail    estimator: the pending first-order dDelta at release (the steps not taken: each harmonic released after
               its own steps once, each run of harmonics released at once with no step between them once), summed
               in |.|, and the geometric remainder past the last
               released harmonic (from the last two changes c_(k-2), c_k: |c_k| r/(1 - r), r = |c_k/c_(k-2)|; none when
               r >= 1)
  err_xcheck_horizon, err_xcheck_M, err_xcheck_machine, err_xcheck_kernel
               empirical: the disagreement of independent computations (u against Moebius, M 32 against 48 or 48
               against 64, one machine against another, fastmul6 against fastmul7), not a bound; null when that check
               does not exist at the point.
               --xcheck "horizon|value|basis;M|value|basis;machine|value|basis", or --M_ref DELTA for M's
  err_steps, err_ks, err_kc, err_kt, err_arith
               measured: changes of Delta when the discretization is refined, at D 300 and D 1000 (I1098 raw folders)
  err_sum      err_resid + err_tail + the err_xcheck present + the measured terms; its basis lists the missing checks

    python3 hp/ladder_record.py STATE OUT.jsonl LOG.jsonl [LOG.jsonl ...] --start START.npz --kstar K --gain G
        [--F_kstar F | --kstar_json KSTAR.json] [--M_ref DELTA] [--xcheck "kind|value|basis;..."]
        [--info "name|value|basis;..."] [--note TEXT]

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
    err_steps=(0.0, "measured: every Taylor step halved changes Delta by 0.0 at D 300 and to 40 digits at D 1000 "
                    "(I1098-mm5/fix, I1098-dh610/fix)"),
    err_ks=(0.0, "measured: Moebius ks 800 and a 4 at ks 600/800 change Delta by 0.0 at D 300 (I1098-mm5/horizon300); "
                 "ks + 200 by 0 at D 1000"),
    err_kc=(0.0, "measured: kc + 20 changes nothing at 1e-48 at D 300 and D 1000 (I1098-dh610/budget300, budget1000)"),
    err_kt=(0.0, "measured: kt + 8 changes nothing at 1e-48 at D 300 and D 1000"),
    err_arith=(1e-30, "measured: the d/dtheta remainders and lane-precision 1/k removed terms of -2.2e-32 (D 300) and "
                      "-1.2e-24 (D 1000); the remainders' own rounding is 1e-16 of those"),
)
XCHECKS = ("horizon", "M", "machine", "kernel")


def opts(argv):
    o = dict(gain="", M_ref="", note="", start="", kstar="", F_kstar="", xcheck="", info="", kstar_json="")
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
    pending, released_by, F_last, F_rel, pend = None, {}, None, {}, {}
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
            pend[k] = abs(pending or 0.0)
            F_last = r.get("F", F_last)
            F_rel[k] = r.get("F")
    ks = sorted(set(per) | set(released_by))
    kstar = int(o["kstar"]) if o["kstar"] else None
    # the geometric remainder past the last released harmonic, from the last two with changes
    kc_ = [k for k in ks if k in per]
    rem, rem_basis = None, "fewer than two released harmonics with changes, so no remainder estimate"
    if len(kc_) >= 2:
        c1, c2 = abs(per[kc_[-2]]), abs(per[kc_[-1]])
        if c1 > 0 and c2 < c1:
            r = c2 / c1
            rem = float(c2 * r / (1 - r))
            rem_basis = (f"the geometric remainder from h{kc_[-2]} ({mp.nstr(per[kc_[-2]], 4)}) and h{kc_[-1]} "
                         f"({mp.nstr(per[kc_[-1]], 4)}), r {mp.nstr(r, 3)}: {rem:.2e}")
        else:
            rem_basis = f"the changes do not fall from h{kc_[-2]} to h{kc_[-1]}, so no remainder estimate"
    beyond = [k for k in ks if kstar is not None and k > kstar]
    s_beyond = mp.fsum(per.get(k, 0) for k in beyond)
    # the steps not taken: each harmonic released after steps of its own has its own pending dDelta; a run of
    # harmonics released at once, with no step between them, shares one state, and their pending dDeltas are nearly
    # one term (D 1000's h25-h31 at M 64: +1.12e-27 each), the base residual's first order, so that run counts once
    terms, run = [], []
    for k in sorted(pend):
        if k in per:
            if run:
                terms.append(max(run))
                run = []
            terms.append(pend[k])
        else:
            run.append(pend[k])
    if run:
        terms.append(max(run))
    pend_sum = float(sum(terms))
    err_tail = pend_sum + (rem or 0.0)
    basis_tail = (f"estimator: the pending first-order dDelta at release, the steps not taken ({pend_sum:.2e}: each "
                  f"harmonic released after its own steps once, each run of harmonics released at once with no step "
                  f"between them once, at its largest), and {rem_basis}. The released changes "
                  f"beyond the bound's kfree {kstar} ({', '.join('h' + str(k) for k in beyond) or 'none'}, "
                  f"{mp.nstr(s_beyond, 6)} in all, changes_beyond_kstar) are measured values with no first-order bound "
                  f"of their own; the err_xcheck fields test them")
    rec = dict(D_str=meta["D_str"], form="S", system="full", method="ladder", M=int(meta["M"]), kfreeze=meta.get("kfreeze"),
               horizon=f"mob a={meta.get('hmob')}" if meta.get("hmob") else "u", ks=int(meta["ks"]), kc=int(meta["kc"]),
               kt=int(meta.get("kt", 36)), taylor_steps=sum(len(g) - 1 for g in meta["segs"]), Delta=delta_digits(z8, 40),
               Delta_parts=[float(z8[q][-1]) for q in range(8)], F_final=F_last, F_check=F_last, F_check_precision="od",
               state=os.path.basename(state), state_blob=git_blob(state),
               per_harmonic={f"h{k}": dict(change=float(per.get(k, 0)), released_by=released_by.get(k, "")) for k in ks},
               kstar=kstar, changes_beyond_kstar=float(s_beyond), err_tail=err_tail, err_tail_basis=basis_tail,
               note=o["note"])
    kj = None
    if o["kstar_json"]:
        with open(o["kstar_json"]) as fh:
            kj = json.load(fh)
    if kj is not None and kj.get("reconv"):
        # the k* stage (kstar_stage.sh): the bound at the re-converged k* system plus the re-converge's move of Delta
        # from the h<k*> state that the later harmonics started from
        rec.update(err_resid=kj["bound_after"], err_resid_before=kj["bound_before"], F_kstar_before=kj["F_before"],
                   F_kstar_after=kj["F_after"], kstar_shift=kj["shift"],
                   err_resid_basis=f"bound, first order: at kfree {kj['kstar']}, gain {kj['gain']:.4e} (kstar_gain.py "
                                   f"at kfreeze {kj['kstar']}) x the re-converged system's |F| {kj['F_after']:.3e}, "
                                   f"plus |the re-converge's move of Delta| {abs(kj['shift']):.3e} from the "
                                   f"h{kj['kstar']} state the later harmonics started from (kstar_stage.sh). At "
                                   f"h{kj['kstar']}'s release it was gain x |F| {kj['F_before']:.3e} = "
                                   f"{kj['bound_before']:.3e} (err_resid_before)")
    elif o["gain"] and kstar is not None:
        Fk = float(o["F_kstar"]) if o["F_kstar"] else F_rel.get(kstar)
        rec.update(err_resid=float(o["gain"]) * float(Fk) if Fk is not None else None,
                   err_resid_basis=f"bound, first order: at kfree {kstar}, gain {o['gain']} (gainq.py at kfreeze "
                                   f"{kstar}) x that system's |F| {Fk} at h{kstar}'s release")
    else:
        rec.update(err_resid=None, err_resid_basis="no gain or kstar given")
    rec["delta_bound_first_order"] = rec["err_resid"]
    xc = {}
    for x in [v for v in o["xcheck"].split(";") if v.strip()]:
        name, val, basis = x.split("|", 2)
        xc[name.strip()] = (float(val), basis.strip())
    if o["M_ref"]:
        dm = abs(mp.mpf(rec["Delta"]) - mp.mpf(o["M_ref"]))
        xc["M"] = (float(dm), f"|Delta - {o['M_ref']}|, the other M's full solve")
    for name in XCHECKS:
        v, b = xc.get(name, (None, "no such check at this point"))
        rec[f"err_xcheck_{name}"] = v
        rec[f"err_xcheck_{name}_basis"] = ("empirical: disagreement of independent computations, not a bound. " + b) \
            if v is not None else b
    for k, (v, b) in MEASURED.items():
        rec[k] = v
        rec[k + "_basis"] = b
    # --info "name|value|basis;...": measured values of the point's history that its final Delta does not carry (not
    # added to err_sum), such as an earlier stage's horizon term
    for x in [v for v in o["info"].split(";") if v.strip()]:
        name, val, basis = x.split("|", 2)
        rec[f"info_{name.strip()}"] = float(val)
        rec[f"info_{name.strip()}_basis"] = basis.strip()
    names = ("err_resid", "err_tail") + tuple(f"err_xcheck_{n}" for n in XCHECKS) + tuple(MEASURED)
    have = [k for k in names if rec.get(k) is not None]
    missing = [k for k in names if rec.get(k) is None]
    rec["err_sum"] = float(sum(rec[k] for k in have))
    rec["err_sum_basis"] = ("the sum of " + ", ".join(have) + ": a first-order bound to kfree k* (err_resid), an "
                            "estimator past it (err_tail), the disagreement of independent computations (err_xcheck_*, "
                            "empirical) and measured discretization changes. Missing: " + (", ".join(missing) or "none"))
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("D_str", "Delta", "kstar", "err_resid", "err_tail", "err_xcheck_horizon",
                                              "err_xcheck_M", "err_xcheck_machine", "err_xcheck_kernel", "err_sum")}), flush=True)


if __name__ == "__main__":
    main()
