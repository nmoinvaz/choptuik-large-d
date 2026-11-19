#!/usr/bin/env python3
"""The checks of amp_flint_par.py --trunc (hier's truncated model with the phys fit, #1016), on finished result files.
Every criterion was fixed before the runs (#1016's start comment and checks comment). Files are amp_flint_par.py
outputs in one folder DIR, named as run_trunc_checks.sh writes them:

  planted   P_K7_J500_{full,t7,t9}.json (procs 1) and P_K9_J500_p2_{full,t9}.json (procs 2): --trunc M with M >= K gives
            sigma_str, f and fit_info identical to the untruncated run's. Control: P_K7_J500_t5.json (M < K) must differ.
  exact     F_K11_J{1000,1500}.json (full), T3_K9_J*.json, T5_K11_J*.json: sigma_n of the truncated model equals the full
            series' for n = 1..M (lemma O, proof/B_tail_M7.md) within 3 x max(e_fit, s_J) at both J, where e_fit is
            the larger phys-fit plateau error of S_1^(2n+1) / |S_1^(1)| in the two runs compared (fit_info) and s_J the
            full run's two-J spread |sigma_n(1500) - sigma_n(1000)|.
  control   C3_K9_J1000.json, C5_K11_J1000.json (--trunc-even M - 1, amp_trunc.py's truncation): sigma_M moves from
            the --trunc M value by more than 10 x max(e_fit, s_J).
  J         M = 3: Delta sigma_4 = sigma_4(full) - sigma_4^[3] at J 1000 and 1500 with the phys fit; it passes if its
            change is below a tenth of the legacy fit's over the same J (L_full_K9_J*.json from amp_flint_par.py without
            --fit, L3_K9_J*.json from solver/proof/amp_trunc_hiermodel.py 3 9 J).

  Jlog      added after the J check failed (#1016), for its diagnosed cause: the truncated model's sequence at its
            first truncated order k = 2M + 3 is c ln j + S + O(1/j). With amp_flint_par.py --fit-logs 1 (T3log_K9_J*,
            T5log_K13_J* against F_K13_J*): the constant's change from J 1000 to 1500 is below a tenth of the legacy
            fit's (the J check's bar), and c changes by less than 1e-6 of itself. Diagnostic, not a check: c / S_1^(1)
            times ln(1500/1000) against the no-log fit's drift of sigma_(M+1).

    python3 amp_trunc_checks.py DIR [planted|exact|control|J|Jlog|all]      exit 0 iff every check run holds
"""
import json
import os
import sys

import mpmath as mp

mp.mp.dps = 30


def load(d, name):
    fn = os.path.join(d, name)
    return json.load(open(fn)) if os.path.exists(fn) else None


def sig(rec, n):
    re, im = rec["sigma_str"][n]
    return mp.mpc(mp.mpf(re), mp.mpf(im))


def planted(d):
    ok, seen = True, 0
    for base, others in (("P_K7_J500_full.json", ("P_K7_J500_t7.json", "P_K7_J500_t9.json")),
                         ("P_K9_J500_p2_full.json", ("P_K9_J500_p2_t9.json",))):
        a = load(d, base)
        for o in others:
            b = load(d, o)
            if a is None or b is None:
                print(f"planted: {base} or {o} missing")
                ok = False
                continue
            seen += 1
            same = all(a[k] == b[k] for k in ("sigma_str", "f", "fit_info"))
            ok &= same
            print(f"planted: {o} against {base}: sigma_str, f, fit_info {'identical' if same else 'DIFFER'}  "
                  f"({'as required' if same else 'NOT AS REQUIRED'})")
    a, c = load(d, "P_K7_J500_full.json"), load(d, "P_K7_J500_t5.json")
    if a and c:
        differ = a["sigma_str"] != c["sigma_str"]
        ok &= differ
        k = next((i for i, (x, y) in enumerate(zip(a["sigma_str"], c["sigma_str"])) if x != y), None)
        print(f"planted control: --trunc 5 at K 7 against full: sigma_str {'differs, first at sigma_' + str(k) if differ else 'IDENTICAL'}"
              f"  ({'as required' if differ else 'NOT AS REQUIRED'})")
    else:
        ok = False
        print("planted control: files missing")
    return ok


def yard(full_a, full_b, tr, n):
    """3-component yardstick for sigma_n: (fit error of the full run, of the truncated run, the full run's two-J spread)"""
    e = max(full_a["fit_info"][f"{2 * n + 1}_1"][1], tr["fit_info"][f"{2 * n + 1}_1"][1]) / s1abs(full_a)
    s = float(abs(sig(full_b, n) - sig(full_a, n)))
    return e, s


def s1abs(rec):
    """|S_1^(1)| = |d_0 / (Gamma(a_1) Gamma(b_1))| with d_0 = -i: the order-1 multiplier, the exact linear solution"""
    w = mp.mpf(rec["wf"]) * 2 * mp.pi / mp.mpf("1.64389375")
    return float(abs(mp.rgamma((1 + 1j * w) / 2) * mp.rgamma((2 + 1j * w) / 2)))


def exact(d):
    ok = True
    F = {J: load(d, f"F_K11_J{J}.json") for J in (1000, 1500)}
    rows = []
    for M, K in ((3, 9), (5, 11)):
        for J in (1000, 1500):
            T = load(d, f"T{M}_K{K}_J{J}.json")
            if T is None or F[1000] is None or F[1500] is None:
                print(f"exact: M {M} J {J}: files missing")
                ok = False
                continue
            for n in range(1, M + 1):
                e, s = yard(F[J], F[1500 if J == 1000 else 1000], T, n)
                diff = float(abs(sig(T, n) - sig(F[J], n)))
                bar = 3 * max(e, s)
                good = diff <= bar
                ok &= good
                rows.append((M, J, n, diff, e, s, good))
    print("| M | J | n | \\|sigma_n^[M] - sigma_n\\| | fit error | two-J spread | 3 x max | holds |")
    print("|---|---|---|---|---|---|---|---|")
    for M, J, n, diff, e, s, good in rows:
        print(f"| {M} | {J} | {n} | {diff:.2e} | {e:.2e} | {s:.2e} | {3 * max(e, s):.2e} | {'yes' if good else 'NO'} |")
    return ok


def control(d):
    ok = True
    F = {J: load(d, f"F_K11_J{J}.json") for J in (1000, 1500)}
    for M, K in ((3, 9), (5, 11)):
        T, C = load(d, f"T{M}_K{K}_J1000.json"), load(d, f"C{M}_K{K}_J1000.json")
        if T is None or C is None or F[1000] is None or F[1500] is None:
            print(f"control: M {M}: files missing")
            ok = False
            continue
        e, s = yard(F[1000], F[1500], T, M)
        mv = float(abs(sig(C, M) - sig(T, M)))
        bar = 10 * max(e, s)
        good = mv > bar
        ok &= good
        first = next((n for n in range(1, M + 1) if C["sigma_str"][n] != T["sigma_str"][n]), None)
        print(f"control M {M}: |sigma_{M}(--trunc-even {M - 1}) - sigma_{M}(--trunc {M})| = {mv:.3e} against 10 x max(fit "
              f"{e:.1e}, spread {s:.1e}) = {bar:.2e}; first differing sigma_n: {first}  "
              f"({'as required' if good else 'NOT AS REQUIRED'})")
    return ok


def Jcheck(d):
    ok = True
    vals = {}
    for kind, fname, tname in (("phys", "F_K11_J{J}.json", "T3_K9_J{J}.json"), ("legacy", "L_full_K9_J{J}.json", "L3_K9_J{J}.json")):
        for J in (1000, 1500):
            a, b = load(d, fname.format(J=J)), load(d, tname.format(J=J))
            if a is None or b is None:
                print(f"J: {kind} J {J}: files missing")
                ok = False
                continue
            vals[(kind, J)] = sig(a, 4) - sig(b, 4)
    if len(vals) < 4:
        return False
    dp = float(abs(vals[("phys", 1500)] - vals[("phys", 1000)]))
    dl = float(abs(vals[("legacy", 1500)] - vals[("legacy", 1000)]))
    good = dp < 0.1 * dl
    ok &= good
    for kind in ("phys", "legacy"):
        print(f"J, {kind}: Delta sigma_4 = sigma_4 - sigma_4^[3] at J 1000 {mp.nstr(vals[(kind, 1000)], 12)}, at J 1500 "
              f"{mp.nstr(vals[(kind, 1500)], 12)}")
    print(f"J: change from J 1000 to 1500, phys {dp:.3e} against legacy {dl:.3e} (bar: phys < 0.1 x legacy)  "
          f"({'as required' if good else 'NOT AS REQUIRED'})")
    return ok


def logc(rec, key):
    v = rec["fit_logc"][key]
    re, im = v[0] if isinstance(v[0], list) else v               # newer files: a list over the log powers
    return mp.mpc(mp.mpf(re), mp.mpf(im))


def Jlog(d):
    ok = True
    legacy = None
    a, b = load(d, "L_full_K9_J1000.json"), load(d, "L3_K9_J1000.json")
    a2, b2 = load(d, "L_full_K9_J1500.json"), load(d, "L3_K9_J1500.json")
    if None not in (a, b, a2, b2):
        legacy = float(abs((sig(a2, 4) - sig(b2, 4)) - (sig(a, 4) - sig(b, 4))))
    for M, K, full, nolog in ((3, 9, "F_K11_J{J}.json", "T3_K9_J{J}.json"), (5, 13, "F_K13_J{J}.json", "T5_K13_J{J}.json")):
        n = M + 1
        F = {J: load(d, full.format(J=J)) for J in (1000, 1500)}
        T = {J: load(d, f"T{M}log_K{K}_J{J}.json") for J in (1000, 1500)}
        N = {J: load(d, nolog.format(J=J)) for J in (1000, 1500)}
        if None in list(F.values()) + list(T.values()):
            print(f"Jlog: M {M}: files missing")
            ok = False
            continue
        D = {J: sig(F[J], n) - sig(T[J], n) for J in (1000, 1500)}
        dc = float(abs(D[1500] - D[1000]))
        key = f"{2 * n + 1}_1"
        c = {J: logc(T[J], key) for J in (1000, 1500)}
        crel = float(abs(c[1500] - c[1000]) / abs(c[1000]))
        bar = 0.1 * legacy if legacy is not None else float("nan")
        good = dc < bar and crel < 1e-6
        ok &= good
        w = mp.mpf(1) * 2 * mp.pi / mp.mpf("1.64389375")
        s1 = -1j * mp.rgamma((1 + 1j * w) / 2) * mp.rgamma((2 + 1j * w) / 2)       # S_1^(1) = d_0 / (Gamma Gamma), d_0 = -i
        cs = c[1000] / s1
        print(f"Jlog M {M}: Delta sigma_{n} (log-separated) at J 1000 {mp.nstr(D[1000], 14)}, at J 1500 {mp.nstr(D[1500], 14)}; "
              f"change {dc:.2e} against 0.1 x legacy {bar:.2e}; log coefficient c = {mp.nstr(c[1000], 14)}, relative J change "
              f"{crel:.1e}  ({'as required' if good else 'NOT AS REQUIRED'})")
        if None not in N.values():
            drift = sig(N[1500], n) - sig(N[1000], n)
            pred = cs * mp.log(mp.mpf(1500) / 1000)
            print(f"   diagnostic: c / S_1^(1) = {mp.nstr(cs, 6)}; times ln 1.5: {mp.nstr(pred, 6)}; the no-log fit's drift of "
                  f"sigma_{n}^[{M}]: {mp.nstr(drift, 6)}; ratio {mp.nstr(abs(drift / pred), 4)}")
    return ok


if __name__ == "__main__":
    d = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    fns = {"planted": planted, "exact": exact, "control": control, "J": Jcheck, "Jlog": Jlog}
    ok = True
    for name, fn in fns.items():
        if which in (name, "all"):
            ok &= fn(d)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
