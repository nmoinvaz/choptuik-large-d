#!/usr/bin/env python3
"""Review of #1079, check 3: every number quoted in the issue's result comments against the raw logs and JSON files.

    python3 review_1079/check_quoted.py RAW [--mutate=K] [--selftest]          from solver/proof
        RAW  the raw-data clone (reads I1079-macm3/ and I1079-dh610/)

Each entry is (what the comment says, the file, how it is read, the expected value typed from the comment). A text
entry passes when the expected string occurs in the file (on a line matching the given key, if one is given); a JSON
entry when the value read, formatted as stated, equals the expected string. Exit 0 iff every entry passes.
--mutate=K changes the expected value of entry K (its last digit): that entry must fail. --selftest runs every
entry's mutation and passes iff each one fails and the clean run passes.
"""
import json
import os
import re
import sys

L1 = "I1079-macm3/levers"
H2 = "I1079-macm3/h2"
DH = "I1079-dh610/out"
X1, X2, X2b, X1b = "M3x1E2_h1e-12_w1e-13", "M3x2E2_h1e-12_w1e-12", "M3x2E2_h1e-12_w1e-13", "M3x1E2_h1e-12_w1e-12"


def T(desc, path, expect, key=None):
    return ("text", desc, path, key, expect)


def J(desc, path, query, fmt, expect):
    return ("json", desc, path, (query, fmt), expect)


def disc_count(raw, d):
    n, bad = 0, 0
    for j in range(12):
        with open(os.path.join(raw, d, f"row_{j:02d}.json")) as fh:
            r = json.load(fh)
        n += len(r["discs"])
        bad += sum(1 for x in r["discs"] if not x["ok"]) + (0 if r["ok"] and r["mutate"] is None else 1)
    return f"{n} discs, {bad} not ok"


ENTRIES = [
    # the Krawczyk step (comments 7107510600, 7107526195, 7109089…: the coordinator's tables)
    T("x1 (1e-12, 1e-13) margins", f"{L1}/logs/A_M_krawczyk_pt_{X1}.log",
      "K(X) in int X: True; relative margins +0.702, +0.758, +0.997, +0.998"),
    T("x2 (1e-12, 1e-12) margins", f"{L1}/logs/A_M_krawczyk_pt_{X2}.log",
      "K(X) in int X: True; relative margins +0.943, +0.945, +0.988, +0.995"),
    T("x2 (1e-12, 1e-13) margins", f"{L1}/logs/A_M_krawczyk_pt_{X2b}.log",
      "K(X) in int X: True; relative margins +0.621, +0.341, +0.830, +0.951"),
    T("x1 (1e-12, 1e-12) margins", f"{L1}/logs/A_M_krawczyk_pt_{X1b}.log",
      "K(X) in int X: True; relative margins +0.272, +0.484, +1.000, +1.000"),
    T("x1 controls", f"{L1}/logs/A_M_controls_pt_{X1}.log", "CONTROLS ALL AS EXPECTED"),
    T("x2 controls", f"{L1}/logs/A_M_controls_pt_{X2}.log", "CONTROLS ALL AS EXPECTED"),
    T("x2 (1e-12, 1e-13) controls", f"{L1}/logs/A_M_controls_pt_{X2b}.log", "CONTROLS ALL AS EXPECTED"),
    T("x1 (1e-12, 1e-12) control C6 unexpected", f"{L1}/logs/A_M_controls_pt_{X1b}.log",
      "got fail (relative margins -2.28e-01, +1.84e-01, +7.50e-01, +7.50e-01)  UNEXPECTED", key="C6 "),
    T("x2 (1e-12, 1e-13) C5", f"{L1}/logs/A_M_controls_pt_{X2b}.log", "(relative margins -1.38e+00,", key="C5 "),
    T("x2 (1e-12, 1e-12) C5", f"{L1}/logs/A_M_controls_pt_{X2}.log", "(relative margins -1.06e+00,", key="C5 "),
    T("x2 (1e-12, 1e-13) C7", f"{L1}/logs/A_M_controls_pt_{X2b}.log", "(relative margins -5.61e+02, -9.13e+02,", key="C7 "),
    T("x2 (1e-12, 1e-13) C8", f"{L1}/logs/A_M_controls_pt_{X2b}.log", "+3.41e-01, -1.17e+00,", key="C8 "),
    T("x2 (1e-12, 1e-12) C8", f"{L1}/logs/A_M_controls_pt_{X2}.log", "+9.45e-01, -1.01e+00,", key="C8 "),
    T("x1 recheck", f"{L1}/logs/A_M_recheck_pt_{X1}.log", "K(X) in int X: True; relative margins +0.702, +0.758, +0.997, +0.998"),
    T("x2 recheck", f"{L1}/logs/A_M_recheck_pt_{X2}.log", "K(X) in int X: True; relative margins +0.943, +0.945, +0.988, +0.995"),
    T("x1 recheck mutation", f"{L1}/logs/A_M_recheck_pt_{X1}_mutate.log", "[mutated: jump sign]"),
    T("x1 recheck mutation fails", f"{L1}/logs/A_M_recheck_pt_{X1}_mutate.log", "K(X) in int X: False"),
    T("x2 recheck mutation fails", f"{L1}/logs/A_M_recheck_pt_{X2}_mutate.log", "K(X) in int X: False"),
    T("x1 enclosure, A", f"{L1}/logs/A_M_krawczyk_pt_{X1}.log", "[3.44621824636642, 3.44621824636703]  (radius 2.98e-13)"),
    T("x1 enclosure, w", f"{L1}/logs/A_M_krawczyk_pt_{X1}.log", "[0.347606965217164, 0.347606965217650]  (radius 2.42e-13)"),
    T("x2 enclosure, A", f"{L1}/logs/A_M_krawczyk_pt_{X2}.log", "[3.52647305669565, 3.52647305669578]  (radius 5.66e-14)"),
    T("x2 enclosure, w", f"{L1}/logs/A_M_krawczyk_pt_{X2}.log", "[0.211497981391159, 0.211497981391271]  (radius 5.55e-14)"),
    # radii of g(p^)
    J("x1 g radius, orders 28/16", f"{L1}/data/A_M_point_M3x1E2.json", "g_rad", "{:.3e}", "1.571e-21"),
    J("x2 g radius, orders 28/16", f"{L1}/data/A_M_point_M3x2E2.json", "g_rad", "{:.3e}", "7.422e-23"),
    J("x2 g radius, E1", f"{L1}/data/A_M_point_M3x2E1.json", "g_rad", "{:.3e}", "6.459e-20"),
    J("x2 g radius, E3", f"{L1}/data/A_M_point_M3x2E3.json", "g_rad", "{:.3e}", "1.967e-28"),
    T("x1 g radius, default orders", "I1079-macm3/steps.txt", "g_rad 5.337e-14", key="round3 point x1"),
    T("x2 g radius, default orders", "I1079-macm3/steps.txt", "g_rad 1.188e-14", key="round3 point x2"),
    # H0
    J("x1 eta", f"{L1}/data/A_M_box_{X1}.json", "start.eta", "{:.2e}", "1.51e-72"),
    J("x2 eta", f"{L1}/data/A_M_box_{X2}.json", "start.eta", "{:.2e}", "6.34e-72"),
    J("x1 start rigorous", f"{L1}/data/A_M_box_{X1}.json", "start.rigorous", "{}", "True"),
    J("x2 start rigorous", f"{L1}/data/A_M_box_{X2}.json", "start.rigorous", "{}", "True"),
    J("x1 point start rigorous", f"{L1}/data/A_M_point_M3x1E2.json", "start.rigorous", "{}", "True"),
    J("x2 point start rigorous", f"{L1}/data/A_M_point_M3x2E2.json", "start.rigorous", "{}", "True"),
    J("x1 box xi1", f"{L1}/data/A_M_box_{X1}.json", "path.3", "{}", "0.3"),
    J("x1 point xi1", f"{L1}/data/A_M_point_M3x1E2.json", "path.3", "{}", "0.3"),
    J("x2 box xi1", f"{L1}/data/A_M_box_{X2}.json", "path.3", "{}", "0.3"),
    J("x2 point xi1", f"{L1}/data/A_M_point_M3x2E2.json", "path.3", "{}", "0.3"),
    # H1
    T("x1 lemma S", f"{L1}/h1/I1079_sector_{X1}.log", "contraction constant (weighted max norm) <= 0.6961"),
    T("x1 old sector lemma", f"{L1}/h1/I1079_sector_{X1}.log", "contraction 0.6839, F + Lip rho < rho: True, ok True"),
    T("x1 flat estimate", f"{L1}/h1/I1079_sector_{X1}.log", "Lambda = 0.2426 < 1: True", key="plain"),
    T("x1 H1 passes", f"{L1}/h1/I1079_sector_{X1}.log", "ALL CHECKS PASS (M = 3): True"),
    T("x1 series mutation", f"{L1}/h1/I1079_sector_{X1}_mutate_series.log", "lemma S holds: False (must be False)"),
    T("x2 lemma S", f"{L1}/h1/I1079_sector_{X2}.log", "contraction constant (weighted max norm) <= 0.6782"),
    T("x2 old sector lemma", f"{L1}/h1/I1079_sector_{X2}.log", "contraction 0.7008, F + Lip rho < rho: True, ok True"),
    T("x2 flat estimate", f"{L1}/h1/I1079_sector_{X2}.log", "Lambda = 0.2368 < 1: True", key="plain"),
    T("x2 H1 passes", f"{L1}/h1/I1079_sector_{X2}.log", "ALL CHECKS PASS (M = 3): True"),
    T("x2 series mutation", f"{L1}/h1/I1079_sector_{X2}_mutate_series.log", "lemma S holds: False (must be False)"),
    # H3
    J("x1 rho_6a", f"{L1}/lemmas_M_runs/lemma6a_{X1}.json", "rho6a", "{}", "[0.01365237958 +/- 4.00e-12]"),
    J("x2 rho_6a", f"{L1}/lemmas_M_runs/lemma6a_{X2}.json", "rho6a", "{}", "[0.01258978538 +/- 1.51e-12]"),
    J("x1 6a ok", f"{L1}/lemmas_M_runs/lemma6a_{X1}.json", "ok", "{}", "True"),
    J("x2 6a ok", f"{L1}/lemmas_M_runs/lemma6a_{X2}.json", "ok", "{}", "True"),
    J("x1 6a xi1", f"{L1}/lemmas_M_runs/lemma6a_{X1}.json", "xi1", "{}", "0.3"),
    T("x1 6a state box", f"{L1}/logs/I1079_lemma6a_{X1}.log", "radius 1.80e-12", key="start box"),
    T("x2 6a state box", f"{L1}/logs/I1079_lemma6a_{X2}.log", "radius 1.21e-12", key="start box"),
    T("x1 6a partial", f"{L1}/logs/I1079_lemma6a_{X1}_partial.log", "G_slow is 9 x 11: not square"),
    T("x2 6a partial", f"{L1}/logs/I1079_lemma6a_{X2}_partial.log", "G_slow is 9 x 11: not square"),
    T("x1 6a swap", f"{L1}/logs/I1079_lemma6a_{X1}_swap.log", "backward integration failed"),
    T("x2 6a swap", f"{L1}/logs/I1079_lemma6a_{X2}_swap.log", "backward integration failed"),
    # H2
    ("call", "x1 covering: discs", f"{H2}/lemma6b1_runs_M3_exc1", disc_count, "183 discs, 0 not ok"),
    ("call", "x2 covering: discs", f"{H2}/lemma6b1_runs_M3_exc2", disc_count, "183 discs, 0 not ok"),
    J("x1 covering half-width", f"{H2}/lemma6b1_runs_M3_exc1/ray.json", "h", "{}", "1e-11"),
    J("x2 covering half-width", f"{H2}/lemma6b1_runs_M3_exc2/ray.json", "h", "{}", "1e-11"),
    T("x1 check", f"{H2}/logs/I1079_h2_check_exc1.log", "LEMMA 6b_1 CHECK: PASS"),
    T("x2 check", f"{H2}/logs/I1079_h2_check_exc2.log", "LEMMA 6b_1 CHECK: PASS"),
    T("x1 gluing", f"{H2}/logs/I1079_h2_check_exc1.log", "tiles glued: 183 (meeting sets: 579 polar rectangles, 461 links)"),
    T("x2 gluing", f"{H2}/logs/I1079_h2_check_exc2.log", "tiles glued: 183 (meeting sets: 579 polar rectangles, 461 links)"),
    T("x1 c1check", f"{H2}/logs/I1079_h2_c1check_exc1.log", "C1 PLANTED TRUTH: every point inside its set: True"),
    T("x2 c1check", f"{H2}/logs/I1079_h2_c1check_exc2.log", "C1 PLANTED TRUTH: every point inside its set: True"),
    T("x1 sign mutation", f"{H2}/logs/I1079_h2_mutate_exc1.log", "0 of 12 discs of row 11 certified (must be 0)"),
    T("x2 sign mutation", f"{H2}/logs/I1079_h2_mutate_exc2.log", "0 of 12 discs of row 11 certified (must be 0)"),
    T("x1 inclusion", f"{H2}/logs/I1079_incl_exc1.log", "inside X_b: True (slack +8.359e-12 at B3)"),
    T("x2 inclusion", f"{H2}/logs/I1079_incl_exc2.log", "inside X_b: True (slack +8.278e-12 at B3)"),
    T("x1 inclusion verdict", f"{H2}/logs/I1079_incl_exc1.log", "THEOREM A SETS (M = 3, mutation None): HOLD"),
    T("x2 inclusion verdict", f"{H2}/logs/I1079_incl_exc2.log", "THEOREM A SETS (M = 3, mutation None): HOLD"),
    T("x1 inclusion selftest", f"{H2}/logs/I1079_incl_exc1_selftest.log", "SELFTEST (M = 3): PASS"),
    T("x2 inclusion selftest", f"{H2}/logs/I1079_incl_exc2_selftest.log", "SELFTEST (M = 3): PASS"),
    # Linux
    T("dh610 x1 margins", f"{DH}/logs/A_M_krawczyk_pt_{X1}.log",
      "K(X) in int X: True; relative margins +0.468, +0.619, +0.997, +0.998"),
    T("dh610 x2 margins", f"{DH}/logs/A_M_krawczyk_pt_{X2}.log",
      "K(X) in int X: True; relative margins +0.711, +0.528, +0.981, +0.992"),
    T("dh610 x1 C6", f"{DH}/logs/A_M_controls_pt_{X1}.log", "got fail (relative margins -3.17e-02, +3.19e-01,", key="C6 "),
    T("dh610 x2 controls", f"{DH}/logs/A_M_controls_pt_{X2}.log", "CONTROLS ALL AS EXPECTED"),
    T("dh610 cross-load x1", f"{DH}/logs/XL_krawczyk_pt_{X1}.log",
      "K(X) in int X: True; relative margins +0.702, +0.758, +0.997, +0.998"),
    T("dh610 cross-load x2", f"{DH}/logs/XL_krawczyk_pt_{X2}.log",
      "K(X) in int X: True; relative margins +0.943, +0.945, +0.988, +0.995"),
    J("dh610 x1 rho_6a", f"{DH}/lemmas_M_runs/lemma6a_{X1}.json", "rho6a", "{:.13}", "[0.0136523760"),
    J("dh610 x2 rho_6a", f"{DH}/lemmas_M_runs/lemma6a_{X2}.json", "rho6a", "{:.13}", "[0.0125897790"),
    ("call", "dh610 x1 covering: discs", f"{DH}/lemma6b1_runs_M3_exc1_linux", disc_count, "183 discs, 0 not ok"),
    ("call", "dh610 x2 covering: discs", f"{DH}/lemma6b1_runs_M3_exc2_linux", disc_count, "183 discs, 0 not ok"),
    T("dh610 x1 sign mutation", f"{DH}/logs/I1079_h2_mutate_exc1.log", "0 of 12 discs of row 11 certified (must be 0)"),
    T("dh610 x2 sign mutation", f"{DH}/logs/I1079_h2_mutate_exc2.log", "0 of 12 discs of row 11 certified (must be 0)"),
    T("dh610 x1 lemma S", f"{DH}/logs/I1079_sector_{X1}.log", "contraction constant (weighted max norm) <= 0.6961"),
    T("dh610 x2 lemma S", f"{DH}/logs/I1079_sector_{X2}.log", "contraction constant (weighted max norm) <= 0.6782"),
    T("dh610 x1 check", f"{DH}/logs/I1079_h2_check_exc1.log", "LEMMA 6b_1 CHECK: PASS"),
    T("dh610 x2 check", f"{DH}/logs/I1079_h2_check_exc2.log", "LEMMA 6b_1 CHECK: PASS"),
    T("dh610 x1 inclusion", f"{DH}/logs/I1079_incl_exc1.log", "inside X_b: True (slack +8.359e-12 at B3)"),
    T("dh610 x2 inclusion", f"{DH}/logs/I1079_incl_exc2.log", "inside X_b: True (slack +8.278e-12 at B3)"),
]


def read(raw, e):
    kind, desc, path, how, expect = e
    full = os.path.join(raw, path)
    if kind == "call":
        return how(raw, path)
    if kind == "json":
        query, fmt = how
        with open(full) as fh:
            v = json.load(fh)
        for k in query.split("."):
            v = v[int(k)] if isinstance(v, list) else v[k]
        return fmt.format(v)
    with open(full) as fh:
        lines = fh.read().splitlines()
    if how is not None:
        lines = [ln for ln in lines if how in ln]
    return "\n".join(lines)


def run(raw, mutate=None, quiet=False):
    bad = []
    for i, e in enumerate(ENTRIES):
        kind, desc, path, how, expect = e
        if mutate == i:                       # the last digit of the expected value changed
            m = re.search(r"\d(?=\D*$)", expect)
            if m:
                expect = expect[:m.start()] + str((int(m.group()) + 1) % 10) + expect[m.end():]
            else:
                expect = expect.replace("True", "False").replace("PASS", "FAIL").replace("HOLD", "FAIL") \
                    if any(w in expect for w in ("True", "PASS", "HOLD")) else expect + " x"
        got = read(raw, e)
        ok = (expect in got) if kind == "text" else (got == expect)
        if not ok:
            bad.append(i)
        if not quiet:
            shown = got if kind != "text" else ("found" if ok else "NOT FOUND")
            print(f"{i:3d} {'ok  ' if ok else 'FAIL'} {desc}: {expect!r} [{path}] -> {shown}")
    return bad


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    raw = args[0]
    if "selftest" in opts:
        clean = run(raw, quiet=True)
        missed = [i for i in range(len(ENTRIES)) if run(raw, mutate=i, quiet=True) != sorted(set(clean + [i]))]
        print(f"clean run: {len(ENTRIES) - len(clean)} of {len(ENTRIES)} entries pass; mutations that did not fire: {missed}")
        good = not clean and not missed
        print("CHECK QUOTED SELFTEST: " + ("PASS" if good else "FAIL"))
        return 0 if good else 1
    bad = run(raw, mutate=int(opts["mutate"]) if "mutate" in opts else None)
    print(f"CHECK QUOTED: {len(ENTRIES) - len(bad)} of {len(ENTRIES)} entries agree" + (f"; FAIL at {bad}" if bad else ": PASS"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
