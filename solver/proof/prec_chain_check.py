#!/usr/bin/env python3
"""The Theorem A chain's documented commands, traced for decimals read before the precision is set (#888).

python-flint starts at 53 bits, and a decimal turned into a ball before the working precision is set is a ball of
relative radius 1e-16: a "point" becomes a small box. That defect was found three times, by chance or by survey
(#484, #867 F2, #879). prec_trace.py reports such reads for one command. This check runs it over the commands that
Theorem A(M) rests on at M = 3 and M = 5, and fails on any early read that is not on the list of known ones.

The chain (CHAIN below) is the documented commands of proof/A_lemmas_M.md 10, proof/A_M.md 8 and proof/A_M2.md 7
for: step (i) (fixed_point_M.py, and the second implementation fixed_point_M_b.py with its harness), hypothesis (H0)
(start_cover_M2.py), (H1) (lemma6_sector_M.py), (H2) (lemma6b1_cover_M.py, theorem_A_sets_M.py,
start_cover_lemmas_M.py), (H3) (lemma6_jump_M.py), and the point certificates (certify_M.py, certify_M2.py,
certify_M_recheck.py). Placeholders are filled from the stored files: the certificates' centres and boxes. Output
options point into a temporary directory, and the covering's ray and row runs get a scratch --tag (the row reads a
copy of the stored ray file there), so no stored file is written. Each command is traced for --limit seconds in its
own process group, which is then killed.

ALLOWED lists the early reads that are known and recorded, by file and function, each with its reason. A read
anywhere else fails the check.

    python3 prec_chain_check.py [--raw=<raw clone>] [--limit=6] [--only=text] [--json=out.json]
        PASS (exit 0) if every trace started and no command shows an early read outside ALLOWED; FAIL (exit 1)
        otherwise, with the file, line, function and string of each read. Without --raw the commands that need raw
        files are skipped and counted, and the verdict says so.
    python3 prec_chain_check.py --mutate=early
        THE CONTROL. A copy of fixed_point_M_b.py in a temporary directory in which the line of main() that sets the
        precision is moved below the lines that read --x, --box and --cbox (the state before #878). Its plain command
        is traced in place of the real one. The check must FAIL (exit 1) and name the read.
    python3 prec_chain_check.py --selftest [--raw=..]
        the clean check must pass and the control must fail; exit 0 only then

Limits, as prec_trace.py's: a read is seen only if the run reaches it within the time limit and on the path its
arguments take, and a command that starts other processes is traced without them.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}

# early reads that are known and recorded (notes/PRECISION_SURVEY.md): (file, function) -> why it is accepted
ALLOWED = {
    ("fixed_point_M.py", "data_table"): "#484: the first row's box is read at 53 bits; held, since the stored rows "
                                        "were made that way and start_cover_M2.py rebuilds the box both ways",
    ("fixed_point_M.py", "W0"): "survey section 1: from r0_inner_np in table and data_table; the value becomes a "
                                "float printed to four decimals",
    ("start_cover_M2.py", "lemma_runs"): "survey section 4: it rebuilds data_table's 53-bit box on purpose (#484)",
}


def stored(name):
    with open(os.path.join(HERE, name)) as fh:
        return json.load(fh)


def chain(raw, tmp):
    """(label, script, arguments, needs the raw clone) for every command of the chain"""
    p3c = ",".join(stored("lemma6b1_runs_M3_cert/ray.json")["x"])          # the M = 3 covering's centre ($P of 10)
    p5 = ",".join(stored("data/A_M2_box_M5_1e-10.json")["p"])              # the M = 5 certificate's centre
    R = raw or "RAW"
    ray5 = f"{R}/I38-macm3/lemma6b1_runs_M5_cert/ray.json"
    t = tmp
    return [
        # step (i), the first implementation
        ("FP(3) at the zero", "fixed_point_M.py", ["--M=3"], False),
        ("FP(3) complex box, 5pi/16", "fixed_point_M.py", ["--M=3", "--diag", "--r0=0.0741", "--arg=5pi/16",
                                                             "--cbox=1e-3"], False),
        ("FP(3) start data", "fixed_point_M.py", ["--M=3", "--data", "--box=1e-9", "--r0=0.0741", "--arg=5pi/16",
                                                   "--diag"], False),
        ("FP(5) complex box, 5pi/16", "fixed_point_M.py", ["--M=5", "--diag", "--r0=0.0622", "--arg=5pi/16",
                                                             "--cbox=1e-3"], False),
        ("FP table, M = 3", "fixed_point_M.py", ["--table", "--Ms=3", f"--json={t}/table3.json"], False),
        ("FP table, M = 5", "fixed_point_M.py", ["--table", "--Ms=5", f"--json={t}/table5.json"], False),
        ("FP data table, M = 3", "fixed_point_M.py", ["--data-table", "--Ms=3", f"--json={t}/data3.json"], False),
        ("FP data table, M = 5", "fixed_point_M.py", ["--data-table", "--Ms=5", f"--json={t}/data5.json"], False),
        # step (i), the second implementation and its harness
        ("FP_b(5) at the centre", "fixed_point_M_b.py", ["--M=5", f"--x={p5}"], False),
        ("FP_b(5) complex box", "fixed_point_M_b.py", ["--M=5", f"--x={p5}", "--box=1e-18", "--cbox=1e-3"], False),
        ("FP_b planted", "fixed_point_M_b.py", ["--planted", "--M=5"], False),
        ("FP_b kernel", "fixed_point_M_b.py", ["--kernel"], False),
        ("FP_b start set", "fixed_point_M_b.py", ["--M=5", "--start", "--rho0=405", "--h=1e-18", "--cauchy=1e-3",
                                                   f"--compare={ray5}"], True),
        ("FP_b harness m1", "fixed_point_M_b_check.py", ["m1"], False),
        ("FP_b harness lip", "fixed_point_M_b_check.py", ["lip", "--M=5"], False),
        ("FP_b harness cert, M = 3", "fixed_point_M_b_check.py", ["cert", "--box=data/A_M_box_M3_1e-10.json"], False),
        ("FP_b harness cert, M = 5", "fixed_point_M_b_check.py", ["cert", "--box=data/A_M2_box_M5_1e-10.json"], False),
        ("FP_b harness cert2", "fixed_point_M_b_check.py", ["cert2", "--box=data/A_M2_box_M5_1e-10.json"], False),
        # (H0)
        ("H0, M = 5", "start_cover_M2.py", [], False),
        ("H0, M = 3", "start_cover_M2.py", ["--box=data/A_M_box_M3_1e-10.json", "--rerun"], False),
        # (H1)
        ("H1 lemma S(3)", "lemma6_sector_M.py", ["--M=3", "--rp=0.1", "--k=20", "--box=1e-6", "--Z=0.004489",
                                                  f"--json={t}/s3.json"], False),
        ("H1 lemma S(5)", "lemma6_sector_M.py", ["--M=5", "--rp=0.08", "--k=20", "--box=1e-6", "--Z=0.004489",
                                                  f"--json={t}/s5.json"], False),
        # (H2)
        ("H2 covering design, M = 3", "lemma6b1_cover_M.py", ["design", "--M=3"], False),
        ("H2 covering ray, M = 3", "lemma6b1_cover_M.py", ["ray", "--c1", "--M=3", "--h=1.1e-9", "--tag=_prectrace",
                                                            f"--x={p3c}"], False),
        ("H2 covering row, M = 3", "lemma6b1_cover_M.py", ["row", "--M=3", "--h=1.1e-9", "--tag=_prectrace",
                                                            f"--x={p3c}", "--j=11"], False),
        ("H2 covering check, M = 3", "lemma6b1_cover_M.py", ["check", "--M=3", "--h=1.1e-9", "--tag=_cert",
                                                              f"--x={p3c}", f"--json={t}/check3.json"], False),
        ("H2 covering c1check, M = 3", "lemma6b1_cover_M.py", ["c1check", "--M=3", "--h=1.1e-9", "--tag=_cert",
                                                                f"--x={p3c}"], False),
        ("H2 covering ray, M = 5", "lemma6b1_cover_M.py", ["ray", "--c1", "--M=5", "--h=1e-18", "--tag=_prectrace",
                                                            f"--x={p5}"], False),
        ("H2 sets, M = 3", "theorem_A_sets_M.py", ["--M=3"], False),
        ("H2 sets, M = 5", "theorem_A_sets_M.py", ["--M=5", f"--raw={R}"], True),
        ("H2 lemmas' start sets, M = 3", "start_cover_lemmas_M.py", ["--M=3", "--rerun"], False),
        ("H2 lemmas' start sets, M = 5", "start_cover_lemmas_M.py", ["--M=5", f"--raw={R}", "--rerun"], True),
        # (H3)
        ("H3 lemma 6a(3)", "lemma6_jump_M.py", ["--M=3", "--box=data/A_M_6a_M3_1e-10_in.json",
                                                 f"--json={t}/6a3.json"], False),
        ("H3 lemma 6a(5)", "lemma6_jump_M.py", ["--M=5", f"--box={R}/I103-macm3/A_M2_6a_M5_sub1e-18.json",
                                                 f"--x={p5}", "--rp=0.08", "--k=20", f"--json={t}/6a5.json"], True),
        # the point certificates
        ("certificate box run, M = 3", "certify_M.py", ["box", "3", "--centre=data/A_M_refine_M3_c.json", "--use_out",
                                                         "--h=1e-10", "--w=4e-16", f"--out={t}/box3.json",
                                                         f"--log={t}/box3.log"], False),
        ("certificate Krawczyk, M = 3", "certify_M.py", ["krawczyk", "--box=data/A_M_box_M3_1e-10.json", "--hier"],
         False),
        ("certificate controls, M = 3", "certify_M.py", ["controls", "--box=data/A_M_box_M3_1e-10.json"], False),
        ("certificate recheck, M = 3", "certify_M_recheck.py", ["data/A_M_box_M3_1e-10.json"], False),
        ("certificate export6a, M = 3", "certify_M.py", ["export6a", "--box=data/A_M_box_M3_1e-10.json",
                                                          f"--out={t}/6a_in3.json"], False),
        ("certificate box run, M = 5", "certify_M2.py", ["box", "5", "--chart=data/A_M2_chart_M5.json", "--h=1e-10",
                                                          "--w=1e-18,1e-19", f"--out={t}/box5.json",
                                                          f"--log={t}/box5.log"], False),
        ("certificate Krawczyk, M = 5", "certify_M2.py", ["krawczyk", "--box=data/A_M2_box_M5_1e-10.json"], False),
        ("certificate Krawczyk, M = 5, sub-box", "certify_M2.py", ["krawczyk", "--box=data/A_M2_box_M5_1e-10.json",
                                                                    "--sub=1e-18,1e-26,1e-27"], False),
        ("certificate controls, M = 5", "certify_M2.py", ["controls", "--box=data/A_M2_box_M5_1e-10.json"], False),
        ("certificate recheck, M = 5", "certify_M_recheck.py", ["data/A_M2_box_M5_1e-10.json"], False),
        ("certificate export6a, M = 5", "certify_M2.py", ["export6a", "--box=data/A_M2_box_M5_1e-10.json",
                                                           f"--out={t}/6a_in5.json", "--sub=1e-18,1e-26,1e-27"], False),
    ]


def trace(script, args, limit, tmp, n):
    """prec_trace.py on one command, in its own process group; returns its record (or None if it did not start)"""
    res = os.path.join(tmp, f"trace_{n}.json")
    cmd = [sys.executable, os.path.join(HERE, "prec_trace.py"), f"--limit={limit}", f"--json={res}", "--", script]
    p = subprocess.Popen(cmd + args, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    try:
        p.wait(timeout=limit + 60)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if not os.path.exists(res):
        return None
    with open(res) as fh:
        return json.load(fh)


def mutated_copy(tmp):
    """fixed_point_M_b.py with main()'s precision line moved below the reads of --x, --box and --cbox"""
    src = open(os.path.join(HERE, "fixed_point_M_b.py")).read()
    line = [ln for ln in src.splitlines(keepends=True) if ln.lstrip().startswith("ctx.prec = prec") and "before --x" in ln]
    assert len(line) == 1, "the precision line of fixed_point_M_b.main was not found"
    anchor = "    st = Setup(x, M, opts.get(\"r0\", \"0.07\")"
    assert src.count(anchor) == 1 and src.index(line[0]) < src.index(anchor)
    src = src.replace(line[0], "").replace(anchor, line[0] + anchor)
    d = os.path.join(tmp, "mutated")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "fixed_point_M_b.py")
    with open(path, "w") as fh:
        fh.write(src)
    return path


def check(raw, limit, mutate=None, quiet=False):
    tmp = tempfile.mkdtemp(prefix="prec_chain_")
    say = (lambda *a, **k: None) if quiet else print
    scratch = [os.path.join(HERE, f"lemma6b1_runs_M{M}_prectrace") for M in (3, 5)]
    pre = [os.path.exists(s) for s in scratch]
    try:
        # the row command reads the ray's file: a copy of the stored one in the scratch run directory, so that the
        # row and ray traces cannot write into the stored run directory
        if not pre[0]:
            os.makedirs(scratch[0])
            shutil.copy(os.path.join(HERE, "lemma6b1_runs_M3_cert", "ray.json"), scratch[0])
        cmds = chain(raw, tmp)
        if "only" in OPTS:
            cmds = [c for c in cmds if OPTS["only"] in c[0] or OPTS["only"] in c[1]]
        mut_path = mutated_copy(tmp) if mutate == "early" else None
        bad, notstarted, skipped, known, t0 = [], [], 0, 0, time.time()
        for n, (label, script, args, needs_raw) in enumerate(cmds):
            if needs_raw and not raw:
                skipped += 1
                continue
            target = script
            if mut_path and (label, script) == ("FP_b(5) at the centre", "fixed_point_M_b.py"):
                target = mut_path
            r = trace(target, args, limit, tmp, n)
            if r is None:
                notstarted.append(label)
                say(f"  {label}: the trace did not start")
                continue
            new = [x for x in r["reads"] if (os.path.basename(x["file"]), x["function"]) not in ALLOWED]
            known += len(r["reads"]) - len(new)
            say(f"  {label}: {r['end']}; early reads {len(r['reads'])}"
                + (f", of which known {len(r['reads']) - len(new)}" if r["reads"] else ""))
            for x in new:
                bad.append((label, x))
                say(f"      EARLY READ {os.path.basename(x['file'])}:{x['line']} in {x['function']}: "
                    f"{x['type']}({x['string']!r}) at {x['prec']} bits")
        ok = not bad and not notstarted
        print(f"PREC CHAIN CHECK{' [MUTATION ' + mutate + ']' if mutate else ''}: {len(cmds) - skipped} commands traced "
              f"for {limit:g} s each ({skipped} skipped for want of --raw), {known} known early reads, "
              f"{len(bad)} new, {len(notstarted)} not started, {time.time() - t0:.0f} s: {'PASS' if ok else 'FAIL'}")
        if "json" in OPTS and not quiet:
            with open(OPTS["json"], "w") as fh:
                json.dump({"traced": len(cmds) - skipped, "skipped": skipped, "known": known, "not_started": notstarted,
                           "new": [{"command": b[0], **b[1]} for b in bad], "ok": ok}, fh, indent=1)
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for s, was in zip(scratch, pre):                 # the scratch run directories of the two ray traces
            if not was and os.path.isdir(s):
                shutil.rmtree(s, ignore_errors=True)


def main():
    raw, limit = OPTS.get("raw"), float(OPTS.get("limit", "6"))
    if "selftest" in OPTS:
        a = check(raw, limit, None, quiet=True)
        b = check(raw, limit, "early", quiet=True)
        ok = a and not b
        print(f"SELFTEST: clean check passes {a} (must be True); control passes {b} (must be False): "
              f"{'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    sys.exit(0 if check(raw, limit, OPTS.get("mutate")) else 1)


if __name__ == "__main__":
    main()
