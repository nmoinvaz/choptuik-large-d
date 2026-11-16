#!/usr/bin/env python3
"""Survey of solver/proof for decimals read into balls before the precision is set (#879), with prec_trace.py.

    python3 prec_survey.py collect [--raw=<raw clone>]    the command lines: every "python3 <script>.py ..." line of
        proof/*.md, PROOF.md, README.md, notes/*.md and solver/proof/**/*.sh, and of the raw-data repository's
        exits.txt, run_*.sh and README.md at origin/main, for scripts present here -> prec_survey_cmds.json
    python3 prec_survey.py run [--limit=5] [--only=script.py]
        traces every collected command, and every script that imports flint with no argument, for --limit seconds
        each, from solver/proof -> prec_survey_runs.json. A command with a placeholder is skipped and counted.
        Output options (--json=, --out=, --log=, --tag=, ...) are pointed into prec_survey_tmp/, so that no stored
        file is written; each trace runs in its own process group, which is killed when it ends.
    python3 prec_survey.py table
        per script: its early reads, the constants it leaves at low precision and the commands that reach them
    python3 prec_survey.py rerun [--limit=300] [--prec=320] [--only=script.py]
        every documented command that showed an early read, run twice: as written, and with the precision set to
        --prec bits before the script and its modules load (prec_trace.py --prec). The two outputs are compared:
        exit status, and the lines that differ -> prec_survey_rerun.json. Outputs go to prec_survey_tmp/.
"""
import glob
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CMDS, RUNS, TMP = (os.path.join(HERE, n) for n in ("prec_survey_cmds.json", "prec_survey_runs.json", "prec_survey_tmp"))
PLACEHOLDER = re.compile(r"[<\[\]]|\.\.\.|\$|\bTAG\b|\bSUB\b|=K\b|=S\b|=D\b|PTS/|\bOUT\b|\bB\.json\b|\bPILOT\b")
INPUTS = {"box", "compare", "centre", "fd", "chart", "load-B", "load_B", "ray", "reference", "x", "p", "second",
          "start", "cover", "sets", "cert", "from", "in", "h8log", "adj", "jump", "lipcheck", "load", "label", "point"}
LOG_IS_INPUT = {"b8m_h8.py", "b8m_lcj.py", "b8m_numcheck.py", "b8m_lc_numcheck.py"}     # they read another run's log
# start_cover_M2.py rebuilds a 53-bit box on purpose, and fixed_point_M_b.py's plain path is #878's
INTENDED = {"start_cover_M2.py", "fixed_point_M_b.py"}
FILE_VALUE = re.compile(r"/|\.(json|log|npz|csv|txt|ckpt|svg|png)$")


def collect(raw):
    cmds = {}

    def add(line, src):
        line = re.sub(r"\s+#.*$", "", line.strip())
        line = re.sub(r"\s*(>|2>&1|\|).*$", "", line)
        m = re.match(r"python3 (\S+\.py)(.*)$", line)
        if not m:
            return
        base = os.path.basename(m.group(1))
        if not os.path.exists(os.path.join(HERE, base)) or ("/" in m.group(1) and "solver/proof" not in m.group(1)):
            return
        cmd = re.sub(r"\s+", " ", f"python3 {base}{m.group(2)}").strip().rstrip(".,;:)`")
        cmds.setdefault(cmd, set()).add(src)
    notes = sorted(glob.glob(f"{ROOT}/proof/*.md") + glob.glob(f"{ROOT}/notes/*.md")) + [f"{ROOT}/PROOF.md",
                                                                                         f"{ROOT}/README.md"]
    for f in notes + sorted(glob.glob(f"{HERE}/**/*.sh", recursive=True)):
        for ln in open(f).read().replace("\\\n", " ").splitlines():
            for m in re.finditer(r"python3 \S+\.py[^`\n]*", ln):
                add(m.group(0), os.path.relpath(f, ROOT))
    if raw:
        out = subprocess.run(["git", "-C", raw, "grep", "-h", "-E", r"python3 [A-Za-z_0-9/.]+\.py", "origin/main", "--",
                              "*/exits.txt", "*/run_*.sh", "*/README.md"], capture_output=True, text=True).stdout
        for ln in out.splitlines():
            for m in re.finditer(r"python3 \S+\.py[^`|>#\n]*", ln):
                add(m.group(0), "raw-data repository")
    with open(CMDS, "w") as fh:
        json.dump([{"cmd": c, "src": sorted(s)} for c, s in sorted(cmds.items())], fh, indent=1)
    print(f"collect: {len(cmds)} command lines of {len({c.split()[1] for c in cmds})} scripts -> {CMDS}")


def safe_args(tokens, n, script=""):
    """the arguments with every output option pointed into TMP"""
    out = []
    for t in tokens:
        m = re.match(r"--([A-Za-z_0-9-]+)=(.*)$", t)
        if t == "--resume":
            continue
        if m and m.group(1) == "tag":
            out.append("--tag=_prectrace")
        elif m and m.group(1) == "log" and script in LOG_IS_INPUT:
            out.append(t)
        elif m and m.group(1) not in INPUTS and FILE_VALUE.search(m.group(2)):
            out.append(f"--{m.group(1)}={TMP}/{n}_{os.path.basename(m.group(2))}")
        else:
            out.append(t)
    return out


def one(script, args, limit, n):
    res = os.path.join(TMP, f"{n}.json")
    cmd = [sys.executable, os.path.join(HERE, "prec_trace.py"), f"--limit={limit}", f"--json={res}", "--", script]
    cmd += args
    p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        p.wait(timeout=limit + 60)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(p.pid, signal.SIGKILL)               # the script's own child processes, if any
    except ProcessLookupError:
        pass
    if os.path.exists(res):
        with open(res) as fh:
            return json.load(fh)
    return {"end": "no result (the import or the trace failed)", "reads": [], "constants": []}


def run(limit, only):
    os.makedirs(TMP, exist_ok=True)
    with open(CMDS) as fh:
        cmds = json.load(fh)
    flint_scripts = [os.path.basename(f) for f in sorted(glob.glob(f"{HERE}/*.py"))
                     if re.search(r"from flint import|import flint", open(f).read())
                     and os.path.basename(f) not in ("prec_trace.py", "prec_survey.py")]
    jobs = [{"cmd": f"python3 {s}", "src": ["no argument"]} for s in flint_scripts] + cmds
    runs, skipped, t0 = [], 0, time.time()
    for n, j in enumerate(jobs):
        script = j["cmd"].split()[1]
        if only and script != only:
            continue
        try:
            args = shlex.split(j["cmd"])[2:]
        except ValueError:
            args = None
        if args is None or PLACEHOLDER.search(j["cmd"]):
            skipped += 1
            runs.append({**j, "script": script, "skipped": "placeholder"})
            continue
        args = [a.replace("RAW/", RAWDIR + "/") if RAWDIR else a for a in args]
        r = one(script, safe_args(args, n, script), limit, n)
        runs.append({**j, "script": script, "end": r["end"], "reads": r["reads"], "constants": r["constants"]})
        if r["reads"]:
            print(f"[{time.time() - t0:5.0f} s] {j['cmd'][:110]}: {len(r['reads'])} early reads", flush=True)
    with open(RUNS, "w") as fh:
        json.dump(runs, fh, indent=1)
    done = [r for r in runs if "skipped" not in r]
    print(f"run: {len(done)} traced, {skipped} skipped for a placeholder, {sum(bool(r['reads']) for r in done)} with "
          f"early reads, {time.time() - t0:.0f} s -> {RUNS}")


def table():
    with open(RUNS) as fh:
        runs = json.load(fh)
    reads, consts, traced = {}, {}, {}
    for r in runs:
        if "skipped" in r:
            continue
        traced[r["script"]] = traced.get(r["script"], 0) + 1
        for x in r["reads"]:
            key = (x["file"], x["line"], x["function"])
            d = reads.setdefault(key, {"strings": set(), "cmds": set(), "scripts": set(), "prec": x["prec"]})
            d["strings"].add(x["string"])
            d["cmds"].add(r["cmd"])
            d["scripts"].add(r["script"])
        for c in r["constants"]:
            consts.setdefault((c["module"], c["name"]), set()).add(r["script"])
    print(f"{len(traced)} scripts traced in {sum(traced.values())} runs; {len(reads)} places read a decimal early")
    for (f, ln, fn), d in sorted(reads.items()):
        ex = sorted(d["strings"])
        print(f"{f}:{ln} in {fn} at {d['prec']} bits: {ex[0]!r}" + (f" and {len(ex) - 1} more" if len(ex) > 1 else "")
              + f"; entered by {', '.join(sorted(d['scripts']))}; {len(d['cmds'])} commands, e.g. "
              f"{sorted(d['cmds'], key=len)[0][:90]}")
    print(f"{len(consts)} module-level constants at low precision")
    for (m, nm), sc in sorted(consts.items()):
        print(f"{m}: {nm}; seen from {len(sc)} scripts")
    clean = sorted(s for s in traced if not any(s in d["scripts"] for d in reads.values()))
    print(f"scripts with no early read on any traced path: {len(clean)}")


def rerun(limit, prec, only):
    os.makedirs(TMP, exist_ok=True)
    with open(RUNS) as fh:
        runs = [r for r in json.load(fh) if r.get("reads") and r["src"] != ["no argument"]]
    out, t0 = [], time.time()
    path = os.path.join(HERE, "prec_survey_rerun.json")
    if only and os.path.exists(path):                  # a rerun of one script replaces that script's entries
        with open(path) as fh:
            out = [x for x in json.load(fh) if x["cmd"].split()[1] != only]
    for n, r in enumerate(runs):
        if (only and r["script"] != only) or (not only and r["script"] in INTENDED):
            continue
        args = [a.replace("RAW/", RAWDIR + "/") if RAWDIR else a for a in shlex.split(r["cmd"])[2:]]
        res = {"cmd": r["cmd"], "src": r["src"]}
        texts = {}
        for tag in ("plain", f"p{prec}"):
            a = safe_args(args, f"rr{n}_{tag}", r["script"])
            cmd = [sys.executable, r["script"]] + a if tag == "plain" else \
                [sys.executable, os.path.join(HERE, "prec_trace.py"), f"--prec={prec}", f"--limit={limit}", "--",
                 r["script"]] + a
            t1 = time.time()
            p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 start_new_session=True)
            try:
                txt, _ = p.communicate(timeout=limit + 30)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                txt, rc = "", "time limit"
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            lines = [ln for ln in (txt or "").splitlines() if not ln.startswith(("prec_trace:", "  read ", "  const "))]
            # the tracer turns the script's exit status into its last line; take it from there
            m = re.search(r"prec_trace: .*: (exit \S+|finished|stopped[^;]*|\w+Error[^;]*);", txt or "")
            if tag != "plain" and m:
                rc = {"finished": 0}.get(m.group(1), m.group(1).replace("exit ", ""))
            texts[tag] = [re.sub(TMP + r"/rr\d+_\w+?_", "", ln) for ln in lines]
            res[tag] = {"exit": str(rc), "lines": len(lines), "sec": round(time.time() - t1, 1)}
        a, b = texts["plain"], texts[f"p{prec}"]
        strip = [re.sub(r"\[?\(?\d+(\.\d+)? ?s\)?\]?", "", ln) for ln in a], \
            [re.sub(r"\[?\(?\d+(\.\d+)? ?s\)?\]?", "", ln) for ln in b]
        diff = [(x, y) for x, y in zip(*strip) if x != y]
        res.update(same_exit=res["plain"]["exit"] == res[f"p{prec}"]["exit"], differing_lines=len(diff)
                   + abs(len(a) - len(b)), first_difference=[d[:200] for d in diff[0]] if diff else None)
        out.append(res)
        print(f"[{time.time() - t0:5.0f} s] {r['cmd'][:95]}: exit {res['plain']['exit']} / {res[f'p{prec}']['exit']}, "
              f"{res['differing_lines']} of {len(a)} lines differ", flush=True)
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)
    print(f"rerun: {len(out)} commands, {sum(not r['same_exit'] for r in out)} with another exit status, "
          f"{sum(r['differing_lines'] > 0 for r in out)} with differing lines")


RAWDIR = None


def main():
    global RAWDIR
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[2:] if a.startswith("--")}
    RAWDIR = opts.get("raw")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "collect":
        collect(RAWDIR)
    elif cmd == "run":
        run(float(opts.get("limit", "5")), opts.get("only"))
    elif cmd == "table":
        table()
    elif cmd == "rerun":
        rerun(float(opts.get("limit", "300")), int(opts.get("prec", "320")), opts.get("only"))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
