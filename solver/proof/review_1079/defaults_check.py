#!/usr/bin/env python3
"""Review of #1079, check 1: with the options unset, does feat/certify-excited run what main runs?

    python3 review_1079/defaults_check.py [--base=origin/main] [--mutate=env]      from solver/proof
    python3 review_1079/defaults_check.py pathmix BOX.json POINT.json               the point/box path test

Part 1 (static). For each of the three files the branch changes, the base version is read with `git show`, and the
two versions are compared function by function (ast): every top-level function and constant whose source differs is
listed. The list must be exactly the expected one (EXPECT), so a change outside the options would show.
Part 2 (dynamic). certify_M of the base and of the worktree are imported in subprocesses with a clean environment
and with the options set. Compared: path_for(M), orders(M) and the box run's checkpoint key fields for M = 1, 3, 5.
Unset, they must be equal for every M. Set, M = 1 must be equal and M = 3, 5 must differ (the control that the
comparison can fail). --mutate=env sets CERTIFY_M_NTAY=28 in the "unset" run of the worktree: the check must FAIL.
pathmix. krawczyk_test is given the box file with a copy of the point file whose recorded matching radius
path[3] is changed (0.3 -> 0.2). A test that compared the two paths would refuse it. Reported, not a pass criterion.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.dirname(HERE)
FILES = ["certify_M.py", "lemma6_sector_M.py", "theorem_A_sets_M.py"]
EXPECT = {"certify_M.py": ["(NTAY, NJAC)", "path_for"],
          "lemma6_sector_M.py": ["main"],
          "theorem_A_sets_M.py": ["check", "main", "selftest", "verdict"]}
OPTS = ("CERTIFY_M_NTAY", "CERTIFY_M_NJAC", "CERTIFY_M_XI1")
DUMP = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import certify_M as CM
out = {}
for M in (1, 3, 5):
    p = CM.path_for(M)
    out[M] = {"path": [str(p[0]), p[1].str(40), str(p[2]), str(p[3])], "orders": list(CM.orders(M)),
              "R0": CM.R0[M], "ARG": CM.ARG[M], "PREC": CM.PREC, "KPREC": CM.KPREC}
print(json.dumps(out, sort_keys=True))
"""


def units(src):
    """{name: source segment} for the top-level functions, classes and assignments (docstrings and comments ignored
    by comparing ast dumps)"""
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                                     and isinstance(getattr(node.body[0], "value", None), ast.Constant)) else node.body
            out[node.name] = ast.dump(ast.Module(body=body, type_ignores=[])) + ast.dump(node.args) \
                if isinstance(node, ast.FunctionDef) else ast.dump(node)
        elif isinstance(node, ast.Assign):
            out[", ".join(ast.unparse(t) for t in node.targets)] = ast.dump(node.value)
    return out


def dump(dirname, env):
    e = {k: v for k, v in os.environ.items() if k not in OPTS and not k.startswith("CERTIFY_M_")}
    e.update(env)
    r = subprocess.run([sys.executable, "-c", DUMP, dirname], capture_output=True, text=True, env=e, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def pathmix(box, point):
    sys.path.insert(0, PROOF)
    import certify_M as CM
    with open(box) as fh:
        bx = json.load(fh)
    with open(point) as fh:
        pt = json.load(fh)
    print(f"box path {bx['path']}, point path {pt['path']}")
    r0 = CM.krawczyk_test(bx, pt, verbose=False)
    pt2 = dict(pt, path=pt["path"][:3] + ["0.2" if pt["path"][3] != "0.2" else "0.3"])
    try:
        r1 = CM.krawczyk_test(bx, pt2, verbose=False)
        print(f"point file relabelled to path {pt2['path']}: krawczyk_test runs, ok = {r1['ok']} (clean: {r0['ok']}), "
              f"margins equal: {r1['rel_margins'] == r0['rel_margins']}")
        print("PATHMIX: NOT REFUSED (krawczyk_test compares p and M of the two files, not their paths)")
    except AssertionError as e:
        print(f"PATHMIX: refused ({e})")
    return 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    if args and args[0] == "pathmix":
        return pathmix(args[1], args[2])
    base = opts.get("base", "origin/main")
    ok = True
    tmp = tempfile.mkdtemp(prefix="defaults_check_")
    root = os.path.dirname(os.path.dirname(PROOF))            # the repository root: ls-tree filters by the cwd's prefix
    names = subprocess.run(["git", "ls-tree", "--name-only", f"{base}:solver/proof/"], capture_output=True, text=True,
                           check=True, cwd=root).stdout.split()
    changed = []
    for nm in names:
        if not nm.endswith(".py"):
            continue
        src = subprocess.run(["git", "show", f"{base}:solver/proof/{nm}"], capture_output=True, text=True, check=True,
                             cwd=root).stdout
        with open(os.path.join(tmp, nm), "w") as fh:
            fh.write(src)
        here = os.path.join(PROOF, nm)
        if not os.path.exists(here) or open(here).read() != src:
            changed.append(nm)
    print(f"base {base}: {len(names)} entries in solver/proof; python files that differ in the worktree: {changed}")
    ok &= sorted(changed) == sorted(FILES)
    for nm in FILES:
        a = units(open(os.path.join(tmp, nm)).read())
        b = units(open(os.path.join(PROOF, nm)).read())
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        good = diff == EXPECT[nm]
        ok &= good
        print(f"  {nm}: units whose code differs: {diff} (expected {EXPECT[nm]}): {'ok' if good else 'UNEXPECTED'}")
    # dynamic
    unset_b = dump(tmp, {})
    unset_w = dump(PROOF, {"CERTIFY_M_NTAY": "28"} if opts.get("mutate") == "env" else {})
    same = unset_b == unset_w
    ok &= same
    print(f"options unset: path_for, orders, R0, ARG, PREC, KPREC for M = 1, 3, 5 equal to the base's: {same}")
    for M in ("1", "3", "5"):
        print(f"    M = {M}: base {unset_b[M]['path'][0]}, {unset_b[M]['path'][2]}, {unset_b[M]['path'][3]}, orders "
              f"{unset_b[M]['orders']}; worktree {unset_w[M]['path'][0]}, {unset_w[M]['path'][2]}, "
              f"{unset_w[M]['path'][3]}, orders {unset_w[M]['orders']}")
    set_w = dump(PROOF, {"CERTIFY_M_NTAY": "28", "CERTIFY_M_NJAC": "16", "CERTIFY_M_XI1": "0.2"})
    c1 = set_w["1"] == unset_b["1"]
    c35 = all(set_w[M] != unset_b[M] for M in ("3", "5"))
    ok &= c1 and c35
    print(f"options set (28, 16, 0.2): M = 1 unchanged: {c1}; M = 3 and 5 changed: {c35} "
          f"(M = 5: path {set_w['5']['path'][3]}, orders {set_w['5']['orders']})")
    print("DEFAULTS CHECK: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
