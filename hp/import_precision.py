#!/usr/bin/env python3
"""Import-time precision side effects (#831): a module that other modules import must not change mpmath's working
precision when it is imported, and importing reseed must leave each of its importers' precision as it was.

    cd hp && python3 import_precision.py [--selftest] [--root=..]

(A) Static, over the modules of hp/, solver/, matter/ and the repository root. A module sets the precision at import
when a statement at its module level (not in a function, a class or the `if __name__ == "__main__"` block) assigns
mp.dps or mp.prec (mp.mp, mpmath.mp or an alias), or when it imports such a module at its module level. A library
module is one another scanned module imports (at module level or in a function). Every library module that sets the
precision at import fails the check unless it is in KNOWN, the offenders recorded when the check was written; KNOWN
entries no longer offending are reported, so that the list only shrinks.
(B) Dynamic, reseed's importers: each module that imports reseed (anywhere) is imported in a fresh interpreter, then
reseed; the check fails when mp.dps or mp.prec changes across reseed's import.
(C) Reseeder keeps D: at the default 15 digits, Reseeder(meta, D) with D_str as a string, and with D parsed at 60
digits, stores D within 1e-45 of D_str.
Exit 1 when any of them fails. (B) and (C) need hpkernel (PYTHONPATH of a build). --selftest runs (A) and (B) on a
planted directory, a module that sets the precision at import, one that imports it at module level, one that imports
it in a function and one that sets it only under its main guard: (A) must list the first two and (B) the third, and
nothing else."""
import ast
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = ("hp", "solver", "matter", ".")
# the library modules that set the precision at import when this check was written (2026-11-15, main 75f94140), reseed
# excepted; fixing them is outside #831
KNOWN = {
    "hp/anm_run.py", "hp/dd_fits_mp.py", "hp/dinf_budget.py", "hp/dz0_weak.py", "hp/export_delta_all.py",
    "hp/frontier_audit.py", "hp/full_release.py", "hp/held_fit.py", "hp/lam_frontier.py", "hp/lam_gauge_fit.py",
    "hp/near_null_eq.py", "hp/short_forms.py", "hp/weak_curv.py",
    "solver/beyond_lc/lin_angle.py", "solver/eeg_profile/rational_fit.py", "solver/harmony/implosion/shoot_mp.py",
    "solver/harmony/stokes/stokes_lines.py",
    "solver/harmony/stokes/stokes_series.py", "solver/inner_amp/arg_crossing.py", "solver/inner_amp/pilot_agreement.py",
    "solver/inner_amp/zero_selection.py", "solver/inner_borel/inner_stokes.py", "solver/inner_pert/first_order_fits.py",
    "solver/inner_stokes/lateral.py", "solver/matter/css_inner.py", "solver/plot_selection.py",
    "solver/proof/b8m_model.py", "solver/proof/b8m_numcheck.py", "solver/proof/entire_cert.py",
    "solver/proof/review_p2/p2_common.py", "solver/proof/review_p2/p2_formal.py",
    "solver/proof/review_p2b/p2b_alg.py", "solver/rt/mu_check.py",
}
MP_BASES = {"mp", "mpmath"}


def is_main_guard(node):
    t = node.test if isinstance(node, ast.If) else None
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in t.comparators))


def sets_precision(target, aliases):
    """target is X.dps or X.prec with X mp.mp, mpmath.mp, or a name bound to mpmath.mp"""
    if not (isinstance(target, ast.Attribute) and target.attr in ("dps", "prec")):
        return False
    v = target.value
    if isinstance(v, ast.Name):
        return v.id in aliases
    return isinstance(v, ast.Attribute) and v.attr == "mp" and isinstance(v.value, ast.Name) and v.value.id in MP_BASES


def module_level(tree):
    """the statements run at import: module level, into if/try/with/for blocks, not functions, classes or the main
    guard"""
    out, todo = [], list(tree.body)
    while todo:
        n = todo.pop(0)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or is_main_guard(n):
            continue
        out.append(n)
        for field in ("body", "orelse", "finalbody", "handlers"):
            todo.extend(getattr(n, field, []) or [])
    return out


def imports_of(nodes):
    names = []
    for n in nodes:
        if isinstance(n, ast.Import):
            names += [a.name.split(".")[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            names.append(n.module.split(".")[0])
    return names


def scan(root, dirs=DIRS):
    """{path: dict(name, direct, top (module-level imports), all (every import))} over the directories' .py files"""
    mods = {}
    for d in dirs:
        base = os.path.join(root, d)
        for dp, dn, fn in os.walk(base):
            dn[:] = [] if d == "." else [x for x in dn if not x.startswith(".") and x not in ("target", "__pycache__")]
            for f in fn:
                if not f.endswith(".py"):
                    continue
                path = os.path.relpath(os.path.join(dp, f), root)
                try:
                    tree = ast.parse(open(os.path.join(root, path), encoding="utf-8").read())
                except (SyntaxError, UnicodeDecodeError):
                    continue
                top = module_level(tree)
                aliases = {a.asname for n in top if isinstance(n, ast.ImportFrom) and n.module == "mpmath"
                           for a in n.names if a.name == "mp" and a.asname} | \
                          {"mp" for n in top if isinstance(n, ast.ImportFrom) and n.module == "mpmath"
                           for a in n.names if a.name == "mp" and not a.asname}
                direct = any(sets_precision(t, aliases) for n in top if isinstance(n, (ast.Assign, ast.AugAssign))
                             for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))
                mods[path] = dict(name=f[:-3], direct=direct, top=imports_of(top), all=imports_of(ast.walk(tree)))
    return mods


def resolve(mods):
    """name -> paths, an importer's directory first"""
    byname = {}
    for p, m in mods.items():
        byname.setdefault(m["name"], []).append(p)

    def find(importer, name):
        c = byname.get(name, [])
        same = [p for p in c if os.path.dirname(p) == os.path.dirname(importer)]
        return same or c
    return find


def offenders(mods):
    """(library modules that set the precision at import, directly or by a module-level import, each with the
    modules that set it directly (its roots); the library set)"""
    find = resolve(mods)
    roots = {p: {p} for p, m in mods.items() if m["direct"]}
    changed = True
    while changed:
        changed = False
        for p, m in mods.items():
            r = set(roots.get(p, ()))
            for n in m["top"]:
                for q in find(p, n):
                    if q != p and q in roots:
                        r |= roots[q]
            if r and r != roots.get(p):
                roots[p] = r
                changed = True
    lib = {q for p, m in mods.items() for n in m["all"] for q in find(p, n) if q != p}
    return {p: sorted(r) for p, r in roots.items() if p in lib}, lib


def importers_of(mods, target):
    find = resolve(mods)
    return sorted(p for p, m in mods.items() if any(target in find(p, n) for n in m["all"]) and p != target)


PROBE = """
import sys, os
sys.path.insert(0, os.path.dirname({path!r}))
import mpmath
import {name}
d1 = (mpmath.mp.dps, mpmath.mp.prec)
import {target}
d2 = (mpmath.mp.dps, mpmath.mp.prec)
print("PREC", d1[0], d1[1], d2[0], d2[1])
"""


def dynamic(root, paths, target_path, timeout=300):
    """(B): per importer, the precision after importing it and after importing the target too"""
    tname = os.path.basename(target_path)[:-3]
    out = {}
    for p in paths:
        full = os.path.join(root, p)
        code = PROBE.format(path=full, name=os.path.basename(p)[:-3], target=tname)
        try:
            r = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(full), capture_output=True, text=True,
                               timeout=timeout)
            line = next((ln for ln in r.stdout.splitlines() if ln.startswith("PREC")), None)
            if line is None:
                out[p] = ("not importable", (r.stderr.strip().splitlines() or ["?"])[-1][:160])
            else:
                v = [int(x) for x in line.split()[1:]]
                out[p] = ("changed" if v[:2] != v[2:] else "kept", f"dps {v[0]} -> {v[2]}, prec {v[1]} -> {v[3]}")
        except subprocess.TimeoutExpired:
            out[p] = ("not importable", f"timeout {timeout} s")
    return out


def reseeder_keeps_D():
    """(C): at 15 digits, Reseeder stores D_str's digits whether D comes as the string or as a 60-digit mpf"""
    code = r"""
import sys
sys.path.insert(0, {here!r})
import mpmath as mp
import reseed
mp.mp.dps = 15
s = "252.61346756517887119315726097412841796875012345678"
meta = dict(M=32, nodes=[0.05, 0.1, 0.2, 0.3, 0.4])
with mp.workdps(60):
    ref = mp.mpf(s)
    D60 = mp.mpf(s)
errs = []
for D in (s, D60):
    rs = reseed.Reseeder(meta, D)
    with mp.workdps(60):
        errs.append(abs(rs.D - ref))
print("ERR", mp.nstr(max(errs), 3), int(max(errs) < mp.mpf(10) ** -45))
""".format(here=HERE)
    r = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True, timeout=300)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("ERR")), None)
    if line is None:
        return False, "not run: " + ((r.stderr.strip().splitlines() or ["?"])[-1][:160])
    _, err, ok = line.split()
    return ok == "1", f"|D - D_str| {err} (bound 1e-45)"


def check(root):
    mods = scan(root)
    off, lib = offenders(mods)
    # a library module fails when a module that sets the precision directly, on its module-level import chain or
    # itself, is not in KNOWN
    new = {p: [q for q in r if q not in KNOWN] for p, r in off.items() if any(q not in KNOWN for q in r)}
    gone = sorted(p for p in KNOWN if not any(p in r for r in off.values()))
    print(f"(A) {len(mods)} modules, {len(lib)} imported by another; {len(off)} of those set the precision at import, "
          f"{len(off) - len(new)} through KNOWN modules only")
    for p, r in sorted(new.items()):
        print(f"    NEW: {p} (" + ("sets it" if r == [p] else "through " + ", ".join(r)) + ")")
    for p in gone:
        print(f"    in KNOWN but no longer setting it: {p} (remove it from KNOWN)")
    ok_a = not new
    target = "hp/reseed.py"
    imp = importers_of(mods, target)
    dyn = dynamic(root, imp, target)
    bad_b = [p for p, (v, _) in dyn.items() if v != "kept"]
    print(f"(B) {len(imp)} modules import reseed; precision across reseed's import:")
    for p in imp:
        print(f"    {p}: {dyn[p][0]} ({dyn[p][1]})")
    ok_c, why = reseeder_keeps_D()
    print(f"(C) Reseeder keeps D at 15 digits: {why}: {'pass' if ok_c else 'FAIL'}")
    ok = ok_a and not bad_b and ok_c
    print(f"import_precision: (A) {'pass' if ok_a else 'FAIL'}, (B) {'pass' if not bad_b else 'FAIL'}, "
          f"(C) {'pass' if ok_c else 'FAIL'}: {'pass' if ok else 'FAIL'}")
    return 0 if ok else 1


def selftest():
    files = {
        "bad_lib.py": "import mpmath as mp\nmp.mp.dps = 50\n",
        "via_top.py": "import bad_lib\n",
        "via_func.py": "import mpmath as mp\nmp.mp.dps = 140\n\n\ndef f():\n    import bad_lib\n",
        "guarded.py": "import mpmath as mp\n\n\ndef main():\n    mp.mp.dps = 50\n\n\nif __name__ == '__main__':\n"
                      "    mp.mp.dps = 50\n",
        "uses.py": "def g():\n    import guarded\n    import via_top\n",
    }
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "hp"))
        for n, src in files.items():
            with open(os.path.join(td, "hp", n), "w") as fh:
                fh.write(src)
        mods = scan(td, ("hp",))
        off, _ = offenders(mods)
        dyn = dynamic(td, importers_of(mods, "hp/bad_lib.py"), "hp/bad_lib.py", timeout=60)
    got_a = sorted(off)
    got_b = sorted(p for p, (v, _) in dyn.items() if v != "kept")
    want_a, want_b = ["hp/bad_lib.py", "hp/via_top.py"], ["hp/via_func.py"]
    ok = got_a == want_a and got_b == want_b
    print(f"import_precision selftest: (A) lists {got_a} (want {want_a}); (B) lists {got_b} (want {want_b}): "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        sys.exit(selftest())
    root = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--root=")), os.path.dirname(HERE))
    sys.exit(check(root))
