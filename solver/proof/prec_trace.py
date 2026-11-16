#!/usr/bin/env python3
"""Which numbers does a script read into arb or acb balls before the working precision is set? (#879)

python-flint starts at 53 bits. A decimal string turned into a ball at that precision is still an enclosure of the
decimal, but its radius is about 1e-16 relative, so a "point" becomes a small box and a box's centre moves by 1e-17.
That happened in fixed_point_M.data_table (#484) and in fixed_point_M_b.main (#867, F2).

This tool runs a script with its arguments under a stand-in for the flint module whose arb and acb record every
construction from a string while ctx.prec is below --low (100 bits): the file, line and function of the caller,
the string and the precision. After the run it also lists the module-level arb and acb values of the modules
loaded from this directory whose relative radius is above 2^-100, which are constants made at import time before
any precision was set. Everything else of flint is the real module, and the balls made are real balls.

    python3 prec_trace.py [--limit=10] [--low=100] [--json=out.json] [--prec=N] -- script.py [arguments ...]
        the script runs as __main__ from the current directory for at most --limit seconds; exit status 0 if no
        early read was recorded, 1 if some were, 2 if the script could not be started. With --prec=N the precision
        is set to N bits before the script and its modules are loaded: the rerun "at the right precision", whose
        output is compared with the plain run's
    python3 prec_trace.py --selftest
        a script that reads its argument before setting the precision must be reported, with a 53-bit constant of
        its own, and the same script with the precision set first must not; exit 0 only then

Limits: a read is seen only if the run reaches it within the time limit, and only on the path the arguments take.
Balls made from floats or from other balls are not reads of a decimal and are not recorded. A script that starts
other scripts as processes is traced without them.
"""
import json
import os
import signal
import sys
import tempfile
import types

import flint as _flint

HERE = os.path.dirname(os.path.abspath(__file__))
LOW = 100
records, seen = [], set()
DIRS = [HERE]


class Limit(Exception):
    pass


def has_str(a):
    if isinstance(a, str):
        return a
    if isinstance(a, (list, tuple)):
        for v in a:
            s = has_str(v)
            if s is not None:
                return s
    return None


def proxy(real, name):
    """a class that stands for the real type: calls are recorded, everything else is the real type's"""
    class Meta(type):
        def __call__(cls, *a, **k):
            prec = _flint.ctx.prec
            if prec < LOW:
                s = has_str(a)
                if s is not None:
                    f = sys._getframe(1)
                    key = (f.f_code.co_filename, f.f_lineno, name, s)
                    if key not in seen:
                        seen.add(key)
                        records.append({"file": os.path.relpath(f.f_code.co_filename, HERE), "line": f.f_lineno,
                                        "function": f.f_code.co_name, "type": name, "string": s[:40], "prec": prec})
            return real(*a, **k)

        def __instancecheck__(cls, obj):
            return isinstance(obj, real)

        def __subclasscheck__(cls, sub):
            return issubclass(sub, real)

        def __getattr__(cls, attr):
            return getattr(real, attr)
    return Meta(name, (), {})


def install():
    mod = types.ModuleType("flint")
    mod.__dict__.update({k: v for k, v in _flint.__dict__.items() if not k.startswith("__") or k in
                         ("__version__", "__FLINT_VERSION__", "__path__", "__file__", "__spec__")})
    mod.arb, mod.acb = proxy(_flint.arb, "arb"), proxy(_flint.acb, "acb")
    sys.modules["flint"] = mod


def constants():
    """module-level balls of the modules loaded from this directory with relative radius above 2^-LOW"""
    out = []

    def rel(v):
        if isinstance(v, _flint.arb):
            m = abs(v.mid())
            return float((v.rad() / m).mid()) if m > 0 else 0.0
        if isinstance(v, _flint.acb):
            return max(rel(v.real), rel(v.imag))
        return 0.0
    for nm, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or not any(os.path.abspath(f).startswith(d) for d in DIRS):
            continue
        if os.path.abspath(f) == os.path.abspath(__file__):
            continue
        for k, v in list(vars(mod).items()):
            vals = v if isinstance(v, (list, tuple)) else list(v.values()) if isinstance(v, dict) else [v]
            try:
                r = max((rel(x) for x in vals), default=0.0)
            except TypeError:
                continue
            if r > 2.0 ** (-LOW):
                out.append({"module": os.path.basename(f), "name": k, "relative radius": r})
    return out


def trace(argv, limit):
    """run argv[0] as __main__ with the stand-in module; returns (records, constants, how it ended)"""
    install()
    sys.argv = list(argv)
    path = os.path.abspath(argv[0])
    DIRS.append(os.path.dirname(path))
    sys.path.insert(0, os.path.dirname(path))
    with open(path) as fh:
        code = compile(fh.read(), path, "exec")
    # the script as the module __main__, kept in sys.modules so that its globals can be read after it stops
    main_mod = types.ModuleType("__main__")
    main_mod.__file__ = path
    sys.modules["__main__"] = main_mod

    def stop(signum, frame):
        raise Limit()
    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, limit)
    end = "finished"
    try:
        exec(code, main_mod.__dict__)
    except Limit:
        end = f"stopped after {limit:g} s"
    except SystemExit as e:
        end = f"exit {e.code}"
    except BaseException as e:                         # the survey goes on past a script's own failure
        end = f"{type(e).__name__}: {str(e)[:80]}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return records, constants(), end


def selftest():
    body = ("import sys\nfrom flint import arb, acb, ctx\n{first}C = 2 * arb.pi() / arb('1.64389375')\n"
            "x = arb(sys.argv[1])\n{second}y = acb(x) * C\nassert isinstance(x, arb)\nprint(y.real.rad() < 1)\n")
    import subprocess
    res = []
    with tempfile.TemporaryDirectory() as d:
        for tag, first, second in (("early", "", "ctx.prec = 320\n"), ("set_first", "ctx.prec = 320\n", "")):
            p = os.path.join(d, f"st_{tag}.py")
            with open(p, "w") as fh:
                fh.write(body.format(first=first, second=second))
            out = os.path.join(d, f"{tag}.json")
            rc = subprocess.run([sys.executable, os.path.abspath(__file__), f"--json={out}", "--", p, "0.1"],
                                capture_output=True, text=True).returncode
            with open(out) as fh:
                r = json.load(fh)
            res.append((tag, rc, len(r["reads"]), len(r["constants"])))
            print(f"{tag}: exit {rc}, {len(r['reads'])} early reads, {len(r['constants'])} low-precision constants")
    ok = res[0][1] == 1 and res[0][2] == 2 and res[0][3] >= 1 and res[1] == ("set_first", 0, 0, 0)
    print(f"SELFTEST: {'PASS' if ok else 'FAIL'} (the script that reads first is reported, the other is not)")
    return ok


def main():
    global LOW
    args = sys.argv[1:]
    if "--selftest" in args:
        sys.exit(0 if selftest() else 1)
    if "--" not in args:
        sys.exit(__doc__)
    cut = args.index("--")
    opts = {a.split("=", 1)[0][2:]: a.split("=", 1)[1] for a in args[:cut] if a.startswith("--") and "=" in a}
    LOW = int(opts.get("low", "100"))
    target = args[cut + 1:]
    if not target or not os.path.exists(target[0]):
        print(f"prec_trace: no such script {target[:1]}")
        sys.exit(2)
    if "prec" in opts:
        _flint.ctx.prec = int(opts["prec"])
    recs, consts, end = trace(target, float(opts.get("limit", "10")))
    print(f"prec_trace: {' '.join(target)}: {end}; {len(recs)} early reads, {len(consts)} low-precision constants")
    for r in recs:
        print(f"  read  {r['file']}:{r['line']} in {r['function']}: {r['type']}({r['string']!r}) at {r['prec']} bits")
    for c in consts:
        print(f"  const {c['module']}: {c['name']}, relative radius {c['relative radius']:.1e}")
    if "json" in opts:
        with open(opts["json"], "w") as fh:
            json.dump({"command": target, "end": end, "reads": recs, "constants": consts}, fh, indent=1)
    sys.exit(1 if recs else 0)


if __name__ == "__main__":
    main()
