"""The driver's old numpy double-double helpers against multidouble's package: bitwise equal on random inputs (with
signs, wide exponents, zeros and the edge of the old split's range), and a control that must differ."""
import ast, sys
import numpy as np
import multidouble as md
src = open(sys.argv[1]).read()
tree = ast.parse(src)
keep = {"two_sum", "dd_add", "dd_norm", "_two_prod", "dd_scale", "dd_axpy"}
ns = {"np": np}
code = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in keep) or
        (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_SPLIT" for t in n.targets))]
exec(compile(ast.Module(body=code, type_ignores=[]), "old", "exec"), ns)
rng = np.random.default_rng(7)
n = 20000
def rnd(emax):
    h = (1 + rng.random(n)) * np.exp2(rng.integers(-emax, emax + 1, n)) * rng.choice([-1.0, 1.0], n)
    h[::97] = 0.0
    l = rng.uniform(-1, 1, n) * np.abs(h) * 2.0**-53
    s = h + l
    return s, l - (s - h)
same = lambda x, y: all(np.array_equal(u, v, equal_nan=True) for u, v in zip(x, y))
bad = 0
for emax in (5, 200, 900, 990):
    ah, al = rnd(emax); bh, bl = rnd(emax)
    a = float(rng.uniform(-3, 3))
    checks = {
        "two_sum": (ns["two_sum"](ah, bh), md.two_sum(ah, bh)),
        "dd_add": (ns["dd_add"](ah, al, bh, bl), md.dd_add(ah, al, bh, bl)),
        "two_prod": (ns["_two_prod"](ah, bh), md.two_prod(ah, bh)),
        "dd_scale": (ns["dd_scale"](ah, al, a), md.dd_scale(ah, al, a)),
        "dd_axpy": (ns["dd_axpy"](a, ah, al, bh, bl), md.dd_axpy(a, ah, al, bh, bl)),
        "dd_norm": ((ns["dd_norm"](ah, al),), (md.dd_norm(ah, al),)),
    }
    for k, (x, y) in checks.items():
        ok = same(x, y)
        bad += not ok
        print(f"emax {emax:4d} {k:9s} {'bitwise equal' if ok else 'DIFFERENT'}")
# control: a perturbed result must be reported different
x = md.dd_add(ah, al, bh, bl)
ctl = same(x, (x[0], x[1] * (1 + 2.0**-40)))
print("control (low parts perturbed):", "caught" if not ctl else "NOT CAUGHT")
print("PASS" if bad == 0 and not ctl else "FAIL")
sys.exit(0 if bad == 0 and not ctl else 1)
