"""Run the sympy 1/n expansion (solver/largeD_expand.py logic) to order 3 with K_k = 0, omega_k = 0 (k >= 1),
and save B_2..B_4, G_1..G_3, S_1..S_3, D_1..D_3 as sympy srepr for the Ecker-Ecker-Grumiller f <= 1 analysis."""
import os
import sys, pickle
sys.argv = ["x", "3"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "largeD_expand.py")).read()
src = src.split("for name, cnd in conds:")[0]
# silence prints
src = src.replace('print(f"order', 'pass  # print(f"order').replace('print(f"   ', 'pass  # print(f"   ')
ns = {}
exec(src, ns)
sp = ns["sp"]; x = ns["x"]; th = ns["th"]; Kf = ns["Kf"]; w = ns["w"]
subs = {Kf[k]: 0 for k in range(len(Kf))}
subs.update({w[k]: 0 for k in range(1, len(w))})
out = {}
for name, arr, rng in (("B", ns["Bs"], range(2, 5)), ("G", ns["G"], range(1, 4)), ("S", ns["S"], range(1, 4)), ("D", ns["D"], range(0, 4))):
    for k in rng:
        out[f"{name}{k}"] = sp.srepr(sp.expand(sp.sympify(arr[k]).subs(subs).doit()))
pickle.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "expand_b.pkl"), "wb"))
print("saved", list(out))
