import numpy as np, warnings, sys
warnings.filterwarnings("ignore")
from run_spec import make
for D, f in [(4.0, "spec_kc31.npy"), (5.0, "solD_5.000_kc31.npy"), (6.0, "solD_6.000_kc31.npy")]:
    S = make(31, n=D - 2); z = np.load(f)
    F, J = S.jacobian(z)
    U, s, Vt = np.linalg.svd(J)
    sizes, K, L = S.layout(); offs = np.concatenate([[0], np.cumsum(sizes)])
    names = ["p1", "c0"] + [f"{q}@{x}" for x in S.nodes for q in "pmac"] + ["mS", "aS", "Delta"]
    pars = S.parities() + ["-"]
    print(f"D={D}: smallest singular values {s[-4:]}")
    for v in Vt[-2:]:
        w = [(np.linalg.norm(v[offs[b]:offs[b + 1]]) if b < len(sizes) else abs(v[-1]), b) for b in range(len(sizes) + 1)]
        w.sort(reverse=True)
        b = w[0][1]; seg = v[offs[b]:offs[b + 1]]
        # which harmonic
        if pars[b] == "odd":
            ks = S.kodd; m = len(ks); amp = np.hypot(seg[:m], seg[m:])
        else:
            ks = S.keven; m = len(ks) - 1; amp = np.concatenate([[abs(seg[0])], np.hypot(seg[1:1 + m], seg[1 + m:])])
        print(f"   vector mostly in {[names[x[1]] + f' ({x[0]:.2f})' for x in w[:4]]}; dominant harmonic in {names[b]}: k={ks[np.argmax(amp)]}")
