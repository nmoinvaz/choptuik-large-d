"""Per-rung residual changes split by harmonic band: the k_c + 8 embedding (tau truncation of products) and every RK
count times 1.07 (the step change of one rung), for each state of a k_c sequence; all blocks summed per band."""
import sys, os, re, json, numpy as np
sys.path.insert(0, "/Users/nathan/Source/choptuik-1103/solver/matter")
import i1103_split as T
BANDS = [(0, 10), (11, 30), (31, 60), (61, 90), (91, 999)]
def bands(S, F):
    no, ne = S.n_odd, S.n_even; blk = 3 * no + 2 * ne; acc = np.zeros(len(BANDS))
    for b in range((len(F) - 1) // blk):
        o = b * blk
        for par in ("odd", "odd", "even", "even", "odd"):
            n_ = no if par == "odd" else ne; c = F[o:o + n_]; o += n_
            ks = S.kodd if par == "odd" else S.keven
            a2 = c[:len(ks)] ** 2 + c[len(ks):] ** 2 if par == "odd" else np.concatenate([[c[0] ** 2], c[1:len(ks)] ** 2 + c[len(ks):] ** 2])
            for i, (lo, hi) in enumerate(BANDS):
                acc[i] += a2[(ks >= lo) & (ks <= hi)].sum()
    return np.sqrt(acc)
out = []
for path in sys.argv[2:]:
    m = re.search(r"kc(\d+)[^_]*_D([0-9.]+)\.npy$", os.path.basename(path)); kc, D = int(m.group(1)), float(m.group(2))
    z = np.load(path); S = T.make(kc, D, z); base = S.freeze_from(z, 0.7); F0 = S.residual(z[None])[0]
    S8 = T.make(kc + 8, D, z); z8 = S8.regrid_from(z, S); S8.steps_map = dict(base)
    dT = T.restrict(S8.residual(z8[None])[0], S8, S) - F0
    S2 = T.make(kc, D, z); S2.steps_map = {k: int(np.ceil(v * 1.07)) for k, v in base.items()}
    dR = S2.residual(z[None])[0] - F0
    bt, br = bands(S, dT), bands(S, dR)
    out.append(dict(state=os.path.basename(path), D=D, kc=kc, trunc=list(bt), rk=list(br)))
    print(f"kc {kc:3d} D {D:7.4f} | trunc " + " ".join(f"{v:.2e}" for v in bt) + " | RK " + " ".join(f"{v:.2e}" for v in br), flush=True)
json.dump(dict(bands=BANDS, rows=out), open(sys.argv[1], "w"))
