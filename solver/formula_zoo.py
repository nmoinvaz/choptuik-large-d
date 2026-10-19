"""Formula zoo: many families of candidate formulas for gamma(D) and Delta(D), each with <= 5 free parameters.
Fit on D <= TRAIN_MAX, score by max error on the exact values beyond (never seen by the fit). n = D - 2."""
import json, sys, warnings, numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaln, digamma
warnings.filterwarnings("ignore")

def load():
    rows = {}
    for l in open("scan_down_kc47.jsonl"):
        r = json.loads(l); rows[round(r["D"], 4)] = dict(D=r["D"], Delta=r["Delta"], gamma=None)
    for l in open("scan_up_kc31.jsonl"):
        r = json.loads(l)
        if r.get("gamma") and r["D"] < 7.7: rows[round(r["D"], 4)] = dict(D=r["D"], Delta=r["Delta"], gamma=r["gamma"])
    for l in open("scan_long_kc23.jsonl"):
        r = json.loads(l); k = round(r["D"], 4)
        rows[k] = dict(D=r["D"], Delta=r["Delta"], gamma=r.get("gamma") or rows.get(k, {}).get("gamma"))
    for l in open("gamma_long.jsonl"):
        r = json.loads(l); k = round(r["D"], 4)
        if k in rows and r.get("gamma"): rows[k]["gamma"] = r["gamma"]
    rows[4.0] = dict(D=4.0, Delta=3.4454523972, gamma=0.3739609)
    rows[5.0] = dict(D=5.0, Delta=3.2217019415, gamma=0.4131269359)
    return rows

G = lambda a: np.exp(gammaln(a))
# ---- families: (name, k, f(D, p), p0) ; n = D - 2, e = D - 3 ----
def fam_gamma():
    F = []
    for k in (3, 4, 5):
        F.append((f"1/2 - sum c_j n^-j            ({k} terms)", k, lambda D, p: 0.5 - sum(c * (D - 2) ** -(j + 1) for j, c in enumerate(p)), np.zeros(k) + 0.1))
        F.append((f"1/2 - sum c_j n^-(j+1)/2      ({k} terms)", k, lambda D, p: 0.5 - sum(c * (D - 2) ** -((j + 1) / 2) for j, c in enumerate(p)), np.zeros(k) + 0.1))
        F.append((f"L - sum c_j n^-j, L free     ({k} params)", k, lambda D, p: p[0] - sum(c * (D - 2) ** -(j + 1) for j, c in enumerate(p[1:])), np.r_[0.5, np.zeros(k - 1) + 0.1]))
        F.append((f"L - sum c_j n^-j/2, L free   ({k} params)", k, lambda D, p: p[0] - sum(c * (D - 2) ** -((j + 1) / 2) for j, c in enumerate(p[1:])), np.r_[0.5, np.zeros(k - 1) + 0.1]))
        F.append((f"1/2 - sum c_j D^-j            ({k} terms)", k, lambda D, p: 0.5 - sum(c * D ** -(j + 1) for j, c in enumerate(p)), np.zeros(k) + 0.1))
    F.append(("1/2 - a n^-alpha (1 + b/n)", 3, lambda D, p: 0.5 - p[0] * (D - 2) ** -p[1] * (1 + p[2] / (D - 2)), [0.3, 0.6, 0.1]))
    F.append(("L - a n^-alpha (1 + b/n)", 4, lambda D, p: p[3] - p[0] * (D - 2) ** -p[1] * (1 + p[2] / (D - 2)), [0.3, 0.6, 0.1, 0.5]))
    F.append(("1/2 - a (n + b)^-alpha", 3, lambda D, p: 0.5 - p[0] * (D - 2 + p[1]) ** -p[2], [0.3, 0.5, 0.6]))
    F.append(("L - a (n + b)^-alpha", 4, lambda D, p: p[3] - p[0] * (D - 2 + p[1]) ** -p[2], [0.3, 0.5, 0.6, 0.5]))
    F.append(("rational [2/2] in D", 5, lambda D, p: (p[0] + p[1] * D + p[2] * D ** 2) / (1 + p[3] * D + p[4] * D ** 2), [0, 0, 0.5, 0, 1]))
    F.append(("rational [1/1] in sqrt(n)", 3, lambda D, p: (p[0] + p[1] * np.sqrt(D - 2)) / (1 + p[2] * np.sqrt(D - 2)), [0, 1, 2]))
    F.append(("rational [2/2] in sqrt(n)", 5, lambda D, p: (p[0] + p[1] * np.sqrt(D - 2) + p[2] * (D - 2)) / (1 + p[3] * np.sqrt(D - 2) + p[4] * (D - 2)), [0, 0, 1, 0, 2]))
    F.append(("Hill: L e^p/(K + e^p), e = D-3", 3, lambda D, p: p[0] * (D - 3) ** p[1] / (p[2] + (D - 3) ** p[1]), [0.5, 0.7, 0.4]))
    F.append(("Hill, L = 1/2", 2, lambda D, p: 0.5 * (D - 3) ** p[0] / (p[1] + (D - 3) ** p[0]), [0.7, 0.4]))
    F.append(("1/2 - a/ln(n + b)", 2, lambda D, p: 0.5 - p[0] / np.log(D - 2 + p[1]), [0.1, 1.0]))
    F.append(("L - a/ln(n + b) - c/n", 4, lambda D, p: p[0] - p[1] / np.log(D - 2 + p[2]) - p[3] / (D - 2), [0.5, 0.1, 1.0, 0.1]))
    F.append(("1/2 - a e^-bn - c/n", 3, lambda D, p: 0.5 - p[0] * np.exp(-p[1] * (D - 2)) - p[2] / (D - 2), [0.3, 0.5, 0.2]))
    F.append(("L - a e^-bn - c/n - d/n^2", 5, lambda D, p: p[0] - p[1] * np.exp(-p[2] * (D - 2)) - p[3] / (D - 2) - p[4] / (D - 2) ** 2, [0.5, 0.3, 0.5, 0.2, 0]))
    F.append(("1/2 - a G(n/2+b)/G(n/2+b+1/2)", 2, lambda D, p: 0.5 - p[0] * G((D - 2) / 2 + p[1]) / G((D - 2) / 2 + p[1] + 0.5), [0.1, 0.5]))
    F.append(("1/2 - a G(n/2+b)/G(n/2+b+c)", 3, lambda D, p: 0.5 - p[0] * G((D - 2) / 2 + p[1]) / G((D - 2) / 2 + p[1] + p[2]), [0.1, 0.5, 0.5]))
    F.append(("1/2 - a/(psi(n/2+b) + c)", 3, lambda D, p: 0.5 - p[0] / (digamma((D - 2) / 2 + p[1]) + p[2]), [0.1, 1.0, 1.0]))
    return F

def fam_delta():
    F = []
    for k in (3, 4, 5):
        F.append((f"sum a_j n^-j, j=0..             ({k} terms)", k, lambda D, p: sum(c * (D - 2) ** -j for j, c in enumerate(p)), np.r_[2, np.zeros(k - 1)]))
        F.append((f"sum a_j n^-j/2, j=0..           ({k} terms)", k, lambda D, p: sum(c * (D - 2) ** -(j / 2) for j, c in enumerate(p)), np.r_[2, np.zeros(k - 1)]))
        F.append((f"n^-1/4 sum a_j n^-j/2           ({k} terms)", k, lambda D, p: (D - 2) ** -0.25 * sum(c * (D - 2) ** -(j / 2) for j, c in enumerate(p)), np.r_[4, np.zeros(k - 1)]))
        F.append((f"n^-1/2 sum a_j n^-j/2           ({k} terms)", k, lambda D, p: (D - 2) ** -0.5 * sum(c * (D - 2) ** -(j / 2) for j, c in enumerate(p)), np.r_[6, np.zeros(k - 1)]))
    F.append(("A n^-alpha (1 + b/n + c/n^2)", 4, lambda D, p: p[0] * (D - 2) ** -p[1] * (1 + p[2] / (D - 2) + p[3] / (D - 2) ** 2), [4, 0.25, 0, 0]))
    F.append(("L + A n^-alpha (1 + b/n)", 4, lambda D, p: p[3] + p[0] * (D - 2) ** -p[1] * (1 + p[2] / (D - 2)), [2, 0.5, 0, 1]))
    F.append(("A (n + b)^-alpha", 3, lambda D, p: p[0] * (D - 2 + p[1]) ** -p[2], [4, 0.5, 0.25]))
    F.append(("L + A (n + b)^-alpha", 4, lambda D, p: p[3] + p[0] * (D - 2 + p[1]) ** -p[2], [3, 0.5, 0.5, 1]))
    F.append(("rational [2/2] in D", 5, lambda D, p: (p[0] + p[1] * D + p[2] * D ** 2) / (1 + p[3] * D + p[4] * D ** 2), [0, 1, 1, 0, 0.3]))
    F.append(("rational [2/2] in sqrt(n)", 5, lambda D, p: (p[0] + p[1] * np.sqrt(D - 2) + p[2] * (D - 2)) / (1 + p[3] * np.sqrt(D - 2) + p[4] * (D - 2)), [3, 0, 1, 0, 0.3]))
    F.append(("a/ln(n/b)", 2, lambda D, p: p[0] / np.log((D - 2) / p[1]), [4, 0.1]))
    F.append(("L + a/ln(n + b)", 3, lambda D, p: p[0] + p[1] / np.log(D - 2 + p[2]), [1, 2, 1]))
    F.append(("L + a/ln(n+b) + c/n", 4, lambda D, p: p[0] + p[1] / np.log(D - 2 + p[2]) + p[3] / (D - 2), [1, 2, 1, 0]))
    F.append(("A n^-alpha ln(n)^beta", 3, lambda D, p: p[0] * (D - 2) ** -p[1] * np.log(D - 2 + 1) ** p[2], [4, 0.25, 0]))
    F.append(("L + a e^-bn + c/n", 4, lambda D, p: p[0] + p[1] * np.exp(-p[2] * (D - 2)) + p[3] / (D - 2), [2, 1, 0.3, 1]))
    F.append(("two-point: A e^p / (1 + B e^q), e = D-3", 4, lambda D, p: p[0] * (D - 3) ** p[1] / (1 + p[2] * (D - 3) ** p[3]), [4, 0.2, 0.3, 0.6]))
    F.append(("A G(n/2+b)/G(n/2+b+c)", 3, lambda D, p: p[0] * G((D - 2) / 2 + p[1]) / G((D - 2) / 2 + p[1] + p[2]), [5, 0.5, 0.25]))
    F.append(("L + A G(n/2+b)/G(n/2+b+c)", 4, lambda D, p: p[3] + p[0] * G((D - 2) / 2 + p[1]) / G((D - 2) / 2 + p[1] + p[2]), [5, 0.5, 0.5, 1]))
    return F

def fit(f, p0, D, y, starts=12, seed=0):
    rng = np.random.default_rng(seed); best = None
    for s in range(starts):
        p = np.array(p0, float) * (1 + (0.5 * rng.standard_normal(len(p0)) if s else 0)) + (0.05 * rng.standard_normal(len(p0)) if s else 0)
        try:
            r = least_squares(lambda q: np.nan_to_num(f(D, q) - y, nan=1e3, posinf=1e3, neginf=-1e3), p, max_nfev=4000)
        except Exception:
            continue
        if best is None or r.cost < best.cost: best = r
    return best.x

def run(target, families, train_max, dmin):
    rows = load(); D = np.array(sorted(d for d in rows if rows[d][target] is not None and d >= dmin)); y = np.array([rows[d][target] for d in D])
    tr, te = D <= train_max, D > train_max
    res = []
    for name, k, f, p0 in families:
        p = fit(f, p0, D[tr], y[tr])
        e_tr = np.abs(f(D[tr], p) - y[tr]).max(); e_te = np.abs(f(D[te], p) - y[te]).max()
        if not np.isfinite(e_te): e_te = np.inf
        res.append(dict(name=name, k=k, train=float(e_tr), test=float(e_te), p=p.tolist(),
                        pred={d: float(f(np.array([float(d)]), p)[0]) for d in (20, 25, 30, 40, 100)}))
    res.sort(key=lambda r: r["test"])
    print(f"\n=== {target}: fit on {dmin} <= D <= {train_max} ({tr.sum()} pts), test on {train_max} < D <= {D.max():.2f} ({te.sum()} pts) ===")
    print(f"{'family':48s} {'k':>2} {'train err':>9} {'TEST err':>9}   prediction at D=20 / 30 / 100")
    for r in res:
        print(f"{r['name']:48s} {r['k']:2d} {r['train']:9.1e} {r['test']:9.1e}   {r['pred'][20]:.5f} / {r['pred'][30]:.5f} / {r['pred'][100]:.5f}")
    return res

if __name__ == "__main__":
    tm = float(sys.argv[1]) if len(sys.argv) > 1 else 11.0
    out = dict(gamma=run("gamma", fam_gamma(), tm, 4.0), Delta=run("Delta", fam_delta(), tm, 3.7))
    json.dump(out, open(f"zoo_train{tm}.json", "w"), indent=1)
