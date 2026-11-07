"""Does alpha_max(D) contain D^-k ln D terms? Fit alpha - 2 with powers u^1..u^K plus u^j ln u for
j >= jlog, and compare the a1, a2 residuals against the pure-power fits (fit_dinf.py)."""
import json, mpmath as mp
mp.mp.dps = 60
data = sorted((mp.mpf(r['D']), mp.mpf(r['alpha_max'])) for r in map(json.loads, open('sweep_D.jsonl')))
def fit(pts, K, jlog, nlog):
    cols = [lambda u, k=k: u**k for k in range(1, K + 1)] + [lambda u, j=j: u**j*mp.log(u) for j in range(jlog, jlog + nlog)]
    A = mp.matrix(len(pts), len(cols)); b = mp.matrix(len(pts), 1)
    for i, (D, al) in enumerate(pts):
        u = 1/D
        for c, f in enumerate(cols): A[i, c] = f(u)
        b[i] = al - 2
    return mp.qr_solve(A, b)[0]
for jlog in (3, 4, 5):
    for Dmin, K, nlog in ((8, 8, 2), (10, 8, 2), (8, 9, 3), (10, 9, 2)):
        pts = [p for p in data if p[0] >= Dmin]
        x = fit(pts, K, jlog, nlog)
        print('ln-terms from u^%d ln u (%d of them), Dmin=%d K=%d: a1-1/2=%s a2-11/16=%s a3=%s log coefs=%s' % (
            jlog, nlog, Dmin, K, mp.nstr(x[0] - 0.5, 2), mp.nstr(x[1] - mp.mpf(11)/16, 2), mp.nstr(x[2], 8), [mp.nstr(x[K + i], 5) for i in range(nlog)]))
