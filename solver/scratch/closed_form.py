"""Search for simple closed forms of Delta (known to ~9 digits) with integer-relation detection.
With only ~9 digits, any 'hit' with small coefficients is very likely a coincidence; the point is to
see whether anything simple fits at all, and how often random numbers produce equally good fits."""
import mpmath as mp, random
mp.mp.dps = 30
Delta = mp.mpf("3.4454523972")   # this work, kc = 63; Martin-Garcia & Gundlach: 3.445452402(3)
tol = mp.mpf("5e-9")
consts = {"pi": mp.pi, "e": mp.e, "ln2": mp.log(2), "sqrt2": mp.sqrt(2), "sqrt3": mp.sqrt(3),
          "euler": mp.euler, "zeta3": mp.zeta(3), "catalan": mp.catalan}
def search(x, maxcoeff=12):
    hits = []
    for name, c in consts.items():
        for target, tname in ((x, "x"), (mp.exp(x), "e^x"), (x / mp.pi, "x/pi"), (x ** 2, "x^2")):
            rel = mp.pslq([target, 1, c], tol=tol, maxcoeff=maxcoeff, maxsteps=10 ** 5)
            if rel and rel[0] != 0:
                approx = -(rel[1] + rel[2] * c) / rel[0]
                if abs(approx - target) < tol * abs(target) * 10:
                    hits.append((tname, name, rel, float(abs(approx - target) / abs(target))))
    return hits
hits = search(Delta)
print("relations a*T + b + c*K = 0 with |coefficients| <= 12, T in {Delta, e^Delta, Delta/pi, Delta^2}:")
print("  ", hits if hits else "none found")
# how often do random numbers in the same range produce a hit?
random.seed(1)
n_hit = sum(1 for _ in range(200) if search(mp.mpf(3 + random.random())))
print(f"control: {n_hit} of 200 random numbers in [3, 4] give a hit at the same tolerance")
