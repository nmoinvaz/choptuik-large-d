import numpy as np, mpmath as mp
import hpkernel as hk
mp.mp.dps = 50
rng = np.random.default_rng(0)
def split(v):
    hi = np.array([float(x) for x in v]); lo = np.array([float(x - mp.mpf(h)) for x, h in zip(v, hi)]); return hi, lo
def tomp(h, l): return [mp.mpf(a) + mp.mpf(b) for a, b in zip(h, l)]
a = [mp.mpf(rng.standard_normal()) / 3 * mp.exp(rng.uniform(-5, 5)) for _ in range(64)]
b = [mp.mpf(rng.standard_normal()) / 7 * mp.exp(rng.uniform(-5, 5)) for _ in range(64)]
ah, al = split(a); bh, bl = split(b)
for op, f in ((0, lambda x, y: x + y), (1, lambda x, y: x * y), (2, lambda x, y: x / y)):
    oh, ol = hk.dd_op(op, ah, al, bh, bl)
    r = tomp(oh, ol); ex = [f(x, y) for x, y in zip(tomp(ah, al), tomp(bh, bl))]
    print("op", op, "max rel err", float(max(abs((u - v) / v) for u, v in zip(r, ex))))
n = 60
A = mp.matrix(n, n)
for i in range(n):
    for j in range(n):
        A[i, j] = mp.mpf(rng.standard_normal()) + (5 if i == j else 0)
bb = mp.matrix([mp.mpf(rng.standard_normal()) for _ in range(n)])
x = mp.lu_solve(A, bb)
Ah, Al = split([A[i, j] for i in range(n) for j in range(n)])
Bh, Bl = split(list(bb))
xh, xl = hk.dd_solve(Ah.reshape(n, n), Al.reshape(n, n), Bh, Bl)
print("dd_solve max err", float(max(abs(u - v) for u, v in zip(tomp(xh, xl), list(x)))), "isa", hk.isa())
