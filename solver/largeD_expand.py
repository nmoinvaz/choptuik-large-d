"""Large-n expansion of the DSS equations, eps = 1/n, n = D - 2.

    p = eps^(1/2) P,  m = eps^(1/2) M,  c = 1 - sum_{k>=1} eps^k G_k,  b = a^2 = 1 + B,  B = sum_{k>=2} eps^k B_k,
    omega = sum_k eps^k w_k.

Exact equations (p, m odd in theta, x the self-similar radius, horizon at x = 1 where c = 1):
    x c_x = (n-1) c B                                   (E1)
    x B_x = (1+B)(p^2+m^2) - (n-1)(1+B) B                 (E2, b = a^2)
    x(x-c) p_x = -w x p_th + G p - c p + (n/2) c (p+m),   G = (n-1) c B      (E3)
    x(x+c) m_x = -w x m_th - G m + c m - (n/2) c (p+m)                        (E4)
With S = P + M, D = P - M (scaled):
    sum:  x c D_x - c D = x^2 S_x + w x S_th - ((1-eps)/eps) c B D
    diff: S = eps [x^2 D_x - x c S_x + w x D_th] / (c (1-eps)(1+B))
At each order: B_k (E2, algebraic), G_{k-1} (E1, integrated from x = 1, G(1) = 0), S_k (diff), then
x D_k,x - D_k = R_k (sum): D_k = x int R_k / x^2 + x K_k(theta). A term r x^1 in R_k would give x log x, not
allowed at the centre: r(theta) = 0 is the solvability condition.
"""
import sympy as sp

x, th, eps = sp.symbols("x theta epsilon")
NORD = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 3
F = sp.Function("F")(th)
w = sp.symbols(f"w0:{NORD + 2}")
Kf = [sp.Function(f"K{k}")(th) for k in range(NORD + 2)]


def ser_mul(a, b, N):
    return [sp.expand(sum(a[i] * b[k - i] for i in range(k + 1) if i < len(a) and k - i < len(b))) for k in range(N)]


def ser_add(*args, N):
    out = [0] * N
    for s in args:
        for k in range(min(N, len(s))):
            out[k] += s[k]
    return [sp.expand(v) for v in out]


def ser_scale(a, c):
    return [sp.expand(c * v) for v in a]


def ser_shift(a, k, N):            # eps^k * a
    return ([0] * k + list(a))[:N]


def ser_d(a, var):
    return [sp.expand(sp.diff(v, var)) for v in a]


def ser_inv(a, N):                 # 1 / a, a[0] != 0
    out = [sp.Integer(1) / a[0]]
    for k in range(1, N):
        out.append(sp.expand(-sum(a[i] * out[k - i] for i in range(1, k + 1) if i < len(a)) / a[0]))
    return out


N = NORD + 3
S = [sp.Integer(0)] * N            # S_0 = 0
D = [2 * x * F] + [0] * (N - 1)    # D_0 = 2 x F (P0 = x F, M0 = -x F)
G = [sp.Integer(0)] * N            # G_k, c = 1 - sum eps^k G_k
Bs = [sp.Integer(0)] * N
W = list(w[:N]) + [0] * max(0, N - len(w))
conds = []
one_m_eps = [sp.Integer(1), sp.Integer(-1)]


def cser():
    return [sp.Integer(1)] + [-G[k] for k in range(1, N)]


def pm():
    P = [(S[i] + D[i]) / 2 for i in range(N)]
    Mm = [(S[i] - D[i]) / 2 for i in range(N)]
    return P, Mm


for k in range(1, NORD + 1):
    # --- B_{k+1} from E2 (scaled by eps): eps x B_x - eps^2 (1+B)(P^2+M^2) + (1-eps)(1+B) B = 0, order k+1;
    # needs P, M to order k-1 and B to order k
    P, Mm = pm()
    q2 = ser_add(ser_mul(P, P, N), ser_mul(Mm, Mm, N), N=N)
    onepB = ser_add([1], Bs, N=N)
    rest = ser_add(ser_shift(ser_scale(ser_d(Bs, x), x), 1, N),
                   ser_scale(ser_shift(ser_mul(onepB, q2, N), 2, N), -1),
                   ser_mul(one_m_eps, ser_mul(onepB, Bs, N), N), N=N)
    Bs[k + 1] = sp.expand(-rest[k + 1])
    onepB = ser_add([1], Bs, N=N)
    # --- G_k from E1 (scaled): eps x c_x - (1-eps) c B = 0 at order k+1: -x G_k,x = [(1-eps) c B]_{k+1}
    c = cser()
    rhs = ser_mul(one_m_eps, ser_mul(c, Bs, N), N)[k + 1]
    integrand = sp.expand(-rhs / x)
    r0 = sp.simplify(sp.expand(-rhs).subs(x, 0))
    if r0 != 0:
        conds.append((f"E1 order {k + 1}: x^0 term (would give log x in c)", r0))
    G[k] = sp.expand(sp.integrate(integrand, (x, 1, x)))
    c = cser()
    # --- S_k from diff: S = eps [x^2 D_x - x c S_x + w x D_th] / (c (1-eps)(1+B))
    num = ser_add(ser_scale(ser_d(D, x), x ** 2), ser_scale(ser_mul(c, ser_d(S, x), N), -x),
                  ser_scale(ser_mul(W, ser_d(D, th), N), x), N=N)
    den = ser_mul(c, ser_mul(one_m_eps, onepB, N), N)
    Snew = ser_shift(ser_mul(num, ser_inv(den, N), N), 1, N)
    S[k] = sp.expand(Snew[k])
    # --- D_k from sum: x c D_x - c D = x^2 S_x + w x S_th - ((1-eps)/eps) c B D at order k
    Dk_sym = sp.Function("Dk")(x, th)
    Dtmp = D[:k] + [Dk_sym] + [0] * (N - k - 1)
    lhs = ser_add(ser_scale(ser_mul(c, ser_d(Dtmp, x), N), x), ser_scale(ser_mul(c, Dtmp, N), -1), N=N)
    cBD = ser_mul(c, ser_mul(Bs, Dtmp, N), N)            # ((1-eps)/eps) c B D = (1/eps - 1) cBD
    rhs = ser_add(ser_scale(ser_d(S, x), x ** 2), ser_scale(ser_mul(W, ser_d(S, th), N), x),
                  ser_scale([cBD[i + 1] if i + 1 < N else 0 for i in range(N)], -1), cBD, N=N)
    Rk = sp.expand(rhs[k] - (lhs[k] - (x * sp.diff(Dk_sym, x) - Dk_sym)))
    assert not Rk.has(Dk_sym), "D_k leaked into its own forcing"
    r1 = sp.Poly(Rk, x).coeff_monomial(x)
    if sp.simplify(r1) != 0:
        conds.append((f"sum order {k}: x^1 term (solvability)", sp.simplify(r1)))
    integrand = sp.expand((Rk - r1 * x) / x ** 2)
    D[k] = sp.expand(x * sp.integrate(integrand, x) + x * Kf[k])
    print(f"order {k}: B_{k+1} = {sp.factor(Bs[k+1])}", flush=True)
    print(f"   G_{k} = {sp.factor(G[k])}", flush=True)
    print(f"   S_{k} = {sp.collect(S[k], x)}", flush=True)
    print(f"   D_{k} = {sp.collect(D[k], x)}", flush=True)
for name, cnd in conds:
    print("\nCONDITION", name, ":\n   ", cnd)
