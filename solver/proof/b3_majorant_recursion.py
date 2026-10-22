"""The plain majorant recursion of Lemma B3.7 (proof/B3_borel.md), iterated in floating point (NOT a certificate).

Same inequalities as b3_gevrey_certificate.py (M = 1, centre gauge, norm sum |p_j| rho^j on the harmonic +1
coefficient), but kept as a recursion in k instead of being turned into a generating-function system. It keeps the
1/k smallness of interior convolutions that the generating function loses, so its growth rate m_{k+1}/(k m_k) -> 1/t
shows the radius a certified induction on this recursion could reach. Printed: the ratio and 1/ratio at several k.

    python3 b3_majorant_recursion.py [rho=0.75] [K=200]
"""
import sys

import mpmath as mp

mp.mp.dps = 30
rho = mp.mpf(sys.argv[1]) if len(sys.argv) > 1 else mp.mpf("0.75")
K = int(sys.argv[2]) if len(sys.argv) > 2 else 200
A = mp.mpf("2.72")
om = 2 * mp.pi / mp.mpf("1.64389375")
m = [A * rho]                       # ||D_0|| = |2 F_1| rho
s = [mp.mpf(0)]
g = [mp.mpf(0)]
beta = [mp.mpf(0), mp.mpf(0)]
cbar = lambda i: mp.mpf(1) if i == 0 else g[i]          # majorant of ||c_i||
bbar = lambda i: mp.mpf(1) if i == 0 else beta[i]       # majorant of ||b_i||
q = []
mu = [mp.mpf(0), mp.mpf(0)]         # majorant of ||[c (b-1)]_j||
for k in range(1, K + 1):
    q.append(sum(s[i] * s[k - 1 - i] + m[i] * m[k - 1 - i] for i in range(k)))
    bb_k = sum(bbar(i) * beta[k - i] for i in range(0, k - 1))
    beta.append(sum(bbar(i) * q[k - 1 - i] for i in range(k)) + (2 * k - 2) * beta[k]
                + sum(beta[i] * beta[k + 1 - i] for i in range(2, k)) + bb_k)
    mu.append(sum(cbar(i) * beta[k + 1 - i] for i in range(0, k)))
    g.append((mu[k + 1] + mu[k]) / 2)
    kap = [sum(cbar(i) * bbar(j - i) for i in range(j + 1)) for j in range(k + 1)]
    num = rho * (2 * k - 1) * m[k - 1] + sum(cbar(i) * 2 * (k - 1 - i) * s[k - 1 - i] for i in range(k)) \
        + om * rho * m[k - 1]
    s.append(num + sum((kap[k - j] + kap[k - j - 1]) * s[j] for j in range(1, k)))
    mk = rho * s[k] + om * rho * s[k] / 2 \
        + (sum(mu[i] * m[k + 1 - i] for i in range(2, k + 2)) + sum(mu[i] * m[k - i] for i in range(2, k + 1))) / 2 \
        + sum(g[i] * (k - i) * m[k - i] for i in range(1, k + 1))
    m.append(mk)
print(f"rho = {mp.nstr(rho, 4)}: m_(k+1)/(k+1) m_k and the radius it implies")
for k in (10, 50, 100, 150, K - 1):
    r = m[k + 1] / (m[k] * (k + 1))
    print(f"  k = {k:4d}: ratio/k = {mp.nstr(r, 6)}  ->  radius 1/ratio = {mp.nstr(1 / r, 5)}")
