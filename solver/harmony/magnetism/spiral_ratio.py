"""Explicit log-periodicity of the excited self-similar family at the equator map, against published shooting data.

Linearizing u = pi/2 + v near r = 0 (static operator) gives v ~ r^beta, beta^2 + (d-2) beta + (d-1) = 0, complex for
d < 4 + 2 sqrt 2, beta = -(d-2)/2 +- i w, w = sqrt(8d - d^2 - 8)/2 [BB11, eq. for a_n]. The shooting slopes then grow
like a_n ~ C exp(n pi / w) [BB11 heuristic]. Check against the wave-map f'(0) values of [BCT00, Table 1] (d = 3).
Control: the same ratio with d = 4 must not match the d = 3 data.
"""
import math, sys
a = [2, 21.757413, 234.50147, 2522.0683, 27113.388]       # [BCT00] Table 1 (ar5iv, summarizing fetch)
def predicted(d):
    w = math.sqrt(8*d - d*d - 8)/2
    return math.exp(math.pi/w), w
r3, w3 = predicted(3); r4, _ = predicted(4)
obs = [a[i+1]/a[i] for i in range(1, len(a)-1)]
print('d = 3: w =', w3, ' predicted ratio exp(pi/w) =', r3)
print('observed a_{n+1}/a_n, n = 1..3:', obs)
rel = abs(obs[-1] - r3)/r3
print('relative difference at n = 3->4:', rel)
print('control d = 4 ratio:', r4, ' relative difference', abs(obs[-1] - r4)/r4)
ok = rel < 1e-3 and abs(obs[-1] - r4)/r4 > 0.1
print('AS EXPECTED' if ok else 'UNEXPECTED'); sys.exit(0 if ok else 1)
