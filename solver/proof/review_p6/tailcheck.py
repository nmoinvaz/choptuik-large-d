"""PC9 of notes/P6_route3_review.md: the review's tail constants against the exact increments.

|dR_j/K_nu| from the review's own Darboux form (model.py's triples, grouped by outer input, mp at 192 bits) for odd
nu = 3..21, each mu, j0 <= j < J, against C'_mu nu/j^2 with the constants of constants.py (B4 mode, tight mode, and the certificate's own C_mu).
Numerical evidence for the certificate, not part of it.

    python3 tailcheck.py CONSTANTS.json [J]          PASS: no violation (J = 600 by default)
    python3 tailcheck.py CONSTANTS.json --mutate=plant   control: three times the bound at five j per run, FAIL, exit 1
"""
import sys
import json
import time
from mpmath import mp
import model


def increments(nu, mu, j_lo, J, C, w):
    """|dR_j/K_nu| for j_lo <= j < J"""
    S = model.Seqs(nu, mu, J + 1, C, w)
    T, Tn = S.T, S.T[nu]
    groups = {}
    for x1, x2, x3 in S.tr:
        key = (x3[0], x3[1])
        g = groups.setdefault(key, {'x3': x3, 'g': [C.mpc(0)] * (J + 2), 'psi': C.mpc(0), 'E': C.mpc(0), 'h': 0})
        G = S.G[(x1[1], x2[1])]
        dd = x1[2] * x2[2]
        for i in range(1, J + 2):
            g['g'][i] += dd * G[i] / (8 * i)
        h = x1[1] + x2[1]
        if h == 0:
            g['psi'] += dd / 8
        else:
            g['E'] += -(dd / 4) / C.mpc(0, h * w)
    a, b = model.ab(C, w, nu)
    snu = C.mpc(0, nu * w)
    for g in groups.values():
        l3, d3 = g['x3'][1], g['x3'][2]
        g['U'] = [d3 * T[l3][l] for l in range(J + 2)]
        g['V'] = [d3 * (2 * l + 1 + C.mpc(0, l3 * w)) * T[l3][l] for l in range(J + 2)]
    out = {}
    for j in range(j_lo, J):
        cj = (2 * j + 2 + snu) / (4 * (j + 1))
        inc = C.mpc(0)
        for g in groups.values():
            gg, U, V = g['g'], g['U'], g['V']
            Ss = C.fsum(gg[i] * V[j - i] for i in range(1, j + 1))
            Sd = C.fsum(gg[i] * U[j + 1 - i] for i in range(1, j + 2))
            inc -= (cj * Ss + Sd) / Tn[j + 1]
            l3, d3 = g['x3'][1], g['x3'][2]
            inc += g['psi'] * d3 * (1 / (a + j) + 1 / (b + j))
            inc += g['E'] * d3 * (T[l3][j + 1] / Tn[j + 1] - T[l3][j] / Tn[j])
        out[j] = abs(inc)
    return out


def main():
    cons = json.load(open(sys.argv[1]))
    J = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 600
    plant = '--mutate=plant' in sys.argv
    C, w = model.ctx('mp', 192)
    t0 = time.time()
    ok = True
    for mode in ('B4', 'tight', 'cert'):
        cons_m = {int(k): mp.mpf(v) for k, v in cons['C_' + mode].items()}
        for k in (-2, 0, 2):
            worst, wat = mp.mpf(0), None
            nviol = 0
            for nu in range(3, 22, 2):
                j0 = max(2 * nu, 10)
                if mode == 'B4':
                    vals = increments(nu, nu + k, j0, J, C, w)
                    cache[(nu, k)] = vals
                else:
                    vals = cache[(nu, k)]
                for idx, (jj, v) in enumerate(sorted(vals.items())):
                    bound = cons_m[k] * nu / mp.mpf(jj) ** 2
                    if plant and idx % ((J - j0) // 5) == 0:
                        v = 3 * bound
                    if v > bound:
                        nviol += 1
                    if v / bound > worst:
                        worst, wat = v / bound, (nu, jj)
            ok &= nviol == 0
            print(f"{mode}: mu = nu{k:+d}, C' = {model.up(cons_m[k], 5)}: worst |dR_j/K|/(C' nu/j^2) = {model.up(worst, 3)}"
                  f" at (nu, j) = {wat}, violations {nviol} [{time.time() - t0:.0f}s]", flush=True)
    print(f"tailcheck.py {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


cache = {}

if __name__ == "__main__":
    main()
