"""Assemble the review's tail constants and whole-sum constants from the three certified parts (#1062 review).

    python3 constants.py EXPLICIT.json INTERIOR.json HEAD.json OUT.json

C'_mu = sum_lam S_lam w_lam + Cr P_mu (notes/P6_route3.md 10.2's assembly, with the review's own S, Cr, P and W):
w = 1 for the families nu - 2 and nu, P for nu + 2, W for +-1; P_mu = P for mu = nu + 2, else 1.
Whole sum: sup over odd nu of e^{-0.736 nu} (H_mu(nu) + C'_mu nu/(j0 - 1)); for nu >= 101 the closed form G and
nu/(2 nu - 1), each times e^{-0.736 nu} decreasing (notes 3), so their values at 101 bound the rest.
Everything is in mpmath.iv; quoted numbers are rounded outward.
"""
import sys
import json
from mpmath import iv, mp
import model

CERT = {'C': {-2: '28.449', 0: '56.874', 2: '4.7661e6'}, 'whole': {-2: '1.0534', 0: '2.1427', 2: '1.7463e5'},
        'Lambda_head': {-2: '0.011012', 0: '0.058818', 2: '3.3073'}}


def main():
    ex, it, hd = (json.load(open(p)) for p in sys.argv[1:4])
    iv.prec = 120
    P = iv.mpf(it['P'])
    W = iv.mpf(it['W'])
    out = {}
    for mode in ('B4', 'tight'):
        Cm = {}
        for k in (-2, 0, 2):
            tot = iv.mpf(0)
            for key, v in ex.items():
                kk, lam = key.split(':', 1)
                if int(kk) != k:
                    continue
                S = iv.mpf(v['U'])
                if v['d'] == 2:
                    wl = W
                elif lam.replace(' ', '') == 'nu+2':
                    wl = P
                else:
                    wl = iv.mpf(1)
                tot += S * wl
            Cr = iv.mpf(it['Cr_' + mode][str(k)])
            tot += Cr * (P if k == 2 else 1)
            Cm[k] = tot
            print(f"C'_(nu{k:+d}) ({mode}) <= {model.up(tot, 5)}   certificate {CERT['C'][k]}: "
                  f"{'implied' if tot.b <= mp.mpf(CERT['C'][k]) else 'NOT implied'}")
        out['C_' + mode] = {str(k): str(v.b) for k, v in Cm.items()}
        for k in (-2, 0, 2):
            best, at = iv.mpf(0), None
            for nu_s, (lo, hi) in hd[str(k)]['H'].items():
                nu = int(nu_s)
                j0 = max(2 * nu, 10)
                v = iv.exp(-iv.mpf('0.736') * nu) * (iv.mpf(hi) + Cm[k] * nu / (j0 - 1))
                if v.b > best.b:
                    best, at = v, nu
            beyond = iv.mpf(hd[str(k)]['eG101']) + iv.exp(-iv.mpf('0.736') * 101) * Cm[k] * 101 / 201
            whole = max(best.b, beyond.b)
            print(f"  whole sum ({mode}), mu = nu{k:+d}: sup e^(-0.736 nu)(H + C' nu/(j0-1)) <= {model.up(whole, 5)} at nu = {at}"
                  f" (nu >= 101: <= {model.up(beyond, 3)});  certificate {CERT['whole'][k]}")
            out.setdefault('whole_' + mode, {})[str(k)] = str(whole)
    for k in (-2, 0, 2):
        print(f"Lambda_head, mu = nu{k:+d}: <= {model.up(iv.mpf(hd[str(k)]['Lambda_head']), 5)} at nu = {hd[str(k)]['at']};"
              f"  certificate {CERT['Lambda_head'][k]}")
    out['C_cert'] = {str(k): v for k, v in CERT['C'].items()}
    json.dump(out, open(sys.argv[4], 'w'), indent=1)


if __name__ == "__main__":
    main()
