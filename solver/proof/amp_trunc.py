#!/usr/bin/env python3
"""The amplitude expansion with the harmonics truncated at |mu| <= M (the analogue of the M-harmonic lateral model),
to measure order by order how much the harmonics |nu| > M change S_1 (gap (b), the harmonic tail).

Same recursion as amp_flint.Amp (serial reference implementation), with H = M instead of H = K: every product keeps
only |mu| <= M, only nu <= M are tuned, and the late-order fits use every neighbour family present (|nu'| <= M, i.e.
shifts up to 2M; with that complete basis the legacy fit is adequate for the truncated problem).

    python3 amp_trunc.py M K J [wf]      writes trunc_M<M>_K<K>_J<J>_w<wf>.json (sigma_k, tuned f)
"""
import sys
import os
import json
import time
import mpmath as mp
from flint import acb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inner_amp"))
import amp_flint as AF  # noqa: E402


class AmpTrunc(AF.Amp):
    def __init__(self, wf, K, J, Mh, prec=320, Mf=4):
        super().__init__(wf, K, J, prec=prec, fams=tuple(range(2, 2 * Mh + 1, 2)), Mf=Mf)
        self.H = Mh
        self.W = 4 * Mh + 1
        Z = acb(0)
        nz = 2 * self.H + 1
        self.zero_row = [Z] * nz
        self.F = {1: [Z] * nz}
        self.F[1][self.H + 1] = acb(0, -1) / 2
        self.F[1][self.H - 1] = acb(0, 1) / 2

    def run(self, verbose=True, tune=True):
        self.P = {}
        t0 = time.time()
        sig, f_out, S1 = [], {}, {}
        for k in range(1, self.K + 1):
            if k % 2 == 0:
                self.pass_even(k)
                continue
            self.pass_odd(k)
            if tune and k >= 3:
                for nu in range(3, min(k, self.H) + 1, 2):
                    S, rs = self.S_at(nu, k)
                    s_ = 1j * nu * self.wmp
                    linS = 1 / (mp.gamma((1 + s_) / 2) * mp.gamma((2 + s_) / 2))
                    f = -S / (2 * linS)
                    fa = AF.acb(AF.arb(mp.nstr(f.real, 50)), AF.arb(mp.nstr(f.imag, 50)))
                    self.add_linear(k, nu, fa)
                    self.add_linear(k, -nu, fa.conjugate())
                    f_out[f"{k}_{nu}"] = [float(f.real), float(f.imag)]
            self.P[("d", k)] = self.poly(self.d[k])
            self.P[("s", k)] = self.poly(self.s[k])
            S, rs = self.S_at(1, k)
            S1[k] = S
            sig.append(S / S1[1])
            if verbose:
                print(f"k {k}: sigma_{(k - 1) // 2} = {mp.nstr(S / S1[1], 15)}  fit resid {float(rs):.1e}  [{time.time() - t0:.0f}s]",
                      flush=True)
        return sig, f_out


if __name__ == "__main__":
    Mh, K, J = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    wf = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    a = AmpTrunc(wf, K, J, Mh)
    sig, f = a.run()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"trunc_M{Mh}_K{K}_J{J}_w{wf:.4f}.json")
    json.dump({"M": Mh, "wf": wf, "K": K, "J": J, "sigma_str": [[mp.nstr(z.real, 20), mp.nstr(z.imag, 20)] for z in sig],
               "f": f}, open(out, "w"))
    print("wrote", out)
