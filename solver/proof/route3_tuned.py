#!/usr/bin/env python3
"""P6 route 3: do tuned inputs keep the remainder's form C nu / j^2? A test on the real amplitude expansion
(notes/P6_route3.md section 10.4, #990).

route3_bound.py bounds the one-step model, whose inputs are Kummer sequences. At harmonic nu's first tuned order
(order nu) the real inputs are the tuned lower harmonics, whose own families are removed (S_mu = 0). This script runs
solver/inner_amp/amp_flint.py's expansion through order K with its tuning, then takes for each target (nu, k) the exact
normalised multiplier sequence R_j of amp_flint.Amp.S_at for every j >= 2 (the sequence S_at fits, without the fit) and
reports j^2 |DR_j / K_nu| / nu, K_nu = 1/(Gamma(a_nu) Gamma(b_nu)). The tuning constants f only shift R_j of the tuned
harmonic by a constant at its own order, so DR_j does not depend on their fit error.

The test (fixed before the run): over the windows j in [J/8, J/4], [J/4, J/2], [J/2, J], the window maxima M1, M2, M3 of
j^2 |DR_j/K_nu| / nu. The form C nu / j^2 survives if M3 / M2 <= 1.25 and M2 / M1 <= 1.25 (bounded up to slow
oscillation; an O(1/j) part would give about 2, a log j factor about log(J)/log(J/2), 1.1 at J = 800). M3 is quoted
against route3_bound's C at kappa 2.

    python3 route3_tuned.py [K] [J]        defaults K = 7, J = 800 (orders 1..K, harmonics |mu| <= K)
    python3 route3_tuned.py --mutate=demix  the de-mixing E x 3/2 in every target: the first tuned orders must fail
                                            (an O(1/j) part). --mutate=psi (h2 x 3/2) changes nothing at a first tuned
                                            order, which has no lower-order sequence of its own harmonic to normalise
"""
import sys
import time

import mpmath as mp

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),
                                              "..", "inner_amp"))
import amp_flint as AF                                  # noqa: E402

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")


class Seq(AF.Amp):
    def R_seq(self, nu, k, jlo=2):
        """amp_flint.Amp.S_at's sequence ys_j = R_j for j = jlo..J (the same code, without the fit)"""
        E, delta = self.theta_series(k)
        H = self.H
        nmax = k // 2 + 1
        s_ = 1j * nu * self.wmp
        a, b = (1 + s_) / 2, (2 + s_) / 2
        hser = [mp.mpc(0)] * (k + 1)
        for kk, v in delta.items():
            if kk <= k:
                hser[kk] = AF.to_mp(v) / 2 * (mp.mpf(3) / 2 if "psi" in MUT else 1)

        def smul(x, y):
            out = [mp.mpc(0)] * (k + 1)
            for i, aa in enumerate(x):
                if aa == 0:
                    continue
                for l in range(k + 1 - i):
                    if y[l] != 0:
                        out[i + l] += aa * y[l]
            return out
        hp = [[mp.mpc(1)] + [mp.mpc(0)] * k]
        for n in range(1, nmax + 1):
            hp.append(smul(hp[-1], [-x for x in hser]))
        Emp = {kk: [AF.to_mp(x) for x in E[kk]] for kk in E}
        ys = []
        for j in range(jlo, self.J + 1):
            base = mp.factorial(j) * mp.rgamma(a + j) * mp.rgamma(b + j)
            ps = [(mp.psi(n - 1, a + j) + mp.psi(n - 1, b + j)) / mp.factorial(n) for n in range(1, nmax + 1)]
            Lser = [mp.mpc(0)] * (k + 1)
            for n in range(1, nmax + 1):
                for q in range(k + 1):
                    Lser[q] -= ps[n - 1] * hp[n][q]
            Nser = [mp.mpc(1)] + [mp.mpc(0)] * k
            term = list(Nser)
            for n in range(1, k + 1):
                term = [x / n for x in smul(term, Lser)]
                if all(t == 0 for t in term):
                    break
                Nser = [x + y for x, y in zip(Nser, term)]
            zser = [mp.mpc(0)] * (k + 1)
            for kp in range(1, k + 1, 2):
                if kp not in self.d:
                    continue
                dj = self.d[kp][j]
                for kpp in range(0, k - kp + 1, 2):
                    if kpp not in Emp:
                        continue
                    Ek = Emp[kpp] if kpp == 0 or "demix" not in MUT else [x * mp.mpf(3) / 2 for x in Emp[kpp]]
                    acc = mp.mpc(0)
                    for mu in range(-min(H, kp), min(H, kp) + 1, 2):
                        e_ = Ek[H + nu - mu] if 0 <= H + nu - mu <= 2 * H else 0
                        if e_ != 0:
                            acc += e_ * AF.to_mp(dj[H + mu])
                    zser[kp + kpp] += acc
            ys.append(base * sum(zser[q] * Nser[k - q] for q in range(k + 1)))
        return ys, mp.gamma(a) * mp.gamma(b)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    K = int(args[0]) if args else 7
    J = int(args[1]) if len(args) > 1 else 800
    t0 = time.time()
    A = Seq(1.0, K, J, prec=320)
    A.run(verbose=False)
    print(f"amp_flint to order {K}, J = {J}, tuned [{time.time() - t0:.0f} s]", flush=True)
    targets = [(nu, nu) for nu in range(3, K + 1, 2)] + [(1, 3), (1, 5), (3, 5)]
    if K >= 7:
        targets.append((5, 7))
    ok = True
    print("| ν | order | first tuned order | M1 [J/8, J/4] | M2 [J/4, J/2] | M3 [J/2, J] | M2/M1 | M3/M2 | form C ν/j² |")
    print("|---|---|---|---|---|---|---|---|---|")
    for nu, k in targets:
        ys, invK = A.R_seq(nu, k)
        f = lambda j: abs((ys[j - 1] - ys[j - 2]) * invK) * j ** 2 / nu           # ys[j - 2] = R_j, ys[j - 1] = R_{j+1}
        win = lambda lo, hi: max(f(j) for j in range(lo, hi + 1))
        M1, M2, M3 = win(J // 8 + 1, J // 4), win(J // 4 + 1, J // 2), win(J // 2 + 1, J - 1)
        good = M3 / M2 <= 1.25 and M2 / M1 <= 1.25
        ok &= good if k == nu else True
        print(f"| {nu} | {k} | {'yes' if k == nu else 'no'} | {mp.nstr(M1, 4)} | {mp.nstr(M2, 4)} | {mp.nstr(M3, 4)} | "
              f"{mp.nstr(M2 / M1, 3)} | {mp.nstr(M3 / M2, 3)} | {'holds' if good else 'does not hold'} |", flush=True)
    print(f"[{time.time() - t0:.0f} s]")
    return ok


if __name__ == "__main__":
    good = main()
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
