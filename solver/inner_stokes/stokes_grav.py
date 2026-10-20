"""Inner Stokes multiplier S(theta0) with gravity, from lateral solutions (route b), Galerkin-truncated in theta.

Small-xi form of the fast solutions of the linearized inner equations (derived: the O(1/xi^2) terms cancel, and the
next order gives  xi a_xi + 2 w a_theta = 2 F^2 a  for the Dd amplitude a, the inner limit of Proposition 3):
    delta Dd ~ e^{-1/(2 xi^2)} xi^{2<F^2>} e^{Pi(theta)} S(theta - 2 w ln xi) (1 + O(xi^2)),
    Pi(theta) = (1/w) int_0^theta (F^2 - <F^2>).
So the jump J(xi, theta) of Dd across the Stokes line, divided by e^{-1/(2xi^2)} xi^{2<F^2>} e^{Pi}, and shifted to the
characteristic label theta0 = theta - 2 w ln xi, gives the multiplier function S(theta0) = sum_nu S_nu e^{i nu theta0}.
Reported: R_nu = S_nu / (2 pi i K'(i nu w) 2 F_nu) at several radii (R = 1 without gravity), and S_nu itself.

    python3 stokes_grav.py M wf A [B3 C3]          F = A sin t + B3 sin 3t + C3 cos 3t, w = wf * w_inf
"""
import sys
import numpy as np
from lateral_np import NP, series_data, Kp, Fharm, W0

R1S = (0.15, 0.175, 0.2, 0.225, 0.25)


def multipliers(M, w, Fh, r0=None, r1s=R1S, gravity=True, NTH=64):
    r0 = r0 or {1: 0.1, 3: 0.085, 5: 0.07}.get(M, 0.07)
    phi0 = 3 * np.pi / 8
    Y0, jt, err = series_data(Fh, w, M, r0 * np.exp(1j * phi0), gravity)
    nm = NP(M, w, gravity)
    th = 2 * np.pi * np.arange(NTH) / NTH
    Fth = sum(v * np.exp(1j * mu * th) for mu, v in Fh.items()).real
    F2m = np.mean(Fth ** 2) if gravity else 0.0
    # Pi(theta) = (1/w) int (F^2 - <F^2>): spectrally
    F2h = np.fft.fft(Fth ** 2) / NTH
    kk = np.fft.fftfreq(NTH, 1 / NTH)
    Pih = np.where(kk != 0, F2h / (1j * kk + (kk == 0)), 0) / w
    Pi = np.real(np.fft.ifft(Pih) * NTH) if gravity else 0 * th
    out = []
    Yr = Y0
    rprev = r0
    for r1 in sorted(r1s):
        Yr = nm.path(Yr, [("ray", phi0, rprev, r1)])
        rprev = r1
        Y = nm.path(Yr, [("arc", r1, phi0, 0.0)])
        D = Y[:nm.no]
        idx = {mu: k for k, mu in enumerate(nm.odd)}
        JD = {mu: D[idx[mu]] - np.conj(D[idx[-mu]]) for mu in nm.odd}
        Jth = sum(v * np.exp(1j * mu * th) for mu, v in JD.items())
        q = Jth * np.exp(1 / (2 * r1 * r1)) * r1 ** (-2 * F2m) * np.exp(-Pi)
        qh = np.fft.fft(q) / NTH
        S = {}
        for nu in nm.odd:
            S[nu] = qh[nu % NTH] * r1 ** (2j * nu * w)          # e^{i nu theta} = e^{i nu theta0} xi^{2 i nu w}
        R = {nu: S[nu] / (2j * np.pi * Kp(1j * nu * w) * 2 * Fh[nu]) for nu in nm.odd if abs(Fh.get(nu, 0)) > 0}
        out.append((r1, S, R, JD))
    return out, err


if __name__ == "__main__":
    M, wf, A = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    B3 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    C3 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    grav = not (len(sys.argv) > 6 and sys.argv[6] == "nograv")
    out, err = multipliers(M, W0 * wf, Fharm(A, B3, C3), gravity=grav)
    print(f"M {M} w/w_inf {wf} A {A} B3 {B3} C3 {C3} gravity {grav} (series truncation {err:.1e})")
    for r1, S, R, JD in out:
        print(f"  xi {r1:5.3f}: " + "  ".join(f"R_{nu} {R[nu]:.5f}" for nu in sorted(R) if nu > 0) +
              "   |S_-1/S_1| %.3e" % (abs(S[-1] / S[1]) if 1 in S else 0), flush=True)
