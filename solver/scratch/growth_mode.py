"""Structure of the unstable (growth) mode at several D: null vector of the mode matrix at lambda0.
For each node x_i: background B_i(tau) = (c q / x), perturbation dQ_i(tau) = d(c q / x).
Questions: is dQ x-independent (like the large-D background)?  How does dB relate to B: amplitude (B), phase (B'),
or something else?"""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
out = {}
for D, f, lam in ((6.0, "solD_6.000_kc31.npy", None), (8.0, "solD_8.000_kc23.npy", 2.238556917), (10.106142578125, "solD_10.106_kc23.npy", 1 / 0.4559589)):
    kc = 31 if "kc31" in f else 23
    S = make(kc, n=D - 2); z = np.load(f)
    if lam is None:
        lam = [json.loads(l)["lam"] for l in open("scan_up_kc31.jsonl") if abs(json.loads(l)["D"] - D) < 1e-6][0]
    M = S.mode_matrix(z, lam)
    U, s, Vt = np.linalg.svd(M)
    v = Vt[-1]
    parts, Dl = S.split(z[None, :]); pars = S.parities()
    dparts, _ = S.split(np.r_[v, 0.0][None, :])
    F = [S.phys(c, p)[0] for c, p in zip(parts, pars)]
    dF = [S.phys(c, p)[0] for c, p in zip(dparts, pars)]
    K = len(S.nodes); N = S.N; tau = np.arange(N) * Dl[0] / N
    Bs, dBs = [], []
    for i in range(K):
        p, m, a, c = F[2 + 4 * i:6 + 4 * i]; dp, dm, da, dc = dF[2 + 4 * i:6 + 4 * i]
        x = S.nodes[i]; q = 0.5 * (p - m); dq = 0.5 * (dp - dm)
        Bs.append(c * q / x); dBs.append((dc * q + c * dq) / x)
    Bs, dBs = np.array(Bs), np.array(dBs)
    sel = [i for i, x in enumerate(S.nodes) if 0.05 <= x <= 0.9]
    B = Bs[sel].mean(0); dB = dBs[sel].mean(0)
    # normalise perturbation sign/size
    k = np.fft.rfftfreq(N, d=Dl[0] / N) * 2 * np.pi
    Bt = np.fft.irfft(1j * k * np.fft.rfft(B), n=N)
    X = np.stack([B, Bt], 1); coef, *_ = np.linalg.lstsq(X, dB, rcond=None)
    resid = dB - X @ coef
    xvar = np.abs(dBs[sel] - dB).max() / np.abs(dB).max()
    # harmonic content of dB vs B
    hB = np.abs(np.fft.rfft(B))[:8] / N; hdB = np.abs(np.fft.rfft(dB))[:8] / N
    print(f"D={D:.3f} lam0={lam:.5f} sigma_min={s[-1]:.1e} (next {s[-2]:.1e})")
    print(f"   x-variation of dQ across nodes (relative): {xvar:.3f}")
    print(f"   dB = {coef[0]:+.4f} B {coef[1]:+.4f} B'  + residual {np.linalg.norm(resid)/np.linalg.norm(dB):.3f} (relative)")
    print(f"   harmonics |B_k|  : " + " ".join(f"{h:.3f}" for h in hB))
    print(f"   harmonics |dB_k| : " + " ".join(f"{h/np.abs(hdB).max():.3f}" for h in hdB) + "  (normalised)")
    # centre lapse perturbation and SSH a perturbation
    c0 = F[1]; dc0 = dF[1]
    print(f"   centre: corr(dc0, c0) = {np.corrcoef(dc0, c0)[0,1]:+.3f};  corr(dc0, B^2) = {np.corrcoef(dc0, B**2)[0,1]:+.3f}; corr(dB, B) = {np.corrcoef(dB, B)[0,1]:+.3f}")
    out[D] = dict(lam=lam, tau=tau.tolist(), B=B.tolist(), dB=dB.tolist())
json.dump(out, open("growth_mode.json", "w"))
