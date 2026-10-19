"""Build a multiple-shooting initial guess from the near-critical simulation slices."""
import numpy as np
from solve4d import resample, shift

def build(N, nodes):
    d = np.load("/home/claude/dss/nearcrit_slices.npz"); g = np.load("/home/claude/dss/guess3.npz")
    Tc, Phi, Pi, a, al, r = d["Tc"], d["Phi"], d["Pi"], d["a"], d["alpha"], d["r"]
    tau = g["tau"]; D = float(g["Delta"])
    E1, E3 = 8.588415, 9.110792
    sel = np.where((Tc >= E1) & (Tc <= E3))[0]
    rs = np.exp(-tau)
    # the collapse code uses G = 1 (8 pi T); the solver uses G_ab = 2n T_ab -> scale the field by sqrt(2 pi)
    sc = np.sqrt(2 * np.pi)
    cols = {"p1": sc * g["p1"], "c0": g["c0"], "mS": sc * g["mS"], "aS": g["aS"]}
    node_fields = []
    for xn in nodes:
        R = xn * rs
        Phi_ = np.array([np.interp(R[k], r, Phi[i]) for k, i in enumerate(sel)])
        Pi_ = np.array([np.interp(R[k], r, Pi[i]) for k, i in enumerate(sel)])
        a_ = np.array([np.interp(R[k], r, a[i]) for k, i in enumerate(sel)])
        c_ = np.array([np.interp(R[k], r, al[i] / a[i]) / np.interp(rs[k], r, al[i] / a[i]) for k, i in enumerate(sel)])
        node_fields.append((sc * R * (Phi_ + Pi_), sc * R * (Phi_ - Pi_), a_, c_))
    parts = [("odd", cols["p1"]), ("even", cols["c0"])]
    for nf in node_fields:
        parts += [("odd", nf[0]), ("odd", nf[1]), ("even", nf[2]), ("even", nf[3])]
    parts += [("odd", cols["mS"]), ("even", cols["aS"])]
    full = [resample(tau, f, N, D, par) for par, f in parts]
    F1 = np.fft.rfft(full[0])[1]
    s = np.angle(F1) / (2 * np.pi / D)
    full = [shift(f, -s, D) for f in full]
    H = N // 2
    return np.concatenate([f[:H] for f in full] + [[D]])
