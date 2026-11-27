"""Which discretisation error does a k_c rung of the l = 1 solver move (#1103)? Residual-only tests on one stored state,
and the gain of the centre datum's harmonics at the first node.

usage: python i1103_split.py STATE D KC OUT_JSON [--planted] [--mutate=NAME]
  Settings are the records' (kappa 0.7, rk 1, u0 0.04, Ks 30, NODES_STD). Each test changes one setting and reports
  the change of the residual, dF = F(changed) - F(record), per shooting block and in norm, beside |F| itself:
    rk2      every RK step count doubled (RK4: F - F(rk2) = 15/16 of the step error)
    rk107    every count times 1.07, the ratio of one k_c rung (k_c + 8) in freeze_from
    Ks45     horizon series to order 45 (the series cut)
    u0_02    horizon series started at u0 = 0.02 (its segment's steps by freeze_from's rule)
    kc+8     the state embedded in k_c + 8 with the same steps (the tail of products past k_c)
    N+25%    the product grid one quarter finer (aliasing)
  gain: d(node-0 state)/d(q_k) by complex step through the centre block, for q's odd harmonics, against the flat-centre
  prediction: the regular harmonic is x^l (Omega x)^-nu J_nu(Omega x) (nu = l + (n - 1)/2, Omega = k omega / c), so its
  psi and p amplitudes at x fall like k^-(nu + 1/2) and k^-(nu - 1/2) past the turning point x_t = nu c / (k omega).
  --planted checks the c_eff fit on an exact Bessel gain; the control --mutate=nuoff fits with nu + 3 and must fail.
"""
import json, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD          # noqa: E402

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")


def env(G, w=2):
    G = np.abs(G) if MUT != "noabs" else np.asarray(G)
    return np.array([G[max(0, i - w):i + w + 1].max() for i in range(len(G))])


def flat_gain(k, nu, om, x, c):
    """|(Omega x)^-nu J_nu(Omega x)| times Gamma(nu + 1) 2^nu (1 at Omega x -> 0), Omega = k om / c"""
    from scipy.special import jv, gammaln
    z = np.asarray(k, float) * om * x / c
    return np.abs(np.exp(gammaln(nu + 1) + nu * np.log(2.0) - nu * np.log(z)) * jv(nu, z))


def fit_ceff(k, G, nu, om, x):
    """c_eff and amplitude A with env(G) ~ A env(flat_gain(c_eff)), least squares in log over a grid of c_eff"""
    Eg = env(G)
    best = (np.inf, None, None)
    for c in np.geomspace(0.05, 3.0, 600):
        Em = env(flat_gain(k, nu, om, x, c))
        r = np.log(Eg) - np.log(Em)
        if not np.all(np.isfinite(r)):
            continue
        a = np.mean(r); rms = float(np.sqrt(np.mean((r - a) ** 2)))
        if rms < best[0]:
            best = (rms, float(c), float(np.exp(a)))
    return dict(rms=best[0], c_eff=best[1], A=best[2])


def planted():
    nu, x, om, c = 11.0, 0.02, 5.35, 0.6
    k = np.arange(1, 136, 2).astype(float)
    G = flat_gain(k, nu, om, x, c) * 3e-2 * (1 + 0.2 * np.cos(0.3 * k))
    r = fit_ceff(k, G, nu + (3 if MUT == 'nuoff' else 0), om, x)
    ok = r["c_eff"] is not None and abs(r["c_eff"] / c - 1) < 0.05
    print(f"planted gain: flat-centre Bessel gain with c 0.6 -> c_eff {r['c_eff']}, rms {r['rms']:.3f} -> "
          f"{'pass' if ok else 'FAIL'}")
    return ok


def make(kc, D, z, **kw):
    a = dict(kc=kc, nodes=NODES_STD, n=D - 2, ell=1, Delta_ref=z[-1], Ks=30, u0=0.04, rk=1)
    a.update(kw)
    return DSSEll(**a)


def restrict(Fb, Sb, S):
    """residual of the larger layout Sb (13 blocks of p, m, a, c, psi spectra, then the phase row) restricted to the
    harmonics of S"""
    PAR = ("odd", "odd", "even", "even", "odd")
    blk_b = 3 * Sb.n_odd + 2 * Sb.n_even
    out = []
    for b in range((len(Fb) - 1) // blk_b):
        o = b * blk_b
        for par in PAR:
            n_ = Sb.n_odd if par == "odd" else Sb.n_even
            c = Fb[o:o + n_]; o += n_
            ko, kn = (Sb.kodd, S.kodd) if par == "odd" else (Sb.keven, S.keven)
            if par == "odd":
                mo = len(ko); re, im = dict(zip(ko, c[:mo])), dict(zip(ko, c[mo:]))
                out.append(np.concatenate([[re[k] for k in kn], [im[k] for k in kn]]))
            else:
                mo = len(ko) - 1; re = dict(zip(ko[1:], c[1:1 + mo])); im = dict(zip(ko[1:], c[1 + mo:]))
                out.append(np.concatenate([[c[0]], [re[k] for k in kn[1:]], [im[k] for k in kn[1:]]]))
    return np.concatenate(out + [Fb[-1:]])


def per_block(S, F):
    blk = 3 * S.n_odd + 2 * S.n_even
    nb = (len(F) - 1) // blk
    return [float(np.linalg.norm(F[i * blk:(i + 1) * blk])) for i in range(nb)]


def main():
    ok = planted()
    if "--planted" in sys.argv or not ok:
        print("planted tests:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    path, D, kc, out = a[0], float(a[1]), int(a[2]), a[3]
    z = np.load(path)
    S = make(kc, D, z)
    base = S.freeze_from(z, 0.7)
    F0 = S.residual(z[None])[0]
    res = dict(state=os.path.basename(path), D=D, kc=kc, steps=int(sum(base.values())), F=float(np.linalg.norm(F0)),
               F_blocks=per_block(S, F0), tests={})
    print(f"{res['state']}: |F| {res['F']:.3e}, steps {res['steps']}")

    def report(name, S2, F2, F2_old=None):
        dF = (F2_old if F2_old is not None else F2) - F0
        r = dict(dF=float(np.linalg.norm(dF)), dF_blocks=per_block(S, dF))
        if F2_old is not None:
            r["F_new_harmonics"] = float(np.linalg.norm(F2) ** 2 - np.linalg.norm(F2_old) ** 2) ** 0.5 \
                if np.linalg.norm(F2) > np.linalg.norm(F2_old) else 0.0
        res["tests"][name] = r
        print(f"  {name:8s} |dF| {r['dF']:.3e}  ({r['dF'] / res['F']:.1f} |F|); per block "
              + " ".join(f"{v:.0e}" for v in r["dF_blocks"])
              + (f"; |F| in the new harmonics {r['F_new_harmonics']:.3e}" if "F_new_harmonics" in r else ""), flush=True)

    for name, fac in (("rk2", 2.0), ("rk107", 1.07)):
        S2 = make(kc, D, z)
        S2.steps_map = {k: int(np.ceil(v * fac)) for k, v in base.items()}
        report(name, S2, S2.residual(z[None])[0])
    S2 = make(kc, D, z, Ks=45); S2.steps_map = dict(base)
    report("Ks45", S2, S2.residual(z[None])[0])
    S2 = make(kc, D, z, u0=0.02); st = S2.freeze_from(z, 0.7)
    report("u0_02", S2, S2.residual(z[None])[0])
    # the tail of products past k_c: embed in k_c + 8, same steps, and read the old harmonics' residual
    S8 = make(kc + 8, D, z); z8 = S8.regrid_from(z, S); S8.steps_map = dict(base)
    F8 = S8.residual(z8[None])[0]
    back = restrict(F8, S8, S)                       # the residual's old harmonics, in k_c's layout (phase row kept)
    report("kc+8", S8, F8, back)
    NN = int(4 * np.ceil(1.25 * S.N / 4))
    S2 = make(kc, D, z, N=NN); S2.steps_map = dict(base)
    report("N+25%", S2, S2.residual(z[None])[0])

    # gain of q's odd harmonics at node 0
    blocks, offs = S._block_list()
    c0_, nf, fun, tgt = blocks[0]
    m = len(S.kodd)
    ks = [k for k in S.kodd if k % 10 == 1 or k == S.kodd[-1]]
    X = np.repeat(z[None, c0_:c0_ + nf], len(ks), axis=0).astype(complex)
    h = 1e-20
    for j, k in enumerate(ks):
        X[j, list(S.kodd).index(k)] += 1j * h                # Re of q_k
    om = np.full(len(ks), 2 * np.pi / z[-1], dtype=complex)
    st0 = fun(X, om)                                         # (5, B, N)
    f = S.fields(z)
    cbar = float(np.mean(f["c0"]))
    nu = 1 + (D - 3) / 2
    G = {}
    for fi, nm in ((0, "p"), (1, "m"), (4, "psi")):
        Y = np.fft.rfft(st0[fi].imag / h, axis=-1) / S.N * 2
        G[nm] = [float(abs(Y[j, k])) for j, k in enumerate(ks)]
    kk = np.array(ks, float)
    omg = 2 * np.pi / z[-1]
    kt = nu * cbar / (omg * NODES_STD[0])
    fp = fit_ceff(kk, np.array(G["psi"]), nu, omg, NODES_STD[0])
    res["gain"] = dict(k=[int(v) for v in ks], G=G, nu=nu, c0_mean=cbar, c0_min=float(f["c0"].min()), k_turn_mean_c0=kt, fit_psi=fp)
    print(f"gain of q_k at x = {NODES_STD[0]} (same harmonic): nu {nu}, c0 mean {cbar:.3f} min {f['c0'].min():.3f}, "
          f"turning harmonic at the mean c0 {kt:.0f}")
    pred = flat_gain(kk, nu, omg, NODES_STD[0], fp["c_eff"]) * fp["A"] if fp["c_eff"] else np.full(len(kk), np.nan)
    for j, k in enumerate(ks):
        print(f"  k {k:4d}: psi {G['psi'][j]:.3e} (flat-centre fit {pred[j]:.3e})  p {G['p'][j]:.3e}  m {G['m'][j]:.3e}")
    print(f"  flat-centre Bessel fit of the psi gain: c_eff {fp['c_eff']}, amplitude {fp['A']}, rms(ln) {fp['rms']:.3f}")
    with open(out, "w") as fh:
        json.dump(res, fh)
    print("wrote", out)


if __name__ == "__main__":
    main()
