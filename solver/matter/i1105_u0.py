"""How does the horizon series start u0 move the last block's residual of a stored l = 1 state (#1105)? Residual-only
sweep over u0 and the tau product grid N, on one state, with the record's settings otherwise.

usage: python i1105_u0.py STATE D KC OUT_JSON [--u0=0.01,0.02,...] [--N=408,544,...] [--mutate=steps]
  Record settings: kappa 0.7, rk 1, u0 0.04, Ks 30, NODES_STD, N = 4 ceil((3 kc + 2)/4). Every segment keeps the
  record's RK4 steps except the horizon segment (1 - u0 -> last node), whose steps follow freeze_from's rule at that
  u0 on the record's grid. Per (u0, N): |F|, the last block's |F| and its change dF against u0 0.04 at the same N, and
  that change by tau-harmonic band. At the record's N and u0 0.02 also: the horizon steps doubled (hz2) and the
  series to order 45 (Ks45).
  Sanity (exit 1 if it fails): u0 0.04 at the record's N reproduces the record system's residual bit for bit.
  Control: --mutate=steps adds one RK step to the horizon segment in the sweep, and the sanity check must fail.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD          # noqa: E402

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")
PAR = ("odd", "odd", "even", "even", "odd")    # p, m, a, c, psi
EDGES = (0, 11, 51, 101, 10 ** 9)               # bands 0-10, 11-50, 51-100, 101-


def opt(name, default):
    v = next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), None)
    return default if v is None else [float(x) if name == "u0" else int(x) for x in v.split(",")]


def make(kc, D, z, **kw):
    a = dict(kc=kc, nodes=NODES_STD, n=D - 2, ell=1, Delta_ref=z[-1], Ks=30, u0=0.04, rk=1)
    a.update(kw)
    return DSSEll(**a)


def hkey(S):
    return (round(1 - S.u0, 9), round(S.nodes[-1], 9))


def last_block(S, F):
    blk = 3 * S.n_odd + 2 * S.n_even
    nb = (len(F) - 1) // blk
    return F[(nb - 1) * blk:nb * blk]


def bands(S, B):
    """norm of one block's entries by tau harmonic band (EDGES); entries laid out as in i1103_split.restrict"""
    acc = np.zeros(len(EDGES) - 1)
    o = 0
    for par in PAR:
        if par == "odd":
            ks = list(S.kodd) * 2
        else:
            ks = [0] + list(S.keven[1:]) * 2
        c = B[o:o + len(ks)]
        o += len(ks)
        for k, v in zip(ks, c):
            acc[np.searchsorted(EDGES, k, side="right") - 1] += float(v) ** 2
    assert o == len(B), (o, len(B))
    return [float(v) for v in np.sqrt(acc)]


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    path, D, kc, out = a[0], float(a[1]), int(a[2]), a[3]
    z = np.load(path)
    S0 = make(kc, D, z)
    base = S0.freeze_from(z, 0.7)
    F0 = S0.residual(z[None])[0]
    u0s = opt("u0", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    Ns = opt("N", [S0.N, 544, 816, 1088] if S0.N < 544 else [S0.N, int(4 * np.ceil(1.5 * S0.N / 4)),
                                                              int(4 * np.ceil(2 * S0.N / 4))])
    if 0.04 not in u0s:
        u0s = [0.04] + u0s
    hsteps = {}
    for u0 in sorted(set(u0s) | {0.02}):            # horizon steps by freeze_from's rule, on the record's grid
        Sh = make(kc, D, z, u0=u0)
        hsteps[u0] = Sh.freeze_from(z, 0.7)[hkey(Sh)] + (1 if MUT == "steps" else 0)
    res = dict(state=os.path.basename(path), D=D, kc=kc, N_record=S0.N, steps=int(sum(base.values())),
               h_steps_record=int(base[hkey(S0)]), F=float(np.linalg.norm(F0)),
               F_last=float(np.linalg.norm(last_block(S0, F0))), bands=list(EDGES[:-1]), sweep=[], extra={})
    print(f"{res['state']}: D {D}, k_c {kc}, N {S0.N}, |F| {res['F']:.4e}, last block {res['F_last']:.4e}, "
          f"steps {res['steps']} (horizon {res['h_steps_record']})", flush=True)

    def run(u0, N, h, **kw):
        S = make(kc, D, z, u0=u0, N=N, **kw)
        st = {k: v for k, v in base.items() if k != hkey(S0)}
        st[hkey(S)] = h
        S.steps_map = st
        t = time.time()
        F = S.residual(z[None])[0]
        return S, F, time.time() - t

    sane = None
    for N in Ns:
        ref = None
        for u0 in sorted(u0s, key=lambda v: v != 0.04):      # u0 0.04 first: the reference at this N
            S, F, sec = run(u0, N, hsteps[u0])
            B = last_block(S, F)
            if u0 == 0.04:
                ref = B
                if N == S0.N:
                    sane = bool(np.array_equal(F, F0))
            dB = B - ref
            r = dict(u0=u0, N=N, h_steps=int(hsteps[u0]), F=float(np.linalg.norm(F)), F_last=float(np.linalg.norm(B)),
                     dF_last=float(np.linalg.norm(dB)), dF_last_bands=bands(S, dB), sec=round(sec, 1))
            res["sweep"].append(r)
            print(f"  N {N:5d} u0 {u0:.3f} h_steps {r['h_steps']:6d}: |F| {r['F']:.3e} last {r['F_last']:.3e} "
                  f"dF_last {r['dF_last']:.3e} bands " + " ".join(f"{v:.1e}" for v in r["dF_last_bands"])
                  + f" ({sec:.0f} s)", flush=True)
    ref = last_block(S0, F0)
    for name, u0, kw, fac in (("hz2", 0.02, {}, 2), ("Ks45", 0.02, {"Ks": 45}, 1)):
        S, F, sec = run(u0, S0.N, hsteps[u0] * fac, **kw)
        B = last_block(S, F)
        r = dict(u0=u0, N=S0.N, h_steps=int(hsteps[u0] * fac), F_last=float(np.linalg.norm(B)),
                 dF_last=float(np.linalg.norm(B - ref)), dF_last_bands=bands(S, B - ref), sec=round(sec, 1))
        res["extra"][name] = r
        print(f"  {name}: u0 {u0} h_steps {r['h_steps']}: last {r['F_last']:.3e} dF_last {r['dF_last']:.3e} bands "
              + " ".join(f"{v:.1e}" for v in r["dF_last_bands"]), flush=True)
    res["sanity"] = sane
    with open(out, "w") as fh:
        json.dump(res, fh)
    print(f"sanity (u0 0.04 at N {S0.N} reproduces the record system's residual bit for bit): "
          f"{'pass' if sane else 'FAIL'}")
    print("wrote", out)
    sys.exit(0 if sane else 1)


if __name__ == "__main__":
    main()
