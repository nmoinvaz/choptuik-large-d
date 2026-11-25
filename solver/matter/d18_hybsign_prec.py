"""#1025: precision check of the l = 1 mode determinant's sign at high k_c. At each lambda, on a stub record's state and
discrete system (d18_hybsign.py's set-up, unchanged):
- hybrid: float64 blocks 0-11, double-double block 12 (d18_hybsign.py's sign, for reference);
- noise: the hybrid sign under relative noise of each amplitude in NOISE on blocks 0-11 (2 seeds each). 1e-15 is
  d18_hybsign.py's test; 1e-12 and 1e-10 bracket the state's own error (record |F| about 4e-10), so a sign that flips
  there is not determined by the converged state;
- alldd: every block in double-double (ell_ddm.DDEll.mode_matrix with only=None), its sign and ln|det|; a sign that
  differs from the hybrid's means float64 rounding in blocks 0-11 decides the hybrid sign.

usage (venv-1025dd): d18_hybsign_prec.py STUB.jsonl STATE.npy D LAMS OUT.jsonl [--noise=1e-15,1e-12,1e-10] [--alldd=1]
       d18_hybsign_prec.py --selftest   (the classifier on planted records; --mutate=class makes it fail)
"""
import json
import os
import sys
import time

import numpy as np

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), None)


def classify(r):
    """the reading of one lambda's record: 'robust' when the hybrid sign survives every noise amplitude and equals the
    all-double-double sign; 'state-limited' when a noise amplitude >= 1e-12 flips it; 'rounding-limited' when the
    all-double-double sign differs; both can hold"""
    out = []
    if any(float(a) >= 1e-12 and any(s != r["sign"] for s in ss) for a, ss in r["noise"].items()):
        out.append("state-limited")
    if r.get("alldd_sign") is not None and r["alldd_sign"] != r["sign"]:
        out.append("rounding-limited")
    if MUT == "class":
        out = ["robust"]
    return ",".join(out) or "robust"


def selftest():
    planted = [
        (dict(sign=1.0, noise={"1e-15": [1.0, 1.0], "1e-10": [1.0, 1.0]}, alldd_sign=1.0), "robust"),
        (dict(sign=1.0, noise={"1e-15": [1.0, 1.0], "1e-10": [1.0, -1.0]}, alldd_sign=1.0), "state-limited"),
        (dict(sign=1.0, noise={"1e-15": [1.0, 1.0], "1e-10": [1.0, 1.0]}, alldd_sign=-1.0), "rounding-limited"),
        (dict(sign=-1.0, noise={"1e-15": [-1.0, -1.0], "1e-12": [1.0, -1.0]}, alldd_sign=1.0), "state-limited,rounding-limited"),
        (dict(sign=1.0, noise={"1e-15": [-1.0, 1.0]}, alldd_sign=None), "robust"),  # 1e-15 alone is not a state test
    ]
    bad = [(r, want, classify(r)) for r, want in planted if classify(r) != want]
    for r, want, got in bad:
        print(f"selftest: {r} -> {got}, want {want}")
    print(f"selftest: {len(planted) - len(bad)} of {len(planted)} planted records classified as written")
    return 0 if not bad else 1


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    recf, st, D, lams, outf = args[0], args[1], float(args[2]), [float(v) for v in args[3].split(",")], args[4]
    amps = [float(v) for v in opts.get("noise", "1e-15,1e-12,1e-10").split(",")]
    alldd = opts.get("alldd", "1") == "1"
    wtm = os.environ.get("P4_WT_MATTER", "/Users/nathan/Source/choptuik-1025/solver/matter")
    sys.path.insert(0, wtm)
    os.chdir(wtm)
    from dss_ell import DSSEll, NODES_STD
    import multidouble as md
    import ell_ddm
    rec = [json.loads(l) for l in open(recf) if json.loads(l).get("D") == D][-1]
    z = np.load(st)
    S = DSSEll(kc=rec["kc"], nodes=NODES_STD, n=D - 2, ell=1, Delta_ref=z[-1], Ks=rec["Ks"], u0=rec["u0"], rk=rec.get("rk", 1))
    steps = S.freeze_from(z, rec["kappa"])
    with np.errstate(all="ignore"):
        F = float(np.linalg.norm(S.residual(z[None, :])[0]))
    same = abs(F - rec["F"]) <= max(1e-2 * rec["F"], 1e-13) and sum(steps.values()) == rec["steps"]
    print(f"D {D} k_c {rec['kc']}: |F| {F:.10e} (record {rec['F']:.10e}), steps {sum(steps.values())} (record {rec['steps']}): "
          f"{'the record system' if same else 'NOT the record system'}", flush=True)
    if not same:
        sys.exit(3)
    E = ell_ddm.DDEll(S)
    blocks, offs = S._block_list()
    blk = 3 * S.n_odd + 2 * S.n_even
    c0, nf, fun, tgt = blocks[12]
    r0 = 12 * blk
    sl = (slice(r0, r0 + blk), slice(c0, c0 + nf))
    mask = np.zeros((S.layout()[2] - 1,) * 2, bool)
    for b, (cc, nn, ff, tt) in enumerate(blocks[:12]):
        mask[b * blk:(b + 1) * blk, cc:cc + nn] = True
    for l in lams:
        t0 = time.time()
        with np.errstate(all="ignore"):
            Mf = S.mode_matrix(z, l)
        sf, lf = np.linalg.slogdet(Mf)
        H, Hl = Mf.copy(), np.zeros_like(Mf)
        Md = E.mode_matrix(z, l, only=[12])
        H[sl] = Md.h[sl]
        Hl[sl] = Md.l[sl]
        s, ld = md.dd_logdet(H, Hl, True)[:2]
        noise = {}
        for a in amps:
            noise[f"{a:g}"] = [float(md.dd_logdet(H + a * np.random.default_rng(100 + k).standard_normal(H.shape) * mask * H, Hl,
                                                  True)[0]) for k in range(2)]
        r = dict(D=D, kc=rec["kc"], lam=l, sign=float(s), ln=float(ld), noise=noise, f64_sign=float(sf), f64_ln=float(lf),
                 state=os.path.basename(st))
        if alldd:
            Ma = E.mode_matrix(z, l)
            sa, la = md.dd_logdet(Ma.h, Ma.l, True)[:2]
            r.update(alldd_sign=float(sa), alldd_ln=float(la))
        r["reading"] = classify(r)
        r["sec"] = round(time.time() - t0)
        print(f"  lam {l!r}: hybrid {s:+.0f} ln|det| {ld:.10f}; noise {noise}; all-dd "
              f"{r.get('alldd_sign', float('nan')):+.0f} {r.get('alldd_ln', float('nan')):.10f}; float64 {sf:+.0f}; "
              f"reading {r['reading']} ({r['sec']} s)", flush=True)
        open(outf, "a").write(json.dumps(r) + "\n")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    main()
