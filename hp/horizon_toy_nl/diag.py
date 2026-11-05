#!/usr/bin/env python3
"""Singular values of each formulation's Jacobian at the planted solution, and the re-seed's horizon-datum error
for node data off by delta (the slab check): b_h(reseed) - b_h(regular member through b0), against delta / G."""
import sys, random, json
sys.path.insert(0, '.')
import mpmath as mp
from mpmath import mpf
import toy_nl as tn
dps = int(sys.argv[1]); mp.mp.dps = dps
out = open(f"diag_{dps}.jsonl", "w")
for s in sys.argv[2].split(","):
    T = tn.Toy(s, dps)
    zs = [T.ref["bh"], T.ref["b0"], T.ref["y0"]]
    rec = dict(s=float(s), dps=dps)
    _, J1, _ = tn.res_series(T)(zs)
    rec["sv_series"] = [float(x) for x in mp.svd_r(mp.matrix(J1), compute_uv=False)]
    for ue in ("0.15", "0.1"):
        _, J2, _ = tn.res_inward(T, ue)([zs[1], zs[2], zs[0]])
        rec[f"sv_inward_{ue}"] = [float(x) for x in mp.svd_r(mp.matrix(J2), compute_uv=False)]
    B, Y, dB, dY, nt, conv = T.hseries(zs[0], T.u0)
    rec["G"] = float(dB)
    # re-seed from node data off by delta: the reseeded b_h's offset from bh*, relative
    rs = []
    for d in ("1e-1", "1e-3"):
        b0 = zs[1] * (1 + mpf(d)); y0 = zs[2] * (1 - mpf(d))
        bh = tn.reseed(T, b0, y0)
        rs.append(dict(delta=float(d), rel_bh_offset=float((bh - zs[0]) / zs[0]) if bh is not None else None))
    rec["reseed"] = rs
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(f"s {s} dps {dps}: G {float(dB):.2e}; sv series {[f'{x:.1e}' for x in rec['sv_series']]}; "
          f"inward 0.15 {[f'{x:.1e}' for x in rec['sv_inward_0.15']]}; inward 0.1 {[f'{x:.1e}' for x in rec['sv_inward_0.1']]}; "
          f"reseed b_h offset for node off 1e-1, 1e-3: {[r['rel_bh_offset'] for r in rs]}", flush=True)
