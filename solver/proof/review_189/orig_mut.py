#!/usr/bin/env python3
"""The original integrator (validated_int.Lohner with lemma6_common, unchanged) under the review's mutations (#189).

The mutations are those of capd_m5 (--sign=-1, --eps=e, a perturbed start), applied to the original's systems by
subclassing lemma6_usys_M.SysUM / SysXiM (rhs and lin both, so the enclosures stay those of the mutated system):
    sign   the sign of [[C E] P] in the V equation flipped: u-form dV += z [CE P]; xi-form dS -= 2 [(cE) D] / xi^2
    eps=e  the 2 of -(2 + s) Q made 2 (1 + e): u-form dV += e z Q; xi-form dS -= 2 e S / xi
    start=i:delta   delta added to real coordinate i of the start (Re y_k for i = 2k, Im y_k for i = 2k + 1)
The run's boxes are written in capd_m5's BOX format, so compare.py checks them against the stored boxes.

    orig_mut.py b1 <ray.json> [--legs=1] [--mutate=sign|eps=1e-9|start=i:delta|none]   the ray legs from sets["0"]
    orig_mut.py a <box.json> --wf=<decimal> [--h=1e-18] [--mutate=...]                 lemma 6a's backward flow
"""
import json
import os
import sys
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.dirname(HERE)
sys.path.insert(0, PROOF)
import validated_int as VI  # noqa: E402
import lemma6_common  # noqa: E402
lemma6_common.install()
import lemma6_usys_M as UM  # noqa: E402
import lemma6b1_cover_M as COV  # noqa: E402
import lemma6_jump_M as JM  # noqa: E402
import fixed_point_M as FPM  # noqa: E402
from flint import arb, acb, ctx  # noqa: E402


def parse_mut(s):
    if not s or s == "none":
        return None, None
    if s == "sign":
        return "sign", None
    k, v = s.split("=", 1)
    return k, v


class SysUMmut(UM.SysUM):
    def __init__(self, M, kind, val):
        super().__init__(M)
        self.kind, self.eps = kind, (acb(arb(val)) if kind == "eps" else None)

    def rhs(self, Y, u, z, zero):
        F, cache = super().rhs(Y, u, z, zero)
        P, V, C, w, s, Cm1, Q, CQ, W, Eh, CE = cache
        no = self.no
        if self.kind == "sign":
            CEP = UM.pr(CE, P, self.odd, zero)
            for i, m in enumerate(self.odd):
                F[no + i] = F[no + i] + z * CEP[m]
        elif self.kind == "eps":
            for i, m in enumerate(self.odd):
                F[no + i] = F[no + i] + self.eps * z * Q[m]
        return F, cache

    def lin(self, cache, dY, u, z, zero):
        F = super().lin(cache, dY, u, z, zero)
        P, V, C, w, s, Cm1, Q, CQ, W, Eh, CE = cache
        odd, even, no = self.odd, self.even, self.no
        dP_, dV_, dCt, dw = self.unpack(dY)
        if self.kind == "sign":
            dC_ = {n: dCt[n] / self.kap for n in even}
            dEh = UM.pr(P, dP_, even, zero)
            a, b = UM.pr(dC_, Eh, even, zero), UM.pr(C, dEh, even, zero)
            dCE = {n: a[n] + b[n] for n in even}
            pa, pb = UM.pr(dCE, P, odd, zero), UM.pr(CE, dP_, odd, zero)
            for i, m in enumerate(odd):
                F[no + i] = F[no + i] + z * (pa[m] + pb[m])
        elif self.kind == "eps":
            for i, m in enumerate(odd):
                ds = UM.I * m * dw
                dQ = dV_[m] + (1 + s[m]) * dP_[m] + ds * P[m]
                F[no + i] = F[no + i] + self.eps * z * dQ
        return F


class SysXiMmut(UM.SysXiM):
    def __init__(self, M, kind, val):
        super().__init__(M)
        self.kind, self.eps = kind, (acb(arb(val)) if kind == "eps" else None)

    def rhs(self, Y, xi, ixi, zero):
        F, cache = super().rhs(Y, xi, ixi, zero)
        D, S, C, w, E, cE, dD = cache
        no = self.no
        if self.kind == "sign":
            b = UM.pr(cE, D, self.odd, zero)
            for i, m in enumerate(self.odd):
                F[no + i] = F[no + i] - 2 * b[m] * ixi * ixi
        elif self.kind == "eps":
            for i, m in enumerate(self.odd):
                F[no + i] = F[no + i] - 2 * self.eps * S[m] * ixi
        return F, cache

    def lin(self, cache, V, xi, ixi, zero):
        F = super().lin(cache, V, xi, ixi, zero)
        D, S, C, w, E, cE, dD = cache
        odd, even, no = self.odd, self.even, self.no
        dD_, dS_, dC_, dw = self.unpack(V)
        if self.kind == "sign":
            dE = UM.pr(D, dD_, even, zero)
            p1, p2 = UM.pr(dC_, E, even, zero), UM.pr(C, dE, even, zero)
            dcE = {n: p1[n] + p2[n] for n in even}
            t2, t3 = UM.pr(dcE, D, odd, zero), UM.pr(cE, dD_, odd, zero)
            for i, m in enumerate(odd):
                F[no + i] = F[no + i] - 2 * (t2[m] + t3[m]) * ixi * ixi
        elif self.kind == "eps":
            for i, m in enumerate(odd):
                F[no + i] = F[no + i] - 2 * self.eps * dS_[m] * ixi
        return F


def box_lines(tag, X, nd):
    """exact endpoints: an arb lower/upper bound is an exact binary number; print it with enough digits and step
    outward by one unit of the last printed digit"""
    ctx.prec = 320
    vals = []
    for k in range(nd):
        vals += [X[k].real, X[k].imag]
    vals.append(X[nd].real)
    lines = []
    for i, v in enumerate(vals):
        lo, hi = arb(v.lower()), arb(v.upper())
        sl = lo.str(70, radius=False)
        sh = hi.str(70, radius=False)
        dl, dh = Decimal(sl), Decimal(sh)
        ulp_l = Decimal(1).scaleb(dl.adjusted() - 69) if dl != 0 else Decimal("1e-400")
        ulp_h = Decimal(1).scaleb(dh.adjusted() - 69) if dh != 0 else Decimal("1e-400")
        lines.append(f"BOX {tag} {i} {dl - abs(ulp_l)} {dh + abs(ulp_h)}")
    return lines


def run_b1(ray, opts):
    kind, val = parse_mut(opts.get("mutate"))
    M = 5
    with open(ray) as fh:
        d = json.load(fh)
    FPM.ZEROS[M] = list(d["x"])
    lo = COV.lohner_restore(M, d["sets"]["0"])
    if kind in ("sign", "eps"):
        lo.s = SysUMmut(M, kind, val)
    elif kind == "start":
        i, delta = val.split(":")
        i = int(i)
        k, part = divmod(i, 2)
        ctx.prec = VI.PREC
        dz = acb(arb(delta), 0) if part == 0 else acb(0, arb(delta))
        lo.c[k] = lo.c[k] + dz
    nd = lo.s.nd
    print(f"original Lohner (validated_int, lemma6_common), mutation {opts.get('mutate')}", flush=True)
    print("\n".join(box_lines("start", lo.box(), nd)))
    legs = int(opts.get("legs", "1"))
    rho = d["rho"]
    for j in range(1, legs + 1):
        leg = UM.URay(str(rho[j - 1]), str(rho[j]))
        n = lo.run(leg, 1e-2)
        X = lo.box()
        print(f"STOP ray{j} steps {n} maxrad {COV.maxrad(X, nd):.3e}", flush=True)
        print("\n".join(box_lines(f"ray{j}", X, nd)), flush=True)
    return 0


def run_a(box, opts):
    kind, val = parse_mut(opts.get("mutate"))
    M = 5
    with open(box) as fh:
        d = json.load(fh)
    y = [JM.parse(p) for p in d["yX_xi1"]]
    ctx.prec = 320
    wf = arb(opts["wf"], arb(opts.get("h", d["h"])))
    w = FPM.W0() * wf
    A_box = arb(FPM.ZEROS[M][0], arb(opts.get("h", d["h"])))
    ctx.prec = VI.PREC
    H = JM.start_box(M, y, w, A_box)
    if kind == "start":
        i, delta = val.split(":")
        k, part = divmod(int(i), 2)
        H[k] = H[k] + (acb(arb(delta), 0) if part == 0 else acb(0, arb(delta)))
    if kind in ("sign", "eps"):
        orig = UM.SysXiM
        UM.SysXiM = lambda M_: SysXiMmut(M_, kind, val)
        try:
            V, X, steps, sec = JM.backward(M, H, "0.3", "0.067")
        finally:
            UM.SysXiM = orig
    else:
        V, X, steps, sec = JM.backward(M, H, "0.3", "0.067")
    nd = 19
    print(f"original backward flow (lemma6_jump_M.backward), mutation {opts.get('mutate')}: {steps} steps, {sec:.0f} s")
    if "save-B" in opts:
        with open(opts["save-B"], "w") as fh:
            json.dump({"M": M, "xi1": "0.3", "xia": "0.067", "steps": steps, "sec": sec, "mutate": opts.get("mutate"),
                       "X": [JM.enc(z) for z in X[:nd]], "B": [[JM.enc(V[i, j]) for j in range(nd)] for i in range(nd)]},
                      fh)
        print(f"saved {opts['save-B']}")
    return 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    if args[0] == "b1":
        return run_b1(args[1], opts)
    if args[0] == "a":
        return run_a(args[1], opts)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
