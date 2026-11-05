#!/usr/bin/env python3
"""P2b review: comparison of the second implementation (p2b_lc.py) with the Lemma LC certificate, written AFTER
p2b_lc.py's own numbers were committed (commit d89f5ca). It reads the certificate's outputs only (b8m_lc_K4.log: the
class constants h and the outputs P(h); b8m_lc_lip.json: the Lipschitz matrix and the derivative constants h'), and
evaluates p2b_lc.py's majorant AT THE CERTIFICATE'S CONSTANTS:

  1. closure: P_p2b(h_cert)/h_cert per component and band, against the certificate's ratios (the independent check
     of the certified closure: every ratio must be < 1);
  2. d0, h_{S,F} (the certificate's P_S on F), h_Sp;
  3. the Lipschitz matrix (forward derivatives here, secants there) entry by entry, and both Perron roots;
  4. diagnostics: the parts of P_S on F (local term, integrands, gains) and of the beta outputs.

    python3 p2b_compare_lc.py [--mode=box --crude_d --notesH] [--log=...]
the class constant h_cc of the F2 component (not in the certificate) is taken from p2b_lc's own closure.
"""
import json
import os
import sys

from mpmath import iv, mp

import p2b_alg as Al
import p2b_lc as L
import p2b_series as Ser
from p2b_alg import mpup, mplo, bto, Bd

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.dirname(HERE)
OPTS = L.OPTS
CMAP = {"V": "V", "S": "S", "B": "b", "C": "c"}


def cert():
    """the certificate's record: by default the certified run that was reviewed (kept as *_preP2b after the F3 rerun)"""
    rec = None
    for line in open(os.path.join(PROOF, OPTS.get("certlog", "b8m_lc_K4_preP2b.log"))):
        r = json.loads(line)
        if r.get("closes") and not r.get("mutate"):
            rec = r
    lip = json.load(open(os.path.join(PROOF, OPTS.get("certlip", "b8m_lc_lip_preP2b.json"))))
    return rec, lip


def main():
    mode = OPTS.get("mode", "box")
    rec, lip = cert()
    st = Al.Setting("73.89")
    T = Ser.tables()
    known = L.Known(st, T)
    cells = L.make_cells(st, mode)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    own = json.load(open(os.path.join(HERE, "p2b_lc_cells.json")))
    H = {}
    for c in L.COMP:
        for b in L.BANDS:
            H[("h", c, b)] = iv.mpf(repr(rec["hyp"][f"{CMAP[c]}.{b}"]))
            H[("d", c, b)] = iv.mpf(repr(lip["hprime"][f"{CMAP[c]}.{b}"]))
    H[("d", "S", "L")] = iv.mpf(repr(rec["hyp"]["Sp"]))
    # F2: the class constant h_cc (not in the certificate): the smallest closed value at the certified constants,
    # h_cc <- 1.01 P_cc(h_cert, h_cc) from p2b's own value
    H[("cc",)] = iv.mpf(own["h"]["cc,L"])
    for _ in range(6):
        o = L.evaluate(st, cells, known, H)
        H[("cc",)] = Al.up(iv.mpf("1.01") * o[("cc",)].v)
    cfg = f"mode {mode}" + (", crude_d" if L.CRUDE_D else "") + (", notesH" if L.NOTESH else "")
    log(f"p2b_lc's majorant ({cfg}) at the certificate's constants ({OPTS.get('certlog', 'b8m_lc_K4_preP2b.log')} hyp, "
        f"{OPTS.get('certlip', 'b8m_lc_lip_preP2b.json')} hprime)")
    out = L.evaluate(st, cells, known, H, detail=True)
    log("  component.band: cert P/h | p2b P/h   (cert P | p2b P)")
    worst = 0
    allbelow = True
    rows = []
    for c in L.COMP:
        for b in L.BANDS:
            k = ("h", c, b)
            hc = float(mpup(H[k]))
            pc = rec["P_hyp"][f"{CMAP[c]}.{b}"]
            pm = float(mpup(out[k].v))
            allbelow &= pm < hc
            worst = max(worst, pm / hc)
            rows.append((f"{CMAP[c]}.{b}", pc / hc, pm / hc, pc, pm))
    pc, pm, hc = rec["P_hyp"]["Sp"], float(mpup(out[("d", "S", "L")].v)), float(mpup(H[("d", "S", "L")]))
    rows.append(("Sp", pc / hc, pm / hc, pc, pm))
    allbelow &= pm < hc
    for nm, rc, rm, a, b in rows:
        log(f"    {nm:5s}: {rc:.4f} | {rm:.4f}   ({a:.5g} | {b:.5g})")
    log("  derivative outputs at the certificate's h' (p2b's D/h'; the certificate's D0 equals h' by construction):")
    log(f"    (F2 component: h_cc = {float(mpup(H[('cc',)])):.5g}, P_cc/h_cc = "
        f"{float(mpup(out[('cc',)].v)) / float(mpup(H[('cc',)])):.4f})")
    drow = []
    for c in L.COMP:
        for b in L.BANDS:
            k = ("d", c, b)
            r = float(mpup(out[k].v)) / float(mpup(H[k]))
            drow.append(f"{CMAP[c]}.{b} {r:.5f}")
            if (c, b) != ("S", "L"):
                allbelow &= r <= 1 + 1e-12          # a derivative bound may be preserved with equality (V)
    log("    " + ", ".join(drow))
    log(f"  d0: cert {rec['diag']['d0']:.6g} | p2b {float(mpup(out['d0'].v)):.6g};  rho4 part: cert "
        f"{rec['diag']['r4']:.6g} | p2b {float(mpup(known.d0_rho4)):.6g}")
    det = out["_detail"]
    res, gains = det["res"], det["gains"]
    g0 = [gains[id(c)] for c in cells if c.kind == "sector" and c.band == "N0"]
    log("  N0 arc gains (per cell): " + ", ".join(f"{float(mpup(g['garc'])):.4g}" for g in g0) +
        f" (cert gamma_N0 {rec['kernel']['gN0']:.5g})")
    gn = [gains[id(c)] for c in det["near"]]
    log("  arc gains at radius 1 +- delta/2 (the data u(x') at Ac): " + ", ".join(f"{float(mpup(g['garc'])):.3g}"
                                                                         for g in gn) +
        f" (cert gamma_Na {rec['kernel']['gNa']:.4g}); |u(x')|/w <= {float(mpup(det['uint'].v)):.4g} + |G2|/w")
    for c in cells:
        if c.kind == "sector" and c.band == "F" and c.rhi is None:
            r = res[id(c)]
            g = gains[id(c)]
            log(f"  F: loc |G1|/w = {float(mpup(r['G1w'].v)):.5g} (cert loc_F {rec['diag']['loc_F']:.5g}); integrand "
                f"(|xG1'| + |xW2|)/w = {float(mpup(r['Iw'].v)):.5g} (cert int_F {rec['diag']['int_F']:.5g}); gains "
                f"arc {float(mpup(g['garc'])):.4g}, ray {float(mpup(g['gtop'])):.4g} (cert gamma_F "
                f"{rec['kernel']['gF']:.5g})")
        if c.kind == "sector" and c.rhi is None:
            kn = det["kres"][id(c)]
            pr = res[id(c)]["parts"]
            pp = {nm: bto(Al.bmul(kn["AKinv"][0], pr[nm][0]), c, L.W).v for nm in ("T1", "T2", "T3")}
            pp["rho2"] = bto(Al.bmul(kn["AKinv"][0], kn["rho2"][0]), c, L.W).v
            log("  F: p2b's G1 pieces, sup |piece|/w: " + ", ".join(f"{a} {float(mpup(b)):.5g}" for a, b in pp.items())
                + "  (certificate, p2b_cert_probe.py: T1 1.8543e6, T2 3.3622e6, T3 1498, rho2 2.6526e5)")
        if c.kind == "ray" and c.band == "F":
            log(f"  F ray: integrand {float(mpup(res[id(c)]['Iw'].v)):.5g}")
    # Lipschitz at the certificate's constants
    M, outad = L.lipschitz(st, cells, known, H)
    idx = [L.idxname(i) for i in range(40)]

    def cname(i):
        nm = L.idxname(i)
        kind = "val" if nm[0] == "h" else "der"
        comp, band = nm[1:].split(",")
        return f"{kind}/{CMAP[comp]}/{band}"
    Mc = lip["M"]
    diffs = []
    for i in range(40):
        for j in range(40):
            a = Mc[f"{cname(i)}<-{cname(j)}"]
            b = float(M[i][j])
            if a > 1e-3 or b > 1e-3:
                diffs.append((cname(i), cname(j), a, b))
    ev_c, th_c, _ = L.perron([[mp.mpf(Mc[f"{cname(i)}<-{cname(j)}"]) for j in range(40)] for i in range(40)])
    ev_m, th_m, _ = L.perron([[M[i][j] for j in range(40)] for i in range(40)])
    log(f"  Lipschitz matrices at the certificate's constants: Perron root cert {ev_c:.5f} (theta {float(th_c):.5f}) |"
        f" p2b {ev_m:.5f} (theta {float(th_m):.5f})")
    log("  entries above 1e-3 in either (row <- column: cert | p2b):")
    for r, c, a, b in sorted(diffs, key=lambda t: -max(t[2], t[3])):
        log(f"    {r:10s} <- {c:10s}: {a:.4g} | {b:.4g}" + ("   *" if abs(a - b) > 0.1 * max(a, b) else ""))
    log(f"  closure of the certified class under p2b's majorant: "
        f"{'every value ratio < 1, every derivative ratio <= 1' if allbelow else 'FAILS'}; "
        f"largest value ratio {worst:.4f}")
    fn = OPTS.get("log")
    if fn:
        with open(os.path.join(HERE, fn), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return 0 if allbelow and th_m < 1 else 1


if __name__ == "__main__":
    sys.exit(main())
