#!/usr/bin/env python3
"""P2c review (#1019): this review's B5(1) record (p2c_b5.py --sector --box) against b5_bounds' stored record
(b5_bounds_k0.3_Z0.0085.json), after the own numbers were committed (a73e6257). The protocol's reading (P2_remaining.md
6.1 (P3)): each pair is a ratio of upper bounds by different cells; a ratio outside [1/3, 3] is investigated.

    python3 p2c_b5_compare.py DIR > DIR/logs/compare.log      # DIR: the raw-data folder I1019-mac/b5
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = {"P": "P", "Phi": "F", "C": "C", "th": "v"}
CASES = {"forcing only": "forcing", "unit P": "unit_P", "unit Phi": "unit_F", "unit C": "unit_C",
         "unit theta": "unit_v", "delta_ref": "data"}


def main():
    d = sys.argv[1]
    mine = json.load(open(os.path.join(d, "p2c_b5_box_nr32_na64.json")))
    b5 = json.load(open(os.path.join(os.path.dirname(HERE), "b5_bounds_k0.3_Z0.0085.json")))[0]
    flagged = []
    print("== constants (here over the box, 3073 cells; b5_bounds 145 cells)")
    for lab, a, b in (("Perron-weighted norm", mine["perron"], b5["lam"]), ("R_Phi", mine["R_Phi"], b5["R_phi"]),
                      ("R_C", mine["R_C"], b5["R_C"]), ("T_F", mine["T_F"], b5["TF"]),
                      ("eps-part ell (largest)", mine["ell"], max(b5["ell"].values()))):
        r = float(a) / float(b)
        print(f"  {lab:24s} here {float(a):12.6g}  b5_bounds {float(b):12.6g}  ratio {r:.4f}")
        if not 1 / 3 <= r <= 3:
            flagged.append(lab)
    for key in ("Phi-1<-C+0", "Phi+1<-C+0", "Phi+1<-C+2", "Phi-1<-C-2", "Phi+1<-P+1", "Phi-1<-P-1"):
        print(f"  L {key:22s} here {float(mine['L'][key]):12.6g}")
    print(f"  b5_bounds' largest L entries: {b5['Lmax_entries']}")
    print("== super-solutions, per component: here / b5_bounds")
    for mc, bc in CASES.items():
        xm, xb = mine["x"][mc], b5["supersolution"][bc]["x"]
        row = []
        for blk in ("P", "Phi", "C", "th"):
            for h in ((-1, 1) if blk in ("P", "Phi") else (-2, 0, 2)):
                km, kb = f"{blk}{h:+d}", f"{NAME[blk]}{h:+d}"
                r = float(xm[km]) / float(xb[kb])
                row.append(f"{km} {r:.3f}")
                floor = float(xb[kb]) <= 0.0011 and float(xm[km]) <= 1e-7       # both are floors (no data, no coupling)
                if not 1 / 3 <= r <= 3 and not floor:
                    flagged.append(f"{mc} {km}")
        print(f"  {mc:13s} certified here {xm['certified']}, b5 {b5['supersolution'][bc]['ok']}: " + ", ".join(row))
    print("  (theta components of the unit P, Phi, C cases are floors on both sides: 1e-8 here, 1e-3 in b5_bounds)")
    print("flagged (ratio outside [1/3, 3], floors excluded): " + (", ".join(flagged) if flagged else "none"))
    # what the flagged ratios come from: R_Phi, and the amplification of Phi data through the P <-> Phi loop
    amp = float(mine["x"]["unit Phi"]["Phi-1"]) / float(b5["supersolution"]["unit_F"]["x"]["F-1"])
    rphi = float(mine["R_Phi"]) / float(b5["R_phi"])
    fr = float(mine["x"]["forcing only"]["Phi-1"]) / float(b5["supersolution"]["forcing"]["x"]["F-1"])
    print(f"  R_Phi ratio {rphi:.4f} x the unit-Phi amplification ratio {amp:.4f} = {rphi * amp:.4f} "
          f"(the forcing-only Phi ratio is {fr:.4f})")
    # C_E that this review's x_Phi would give (B_chain.md 3.2: C_E is proportional to x_Phi,-1 + x_Phi,+1), scaled from
    # b5_jump's C_E for the same rounded-up delta (I1019-mac/delta/b5_jump_rounded_up.log: 2.532706e33 with
    # x_Phi = (39789.7, 38781.0))
    xh = mine["x"]["(H8) rounded up"]
    s_here = float(xh["Phi-1"]) + float(xh["Phi+1"])
    ce = 2.532706e33 * s_here / (39789.7 + 38781.0)
    import math
    print(f"  C_E with this review's x_Phi for (H8) rounded up ({float(xh['Phi-1']):.6g}, {float(xh['Phi+1']):.6g}): "
          f"{ce:.6g} = e^{math.log(ce):.4f} (b5_jump: 2.532706e+33 = e^76.9146); not certified by reviewed code")
    return 0


if __name__ == "__main__":
    sys.exit(main())
