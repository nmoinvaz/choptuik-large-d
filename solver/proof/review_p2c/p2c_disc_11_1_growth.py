#!/usr/bin/env python3
"""P2c review (#992): how disc (11, 1)'s bound grows with its input box, from the records of
p2c_disc.py --discs=11:1 --infl=I --tag=diag_11_1_inflI (I = 0..3), against the Lipschitz growth over the disc.

    python3 p2c_disc_11_1_growth.py DIR > DIR/p2c_disc_11_1_growth.log     # DIR: the records' folder (I992-mac)
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = sys.argv[1]
R = {i: json.loads(open(os.path.join(DIR, f"p2c_disc_diag_11_1_infl{i}.jsonl")).readline()) for i in range(4)}
row = json.load(open(os.path.join(os.path.dirname(HERE), "lemma6b1_runs", "row_11.json")))
box, n = row["discs"][1]["box_rad"], row["n"]
print(f"disc (11, 1): n {n}, r {row['r']}, stored box_rad {box:.4g}")
for i in range(4):
    print(f"  input box {i} x box_rad + 1e-15: Lambda {R[i]['Lam']:.4f}, max M {R[i]['Mmax']:.4g}")
print(f"  growth of max M per unit of input box radius, 0 -> 1 x: {(R[1]['Mmax'] - R[0]['Mmax']) / box:.2e}")
print(f"  Lipschitz growth over the disc, exp((n + 2) Lambda) at the thin input: {math.exp((n + 2) * R[0]['Lam']):.2e}")
