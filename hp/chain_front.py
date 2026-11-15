#!/usr/bin/env python3
"""The chain's front for the figures of the large-D chain (#659): the chain_fit row of data/delta_all.csv they are drawn
at, and its state (relative to the raw-data repository).

rule "top" (the default): the highest chain_fit D, the chain's reach. rule "fits": the highest chain_fit row whose err
is the spread over the held fit's degree, track C's comparison frontier, below the rows whose err is the weak pivot's
(#806; past it #603's chain leaves the 1/n series by a term that grows toward its fold near D 259.9, #805).

    cd hp && python3 chain_front.py [top|fits]       prints D and the state
"""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# the rule the figures are drawn at (the scripts' --front default) and that check_numbers.py's figure_front checks their
# captions against: "fits", the comparison frontier, the last D where the chain agrees with track C's series within its
# err; past it the weak pivot's term grows, and near D 259.9 the discrete system's fold distorts Delta and the harmonics
# (#659). "top" draws them at the chain's reach instead.
FIGURES = "fits"


def front(rule="top"):
    """(D, state path) of the front's chain_fit row"""
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_all.csv")))
            if r["kind"] == "chain_fit" and r["state"]]
    if rule == "fits":
        rows = [r for r in rows if r["err_kind"].startswith("fits")]
    elif rule != "top":
        raise SystemExit(f"chain_front: rule {rule!r} is top or fits")
    r = max(rows, key=lambda r: float(r["D"]))
    return float(r["D"]), r["state"]


def x_hi(D):
    """the right end of a D axis that reaches the front: the next multiple of 10"""
    return 10 * math.ceil(D / 10 + 1e-9)


if __name__ == "__main__":
    D, st = front(sys.argv[1] if len(sys.argv) > 1 else "top")
    print(f"{D!r} {st}")
