#!/usr/bin/env python3
"""Regenerate the inputs of the lemma 6b1(5) disc certificates (#39 review).

The stored rows (row_JJ.json) keep each disc's verdict and Lam, not the Lohner box the disc lemma was given. This
driver reruns the original row computation, lemma6b1_cover_M.row_run unchanged, on the stored C^1 ray (ray.json, its
Lohner states at the row radii), and records for every disc the exact arguments the original passes to
lemma6b1_disc_M.certify: the centre ball c, the state box Y0, the w ball, r and n. Balls are written exactly, as the
(mantissa, exponent) pairs of their midpoints and radii. The original certify runs too, so its verdict on these inputs
is recorded beside the stored one. This is the original machinery (validated_int.Lohner, lemma6_common, SysUM), used as
the source of the inputs. The independent part of the review is check_6b1.py.

    python3 dump_inputs.py --ray=<dir with ray.json> --out=<dir> --rows=17,16,...
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.dirname(HERE)
sys.path.insert(0, PROOF)
import lemma6b1_cover_M as CM  # noqa: E402
import fixed_point_M as FPM  # noqa: E402
from flint import arb  # noqa: E402

M = 5
H = "1e-18"


def me(x):
    """an arb as exact [[mid man, mid exp], [rad man, rad exp]]"""
    m = x.mid().man_exp()
    r = x.rad().man_exp()
    return [[int(m[0]), int(m[1])], [int(r[0]), int(r[1])]]


def mec(z):
    return [me(z.real), me(z.imag)]


class Proxy:
    """stands in for lemma6b1_disc_M inside lemma6b1_cover_M: records each certify call, then runs the original"""

    def __init__(self, real, sink):
        self.real, self.sink = real, sink

    def certify(self, Mh, c, Y0, w, r, n, mutate=None):
        t0 = time.time()
        res = self.real.certify(Mh, c, Y0, w, r, n, mutate=mutate)
        self.sink.append({"c": mec(c), "Y0": [mec(y) for y in Y0], "w": mec(w), "r": str(r), "n": int(n),
                          "orig": {"ok": bool(res["ok"]), "Lam": float(res["Lam"].upper()),
                                   "F0": float(res["F0"].upper()),
                                   "sup": max(float(m.upper()) for m in res["Msup"]) if "Msup" in res else None,
                                   "it": res.get("it"), "sec": time.time() - t0}})
        return res


def main():
    opts = {a.split("=", 1)[0][2:]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--")}
    raydir, out = opts["ray"], opts["out"]
    rows = [int(v) for v in opts["rows"].split(",")]
    with open(os.path.join(raydir, "ray.json")) as fh:
        x = json.load(fh)["x"]
    FPM.ZEROS[M] = list(x)                      # the --x of the stored runs (row_run asserts it)
    CM.runs_dir = lambda _M: raydir             # row_run reads ray.json and writes its row_JJ.json here
    os.makedirs(out, exist_ok=True)
    for j in rows:
        sink = []
        CM.LDM = Proxy(CM.LDM if not isinstance(CM.LDM, Proxy) else CM.LDM.real, sink)
        t0 = time.time()
        res = CM.row_run(M, j, H)
        rec = {"M": M, "j": j, "h": H, "x": x, "rho": res["rho"], "r": res["r"], "n": res["n"],
               "t": [d["t"] for d in res["discs"]], "box_rad": [d["box_rad"] for d in res["discs"]],
               "steps": res["steps"], "sec": time.time() - t0, "discs": sink}
        with open(os.path.join(out, f"row_{j:02d}.json"), "w") as fh:
            json.dump(rec, fh)
        print(f"row {j}: {len(sink)} discs dumped, all ok {res['ok']}, {res['steps']} steps, {time.time() - t0:.0f} s",
              flush=True)


if __name__ == "__main__":
    main()
