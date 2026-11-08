#!/usr/bin/env python3
"""Compare capd_m5's enclosures with the stored certificate boxes (#189 review). Exact rational arithmetic.

    compare.py b1 <capd.log> <ray.json> <disc_inputs/row_JJ.json> [--expect=overlap|miss] [--json=f.json]
        lemma 6b1: every BOX of capd.log tagged rayJ against ray.json "starts"[J] (the box over X at |u| = rho_J), and
        every BOX tagged arcJJ_K against disc K of the disc inputs (Y0 and w, the box the disc lemma got)
    compare.py a <capd.log> <B.json> [--expect=...] [--json=f.json]
        lemma 6a: the C^1 run's final hull (HBOX xia) against B.json "X", and its monodromy (JAC xia) against "B":
        dy_i/dy_j = J[2i][2j] + i J[2i+1][2j] = J[2i+1][2j+1] - i J[2i][2j+1] (Cauchy-Riemann), all four real
        enclosures against Re B_ij and Im B_ij
    compare.py logs <a.log> <b.log> [--expect=...] [--json=f.json]
        two runs in the BOX/HBOX format (e.g. capd_m5 and orig_mut.py under the same mutation): every tag of a.log
        other than start that b.log also has
    compare.py BB <a_B.json> <b_B.json> [--expect=...] [--json=f.json]
        two lemma 6a B files (lemma6_jump_M --save-B format, e.g. orig_mut.py a --save-B against I103's): X and B
A comparison passes when every component overlaps. --expect=miss (a mutation) inverts the exit status: the run
passes only if some component fails to overlap (or the CAPD run failed). Also reported: the largest ratio of the
review's radius to the stored one, and the largest |mid difference| / (r_review + r_stored).
"""
import json
import sys
from fractions import Fraction

import stored as S


def hexfrac(s):
    return Fraction(float.fromhex(s))


def read_log(fn):
    boxes, jac, stops, failed = {}, {}, [], None
    with open(fn) as fh:
        for ln in fh:
            p = ln.split()
            if not p:
                continue
            if p[0] == "BOX":
                boxes.setdefault(p[1], {})[int(p[2])] = (Fraction(p[3]), Fraction(p[4]))
            elif p[0] == "HBOX":
                boxes.setdefault(p[1], {})[int(p[2])] = (hexfrac(p[3]), hexfrac(p[4]))
            elif p[0] == "JAC":
                jac.setdefault(p[1], {})[(int(p[2]), int(p[3]))] = (hexfrac(p[4]), hexfrac(p[5]))
            elif p[0] == "STOP":
                stops.append(ln.strip())
            elif p[0] == "FAILED:":
                failed = ln.strip()
    return boxes, jac, stops, failed


class Tally:
    def __init__(self):
        self.n, self.miss, self.ratio, self.sep = 0, [], 0.0, 0.0
        self.rows = []

    def add(self, name, mine, ref):
        self.n += 1
        ok = S.overlaps(mine, ref)
        if not ok:
            self.miss.append(name)
        rr = float(S.rad(ref))
        rm = float(S.rad(mine))
        if rr > 0:
            self.ratio = max(self.ratio, rm / rr)
        den = S.rad(mine) + S.rad(ref)
        if den > 0:
            self.sep = max(self.sep, float(abs(S.mid(mine) - S.mid(ref)) / den))
        return ok


def cmp_b1(log, ray, row, opts):
    boxes, _, stops, failed = read_log(log)
    with open(ray) as fh:
        d = json.load(fh)
    with open(row) as fh:
        r = json.load(fh)
    j = int(r["j"])
    out, allok, tot = [], True, Tally()
    tags = sorted(boxes, key=lambda t: (0 if t == "start" else 1 if t.startswith("ray") else 2,
                                        int(t[3:]) if t.startswith("ray") else (int(t.split("_")[1]) if "_" in t else 0)))
    for tag in tags:
        b = boxes[tag]
        if tag == "start":
            continue
        if tag.startswith("ray"):
            J = tag[3:]
            ref = d["starts"][J]
            refs = [(S.arb_str(re), S.arb_str(im)) for re, im in ref[:19]] + [(S.arb_str(ref[19][0]), None)]
            where = f"|u| = {d['rho'][int(J)]} on the start ray"
        else:
            k = int(tag.split("_")[1])
            dk = r["discs"][k]
            refs = [(S.manexp(re), S.manexp(im)) for re, im in dk["Y0"]] + [(S.manexp(dk["w"][0]), None)]
            where = f"row {j} disc {k} (t = {r['t'][k]})"
        t = Tally()
        for kk in range(19):
            t.add(f"{tag}:{kk}re", b[2 * kk], refs[kk][0])
            t.add(f"{tag}:{kk}im", b[2 * kk + 1], refs[kk][1])
        t.add(f"{tag}:w", b[38], refs[19][0])
        rr = max(float(S.rad(refs[kk][q])) for kk in range(19) for q in (0, 1))
        rm = max(float(S.rad(b[i])) for i in range(38))
        ok = not t.miss
        allok = allok and ok
        tot.n += t.n
        tot.miss += t.miss
        tot.ratio = max(tot.ratio, t.ratio)
        tot.sep = max(tot.sep, t.sep)
        row_ = dict(tag=tag, where=where, overlap=ok, n=t.n, missed=t.miss[:6], r_review=rm, r_stored=rr,
                    worst_ratio=t.ratio, worst_sep=t.sep)
        out.append(row_)
        print(f"{tag:10s} {where:38s} overlap {ok!s:5s} ({t.n - len(t.miss)}/{t.n}); max radius review {rm:.2e}, "
              f"stored {rr:.2e}; worst radius ratio {t.ratio:.2e}; worst |mid diff|/(r+r) {t.sep:.2e}")
    return finish(out, allok and failed is None and len(out) > 0, tot, failed, stops, opts)


def cmp_a(log, bfile, opts):
    boxes, jac, stops, failed = read_log(log)
    with open(bfile) as fh:
        d = json.load(fh)
    out, tot = [], Tally()
    if "xia" not in boxes or "xia" not in jac:
        return finish(out, False, tot, failed or "no final stop in the log", stops, opts)
    b, J = boxes["xia"], jac["xia"]
    X = [(S.midrad(re), S.midrad(im)) for re, im in d["X"]]
    tx = Tally()
    for k in range(19):
        tx.add(f"X{k}re", b[2 * k], X[k][0])
        tx.add(f"X{k}im", b[2 * k + 1], X[k][1])
    print(f"final state X: overlap {not tx.miss} ({tx.n - len(tx.miss)}/{tx.n}); review max radius "
          f"{max(float(S.rad(b[i])) for i in range(38)):.2e}, stored {max(float(S.rad(X[k][q])) for k in range(19) for q in (0, 1)):.2e}; "
          f"worst radius ratio {tx.ratio:.2e}; worst |mid diff|/(r+r) {tx.sep:.2e}")
    tb = Tally()
    neg = lambda a: (-a[1], -a[0])    # noqa: E731
    for i in range(19):
        for j in range(19):
            re, im = S.midrad(d["B"][i][j][0]), S.midrad(d["B"][i][j][1])
            tb.add(f"B{i},{j} re(a)", J[(2 * i, 2 * j)], re)
            tb.add(f"B{i},{j} re(b)", J[(2 * i + 1, 2 * j + 1)], re)
            tb.add(f"B{i},{j} im(a)", J[(2 * i + 1, 2 * j)], im)
            tb.add(f"B{i},{j} im(b)", neg(J[(2 * i, 2 * j + 1)]), im)
    rB = max(float(S.rad(S.midrad(d["B"][i][j][q]))) for i in range(19) for j in range(19) for q in (0, 1))
    rJ = max(float(S.rad(J[(a, c)])) for a in range(38) for c in range(38))
    print(f"monodromy B (19 x 19 complex, 4 real enclosures each): overlap {not tb.miss} ({tb.n - len(tb.miss)}/{tb.n}); "
          f"review max radius {rJ:.2e}, stored {rB:.2e}; worst radius ratio {tb.ratio:.2e}; worst |mid diff|/(r+r) "
          f"{tb.sep:.2e}")
    if tb.miss:
        print("  first misses: " + ", ".join(tb.miss[:8]))
    out = [dict(what="X", overlap=not tx.miss, n=tx.n, missed=tx.miss[:8], worst_ratio=tx.ratio, worst_sep=tx.sep),
           dict(what="B", overlap=not tb.miss, n=tb.n, missed=tb.miss[:8], worst_ratio=tb.ratio, worst_sep=tb.sep,
                r_review=rJ, r_stored=rB)]
    tot.n, tot.miss = tx.n + tb.n, tx.miss + tb.miss
    tot.ratio, tot.sep = max(tx.ratio, tb.ratio), max(tx.sep, tb.sep)
    return finish(out, not tot.miss and failed is None, tot, failed, stops, opts)


def cmp_logs(la, lb, opts):
    ba, _, sa, fa = read_log(la)
    bb, _, sb, fb = read_log(lb)
    out, allok, tot = [], True, Tally()
    common = [t for t in ba if t != "start" and t in bb]
    for tag in common:
        t = Tally()
        for i in sorted(ba[tag]):
            if i in bb[tag]:
                t.add(f"{tag}:{i}", ba[tag][i], bb[tag][i])
        ok = not t.miss
        allok = allok and ok
        tot.n += t.n
        tot.miss += t.miss
        tot.ratio, tot.sep = max(tot.ratio, t.ratio), max(tot.sep, t.sep)
        out.append(dict(tag=tag, overlap=ok, n=t.n, missed=t.miss[:6], worst_ratio=t.ratio, worst_sep=t.sep))
        print(f"{tag:10s} overlap {ok!s:5s} ({t.n - len(t.miss)}/{t.n}); worst radius ratio a/b {t.ratio:.2e}; worst "
              f"|mid diff|/(r+r) {t.sep:.2e}")
    failed = fa or fb
    return finish(out, allok and failed is None and len(common) > 0, tot, failed, sa + sb, opts)


def cmp_BB(fa, fb, opts):
    with open(fa) as fh:
        a = json.load(fh)
    with open(fb) as fh:
        b = json.load(fh)
    tot = Tally()
    tx, tb = Tally(), Tally()
    for k in range(19):
        for q in (0, 1):
            tx.add(f"X{k}.{q}", S.midrad(a["X"][k][q]), S.midrad(b["X"][k][q]))
    for i in range(19):
        for j in range(19):
            for q in (0, 1):
                tb.add(f"B{i},{j}.{q}", S.midrad(a["B"][i][j][q]), S.midrad(b["B"][i][j][q]))
    print(f"X: overlap {not tx.miss} ({tx.n - len(tx.miss)}/{tx.n}), worst |mid diff|/(r+r) {tx.sep:.2e}")
    print(f"B: overlap {not tb.miss} ({tb.n - len(tb.miss)}/{tb.n}), worst |mid diff|/(r+r) {tb.sep:.2e}")
    tot.n, tot.miss = tx.n + tb.n, tx.miss + tb.miss
    tot.ratio, tot.sep = max(tx.ratio, tb.ratio), max(tx.sep, tb.sep)
    out = [dict(what="X", overlap=not tx.miss, missed=tx.miss[:8]), dict(what="B", overlap=not tb.miss,
                                                                         missed=tb.miss[:8])]
    return finish(out, not tot.miss, tot, None, [], opts)


def finish(out, ok, tot, failed, stops, opts):
    expect = opts.get("expect", "overlap")
    if failed:
        print(f"CAPD run: {failed}")
    print(f"ALL COMPONENTS OVERLAP: {ok} ({tot.n - len(tot.miss)}/{tot.n} overlap; worst radius ratio review/stored "
          f"{tot.ratio:.2e}; worst |mid diff|/(r+r) {tot.sep:.2e})")
    passed = ok if expect == "overlap" else not ok
    print(f"VERDICT ({'clean run: every component must overlap' if expect == 'overlap' else 'mutation: some component must miss'}): "
          f"{'PASS' if passed else 'FAIL'}")
    if "json" in opts:
        with open(opts["json"], "w") as fh:
            json.dump(dict(ok=ok, expect=expect, passed=passed, n=tot.n, missed=len(tot.miss), worst_ratio=tot.ratio,
                           worst_sep=tot.sep, failed=failed, rows=out, stops=stops), fh, indent=1)
    return 0 if passed else 3


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    if args[0] == "b1":
        return cmp_b1(args[1], args[2], args[3], opts)
    if args[0] == "a":
        return cmp_a(args[1], args[2], opts)
    if args[0] == "logs":
        return cmp_logs(args[1], args[2], opts)
    if args[0] == "BB":
        return cmp_BB(args[1], args[2], opts)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
