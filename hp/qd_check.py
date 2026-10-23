#!/usr/bin/env python3
"""Independent checks of the quad-double arithmetic in hp/src/qd.rs and hp/src/qv.rs (notes/QUAD_DOUBLE.md).

    cd hp && python3 qd_check.py mpmath [--n=2000] [--seed=1] [--dump=DIR] [--mutate=ID]
    cd hp && python3 qd_check.py backends [--n=4000] [--only=avx512,avx2,portable] [--fresh] [--out=DIR]

mpmath: the Rust test qd_dump (built for AVX-512, as the backends check's avx512 build, into hp/target/qd_native) writes inputs and scalar
  results of every operation and input category, of the dot accumulator, and of decimal parsing and printing; this
  script recomputes every reference with mpmath (working precision 3000 bits, so that every sum of parts is exact
  even under deep cancellation; the 120 digits asked for are a lower bound) and reports, per operation and
  category, the max and the 99.9th percentile of the error in units of 2^-209, with the pass thresholds and the
  near-underflow allowance of the Rust tests. It shares no code with the Rust reference (src/exact.rs). Parsing is
  checked against the exact decimal value (greedy round-to-nearest: |x - v| <= ulp(x3)/2), printing against correct
  rounding of the exact value. --dump=DIR reads an existing dump instead of running cargo. Exit 1 on any failure.
  --mutate=ID is the control: the dump is written by a build with the qd_mutate feature and mutation ID active
  (qd::mutate in hp/src/qd.rs, e.g. 5: sloppy mul drops the error of a0 b2; 14: parse truncates its last part), so
  this check must then fail (exit 1).
backends: runs `cargo test --release --lib qd_` (the accuracy, agreement and decimal tests) with the AVX-512, the
  AVX2 and the portable v8 backend (RUSTFLAGS "-C target-cpu=x86-64-v4", "-C target-feature=+avx2,+fma",
  none: explicit features, so that a build does not depend on the host it was made on; build
  directories hp/target/qd_<isa>); every build must pass, and the digests of all scalar results must agree (each
  build also checks that its 8-lane results equal the scalar ones bit for bit). Exit 1 otherwise. Each backend's
  result is written to DIR/<isa>.txt (default data/quad_double/backends) as soon as it is known, and a rerun skips
  the backends recorded there as passed with the same n and RUSTFLAGS (--fresh reruns them; --only=... runs a subset, and the
  digests are then compared over every recorded backend). The portable build takes 15-20 minutes on two cores.
"""
import os
import re
import struct
import subprocess
import sys
import tempfile

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
mp.mp.prec = 3000
U = mp.mpf(2) ** -209
TINY = 16 * mp.mpf(2) ** -1074
BUILDS = {"avx512": "-C target-cpu=x86-64-v4", "avx2": "-C target-feature=+avx2,+fma", "portable": ""}

KIND = {"sloppy add": "add", "accurate add": "add", "qd + f64": "add", "qd + dd": "add", "sloppy mul": "mul",
        "accurate mul": "mul", "qd x f64": "mul", "qd x dd": "mul", "sqr": "sqr", "sloppy div": "div",
        "accurate div": "div", "sqrt": "sqrt"}
CRAY = {"sloppy add", "qd + f64", "qd + dd"}


def opts():
    return dict((a[2:].split("=", 1) + [True])[:2] for a in sys.argv[2:] if a.startswith("--"))


def cargo(args, rustflags, target, env=None):
    e = dict(os.environ, RUSTFLAGS=rustflags, CARGO_TARGET_DIR=os.path.join(HERE, "target", target))
    e.update(env or {})
    r = subprocess.run(["nice", "-n", "10", "cargo", "test", "--release", "--lib"] + args, cwd=HERE, env=e,
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def val(parts):
    s = mp.mpf(0)
    for p in parts:
        s += mp.mpf(p)  # exact at 3000 bits for parts spanning less than 3000 bits
    return s


def ulp(x):
    if x == 0:
        return mp.mpf(2) ** -1074
    m, e = mp.frexp(abs(mp.mpf(x)))
    return mp.mpf(2) ** max(e - 53, -1074)


def pct(v, q):
    v = sorted(v)
    return v[int((len(v) - 1) * q)] if v else 0.0


def fmt(x):
    x = float(x)
    if x == 0:
        return "0"
    if x == float("inf"):
        return "inf"
    return f"{x:.3f}" if 0.01 <= x < 1000 else f"{x:.2e}"


def check_ops(d):
    rows, bad = [], 0
    raw = open(os.path.join(d, "ops.bin"), "rb").read()
    off = 0
    blocks = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(d, "ops.txt"))]
    print(f"{'operation':<14} {'inputs':<10} {'n':>6} {'max':>10} {'99.9%':>10} {'max rel':>10} {'thr':>6} {'fails':>6}")
    for b in blocks:
        if b[0] == "dot":
            off = check_dot(raw, off, int(b[1]), int(b[2]), float(b[3]))
            continue
        op, cat, n, thr = b[0], b[1], int(b[2]), mp.mpf(b[3])
        ms, rels, fails = [], [], 0
        for _ in range(n):
            v = struct.unpack_from("<12d", raw, off)
            off += 96
            a, bb, r = v[0:4], v[4:8], v[8:12]
            if op in ("qd + f64", "qd x f64"):
                bb = bb[:1]
            elif op in ("qd + dd", "qd x dd"):
                bb = bb[:2]
            A, B, R = val(a), val(bb), val(r)
            under = cat == "underflow"
            k = KIND[op]
            allow = TINY if under else mp.mpf(0)
            if k == "add":
                X = A + B
                err = abs(R - X)
                scale = abs(A) + abs(B) if op in CRAY else abs(X)
                rel = err / abs(X) if X != 0 else (mp.mpf(0) if err == 0 else mp.inf)
            elif k in ("mul", "sqr"):
                X = A * A if k == "sqr" else A * B
                err, scale = abs(R - X), abs(X)
                rel = err / scale if scale else mp.mpf(0)
            elif k == "div":
                # mpmath quotient; err in the residual form |r b - a| of the Rust test
                Q = A / B
                rel = abs(R - Q) / abs(Q)
                err, scale = abs(R * B - A), abs(A)
                if under:
                    allow = TINY + TINY * abs(B)
            else:
                S = mp.sqrt(A)
                rel = abs(R - S) / S
                err, scale = abs(R * R - A), 2 * abs(A)
                if under:
                    allow = TINY + 2 * TINY * abs(R)
            if not all(map(lambda t: t == t and abs(t) != float("inf"), r)) or err > thr * U * scale + allow:
                fails += 1
            ms.append(err / scale / U if scale else mp.mpf(0))
            rels.append(rel / U)
        bad += fails
        rows.append((op, cat, n, max(ms), pct(ms, 0.999), max(rels), thr, fails))
        print(f"{op:<14} {cat:<10} {n:>6} {fmt(max(ms)):>10} {fmt(pct(ms, 0.999)):>10} {fmt(max(rels)):>10} "
              f"{fmt(thr):>6} {fails:>6}")
    return rows, bad


def check_dot(raw, off, ln, nsamp, thr):
    ms, fails = [], 0
    for _ in range(nsamp):
        terms = []
        for i in range(ln):
            for lane in range(8):
                v = struct.unpack_from("<8d", raw, off)
                off += 64
                terms.append((i, lane, val(v[0:4]) * val(v[4:8])))
        for lane in range(8):
            v = struct.unpack_from("<4d", raw, off)
            off += 32
            X = sum(t[2] for t in terms if t[1] == lane)
            S = sum(abs(t[2]) for t in terms if t[1] == lane)
            err = abs(val(v) - X)
            ms.append(err / S / U)
            fails += err > thr * U * S
    print(f"{'dot n=' + str(ln):<14} {'random':<10} {len(ms):>6} {fmt(max(ms)):>10} {fmt(pct(ms, 0.999)):>10} "
          f"{'':>10} {fmt(thr):>6} {fails:>6}")
    global DOT_FAILS
    DOT_FAILS = fails
    return off


DOT_FAILS = 0


def check_io(d):
    bad = 0
    pmax, pms, gfail, sfail, sn, mpdiff = mp.mpf(0), [], 0, 0, 0, 0
    for line in open(os.path.join(d, "io.txt")):
        f = line.split()
        if f[0] == "P":
            s, parts = f[1], [struct.unpack("<d", bytes.fromhex(h)[::-1])[0] for h in f[2:6]]
            v = mp.mpf(s)  # 3000 bits: exact for these decimals' needs (error far below 2^-212 relative)
            x = val(parts)
            e = abs(x - v)
            pms.append(e / abs(v) / U)
            last = max(i for i in range(4) if parts[i] != 0)
            bound = ulp(parts[last]) / 2 if last == 3 else mp.mpf(0)
            # v is not exact at 3000 bits when it has more than ~900 digits; these have at most 90
            if e > bound * (1 + mp.mpf(2) ** -100) + abs(v) * mp.mpf(2) ** -2900:
                gfail += 1
        else:
            parts = [struct.unpack("<d", bytes.fromhex(h)[::-1])[0] for h in f[1:5]]
            digits, s = int(f[5]), f[6]
            sn += 1
            x = val(parts)
            m = re.fullmatch(r"(-?)(\d)(?:\.(\d+))?e([+-]\d+)", s)
            ok = m is not None
            if ok:
                ds = m.group(2) + (m.group(3) or "")
                e_last = int(m.group(4)) - (len(ds) - 1)
                D = (-1 if m.group(1) else 1) * mp.mpf(int(ds)) * mp.mpf(10) ** e_last
                half = 5 * mp.mpf(10) ** (e_last - 1)
                dist = abs(x - D)
                ok = len(ds) == digits and ds[0] != "0" and (dist < half or (dist == half and int(ds[-1]) % 2 == 0))
                ok = ok and (m.group(1) == "-") == (x < 0)
                # mpmath's own rounding to the same digits (from the 3000-bit value)
                ref = mp.nstr(x, digits, min_fixed=1, max_fixed=0, strip_zeros=False)
                mm = re.fullmatch(r"(-?)(\d)(?:\.(\d+))?e([+-]\d+)", ref)
                if mm and (mm.group(2) + (mm.group(3) or ""), int(mm.group(4))) != (ds, int(m.group(4))):
                    mpdiff += 1
            sfail += not ok
    print(f"parse: n {len(pms)}, relative error max {fmt(max(pms))} u, 99.9% {fmt(pct(pms, 0.999))} u; greedy "
          f"round-to-nearest violated: {gfail}")
    print(f"print: n {sn}, not correctly rounded: {sfail}; differing from mpmath's nstr: {mpdiff}")
    bad += gfail + sfail + sum(1 for p in pms if p > 0.25)
    return bad


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("mpmath", "backends"):
        print(__doc__)
        sys.exit(2)
    o = opts()
    if sys.argv[1] == "backends":
        n = str(o.get("n", 4000))
        out = os.path.join(HERE, str(o.get("out", os.path.join("data", "quad_double", "backends"))))
        os.makedirs(out, exist_ok=True)
        only = str(o.get("only", ",".join(BUILDS))).split(",")
        for isa in only:
            path = os.path.join(out, f"{isa}.txt")
            if os.path.exists(path) and not o.get("fresh"):
                rec = dict(ln.rstrip("\n").split("\t", 1) for ln in open(path))
                if rec.get("n") == n and rec.get("exit") == "0" and rec.get("rustflags") == BUILDS[isa]:
                    print(f"{isa:<9} passed before, recorded in {path}, skipped")
                    continue
            rc, outp = cargo(["qd_", "--", "--test-threads=1", "--nocapture"], BUILDS[isa], f"qd_{isa}", {"QD_N": n})
            dg = re.search(r"qd digest (\w+) \(n \d+, seed \d+, isa (\w+)\)", outp)
            res = re.findall(r"test result: .*", outp)
            rec = {"isa": isa, "n": n, "rustflags": BUILDS[isa], "exit": str(rc), "isa_reported": dg.group(2) if dg else "?",
                   "digest": dg.group(1) if dg else "?", "result": res[-1] if res else "no result"}
            with open(path, "w") as f:
                f.write("".join(f"{k}\t{v}\n" for k, v in rec.items()))
        digests, bad = {}, 0
        for isa in BUILDS:
            path = os.path.join(out, f"{isa}.txt")
            if not os.path.exists(path):
                print(f"{isa:<9} not run yet")
                bad += 1
                continue
            rec = dict(ln.rstrip("\n").split("\t", 1) for ln in open(path))
            print(f"{isa:<9} n {rec['n']}  exit {rec['exit']}  isa reported {rec['isa_reported']}  digest {rec['digest']}  {rec['result']}")
            digests[isa] = rec["digest"] if rec["n"] == n else None
            bad += rec["exit"] != "0" or rec["digest"] == "?" or rec["isa_reported"] != isa
        same = len(digests) == len(BUILDS) and len(set(digests.values())) == 1 and None not in digests.values()
        print("digests agree across backends" if same else "DIGESTS DIFFER OR MISSING")
        sys.exit(0 if same and not bad else 1)
    n, seed = str(o.get("n", 2000)), str(o.get("seed", 1))
    d = o.get("dump")
    if not d:
        d = tempfile.mkdtemp(prefix="qd_dump_")
        mut = str(o.get("mutate", 0))
        feat = ["--features", "qd_mutate"] if mut != "0" else []
        rc, out = cargo(feat + ["qd_dump", "--", "--ignored", "--nocapture"], BUILDS["avx512"],
                        "qd_native_mutate" if feat else "qd_native", {"QD_DUMP": d, "QD_N": n, "QD_SEED": seed,
                                                                      "QD_MUTATE": mut})
        if rc != 0:
            print(out)
            sys.exit(1)
    print(f"quad-double results against mpmath (3000-bit working precision); units of 2^-209; dump {d}")
    rows, bad = check_ops(d)
    bad += DOT_FAILS + check_io(d)
    print("all within the thresholds" if not bad else f"{bad} FAILURES")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
