"""#1084: value_mp's stop at the rounding floor on a planted residual, through value_mp.solve and value_v unchanged.

The planted residual is a smooth map with a known zero plus rounding-like noise, a pseudo-random function of the
bits of (A, w_f, v) at 10^-(dps - 4) of its scale: Newton reaches a floor where steps are draws of that noise, as in
inner_mp (#1084's probe). It replaces inner_mp.residual_demixed in this process only.

    python3 value_mp_floor_check.py [--mutate=off|tau-low|tau-high|keepj]      exit 0 iff every test passes

Tests (each run with the floor stop and with --floor=0, the old solve):
  (A) identical results: value_v rows at M = 3 and 5 from four starts each (every field but seconds and calls, with
      want_J's Jacobian), and a second-precision-like solve started from a Jacobian formed far away (v, R, r);
  (B) the stop acts: in the value_v runs at least one refresh is skipped, the calls drop, and no residual is
      evaluated twice at the same v (the old solve does so after a refresh at the floor);
  (C) the guard acts: the far Jacobian's solve refreshes at least once where a skip was tested and refused;
  (D) zero_mp's walk (M = 5, at most 3 iterations from (3.5001, 0.2999)) gives the same rows but for sig and valued.
Controls: off (the floor stop disabled) breaks (B); tau-low (FLOOR_TAU 0) breaks (B); tau-high (FLOOR_TAU inf, the
stop moved off the floor) breaks (A) and (C); keepj (the first precision skips refreshes too) breaks (A).
"""
import hashlib
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import value_mp as VM  # noqa: E402

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), None)
SEEN = []                                       # (dps, v) of every planted residual call


def _noise(x, k):
    """a value in [-1, 1) from the bits of x and the row k"""
    h = hashlib.sha256(repr((k, mp.mp.prec, tuple(mp.mpf(t)._mpf_ for t in x))).encode()).digest()
    return mp.mpf(int.from_bytes(h[:8], "big")) / 2 ** 63 - 1


def planted(x, M=3, r1="0.17", NTH=64, p=40, margin=12):
    x = [mp.mpf(t) for t in x]
    A, wf, v = x[0], x[1], x[2:]
    n = M - 1
    SEEN.append((mp.mp.dps, tuple(v)))
    vs = [mp.mpf(k + 1) / (3 * (k + 2)) + A / 1000 + wf / 7000 for k in range(n)]      # the planted zero
    d = [v[k] - vs[k] for k in range(n)]
    eps = mp.mpf(10) ** (-(mp.mp.dps - 4))
    S = mp.mpf(10) ** 6
    r = [S * (sum((2 if i == k else mp.mpf("0.3") / (1 + abs(i - k))) * d[k] for k in range(n)) + d[i] ** 2 / 2
              + eps * _noise(x, i)) for i in range(n)]
    a, w = A - mp.mpf("3.5"), wf - mp.mpf("0.3")              # R_1's planted zero at (3.5, 0.3)
    R = [S * (a + 2 * w + sum(d) + eps * _noise(x, n)), S * (-a + 3 * w - d[0] + eps * _noise(x, n + 1))]
    return R + r, dict(ymax=mp.mpf(1), steps=1, terr=mp.mpf(0))


def run(fn, *a, **k):
    tr = []
    _solve = VM.solve

    def solve(*aa, **kk):
        if MUT == "keepj":
            kk["keep_J"] = False
        out = _solve(*aa, trace=tr, **kk)
        return out

    VM.solve = solve
    del SEEN[:]
    try:
        out = fn(*a, **k)
    finally:
        VM.solve = _solve
    rep = len(SEEN) - len(set(SEEN))
    return out, tr, rep


def main():
    VM.IM.residual_demixed = planted
    if MUT == "tau-low":
        VM.FLOOR_TAU = "0"
    if MUT == "tau-high":
        VM.FLOOR_TAU = "inf"
    on = MUT != "off"
    ok_all = True
    skips = calls_on = calls_off = rep_on = 0
    same = 0
    total = 0
    # (A), (B): value_v at M = 3 and 5, four starts each, as argcount and zero_mp's centre call it (want_J)
    for M in (3, 5):
        n = M - 1
        for s in range(4):
            mp.mp.dps = 50
            v0 = [mp.mpf(k + 1) / (3 * (k + 2)) + mp.mpf("3.5") / 1000 + mp.mpf("0.3") / 7000
                  + mp.mpf(10) ** -(5 + s) * (-1) ** (k + s) for k in range(n)]
            v0 = [mp.nstr(t, 40) for t in v0]
            kw = dict(iters=12, want_J=True)
            a, tra, ra = run(VM.value_v, "3.5", "0.3", v0, M, floor=on, **kw)
            b, trb, rb = run(VM.value_v, "3.5", "0.3", v0, M, floor=False, **kw)
            keys = [k for k in b if k not in ("seconds", "calls")]
            diff = [k for k in keys if a.get(k) != b.get(k)]
            total += 1
            same += not diff
            skips += sum(t[0] == "skip" for t in tra)
            calls_on += sum(a["calls"])
            calls_off += sum(b["calls"])
            rep_on += ra
            print(f"M {M} start {s}: calls {a['calls']} (floor stop) vs {b['calls']} (old); skips "
                  f"{sum(t[0] == 'skip' for t in tra)}; repeated residuals {ra} vs {rb}; "
                  f"{'identical' if not diff else 'DIFFER in ' + ','.join(diff)}", flush=True)
    okA1 = same == total
    okB = skips >= 1 and calls_on < calls_off and rep_on == 0
    print(f"(A) value_v rows identical with and without the floor stop: {same} of {total}: "
          f"{'PASS' if okA1 else 'FAIL'}", flush=True)
    print(f"(B) the stop acts: {skips} skipped refreshes, calls {calls_on} against {calls_off}, {rep_on} repeated "
          f"residual calls: {'PASS' if okB else 'FAIL'}", flush=True)
    # (A), (C): a second-precision-like solve from a Jacobian formed 1e-3 away (an offset point's case)
    mp.mp.dps = 34
    far = [mp.mpf(k + 1) / (3 * (k + 2)) + mp.mpf("3.5") / 1000 + mp.mpf("0.3") / 7000 + mp.mpf("1e-3")
           for k in range(2)]
    _, _, _, J1, J3, _, _, base = VM.solve("3.5", "0.3", far, "0.17", 34, 40, iters=0, floor=False)
    near = [mp.nstr(t - mp.mpf("1e-3") + mp.mpf("1e-7"), 40) for t in far]
    res = []
    for fl in (on, False):
        out, tr, _ = run(lambda: VM.solve("3.5", "0.3", near, "0.17", 42, 50, iters=12, J=(J1, J3, base), floor=fl,
                                          keep_J=False))
        res.append((out[0], out[1], tuple(out[2]), tr))
    okA2 = res[0][:3] == res[1][:3]
    refused = sum(t[0] == "jac" and t[2] is not None for t in res[0][3])
    okC = refused >= 1
    print(f"(A) far-Jacobian solve: v, R, r {'identical' if okA2 else 'DIFFER'} with and without the floor stop: "
          f"{'PASS' if okA2 else 'FAIL'}", flush=True)
    print(f"(C) the guard refuses a skip there {refused} times (skips {sum(t[0] == 'skip' for t in res[0][3])}): "
          f"{'PASS' if okC else 'FAIL'}", flush=True)
    # (D) zero_mp's walk (centre with want_J, offsets single=True from the centre's Jacobian), with VM.FLOOR as the
    # walk reads it: the rows' located point, R, outer Jacobian, step and inner Jacobian are the same
    import zero_mp as ZM
    walks = []
    for fl in (on, False):
        VM.FLOOR = fl
        _solve = VM.solve
        if MUT == "keepj":
            VM.solve = lambda *aa, **kk: _solve(*aa, **dict(kk, keep_J=False))
        try:
            mp.mp.dps = 34
            v0 = [mp.nstr(mp.mpf(k + 1) / (3 * (k + 2)) + mp.mpf("3.5") / 1000 + mp.mpf("0.3") / 7000, 20)
                  for k in range(4)]
            walks.append(ZM.newton(3.5001, 0.2999, v0, ZM._value_v, 2, 1e-7, 1e-8, 1, M=5))
        finally:
            VM.solve = _solve
    keys = ("it", "A", "wf", "re", "im", "J", "orientation", "dA", "dwf", "F3", "J_inner")
    okD = len(walks[0]) == len(walks[1]) and all(a[k] == b[k] for a, b in zip(*walks) for k in keys)
    side = sum(a[k] != b[k] for a, b in zip(*walks) for k in ("sig", "R_over_sig", "valued"))
    print(f"(D) zero_mp walk, {len(walks[0])} rows: located point, R, outer J, step and inner J "
          f"{'identical' if okD else 'DIFFER'} with and without the floor stop ({side} sig/valued fields differ): "
          f"{'PASS' if okD else 'FAIL'}", flush=True)
    ok_all = okA1 and okB and okA2 and okC and okD
    print(f"value_mp floor check{' (mutate=' + MUT + ')' if MUT else ''}: {'PASS' if ok_all else 'FAIL'}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
