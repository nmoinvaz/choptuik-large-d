"""T43 pilot diagnosis, part 2: at D 60.023452 in the 0.3-0.6 layout, the truncated base solution z* (|F| 1.2e-19) and
a seed z* + e. e is (a) the node06 run's actual seed error, rebuilt as in exp_node06.py, and (b) the same e restricted
to each block in turn. For each: |F(z* + t e)| at t = 1, 0.5, 0.1, 0.01 (is F linear along e?), and the Newton step
dz at z* + e against -e (is the step right?). The same test at 0.7 with base's own layout for comparison.
    python exp_node06b.py HP_DIR PILOT_DIR"""
import ctypes, glob, os, sys, types
import numpy as np

H, P = sys.argv[1], sys.argv[2]
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)    # below normal
sys.path.insert(0, H)
os.chdir(H)
import scan_hp  # noqa: E402
from hpkernel.driver import dd_add, dd_norm, load, make_solver, renode  # noqa: E402
from pinning import predict_dd  # noqa: E402

M = 32
N6, N7 = [0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6, 0.7]
a6 = types.SimpleNamespace(kt=36, kc=220, ks=160, threads=6, kfreeze=13, pin=None, tol_step=1e-32, hmax=0.05,
                           corrector="dnp", tol=5.45e-19, w=16, verbose=False, junk_only=False, delta_tol=1e-14,
                           polish_from=1e-12, polish_its=4, lm=0)
a7 = types.SimpleNamespace(**dict(vars(a6), ks=100, tol=8e-23))


def build(nodes, a):
    """the run's predictor history in this layout: the last 3 seeds re-noded and re-converged, then the start"""
    hist = []
    for f in sorted(glob.glob(os.path.join(H, "data", "seeds", "step_test", "up13_D*.npz")))[-3:]:
        hh, hl, hm = load(f)
        if list(hm["nodes"]) != nodes:
            s0 = make_solver(hm["D"], M, nodes=list(hm["nodes"]), kt=36, kc=220, ks=a.ks, threads=6)
            hh, hl = renode(s0, hh, hl, list(hm["nodes"]), nodes, M)
        hh, hl = scan_hp.freeze_p1(hh, hl, a, M)
        _, hh, hl, _, _, _ = scan_hp.solve_at(hm["D"], hh, hl, a, nodes, M, 25)
        hist.append((hm["D"], hh, hl))
    last = 0.6 if nodes == N6 else 0.7
    zh, zl, m = load(os.path.join(P, f"start_D60.0020_x{last}_kf13.npz"))
    _, zh, zl, _, _, _ = scan_hp.solve_at(m["D"], zh, zl, a, nodes, M, 25)
    hist.append((m["D"], zh, zl))
    return hist


bh, bl, bm = load(os.path.join(P, "base_D60.00_D60.0235.npz"))
D = bm["D"]
for nodes, a in ((N7, a7), (N6, a6)):
    print(f"\n==== layout {nodes}, ks {a.ks}", flush=True)
    if nodes == N7:
        sh, sl = bh, bl
    else:
        s7 = make_solver(D, M, nodes=N7, kt=36, kc=220, ks=100, threads=6)
        sh, sl = renode(s7, bh, bl, N7, N6, M)
    s = make_solver(D, M, nodes=nodes, kt=36, kc=220, ks=a.ks, threads=6, kfreeze=13)
    s.set_segs(s.plan(sh, sl, 1e-32, 0.05))
    print(f"|F(z*)| {dd_norm(*s.residual(sh, sl)):.1e}", flush=True)
    zp, zpl = predict_dd(build(nodes, a), D, 4)
    e = (zp - sh) + (zpl - sl)
    K = len(nodes)
    blocks = {"all": slice(None), "centre": slice(0, 2 * M)}
    for i, x in enumerate(nodes):
        blocks[f"node {x}"] = slice(2 * M + 4 * M * i, 2 * M + 4 * M * (i + 1))
    blocks["horizon+Delta"] = slice(2 * M + 4 * M * K, None)
    for name, sl_ in blocks.items():
        eb = np.zeros_like(e)
        eb[sl_] = e[sl_]
        if not np.any(eb):
            continue
        Fs = []
        for t in (1.0, 0.5, 0.1, 0.01):
            th, tl = dd_add(sh, sl, t * eb, np.zeros_like(eb))
            Fs.append(dd_norm(*s.residual(th, tl)))
        th, tl = dd_add(sh, sl, eb, np.zeros_like(eb))
        dh, dl, Fh, Fl, pmin, pmax = s.newton(th, tl)
        dz = np.asarray(dh) + np.asarray(dl)
        nh, nl = dd_add(th, tl, dh, dl)
        print(f"{name:14s} |e| {np.abs(eb).max():.1e}: |F(z*+t e)| t=1,.5,.1,.01: "
              + " ".join(f"{f:.1e}" for f in Fs)
              + f" | Newton: |dz+e|/|e| {np.abs(dz + eb).max() / np.abs(eb).max():.1e}, |F| after full step "
              f"{dd_norm(*s.residual(nh, nl)):.1e}, piv {pmin:.0e}..{pmax:.0e}", flush=True)
print("done", flush=True)
