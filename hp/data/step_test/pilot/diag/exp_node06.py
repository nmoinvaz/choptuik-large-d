"""T43 pilot diagnosis: why node06 fails every step from D 60.0020 while base converges.
    python exp_node06.py HP_DIR PILOT_DIR
Rebuilds the node06 run's predictor history as scan_hp.py does (seeds re-noded to 0.3-0.6, kfreeze 13, re-converged
with dnp), predicts the run's first and last failed targets, and runs plain and deflated Newton from each seed
verbosely. Compares the seed with base's converged state at 60.023452 truncated to the 0.6 layout."""
import ctypes, glob, os, sys, time, types
import numpy as np

H, P = sys.argv[1], sys.argv[2]
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)    # below normal
sys.path.insert(0, H)
os.chdir(H)
import scan_hp  # noqa: E402
from hpkernel.driver import deflated_newton, dd_norm, load, make_solver, newton_solve, renode  # noqa: E402
from pinning import predict_dd  # noqa: E402

M, N6 = 32, [0.3, 0.4, 0.5, 0.6]
a = types.SimpleNamespace(kt=36, kc=220, ks=160, threads=3, kfreeze=13, pin=None, tol_step=1e-32, hmax=0.05,
                          corrector="dnp", tol=5.45e-19, w=16, verbose=False, junk_only=False, delta_tol=1e-14,
                          polish_from=1e-12, polish_its=4, lm=0)


def solver(D):
    return make_solver(D, M, nodes=N6, kt=36, kc=220, ks=160, threads=3, kfreeze=13)


hist = []
for f in sorted(glob.glob(os.path.join(H, "data", "seeds", "step_test", "up13_D*.npz")))[-3:]:
    hh, hl, hm = load(f)
    s0 = make_solver(hm["D"], M, nodes=list(hm["nodes"]), kt=36, kc=220, ks=160, threads=3)
    hh, hl = renode(s0, hh, hl, list(hm["nodes"]), N6, M)
    hh, hl = scan_hp.freeze_p1(hh, hl, a, M)
    s, hh, hl, hst, ok, dt = scan_hp.solve_at(hm["D"], hh, hl, a, N6, M, 25)
    print(f"history {hm['D']}: ok {ok} |F| {hst[0]['F']:.1e} -> {hst[-1]['F']:.1e} in {len(hst)} its ({dt:.0f} s)",
          flush=True)
    hist.append((hm["D"], hh, hl))
zh, zl, m = load(os.path.join(P, "start_D60.0020_x0.6_kf13.npz"))
s, zh, zl, hst, ok, dt = scan_hp.solve_at(m["D"], zh, zl, a, N6, M, 25)
print(f"start {m['D']}: ok {ok} |F| {hst[0]['F']:.1e} -> {hst[-1]['F']:.1e} in {len(hst)} its", flush=True)
hist.append((m["D"], zh, zl))

bh, bl, bm = load(os.path.join(P, "base_D60.00_D60.0235.npz"))
s7 = make_solver(bm["D"], M, nodes=list(bm["nodes"]), kt=36, kc=220, ks=100, threads=3)
th, tl = renode(s7, bh, bl, list(bm["nodes"]), N6, M)
sb = solver(bm["D"])
sb.set_segs(sb.plan(th, tl, 1e-32, 0.05))
print(f"base's solution at {bm['D']} truncated to 0.3-0.6: |F| in the 0.6 system {dd_norm(*sb.residual(th, tl)):.1e}",
      flush=True)

for Dn in (60.002624, bm["D"]):
    zp, zpl = predict_dd(hist, Dn, 4)
    s = solver(Dn)
    s.set_segs(s.plan(zp, zpl, 1e-32, 0.05))
    print(f"\n== D {Dn}: seed |F| {dd_norm(*s.residual(zp, zpl)):.2e}", flush=True)
    if abs(Dn - bm["D"]) < 1e-9:
        d = (zp - th) + (zpl - tl)
        print(f"seed - truncated base solution: max {np.abs(d).max():.1e}, Delta {d[-1]:+.1e}, "
              f"centre/nodes {np.abs(d[:2 * M + 4 * M * 4]).max():.1e}, horizon {np.abs(d[2 * M + 4 * M * 4:]).max():.1e}",
              flush=True)
    t0 = time.time()
    print("-- plain Newton", flush=True)
    newton_solve(s, zp, zpl, tol=1e-40, it_max=8, verbose=True)
    print(f"-- deflated Newton, w 16 ({time.time() - t0:.0f} s so far)", flush=True)
    deflated_newton(s, zp, zpl, tol=5.45e-19, it_max=8, w=16, verbose=True)
print("done", flush=True)
