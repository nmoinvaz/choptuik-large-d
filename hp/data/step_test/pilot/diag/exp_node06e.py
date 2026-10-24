"""T43 pilot diagnosis, part 5: does Levenberg-Marquardt (off in the step_test runs, --lm 0) converge from the node06
run's seeds? The seeds at D 60.002624 and 60.023452 rebuilt as in exp_node06.py, then lm_solve (15 its, Newton after
1e-22, as scan_hp.py's --lm 15) and plain Newton with 30 halvings.
    python exp_node06e.py HP_DIR PILOT_DIR"""
import ctypes, glob, os, sys, time, types
import numpy as np

H, P = sys.argv[1], sys.argv[2]
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)    # below normal
sys.path.insert(0, H)
os.chdir(H)
import scan_hp  # noqa: E402
from hpkernel.driver import dd_norm, lm_solve, load, make_solver, newton_solve, renode  # noqa: E402
from pinning import predict_dd  # noqa: E402

M, N6 = 32, [0.3, 0.4, 0.5, 0.6]
a = types.SimpleNamespace(kt=36, kc=220, ks=160, threads=6, kfreeze=13, pin=None, tol_step=1e-32, hmax=0.05,
                          corrector="dnp", tol=5.45e-19, w=16, verbose=False, junk_only=False, delta_tol=1e-14,
                          polish_from=1e-12, polish_its=4, lm=0)
hist = []
for f in sorted(glob.glob(os.path.join(H, "data", "seeds", "step_test", "up13_D*.npz")))[-3:]:
    hh, hl, hm = load(f)
    s0 = make_solver(hm["D"], M, nodes=list(hm["nodes"]), kt=36, kc=220, ks=160, threads=6)
    hh, hl = renode(s0, hh, hl, list(hm["nodes"]), N6, M)
    hh, hl = scan_hp.freeze_p1(hh, hl, a, M)
    _, hh, hl, _, _, _ = scan_hp.solve_at(hm["D"], hh, hl, a, N6, M, 25)
    hist.append((hm["D"], hh, hl))
zh, zl, m = load(os.path.join(P, "start_D60.0020_x0.6_kf13.npz"))
_, zh, zl, _, _, _ = scan_hp.solve_at(m["D"], zh, zl, a, N6, M, 25)
hist.append((m["D"], zh, zl))
for Dn in (60.002624, 60.023452):
    zp, zpl = predict_dd(hist, Dn, 4)
    s = make_solver(Dn, M, nodes=N6, kt=36, kc=220, ks=160, threads=6, kfreeze=13)
    s.set_segs(s.plan(zp, zpl, 1e-32, 0.05))
    print(f"\n== D {Dn}: seed |F| {dd_norm(*s.residual(zp, zpl)):.2e}", flush=True)
    t0 = time.time()
    print("-- Levenberg-Marquardt", flush=True)
    _, _, h = lm_solve(s, zp, zpl, tol=5.45e-19, it_max=15, verbose=True, newton_after=1e-22)
    print(f"   -> |F| {h[-1]['F'] if h else float('nan'):.1e} ({time.time() - t0:.0f} s)", flush=True)
    print("-- plain Newton, 30 halvings", flush=True)
    newton_solve(s, zp, zpl, tol=5.45e-19, it_max=6, verbose=True, max_halvings=30)
print("done", flush=True)
