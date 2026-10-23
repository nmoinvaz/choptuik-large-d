"""M = 512 small check (notes/HIGH_M.md 9.2): the D = 3.7 M = 256 solution resampled to M = 384 and 512, its own Taylor plan
(1e-30, 0.05), the residual (|F| and seconds), the series tails and the peak RSS. Run from hp/ with PYTHONPATH at the build."""
import json, os, resource, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
import lowD_dd
from hpkernel.driver import load, make_solver, resample_z
zh, zl, m = load('data/lowD_states/highm15_D3.7000_M256.npz')
for M in [int(v) for v in sys.argv[1:]] or [384, 512]:
    z, _ = resample_z(np.asarray(zh) + np.asarray(zl), 256, M, len(m['nodes']))
    t = time.time()
    s = make_solver(m['D'], M, nodes=m['nodes'], kt=m['kt'], kc=m['kc'], ks=m['ks'], threads=2)
    t_make = time.time() - t
    t = time.time(); sg = s.plan(z, 0 * z, 1e-30, 0.05); t_plan = time.time() - t
    s.set_segs(sg)
    t = time.time(); Fh, Fl = s.residual(z, 0 * z); t_res = time.time() - t
    F = float(np.abs(np.asarray(Fh) + np.asarray(Fl)).max())
    tc, th = lowD_dd.tails(s, z, 0 * z, m['nodes'], M)
    rec = dict(M=M, N=s.size, steps=sum(len(g) - 1 for g in sg), F=F, s_make=round(t_make, 1), s_plan=round(t_plan, 1),
               s_residual=round(t_res, 1), tail_c=tc, tail_h=th, peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
               load=round(os.getloadavg()[0], 2))
    print(json.dumps(rec), flush=True)
    with open('data/lowD_highM/check512.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
