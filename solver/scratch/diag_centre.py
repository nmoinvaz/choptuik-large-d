import numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
for D, f in [(4.0, "spec_kc31.npy"), (6.0, "solD_6.000_kc31.npy")]:
    S = make(31, n=D - 2); z = np.load(f)
    parts, Dl = S.split(z[None, :]); pars = S.parities()
    p1c, c0c = parts[0], parts[1]
    om = np.full(1, 2 * np.pi / Dl[0])
    base = S._centre_to_node0(S.phys(p1c, "odd"), S.phys(c0c, "even"), om)
    for k in (5, 15, 21, 27, 31):
        j = list(S.kodd).index(k)
        pc = p1c.copy(); pc[0, j] += 1e-6
        s = S._centre_to_node0(S.phys(pc, "odd"), S.phys(c0c, "even"), om)
        d = (s - base) / 1e-6
        # response at node 0.02: magnitude of mode k in p and m
        Fp = np.abs(np.fft.rfft(d[0, 0])) / S.N; Fm = np.abs(np.fft.rfft(d[1, 0])) / S.N
        print(f"D={D} k={k:2d}: response at x=0.02: |dp_k|={Fp[k]:.2e} |dm_k|={Fm[k]:.2e}   (naive x*dp1 = {0.02*0.5:.1e})  max over modes p {Fp.max():.2e}")
