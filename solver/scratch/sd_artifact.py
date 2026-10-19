import numpy as np, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from spec_long import make_long
from spec_c import freeze_steps, jacobian_cs
for D in (24.235037500000008, 24.53503750000001, 24.93503750000001, 25.195):
    try: z = np.load(f"longB_D{D:.4f}.npy")
    except FileNotFoundError: continue
    S = freeze_steps(make_long(23, D - 2, nodes=[0.02,0.05,0.1,0.16,0.23,0.3,0.4,0.5,0.6], Ks=60), z[-1])
    F, J = jacobian_cs(S, z)
    out = []
    for rc in (1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10):
        y = np.linalg.lstsq(J[:, :-1], J[:, -1], rcond=rc)[0]; out.append(np.linalg.norm(J[:, -1] - J[:, :-1] @ y))
    s = np.linalg.svd(J[:, :-1], compute_uv=False)
    print(f"D={D:.3f}: s_Delta for rcond 1e-15..1e-10: " + " ".join(f"{v:.1e}" for v in out) +
          f" | #sv of J_others below 1e-13*smax: {(s < 1e-13*s[0]).sum()}, below 1e-12: {(s < 1e-12*s[0]).sum()}")
