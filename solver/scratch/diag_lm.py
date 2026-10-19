import numpy as np, warnings, time
warnings.filterwarnings("ignore")
from run_spec import make
D = 6.0
S = make(31, n=D - 2); z = np.load("solD_6.000_kc31.npy")
F, J = S.jacobian(z)
print("|F| =", np.linalg.norm(F))
sv = np.linalg.svd(J, compute_uv=False)
print("singular values: largest", sv[0], " smallest 6:", sv[-6:])
# plain Newton steps with this Jacobian
for it in range(6):
    F, J = S.jacobian(z)
    dz = np.linalg.solve(J, -F)
    for t in (1.0, 0.5, 0.25):
        zt = z + t * dz; Ft = S.residual(zt[None, :])[0]
        if np.linalg.norm(Ft) < np.linalg.norm(F): break
    print(f"it {it}: |F| {np.linalg.norm(F):.3e} -> {np.linalg.norm(Ft):.3e} (t={t}), |dz| {np.linalg.norm(dz):.2e}, Delta {zt[-1]:.12f}")
    z = zt
