import numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
D = 6.0; S = make(31, n=D - 2); z = np.load("solD_6.000_kc31.npy")
F, J = S.jacobian(z)
U, s, Vt = np.linalg.svd(J)
print("Delta-component of the 8 weakest directions:", np.abs(Vt[-8:, -1]))
print("residual projected on weakest left vectors:", np.abs(U[:, -8:].T @ F))
print("residual norm outside the 8 weakest:", np.linalg.norm(F - U[:, -8:] @ (U[:, -8:].T @ F)))
# truncated-SVD Newton
def tsvd_step(F, J, rel=1e-8):
    U, s, Vt = np.linalg.svd(J); keep = s > rel * s[0]
    return -(Vt[keep].T @ ((U[:, keep].T @ F) / s[keep])), (~keep).sum()
for it in range(5):
    F, J = S.jacobian(z)
    dz, nd = tsvd_step(F, J)
    z = z + dz
    print(f"it {it}: |F| {np.linalg.norm(F):.3e}, dropped {nd} directions, |dz| {np.linalg.norm(dz):.2e}, Delta {z[-1]:.12f}")
print("final |F|", np.linalg.norm(S.residual(z[None, :])[0]))
np.save("solD_6.000_kc31_tsvd.npy", z)
