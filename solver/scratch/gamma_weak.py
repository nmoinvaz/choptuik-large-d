"""How much does gamma move along the weakly determined Delta direction?  Two converged solutions at the same D
that differ along it (different solvers): compare their growth rates."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
from gamma import growth_rate
D = 9.35
for f in ("diag_D9.350_mu1e-08.npy", "lm_D9.350.npy"):
    S = make(23, n=D - 2); z = np.load(f)
    lam, _ = growth_rate(S, z, 2.19, 2.23, xtol=1e-11)
    print(f"{f}: Delta={z[-1]:.8f}  lambda0={lam:.8f}  gamma={1/lam:.8f}", flush=True)
