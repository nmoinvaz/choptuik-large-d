"""Long-horizon-series variant: the horizon (SSH) Taylor series is evaluated directly at the last node x_K (u = 1 - x_K
~ 0.3) with many orders, replacing the stiff integration from u0 = 0.03.  This enlarges the horizon-singular signal
(selection strength ~ u_K^{s0}) without raising the residual floor."""
import types, numpy as np
from spec import DSSSpec
from run_spec import NODES
from spec_c import freeze_steps

def make_long(kc, n, nodes=None, Ks=50):
    nodes = nodes or [x for x in NODES if x <= 0.7 + 1e-12]
    S = DSSSpec(kc=kc, nodes=nodes, n=n, Ks=Ks, u0=1 - nodes[-1], seg_h=0.002, log_steps=150)
    def _ssh_to_last(self, mS, aS, omega):
        P, M, A, C = self.ssh_series(mS, aS, omega)
        return self.trunc(np.stack([self.eval_series(Q, self.u0) for Q in (P, M, A, C)]))
    S._ssh_to_last = types.MethodType(_ssh_to_last, S)
    return S

def from_standard(S_std, z, S_long):
    """Drop the node blocks of S_std that S_long does not have (nodes beyond x_K)."""
    sizes, K, L = S_std.layout(); offs = np.concatenate([[0], np.cumsum(sizes)])
    keep_nodes = len(S_long.nodes)
    j0 = 2 + 4 * keep_nodes; j1 = 2 + 4 * K
    keep = np.r_[0:offs[j0], offs[j1]:L]
    return z[keep]
