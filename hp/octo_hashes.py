"""SHA-256 of the octo-double kernel's outputs at a stored eight-part state, for a bit-for-bit comparison of two hp
builds (multidouble 4.8.2 against the branch perf/od-dd-const-products): the residual, the Jacobian with one and with
two tangents, the Newton step with F and the LU's extreme pivots, and the series by order to N. Run from hp/ with each
build, then diff.

    python3 octo_hashes.py RAW STATE.npz [--threads 6] [--N 6] [--mutate_dz 0] > hashes_<build>.txt

Control: --mutate_dz 1 moves one entry of the step's leading part by one ulp before hashing, so its line must differ.
"""
import hashlib
import sys

import numpy as np


def h(arrs):
    """the SHA-256 of the arrays' float64 bytes, in order"""
    d = hashlib.sha256()
    for a in arrs:
        d.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return d.hexdigest()


def main(argv):
    # LF line ends on every platform, so a Windows output compares byte for byte with Linux's (#962)
    sys.stdout.reconfigure(newline="\n")
    from anm_run import coefficients_o
    from octo_corrector import truth_system
    raw, state = argv[0], argv[1]
    opt = dict(threads=6, N=6, mutate_dz=0)
    for k in list(opt):
        if "--" + k in argv:
            opt[k] = int(argv[argv.index("--" + k) + 1])
    z, meta, (s, _, _, _), info = truth_system(raw, state, threads=opt["threads"])
    zo = [np.ascontiguousarray(np.asarray(p)) for p in z]
    print("state", state, "D", meta.get("D_str", meta.get("D")), info["plan"])
    print("residual_o", h(s.residual_o(list(zo))))
    for t in (1, 2):
        print(f"jacobian_o tangents {t}", h(s.jacobian_o(list(zo), tangents=t)))
    st = s.newton_o(list(zo))
    dz = [np.array(p) for p in st[0]]
    if opt["mutate_dz"]:
        dz[0][0] = np.nextafter(dz[0][0], np.inf)
    print("newton_o dz", h(dz))
    print("newton_o F, min and max |u_kk|", h([st[1], st[2], np.float64(st[3]), np.float64(st[4])]))
    A, _ = coefficients_o(s, zo, opt["N"], by_order=True)
    for k in range(1, len(A)):
        print(f"series_o z_{k}", h(A[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
