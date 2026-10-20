"""Fixed A and w: solve the higher-harmonic conditions S_3 = ... = S_M = 0 (harmonic 1 left free) with de-mixed
residuals, report F_nu/F_1 (phases included) and the harmonic-1 residual r_1. Thin wrapper around full_M.py --fixA.

F_3 is seeded from the M = 3 result at (2.72, w_inf), F3/F1 = 0.054180 + 0.004186i (scaled to the given A by agent C's
leading law F3/F1 ~ q A^2), higher harmonics from F_{nu+2} = q A^2 F_nu, q = 0.0085 + 0.0011i.

    python -u fixedA.py M wf [A] [--procs N]     e.g.  python -u fixedA.py 5 1.0 --procs 8   (A defaults to 2.72)
M = 3 at (2.72, w_inf) gives F3/F1 = 0.054180 + 0.004186i in about 25 s (one core).
"""
import sys
import full_M


def run():
    argv = sys.argv[1:]
    extra = []
    if "--procs" in argv:
        k = argv.index("--procs")
        extra = argv[k:k + 2]
        del argv[k:k + 2]
    M, wf = int(argv[0]), float(argv[1])
    A = float(argv[2]) if len(argv) > 2 else 2.72
    ratio3 = (0.054180 + 0.004186j) * (A / 2.72) ** 2
    z = (A / 2j) * ratio3
    sys.argv = ["full_M.py", str(M), repr(A), repr(wf), repr(-2 * z.imag), repr(2 * z.real), "--fixA"] + extra
    full_M.main()


if __name__ == "__main__":
    run()
