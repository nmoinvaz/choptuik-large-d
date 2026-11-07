"""Second and third alpha = 2 solutions of SC02 (97) at D = 3 (after the explicit z = 6): the
Keller-Segel self-similar profiles beyond the explicit one, as crossings of alpha(z) = 2.
Two precisions; the explicit z = 6 is the planted check of the same call."""
import flint
from flint import arb
import sp_bvp2 as B
for prec, X in ((110, '13'), (150, '14')):
    flint.ctx.prec = prec
    tol = arb(2)**(-prec + 20)
    P = B.Prob('3', X, '2', 26)
    out = []
    for z0, A0 in (('6.3', '4'), ('2300', '1.6'), ('150000', '1.3')):
        z, A = B.z_of_alpha(P, arb(2), arb(A0), arb(z0), tol)
        out.append(z.str(25, radius=False))
    print('prec %d X %s: z = %s' % (prec, X, ', '.join(out)), flush=True)
