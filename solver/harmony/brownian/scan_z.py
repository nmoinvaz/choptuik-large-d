import sys, time, flint
from flint import arb
sys.argv = [sys.argv[0]] + sys.argv[1:]
import sp_bvp2 as B
D = sys.argv[1]; flint.ctx.prec = int(sys.argv[2]); X = sys.argv[3]
P = B.Prob(D, X, '2', 24)
tol = arb(2)**(-flint.ctx.prec + 20)
A, al = arb(4), arb('2.05')
for zs in sys.argv[4:]:
    t = time.time()
    al, A = B.alpha_of_z(P, arb(zs), A, al, tol)
    print('D=%s z=%s alpha=%s A=%s [%.1fs]' % (D, zs, al.str(20, radius=False), A.str(10, radius=False), time.time() - t), flush=True)
