"""Faster, multi-process version of amp_flint.Amp: same problem, same JSON output, results agree within ball radii.

What changed against amp_flint.py (details in the docstrings below):

1. Products (the convolutions over (j, mu)).  Order eps^k has only the harmonics mu = -k, -k+2, ..., k, so a product of
   orders k1 + k2 = k is packed with stride k + 1 (index (mu + k_i)/2 in a row) instead of 4K + 1, and only the first
   (J + 1)(k + 1) coefficients are formed (acb_series, a truncated product).  The two d d products of pass_even that are
   mirror images are formed once.  (Packing (j, mu) into one polynomial is Kronecker substitution, Kronecker 1882, as
   in D. Harvey, arXiv:0712.4046; a truncated product is a short product, T. Mulders, AAECC 11, 69 (2000).)  All products of one order are independent; with --procs N they are spread over N
   worker processes, which keep their own copies of the finished tables (each table is sent once, when its order is done).
2. The late-order fits (S_at).  The least-squares matrix depends only on nu, so it is built once per nu, in ball
   arithmetic, and reduced to G = (A^H A)^{-1} A^H; each fit is then c = G y.  The psi tables are built once per nu (acb
   polygamma), the exponential of the normalisation series uses N' = L' N (O(k^2) instead of O(k^3)), and every row is
   done in acb instead of mpmath.  The fits of one order are independent of one another (S_nu at order k uses d^(k) only
   at harmonic nu, and E, <F^2> use orders below k), so tuning order matches the serial code.
3. Nothing else is approximated: the tuned f_nu is computed in acb at full precision (the serial code rounds it to 50
   digits), so the two codes differ only at the level of the serial code's own rounding.

Threads do not help: python-flint keeps the GIL during a product (two threads take 2.3x one product).  Processes are
started with 'spawn', so this runs the same way on Windows; keep calls under an  if __name__ == "__main__":  guard.

    python3 amp_flint_par.py wf K J [prec_bits] [--procs N] [--fit phys [--mf LO:HI] [--fit-step S] [--phys-prec B]]
                             [--out FILE]
        writes sigF_w<wf>_K<K>_J<J>.json (or FILE), same keys as amp_flint.py (plus 'procs', 'seconds', 'timing',
        'order_times', 'sigma_rad'; with --fit phys also 'sigma_err', 'fit_info', 'mf', 'fit_step', 'phys_prec')

4. The fit basis (--fit phys; the default 'legacy' keeps amp_flint.py's basis and its results).  At order k harmonic nu
   contains, besides its own family (exponent g_nu), every family nu' with |nu'| <= k: products of the fast mode with
   functions of the characteristic label theta_0 = theta - 2 w ln xi (from integrating the transport equation) carry
   e^{i nu' theta_0} whatever S_nu' is, so they appear at relative order 1/j with amplitude ~ eps^{|nu' - nu|}.  The
   legacy basis has only nu' - nu = +-2, +-4, +-6 with 4 terms each.  From k = 11 (nu >= 7) and k = 13 (nu = 1, 3) the
   missing families (and too few terms) bias the fits: S_3^(13) came out as 1e2 instead of ~1e-4, the tuned f_3^(13)
   was wrong by 1e-5, and from there sigma_7, sigma_8 blew up.  'phys' fits the main family (M terms) and every family
   nu' (odd, |nu'| <= k, nu' != nu) with Mf terms 1/j^1..1/j^Mf each, for Mf = LO..HI (one normal matrix at
   --phys-prec bits serves all Mf as prefixes), and takes S at the Mf where it changes least from Mf - 1; that change is
   the error estimate ('fit_info', 'sigma_err').  Use prec 640: the data's noise floor is then ~1e-187 relative
   (~1e-90 at prec 320), which the high-Mf fits need.  The fits for nu = 3..k at one order run in parallel on the
   workers.  --fit-step S uses every S-th j (the basis has 13 + k*HI columns; keep the rows (J - 2J/5)/S above that).
   The 'fit resid' printed in legacy mode is now computed on midpoints; before, it was dominated by the radii of y,
   which |G| ~ 1e29 amplifies, and printed values > 1 that did not mean the fit had failed.
For K >= 13 use --fit phys with prec 640 (see the docstring of fit_phys); the legacy basis is reliable only to k = 11
(sigma_5, to ~1e-9 at J = 1000; sigma_1..4 are biased by ~3e-10 at J = 500).
"""
import sys
import json
import time
import pickle
import multiprocessing as mpr
import mpmath as mp
from flint import acb, acb_mat, acb_series, arb, arf, ctx

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from amp_flint import Amp, to_mp  # noqa: E402

ZERO = acb(0)


# ---------- ball (de)serialisation: exact midpoints, radii rounded up; works across spawn

def _ser_rows(rows):
    flat = []
    for row in rows:
        for z in row:
            re, im = z.real, z.imag
            a, b = re.mid().man_exp()
            c, d = re.rad().mid().man_exp()
            e, f = im.mid().man_exp()
            g, h = im.rad().mid().man_exp()
            flat.extend((int(a), int(b), int(c), int(d), int(e), int(f), int(g), int(h)))
    return len(rows), (len(rows[0]) if rows else 0), flat


def _de_rows(blob):
    nr, nc, flat = blob
    out, p = [], 0
    for _ in range(nr):
        row = []
        for _ in range(nc):
            a, b, c, d, e, f, g, h = flat[p:p + 8]
            p += 8
            row.append(acb(arb(arf((a, b)), arf((c, d))), arb(arf((e, f)), arf((g, h)))))
        out.append(row)
    return out


# ---------- products of compressed tables (row j holds mu = -o, -o+2, ..., o at positions 0..o)

def _series(T, o, Wp, J):
    c = [ZERO] * ((J + 1) * Wp)
    for j, row in enumerate(T):
        c[j * Wp: j * Wp + o + 1] = row
    return acb_series(c)


def _products(tabs, k, J, tasks):
    """tasks: (group, keyX, keyY, coef) with order(X) + order(Y) = k; returns {group: compressed (J+1) x (k+1) table}"""
    Wp = k + 1
    n = (J + 1) * Wp
    ctx.cap = n
    acc = {}
    for grp, kx, ky, coef in tasks:
        P = _series(tabs[kx], kx[1], Wp, J) * _series(tabs[ky], ky[1], Wp, J)
        if coef != 1:
            P = P * coef
        acc[grp] = P if grp not in acc else acc[grp] + P
    out = {}
    for grp, P in acc.items():
        cs = P.coeffs()
        cs = cs + [ZERO] * (n - len(cs))
        out[grp] = [cs[j * Wp:(j + 1) * Wp] for j in range(J + 1)]
    return out


def phys_families(nu, kmax):
    """signed shifts t = nu' - nu of the families present in harmonic nu at order kmax: every odd nu' != nu, |nu'| <= kmax"""
    return [nup - nu for nup in range(-kmax, kmax + 1, 2) if nup != nu]


def fit_phys(wmid, nu, k, jlo, J, M, mf_lo, mf_hi, step, fit_prec, ys):
    """Late-order fit of y_j (j = jlo, jlo + step, ..., <= J) with the 'physical' basis: the main family (1 and M terms
    1/(j+g0-1)...(j+g0-m)) plus every family nu' (odd, |nu'| <= k, nu' != nu), each with Mf terms
    Gamma(j+g')/Gamma(j+g0) / ((j+g'-1)...(j+g'-m)), m = 1..Mf. Columns are ordered by level m, so the bases for
    Mf = mf_lo..mf_hi are prefixes of one matrix: one normal matrix N = A^H A (at fit_prec, on midpoints) serves them all.
    Returns [(Mf, S, relative residual)] with S = the coefficient of the constant column (the multiplier)."""
    p0 = ctx.prec
    ctx.prec = fit_prec
    try:
        w = _de_rows(wmid)[0][0] if isinstance(wmid, tuple) else acb(wmid)
        g0 = acb(0, nu) * w + acb(1) / 2
        fams = phys_families(nu, k)
        js = list(range(jlo, J + 1, step))
        rows = []
        for j in js:
            main = [acb(1)]
            t = acb(1)
            for m in range(1, M + 1):
                t = t / (j + g0 - m)
                main.append(t)
            lg0 = (j + g0).lgamma()
            levels = []
            ts = []
            for tsh in fams:
                gs = g0 + acb(0, tsh) * w
                ts.append((gs, ((j + gs).lgamma() - lg0).exp()))
            for m in range(1, mf_hi + 1):
                lev = []
                for i, (gs, t) in enumerate(ts):
                    t = t / (j + gs - m)
                    ts[i] = (gs, t)
                    lev.append(t)
                levels.extend(lev)
            rows.append(main + levels)
        n = len(rows[0])
        sc = [max(abs(r[c]).mid() for r in rows) for c in range(n)]
        A = acb_mat([[r[c].mid() / sc[c] for c in range(n)] for r in rows])
        yv = [acb(v.mid()) for v in ys]
        y = acb_mat([[v] for v in yv])
        AH = A.transpose().conjugate()
        N = AH * A
        b = AH * y
        ny2 = sum((abs(v) ** 2 for v in yv), arb(0))
        out = []
        nf = len(fams)
        for Mf in range(mf_lo, mf_hi + 1):
            p = M + 1 + nf * Mf
            Np = acb_mat([[N[i, l] for l in range(p)] for i in range(p)])
            bp = acb_mat([[b[i, 0]] for i in range(p)])
            c = Np.solve(bp, algorithm="approx")
            # |A c - y|^2 = |y|^2 - Re(c^H b) at the least-squares solution (fit_prec carries the cancellation)
            cb = sum((c[l, 0].conjugate() * b[l, 0] for l in range(p)), acb(0))
            r2 = (ny2 - cb.real).mid()
            rel = float((abs(r2) / ny2).sqrt().mid()) if ny2 != 0 else float("nan")
            S = (c[0, 0] / sc[0]).mid()
            out.append((Mf, S, rel))
        return out
    finally:
        ctx.prec = p0


def pick_plateau(res):
    """choose among [(Mf, S, resid)] the Mf where S changes least from Mf - 1 (and the residual still falls);
    returns (Mf, S, err = |S(Mf) - S(Mf-1)|, resid)"""
    best = None
    for (m0, S0, r0), (m1, S1, r1) in zip(res, res[1:]):
        if not (r1 < r0):
            continue
        d = float(abs(S1 - S0).mid())
        if best is None or d < best[2]:
            best = (m1, S1, d, r1)
    if best is None:
        m, S, r = res[-1]
        best = (m, S, float("inf"), r)
    return best


def _worker(conn, prec):
    ctx.prec = prec
    tabs = {}
    while True:
        msg = pickle.loads(conn.recv_bytes())
        op = msg[0]
        if op == "fitphys":
            _, args, yblob = msg
            ys = _de_rows(yblob)[0]
            res = fit_phys(*args, ys)
            conn.send_bytes(pickle.dumps([(m, _ser_rows([[S]]), r) for m, S, r in res],
                                         protocol=pickle.HIGHEST_PROTOCOL))
            ctx.prec = prec
        elif op == "put":
            for key, blob in msg[1]:
                tabs[key] = _de_rows(blob)
        elif op == "prod":
            _, k, J, tasks = msg
            res = _products(tabs, k, J, tasks)
            conn.send_bytes(pickle.dumps({g: _ser_rows(T) for g, T in res.items()}, protocol=pickle.HIGHEST_PROTOCOL))
        elif op == "fitprep":
            _, params, nus = msg
            amp = AmpPar(**params)
            out = {}
            for nu in nus:
                A, G, sca, tab = amp.fit_data(nu)
                out[nu] = (_ser_rows(A.tolist()), _ser_rows(G.tolist()), [float(x) for x in sca],
                           _ser_rows([[acb(base)] + ps for base, ps in tab]))
            conn.send_bytes(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
            ctx.prec = prec
        elif op == "stop":
            conn.close()
            return


def _add_rows(A, B):
    return [[x + y for x, y in zip(a, b)] for a, b in zip(A, B)]


class AmpPar(Amp):
    def __init__(self, wf, K, J, prec=320, procs=1, **kw):
        fit_prec = kw.pop("fit_prec", None)
        # late-order fit: "legacy" (default; the fixed basis of amp_flint.py) or "phys" (every family nu' present at the
        # order, Mf = mf[0]..mf[1] terms each, Mf chosen at the plateau of S; see fit_phys)
        self.fit_mode = kw.pop("fit", "legacy")
        self.mf = tuple(kw.pop("mf", (4, 12)))
        self.fit_step = int(kw.pop("fit_step", 1))
        self.phys_prec = kw.pop("phys_prec", None)
        super().__init__(wf, K, J, prec=prec, **kw)
        self.params = dict(wf=wf, K=K, J=J, prec=prec, fit_prec=fit_prec, **kw)
        self.prec, self.procs = prec, max(1, int(procs))
        self.fit_prec = fit_prec or max(1024, 3 * prec)
        self.phys_prec = self.phys_prec or max(1600, 3 * prec)
        self.fit_info = {}                    # "k_nu" -> (Mf chosen, err estimate of S, relative residual)
        self.workers = []
        self.ctab = {}                      # compressed finished tables, keyed (name, order)
        self.fit_cache = {}
        self.theta_cache = {}
        self.timing = {"products": 0.0, "fit_setup": 0.0, "fits": 0.0}
        self.order_times = {}                 # k -> (seconds in products, number of products)

    # --- worker pool (spawn; one pipe per worker so each keeps its own copy of the tables)
    def start(self):
        if self.procs > 1 and not self.workers:
            cx = mpr.get_context("spawn")
            for _ in range(self.procs):
                a, b = cx.Pipe()
                p = cx.Process(target=_worker, args=(b, self.prec))
                p.start()
                self.workers.append((p, a))
            self.prefetch_fits()

    def prefetch_fits(self):
        """build the per-nu fit data (nu = 1, 3, ..., K) in the workers, in parallel"""
        t0 = time.time()
        nus = [nu for nu in range(1, self.K + 1, 2) if nu not in self.fit_cache]
        nw = len(self.workers)
        busy = []
        for i, (p, conn) in enumerate(self.workers):
            mine = nus[i::nw]
            if mine:
                conn.send_bytes(pickle.dumps(("fitprep", self.params, mine), protocol=pickle.HIGHEST_PROTOCOL))
                busy.append(conn)
        for conn in busy:
            for nu, (Ab, Gb, sc, tb) in pickle.loads(conn.recv_bytes()).items():
                tab = [(row[0], row[1:]) for row in _de_rows(tb)]      # base_j is complex
                self.fit_cache[nu] = (acb_mat(_de_rows(Ab)), acb_mat(_de_rows(Gb)), [arb(x) for x in sc], tab)
        self.timing["fit_setup"] += time.time() - t0

    def stop(self):
        for p, conn in self.workers:
            try:
                conn.send_bytes(pickle.dumps(("stop",)))
            except (BrokenPipeError, OSError):
                pass
        for p, conn in self.workers:
            p.join(timeout=30)
        self.workers = []

    def publish(self, name, k, T):
        H = self.H
        C = [row[H - k: H + k + 1: 2] for row in T]
        self.ctab[(name, k)] = C
        if self.workers:
            msg = pickle.dumps(("put", [((name, k), _ser_rows(C))]), protocol=pickle.HIGHEST_PROTOCOL)
            for p, conn in self.workers:
                conn.send_bytes(msg)

    def expand(self, C, k):
        H = self.H
        out = []
        for row in C:
            full = list(self.zero_row)
            full[H - k: H + k + 1: 2] = row
            out.append(full)
        return out

    def products(self, k, tasks):
        """sum of the products in tasks (all of order k), by group, as full-width (2H+1) rows"""
        t0 = time.time()
        try:
            return self._products(k, tasks)
        finally:
            dt = time.time() - t0
            self.timing["products"] += dt
            self.order_times[k] = (round(dt, 2), len(tasks))

    def _products(self, k, tasks):
        J = self.J
        if not self.workers:
            res = _products(self.ctab, k, J, tasks)
        else:
            nw = len(self.workers)
            # balance: same cost per task at fixed k, so deal them out in turn
            busy = []
            for i, (p, conn) in enumerate(self.workers):
                chunk = tasks[i::nw]
                if chunk:
                    conn.send_bytes(pickle.dumps(("prod", k, J, chunk), protocol=pickle.HIGHEST_PROTOCOL))
                    busy.append(conn)
            res = {}
            for conn in busy:
                part = pickle.loads(conn.recv_bytes())
                for g, blob in part.items():
                    T = _de_rows(blob)
                    res[g] = T if g not in res else _add_rows(res[g], T)
        zero = [[ZERO] * (k + 1) for _ in range(J + 1)]
        return {g: self.expand(res.get(g, zero), k) for g in set(t[0] for t in tasks) | set(res)}

    # --- passes (same algebra as amp_flint.Amp)
    def pass_even(self, k):
        J = self.J
        tasks = []
        for k1 in range(1, k, 2):
            k2 = k - k1
            if k1 < k2:
                tasks.append(("D", ("d", k1), ("d", k2), 2))
            elif k1 == k2:
                tasks.append(("D", ("d", k1), ("d", k2), 1))
        for k1 in range(2, k - 1, 2):
            tasks.append(("G", ("g", k1), ("e", k - k1), 1))
        R = self.products(k, tasks)
        D = R["D"]
        e = [list(self.zero_row)] + [[x / 4 for x in D[m - 1]] for m in range(1, J + 1)]
        self.e[k] = e
        G = [list(row) for row in e]
        if "G" in R:
            G = _add_rows(G, R["G"])
        g = [list(self.zero_row)] + [[x / (2 * m) for x in G[m]] for m in range(1, J + 1)]
        self.g[k] = g
        self.publish("e", k, e)
        self.publish("g", k, g)

    def pass_odd(self, k):
        H, J = self.H, self.J
        tasks = []
        for k1 in range(2, k, 2):
            tasks.append(("Ss", ("g", k1), ("s", k - k1), 1))
            tasks.append(("Sd", ("g", k1), ("d", k - k1), 1))
        zero = [list(self.zero_row) for _ in range(J + 1)]
        R = self.products(k, tasks) if tasks else {}
        Ss, Sd = R.get("Ss", zero), R.get("Sd", zero)
        Dth = [acb(0, mu) * self.w for mu in range(-H, H + 1)]
        d = [[2 * v for v in self.F.get(k, self.zero_row)]]
        s = []
        for j in range(J + 1):
            sj = [(2 * j + 1 + Dth[i]) * d[j][i] - Ss[j][i] for i in range(2 * H + 1)]
            s.append(sj)
            if j < J:
                d.append([(2 * j + 2 + Dth[i]) * sj[i] / (4 * (j + 1)) - Sd[j + 1][i] for i in range(2 * H + 1)])
        self.d[k], self.s[k] = d, s

    # --- late-order fits, in acb
    def fit_data(self, nu):
        """per-nu: column-scaled basis A, G = (A^H A)^{-1} A^H, and rows (base_j, [psi sums / n!])"""
        if nu in self.fit_cache:
            return self.fit_cache[nu]
        t0 = time.time()
        s_ = acb(0, nu) * self.w
        g0 = s_ + acb(1) / 2
        a, b = (1 + s_) / 2, (2 + s_) / 2
        nmaxK = self.K // 2 + 1
        rows, tab = [], []
        fact = [arb(1)]
        for n in range(1, nmaxK + 1):
            fact.append(fact[-1] * n)
        # psi^(m)(z + 1) = psi^(m)(z) + (-1)^m m! / z^(m+1) and base_(j+1) = base_j (j+1) / ((a+j)(b+j)), from j = jlo
        za, zb = a + self.jlo, b + self.jlo
        psa = [za.polygamma(m) for m in range(nmaxK)]
        psb = [zb.polygamma(m) for m in range(nmaxK)]
        base = arb(self.jlo + 1).gamma() * za.rgamma() * zb.rgamma()
        for j in range(self.jlo, self.J + 1):
            if j > self.jlo:
                base = base * j / ((a + j - 1) * (b + j - 1))
                ia, ib = 1 / (a + j - 1), 1 / (b + j - 1)
                pa, pb = ia, ib
                for m in range(nmaxK):
                    sgn = fact[m] if m % 2 == 0 else -fact[m]
                    psa[m] += sgn * pa
                    psb[m] += sgn * pb
                    pa, pb = pa * ia, pb * ib
            row = [acb(1)]
            t = acb(1)
            for m in range(1, self.M + 1):
                t = t / (j + g0 - m)
                row.append(t)
            lg0 = (j + g0).lgamma()
            for sh in self.fams:
                for sgn in (-1, 1):
                    gs = g0 + acb(0, sgn * sh) * self.w
                    t = ((j + gs).lgamma() - lg0).exp()
                    for m in range(1, self.Mf + 1):
                        t = t / (j + gs - m)
                        row.append(t)
            rows.append(row)
            tab.append((base, [(psa[n - 1] + psb[n - 1]) / fact[n] for n in range(1, nmaxK + 1)]))
        m_, n_ = len(rows), len(rows[0])
        sc = [max(float(abs(rows[i][c]).mid()) for i in range(m_)) for c in range(n_)]
        sca = [arb(x) for x in sc]
        A = acb_mat([[rows[i][c] / sca[c] for c in range(n_)] for i in range(m_)])
        # least-squares projector G = (A^H A)^{-1} A^H, formed on midpoints at high precision (normal equations square
        # the condition number, so use >= 3x the working precision); G is then a fixed linear map with exact entries,
        # so c = G y carries exactly the ball radius of the data y (the fit itself, like the serial QR, is not bounded)
        p0 = ctx.prec
        try:
            ctx.prec = self.fit_prec
            Am = A.mid()
            AH = Am.transpose().conjugate()
            G = (AH * Am).solve(AH, algorithm="approx").mid()
        finally:
            ctx.prec = p0
        G = (G * acb(1)).mid()                  # round the fixed projector to the working precision
        out = (A, G, sca, tab)
        self.fit_cache[nu] = out
        self.timing["fit_setup"] += time.time() - t0
        return out

    def theta(self, k):
        if k not in self.theta_cache:
            self.theta_cache[k] = self.theta_series(k)
        return self.theta_cache[k]

    def S_at(self, nu, k):
        A, G, sca, tab = self.fit_data(nu)
        t0 = time.time()
        try:
            if self.fit_mode == "phys":
                return self.S_many([nu], k)[nu]
            return self._S_at(nu, k, A, G, sca, tab)
        finally:
            self.timing["fits"] += time.time() - t0

    def _S_at(self, nu, k, A, G, sca, tab):
        ys = self._ys(nu, k, tab)
        y = acb_mat([[v] for v in ys])
        c = G * y
        # residual on midpoints: the ball residual is dominated by the radii of y, which |G| ~ 1e29 amplifies
        r = A.mid() * c.mid() - y.mid()
        nr = sum(float(abs(r[i, 0].mid())) ** 2 for i in range(r.nrows())) ** 0.5
        ny = sum(float(abs(v.mid())) ** 2 for v in ys) ** 0.5
        return c[0, 0] / sca[0], (nr / ny if ny else float("nan"))

    def S_many(self, nus, k):
        """phys fits of S_nu^(k) for several nu (independent at one order), on the worker pool when there is one.
        Returns {nu: (S, err)}; err = |S(Mf) - S(Mf - 1)| at the chosen Mf. Details go to self.fit_info."""
        args, yb = {}, {}
        for nu in nus:
            tab = self.fit_data(nu)[3]
            ys = self._ys(nu, k, tab, self.fit_step)
            args[nu] = (_ser_rows([[self.w.mid()]]), nu, k, self.jlo, self.J, self.M, self.mf[0], self.mf[1], self.fit_step,
                        self.phys_prec)
            yb[nu] = ys
        res = {}
        if self.workers:
            nw = len(self.workers)
            pending = list(nus)
            while pending:
                batch, pending = pending[:nw], pending[nw:]
                for (p, conn), nu in zip(self.workers, batch):
                    conn.send_bytes(pickle.dumps(("fitphys", args[nu], _ser_rows([yb[nu]])),
                                                 protocol=pickle.HIGHEST_PROTOCOL))
                for (p, conn), nu in zip(self.workers, batch):
                    out = pickle.loads(conn.recv_bytes())
                    res[nu] = [(m, _de_rows(Sb)[0][0], r) for m, Sb, r in out]
        else:
            for nu in nus:
                res[nu] = fit_phys(*args[nu], yb[nu])
                ctx.prec = self.prec
        out = {}
        for nu in nus:
            Mf, S, err, rel = pick_plateau(res[nu])
            self.fit_info[f"{k}_{nu}"] = (Mf, err, rel)
            out[nu] = (S, err)
        return out

    def _ys(self, nu, k, tab, step=1):
        """the normalized late-order sequence y_j, j = jlo, jlo + step, ..., the fit for S_nu^(k) uses"""
        E, delta = self.theta(k)
        H = self.H
        nmax = k // 2 + 1
        hser = [ZERO] * (k + 1)
        for kk, v in delta.items():
            if kk <= k:
                hser[kk] = v / 2
        mh = [-x for x in hser]
        hp = [[acb(1)] + [ZERO] * k]
        for n in range(1, nmax + 1):
            prev, out = hp[-1], [ZERO] * (k + 1)
            for i, x in enumerate(prev):
                if x == 0:
                    continue
                for l in range(1, k + 1 - i):
                    if not (mh[l] == 0):
                        out[i + l] += x * mh[l]
            hp.append(out)
        qs = [q for q in range(1, k + 1) if any(not (hp[n][q] == 0) for n in range(1, nmax + 1))]
        # z-series pattern: for each (kp, kpp), the (H + mu, E value) pairs that contribute
        pat = []
        for kp in range(1, k + 1, 2):
            if kp not in self.d:
                continue
            for kpp in range(0, k - kp + 1, 2):
                if kpp not in E:
                    continue
                Ek = E[kpp]
                terms = []
                for mu in range(-min(H, kp), min(H, kp) + 1, 2):
                    i = H + nu - mu
                    if 0 <= i <= 2 * H and not (Ek[i] == 0):
                        terms.append((H + mu, Ek[i]))
                if terms:
                    pat.append((kp, kp + kpp, terms))
        ys = []
        for idx, j in enumerate(range(self.jlo, self.J + 1)):
            if idx % step:
                continue
            base, ps = tab[idx]
            L = [ZERO] * (k + 1)
            for q in qs:
                acc = ZERO
                for n in range(1, nmax + 1):
                    h = hp[n][q]
                    if not (h == 0):
                        acc -= ps[n - 1] * h
                L[q] = acc
            N = [acb(1)] + [ZERO] * k                 # N = exp(L): q N_q = sum_m m L_m N_{q-m}
            for q in range(1, k + 1):
                acc = ZERO
                for m in qs:
                    if m > q:
                        break
                    if not (N[q - m] == 0):
                        acc += m * L[m] * N[q - m]
                N[q] = acc / q
            z = [ZERO] * (k + 1)
            for kp, q, terms in pat:
                dj = self.d[kp][j]
                acc = ZERO
                for i, e_ in terms:
                    acc += e_ * dj[i]
                z[q] += acc
            ys.append(base * sum((z[q] * N[k - q] for q in range(k + 1)), ZERO))
        return ys

    def run(self, verbose=True, tune=True):
        self.start()
        try:
            return self._run(verbose, tune)
        finally:
            self.stop()

    def _run(self, verbose, tune):
        t0 = time.time()
        sig, f_out, S1 = [], {}, {}
        self.f_acb = {}
        self.S1_err = {}
        for k in range(1, self.K + 1):
            if k % 2 == 0:
                self.pass_even(k)
            else:
                self.pass_odd(k)
                if tune and k >= 3:
                    nus = list(range(3, k + 1, 2))
                    if self.fit_mode == "phys":
                        t1 = time.time()
                        pre = self.S_many(nus, k)
                        self.timing["fits"] += time.time() - t1
                    for nu in nus:
                        S, rs = pre[nu] if self.fit_mode == "phys" else self.S_at(nu, k)
                        s_ = acb(0, nu) * self.w
                        linS = ((1 + s_) / 2).rgamma() * ((2 + s_) / 2).rgamma()
                        f = -S / (2 * linS)
                        # the tuning constant is a choice: add it as an exact number (its midpoint), as the serial code
                        # does, so the fit's sensitivity (radius of f) is reported but not compounded into later orders
                        fx = f.mid()
                        self.add_linear(k, nu, fx)
                        self.add_linear(k, -nu, fx.conjugate())
                        fm = to_mp(f)
                        self.f_acb[f"{k}_{nu}"] = f
                        f_out[f"{k}_{nu}"] = [float(fm.real), float(fm.imag)]
                        if verbose:
                            print(f"   k {k} nu {nu}: |S_gen/linS| {float(abs(S / linS).mid()):.3e} {self._fitlabel(k, nu, rs)} "
                                  f"f {mp.nstr(fm, 10)}", flush=True)
                self.publish("d", k, self.d[k])
                self.publish("s", k, self.s[k])
                S, rs = self.S_at(1, k)
                S1[k] = S
                if self.fit_mode == "phys":
                    self.S1_err[k] = rs
                if verbose:
                    print(f"k {k}: S_1^({k}) = {mp.nstr(to_mp(S), 15)}  {self._fitlabel(k, 1, rs)}  sigma = "
                          f"{mp.nstr(to_mp(S / S1[1]), 15)}  [{time.time() - t0:.0f}s]", flush=True)
                sig.append(S / S1[1])
        self.sig_acb = sig
        if self.fit_mode == "phys":
            # error of sigma_n = S^(2n+1)/S^(1) from the fit plateaus of the two multipliers
            e1, a1 = self.S1_err.get(1, 0.0), float(abs(S1[1]).mid())
            self.sigma_err = [self.S1_err.get(2 * n + 1, 0.0) / a1 + float(abs(z).mid()) * e1 / a1
                              for n, z in enumerate(sig)]
        return [to_mp(z) for z in sig], f_out

    def _fitlabel(self, k, nu, rs):
        if self.fit_mode == "phys":
            Mf, err, rel = self.fit_info[f"{k}_{nu}"]
            return f"fit Mf {Mf} err {err:.1e} resid {rel:.1e}"
        return f"fit resid {rs:.1e}"


def _opt(argv, name, default=None, conv=str):
    if name in argv:
        i = argv.index(name)
        v = conv(argv[i + 1])
        del argv[i:i + 2]
        return v
    return default


def main(argv):
    argv = list(argv)
    procs = _opt(argv, "--procs", 1, int)
    fit = _opt(argv, "--fit", "legacy")
    mf = _opt(argv, "--mf", "4:12")
    fit_step = _opt(argv, "--fit-step", 1, int)
    phys_prec = _opt(argv, "--phys-prec", None, int)
    out = _opt(argv, "--out", None)
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hp"))
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    wf, K, J = float(argv[1]), int(argv[2]), int(argv[3])
    prec = int(argv[4]) if len(argv) > 4 else 320
    out = out or f"sigF_w{wf:.4f}_K{K}_J{J}.json"
    require_raw(out, allow=allow)       # a raw-data folder, or --allow_unraw
    t0 = time.time()
    kw = {}
    if fit != "legacy":
        kw = dict(fit=fit, mf=tuple(int(x) for x in mf.split(":")), fit_step=fit_step, phys_prec=phys_prec)
    A = AmpPar(wf, K, J, prec=prec, procs=procs, **kw)
    sig, f = A.run()
    secs = time.time() - t0
    tm = {k: round(v, 1) for k, v in A.timing.items()}
    print(f"total {secs:.0f}s with {procs} process(es); products {tm['products']}s, fit setup {tm['fit_setup']}s, "
          f"fits {tm['fits']}s", flush=True)
    rec = {"wf": wf, "K": K, "J": J, "prec": prec, "sigma": [[float(z.real), float(z.imag)] for z in sig],
           "sigma_str": [[mp.nstr(z.real, 20), mp.nstr(z.imag, 20)] for z in sig], "f": f,
           "sigma_rad": [max(float(z.real.rad()), float(z.imag.rad())) for z in A.sig_acb],
           "procs": procs, "seconds": round(secs, 1), "timing": tm, "order_times": A.order_times}
    if fit != "legacy":
        rec.update({"fit": fit, "mf": list(A.mf), "fit_step": A.fit_step, "phys_prec": A.phys_prec,
                    "sigma_err": A.sigma_err, "fit_info": A.fit_info})
    json.dump(rec, open(out or f"sigF_w{wf:.4f}_K{K}_J{J}.json", "w"))


if __name__ == "__main__":
    main(sys.argv)
