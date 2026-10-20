"""Faster, multi-process version of amp_flint.Amp: same problem, same JSON output, results agree within ball radii.

What changed against amp_flint.py (see PARALLEL notes in the docstrings below):

1. Products (the convolutions over (j, mu)).  Order eps^k has only the harmonics mu = -k, -k+2, ..., k, so a product of
   orders k1 + k2 = k is packed with stride k + 1 (index (mu + k_i)/2 in a row) instead of 4K + 1, and only the first
   (J + 1)(k + 1) coefficients are formed (acb_series, a truncated product).  The two d d products of pass_even that are
   mirror images are formed once.  All products of one order are independent; with --procs N they are spread over N
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

    python3 amp_flint_par.py wf K J [prec_bits] [--procs N]
        writes sigF_w<wf>_K<K>_J<J>.json, same keys as amp_flint.py (plus 'procs', 'seconds', 'timing', 'order_times',
        'sigma_rad')
For K >= 13 use J >= 1000 (and compare two J): at J = 500 the nu = 3 fit at order 13 is not converged and sigma_7, sigma_8
blow up (see scan_omega.py).
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


def _worker(conn, prec):
    ctx.prec = prec
    tabs = {}
    while True:
        msg = pickle.loads(conn.recv_bytes())
        op = msg[0]
        if op == "put":
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
        super().__init__(wf, K, J, prec=prec, **kw)
        self.params = dict(wf=wf, K=K, J=J, prec=prec, fit_prec=fit_prec, **kw)
        self.prec, self.procs = prec, max(1, int(procs))
        self.fit_prec = fit_prec or max(1024, 3 * prec)
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
            return self._S_at(nu, k, A, G, sca, tab)
        finally:
            self.timing["fits"] += time.time() - t0

    def _S_at(self, nu, k, A, G, sca, tab):
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
        y = acb_mat([[v] for v in ys])
        c = G * y
        r = A * c - y
        nr = sum(float(abs(r[i, 0]).mid()) ** 2 for i in range(r.nrows())) ** 0.5
        ny = sum(float(abs(v).mid()) ** 2 for v in ys) ** 0.5
        return c[0, 0] / sca[0], (nr / ny if ny else float("nan"))

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
        for k in range(1, self.K + 1):
            if k % 2 == 0:
                self.pass_even(k)
            else:
                self.pass_odd(k)
                if tune and k >= 3:
                    for nu in range(3, k + 1, 2):
                        S, rs = self.S_at(nu, k)
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
                            print(f"   k {k} nu {nu}: |S_gen/linS| {float(abs(S / linS).mid()):.3e} fit resid {rs:.1e} "
                                  f"f {mp.nstr(fm, 10)}", flush=True)
                self.publish("d", k, self.d[k])
                self.publish("s", k, self.s[k])
                S, rs = self.S_at(1, k)
                S1[k] = S
                if verbose:
                    print(f"k {k}: S_1^({k}) = {mp.nstr(to_mp(S), 15)}  fit resid {rs:.1e}  sigma = "
                          f"{mp.nstr(to_mp(S / S1[1]), 15)}  [{time.time() - t0:.0f}s]", flush=True)
                sig.append(S / S1[1])
        self.sig_acb = sig
        return [to_mp(z) for z in sig], f_out


def main(argv):
    procs = 1
    if "--procs" in argv:
        i = argv.index("--procs")
        procs = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    wf, K, J = float(argv[1]), int(argv[2]), int(argv[3])
    prec = int(argv[4]) if len(argv) > 4 else 320
    t0 = time.time()
    A = AmpPar(wf, K, J, prec=prec, procs=procs)
    sig, f = A.run()
    secs = time.time() - t0
    tm = {k: round(v, 1) for k, v in A.timing.items()}
    print(f"total {secs:.0f}s with {procs} process(es); products {tm['products']}s, fit setup {tm['fit_setup']}s, "
          f"fits {tm['fits']}s", flush=True)
    json.dump({"wf": wf, "K": K, "J": J, "prec": prec, "sigma": [[float(z.real), float(z.imag)] for z in sig],
               "sigma_str": [[mp.nstr(z.real, 20), mp.nstr(z.imag, 20)] for z in sig], "f": f,
               "sigma_rad": [max(float(z.real.rad()), float(z.imag.rad())) for z in A.sig_acb],
               "procs": procs, "seconds": round(secs, 1), "timing": tm, "order_times": A.order_times},
              open(f"sigF_w{wf:.4f}_K{K}_J{J}.json", "w"))


if __name__ == "__main__":
    main(sys.argv)
