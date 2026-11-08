// capd_m5.cpp -- #189 review: a second validated integrator for the M = 5 flows of lemmas 6b1(5) and 6a(5), with the
// CAPD library (Lohner-type doubleton sets, CAPD 6.1.0). Nothing is shared with solver/proof/validated_int.py,
// lemma6_common.py or lemma6_usys_M.py. The vector field below is transcribed from the (P, V, C) equations of
// proof/A_lemmas_M.md section 1, and both forms the lemmas integrate are derived from that one transcription by the
// exact changes of variables (u = 1/z; D = xi P, S = z Q).
//
// Harmonics: odd mu in {-5, ..., 5} (index (mu + 5)/2), even nu in {-6, ..., 6} (index (nu + 6)/2). Every product is
// projected onto its result's harmonics. With s = i mu w, Q = V + (1 + s) P, W = V + [(C - 1) Q], E = [P P] / 2:
//     2 z dP/dz   = V + [(C-1)Q]                                                      =: Np
//     2 z^2 dV/dz = V + [(C-1)Q] + [(C-1)W] + z (-(2 + s) Q - (1 + s) W + [[C E] P])  =: Nv
//     dC/dz       = [C E] / 2                                                         (CE := [C E])
// u-form (lemma 6b1; state P, V, c~ = kappa (C - delta_nu0), w):  dP/du = -z Np / 2,  dV/du = -Nv / 2,
//     dc~/du = -kappa z^2 CE / 2, z = 1/u.
// xi-form (lemma 6a; state D, S, C, w; real xi):  dD/dxi = P + Np,  dS/dxi = 2 xi Q + Nv / xi + xi (1 + s) Np,
//     dC/dxi = xi CE, with P = D / xi, Q = S / xi^2, V = Q - (1 + s) P.
// Real state layout (dim 39): Re y_k, Im y_k at 2k, 2k + 1 for the 19 complex components in the order of the stored
// files ([P or D] mu ascending, [V or S] mu ascending, [c~ or C] nu ascending), then w (real) at 38.
//
// Paths (one real time s in [0, 1] per leg, as in the original):
//     ray   u = (ra + (rb - ra) s) e^{-3 i pi/4}
//     arc   u = rho e^{-i (3 pi/4)(1 - s)}
//     real  xi = xi1 + (xia - xi1) s
//
// Modes:
//   capd_m5 c0 <start.txt> <path.txt> [--prec=B] [--order=N] [--tol=x] [--sign=-1] [--eps=x]
//       multiprecision C^0 doubleton set (MpC0Rect2Set) x + C r0 (+ B r), started from start.txt; the legs of path.txt;
//       prints the interval hull at every stop ("BOX tag i lo hi", decimal, rounded outward)
//   capd_m5 c1 <start.txt> <path.txt> [--order=N] [--tol=x] [--sign=-1] [--eps=x]
//       double C^1 doubleton set (C1Rect2Set) from the box of start.txt; prints the hull and the monodromy matrix
//       (D Phi over the box, "JAC tag i j lo hi", hex) at every stop
//   --sign=-1 flips the sign of [[C E] P] in Nv and --eps=e replaces the 2 in -(2 + s) Q by 2 (1 + e): the mutations.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "capd/capdlib.h"
#include "capd/mpcapdlib.h"
#include "capd/dynsys/MpStepControl.h"
#include "capd/dynsys/OdeSolver.hpp"
#include "capd/poincare/TimeMap.hpp"

using capd::autodiff::Node;
using std::cerr;
using std::cout;
using std::endl;

static const int NO = 6, NE = 7, NC = 19, DIM = 39, NPAR = 11;
// parameters: 0 cos(3pi/4), 1 sin(3pi/4), 2 ra, 3 rb, 4 rho, 5 3pi/4, 6 kappa, 7 sign of [[CE]P], 8 eps, 9 xi1, 10 xia

struct Cx {
  Node re, im;
};
static Cx operator+(const Cx& a, const Cx& b) { return Cx{a.re + b.re, a.im + b.im}; }
static Cx operator-(const Cx& a, const Cx& b) { return Cx{a.re - b.re, a.im - b.im}; }
static Cx operator*(const Cx& a, const Cx& b) { return Cx{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
static Cx sc(const Cx& a, const Node& s) { return Cx{a.re * s, a.im * s}; }
static Cx sc(const Cx& a, double s) { return Cx{a.re * s, a.im * s}; }
static Cx cz() { return Cx{Node(0.), Node(0.)}; }

static int MU(int k) { return -5 + 2 * k; }
static int NU(int k) { return -6 + 2 * k; }
static int iodd(int mu) { return (mu >= -5 && mu <= 5 && (mu % 2 != 0)) ? (mu + 5) / 2 : -1; }
static int ieven(int nu) { return (nu >= -6 && nu <= 6 && (nu % 2 == 0)) ? (nu + 6) / 2 : -1; }

// [X Y]_mu, X even, Y odd -> odd
static void evenOdd(const Cx X[NE], const Cx Y[NO], Cx out[NO]) {
  for (int a = 0; a < NO; ++a) {
    Cx acc = cz();
    bool any = false;
    for (int b = 0; b < NE; ++b) {
      int j = iodd(MU(a) - NU(b));
      if (j < 0) continue;
      acc = any ? acc + X[b] * Y[j] : X[b] * Y[j];
      any = true;
    }
    out[a] = acc;
  }
}
// [X Y]_nu, X odd, Y odd -> even
static void oddOdd(const Cx X[NO], const Cx Y[NO], Cx out[NE]) {
  for (int b = 0; b < NE; ++b) {
    Cx acc = cz();
    bool any = false;
    for (int a = 0; a < NO; ++a) {
      int j = iodd(NU(b) - MU(a));
      if (j < 0) continue;
      acc = any ? acc + X[a] * Y[j] : X[a] * Y[j];
      any = true;
    }
    out[b] = acc;
  }
}
// [X Y]_nu, X even, Y even -> even
static void evenEven(const Cx X[NE], const Cx Y[NE], Cx out[NE]) {
  for (int b = 0; b < NE; ++b) {
    Cx acc = cz();
    bool any = false;
    for (int c = 0; c < NE; ++c) {
      int j = ieven(NU(b) - NU(c));
      if (j < 0) continue;
      acc = any ? acc + X[c] * Y[j] : X[c] * Y[j];
      any = true;
    }
    out[b] = acc;
  }
}

// the (P, V, C) right-hand sides of section 1: Np, Nv, CE from P, V, C - 1, w, z
static void core(const Cx P[NO], const Cx V[NO], const Cx Cm1[NE], const Node& w, const Cx& z, const Node& sgn,
                 const Node& eps, Cx Np[NO], Cx Nv[NO], Cx CE[NE]) {
  Cx s[NO], Q[NO], CQ[NO], W[NO], CW[NO], E[NE], C[NE], CEP[NO], PP[NE];
  for (int a = 0; a < NO; ++a) {
    s[a] = Cx{Node(0.), w * double(MU(a))};
    Q[a] = V[a] + P[a] + s[a] * P[a];
  }
  evenOdd(Cm1, Q, CQ);
  for (int a = 0; a < NO; ++a) W[a] = V[a] + CQ[a];
  evenOdd(Cm1, W, CW);
  oddOdd(P, P, PP);
  for (int b = 0; b < NE; ++b) E[b] = sc(PP[b], 0.5);
  for (int b = 0; b < NE; ++b) C[b] = (NU(b) == 0) ? Cx{Cm1[b].re + 1., Cm1[b].im} : Cm1[b];
  evenEven(C, E, CE);
  evenOdd(CE, P, CEP);
  for (int a = 0; a < NO; ++a) {
    Np[a] = V[a] + CQ[a];
    Cx two = Cx{(eps + 1.) * 2., Node(0.)};        // the coefficient 2 of -(2 + s) Q (mutation: 2 (1 + eps))
    Cx g = cz() - (two + s[a]) * Q[a] - (Cx{Node(1.), Node(0.)} + s[a]) * W[a] + sc(CEP[a], sgn);
    Nv[a] = V[a] + CQ[a] + CW[a] + z * g;
  }
}

static void unpack(Node in[], Cx X[NO], Cx Y[NO], Cx Z[NE]) {
  for (int a = 0; a < NO; ++a) X[a] = Cx{in[2 * a], in[2 * a + 1]};
  for (int a = 0; a < NO; ++a) Y[a] = Cx{in[2 * (NO + a)], in[2 * (NO + a) + 1]};
  for (int b = 0; b < NE; ++b) Z[b] = Cx{in[2 * (2 * NO + b)], in[2 * (2 * NO + b) + 1]};
}
static void pack(const Cx dX[NO], const Cx dY[NO], const Cx dZ[NE], const Cx& f, Node out[]) {
  Cx all[NC];
  for (int a = 0; a < NO; ++a) all[a] = dX[a] * f;
  for (int a = 0; a < NO; ++a) all[NO + a] = dY[a] * f;
  for (int b = 0; b < NE; ++b) all[2 * NO + b] = dZ[b] * f;
  for (int k = 0; k < NC; ++k) {
    out[2 * k] = all[k].re;
    out[2 * k + 1] = all[k].im;
  }
  out[38] = Node(0.);
}

// u-form at u (complex), z = 1/u, multiplied by du/ds
static void ufield(const Cx& z, const Cx& duds, Node in[], Node out[], Node params[]) {
  Cx P[NO], V[NO], Ct[NE], Cm1[NE], Np[NO], Nv[NO], CE[NE];
  unpack(in, P, V, Ct);
  Node w = in[38];
  Node kap = params[6];
  for (int b = 0; b < NE; ++b) Cm1[b] = Cx{Ct[b].re / kap, Ct[b].im / kap};
  core(P, V, Cm1, w, z, params[7], params[8], Np, Nv, CE);
  Cx dP[NO], dV[NO], dC[NE];
  Cx z2 = z * z;
  for (int a = 0; a < NO; ++a) {
    dP[a] = sc(z * Np[a], -0.5);
    dV[a] = sc(Nv[a], -0.5);
  }
  for (int b = 0; b < NE; ++b) dC[b] = sc(sc(z2 * CE[b], -0.5), kap);
  pack(dP, dV, dC, duds, out);
}

static void rayVF(Node t, Node in[], int, Node out[], int, Node params[], int) {
  Node r = params[2] + (params[3] - params[2]) * t;
  // u = r (c - i s), z = (c + i s) / r with c = cos(3pi/4), s = sin(3pi/4); du/ds = (rb - ra)(c - i s)
  Cx z{params[0] / r, params[1] / r};
  Node dr = params[3] - params[2];
  Cx duds{dr * params[0], -(dr * params[1])};
  ufield(z, duds, in, out, params);
}

static void arcVF(Node t, Node in[], int, Node out[], int, Node params[], int) {
  Node th = params[5] * (1. - t);
  Node c = cos(th), s = sin(th);
  Cx z{c / params[4], s / params[4]};                                   // e^{i th} / rho
  Cx duds{params[5] * params[4] * s, params[5] * params[4] * c};        // i (3pi/4) rho e^{-i th}
  ufield(z, duds, in, out, params);
}

static void realVF(Node t, Node in[], int, Node out[], int, Node params[], int) {
  Node xi = params[9] + (params[10] - params[9]) * t;
  Node dxi = params[10] - params[9];
  Cx D[NO], S[NO], C[NE], P[NO], Q[NO], V[NO], Cm1[NE], Np[NO], Nv[NO], CE[NE];
  unpack(in, D, S, C);
  Node w = in[38];
  Node z = xi * xi;
  for (int a = 0; a < NO; ++a) {
    P[a] = Cx{D[a].re / xi, D[a].im / xi};
    Q[a] = Cx{S[a].re / z, S[a].im / z};
    Cx s{Node(0.), w * double(MU(a))};
    V[a] = Q[a] - P[a] - s * P[a];
  }
  for (int b = 0; b < NE; ++b) Cm1[b] = (NU(b) == 0) ? Cx{C[b].re - 1., C[b].im} : C[b];
  core(P, V, Cm1, w, Cx{z, Node(0.)}, params[7], params[8], Np, Nv, CE);
  Cx dD[NO], dS[NO], dC[NE];
  for (int a = 0; a < NO; ++a) {
    Cx s{Node(0.), w * double(MU(a))};
    dD[a] = P[a] + Np[a];
    dS[a] = sc(Q[a], xi * 2.) + Cx{Nv[a].re / xi, Nv[a].im / xi} + sc((Cx{Node(1.), Node(0.)} + s) * Np[a], xi);
  }
  for (int b = 0; b < NE; ++b) dC[b] = sc(CE[b], xi);
  pack(dD, dS, dC, Cx{dxi, Node(0.)}, out);
}

// ---------------------------------------------------------------- input
struct Leg {
  std::string kind, tag;
  std::string a, b;                  // ray: ra, rb; arc: rho; real: xi1, xia (decimal strings)
  std::vector<std::string> stops;    // arc: stop times (exact binary doubles as decimal), each with a tag
  std::vector<std::string> stoptags;
};

struct Start {
  std::vector<std::string> xlo, xhi;                           // DIM
  std::vector<std::vector<std::string>> Clo, Chi;              // DIM x DIM ("" = 0)
  std::vector<std::string> rlo, rhi;                           // DIM
};

static Start readStart(const std::string& fn) {
  std::ifstream f(fn);
  if (!f) { std::cerr << "cannot open " << fn << endl; std::exit(2); }
  Start st;
  st.xlo.assign(DIM, "0"); st.xhi.assign(DIM, "0");
  st.rlo.assign(DIM, "0"); st.rhi.assign(DIM, "0");
  st.Clo.assign(DIM, std::vector<std::string>(DIM, "0"));
  st.Chi.assign(DIM, std::vector<std::string>(DIM, "0"));
  std::string ln;
  int nx = 0;
  while (std::getline(f, ln)) {
    if (ln.empty() || ln[0] == '#') continue;
    std::istringstream is(ln);
    std::string k;
    is >> k;
    if (k == "x") { int i; is >> i; is >> st.xlo[i] >> st.xhi[i]; ++nx; }
    else if (k == "C") { int i, j; is >> i >> j; is >> st.Clo[i][j] >> st.Chi[i][j]; }
    else if (k == "r0") { int j; is >> j; is >> st.rlo[j] >> st.rhi[j]; }
  }
  if (nx != DIM) { std::cerr << fn << ": expected " << DIM << " x lines, got " << nx << endl; std::exit(2); }
  return st;
}

static std::vector<Leg> readPath(const std::string& fn) {
  std::ifstream f(fn);
  if (!f) { std::cerr << "cannot open " << fn << endl; std::exit(2); }
  std::vector<Leg> legs;
  std::string ln;
  while (std::getline(f, ln)) {
    if (ln.empty() || ln[0] == '#') continue;
    std::istringstream is(ln);
    Leg L;
    is >> L.kind;
    if (L.kind == "ray" || L.kind == "real") { is >> L.a >> L.b >> L.tag; }
    else if (L.kind == "arc") {
      is >> L.a;
      std::string t, g;
      while (is >> t >> g) { L.stops.push_back(t); L.stoptags.push_back(g); }
    } else { std::cerr << "bad path line: " << ln << endl; std::exit(2); }
    legs.push_back(L);
  }
  return legs;
}

// ---------------------------------------------------------------- interval types
static capd::MpInterval mpI(const std::string& lo, const std::string& hi) {
  using capd::MpFloat;
  return capd::MpInterval(MpFloat(lo, MpFloat::RoundDown), MpFloat(hi, MpFloat::RoundUp));
}
static capd::MpInterval mpD(const std::string& s) { return mpI(s, s); }
static capd::DInterval dI(const std::string& lo, const std::string& hi) {
  // a decimal string rounded outward to double: through a 256-bit MpFloat, then directed conversion
  using capd::MpFloat;
  MpFloat a(lo, MpFloat::RoundDown, 256), b(hi, MpFloat::RoundUp, 256);
  return capd::DInterval(toDouble(a, MpFloat::RoundDown), toDouble(b, MpFloat::RoundUp));
}

static std::string decDown(const capd::MpFloat& x) {
  std::ostringstream o;
  x.put(o, capd::MpFloat::RoundDown, 0, 10, 60);
  std::string s = o.str();
  while (!s.empty() && s.back() == ' ') s.pop_back();
  return s;
}
static std::string decUp(const capd::MpFloat& x) {
  std::ostringstream o;
  x.put(o, capd::MpFloat::RoundUp, 0, 10, 60);
  std::string s = o.str();
  while (!s.empty() && s.back() == ' ') s.pop_back();
  return s;
}
static std::string hexd(double v) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%a", v);
  return buf;
}

template <class Itv>
static void setParams(capd::map::Map<capd::vectalg::Matrix<Itv, 0, 0>>& f, const Leg& L, const Itv& sgn, const Itv& eps,
                      Itv (*mk)(const std::string&)) {
  Itv pi = Itv::pi();
  Itv a34 = pi * 3 / 4;
  Itv par[NPAR] = {cos(a34), sin(a34), Itv(0), Itv(0), Itv(0), a34, Itv(100), sgn, eps, Itv(0), Itv(0)};
  if (L.kind == "ray") { par[2] = mk(L.a); par[3] = mk(L.b); }
  if (L.kind == "arc") par[4] = mk(L.a);
  if (L.kind == "real") { par[9] = mk(L.a); par[10] = mk(L.b); }
  for (int i = 0; i < NPAR; ++i) f.setParameter(i, par[i]);
}

static double radMp(const capd::MpInterval& x) {
  using capd::MpFloat;
  MpFloat d = x.rightBound() - x.leftBound();
  return toDouble(d, MpFloat::RoundUp) / 2;
}

static capd::DInterval dmk(const std::string& s) { return dI(s, s); }
static const int progress = std::getenv("CAPD_M5_PROGRESS") ? std::atoi(std::getenv("CAPD_M5_PROGRESS")) : 0;
static double mid_of(const capd::MpInterval& t) { return toDouble(t.leftBound(), capd::MpFloat::RoundNearest); }
static double radAll(const capd::MpIVector& v) {
  double r = 0;
  for (int i = 0; i < 38; ++i) r = std::max(r, radMp(v[i]));
  return r;
}

// ---------------------------------------------------------------- C^0, multiprecision
static int runC0(const Start& st, const std::vector<Leg>& legs, int prec, int order, double tol, double sgnv, double epsv) {
  using namespace capd;
  MpFloat::setDefaultPrecision(prec);
  MpFloat::setDefaultRndMode(MpFloat::RoundNearest);
  MpIVector x(DIM), r0(DIM);
  MpIMatrix C(DIM, DIM);
  for (int i = 0; i < DIM; ++i) {
    x[i] = mpI(st.xlo[i], st.xhi[i]);
    r0[i] = mpI(st.rlo[i], st.rhi[i]);
    for (int j = 0; j < DIM; ++j) C[i][j] = mpI(st.Clo[i][j], st.Chi[i][j]);
  }
  MpC0Rect2Set set(x, C, r0, MpInterval(0));
  MpInterval sgn(sgnv), eps(epsv);
  // eps is a decimal like 1e-9: build it exactly enough as an interval from its string
  {
    std::ostringstream o;
    o << std::setprecision(17) << epsv;
    eps = mpD(o.str());
  }
  MpIMap ray(rayVF, DIM, DIM, NPAR, 1), arc(arcVF, DIM, DIM, NPAR, 1), real(realVF, DIM, DIM, NPAR, 1);
  auto t00 = std::chrono::steady_clock::now();
  int total = 0;
  auto report = [&](const std::string& tag) {
    MpIVector v = (MpIVector)set;
    double r = 0;
    for (int i = 0; i < 38; ++i) r = std::max(r, radMp(v[i]));
    cout << "STOP " << tag << " steps " << total << " maxrad " << std::scientific << std::setprecision(3) << r << " sec "
         << std::fixed << std::setprecision(1)
         << std::chrono::duration<double>(std::chrono::steady_clock::now() - t00).count() << endl;
    for (int i = 0; i < DIM; ++i) cout << "BOX " << tag << " " << i << " " << decDown(v[i].leftBound()) << " "
                                         << decUp(v[i].rightBound()) << "\n";
    cout.flush();
  };
  report("start");
  for (const Leg& L : legs) {
    MpIMap& f = (L.kind == "ray") ? ray : (L.kind == "arc" ? arc : real);
    setParams<MpInterval>(f, L, sgn, eps, mpD);
    typedef dynsys::OdeSolver<MpIMap, dynsys::MpLastTermsStepControl> Solver;
    Solver solver(f, order);
    if (tol > 0) { solver.setAbsoluteTolerance(tol); solver.setRelativeTolerance(tol); }
    poincare::TimeMap<Solver> tm(solver);
    tm.stopAfterStep(true);
    set.setCurrentTime(MpInterval(0));
    std::vector<std::string> stops = L.stops, tags = L.stoptags;
    if (L.kind != "arc") { stops = {"1"}; tags = {L.tag}; }
    for (size_t k = 0; k < stops.size(); ++k) {
      MpInterval T = mpD(stops[k]);
      if (!(T.leftBound() > 0)) { report(tags[k]); continue; }       // a stop at s = 0
      do {
        tm(T, set);
        ++total;
        if (progress > 0 && total % progress == 0)
          cerr << "  step " << total << " s = " << mid_of(set.getCurrentTime()) << " maxrad "
               << radAll((MpIVector)set) << " "
               << std::chrono::duration<double>(std::chrono::steady_clock::now() - t00).count() << " s" << endl;
      } while (!tm.completed());
      report(tags[k]);
    }
  }
  return 0;
}

// ---------------------------------------------------------------- C^1, double
static int runC1(const Start& st, const std::vector<Leg>& legs, int order, double tol, double sgnv, double epsv) {
  using namespace capd;
  IVector x(DIM);
  for (int i = 0; i < DIM; ++i) x[i] = dI(st.xlo[i], st.xhi[i]);
  C1Rect2Set set(x, DInterval(0));
  DInterval sgn(sgnv), eps;
  {
    std::ostringstream o;
    o << std::setprecision(17) << epsv;
    eps = dI(o.str(), o.str());
  }
  IMap ray(rayVF, DIM, DIM, NPAR, 1), arc(arcVF, DIM, DIM, NPAR, 1), real(realVF, DIM, DIM, NPAR, 1);
  auto t00 = std::chrono::steady_clock::now();
  int total = 0;
  auto report = [&](const std::string& tag) {
    IVector v = (IVector)set;
    IMatrix J = (IMatrix)set;
    double r = 0, rj = 0;
    for (int i = 0; i < 38; ++i) {
      r = std::max(r, v[i].rightBound() - v[i].leftBound());
      for (int j = 0; j < 38; ++j) rj = std::max(rj, J[i][j].rightBound() - J[i][j].leftBound());
    }
    cout << "STOP " << tag << " steps " << total << " maxrad " << std::scientific << std::setprecision(3) << r / 2
         << " jacrad " << rj / 2 << " sec " << std::fixed << std::setprecision(1)
         << std::chrono::duration<double>(std::chrono::steady_clock::now() - t00).count() << endl;
    for (int i = 0; i < DIM; ++i) cout << "HBOX " << tag << " " << i << " " << hexd(v[i].leftBound()) << " "
                                         << hexd(v[i].rightBound()) << "\n";
    for (int i = 0; i < DIM; ++i)
      for (int j = 0; j < DIM; ++j)
        cout << "JAC " << tag << " " << i << " " << j << " " << hexd(J[i][j].leftBound()) << " "
             << hexd(J[i][j].rightBound()) << "\n";
    cout.flush();
  };
  report("start");
  for (const Leg& L : legs) {
    IMap& f = (L.kind == "ray") ? ray : (L.kind == "arc" ? arc : real);
    setParams<DInterval>(f, L, sgn, eps, dmk);
    IOdeSolver solver(f, order);
    if (tol > 0) { solver.setAbsoluteTolerance(tol); solver.setRelativeTolerance(tol); }
    ITimeMap tm(solver);
    tm.stopAfterStep(true);
    set.setCurrentTime(DInterval(0));
    std::vector<std::string> stops = L.stops, tags = L.stoptags;
    if (L.kind != "arc") { stops = {"1"}; tags = {L.tag}; }
    for (size_t k = 0; k < stops.size(); ++k) {
      DInterval T = dmk(stops[k]);
      if (!(T.leftBound() > 0)) { report(tags[k]); continue; }
      do {
        tm(T, set);
        ++total;
        if (progress > 0 && total % progress == 0)
          cerr << "  step " << total << " s = " << set.getCurrentTime().leftBound() << " "
               << std::chrono::duration<double>(std::chrono::steady_clock::now() - t00).count() << " s" << endl;
      } while (!tm.completed());
      report(tags[k]);
    }
  }
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage: capd_m5 c0|c1 start.txt path.txt [--prec=B] [--order=N] [--tol=x] [--sign=-1] [--eps=x]" << endl;
    return 2;
  }
  std::string mode = argv[1];
  int prec = 160, order = 24;
  double tol = 0, sgn = 1, eps = 0;
  for (int i = 4; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--prec=", 0) == 0) prec = std::atoi(a.c_str() + 7);
    else if (a.rfind("--order=", 0) == 0) order = std::atoi(a.c_str() + 8);
    else if (a.rfind("--tol=", 0) == 0) tol = std::atof(a.c_str() + 6);
    else if (a.rfind("--sign=", 0) == 0) sgn = std::atof(a.c_str() + 7);
    else if (a.rfind("--eps=", 0) == 0) eps = std::atof(a.c_str() + 6);
    else { std::cerr << "unknown option " << a << endl; return 2; }
  }
  Start st = readStart(argv[2]);
  std::vector<Leg> legs = readPath(argv[3]);
  cout << "capd_m5 " << mode << " prec " << prec << " order " << order << " tol " << tol << " sign " << sgn << " eps "
       << eps << endl;
  try {
    if (mode == "c0") return runC0(st, legs, prec, order, tol, sgn, eps);
    if (mode == "c1") return runC1(st, legs, order, tol, sgn, eps);
  } catch (std::exception& e) {
    cout << "FAILED: " << e.what() << endl;
    return 4;
  }
  std::cerr << "unknown mode " << mode << endl;
  return 2;
}
