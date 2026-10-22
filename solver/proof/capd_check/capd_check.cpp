// capd_check.cpp -- an independent check, with the CAPD library, of the one-harmonic (M = 1) certificate of
// proof/A_certify.md section 9: validated integration of the inner system from xi0 = 0.1 e^{3 i pi/8} along the ray
// to |xi| = 0.3 and the arc to the real axis, then a Krawczyk test for J_D(0.3) = D+_{+1} - conj(D+_{-1}) = 0, the
// D-component of the jump (the selection condition needs lemmas 6a and 6b of proof/A_review.md on top), mutation
// controls, and enclosures of the other jump components at the certified zero.
//
// Nothing here is shared with validated_int.py, fixed_point.py or krawczyk.py except the start data of step (i),
// which export_start.py writes to data/ as outward-rounded intervals, and kappa, used only for the second, kappa-scaled
// form of the test. The vector field is written below from the
// equations of proof/A_certify.md section 2. The integration is CAPD's Lohner algorithm:
//   - point run at the box centre x^: multiprecision intervals (MPFR, default 128 bits), C^0 doubleton set
//     (MpC0Rect2Set), started from double-double bounds of the start data;
//   - box run over X = x^ + [-h, h]^2: double-precision intervals, C^1 Lohner set (C1Rect2Set, or the
//     Hermite-Obreshkov variant C1HORect2Set with --ho), started from the mean value form of the start data over X.
// The Krawczyk test is written here too.
//
// The system. Unknowns D_mu, S_mu (mu = -1, +1) and C_nu (nu = -2, 0, +2), complex; w = w_inf * wf with
// w_inf = 2 pi / 1.64389375. Every product is projected onto its result's harmonics:
//     dD_mu/dxi = ( [C S]_mu - i mu w xi D_mu ) / xi^2
//     dS_mu/dxi = ( [C (xi dD/dxi)]_mu + [(C E) D]_mu - [C D]_mu - i mu w xi S_mu ) / xi^2,   E = [D D] / 2
//     dC_nu/dxi = [C E]_nu / xi
// State (15 reals): Re, Im of [D-1, D+1, S-1, S+1, C-2, C0, C+2], then wf (constant, so the C^1 part of the
// flow carries d/dwf; A enters only through the start data).
//
// The path, one real time s in [0, 2]:
//     ray  s in [0, 1]:  xi = (r0 + (r1 - r0) s) e^{i a},   a = 3 pi / 8, r0 = 0.1, r1 = 0.3
//     arc  s in [1, 2]:  xi = r1 e^{i a (2 - s)}
// and dY/ds = F(xi(s), Y) xi'(s). One set object is carried through both pieces.
//
// Usage:  capd_check <start_point.txt> [<start_box.txt> ...] [options]   (one point run, then one box run per box file)
//   --prec=B        MPFR precision of the point run in bits (default 128); --prec=0: point run in double instead
//   --mp-order=N    Taylor order of the point run (default 40)
//   --mp-tol=x      step-size tolerance of the point run (default 1e-28; CAPD's default at 128 bits is 1e-41)
//   --order=N       Taylor order of the double runs (default 20)
//   --ho            C1HORect2Set instead of C1Rect2Set for the box run
//   --double-point  also run the point in double precision with the C^1 set (derivatives at x^, for comparison)
//   --save-point=f  save the jump functionals at x^ from the point run;  --load-point=f  use them, skip the point run
//   --enclosures=d  write the Krawczyk inputs (F(x^), DF over X) as exact hex bounds to d/capd_enclosures_<h>.json
//   --rho=0.2       the lower path: ray to |xi| = 0.2, arc at 0.2, then the real axis from 0.2 to 0.3 (s in [2, 3])

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

using namespace capd;
using capd::autodiff::Node;
using std::cout;
using std::endl;

// ---------------------------------------------------------------- complex arithmetic on CAPD autodiff nodes
struct Cx {
  Node re, im;
};
static Cx operator+(const Cx& a, const Cx& b) { return Cx{a.re + b.re, a.im + b.im}; }
static Cx operator-(const Cx& a, const Cx& b) { return Cx{a.re - b.re, a.im - b.im}; }
static Cx operator*(const Cx& a, const Cx& b) { return Cx{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
static Cx scale(const Cx& a, const Node& s) { return Cx{a.re * s, a.im * s}; }
static Cx scale(const Cx& a, double s) { return Cx{a.re * s, a.im * s}; }
static Cx timesI(const Cx& a) { return Cx{-a.im, a.re}; }

// projected products. Odd vectors X[0] = X_{-1}, X[1] = X_{+1}; even vectors C[0] = C_{-2}, C[1] = C_0, C[2] = C_{+2}
static void evenTimesOdd(const Cx C[3], const Cx X[2], Cx out[2]) {
  out[0] = C[0] * X[1] + C[1] * X[0];   // nu + mu' = -1: (-2, +1), (0, -1)
  out[1] = C[1] * X[1] + C[2] * X[0];   // nu + mu' = +1: (0, +1), (+2, -1)
}
static void evenTimesEven(const Cx C[3], const Cx E[3], Cx out[3]) {
  out[0] = C[0] * E[1] + C[1] * E[0];                  // -2: (-2, 0), (0, -2)
  out[1] = C[0] * E[2] + C[1] * E[1] + C[2] * E[0];    //  0: (-2, 2), (0, 0), (2, -2)
  out[2] = C[1] * E[2] + C[2] * E[1];                  // +2: (0, 2), (2, 0)
}

// parameters: 0 w_inf, 1 r0, 2 r1, 3 cos a, 4 sin a, 5 a, 6 rho (the arc radius; rho = r1 on the standard path)
static void field(const Cx& xi, const Cx& dxi, Node in[], Node out[], Node params[]) {
  Cx D[2] = {Cx{in[0], in[1]}, Cx{in[2], in[3]}};
  Cx S[2] = {Cx{in[4], in[5]}, Cx{in[6], in[7]}};
  Cx C[3] = {Cx{in[8], in[9]}, Cx{in[10], in[11]}, Cx{in[12], in[13]}};
  Node w = params[0] * in[14];
  Node n2 = xi.re * xi.re + xi.im * xi.im;
  Cx inv{xi.re / n2, -(xi.im / n2)};          // 1 / xi
  Cx inv2 = inv * inv;                        // 1 / xi^2
  const double mu[2] = {-1., 1.};

  Cx E[3] = {scale(D[0] * D[0], 0.5), D[0] * D[1], scale(D[1] * D[1], 0.5)};
  Cx cE[3];
  evenTimesEven(C, E, cE);
  Cx cS[2], cD[2], cED[2];
  evenTimesOdd(C, S, cS);
  evenTimesOdd(C, D, cD);
  evenTimesOdd(cE, D, cED);

  Cx dD[2], xidD[2], dS[2];
  for (int m = 0; m < 2; ++m) {
    Cx num = cS[m] - scale(timesI(xi * D[m]), w * mu[m]);    // [C S] - i mu w xi D
    dD[m] = num * inv2;
    xidD[m] = num * inv;
  }
  Cx cP[2];
  evenTimesOdd(C, xidD, cP);
  for (int m = 0; m < 2; ++m) {
    Cx num = cP[m] + cED[m] - cD[m] - scale(timesI(xi * S[m]), w * mu[m]);
    dS[m] = num * inv2;
  }
  Cx dC[3];
  for (int k = 0; k < 3; ++k) dC[k] = cE[k] * inv;

  Cx rhs[7] = {dD[0], dD[1], dS[0], dS[1], dC[0], dC[1], dC[2]};
  for (int k = 0; k < 7; ++k) {
    Cx v = rhs[k] * dxi;
    out[2 * k] = v.re;
    out[2 * k + 1] = v.im;
  }
  out[14] = Node(0.);
}

static void rayVF(Node t, Node in[], int, Node out[], int, Node params[], int) {
  Node dr = params[6] - params[1];
  Node rho = params[1] + dr * t;
  Cx xi{rho * params[3], rho * params[4]};
  Cx dxi{dr * params[3], dr * params[4]};
  field(xi, dxi, in, out, params);
}

static void arcVF(Node t, Node in[], int, Node out[], int, Node params[], int) {
  Node th = params[5] * (2. - t);
  Cx xi{params[6] * cos(th), params[6] * sin(th)};
  Cx dxi{params[5] * xi.im, -(params[5] * xi.re)};      // d xi / ds = -i a xi
  field(xi, dxi, in, out, params);
}

static void realVF(Node t, Node in[], int, Node out[], int, Node params[], int) {   // s in [2, 3]: rho -> r1
  Node dr = params[2] - params[6];
  Cx xi{params[6] + dr * (t - 2.), Node(0.)};
  Cx dxi{dr, Node(0.)};
  field(xi, dxi, in, out, params);
}

// ---------------------------------------------------------------- interval types
struct DTypes {
  typedef capd::DInterval Itv;
  typedef capd::IVector Vec;
  typedef capd::IMatrix Mat;
  typedef capd::IMap Map;
  typedef capd::IOdeSolver Solver;
  typedef capd::ITimeMap TimeMap;
  static double dbl(const Itv& x) { return x.leftBound(); }
};
struct MpTypes {
  typedef capd::MpInterval Itv;
  typedef capd::MpIVector Vec;
  typedef capd::MpIMatrix Mat;
  typedef capd::MpIMap Map;
  typedef capd::dynsys::OdeSolver<capd::MpIMap, capd::dynsys::MpLastTermsStepControl> Solver;
  typedef capd::poincare::TimeMap<Solver> TimeMap;
  static double dbl(const MpFloat& x) { return toDouble(x, MpFloat::RoundNearest); }
};

template <class Itv> static double mid_d(const Itv& x);
template <> double mid_d(const DInterval& x) { return x.mid().leftBound(); }
template <> double mid_d(const MpInterval& x) { return toDouble(x.mid().leftBound(), MpFloat::RoundNearest); }
template <class Itv> static double rad_d(const Itv& x);     // an upper bound of the radius
template <> double rad_d(const DInterval& x) {
  return std::max((interval(x.rightBound()) - x.mid()).rightBound(), (x.mid() - x.leftBound()).rightBound());
}
template <> double rad_d(const MpInterval& x) {
  MpInterval m = x.mid();
  MpInterval a = MpInterval(x.rightBound()) - m, b = m - MpInterval(x.leftBound());
  return std::max(toDouble(a.rightBound(), MpFloat::RoundUp), toDouble(b.rightBound(), MpFloat::RoundUp));
}
// outward conversion to a double interval
static DInterval toD(const DInterval& x) { return x; }
static DInterval toD(const MpInterval& x) {
  return DInterval(toDouble(x.leftBound(), MpFloat::RoundDown), toDouble(x.rightBound(), MpFloat::RoundUp));
}

template <class Itv>
static std::string fmt(const Itv& x, int prec = 17) {
  std::ostringstream o;
  o << std::setprecision(prec) << mid_d(x) << " +- " << std::setprecision(2) << std::scientific << rad_d(x);
  return o.str();
}
template <class Itv>
static std::string fmtc(const Itv& re, const Itv& im, int prec = 17) {
  std::ostringstream o;
  double r = mid_d(re), i = mid_d(im);
  o << std::setprecision(prec) << r << (i < 0 ? " - " : " + ") << std::fabs(i) << "i  (radii " << std::setprecision(2)
    << std::scientific << rad_d(re) << ", " << rad_d(im) << ")";
  return o.str();
}

// ---------------------------------------------------------------- input
struct Start {
  DInterval A, W, h, hX;
  MpInterval A2, W2;                  // double-double bounds (when present)
  bool box = false, dd = false;
  IVector Y = IVector(14), dA = IVector(14), dW = IVector(14);
  std::vector<MpInterval> Y2 = std::vector<MpInterval>(14);
  DInterval kre, kim;
  bool kappa = false;
};

static double hexd(const std::string& s) { return std::strtod(s.c_str(), nullptr); }
static MpInterval ddInterval(const std::string& l0, const std::string& l1, const std::string& h0, const std::string& h1) {
  MpInterval lo = MpInterval(hexd(l0)) + MpInterval(hexd(l1));
  MpInterval hi = MpInterval(hexd(h0)) + MpInterval(hexd(h1));
  return MpInterval(lo.leftBound(), hi.rightBound());
}

static Start readStart(const std::string& fn) {
  std::ifstream f(fn);
  if (!f) { std::cerr << "cannot open " << fn << endl; std::exit(2); }
  Start st;
  std::string ln;
  int seen = 0, seen2 = 0;
  while (std::getline(f, ln)) {
    auto p = ln.find('#');
    if (p != std::string::npos) ln = ln.substr(0, p);
    std::istringstream is(ln);
    std::vector<std::string> tok;
    std::string t;
    while (is >> t) tok.push_back(t);
    if (tok.empty()) continue;
    if (tok[0] == "xhatA") st.A = DInterval(hexd(tok[1]), hexd(tok[2]));
    else if (tok[0] == "xhatW") st.W = DInterval(hexd(tok[1]), hexd(tok[2]));
    else if (tok[0] == "xhatA2") st.A2 = ddInterval(tok[1], tok[2], tok[3], tok[4]);
    else if (tok[0] == "xhatW2") st.W2 = ddInterval(tok[1], tok[2], tok[3], tok[4]);
    else if (tok[0] == "h") { st.h = DInterval(hexd(tok[1])); st.box = true; }
    else if (tok[0] == "hX") st.hX = DInterval(hexd(tok[1]), hexd(tok[2]));
    else if (tok[0] == "kappa") {
      st.kre = DInterval(hexd(tok[2]), hexd(tok[3]));
      st.kim = DInterval(hexd(tok[4]), hexd(tok[5]));
      st.kappa = true;
    } else if (tok[0] == "Y" || tok[0] == "dA" || tok[0] == "dw") {
      int k = std::atoi(tok[1].c_str());
      IVector& v = tok[0] == "Y" ? st.Y : (tok[0] == "dA" ? st.dA : st.dW);
      v[2 * k] = DInterval(hexd(tok[2]), hexd(tok[3]));
      v[2 * k + 1] = DInterval(hexd(tok[4]), hexd(tok[5]));
      ++seen;
    } else if (tok[0] == "Y2") {
      int k = std::atoi(tok[1].c_str());
      st.Y2[2 * k] = ddInterval(tok[2], tok[3], tok[4], tok[5]);
      st.Y2[2 * k + 1] = ddInterval(tok[6], tok[7], tok[8], tok[9]);
      ++seen2;
    }
  }
  if (seen != 21) { std::cerr << fn << ": expected 21 data lines, got " << seen << endl; std::exit(2); }
  st.dd = (seen2 == 7);
  return st;
}

// ---------------------------------------------------------------- the integration
template <class T>
struct Engine {
  typedef typename T::Itv Itv;
  typedef typename T::Vec Vec;
  typedef typename T::Mat Mat;
  typename T::Map ray, arc, real;
  int order;
  double tol;     // 0: CAPD's default
  int rho100;     // the arc radius in hundredths: 30 = the standard path; 20 = ray to 0.2, arc at 0.2, real axis to 0.3
  bool lower;     // with the real-axis piece s in [2, 3]
  std::string rhoText;
  explicit Engine(int order_, double tol_ = 0, int rho100_ = 30)
      : ray(rayVF, 15, 15, 7, 1), arc(arcVF, 15, 15, 7, 1), real(realVF, 15, 15, 7, 1), order(order_), tol(tol_),
        rho100(rho100_), lower(rho100_ != 30), rhoText("0." + std::to_string(rho100_)) {
    Itv pi = Itv::pi();
    Itv a = pi * 3 / 8;
    Itv winf = 2 * pi * Itv(160000) / Itv(263023);   // 1.64389375 = 263023 / 160000 exactly
    Itv par[7] = {winf, Itv(1) / 10, Itv(3) / 10, cos(a), sin(a), a, Itv(rho100) / 100};
    for (int i = 0; i < 7; ++i) {
      ray.setParameter(i, par[i]);
      arc.setParameter(i, par[i]);
      real.setParameter(i, par[i]);
    }
  }
  template <class SetT>
  int integrate(SetT& set, const char* label) {
    typename T::Solver sray(ray, order), sarc(arc, order), sreal(real, order);
    if (tol > 0) {
      sray.setAbsoluteTolerance(tol); sray.setRelativeTolerance(tol);
      sarc.setAbsoluteTolerance(tol); sarc.setRelativeTolerance(tol);
      sreal.setAbsoluteTolerance(tol); sreal.setRelativeTolerance(tol);
    }
    typename T::TimeMap tray(sray), tarc(sarc), treal(sreal);
    tray.stopAfterStep(true);
    tarc.stopAfterStep(true);
    treal.stopAfterStep(true);
    int steps = 0;
    const bool progress = std::getenv("CAPD_CHECK_PROGRESS") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    auto report = [&]() {
      if (progress)
        cout << "      step " << steps << ", s = " << mid_d(Itv(set.getCurrentTime())) << ", radius " << radius(set) << ", "
             << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s" << endl;
    };
    do { tray(Itv(1.), set); ++steps; report(); } while (!tray.completed());
    cout << "   " << label << ": ray to |xi| = " << rhoText << " in " << steps << " steps, state radius " << radius(set);
    int s1 = steps;
    do { tarc(Itv(2.), set); ++steps; report(); } while (!tarc.completed());
    cout << "; arc to the real axis in " << steps - s1 << " steps, state radius " << radius(set);
    if (lower) {
      int s2 = steps;
      do { treal(Itv(3.), set); ++steps; report(); } while (!treal.completed());
      cout << "; real axis to 0.3 in " << steps - s2 << " steps, state radius " << radius(set);
    }
    cout << endl;
    return steps;
  }
  template <class SetT>
  static double radius(const SetT& set) {
    Vec v = (Vec)set;
    double r = 0;
    for (int i = 0; i < 14; ++i) r = std::max(r, rad_d(v[i]));
    return r;
  }
};

// J_D = D+_{+1} - conj(D+_{-1}) is linear in the state: Re J_D = l_0 . y, Im J_D = l_1 . y. Besides the interval hull,
// it is evaluated on CAPD's doubleton representation of the set (x + C r0 + B r for the state, D + Cjac R0 + Bjac R
// for the monodromy), applying the functional before the interval products. Both are rigorous enclosures; the result
// is their intersection.
// The jump components, J_k = y+_k - conj(y+_{-k}) for real parameters (y-_k = conj(y+_{-k})): index 2c + part,
// c = 0: D (J_D, the one the certificate uses), 1: S, 2: C at nu = +2, 3: C at nu = 0 (= 2i Im C_0); part 0 = Re, 1 = Im.
// q = 8, 9 serve only the mutation controls: Im D_{+1} - Im D_{-1} and Re D_{+1} + Re D_{-1}.
static const int NJ = 10;
template <class Vec>
static Vec functional(int q, int dim = 15) {
  Vec l(dim);
  switch (q) {
    case 0: l[2] = 1; l[0] = -1; break;     // Re D_{+1} - Re D_{-1}
    case 1: l[3] = 1; l[1] = 1; break;      // Im D_{+1} + Im D_{-1}
    case 2: l[6] = 1; l[4] = -1; break;     // Re S_{+1} - Re S_{-1}
    case 3: l[7] = 1; l[5] = 1; break;      // Im S_{+1} + Im S_{-1}
    case 4: l[12] = 1; l[8] = -1; break;    // Re C_{+2} - Re C_{-2}
    case 5: l[13] = 1; l[9] = 1; break;     // Im C_{+2} + Im C_{-2}
    case 6: break;                          // Re (C_0 - conj C_0) = 0
    case 7: l[11] = 2; break;               // Im (C_0 - conj C_0) = 2 Im C_0
    case 8: l[3] = 1; l[1] = -1; break;     // Im D_{+1} - Im D_{-1}   (controls only)
    case 9: l[2] = 1; l[0] = 1; break;      // Re D_{+1} + Re D_{-1}   (controls only)
  }
  return l;
}
template <class Itv>
static Itv meet(const Itv& a, const Itv& b) {
  Itv r;
  if (!intersection(a, b, r)) { std::cerr << "empty intersection of two enclosures: a bug" << endl; std::exit(3); }
  return r;
}

template <class T>
struct Out {
  typename T::Vec y = typename T::Vec(15);
  typename T::Mat Dphi = typename T::Mat(15, 15);
  typename T::Itv J[NJ];                           // jump components; J[0], J[1] = Re, Im J_D
  typename T::Mat dJ = typename T::Mat(NJ, 15);    // rows l_q^T Dphi (C^1 sets only)
};

template <class T, class SetT>
static void extractC0(const SetT& set, Out<T>& o) {
  typedef typename T::Vec Vec;
  Vec y = (Vec)set;
  const int dim = y.dimension();
  for (int i = 0; i < 15; ++i) o.y[i] = y[i];
  Vec zero(dim);
  for (int p = 0; p < NJ; ++p) {
    Vec l = functional<Vec>(p, dim);
    o.J[p] = meet(l * y, set.evalAffineFunctional(l, zero));
  }
}

template <class SetT>
static void extractC1Hull(const SetT& set, Out<DTypes>& o) {
  o.y = (IVector)set;
  o.Dphi = (IMatrix)set;
  for (int p = 0; p < NJ; ++p) {
    IVector l = functional<IVector>(p);
    o.J[p] = l * o.y;
    for (int j = 0; j < 15; ++j) {
      interval s = 0;
      for (int i = 0; i < 15; ++i) s += l[i] * o.Dphi[i][j];
      o.dJ[p][j] = s;
    }
  }
}
static void extractC1(const C1HORect2Set& set, Out<DTypes>& o) { extractC1Hull(set, o); }
static void extractC1(const C1Rect2Set& set, Out<DTypes>& o) {
  extractC1Hull(set, o);
  const capd::geomset::CenteredDoubletonSet<IMatrix>& c0 = set;
  const capd::geomset::MatrixDoubletonSet<IMatrix>& c1 = set;
  IVector zero(15);
  const IMatrix &D = c1.get_D(), &Cj = c1.get_Cjac(), &R0 = c1.get_R0(), &Bj = c1.get_Bjac(), &R = c1.get_R();
  for (int p = 0; p < NJ; ++p) {
    IVector l = functional<IVector>(p);
    o.J[p] = meet(o.J[p], c0.evalAffineFunctional(l, zero));
    IVector lD(15), lC(15), lB(15);
    for (int j = 0; j < 15; ++j)
      for (int i = 0; i < 15; ++i) { lD[j] += l[i] * D[i][j]; lC[j] += l[i] * Cj[i][j]; lB[j] += l[i] * Bj[i][j]; }
    for (int j = 0; j < 15; ++j) {
      interval s = lD[j];
      for (int k = 0; k < 15; ++k) s += lC[k] * R0[k][j] + lB[k] * R[k][j];
      o.dJ[p][j] = meet(o.dJ[p][j], s);
    }
  }
}

static const char* NAMES[7] = {"D-1", "D+1", "S-1", "S+1", "C-2", "C0 ", "C+2"};

template <class T>
static void printState(const Out<T>& o) {
  for (int k = 0; k < 7; ++k) cout << "   " << NAMES[k] << "(xi1) = " << fmtc(o.y[2 * k], o.y[2 * k + 1]) << endl;
  cout << "   J_D(x^) = " << fmtc(o.J[0], o.J[1], 10) << "   (the hull of the state gives radii "
       << std::setprecision(2) << std::scientific << rad_d(o.y[2] - o.y[0]) << ", " << rad_d(o.y[3] + o.y[1]) << ")"
       << std::defaultfloat << endl;
}

static double seconds(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char* argv[]) {
  int order = 20, mpOrder = 40, prec = 128;
  double mpTol = 1e-28;
  int rho100 = 30;
  bool ho = false, doublePoint = false;
  std::string savePoint, loadPoint, enclDir;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--order=", 0) == 0) order = std::atoi(a.c_str() + 8);
    else if (a.rfind("--mp-order=", 0) == 0) mpOrder = std::atoi(a.c_str() + 11);
    else if (a.rfind("--prec=", 0) == 0) prec = std::atoi(a.c_str() + 7);
    else if (a.rfind("--mp-tol=", 0) == 0) mpTol = std::atof(a.c_str() + 9);
    else if (a == "--ho") ho = true;
    else if (a.rfind("--rho=", 0) == 0) rho100 = (int)std::lround(std::atof(a.c_str() + 6) * 100);
    else if (a == "--double-point") doublePoint = true;
    else if (a.rfind("--save-point=", 0) == 0) savePoint = a.substr(13);
    else if (a.rfind("--load-point=", 0) == 0) loadPoint = a.substr(13);
    else if (a.rfind("--enclosures=", 0) == 0) enclDir = a.substr(13);
    else if (a.rfind("--", 0) == 0) { std::cerr << "unknown option " << a << endl; return 2; }
    else files.push_back(a);
  }
  if (files.empty()) {
    std::cerr << "usage: capd_check <start_point.txt> [<start_box.txt>] [--prec=B] [--mp-order=N] [--order=N] [--ho] "
                 "[--double-point]" << endl;
    return 2;
  }
  if (prec > 0) {               // before any MpInterval is made (the double-double start data are read at this precision)
    MpFloat::setDefaultPrecision(prec);
    MpFloat::setDefaultRndMode(MpFloat::RoundNearest);
  }
  Start sp = readStart(files[0]);
  if (prec > 0 && !sp.dd) { std::cerr << files[0] << " has no double-double (Y2) lines" << endl; return 2; }
  if (rho100 < 11 || rho100 > 30) { std::cerr << "--rho must be in [0.11, 0.3]" << endl; return 2; }
  cout << "capd_check (CAPD, independent of validated_int.py): xi0 = 0.1 e^{3 i pi/8} -> ray -> |xi| = 0." << rho100
       << " -> arc -> " << (rho100 == 30 ? "xi1 = 0.3" : "real axis -> xi1 = 0.3") << endl;

  // ------------------------------------------------ point run at x^
  DInterval Jhat[NJ];          // the jump components at x^ (J_D = Jhat[0] + i Jhat[1])
  if (!loadPoint.empty()) {
    // the jump functionals at x^ from an earlier point run (--save-point): outward double intervals, exact hex
    std::ifstream f(loadPoint);
    std::string ln;
    int n = 0, rho = -1;
    while (std::getline(f, ln)) {
      std::istringstream is(ln);
      std::string tag;
      is >> tag;
      if (tag == "rho100") is >> rho;
      if (tag == "J") {
        int q;
        std::string lo, hi;
        is >> q >> lo >> hi;
        if (q >= 0 && q < NJ) { Jhat[q] = DInterval(hexd(lo), hexd(hi)); ++n; }
      }
    }
    if (n != NJ || rho != rho100) { std::cerr << loadPoint << ": not a point file for this path" << endl; return 2; }
    cout << "point run at x^: loaded from " << loadPoint << endl;
    cout << "   J_D(x^) = " << fmtc(Jhat[0], Jhat[1], 10) << endl;
    prec = -1;      // no point run below
  }
  if (prec > 0) {
    auto t0 = std::chrono::steady_clock::now();
    Engine<MpTypes> E(mpOrder, mpTol, rho100);
    MpIVector y0(15);
    for (int i = 0; i < 14; ++i) y0[i] = sp.Y2[i];
    y0[14] = sp.W2;
    MpC0Rect2Set set(y0, MpInterval(0.));
    int steps = E.integrate(set, "point run");
    Out<MpTypes> op;
    extractC0(set, op);
    cout << "point run at x^ = (2.5421480475, 0.7769940136): MpC0Rect2Set, " << prec << "-bit MPFR intervals, Taylor order "
         << mpOrder << ", step tolerance " << mpTol << ", " << steps << " steps, " << std::setprecision(3) << seconds(t0) << " s" << endl;
    printState(op);
    for (int q = 0; q < NJ; ++q) Jhat[q] = toD(op.J[q]);
    if (!savePoint.empty()) {
      std::ofstream f(savePoint);
      f << "# capd_check point run at x^ = (2.5421480475, 0.7769940136): MpC0Rect2Set, " << prec << " bits, Taylor order "
        << mpOrder << ", tolerance " << mpTol << ", " << steps << " steps\n";
      f << "# J q lo hi: jump functional q at xi1 (0, 1: Re, Im J_D; 2, 3: J_S; 4, 5: J_C2; 6, 7: J_C0; 8, 9: controls),"
           " outward double bounds in hex\n";
      f << "rho100 " << rho100 << "\n";
      for (int q = 0; q < NJ; ++q) {
        char lo[40], hi[40];
        std::snprintf(lo, sizeof lo, "%a", Jhat[q].leftBound());
        std::snprintf(hi, sizeof hi, "%a", Jhat[q].rightBound());
        f << "J " << q << " " << lo << " " << hi << "   # " << fmt(Jhat[q], 12) << "\n";
      }
      for (int k = 0; k < 7; ++k)
        f << "# " << NAMES[k] << "(xi1) = " << fmtc(op.y[2 * k], op.y[2 * k + 1], 20) << "\n";
      cout << "   saved to " << savePoint << endl;
    }
  }
  IMatrix M0(15, 2);
  for (int i = 0; i < 14; ++i) { M0[i][0] = sp.dA[i]; M0[i][1] = sp.dW[i]; }
  M0[14][0] = 0; M0[14][1] = 1;
  if (prec == 0 || (doublePoint && prec != -1)) {
    auto t0 = std::chrono::steady_clock::now();
    Engine<DTypes> E(order, 0, rho100);
    IVector y0(15);
    for (int i = 0; i < 14; ++i) y0[i] = sp.Y[i];
    y0[14] = sp.W;
    C1Rect2Set set(y0, DInterval(0.));
    int steps = E.integrate(set, "point run (double)");
    Out<DTypes> op;
    extractC1(set, op);
    cout << "point run at x^ in double precision: C1Rect2Set, Taylor order " << order << ", " << steps << " steps, "
         << std::setprecision(3) << seconds(t0) << " s" << endl;
    printState(op);
    IMatrix Vp = op.Dphi * M0;
    cout << "   at x^: dD+_{-1}/d(A, wf) = " << fmtc(Vp[0][0], Vp[1][0], 6) << ",  " << fmtc(Vp[0][1], Vp[1][1], 6) << endl;
    cout << "          dD+_{+1}/d(A, wf) = " << fmtc(Vp[2][0], Vp[3][0], 6) << ",  " << fmtc(Vp[2][1], Vp[3][1], 6) << endl;
    if (prec == 0)
      for (int q = 0; q < NJ; ++q) Jhat[q] = op.J[q];
  }
  DInterval JDre = Jhat[0], JDim = Jhat[1];
  DInterval gre, gim;
  if (sp.kappa) {
    gre = sp.kre * JDre - sp.kim * JDim;
    gim = sp.kim * JDre + sp.kre * JDim;
    cout << "   g(x^) = kappa J_D = " << fmtc(gre, gim, 6) << "   (kappa = " << fmtc(sp.kre, sp.kim, 12) << ")" << endl;
  }
  bool allOk = true;
  for (size_t fb = 1; fb < files.size(); ++fb) {
    // ------------------------------------------------ box run over X = x^ + [-h, h]^2
    Start sb = readStart(files[fb]);
    if (!sb.box) { std::cerr << files[fb] << " is not a box file" << endl; return 2; }
    DInterval h = sb.h;
    double wA = sp.A.rightBound() - sp.A.leftBound(), wW = sp.W.rightBound() - sp.W.leftBound();
    DInterval dlt = DInterval(-1, 1) * (h + DInterval(std::max(wA, wW)));    // contains X - x^ for the exact x^
    if (!(dlt.rightBound() <= sb.hX.leftBound())) {
      std::cerr << "the start box x^ +- hX does not cover X" << endl;
      return 2;
    }
    // start set in mean value form: z0(p) in [z0(x^)] + [dz0/dp](X) (p - x^) = x + C r0 with
    //   C columns 0, 1: mid dz0/dA, mid dz0/dwf; r0 = (p - x^, 0, ..., 0); x = [z0(x^)] + ([dz0/dp](X) - mid) (p - x^)
    IVector x(15), r0(15);
    IMatrix C = IMatrix::Identity(15);      // columns 2..14 are unused (their r0 is 0)
    IMatrix MX(15, 2);
    for (int i = 0; i < 14; ++i) { MX[i][0] = sb.dA[i]; MX[i][1] = sb.dW[i]; }
    MX[14][0] = 0; MX[14][1] = 1;
    for (int i = 0; i < 15; ++i) {
      DInterval mA = MX[i][0].mid(), mW = MX[i][1].mid();
      C[i][0] = mA; C[i][1] = mW;
      x[i] = (i < 14 ? sp.Y[i] : sp.W) + (MX[i][0] - mA) * dlt + (MX[i][1] - mW) * dlt;
    }
    r0[0] = dlt; r0[1] = dlt;
    auto t1 = std::chrono::steady_clock::now();
    Engine<DTypes> E(order, 0, rho100);
    Out<DTypes> ob;
    int steps;
    if (ho) {
      C1HORect2Set set(x, C, r0, DInterval(0.));
      steps = E.integrate(set, "box run");
      extractC1(set, ob);
    } else {
      C1Rect2Set set(x, C, r0, DInterval(0.));
      steps = E.integrate(set, "box run");
      extractC1(set, ob);
    }
    IMatrix V = ob.Dphi * MX;     // dz(xi1)/dp over X: chain rule, Dphi enclosed over the whole start set
    cout << "box run over x^ + [-h, h]^2, h = " << std::setprecision(6) << h.leftBound() << ": "
         << (ho ? "C1HORect2Set" : "C1Rect2Set") << ", double, Taylor order " << order << ", " << steps << " steps, "
         << std::setprecision(3) << seconds(t1) << " s" << endl;
    cout << "   dD+_{-1}/d(A, wf) = " << fmtc(V[0][0], V[1][0], 6) << ",  " << fmtc(V[0][1], V[1][1], 6) << endl;
    cout << "   dD+_{+1}/d(A, wf) = " << fmtc(V[2][0], V[3][0], 6) << ",  " << fmtc(V[2][1], V[3][1], 6) << endl;

    // ------------------------------------------------ Krawczyk, twice: on F = (Re J_D, Im J_D) and on g = kappa J_D
    // dJ_D/dp = dD+_{+1}/dp - conj(dD+_{-1}/dp), for real p
    IMatrix JF = ob.dJ * MX;
    for (int j = 0; j < 2; ++j) {
      JF[0][j] = meet(JF[0][j], V[2][j] - V[0][j]);
      JF[1][j] = meet(JF[1][j], V[3][j] + V[1][j]);
    }
    cout << "   dJ_D/d(A, wf) over X = " << fmtc(JF[0][0], JF[1][0], 4) << ",  " << fmtc(JF[0][1], JF[1][1], 4) << endl;
    DInterval xh[2] = {sp.A, sp.W};
    DInterval Kzero[2];          // K(X) of the test on F: it contains the zero
    // K(X') for X' = x^ + [-hp, hp]^2 with hp <= h: J(X) encloses the derivative over X' too
    auto kraw = [&](const DInterval g0[2], const IMatrix& J, double hp, DInterval K[2], double margin[2]) {
      double a = J[0][0].mid().leftBound(), b = J[0][1].mid().leftBound();
      double c = J[1][0].mid().leftBound(), d = J[1][1].mid().leftBound();
      double det = a * d - b * c;
      double Yf[2][2] = {{d / det, -b / det}, {-c / det, a / det}};    // any real matrix is allowed; used as exact
      DInterval hpI(hp);
      DInterval dl = DInterval(-1, 1) * (hpI + DInterval(std::max(wA, wW)));   // contains X' - x^ for the exact x^
      bool ok = true;
      for (int i = 0; i < 2; ++i) {
        DInterval Xi = xh[i] + DInterval(-1, 1) * hpI;
        DInterval t = xh[i] - (DInterval(Yf[i][0]) * g0[0] + DInterval(Yf[i][1]) * g0[1]);
        for (int j = 0; j < 2; ++j) {
          DInterval Mij = DInterval(i == j ? 1. : 0.) - (DInterval(Yf[i][0]) * J[0][j] + DInterval(Yf[i][1]) * J[1][j]);
          t += Mij * dl;
        }
        K[i] = t;
        ok = ok && (K[i].leftBound() > Xi.leftBound()) && (K[i].rightBound() < Xi.rightBound());
        margin[i] = std::min((DInterval(K[i].leftBound()) - Xi.leftBound()).leftBound(),
                             (DInterval(Xi.rightBound()) - K[i].rightBound()).leftBound());
      }
      return ok;
    };
    auto krawczyk = [&](const char* name, const DInterval g0[2], const IMatrix& J, DInterval* Kout) {
      DInterval K[2];
      double margin[2];
      bool ok = kraw(g0, J, h.rightBound(), K, margin);
      cout << "   Krawczyk on " << name << ": J(X) = [[" << fmt(J[0][0], 5) << ", " << fmt(J[0][1], 5) << "], ["
           << fmt(J[1][0], 5) << ", " << fmt(J[1][1], 5) << "]]" << endl;
      cout << "      K(X) = (" << fmt(K[0], 13) << ", " << fmt(K[1], 13) << ")" << endl;
      cout << "      K(X) in int X: " << (ok ? "true" : "false") << "; margins " << std::setprecision(4) << std::scientific
           << margin[0] << ", " << margin[1] << " (h = " << h.leftBound() << ")" << std::defaultfloat << endl;
      if (ok && Kout) {
        // the tightest sub-box X' = x^ +- hp (hp = h 0.95^k) that still passes: a sharper enclosure of the same zero
        double hp = h.rightBound(), best = hp;
        DInterval Kb[2] = {K[0], K[1]}, Kt[2];
        double mt[2];
        for (int k = 0; k < 400; ++k) {
          hp *= 0.95;
          if (!kraw(g0, J, hp, Kt, mt)) break;
          best = hp; Kb[0] = Kt[0]; Kb[1] = Kt[1];
        }
        Kout[0] = Kb[0]; Kout[1] = Kb[1];
        cout << "      smallest passing sub-box tried: hp = " << std::setprecision(3) << std::scientific << best
             << std::defaultfloat << "; its K(X') encloses the zero:" << endl;
        cout << std::setprecision(17) << "      A in [" << Kb[0].leftBound() << ", " << Kb[0].rightBound() << "],  wf in ["
             << Kb[1].leftBound() << ", " << Kb[1].rightBound() << "]   (radii " << std::setprecision(2) << std::scientific
             << rad_d(Kb[0]) << ", " << rad_d(Kb[1]) << ")" << std::defaultfloat << endl;
      }
      return ok;
    };
    DInterval F0[2] = {JDre, JDim};
    bool okF = krawczyk("F = (Re J_D, Im J_D)", F0, JF, Kzero);
    bool okg = true;
    if (sp.kappa) {
      IMatrix Jg(2, 2);
      for (int j = 0; j < 2; ++j) {
        Jg[0][j] = sp.kre * JF[0][j] - sp.kim * JF[1][j];
        Jg[1][j] = sp.kim * JF[0][j] + sp.kre * JF[1][j];
      }
      DInterval g0[2] = {gre, gim};
      cout << "   g(x^) = (" << fmt(gre, 6) << ", " << fmt(gim, 6) << ")" << endl;
      okg = krawczyk("g = kappa J_D", g0, Jg, nullptr);
    }
    if (okF) {
      // the zero z lies in K(X), a subset of X, so by the mean value theorem every jump component satisfies
      //   J_q(z) in J_q(x^) + [dJ_q/dp](X) (K(X) - x^)
      const char* jn[4] = {"J_D  (D, mu = 1)  ", "J_S  (S, mu = 1)  ", "J_C2 (C, nu = 2)  ", "J_C0 (C, nu = 0)  "};
      cout << "   jump components J_k = y+_k - conj(y+_{-k}) at xi1, at x^ and at the zero:" << endl;
      for (int c = 0; c < 4; ++c) {
        DInterval at[2];
        for (int p = 0; p < 2; ++p) {
          int q = 2 * c + p;
          at[p] = Jhat[q];
          for (int j = 0; j < 2; ++j) at[p] += JF[q][j] * (Kzero[j] - xh[j]);
        }
        cout << "      " << jn[c] << "at x^: " << fmtc(Jhat[2 * c], Jhat[2 * c + 1], 6) << ";  at the zero: |Re| <= "
             << std::setprecision(2) << std::scientific << std::max(std::fabs(at[0].leftBound()), std::fabs(at[0].rightBound()))
             << ", |Im| <= " << std::max(std::fabs(at[1].leftBound()), std::fabs(at[1].rightBound())) << std::defaultfloat
             << (at[0].contains(0.) && at[1].contains(0.) ? "  (contains 0)" : "  (excludes 0)") << endl;
      }
    }
    // mutation controls: change the function (never the enclosures) and check the verdict a sound test must give
    {
      auto pick = [&](int r0_, int r1_, DInterval f[2], IMatrix& J) {
        f[0] = Jhat[r0_]; f[1] = Jhat[r1_];
        for (int j = 0; j < 2; ++j) { J[0][j] = JF[r0_][j]; J[1][j] = JF[r1_][j]; }
      };
      IMatrix Lm(2, 2);                  // exact binary midpoint of J(X), for planting zeros: F - L d moves the zero by ~d
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) Lm[i][j] = JF[i][j].mid();
      // Planting shifts K(X) by Y L d = d (up to rounding), so a zero moved by d_i = +-m_i / 2, with m the clean
      // margins, must still pass, and one moved by 2h must fail.
      DInterval f0[2], Kc[2];
      IMatrix J0(2, 2);
      double mc[2];
      pick(0, 1, f0, J0);
      kraw(f0, J0, h.rightBound(), Kc, mc);
      struct Ctl { const char* name; int kind; double d0, d1; bool expect; };
      const Ctl ctl[5] = {{"C1 jump sign flipped, D+1 + conj(D-1)", 1, 0, 0, false},
                          {"C2 conjugation dropped, D+1 - D-1", 2, 0, 0, false},
                          {"C5 zero planted outside, moved by (2h, 0)", 3, 2 * h.rightBound(), 0, false},
                          {"C6 zero planted inside, moved by (m_A/2, -m_w/2)", 3, mc[0] / 2, -mc[1] / 2, true},
                          {"C7 J_D(x^) widened by 1e3 h", 4, 0, 0, false}};
      const bool cleanOk = mc[0] > 0 && mc[1] > 0;
      bool allMatch = true;
      for (const Ctl& c : ctl) {
        if (c.kind == 3 && c.expect && !cleanOk) {     // planting inside the margins needs a passing clean test
          cout << "   control " << c.name << ": skipped (the clean test fails)" << endl;
          continue;
        }
        DInterval f[2];
        IMatrix J(2, 2);
        if (c.kind == 1) pick(9, 8, f, J);
        else if (c.kind == 2) pick(0, 8, f, J);
        else {
          pick(0, 1, f, J);
          if (c.kind == 3) {
            DInterval d[2] = {DInterval(c.d0), DInterval(c.d1)};
            for (int i = 0; i < 2; ++i) f[i] -= Lm[i][0] * d[0] + Lm[i][1] * d[1];
          } else {
            for (int i = 0; i < 2; ++i) f[i] += DInterval(-1, 1) * 1e3 * h.rightBound();
          }
        }
        DInterval K[2];
        double m[2];
        bool ok = kraw(f, J, h.rightBound(), K, m);
        allMatch = allMatch && (ok == c.expect);
        cout << "   control " << c.name << ": " << (ok ? "pass" : "fail") << " (expected " << (c.expect ? "pass" : "fail")
             << "), margins " << std::setprecision(2) << std::scientific << m[0] << ", " << m[1] << std::defaultfloat
             << endl;
      }
      cout << "   controls: " << (allMatch ? "every verdict as expected" : "A CONTROL GAVE THE WRONG VERDICT") << endl;
      okF = okF && allMatch;
    }
    cout << "RESULT: " << (okF && okg ? "the Krawczyk test passes" : "the Krawczyk test FAILS") << " on X = x^ + [-h, h]^2, h = "
         << std::setprecision(6) << h.leftBound() << endl;
    if (!enclDir.empty()) {
      // the inputs of the Krawczyk test as exact outward double bounds (hex), for an external checker
      std::string tag = files[fb];
      auto sl = tag.find_last_of('/');
      if (sl != std::string::npos) tag = tag.substr(sl + 1);
      if (tag.rfind("start_box_", 0) == 0) tag = tag.substr(10);
      if (tag.size() > 4 && tag.substr(tag.size() - 4) == ".txt") tag = tag.substr(0, tag.size() - 4);
      std::string fn = enclDir + "/capd_enclosures_" + tag + (rho100 == 30 ? "" : "_rho0." + std::to_string(rho100)) + ".json";
      std::ofstream f(fn);
      auto hx = [](double v) { char b[40]; std::snprintf(b, sizeof b, "\"%a\"", v); return std::string(b); };
      auto iv = [&](const DInterval& v) { return "[" + hx(v.leftBound()) + ", " + hx(v.rightBound()) + "]"; };
      f << "{\n  \"producer\": \"solver/proof/capd_check (CAPD 37057e5899bccaa6cd5701057b7e6ea5e232ef8b)\",\n"
        << "  \"map\": \"F(A, wf) = (Re J_D, Im J_D), J_D = D+_{+1}(0.3) - conj(D+_{-1}(0.3)), path: ray arg xi = 3pi/8 to 0."
        << rho100 << (rho100 == 30 ? ", arc at 0.3" : ", arc, real axis to 0.3") << "\",\n"
        << "  \"numbers\": \"every interval is [lo, hi] with lo, hi exact binary doubles in C99 hex\",\n"
        << "  \"xhat\": [\"2.5421480475\", \"0.7769940136\"],\n"
        << "  \"xhat_enclosure\": [" << iv(sp.A) << ", " << iv(sp.W) << "],\n"
        << "  \"h\": " << hx(h.rightBound()) << ",\n"
        << "  \"F_at_xhat\": [" << iv(Jhat[0]) << ", " << iv(Jhat[1]) << "],\n"
        << "  \"DF_over_X\": [[" << iv(JF[0][0]) << ", " << iv(JF[0][1]) << "], [" << iv(JF[1][0]) << ", " << iv(JF[1][1])
        << "]],\n"
        << "  \"kappa\": [" << iv(sp.kre) << ", " << iv(sp.kim) << "],\n"
        << (okF ? "  \"zero_enclosure\": [" + iv(Kzero[0]) + ", " + iv(Kzero[1]) + "],\n" : std::string())
        << "  \"krawczyk_F_passes\": " << (okF ? "true" : "false") << "\n}\n";
      cout << "   enclosures written to " << fn << endl;
    }
    allOk = allOk && okF && okg;
  }
  return allOk ? 0 : 1;
}
