"""Which endpoint exponent does the ratio law sigma_k = kappa^k/Gamma(k+beta) imply? Borel transform B(t) = sum k! sigma_k
t^(-k-1) has jump density rho on [0, kappa]; k! sigma_k = int rho(t) t^k dt. Claim: rho(t) = (kappa - t)^(beta-2) kappa^(1-beta)
/Gamma(beta-1) (Mittag-Leffler E_{1,beta}). Check the moments for k = 0..30 at beta = 4.16-1.60i; control: exponent
beta - 1 (the paper's and D_entire's statement) must fail. Exit 1 otherwise."""
import sys
import mpmath as mp
mp.mp.dps = 30
kap, beta = mp.mpc("1.279", "1.081"), mp.mpc("4.16", "-1.60")
def moment(k, p):
    # t = kappa s, s in [0, 1]: int_0^1 (kappa(1-s))^p (kappa s)^k kappa ds, normalised as claimed
    return kap ** (k + p + 1) * mp.beta(k + 1, p + 1) * kap ** (1 - beta) / mp.gamma(beta - 1)
def rel(p):
    return max(abs(moment(k, p) - mp.factorial(k) * kap ** k / mp.gamma(k + beta)) / abs(mp.factorial(k) * kap ** k / mp.gamma(k + beta)) for k in range(31))
e2, e1 = rel(beta - 2), rel(beta - 1)
print(f"density exponent beta-2: max rel moment error {mp.nstr(e2,3)};  beta-1 (control): {mp.nstr(e1,3)}")
ok = e2 < 1e-20 and e1 > 1e-2
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
