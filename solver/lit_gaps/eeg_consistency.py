"""Consistency of arXiv:2611.07007's large-D numbers with ours (#1089).

Inputs, all printed below with their sources:
  EEG26b: Table I's D -> inf entry 1.643911 (SM), mu_0 0.130818 and mu_1 -0.1449 (Eq. (7)), Delta_1 10.37 (Eq. (9)).
  Ours:   Delta_inf 1.643893746 +- 1.5e-8 (README), Delta_1 10.3753 (notes/FIRST_ORDER.md, fit row), the tuned D = inf
          profile F (notes/SECOND_ORDER.md 1.2, harmonics nu <= 7, b = sum 2 Re(c_nu e^{i nu th})).
Relations used:
  Delta = 4 pi mu / (1 - gbar)       (EEG26b Eq. (3); notes/RT_FORMULATION.md 2, gbar = (n-1)<a_h^2 - 1>)
  gbar = N0 eps + O(eps^2), N0 = (Delta_1 - 4 pi mu_1)/(4 pi mu_0) = <beta~_0^2>   (EEG26b Eq. (15))
  beta~ = sqrt(2(1 + 1/n)) F          (notes/SECOND_ORDER.md 1.2), so <beta~_0^2> = 2 <F^2> = 4 sum |c_nu|^2
  sigma_R = (n/2 - 1 + gbar)/(1 - gbar) = n/2 - 1 + N0/2 + O(1/n)   (notes/LARGE_D_BASICS.md 2)
Usage: python3 eeg_consistency.py [--mutate=norm]
  --mutate=norm uses <F^2> = sum |c|^2 (a factor 2 short); the N0 check against EEG's 7.4158 must then fail (exit 1).
"""
import math
import sys

mutate = '--mutate=norm' in sys.argv
dinf_ours, dinf_err = 1.643893746, 1.5e-8
d1_ours = 10.3753
dinf_eeg, mu0_eeg, mu1_eeg, d1_eeg = 1.643911, 0.130818, -0.1449, 10.37
F = {1: complex(0, -1.35959), 3: complex(0.007383, -0.074488), 5: complex(0.000756, -0.004212),
     7: complex(0.0000613, -0.0002377)}

print(f"Table I inf entry - ours: {dinf_eeg - dinf_ours:+.3e} ({(dinf_eeg - dinf_ours) / dinf_err:.0f} x our +-)")
print(f"4 pi mu_0 (EEG Eq. (7)): {4 * math.pi * mu0_eeg:.7f}; its rounding +-{4 * math.pi * 5e-7:.1e}; "
      f"minus ours {4 * math.pi * mu0_eeg - dinf_ours:+.2e}")
print(f"mu_0 implied by our Delta_inf (gbar -> 0): {dinf_ours / (4 * math.pi):.9f}")
sq = sum(abs(c) ** 2 for c in F.values())
N0_ours = (2 if mutate else 4) * sq
N0_eeg = (d1_eeg - 4 * math.pi * mu1_eeg) / (4 * math.pi * mu0_eeg)
print(f"N0 = <beta~_0^2>: ours (tuned F) {N0_ours:.5f}; EEG Eq. (15) from their Delta_1, mu_1, mu_0 {N0_eeg:.5f}")
mu1_ours = (d1_ours - dinf_ours * N0_ours) / (4 * math.pi)
print(f"mu_1 implied by our Delta_1 and N0: {mu1_ours:.5f} (EEG {mu1_eeg})")
print(f"sigma_R - n/2 at D = inf: N0/2 - 1 = {N0_ours / 2 - 1:.3f}; ks >= sigma_R + 7 gives ks = n/2 + {N0_ours / 2 + 6:.1f}")
for D in (263.6, 400, 2000, 10000):
    n = D - 2
    print(f"  D {D:>7}: sigma_R ~ {n / 2 - 1 + N0_ours / 2:.1f}, horizon order ks ~ {math.ceil(n / 2 + N0_ours / 2 + 6)}")
ok = abs(N0_ours - N0_eeg) < 2e-3
print("N0 check " + ("agrees" if ok else "DISAGREES") + (" (mutated)" if mutate else ""))
sys.exit(0 if ok else 1)
