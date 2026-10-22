# Low D: how close to D = 3 the solvers get (draft, in progress)

Draft written while the runs are going; the final version replaces it.

## Literature read so far

- Ecker, Ecker, Grumiller & Jechtl, SciPost Phys. 20, 176 (2026), arXiv:2602.10185 (Secs. 1-5 read in the arXiv PDF
  text; Sec. 6.2 and Appendices C, D could not be read: the fetched text ends at Sec. 6). Range 3.05 <= D <= 5.5;
  "numerically too costly to go below D = 3.05"; obstacles: narrow, large-amplitude peaks of Psi near the centre (more
  Fourier modes) and many x points (N_x = 1.55e5 at D = 3.05, 8000 at D >= 4); N_tau = 512 enough for D >~ 3.2, at
  least 1024 below. Claimed accuracy about 1e-6 at D = 4 and at the maximum. Conjecture: Delta -> 0 and gamma -> 0 as
  D -> 3+, gamma faster. Their data repository (github.com/EckerChristian/CriticalSpacetimeCrystal) has Delta and gamma
  per resolution; hp/lowD_ecker.py tabulates it.
- Jalmuzna, Gundlach & Chmaj, PRD 92, 124044 (2015), arXiv:1510.02592: in 2+1 (Lambda < 0 needed) the critical
  solution is CSS with c != 0, modelled by Garfinkle's n = 4 solution; gamma = 8/7 predicted, 1.23(4) measured.
- Bland et al. 2005 (gr-qc/0507088), Table I: gamma(3.5) = 0.349 +- 0.003 (about 1%, from the a* fit range).
