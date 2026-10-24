#!/bin/sh
# P2: the LC certificate with the proposed corrections (patched copy), then with the |R_c''| bound inflated by 1e20
cd "$(dirname "$0")"
nice -n 10 python3 p2_lc_fix.py 4 73.89 --rA=2e-6 --rW=2e-6 --lip --log=p2_lc_fix.log --lipout=p2_lc_fix_lip.json > p2_lc_fix.out 2>&1
nice -n 10 python3 p2_lc_fix.py 4 73.89 --rA=2e-6 --rW=2e-6 --lip --rcpp=1e20 --log=p2_lc_fix_rcpp.log --lipout=p2_lc_fix_rcpp_lip.json > p2_lc_fix_rcpp.out 2>&1
