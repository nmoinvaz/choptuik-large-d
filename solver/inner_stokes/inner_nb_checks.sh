#!/bin/sh
# The compiled backend's checks and their mutation controls, with logs in logs/ (run from solver/inner_stokes/).
# Each line's exit status is appended to its log; the controls must end with "exit 1".
cd "$(dirname "$0")"
nice -n 10 python3 -u inner_nb.py capcheck > logs/inner_nb_capcheck.log 2>&1; echo "exit $?" >> logs/inner_nb_capcheck.log
INNER_NB_MAXFEV=0 nice -n 10 python3 -u inner_nb.py capcheck > logs/inner_nb_capcheck_mutate.log 2>&1; echo "exit $?" >> logs/inner_nb_capcheck_mutate.log
nice -n 10 python3 -u inner_nb.py check 1 3 5 7 > logs/inner_nb_check.log 2>&1; echo "exit $?" >> logs/inner_nb_check.log
nice -n 10 python3 -u inner_nb.py check 3 5 --mutate=sign > logs/inner_nb_check_mutate.log 2>&1; echo "exit $?" >> logs/inner_nb_check_mutate.log
nice -n 10 python3 -u hier_nb.py selftest > logs/hier_nb_selftest.log 2>&1; echo "exit $?" >> logs/hier_nb_selftest.log
nice -n 10 python3 -u hier_nb.py selftest --mutate=radius > logs/hier_nb_selftest_mutate.log 2>&1; echo "exit $?" >> logs/hier_nb_selftest_mutate.log
