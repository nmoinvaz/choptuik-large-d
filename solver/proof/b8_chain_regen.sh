#!/bin/bash
# Regenerates the stored outputs of the B8 part of Theorem B(1)'s chain with the code of this checkout (#890): the
# commands of proof/B8_constants.md section 13 and proof/B_chain.md section 11 from b8m_atoms.py on, without the
# b7_adjoint.py runs (Lemma P), whose inputs do not change. Each output is written under its stored name and in its
# stored layout.
#
#     bash b8_chain_regen.sh [python]        one process at nice 10, about 20 min on an Apple M5
#
# b8m_scan.py and b8m_lc.py append to their --log, so each such log is emptied first and holds this run's records
# only. A mutation log keeps the last lines of each mutated run. A line "exit N" follows the output where the stored
# file has one. The exit status of every command is printed, and the script exits 1 if one is not 0. b8m_scan.py
# exits 0 also where the closure fails: its record then has no field "closes".
cd "$(dirname "$0")" || exit 9
PY=${1:-python3}
BOX="--rA=2e-6 --rW=2e-6"
BAD=0

status() {  # exit status, what ran
    echo "exit $1: $2"
    [ "$1" = 0 ] || BAD=1
}
out() {     # file, command: the output alone
    local f=$1; shift
    nice -n 10 "$PY" "$@" > "$f" 2>&1
    status $? "$* > $f"
}
outx() {    # file, command: the output, then a line with the exit status
    local f=$1 rc; shift
    nice -n 10 "$PY" "$@" > "$f" 2>&1
    rc=$?
    echo "exit $rc" >> "$f"
    status $rc "$* > $f"
}
add() {     # file, command: the output added to the file
    local f=$1; shift
    nice -n 10 "$PY" "$@" >> "$f" 2>&1
    status $? "$* >> $f"
}
last() {    # lines, file, command: the last lines of the output added to the file
    local n=$1 f=$2 rc; shift 2
    nice -n 10 "$PY" "$@" 2>&1 | tail -"$n" >> "$f"
    rc=${PIPESTATUS[0]}
    status "$rc" "$* | tail -$n >> $f"
}

# --- B8_constants.md 13: the atoms at the centre and on the box, the closure, (H8)
out b8m_atoms_K4.log b8m_atoms.py 4 0 0
add b8m_atoms_K4.log b8m_atoms.py 4 2e-6 2e-6
: > b8m_scan_K4_box.log                 # ln n0 = 58 does not close: B8(1)'s threshold lies between 58 and 62
out b8m_scan_K4_box.out b8m_scan.py 4 73.89,70,66,62,58 $BOX --log=b8m_scan_K4_box.log
out b8m_h8_K4_L73.89.log b8m_h8.py 4 73.89 --log=b8m_scan_K4_box.log $BOX
: > b8m_scan_K4_k0.85.log
out /dev/null b8m_scan.py 4 104.68 --kappa1=0.85 $BOX --log=b8m_scan_K4_k0.85.log
outx b8m_h8_K4_L104.68_k0.85.log b8m_h8.py 4 104.68 --log=b8m_scan_K4_k0.85.log --kappa1=0.85
: > b8m_scan_K4_fixedZ.log
out /dev/null b8m_scan.py 4 90 --kappa1=0.7308324141810887 $BOX --log=b8m_scan_K4_fixedZ.log

# --- B_chain.md 11: B5(2) without the offset, its threshold
out b5_jump.log b5_jump.py --json=b5_jump.json
outx b5_threshold.log b5_threshold.py

# --- B8 with the offset d0, (H8), B5(2), B7. The scan over d0 does not close at 1e8 and 1e9
: > b8m_scan_K4_d0.log
for d0 in 1e5 1e6 1e7 1e8 1e9; do out /dev/null b8m_scan.py 4 73.89 --d0=$d0 $BOX --log=b8m_scan_K4_d0.log; done
for d0 in 1e6 1e7; do
    : > b8m_scan_K4_d0_$d0.log
    out /dev/null b8m_scan.py 4 73.89 --d0=$d0 $BOX --log=b8m_scan_K4_d0_$d0.log
    outx b8m_h8_K4_L73.89_d0_$d0.log b8m_h8.py 4 73.89 --log=b8m_scan_K4_d0_$d0.log --d0=$d0
    out b5_jump_d0_$d0.log b5_jump.py --h8log=b8m_h8_K4_L73.89_d0_$d0.log --json=b5_jump_d0_$d0.json
done
out b7_ift.log b7_ift.py --adj=b7_adjoint.json --jump=b5_jump_d0_1e7.json --json=b7_ift.json
out b7_ift_mutJD.log b7_ift.py --adj=b7_adjoint.json --jump=b5_jump_d0_1e7.json --mutate=JD
outx b7_ift_d0_1e6.log b7_ift.py --adj=b7_adjoint.json --jump=b5_jump_d0_1e6.json
outx b7_ift_d0_0.log b7_ift.py --adj=b7_adjoint.json --jump=b5_jump.json

# --- Lemma LC, its Lipschitz matrix, the mutated certificate, the numerical checks
: > b8m_lc_K4.log
outx b8m_lc_K4.out b8m_lc.py 4 73.89 $BOX --lip --log=b8m_lc_K4.log
outx b8m_lc_lipcheck.log b8m_lc.py --lipcheck=b8m_lc_lip.json
: > b8m_lc_K4_mut_rho2L.log
: > b8m_lc_K4_mut_rho2L.out
last 2 b8m_lc_K4_mut_rho2L.out b8m_lc.py 4 73.89 $BOX --mutate=rho2L --log=b8m_lc_K4_mut_rho2L.log
outx b8m_lc_numcheck_n1e3.log b8m_lc_numcheck.py 1000 --log=b8m_lc_K4.log
: > b8m_lc_numcheck_mut.log
last 3 b8m_lc_numcheck_mut.log b8m_lc_numcheck.py 1000 --log=b8m_lc_K4.log --mutate=d0
last 3 b8m_lc_numcheck_mut.log b8m_lc_numcheck.py 1000 --log=b8m_lc_K4.log --mutate=weight
last 3 b8m_lc_numcheck_mut.log b8m_lc_numcheck.py 1000 --log=b8m_lc_K4_mut_rho2L.log --mutlc
echo rho2L-done >> b8m_lc_numcheck_mut.log

# --- Lemma J and its mutations
outx b8m_lcj.log b8m_lcj.py --log=b8m_lc_K4.log --json=b8m_lcj.json
: > b8m_lcj_mutations.log
last 2 b8m_lcj_mutations.log b8m_lcj.py --log=b8m_lc_K4.log --mutate=kernel
last 2 b8m_lcj_mutations.log b8m_lcj.py --log=b8m_lc_K4.log --mutate=Om

echo "b8_chain_regen: $([ $BAD = 0 ] && echo "every command exits 0" || echo "SOME COMMAND DID NOT EXIT 0")"
exit $BAD
