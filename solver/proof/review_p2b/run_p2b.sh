#!/bin/sh
# P2b review: every run, in order (from solver/proof/review_p2b; one process, nice -n 10). Times on the Mac (M5,
# shared): the series ~4 s, a box-mode LC run ~15 s, a cell-mode LC run ~3 min, Lemma J ~1 min each.
set -e
PY=${PY:-python3}
N="nice -n 10"
$N $PY p2b_series.py --centre            # exact series at the centre (p2b_series_centre.json)
$N $PY p2b_series.py                     # over B8's box (p2b_series.json); Lemma F's orders verified exactly
$N $PY p2b_lc.py --mode=cells --log=p2b_lc_cells.log                          # own closure + Lipschitz (primary)
$N $PY p2b_lc.py --mode=box --crude_d --notesH --log=p2b_lc_box_crude_d_notesH.log   # certificate-like settings
# mutations (each must be caught; p2b_lc_mutations.log)
: > p2b_lc_mutations.log
for m in tauD taub rho4 kernelH NL nod0 gamma3; do
  echo "== --mutate=$m" >> p2b_lc_mutations.log
  $N $PY p2b_lc.py --mode=box --crude_d --notesH --mutate=$m 2>&1 | grep -E 'h_S:|h_B:|d0 =|Lipschitz matrix 40|Error' >> p2b_lc_mutations.log || true
done
$N $PY p2b_lcj.py --h=p2b_lc_cells.json --log=p2b_lcj_cells.log              # Lemma J on p2b's own class
$N $PY p2b_lcj.py --h=cert --tag=cert --log=p2b_lcj_cert.log                  # Lemma J on the certificate's class
: > p2b_lcj_mutations.log
for m in noend omega nodecay shape; do
  echo "== --mutate=$m" >> p2b_lcj_mutations.log
  $N $PY p2b_lcj.py --h=p2b_lc_cells.json --mutate=$m 2>&1 | tail -2 >> p2b_lcj_mutations.log || true
done
# comparison with the certificate (reads its logs only; written after commit d89f5ca)
$N $PY p2b_compare_lc.py --mode=box --crude_d --notesH --certlog=b8m_lc_K4_preP2b.log --certlip=b8m_lc_lip_preP2b.json --log=p2b_compare_lc_box.log
$N $PY p2b_compare_lc.py --mode=cells --certlog=b8m_lc_K4_preP2b.log --certlip=b8m_lc_lip_preP2b.json --log=p2b_compare_lc_cells.log
$N $PY p2b_compare_lc.py --mode=cells --certlog=b8m_lc_K4.log --certlip=b8m_lc_lip.json --log=p2b_compare_lc_cells_F3.log
$N $PY p2b_cert_probe.py > p2b_cert_probe.log
$N $PY p2b_cert_probe.py lcj >> p2b_cert_probe.log
$N $PY p2b_atoms_cmp.py
