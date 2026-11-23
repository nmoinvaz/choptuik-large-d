#!/usr/bin/env python3
"""Review of #1079: writes the margins table of perturb_table.py into notes/EXCITED_REVIEW_1079.md, between the
markers `<!-- perturb table -->` and `<!-- end perturb table -->`, so that no number of that table is typed by hand.

    python3 review_1079/fill_note.py RAW          from solver/proof
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NOTE = os.path.join(HERE, "..", "..", "..", "notes", "EXCITED_REVIEW_1079.md")
A, B = "<!-- perturb table -->", "<!-- end perturb table -->"

out = subprocess.run([sys.executable, "-I", os.path.join(HERE, "perturb_table.py"), sys.argv[1]], capture_output=True,
                     text=True, check=True).stdout.strip()
s = open(NOTE, encoding="utf-8").read()
i, j = s.index(A), s.index(B)
s = s[:i] + A + "\n" + out + "\n\n" + s[j:]
open(NOTE, "w", encoding="utf-8").write(s)
print(out)
