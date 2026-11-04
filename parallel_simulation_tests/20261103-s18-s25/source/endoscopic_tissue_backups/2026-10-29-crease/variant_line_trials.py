"""Variant for a test build: the material line search tries eight step lengths, 1 down to 1/64, instead of 1 down to
1/8 (MaterialLineSearch keeps eight energies).

Usage: python variant_line_trials.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
"""
import sys
from pathlib import Path

assert sys.argv[1] == "apply"
path = Path("solver/material/material_line_search.cuh")
text = path.read_text(encoding="utf-8")
for old, new, count in (("double energy[5]", "double energy[8]", 1), ("double values[5]", "double values[8]", 2),
                        ("k==0?0.f:k==1?1.f:k==2?.5f:k==3?.25f:.125f", "k==0?0.f:ldexpf(1.f,1-k)", 1)):
    assert text.count(old) == count, old
    text = text.replace(old, new)
assert text.count("k<5") >= 6
text = text.replace("k<5", "k<8")
path.write_text(text, encoding="utf-8")
print("applied")
