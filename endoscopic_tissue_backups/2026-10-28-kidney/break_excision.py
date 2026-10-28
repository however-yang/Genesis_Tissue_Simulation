"""Run the excision acceptance test against broken checks, to see that each break is caught."""
import unittest

import numpy as np

from endoscopic_tissue import kidney, scenario, test_core

name = "test_kidney_excision_leaves_the_whole_tumor_in_the_jaws"
original_weight, original_pieces = kidney.Kidney.tumor_weight, scenario.Sequence.material_pieces
breaks = {
    # The tumor check sees the tumor everywhere it is held: tumor left on the kidney goes unnoticed.
    "tumor ignored": lambda: setattr(kidney.Kidney, "tumor_weight", lambda self, p, w: np.zeros(len(p))),
    # Pieces linked through any node, so an uncut outer column is never noticed... by merging everything.
    "pieces merged": lambda: setattr(scenario.Sequence, "material_pieces", lambda self, n: np.zeros(len(n), int)),
}
for label, patch in breaks.items():
    patch()
    result = unittest.TextTestRunner(verbosity=0).run(unittest.TestSuite([test_core.CoreTests(name)]))
    print(label, "->", "caught" if result.failures or result.errors else "MISSED")
    kidney.Kidney.tumor_weight, scenario.Sequence.material_pieces = original_weight, original_pieces
