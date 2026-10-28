"""Run the excision test with the shell ignored in the lattice count, to see that the tethered case is caught."""
import unittest

from endoscopic_tissue import scenario, test_core

pieces = scenario.Sequence.pieces
scenario.Sequence.pieces = lambda self, neighbors, members: pieces(self, neighbors, self.inside)
result = unittest.TextTestRunner(verbosity=0).run(
    unittest.TestSuite([test_core.CoreTests("test_kidney_excision_leaves_the_whole_tumor_in_the_jaws")]))
print("shell ignored ->", "caught" if result.failures or result.errors else "MISSED")
