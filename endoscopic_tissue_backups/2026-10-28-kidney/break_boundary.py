"""Run the kidney lattice test with a broken crossing search and a broken bed, to check that the test catches both."""
import unittest

import numpy as np

from endoscopic_tissue import kidney, test_core


def linear_boundary(distance, positions, a, b):
    da, db = distance(positions[a]), distance(positions[b])
    t = da / (da - db)
    normals = np.zeros((len(a), 3), dtype=np.float32)
    normals[:, 1] = 1
    return t, normals


name = "test_kidney_surface_lies_on_its_zero_set_and_frees_the_tumor"
for label, patch in (("linear crossings", lambda: setattr(kidney, "implicit_boundary", linear_boundary)),
                     ("bed pins everything", lambda: setattr(kidney.Kidney, "bed", lambda self, p: p[:, 1] < 9))):
    original = (kidney.implicit_boundary, kidney.Kidney.bed)
    patch()
    result = unittest.TextTestRunner(verbosity=0).run(unittest.TestSuite([test_core.CoreTests(name)]))
    print(label, "->", "caught" if result.failures or result.errors else "MISSED")
    kidney.implicit_boundary, kidney.Kidney.bed = original
