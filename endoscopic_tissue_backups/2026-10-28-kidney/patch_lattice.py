"""Require excisions to part through the shell too, and extend the excision test with a tethered case."""
from pathlib import Path

root = Path(r"D:\Simulation\genesis-world\endoscopic_tissue")
scenario = root / "scenario.py"
s = scenario.read_text(encoding="utf8")
replacements = [
    ("""    gripped_pieces: int = 0
    tumor_nodes_left: int = 0
""",
     """    gripped_pieces: int = 0
    tumor_nodes_left: int = 0
    # Material pieces counted over every simulated link, the shell around the
    # material included: a piece cut off but still tied to the rest through
    # the shell counts once with it.
    lattice_pieces: int = 0
"""),
    ('''    def material_pieces(self, neighbors):
        """Return the piece of every node: material nodes linked through material nodes share their piece."""
        inside = np.flatnonzero(self.inside)
        rows = np.repeat(inside, 6)
        cols = neighbors[inside].reshape(-1)
        keep = (cols >= 0) & self.inside[np.maximum(cols, 0)]
''',
     '''    def pieces(self, neighbors, members):
        """Return the piece of every node: nodes of the mask `members` linked through intact links among them share
        their piece."""
        ids = np.flatnonzero(members)
        rows = np.repeat(ids, 6)
        cols = neighbors[ids].reshape(-1)
        keep = (cols >= 0) & members[np.maximum(cols, 0)]
'''),
    ("""                pieces = self.material_pieces(obs.neighbors)
                m.material_components = len(np.unique(pieces[self.inside]))
""",
     """                pieces = self.pieces(obs.neighbors, self.inside)
                m.material_components = len(np.unique(pieces[self.inside]))
                lattice = self.pieces(obs.neighbors, np.ones(len(self.inside), dtype=bool))
                m.lattice_pieces = len(np.unique(lattice[self.inside]))
"""),
    ("""                               and m.material_components == self.scene.pieces and m.uncut_columns == 0""",
     """                               and m.material_components == m.lattice_pieces == self.scene.pieces
                               and m.uncut_columns == 0"""),
    ('''                                   "leaves tumor behind or that the grasper does not hold")''',
     '''                                   "leaves tumor behind, stays tied to the rest or that the grasper does not "
                                   "hold it")'''),
]
for old, new in replacements:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
scenario.write_text(s, encoding="utf8")

test = root / "test_core.py"
s = test.read_text(encoding="utf8")
replacements = [
    ("""        # Three columns of four nodes across the neck, linked along x within rows. The tumor's ellipsoid reaches down
        # past the upper two rows of the inner columns. The traction holds the nodes 7 mm up, so the links between
        # the middle rows cross the shear plane.""",
     """        # Three material columns of four nodes across the neck and a column of shell beside them, linked along x
        # within rows. The tumor's ellipsoid reaches down past the upper two rows of the inner columns. The traction
        # holds the nodes 7 mm up, so the links between the middle rows cross the shear plane."""),
    ("""        rest = np.array([(x, y, 0) for x in (-.1, .1, .5) for y in (2.875, 2.975, 3.075, 3.175)], dtype=np.float32)
        neighbors = np.full((12, 6), -1, dtype=np.int32)
        for node in range(12):
            if node % 4 < 3:
                neighbors[node, 3], neighbors[node + 1, 2] = node + 1, node
            if node < 8:
                neighbors[node, 1], neighbors[node + 4, 0] = node + 4, node""",
     """        rest = np.array([(x, y, 0) for x in (-.1, .1, .5, .6) for y in (2.875, 2.975, 3.075, 3.175)], dtype=np.float32)
        inside = np.repeat((1, 1, 1, 0), 4)
        neighbors = np.full((16, 6), -1, dtype=np.int32)
        for node in range(16):
            if node % 4 < 3:
                neighbors[node, 3], neighbors[node + 1, 2] = node + 1, node
            if node < 12:
                neighbors[node, 1], neighbors[node + 4, 0] = node + 4, node"""),
    ("""        # Every column severed between its middle rows; or the first column one row higher with the step beside it,
        # which leaves a tumor node on the kidney; or the outer column left whole.
        excised = cut((1, 3), (5, 3), (9, 3))
        cases = {"valid": (excised, (3, 7)), "tumor_left": (cut((2, 3), (5, 3), (9, 3), (2, 1)), (3, 7)),
                 "not_held": (excised, (0, 4)), "attached": (cut((1, 3), (5, 3)), (3, 7))}""",
     """        # Every column severed between its middle rows; or the first column one row higher with the step beside it,
        # which leaves a tumor node on the kidney; or the outer material column, or the shell beside it, left whole.
        excised = cut((1, 3), (5, 3), (9, 3), (13, 3))
        cases = {"valid": (excised, (3, 7)), "tumor_left": (cut((2, 3), (5, 3), (9, 3), (13, 3), (2, 1)), (3, 7)),
                 "not_held": (excised, (0, 4)), "attached": (cut((1, 3), (5, 3), (13, 3)), (3, 7)),
                 "tethered": (cut((1, 3), (5, 3), (9, 3)), (3, 7))}"""),
    ("""                flags = np.zeros(12)""", """                flags = np.zeros(16)"""),
    ("""                sequence = Sequence(initial, scissors_settings(), scene, rest, np.ones(12))""",
     """                sequence = Sequence(initial, scissors_settings(), scene, rest, inside)"""),
    ("""                    self.assertEqual((m.material_components, m.gripped_pieces, m.tumor_nodes_left), (2, 1, 0))""",
     """                    self.assertEqual((m.material_components, m.lattice_pieces, m.gripped_pieces, m.tumor_nodes_left),
                                     (2, 2, 1, 0))"""),
    ("""                    culprit = {"tumor_left": m.tumor_nodes_left == 1, "not_held": m.tumor_nodes_left == 4,
                               "attached": m.material_components == 1}""",
     """                    culprit = {"tumor_left": m.tumor_nodes_left == 1, "not_held": m.tumor_nodes_left == 4,
                               "attached": m.material_components == 1,
                               "tethered": m.material_components == 2 and m.lattice_pieces == 1}"""),
]
for old, new in replacements:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
test.write_text(s, encoding="utf8")
print("patched")
