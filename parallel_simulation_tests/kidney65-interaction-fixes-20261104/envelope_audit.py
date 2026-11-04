"""Explicit, fail-closed distinction between a wound and a proxy-envelope release.

The original distance diagnostic remains in the certificate. A release is
exempt only when native provenance, material endpoints, emitted cut points,
and final connectivity independently agree. This never changes native state.
"""
import ctypes
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def capture(native, edge_count):
    flags = np.zeros(edge_count, np.uint8)
    fn = native.LCS_GetEnvelopeReleaseEdges
    fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
    if fn(flags.ctypes.data, len(flags)) != 0:
        raise RuntimeError("Envelope provenance read failed")
    counts = np.zeros(9, np.int32)
    fn = native.LCS_ReadCutWitnesses
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                   ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    if fn(counts.ctypes.data, None, 0, None, 0, None, 0, None, None) != 0:
        raise RuntimeError("Cut witness size query failed")
    if tuple(counts[6:]) != (32, 16, 36) or counts[4] != edge_count:
        raise RuntimeError("Unsupported cut witness layout")
    points = np.zeros((max(1, int(counts[5])), 32), np.uint8)
    edges = np.zeros((int(counts[4]), 4), np.int32)
    rotations = np.zeros((int(counts[3]), 9), np.float32)
    cuts = np.zeros((int(counts[3]), 6), np.float32)
    skin = np.zeros_like(cuts)
    rc = fn(counts.ctypes.data, points.ctypes.data, len(points),
            edges.ctypes.data, len(edges), rotations.ctypes.data, len(rotations),
            cuts.ctypes.data, skin.ctypes.data)
    if rc != 0:
        raise RuntimeError(f"Cut witness read failed: {rc}")
    return dict(flags=flags, counts=counts, points=points[:counts[5]],
                edges=edges, cuts=cuts, skin=skin)


def components(neighbors, active):
    ids = np.flatnonzero(active)
    a, b = np.repeat(ids, 6), neighbors[ids].reshape(-1)
    keep = (b >= 0) & active[np.maximum(b, 0)]
    graph = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])),
                       shape=(len(active), len(active)))
    return connected_components(graph, directed=False)[1]


def certify(grid, neighbors, evidence):
    edges = evidence['edges']
    if not np.array_equal(edges, grid.edges):
        raise ValueError("Witness grid does not match the running scene")
    flags = evidence['flags']
    if flags.shape != (len(edges),) or not np.isin(flags, [0, 1]).all():
        raise ValueError("Invalid release provenance")
    dims = np.asarray(evidence['counts'][:3]) + 1
    if int(np.prod(dims)) != len(grid.inside):
        raise ValueError("Witness dimensions do not match the running scene")
    a = edges[:, 1] + dims[0] * (edges[:, 2] + dims[1] * edges[:, 3])
    axis = edges[:, 0]
    b = a + np.asarray([1, dims[0], dims[0]*dims[1]])[axis]
    ids = np.flatnonzero(flags)
    aa, bb, slot = a[ids], b[ids], 2*axis[ids]+1
    # CutPointGpu.edgeId is an int32 at byte 16 of its 32-byte record.
    emitted = evidence['points'].copy().view('<i4').reshape(-1, 8)[:, 4]
    if np.any((emitted < 0) | (emitted >= len(edges))):
        raise ValueError("Invalid emitted cut-point edge")
    groups = components(neighbors, grid.active != 0)
    material_groups = np.unique(groups[grid.inside != 0])
    air = (grid.inside[aa] == 0) & (grid.inside[bb] == 0)
    severed = neighbors[aa, slot] < 0
    no_points = ~np.isin(ids, emitted)
    separate = (groups[aa] != groups[bb]) & np.isin(groups[aa], material_groups) & np.isin(groups[bb], material_groups)
    valid = air & severed & no_points & separate
    result = dict(released_edges=len(ids), certified_air_releases=int(valid.sum()),
                  material_endpoint_releases=int((~air).sum()),
                  released_edges_emitting_cut_points=int((~no_points).sum()),
                  releases_not_separating_material_components=int((~separate).sum()),
                  invalid_release_edges=ids[~valid].tolist(),
                  certified_edge_ids=ids[valid].tolist())
    # Restore ONLY certified air links in a temporary diagnostic copy. The
    # original material midpoints and all unclassified skin cuts stay intact.
    diagnostic_neighbors = neighbors.copy()
    take = ids[valid]
    diagnostic_neighbors[a[take], 2*axis[take]+1] = b[take]
    diagnostic_neighbors[b[take], 2*axis[take]] = a[take]
    return result, diagnostic_neighbors


def install(run, native, output):
    base = run.Sequence
    original_open = run.Flow.open
    class CertifiedSequence(base):
        def stray_skin_cuts(self, neighbors):
            raw = super().stray_skin_cuts(neighbors)
            evidence = capture(native, len(self.audit_grid.edges))
            result, diagnostic = certify(self.audit_grid, neighbors, evidence)
            unexplained = super().stray_skin_cuts(diagnostic)
            result.update(raw_distance_stray_skin_edges=raw,
                          unexplained_stray_skin_edges=unexplained,
                          certification_version=1)
            np.savez_compressed(output / 'release-provenance.npz', **evidence)
            (output / 'envelope-certificate.json').write_text(json.dumps(result, indent=2)+'\n')
            if result['invalid_release_edges']:
                raise RuntimeError("Uncertified envelope release; inspect envelope-certificate.json")
            return unexplained
    run.Sequence = CertifiedSequence
    def open_with_grid(self, tissue):
        original_open(self, tissue)
        self.sequence.audit_grid = self.grid
    run.Flow.open = open_with_grid
