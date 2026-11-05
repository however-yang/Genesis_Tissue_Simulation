"""CPU-only count of exact geometric-query reuse; never changes a fixture.

Counts are prospective, not a GPU speedup or physical-correctness claim.
Mode 2 preserves held-volume contact and all original per-corner constraints.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = args.fixture.read_bytes()
    header = np.frombuffer(data, '<i4', count=4)
    magic, n, triangles, vertex_bytes = map(int, header)
    assert magic == 0x46524731 and n > 0 and triangles > 0 and vertex_bytes == 92
    offset = 16

    def take(dtype, count):
        nonlocal offset
        array = np.frombuffer(data, dtype, count=count, offset=offset)
        offset += array.nbytes
        return array

    take('<f4', 9*n)  # current, previous, velocity
    component = take('<i4', n)
    inverse_mass = take('<f4', n)
    support = take('<i4', n)
    take('<i4', n)  # grasp: deliberately not a query exclusion in mode 2
    active = take('<i4', n)
    words = take('<u4', 23*3*triangles).reshape(-1, 23)
    assert offset == len(data)
    integer = words.view('<i4')
    ids, counts = integer[:, :8], integer[:, 8]
    assert np.all((counts >= 0) & (counts <= 8))
    roots = np.full(len(words), -1, dtype=np.int32)
    for i in np.flatnonzero(counts > 0):
        indices = ids[i, :counts[i]]
        assert np.all((indices >= 0) & (indices < n))
        labels = component[indices]
        if np.all(labels == labels[0]):
            roots[i] = labels[0]
    selected = np.flatnonzero((roots >= 0) & ((support[np.maximum(roots, 0)] & 1) == 0)
                              & ((support[np.maximum(roots, 0)] & 2) != 0))
    node_count = int(np.count_nonzero((active != 0) & (inverse_mass > 0)
        & ((support[component] & 1) == 0) & ((support[component] & 2) != 0)))
    buckets, invalid_voxel = {}, 0
    for i in selected:
        voxel = int(integer[i, 9])
        if not 0 <= voxel < n:
            invalid_voxel += 1
            continue
        full = words[i].tobytes()
        count, bindings = buckets.setdefault(voxel, [0, set()])
        buckets[voxel][0] = count + 1
        bindings.add(full)
    old_kept = invalid_voxel + sum(count if len(bindings) > 1 else 1
                                  for count, bindings in buckets.values())
    geometry = np.ascontiguousarray(np.concatenate((words[selected, :9], words[selected, 12:23]), axis=1))
    unique_bindings = len(np.unique(geometry, axis=0))
    record = dict(fixture=str(args.fixture), sha256=hashlib.sha256(data).hexdigest(),
        nodes=n, triangles=triangles, surface_queries=len(selected), node_queries=node_count,
        original_queries=len(selected) + node_count,
        current_full_vertex_voxel_dedup_queries=old_kept + node_count,
        prospective_exact_binding_dedup_queries=unique_bindings + node_count,
        scope='CPU exact-binding count only; no GPU algorithm, physical forces or performance validated',
        constraints_must_remain_per_original_source=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
