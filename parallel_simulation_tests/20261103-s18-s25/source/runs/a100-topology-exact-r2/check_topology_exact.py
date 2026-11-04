"""Exact dictionary comparison on adversarial surfaces and every replay frame."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import struct
import sys
import time
import zlib

import numpy as np
import trimesh
from threadpoolctl import threadpool_limits
from topology_exact import topology_statistics as candidate


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('replays', nargs='+', type=Path)
    args = p.parse_args()
    sys.path[:0] = [str(args.release), str(args.release/'endoscopic_tissue_backups/2026-10-31-grasp-cut-validation')]
    from replay_archive import open_replay
    from endoscopic_tissue.native_backend import Surface
    from endoscopic_tissue.surface_diagnostics import topology_statistics as reference
    from endoscopic_tissue.training.data import atomic_json
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, default_replaced=False, criteria_changed=False,
                  numpy_version=np.__version__, trimesh_version=trimesh.__version__,
                  candidate_sha256=sha256(Path(__file__).with_name('topology_exact.py')),
                  reference_sha256=sha256(args.release/'endoscopic_tissue/surface_diagnostics.py'),
                  synthetic_cases=0, replay_cases=[])
    def check(surface, label, timings=None):
        functions = [('reference', reference), ('candidate', candidate)]
        if timings is not None and len(timings['reference']) % 2:
            functions.reverse()
        values = {}
        for name, function in functions:
            start = time.perf_counter()
            values[name] = function(surface)
            elapsed = (time.perf_counter()-start)*1000
            if timings is not None:
                timings[name].append(elapsed)
        if values['reference'] != values['candidate']:
            atomic_json(args.output/'mismatch.json', {'label': label, **values})
            np.savez_compressed(args.output/'mismatch.npz', vertices=surface.vertices,
                                faces=surface.triangles, anchors=surface.anchors)
            raise AssertionError('Exact topology diagnostic mismatch: '+label)
    try:
        with threadpool_limits(limits=1):
            rng = np.random.default_rng(83149)
            verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
            faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
            examples = [(verts, faces), (verts, faces[:3]), (verts, np.r_[faces, faces[:1]]),
                        (verts, np.r_[faces, [[0, 1, 2]]]), (verts, faces[:, ::-1]),
                        (verts, np.empty((0, 3), np.int32)),
                        (np.empty((0, 3)), np.empty((0, 3), np.int32)),
                        (verts, np.r_[faces, [[0, 0, 1]]]), (verts * 1e-10, faces)]
            reversed_one = faces.copy()
            reversed_one[0] = reversed_one[0, ::-1]
            examples.append((verts, reversed_one))
            for index in range(160):
                v = rng.integers(-30, 31, (90, 3)).astype(np.float64) / 100000
                # Half-unit rounding, coincident sides, unused vertices and
                # high coordinate ranges exercise all positional hash paths.
                v += rng.choice([0, .5e-5, -.5e-5, 1e-8], v.shape)
                if index % 3 == 0:
                    v *= 100000
                f = rng.integers(0, 80, (60, 3), dtype=np.int32)
                f[-2:] = f[:2]
                examples.append((v.astype(np.float32 if index % 2 else np.float64), f))
            for index, (v, f) in enumerate(examples):
                anchors = np.zeros((len(v), 4), np.float32)
                anchors[::3, 3] = 1
                surface = Surface(v, np.zeros_like(v), f, anchors)
                check(surface, f'synthetic-{index}')
                result['synthetic_cases'] += 1
            for source in args.replays:
                timings = {'reference': [], 'candidate': []}
                record = dict(source=str(source), frames=0, passed=False)
                result['replay_cases'].append(record)
                with open_replay(source) as stream:
                    pickle.load(stream)
                    while prefix := stream.read(4):
                        step, stage, surface, _ = pickle.loads(zlib.decompress(stream.read(struct.unpack('<I', prefix)[0])))
                        check(surface, f'{source.parent.name}:{step}:{stage}', timings)
                        record['frames'] += 1
                        if record['frames'] % 120 == 0:
                            print(source.parent.name, record['frames'], 'exact', flush=True)
                            atomic_json(args.output/'progress.json', result)
                record.update(passed=True, timings_ms={name: {
                    'mean': float(np.mean(values)), 'median': float(np.median(values)),
                    'p95': float(np.percentile(values, 95))} for name, values in timings.items()})
                atomic_json(args.output/'progress.json', result)
            result['passed'] = True
            result['scope'] = 'Pure read-only diagnostic equivalence; no solver or parameter changes, no end-to-end FPS claim'
        return 0
    except AssertionError as error:
        result['error'] = repr(error)
        print('Rejected equivalence candidate:', error, flush=True)
        return 2
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        atomic_json(args.output/'result.json', result)


if __name__ == '__main__':
    raise SystemExit(main())
