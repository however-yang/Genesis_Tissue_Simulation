"""Compare complete payloads of naive and cached overlapping windows."""
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from endoscopic_tissue.training.data import EpisodeReader, atomic_json, load_arrays


def payload_hash(states, digest):
    for metadata, arrays, topology in states:
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        for group in (arrays, topology):
            for name in sorted(group):
                digest.update(name.encode())
                digest.update(group[name].tobytes())


def run(reader, cached):
    started = time.perf_counter()
    digest = hashlib.sha256()
    count = 0
    with patch('endoscopic_tissue.training.data.load_arrays', wraps=load_arrays) as load:
        if cached:
            for window in reader.windows(4, allow_unvalidated_scope=True):
                payload_hash(window['states'], digest)
                count += 1
        else:
            for start in range(len(reader.transitions)-3):
                rows = reader.transitions[start:start+4]
                assert all(r['after'] is not None for r in rows)
                states = [reader.state(rows[0]['before'])]+[reader.state(r['after']) for r in rows]
                if any(not s[0]['quality']['checked_metrics_passed'] or not s[0]['quality']['within_declared_local_scope'] for s in states):
                    continue
                payload_hash(states, digest)
                count += 1
    return dict(cached=cached, windows=count, payload_sha256=digest.hexdigest(),
                array_decompressions=load.call_count, wall_time_s=time.perf_counter()-started)


reader = EpisodeReader(ROOT/'endoscopic_tissue/outputs/training-foundation-20261101/pilot-01/episode-0002')
# Counterbalanced order; whole payload hashing is included in both timings.
runs = [run(reader, cached) for cached in (False, True, True, False)]
assert len({r['payload_sha256'] for r in runs}) == 1
result = dict(runs=runs, exact_payload_equality=True,
              scope='offline 4-action/5-state windows; not native solver speed or interaction FPS')
atomic_json(Path(__file__).parent/'reader-benchmark.json', result)
print(json.dumps(result, indent=2))
