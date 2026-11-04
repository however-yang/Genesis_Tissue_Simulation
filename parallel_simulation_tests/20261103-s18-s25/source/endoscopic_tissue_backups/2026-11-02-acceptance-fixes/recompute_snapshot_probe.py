"""Correct only derived material-mask fields; preserve the original report."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from snapshot_step_probe import difference

p = argparse.ArgumentParser()
p.add_argument('folder', type=Path)
args = p.parse_args()
original_path = args.folder/'result.json'
result = json.loads(original_path.read_text(encoding='utf8'))
with np.load(args.folder/'original-step.npz', allow_pickle=False) as data:
    reference = data['positions_m'].copy()
    grid = SimpleNamespace(inside=data['material'].astype(bool), active=data['active'].astype(bool))
previous = []
for item in result['repeats']:
    with np.load(args.folder/f'restored-{item["repeat"]:02}.npz', allow_pickle=False) as data:
        current = data['positions_m'].copy()
    corrected = difference(reference, current, grid)
    if corrected['max_node_mm'] != item['vs_original']['max_node_mm']:
        raise ValueError('Unexpected change to the all-node T3 measurement')
    item['vs_original'] = corrected
    item['vs_restored'] = [difference(array, current, grid) for array in previous]
    previous.append(current)
result['derived_field_correction'] = 'Integer material mask treated as bool; all-node T3 values and verdict unchanged'
out = args.folder/'result-material-mask-corrected.json'
with out.open('x', encoding='utf8') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
print(out)
