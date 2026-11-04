"""Recompute benchmark tables and state comparisons from retained raw artifacts."""
import itertools
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    rows=[]
    for f in sorted((ROOT/'runs').glob('*/summary.json')):
        d=json.loads(f.read_text());c=json.loads((f.parent/'configuration.json').read_text())
        stat=lambda k,field='mean':(d[k][field] if d.get(k) else None)
        rows.append(dict(name=f.parent.name,workers=d['workers'],episodes=d['recorded_episodes'],
            cpu_count=min(d['workers'],len(set(c['cpus'].split(',')))),
            passed=d['passed_full_episodes'],complete=d['complete'],instrumented=c['profile'],
            aggregate_steps_s=d['aggregate_steps_s'],valid_episodes_hour=d['valid_episodes_per_hour'],
            per_env_steps_s=[r['steps_s'] for r in d['worker_rates']],
            peak_gpu_mib=stat('gpu_memory_mib','max'),mean_gpu_percent=stat('gpu_utilization_percent'),
            mean_memory_percent=stat('gpu_memory_utilization_percent'),host_cpu_percent=stat('host_cpu_percent'),
            native_step_ms=d['timings_ms'].get('native_step',{}).get('mean'),
            quality_and_export_ms=d['timings_ms'].get('quality_and_export',{}).get('mean'),
            total_step_ms=d['timings_ms'].get('total',{}).get('mean'),
            mps=bool(c['mps_pipe']),stop_after=c['stop_after']))
    (ROOT/'performance.json').write_text(json.dumps(rows,indent=2))
    lines=['| Run | N | Full passes/episodes | Total steps/s | Valid episodes/h | GPU MiB peak | GPU % mean | Native ms | Full step ms |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda n:'—' if n is None else f'{n:.2f}'
    for r in rows:
        lines.append(f"| {r['name'] + (' [INCOMPLETE]' if not r['complete'] else '')} | {r['workers']} | {r['passed']}/{r['episodes']} | {fmt(r['aggregate_steps_s'])} | {fmt(r['valid_episodes_hour'])} | {fmt(r['peak_gpu_mib'])} | {fmt(r['mean_gpu_percent'])} | {fmt(r['native_step_ms'])} | {fmt(r['total_step_ms'])} |")
    (ROOT/'performance.md').write_text('\n'.join(lines)+'\n')
    checks=[]
    reference=ROOT/'runs/plain-1-r0/worker-000/episode-000/final-state.npz'
    for path in sorted((ROOT/'runs').glob('*/worker-*/episode-*/final-state.npz')):
        with np.load(path) as z:
            neighbors=z['neighbors'];n=len(z['nodes_cm']);valid=(neighbors>=0)&(neighbors<n)
            record={'path':str(path.relative_to(ROOT)),'nodes':n,'finite':bool(np.isfinite(z['nodes_cm']).all()),
                'neighbor_ids_in_range':bool(((neighbors==-1)|valid).all()),
                'max_material_displacement_mm':float(np.linalg.norm((z['nodes_cm']-z['rest_cm'])[z['inside'].astype(bool)],axis=1).max()*10)}
            # Full graph equality is reported, not required: CUDA atomic ordering may diverge after cuts.
            if (reference.exists() and (path.parent/'result.json').exists()
                    and json.loads((path.parent/'result.json').read_text()).get('completed_requested_stages')
                    and json.loads((path.parent/'result.json').read_text()).get('full_script')):
                with np.load(reference) as ref:
                    if z['rest_cm'].shape==ref['rest_cm'].shape and np.array_equal(z['rest_cm'],ref['rest_cm']):
                        delta=(z['nodes_cm']-ref['nodes_cm'])*10
                        record.update(reference_geometry_equal=True,max_node_delta_mm=float(np.linalg.norm(delta,axis=1).max()),
                            rms_node_delta_mm=float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),
                            neighbors_equal=bool(np.array_equal(z['neighbors'],ref['neighbors'])),
                            max_material_node_delta_mm=float(np.linalg.norm(delta[z['inside'].astype(bool)],axis=1).max()),
                            rms_material_node_delta_mm=float(np.sqrt(np.mean(np.sum(delta[z['inside'].astype(bool)]**2,axis=1)))))
            checks.append(record)
    (ROOT/'state-comparisons.json').write_text(json.dumps(checks,indent=2))
    print('\n'.join(lines))

if __name__=='__main__':main()
