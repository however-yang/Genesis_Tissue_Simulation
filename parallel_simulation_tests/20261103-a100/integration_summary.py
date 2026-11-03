"""Report observed IPC/recording costs; never label this as learner throughput."""
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parent

def main():
    rows=[]
    for f in sorted((ROOT/'runs').glob('*/integration-result.json')):
        d=json.loads(f.read_text());steps=[r for r in d['events'] if r['op']=='step']
        files=[f for ep in f.parent.glob('worker-*/episode-*') for f in ep.rglob('*') if f.is_file()]
        rows.append({'name':f.parent.name,'passed':d['passed'],'workers':len(list(f.parent.glob('worker-*'))),
            'recorded_transitions':d['recorded_valid_windows'],'wall_s':d['wall_s'],
            'episode_files_mib':sum(f.stat().st_size for f in files)/2**20,
            'sampled_step_responses':len(steps),
            'response_mean_bytes':statistics.mean(r['response_bytes'] for r in steps),
            'mean_roundtrip_ms':statistics.mean(r['roundtrip_ms'] for r in steps),
            'mean_worker_step_and_write_ms':statistics.mean(r['worker_elapsed_ms'] for r in steps),
            'mean_roundtrip_minus_worker_ms':statistics.mean(r['roundtrip_ms']-r['worker_elapsed_ms'] for r in steps),
            'scope':'Short non-cutting integration only; roundtrip includes pipe transport, collector scheduling and resource monitoring; not an isolated IPC latency or trained policy throughput.'})
    (ROOT/'integration-summary.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
