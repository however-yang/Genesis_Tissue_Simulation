"""CPU-only timing/video provenance for the four S15 preflight pairs."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PAIRS = {
    's15-before-contact-pair00': 's15-before-contact',
    's15-v1-preflight-02-first-cut-00': 's15-first-cut',
    's15-v1-preflight-02-squeezing-00': 's15-squeezing',
    's15-v1-preflight-02-regrasp-00': 's15-regrasp',
}


def distribution(rows):
    values = [r['native_ms'] for r in rows]
    assert values and all(v is not None and np.isfinite(v) and v > 0 for v in values)
    return dict(frames=len(values), mean_ms=float(np.mean(values)),
                p95_ms=float(np.percentile(values, 95)), max_ms=float(np.max(values)))


def main():
    result = dict(scope='Four single-pair preflights; not the 20-pair T4 acceptance',
        timing_scope='LCS_Step wall time only; complete QA/rendering/readback excluded',
        video_fps_is_simulation_dt_not_throughput=True, pairs=[])
    for name, label in PAIRS.items():
        inputs = [ROOT/'results'/(name+'.json'),
                  ROOT/'review'/(label+'-comparison')/'decode-report.json']
        case, decode = [json.loads(p.read_text()) for p in inputs]
        item = dict(name=name, fork=case['fork'], preflight_passed=case['passed'],
                    restore_ms=case['restore_ms'], complete_video_decode_passed=decode['passed'], arms={})
        for arm in ('original', 'restored'):
            folder = ROOT/'review'/(label+'-'+arm)
            paths = [folder/'trajectory.jsonl', folder/'report.json']
            inputs.extend(paths)
            rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
            report = json.loads(paths[1].read_text())
            assert report['contiguous_native_frames'] and report['completed_requested_continuation']
            assert len(rows)+1 == decode['decoded_frames'][label+'-'+arm]
            item['arms'][arm] = dict(full=distribution(rows),
                continuation=distribution([r for r in rows if r['step'] > case['fork_native_tick']]),
                outcome=case[arm]['outcome'])
        item['input_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        result['pairs'].append(item)
    with (ROOT/'results/s15-four-fork-preflight-videos.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
