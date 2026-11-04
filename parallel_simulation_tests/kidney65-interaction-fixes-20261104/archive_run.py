"""Archive small, reproducible summaries; retain all failed runs and raw outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--platform',required=True)
    args=parser.parse_args()
    if not args.name.replace('-','').replace('_','').isalnum():
        parser.error('Invalid run name')
    root=Path(__file__).resolve().parent
    folder=root/'outputs'/args.name
    result={'run':args.name,'platform':args.platform,'source_sha256':{}}
    for filename,key in [('report.json','report'),('scene-config.json','scene'),
                         ('training-stress.json','stress'),('interference.json','interference')]:
        path=folder/filename
        if path.exists():
            data=path.read_bytes()
            result[key]=json.loads(data)
            result['source_sha256'][filename]=hashlib.sha256(data).hexdigest()
    path=folder/'native-timing.json'
    if path.exists():
        data=path.read_bytes();calls=json.loads(data)['calls']
        # Modern runs explicitly mark replay calls. Refuse to guess which calls
        # are replay in historical files lacking the phase field.
        if any('phase' not in row for row in calls):
            raise ValueError('Legacy timing needs manual provenance, not inferred phases')
        calls=[row['native_ms'] for row in calls if row['tick']>=0 and row['phase']=='trajectory']
        for key,values in [('original',calls[:1110]),('extra',calls[1110:])]:
            if values:
                result[f'native_{key}_calls']=len(values)
                result[f'native_{key}_mean_ms']=float(np.mean(values))
                result[f'native_{key}_p95_ms']=float(np.percentile(values,95))
                result[f'native_{key}_max_ms']=float(np.max(values))
        result['source_sha256'][path.name]=hashlib.sha256(data).hexdigest()
    result['timing_scope']='Native LCS_Step wall time; excludes init/replay, render, encoding, readback and Python checks. Not minimum FPS.'
    profiled=any('profile_gpu_sections_ms_per_step' in stage
                 for stage in result.get('stress',{}).get('stages',[]))
    result['direct_profile_in_extra_steps']=profiled
    probes=bool(result.get('scene',{}).get('stencil_probes_in_run'))
    result['assembly_probes_in_run']=probes
    result['performance_usable']=not bool(result.get('interference')) and not profiled and not probes
    if probes:result['timing_scope']+=' Private stage assembly probes are present; run a separate normal-throughput comparison.'
    if profiled:
        result['timing_scope']+=' Extra stages use diagnostic direct-kernel profiling; those values are not production performance.'
    target=root/'results'/(args.name+'-summary.json')
    with target.open('x',encoding='utf8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k.startswith('native_') or k in ('run','performance_usable')}))


if __name__=='__main__':
    main()
