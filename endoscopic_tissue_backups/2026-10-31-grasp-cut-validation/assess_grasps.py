"""Report current material between the pads, not just retention of the initial bite.

Sliding is allowed. Losing initial sample nodes is not itself a failed grasp.
No clinical force/realism threshold is inferred from these measurements.
"""
import argparse
import json
from pathlib import Path
import sys
from dataclasses import replace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from endoscopic_tissue.native_backend import Settings
from endoscopic_tissue.scenario import SCISSORS_SCENES,Sequence
from endoscopic_tissue.surface_diagnostics import load_state


@threadpool_limits.wrap(limits=1,user_api='blas')
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'endoscopic_tissue/outputs/grasp-cut-validation-20261031')
    base=parser.parse_args().root
    reports=[]
    for path in sorted(base.glob('*/report.json')):
        r=json.loads(path.read_text(encoding='utf8'))
        if 'metadata' not in r or r['metadata']['kind']!='grasp' or r['metadata'].get('fixture_valid') is False:continue
        metadata=r['metadata']
        settings=replace(Settings(**metadata['settings']),contact_model=1)
        scene=SCISSORS_SCENES[metadata['scene']]
        trajectory=[json.loads(line) for line in (path.parent/'trajectory.jsonl').read_text(encoding='utf8').splitlines()]
        result=dict(run=path.parent.name,case=metadata['case'],native_contact_model=metadata['settings']['contact_model'],stages=[])
        for stage in r['stages']:
            name=stage['stage']
            if name not in ('close','lift','sustain','reclose','relift','rehold','jaw_open','open_retract','relax'):continue
            file=path.parent/(name+'.npz')
            obs,_=load_state(file)
            d=np.load(file)
            probe=Sequence(obs,settings,scene,d['rest_cm'],d['inside'])
            enclosed,_=probe.between_pads(obs.nodes,obs)
            row=dict(stage=name,current_enclosed_material_nodes=int(np.count_nonzero(enclosed & d['inside'].astype(bool))),
                initial_bite_nodes=stage['end']['bite_nodes'],initial_bite_still_enclosed=stage['end']['held_bite_nodes'],
                jaw_load_n=stage['end']['jaw_load_n'],pad_slip_mm=stage['end']['pad_slip_mm'],
                bite_lift_mm=stage['end']['bite_lift_mm'],anchors=stage['end']['grip_points'],
                tumor_com_mm=stage['end']['tumor_com_mm'])
            samples=[t for t in trajectory if t['stage']==name][-31:]
            if samples and samples[0]['pad_slip_mm'] is not None:
                row['last_second_slip_change_mm']=samples[-1]['pad_slip_mm']-samples[0]['pad_slip_mm']
                row['last_second_min_pad_load_n']=np.min([t['jaw_load_n'] for t in samples],axis=0).tolist()
            result['stages'].append(row)
        reports.append(result)
    (base/'grasp-assessment.json').write_text(json.dumps(reports,indent=2,allow_nan=False),encoding='utf8')
    for r in reports:
        held=next((s for s in r['stages'] if s['stage']=='sustain'),{})
        print(r['run'],json.dumps(held),flush=True)


if __name__=='__main__':main()
