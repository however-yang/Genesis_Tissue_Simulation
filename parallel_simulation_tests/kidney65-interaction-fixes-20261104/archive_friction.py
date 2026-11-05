"""Create a small immutable friction-trial summary without deleting raw evidence."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('names',nargs='+');args=parser.parse_args()
    for name in args.names:
        if not name.replace('-','').replace('_','').isalnum():raise ValueError('Invalid label')
        folder=ROOT/'outputs'/name;data=json.loads((folder/'result.json').read_text())
        summary={k:v for k,v in data.items() if k!='reset_diagnostic'}
        summary['run']=name
        diagnostic=data.get('reset_diagnostic') or {}
        if diagnostic:
            summary['reset_grasp']=diagnostic.get('grasp')
            preparation=diagnostic.get('preparation') or []
            if preparation:
                summary['reset_last_jaw_load_n']=preparation[-1]['jaw_load_n']
                summary['preparation_steps']=len(preparation)
        summary['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.iterdir() if p.name in ('result.json','frame-timing.json','dose-steps.json','preparation.json','context.json','driver.py','training-adapter.py','interactions.mp4')}
        summary['not_full_s16d_matrix']=True
        with (ROOT/'results'/(name+'-summary.json')).open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')
        print(json.dumps({k:summary.get(k) for k in ('run','passed','completed','failure','frames','native_ms')}))


if __name__=='__main__':main()
