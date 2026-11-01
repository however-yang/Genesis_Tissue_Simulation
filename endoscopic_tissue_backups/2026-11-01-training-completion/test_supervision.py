"""Real child timeout and disk-budget fault injection without loading a native DLL."""
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from endoscopic_tissue.training.experiment import supervise

P=Path(__file__).resolve().parent
spec={'schema':'tissue_experiment_v1','purpose':'collection','dll':'unused.dll','dll_sha256':'0'*64,
      'episodes':[{'id':'timeout-a','seed':1,'policy':{'kind':'hold'}},
                  {'id':'timeout-b','seed':2,'policy':{'kind':'hold'}}],
      'episode_timeout_s':.001,'min_free_disk_bytes':1}
out=P/'supervision-timeout'
supervise(spec,out)
result=json.loads((out/'results.json').read_text())
assert len(result['episodes'])==2 and all(r['status']=='timeout' for r in result['episodes'])
spec['min_free_disk_bytes']=10**18
out=P/'supervision-disk-budget'
supervise(spec,out)
disk=json.loads((out/'results.json').read_text())
assert all(r['status']=='disk_budget_exceeded' for r in disk['episodes'])
(P/'supervision-test-result.json').write_text(json.dumps({'timeout_owned_children':2,
    'recovered_to_next_episode':True,'disk_budget_prevents_worker':True,'passes':True},indent=2))
print('Timeout recovery and disk guard passed')
