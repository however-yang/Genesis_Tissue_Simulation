"""Compare four serial ABBA runs without treating a mean screen as approval."""
import argparse,hashlib,json,statistics
from pathlib import Path
def describe(rows,key):
 values=sorted(r[key] for r in rows)
 if not values:return dict(count=0)
 q=.95*(len(values)-1);i=int(q);p95=values[i]+(values[min(i+1,len(values)-1)]-values[i])*(q-i)
 return dict(count=len(values),mean=statistics.fmean(values),median=statistics.median(values),p95=p95,maximum=values[-1],over_100ms=sum(x>100 for x in values))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['candidate-a','baseline-a','baseline-b','candidate-b']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--kind',choices=['release','wall','friction'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();runs={};stages=[]
 for label in ['candidate_a','baseline_a','baseline_b','candidate_b']:
  folder=getattr(a,label);r=json.loads((folder/'report.json').read_text());rows=json.loads((folder/'steps.json').read_text())
  runs[label]=dict(name=folder.name,case=r['case'],settings=r['settings'],dll_sha256=r['dll_sha256'],steps=r['steps'],expected_steps=r['expected_steps'],final_state=r.get('final_state'),telemetry=r.get('telemetry'),spike_policy=r.get('spike_policy'),report_sha256=hashlib.sha256((folder/'report.json').read_bytes()).hexdigest(),samples_sha256=hashlib.sha256((folder/'steps.json').read_bytes()).hexdigest(),rows=rows)
  for row in rows:
   if row['stage'] not in stages:stages.append(row['stage'])
 reference=runs['baseline_a']
 assert all(x['case']==reference['case'] and x['settings']==reference['settings'] and x['expected_steps']==reference['expected_steps'] for x in runs.values()),'Mismatched timing inputs'
 regions={name:lambda row,n=name:row['stage']==n for name in stages}
 regions['all_common_after_warmup']=lambda row:30<row['step']<=min(x['steps'] for x in runs.values())
 if a.kind=='release':
  for name,lo,hi in [('before_second_cut',31,690),('held_after_cut',841,1215),('full_free_candidate',1244,1320)]:regions[name]=lambda row,l=lo,h=hi:l<=row['step']<=h
 out=dict(kind=a.kind,default_promotion=False,matching_inputs_verified=True,runs={},comparison={},note='Core and native pipeline timings exclude recording/diagnostic auditing. Two ABBA repetitions describe these runs only; inspect p95/worst-frame results and trajectory quality separately. Profiled frames are excluded but their IDs retained.')
 for label,run in runs.items():
  rows=run.pop('rows');out['runs'][label]=dict(**run,profiled_steps=[r['step'] for r in rows if r.get('section_profile')],regions={})
  for name,select in regions.items():
   chosen=[r for r in rows if r['step']>30 and select(r) and not r.get('section_profile')]
   out['runs'][label]['regions'][name]={k:describe(chosen,k) for k in ['physics_ms','pipeline_ms']}
 for name in regions:
  q={}
  for key in ['physics_ms','pipeline_ms']:
   b=[out['runs'][x]['regions'][name][key] for x in ['baseline_a','baseline_b']];c=[out['runs'][x]['regions'][name][key] for x in ['candidate_a','candidate_b']]
   if not all(x['count'] for x in b+c):continue
   bm=statistics.fmean(x['mean'] for x in b);cm=statistics.fmean(x['mean'] for x in c);q[key]=dict(baseline_mean=bm,candidate_mean=cm,delta=cm-bm,delta_percent=(cm/bm-1)*100,baseline_p95_mean=statistics.fmean(x['p95'] for x in b),candidate_p95_mean=statistics.fmean(x['p95'] for x in c),baseline_max=max(x['maximum'] for x in b),candidate_max=max(x['maximum'] for x in c))
  out['comparison'][name]=q
 a.output.write_text(json.dumps(out,indent=2));print(json.dumps(dict(output=str(a.output),comparison=out['comparison']),indent=2))
if __name__=='__main__':main()
