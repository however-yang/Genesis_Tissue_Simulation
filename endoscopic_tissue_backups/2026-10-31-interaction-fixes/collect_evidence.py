"""Collect reproducible evidence; artifact completeness is not a quality approval."""
import argparse, hashlib, json
from pathlib import Path
W=Path(__file__).resolve().parent
R=W.parents[1]
D=R/'endoscopic_tissue/outputs/interaction-fixes-20261031'
E=W/'package/evidence'
CASES=['release-baseline','r16-strict-kidney-reference','r17-strict-kidney-reference','r19-strict-kidney-reference',
'baseline31-wall-tip-completion-wide','r16-wall-tip-completion-wide','r18-wall-tip-completion-wide','r18-wall-reference',
'r13-wall-rim-1','r13-wall-rim-2','baseline-wall-tip-short','r13-wall-tip-short','baseline-wall-tip-completion','r13-wall-tip-completion',
'baseline-friction-central','r13-friction-central','r13-friction-long','r13-friction-deeper','r13-friction-zero','r13-friction-regrasp','r13-friction-wall','r13-friction-air','r13-friction-marginal',
'r18-friction-central','r18-friction-marginal','baseline31-friction-deeper','r18-friction-deeper','baseline31-wall-wide-repeat','baseline31-friction-central-repeat','baseline31-friction-regrasp']
OPTIONAL=['r19-wall-tip-completion-wide','r19-friction-central','r19-friction-deeper','r19-friction-regrasp']
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-complete',action='store_true');p.add_argument('--hash-videos',action='store_true');a=p.parse_args()
 E.mkdir(exist_ok=True,parents=True);records=[];missing=[]
 for name in CASES+[x for x in OPTIONAL if (D/x).exists()]:
  folder=D/name;rp=folder/'report.json';vp=folder/'review/video-verification.json'
  if not rp.exists():missing.append(name+': report');continue
  r=json.loads(rp.read_text());meta=r.get('metadata',{});q=dict(name=name,case=meta.get('case'),dll_sha256=meta.get('dll_sha256'),settings=meta.get('settings'),material_nodes=meta.get('material_nodes'),active_nodes=meta.get('active_nodes'),total_grid_nodes=meta.get('total_grid_nodes'),completed=r.get('completed'),steps=r.get('steps'),expected_steps=r.get('expected_steps'),diagnostic_errors=r.get('diagnostic_errors'),max_invalid_tets=r.get('max_invalid_tets'),max_blade_overlap_mm=r.get('max_blade_overlap_mm'),max_wrist_overlap_mm=r.get('max_wrist_overlap_mm'),original_metrics=r.get('original_metrics'),report_sha256=digest(rp),path=str(folder.relative_to(R)))
  q['saved_audits']={}
  if meta.get('kind')=='grasp':
   rows=[json.loads(line) for line in (folder/'trajectory.jsonl').read_text().splitlines()]
   assert len(rows)==r['steps'],name
   q['grasp_frame_summary']=dict(frames=len(rows),maximum_auxiliary_grip_points=max((x['grip_points'] for x in rows),default=0),maximum_cut_edges=max((x['cut_edges'] for x in rows),default=0),maximum_invalid_tets=max((x['invalid_tets'] for x in rows),default=0),maximum_blade_overlap_mm=max((x['blade_overlap_mm'] for x in rows),default=0.0),note='All recorded frame rows; no inference of successful or realistic grasp from these invariants alone.')
  if meta.get('case')=='grasp-central-zero-friction':
   marker=folder/'NEGATIVE-CONTROL.txt'
   assert marker.exists(),'Missing explicit zero-friction provenance'
   q['explicit_runtime_override']=dict(contact_model=1,mu_static=0.0,mu_dynamic=0.0,source=marker.read_text())
  for audit in ['release-overlap-audit.json','conservative-flight-audit.json','fragment-query-gate-audit.json','wall-skin-bridge-audit.json','wall-to-base-audit.json','grasp-displacement-audit.json','friction-audit.json']:
   ap=folder/audit
   if ap.exists():q['saved_audits'][audit]=json.loads(ap.read_text())
  if vp.exists():
   v=json.loads(vp.read_text());q['video']=v
   assert v['frames']==v['decoded']==v['expected_frames'],name
   assert v['frames']==r['steps']+1,(name,v['frames'],r['steps'])
   if a.hash_videos:assert digest(folder/'review/interactions-h264.mp4')==v['movie_sha256'],name
  else:missing.append(name+': complete video')
  records.append(q)
 protected=json.loads((W/'protected-files-before.json').read_text());check={path:dict(expected=value,actual=digest(R/path)) for path,value in protected.items()}
 assert all(x['expected']==x['actual'] for x in check.values()),'Protected default changed'
 result=dict(default_promoted=False,protected_files=check,records=records,missing=missing,video_hashes_checked=a.hash_videos,note='Artifact completion does not imply realism, cutting, grasping, or performance acceptance. Keep recorded failures.')
 (E/'interaction-evidence-index.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 if not missing:
  lines=['# 完整交互录像索引','', '全部录像保留已记录帧并再次解码核对帧数。30fps是仿真时间回放，不是实际计算帧率。基线切除在1247步触发保护，录像止于1246步；各条轨迹是否完成见下表。录像存在不等于效果验收通过，详细结论见交互修复报告。','', '| 算例 | 已完成步数 / 计划步数 | 轨迹完成 | 完整录像 |','|---|---:|---|---|']
  for q in records:
   movie=(R/q['path']/'review/interactions-h264.mp4').as_posix()
   lines.append(f"| {q['name']} | {q['steps']} / {q['expected_steps']} | {'是' if q['completed'] else '中止，失败保留'} | [播放]({movie}) |")
  (E.parent/'video-index.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print(json.dumps(dict(cases=len(records),missing=missing,protected=len(check),default_promoted=False),indent=2))
 if a.require_complete:assert not missing,missing
if __name__=='__main__':main()
