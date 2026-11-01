"""Aggregate evidence without turning missing/failed physical checks into passes."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from endoscopic_tissue.training.provenance import sha256
P=Path(__file__).resolve().parent
OUT=ROOT/'endoscopic_tissue/outputs/training-completion-20261101'


def main():
    result={'schema':'training_completion_evidence_v1','default_replaced':False,'training_domain_approved':False}
    reports=[json.loads((OUT/f'timing-{i}/report.json').read_text()) for i in range(4)]
    result['abba']=dict(baseline_core_mean_ms=float(np.mean([reports[i]['physics_ms']['mean'] for i in [0,3]])),
        candidate_core_mean_ms=float(np.mean([reports[i]['physics_ms']['mean'] for i in [1,2]])),
        baseline_pipeline_mean_ms=float(np.mean([reports[i]['pipeline_ms']['mean'] for i in [0,3]])),
        candidate_pipeline_mean_ms=float(np.mean([reports[i]['pipeline_ms']['mean'] for i in [1,2]])),
        settings_identical=all(r['settings']==reports[0]['settings'] for r in reports),
        steps=[r['steps'] for r in reports],raw_reports=[str(OUT/f'timing-{i}/report.json') for i in range(4)])
    support=json.loads((OUT/'r20-kidney/support-cache-audit.json').read_text())
    overlap=json.loads((OUT/'r20-kidney/release-overlap-audit.json').read_text())
    report=json.loads((OUT/'r20-kidney/report.json').read_text())
    result['r20']={'steps':report['steps'],'all_support_flags_equal':support['all_equal'],
                   'support_checks':len(support['checks']),'release_frames':len(overlap),
                   'closed_body_frames':sum(r['body_closed'] for r in overlap),
                   'observed_vertex_penetration_max_mm':max(r['max_penetration_mm'] for r in overlap),
                   'penetration_result_reliable':all(r['body_closed'] and r['piece_closed'] for r in overlap),
                   'accepted':False,'rejection':'no demonstrated performance nonregression; nonmanifold body makes containment audit insufficient'}
    datasets=[]
    for path in sorted(OUT.glob('final-*/**/results.json')):
        data=json.loads(path.read_text(encoding='utf8'))
        cases=[]
        for row in data['episodes']:
            context_file=path.parent/row['id']/'episode/context.json'
            matching=frozen_matching=None
            if context_file.exists():
                context=json.loads(context_file.read_text(encoding='utf8'))
                matching=all(sha256(ROOT/name)==digest for name,digest in context['current_adapter_and_asset_sha256'].items())
                snapshot=P/path.relative_to(OUT).parts[0]/'adapter-source'
                frozen_matching=all(sha256(snapshot/Path(name).name if name.startswith('endoscopic_tissue/training/')
                                           else ROOT/name)==digest
                                    for name,digest in context['current_adapter_and_asset_sha256'].items())
            cases.append({k:row.get(k) for k in ['id','seed','status','reason','error','transitions','states',
                          'task_success','task_failure','offline_reward_match','worker_peak_rss_bytes']})
            cases[-1]['current_source_assets_match_context']=matching
            cases[-1]['frozen_source_assets_match_context']=frozen_matching
        datasets.append({'path':str(path),'summary':data['summary'],'cases':cases})
    result['datasets']=datasets
    protected=json.loads((P/'machine.json').read_text())['protected']
    result['default_protection']={p:sha256(ROOT/p)==digest for p,digest in protected.items()}
    result['all_protected_unchanged']=all(result['default_protection'].values())
    correction=P/'strict-id-offline-equivalence.json'
    if correction.exists():
        detail=json.loads(correction.read_text())
        result['post_capture_input_validation']={k:detail[k] for k in
            ('change','native_physics_changed','total_transitions','all_online_records_equal','cpu_tests','cpu_exit_code')}
        result['post_capture_input_validation']['recomputed_goal_episodes']=len(detail['episodes'])
    (P/'evidence-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
