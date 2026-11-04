import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'endoscopic_tissue/outputs/grasp-cut-validation-20261031'
rows=[]
for path in sorted(BASE.glob('*/report.json')):
    r=json.loads(path.read_text(encoding='utf8'))
    if 'metadata' not in r: continue
    stages={s['stage']:s for s in r['stages']}
    row=dict(fixture_valid=r['metadata'].get('fixture_valid', True),run=path.parent.name,dll_sha256=r['metadata']['dll_sha256'],case=r['metadata']['case'],completed=r['completed'],model=r['metadata']['settings']['contact_model'],nodes=r['metadata']['material_nodes'],physics=r['physics_ms'],pipeline=r['pipeline_ms'],healed=r['healed_connections'],disabled_cuts=r['cuts_while_disabled'],max_blade_overlap_mm=r['max_blade_overlap_mm'],max_wrist_overlap_mm=r['max_wrist_overlap_mm'],invalid_tets=r['max_invalid_tets'])
    row['stages']={}
    for key in ('close','lift','sustain','reclose','relift','rehold','incise_1','incise_2','incise_repeat_1','incise_repeat_2','hold','side_hold','jaw_open','relax','open_retract'):
        if key not in stages: continue
        s=stages[key]; e=s['end']; a=s['audit']
        row['stages'][key]={k:e[k] for k in ('grip_points','jaw_load_n','bite_nodes','held_bite_nodes','pad_slip_mm','bite_lift_mm','cut_edges','fractures','jaw_gap_mm','tumor_com_mm')}
        row['stages'][key].update({k:a[k] for k in ('material_components','lattice_components','stray_skin_cuts','positional_nonmanifold_edges','degenerate_triangles','max_triangle_edge_mm')})
    row['original_metrics']=r['original_metrics']
    row['errors']=r['diagnostic_errors']
    rows.append(row)
(BASE/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
for r in rows:
    print(r['run'],r['completed'],'core ms',round(r['physics']['mean'],2),'p95',round(r['physics']['p95'],2),'overlap',round(r['max_wrist_overlap_mm'],3))
