"""Separate serial numerical repeatability from the parallel acceptance evidence."""
import itertools
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def summary(values):
    return {'min':min(values),'max':max(values),'mean':float(np.mean(values))} if values else None

def main():
    records=[]
    for f in sorted((ROOT/'runs').glob('*/worker-*/episode-*/result.json')):
        config=f.parents[2]/'configuration.json'
        if not config.exists():continue
        c=json.loads(config.read_text());r=json.loads(f.read_text())
        if c['profile'] or not r['full_script'] or not r['completed_requested_stages']:continue
        records.append({'path':str(f.relative_to(ROOT)), 'workers':c['workers'], 'mps':bool(c['mps_pipe']),
            'passed':r['passed'],'snips':r['snips'],'quality_failures':r['quality_failures'],'scope_failures':r['scope_failures']})
    (ROOT/'cut-repeatability.json').write_text(json.dumps(records,indent=2))
    groups={}
    for name,rows in [('serial',[r for r in records if r['workers']==1]),('parallel',[r for r in records if r['workers']>1])]:
        groups[name]={'episodes':len(rows),'all_passed':all(r['passed'] for r in rows),'stages':{}}
        for stage in ('incise_1','incise_2','incise_3'):
            groups[name]['stages'][stage]={key:summary([r['snips'][stage][key] for r in rows]) for key in ('first_material_cut_stage_step','material_edges')}
    serial=[ROOT/r['path'] for r in records if r['workers']==1]
    pairs=[]
    for left,right in itertools.combinations(serial,2):
        with np.load(left.parent/'final-state.npz') as a,np.load(right.parent/'final-state.npz') as b:
            mask=a['inside'].astype(bool);delta=(a['nodes_cm']-b['nodes_cm'])[mask]*10
            norm=np.linalg.norm(delta,axis=1)
            pairs.append({'left':str(left.relative_to(ROOT)),'right':str(right.relative_to(ROOT)),
                'max_material_delta_mm':float(norm.max()),'rms_material_delta_mm':float(np.sqrt(np.mean(norm**2))),
                'neighbors_equal':bool(np.array_equal(a['neighbors'],b['neighbors']))})
    states=json.loads((ROOT/'state-comparisons.json').read_text())
    parallel_paths={str(Path(r['path']).parent/'final-state.npz') for r in records if r['workers']>1}
    parallel=[r for r in states if r['path'] in parallel_paths and 'rms_material_node_delta_mm' in r]
    out={'groups':groups,'serial_pairwise':pairs,
        'serial_pairwise_rms_mm':summary([r['rms_material_delta_mm'] for r in pairs]),
        'serial_pairwise_max_mm':summary([r['max_material_delta_mm'] for r in pairs]),
        'parallel_vs_first_serial_rms_mm':summary([r['rms_material_node_delta_mm'] for r in parallel]),
        'parallel_vs_first_serial_max_mm':summary([r['max_material_node_delta_mm'] for r in parallel]),
        'scope':'Unchanged acceptance checks; few serial repeats establish non-bitwise repeatability, not statistical equivalence or clinical accuracy.'}
    (ROOT/'correctness-summary.json').write_text(json.dumps(out,indent=2))
    lines=['# 切割正确性与串行重复对照','',
        '沿用原完整验收阈值。下列范围是小样本描述，不是等价性检验或新的合格阈值。seed 只记录在当前固定脚本上下文，不改变预设墙面和动作；代码没有借助不同随机任务解释差异。','',
        '| 组别 | 完整回合 | 现有检查全部通过 | 第一/二/三剪起切步范围 | 第一/二/三剪材料断边数范围 |',
        '|---|---:|---|---|---|']
    for name,g in groups.items():
        def ranges(key):
            return ' / '.join(f"{s[key]['min']}–{s[key]['max']}" for s in g['stages'].values()) if g['episodes'] else '未测'
        lines.append(f"| {name} | {g['episodes']} | {g['all_passed']} | {ranges('first_material_cut_stage_step')} | {ranges('material_edges')} |")
    lines += ['', '| 对照 | 材料节点 RMS 差范围 mm | 材料节点最大差范围 mm |','|---|---:|---:|']
    for label,rms,maxi in [('串行重复两两',out['serial_pairwise_rms_mm'],out['serial_pairwise_max_mm']),('并行回合对首个串行',out['parallel_vs_first_serial_rms_mm'],out['parallel_vs_first_serial_max_mm'])]:
        fmt=lambda d:'未测' if d is None else f"{d['min']:.6f}–{d['max']:.6f}"
        lines.append(f'| {label} | {fmt(rms)} | {fmt(maxi)} |')
    lines += ['', '最终切割拓扑存在离散差异，最大节点差需要结合 RMS、位置、原验收和更多重复解释，不能单凭平均小就忽略局部差异。进程隔离和独立重置测试未观察到跨环境状态串扰；它们不构成所有动作域的物理等价性证明。', '', '原始明细：cut-repeatability.json、state-comparisons.json、correctness-summary.json。']
    (ROOT/'correctness-summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='serial_pairwise'},indent=2))
if __name__=='__main__':main()
