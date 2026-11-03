"""Summarize intrusive native section samples without treating them as normal FPS."""
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parent

def stats(values):
    return {'n':len(values),'mean_ms':statistics.mean(values),'min_ms':min(values),'max_ms':max(values)}

def main():
    samples=[]
    for f in (ROOT/'runs').glob('*/worker-*/profile.json'):
        for row in json.loads(f.read_text()):samples.append(dict(row,path=str(f.relative_to(ROOT))))
    sections={};apis={}
    for s in samples:
        for k,v in s['sections_ms'].items():sections.setdefault(k,[]).append(v)
        for k,v in s['api_wall_ms'].items():apis.setdefault(k,[]).append(v)
    out={'samples':samples,'section_statistics':{k:stats(v) for k,v in sections.items()},
        'api_statistics':{k:stats(v) for k,v in apis.items()},
        'limitations':['Profiling bypasses some CUDA graphs; not representative normal FPS.',
        'Material/contact iterates are coupled; labels are actual event intervals, not exclusive physical subsystems.',
        'Cut tick and substep topology rebuild share an event interval; surface reconstruction Finalize is separate.',
        'API wall time may include CPU work, synchronization and device transfer. CUDA event intervals may include host launch gaps.']}
    out['topology_change_split']={}
    for name,changed in [('changed',True),('unchanged',False)]:
        rows=[r for r in samples if (r['topology_before']!=r['topology_after'])==changed]
        if rows:out['topology_change_split'][name]=stats([r['sections_ms']['begin frame, cut tick and rebuild between substeps'] for r in rows])
    (ROOT/'profile-breakdown.json').write_text(json.dumps(out,indent=2))
    lines=['# 原生分段耗时诊断','',
        f'共 {len(samples)} 个插桩步骤，覆盖牵拉、第一剪附近及剪后状态。此模式绕过部分 CUDA Graph 并增加事件；不混入常规性能表。','',
        '| CUDA 事件区间（原标签） | 样本数 | 平均 ms/步 | 最小 | 最大 |','|---|---:|---:|---:|---:|']
    for name,v in out['section_statistics'].items():lines.append(f"| {name} | {v['n']} | {v['mean_ms']:.3f} | {v['min_ms']:.3f} | {v['max_ms']:.3f} |")
    lines+=['','| C ABI 主机墙钟（包含同步/读回） | 样本数 | 平均 ms/步 | 最小 | 最大 |','|---|---:|---:|---:|---:|']
    for name,v in out['api_statistics'].items():lines.append(f"| {name} | {v['n']} | {v['mean_ms']:.3f} | {v['min_ms']:.3f} | {v['max_ms']:.3f} |")
    lines+=['','| 拓扑版本变化 | 样本数 | 帧开始+cut tick+子步间重建，平均 ms |','|---|---:|---:|']
    for name,v in out['topology_change_split'].items():lines.append(f"| {name} | {v['n']} | {v['mean_ms']:.3f} |")
    lines+=['','接触发现/流形、材料与接触耦合迭代、cut tick 与子步间重建、Finalize 表面重建应分开看。现有计时点不提供独立的“纯形变、纯接触、纯切割、纯拓扑”四项精确分解；本轮不通过推测填补。若需要该粒度，应在隔离的原生构建中增加 CUDA/NVTX 区间并量化插桩误差。','',
        '原始 tick、拓扑版本和每个 API 数值见 profile-breakdown.json。']
    (ROOT/'profile-breakdown.md').write_text('\n'.join(lines)+'\n')
    print('profile samples',len(samples))
if __name__=='__main__':main()
