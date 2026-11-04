"""Generate report tables from recorded measurements; failures remain visible."""
import json,statistics
from pathlib import Path
p=Path(__file__).resolve().parents[1];data=json.loads((p/'results/performance-summary.json').read_text());by={d['name']:d for d in data}
selected=[('baseline-a1','基线 A1，固定 4 核'),('baseline-a2','基线 A2，固定 4 核'),('cpu-fused-b1','仅 CPU 融合'),('cpu-gpu-c1','V1 CPU 融合 + GPU 后处理'),('baseline-rgbd-a3','基线 A3'),('final-rgbd-b1-retry','V2 重复 B1'),('final-rgbd-b2','V2 重复 B2'),('baseline-rgbd-unbound','基线，不固定 CPU 亲和性'),('final-rgbd-unbound','V2，不固定 CPU 亲和性'),('baseline-physics','基线：物理 + 全检查，无图像'),('final-physics','V2：物理 + 全检查，无图像'),('baseline-vision','基线：训练 RGB-D'),('final-vision','V2：训练 RGB-D，失败'),('vision-v2-paired','V2：训练 RGB-D，缓存关闭'),('vision-final-paired','V4：再加 GPU 深度对齐')]
lines=['## 未插桩完整场景性能对照','', '以下完整场景使用同一 1410 步动作脚本和原验收阈值。每组只比较相同模式；CPU 亲和性变更另列，训练图像和直接 RGB-D 不混算。共享服务器负载变化明显，保留全部重复结果，不挑最高 FPS 代替稳定性能。','', '| 模式 / 版本 | 帧数 | 均值 Hz | 均值 ms | P50 ms | P95 ms | GPU 利用率均值 % | 显存峰值 MiB | 脚本检查 |','|---|---:|---:|---:|---:|---:|---:|---:|---|']
for name,label in selected:
 if name not in by:continue
 d=by[name];steady=d['steady_resource_samples']>0;gpu=d['gpu_percent_steady_mean'] if steady else d['gpu_percent_process_mean'];memory=d['gpu_mib_steady_peak'] if steady else d['gpu_mib_process_peak'];suffix='' if steady else '†'
 lines.append(f"| {label} | {d['frames']} | {d['fps']:.2f} | {d['mean_ms']:.2f} | {d['p50_ms']:.2f} | {d['p95_ms']:.2f} | {gpu:.1f}{suffix} | {memory:.0f}{suffix} | {'完成，通过已启用检查' if d['completed'] and d['quality_passed'] else '**失败，非有效整段成绩**'} |")
lines.extend(['','资源列为 NVML 约 2 s 一次的采样，不是 SM occupancy。† 表示该旧记录缺少精确测量起止时间，资源均值/峰值包括初始化；其余仅筛选逐帧测量窗口。显存为卡级用量，包含已披露的后台显示上下文。完整逐帧值、P99、超 50 ms 帧数和资源范围见 [JSON](results/performance-summary.json) / [CSV](results/performance-summary.csv)。','', 'V4 训练 RGB-D 的 1410 帧**全部超过 50 ms**，最低 20 帧滚动速率约 7.60 Hz；平均 11.16 Hz、P95 120.35 ms 均未通过 20 Hz 目标。该行通过的只是固定 nominal 场景已启用的检查，完整物理域仍未验收。直接 RGB-D 的最高完整记录 12.73 Hz 同样未通过。','', 'V1 相对重复 A1/A2 的完整 RGB-D 帧耗时约降低 30%，吞吐约提高 42%；去掉绑定 CPU 亲和性的独立组为 9.11→12.73 Hz。GPU 深度对齐的相邻训练对照为 9.78→11.16 Hz（约 +14.1%），视觉阶段 47.70→34.27 ms，而原生物理约 33.07→33.35 ms，支持收益来自视觉路径。不同时间的 V1/V2 单次绝对值不用于证明每个附加优化都有整帧增益。','', '曾有一轮 `final-rgbd-b1` 在 927 帧遇到外来 GPU 进程，由守卫中止。其约 12.31 Hz 是部分结果，排除有效性能结论；在再次确认空闲后另建 `final-rgbd-b1-retry` 完整重跑，原失败日志保留。','', '## 包含图像 IPC 的实际池端到端性能','', '每 worker 一个 CPU 核；固定 60 步切割窗口、完整 1280×960 新图像、深度对齐、有效掩码、策略观测复制及父进程接收。表中 P50/P95 为请求到响应往返延迟；总吞吐另按全窗口 wall-clock 计算。所有列出的成功池运行仅是该短窗口通过，不是完整 episode 或 20 Hz 验收。','', '| 模式 | workers | 总 steps/s | 每环境 steps/s | 往返均值 ms | P50 ms | P95 ms | GPU % | 显存 MiB |','|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
q=json.loads((p/'results/pool-analysis/summary.json').read_text())['runs'];pool={d['name']:d for d in q}
for name,label in [('pool-baseline-1','基线 Pipe'),('pool-baseline-2','基线 Pipe'),('pool-final-1','V2 Pipe'),('pool-final-2','V2 Pipe'),('pool-v4-1','V4 GPU 对齐 + Pipe'),('pool-v4-2','V4 GPU 对齐 + Pipe'),('pool-shared-1','V5 GPU 对齐 + 共享内存'),('pool-shared-2','V5 GPU 对齐 + 共享内存'),('pool-physics-1','V2 无图像，仅物理与检查'),('pool-physics-2','V2 无图像，仅物理与检查')]:
 if name not in pool:continue
 d=pool[name];lines.append(f"| {label} | {d['workers']} | {d['total_steps_s']:.2f} | {min(d['per_worker_steps_s']):.2f} | {d['roundtrip_mean_ms']:.2f} | {d['roundtrip_p50_ms']:.2f} | {d['roundtrip_p95_ms']:.2f} | {d.get('gpu_mean_pct',0):.1f} | {d.get('peak_vram_mib',0):.0f} |")
lines.extend(['','池资源列只取测量窗口采样，最短无图像运行只有 1–4 个样本，不能视为精确利用率预测。未运行学习器、没有实测 episodes/hour；不能用短窗口 steps/s 换算可信有效实验数量。所有 worker 切割事件、质量结果和请求号保留在原始结果中。'])
(p/'results/performance-tables.md').write_text('\n'.join(lines)+'\n')
# Preserve physical failures and variation across every unprofiled full/partial run.
physical=[]
for f in list((p/'results/comparison-v1').rglob('measurement/result.json'))+list((p/'results').glob('*/result.json')):
 d=json.loads(f.read_text())
 if d.get('profiled') is not False or not isinstance(d.get('frames'),list) or not d['frames']:continue
 rows=d['frames'];metrics=[x['quality']['metrics'] for x in rows];name=f.parents[2].name if f.parent.name=='measurement' else f.parent.name
 physical.append(dict(name=name,completed=d['completed'],checks_pass=d['quality_passed'],frames=len(rows),cut_edges_total=sum(m['new_cut_edges'] for m in metrics),invalid_tets_max=max(m['invalid_tets'] for m in metrics),healed_edges=sum(m['healed_edges'] for m in metrics),components_max=max(m['material_components'] for m in metrics),volume_ratio_min=min(m['min_volume_ratio'] for m in metrics),displacement_max_m=max(m['max_displacement_m'] for m in metrics),body_overlap_max_m=max(m['blade_body_overlap_m'] for m in metrics),nonmanifold_edges_max=max(m.get('surface_topology',{}).get('positional_nonmanifold_edges',0) for m in metrics),final_triangles=metrics[-1].get('surface_topology',{}).get('triangles'),failure_reasons=sorted({reason for x in rows for reason in x['quality']['numerical_failure_reasons']}),training_domain_approved=False))
(p/'results/physical-regression-summary.json').write_text(json.dumps(physical,indent=2))
print('Generated tables for',len(by),'frame benchmarks and',len(pool),'pool benchmarks')
