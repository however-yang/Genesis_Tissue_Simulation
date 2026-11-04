from pathlib import Path
import json
h=Path(__file__).resolve().parent
profile=h/'recovery-run/r3-complete-frame-profile/attempt-000/profile'
p=json.loads((profile/'result.json').read_text());r=json.loads((profile/'full-run/report.json').read_text())
summary=dict(profile_completed=p['completed'],library_sha256=p['library_sha256'],material_nodes=p['frames'][0]['material_nodes'],original_metrics=r['original_metrics'],physics_ms=r['physics_ms'],pipeline_ms=r['pipeline_ms'],frames=[{k:f[k] for k in ['native_tick','profiled_frame_wall_ms','kernel_union_active_ms','unattributed_kernel_launches','apis','kernels']} for f in p['frames']],performance_accepted=False,repair_diagnostics=[])
for path in sorted((h/'repair-run').glob('*/attempt-*/diagnostic/result.json')):
 d=json.loads(path.read_text());summary['repair_diagnostics'].append(dict(path=str(path.relative_to(h)),native_tick=d['native_tick'],native_import_exact=d['native_import_exact'],samples=d['samples']))
(h/'continuation-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# 剩余清单续做证据','', '本轮不替换默认库；物理失败保留。','', '## 自动验收恢复与视频','', '新的 acceptance_queue.py 使用子日志/产物进度、不可覆盖的 attempt、输入和结果哈希、独占队列锁、原进程身份校验及 GPU 资源保护。5 项真实子进程回归通过。实际最后视频已在 attempt-001 完成，后续完整帧剖析也完成；原始录像文件未改。466 帧全部可解码，三个 storyboard 均已人工式抽帧检查。视频恢复不改变原来的物理判断。','', '## 102710 材料节点完整帧剖析','', '| native tick | CPU 完整原生步进 ms（含同步/剖析） | GPU kernel 活跃并集 ms | 无 API 归属 kernel 数 |','|---:|---:|---:|---:|']
for f in p['frames']:lines.append(f"| {f['native_tick']} | {f['profiled_frame_wall_ms']:.2f} | {f['kernel_union_active_ms']:.2f} | {f['unattributed_kernel_launches']} |")
lines += ['', '完整 1320 步运行结束。GPU 时间与 CPU 时间有包含关系，不能相加；CUPTI、状态导出及共享服务器负载使它不能作为无记录的 FPS 签署。1244/1245/1300 帧存在原生调用关联缺失，已保留 unattributed，不能把对应时间分摊成已证实的材料/接触比例。', '', '已定位：tick 600 接触搜索 k_FindSurface 17.35 ms；tick 1300 游离块树更新 k_FragmentTreeParents 9.79 ms。整步 33 Hz 目标未达到。原严格报告仍有 stray_skin_cuts，未将执行完成当作物理通过。','', '## 原失败快照与接触收敛','', '两份 R3 原始/恢复分支失败快照均按原库完整导入并逐字节核验，不推进时间、不改变刀具位姿。额外调用只用于诊断，没有接入正常步进。']
for d in summary['repair_diagnostics']:
 a,b=d['samples'][0],d['samples'][-1];lines.append(f"- tick {d['native_tick']}：刀体重叠 {a['body_overlap_mm']:.6f} → {b['body_overlap_mm']:.6f} mm，额外 8 次修复耗时 {b['elapsed_ms']:.2f} ms，最终 minJ={b['material_quality'][0]:.8f}。")
lines += ['', '因此不能将“增加修复次数”作为已验证解决办法。R9 隔离实验仅在重建修复中尝试受体积约束的顶点下降方向：保持原二次模型、力律、材料、子步和 J 下界；所有 incident tets 再次检查，模型能量不优于原步时回退。它针对局部步被单一体积约束整体截断的假设，尚不是已确认根因或已验收修复。','', '## 尚未关闭','', 'R9 必须先过原 20 对换抓分叉，再谈录像及同库其他回归。纯摩擦完整 900 次和 tear/force 完整 900 次、释放扰动/CCD、切口拓扑重复、同库整体验收、≤1 秒恢复、33 Hz 与多 GPU 闭环仍未关闭；不能把不同库历史成功拼接为最终通过。']
(h/'CONTINUATION_REPORT.md').write_text('\n'.join(lines)+'\n')
print('Continuation evidence summarized')
