"""Create reviewable artifacts after final validation; no GPU work or deployment."""
import csv,difflib,hashlib,json,os,shutil,statistics,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parents[1];prof=p.parent;root=prof.parent;py='/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python'
final=json.loads((p/'results/final-selection.json').read_text());assert final['completed'];source=Path(final['source'])
subprocess.run([py,str(p/'tools/summarize.py')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
subprocess.run(['/opt/nvidia/nsight-systems/2025.3.2/bin/nsys','export','--type','sqlite','--force-overwrite','true','--output',str(p/'results/final-profile-trace.sqlite'),str(p/'results/final-profile-trace.nsys-rep')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
subprocess.run([py,str(p/'tools/analyze_timeline.py'),str(p/'results/final-profile-trace.sqlite'),'--output',str(p/'results/analysis-final'),'--cpu-spans',str(p/'results/final-profile/cpu-spans.json'),'--plot'],check=True)

def digest(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def patch(before,after,target):
 paths=set()
 for tree in [before,after]:
  paths.update(x.relative_to(tree) for x in tree.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix not in ['.pyc','.nbi','.nbc'] and not x.is_symlink())
 chunks=[];changed=[]
 for path in sorted(paths):
  a,b=before/path,after/path
  if a.exists() and b.exists() and digest(a)==digest(b):continue
  try:left=a.read_bytes().decode('utf8').splitlines(True) if a.exists() else [];right=b.read_bytes().decode('utf8').splitlines(True) if b.exists() else []
  except UnicodeDecodeError:raise RuntimeError('Unexpected changed binary: '+str(path))
  chunks.extend(difflib.unified_diff(left,right,fromfile='a/'+str(path) if a.exists() else '/dev/null',tofile='b/'+str(path) if b.exists() else '/dev/null'));changed.append(str(path))
 target.write_text(''.join(chunks));return changed
package=p/'delivery';package.mkdir(exist_ok=True)
changed=patch(prof/'candidate-v4-source/endoscopic_tissue',source/'endoscopic_tissue',package/'python-from-v4.patch')
# This patch is relative to endoscopic_tissue, not the repository root.
subprocess.run(['git','apply','--check',str(package/'python-from-v4.patch')],cwd=prof/'candidate-v4-source/endoscopic_tissue',check=True)
native_tree=Path(final['dll']).parents[2]/'source';native_changed=[]
if native_tree.exists() and native_tree!=root/'native-s25/source':
 native_changed=patch(p/'native-baseline/source',native_tree,package/'native-from-baseline.patch')
 subprocess.run(['git','apply','--check',str(package/'native-from-baseline.patch')],cwd=p/'native-baseline/source',check=True)
manifest=json.loads((prof/'baseline-manifest.json').read_text());integrity={}
for key,tree in [('frozen_baseline',prof/'baseline-source'),('original_s18_s25_source',root/'source')]:
 bad=[name for name,expected in manifest['source_sha256'].items() if not (tree/name).exists() or digest(tree/name)!=expected];integrity[key]=dict(files_checked=len(manifest['source_sha256']),mismatches=bad)
integrity['native_library']=dict(path=manifest['native_library'],expected_sha256=manifest['native_sha256'],actual_sha256=digest(Path(manifest['native_library'])));integrity['checked_unix_s']=time.time();integrity['passed']=not any(x['mismatches'] for x in [integrity['frozen_baseline'],integrity['original_s18_s25_source']]) and integrity['native_library']['expected_sha256']==integrity['native_library']['actual_sha256'];(p/'results/source-integrity.json').write_text(json.dumps(integrity,indent=2));assert integrity['passed']
package_manifest=dict(source=str(source),source_sha256=json.loads((source/'ASSEMBLY.json').read_text())['source_sha256'],dll=final['dll'],dll_sha256=final['dll_sha256'],flags=final['flags'],python_patch_root=str(prof/'candidate-v4-source/endoscopic_tissue'),changed_python_files=changed,native_patch_root=str(p/'native-baseline/source'),changed_native_files=native_changed,python_patch_sha256=digest(package/'python-from-v4.patch'),production_replaced=False,stable_20hz_passed=final['stable_20hz_passed'],formal_physics_domain_approved=False);(package/'manifest.json').write_text(json.dumps(package_manifest,indent=2))
# Collect nonoverlapping phase and stage costs from unprofiled serial frames.
rows=json.loads((p/'results/performance-summary.json').read_text());by_name={x['name']:x for x in rows}
stages=[]
for name in ['final-v4-control-a','final-rgbd-a','final-rgbd-b','final-v4-control-b']:
 d=json.loads((p/'results'/name/'result.json').read_text());names=list(dict.fromkeys(x['stage'] for x in d['frames']))
 for stage in names:
  frames=[x for x in d['frames'] if x['stage']==stage];values=sorted(x['frame_ms'] for x in frames)
  item=dict(run=name,stage=stage,frames=len(frames),mean_ms=statistics.mean(values),p95_ms=values[min(len(values)-1,int(.95*len(values)))],max_ms=max(values),over_50ms=sum(x>50 for x in values))
  for phase in ['native_step','quality_and_export','snapshot']:item[phase+'_ms']=statistics.mean(x['timings_ms'].get(phase,0) for x in frames)
  item['vision_ms']=statistics.mean(x['render_ms'] for x in frames);stages.append(item)
with (p/'results/final-stage-times.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=stages[0]);w.writeheader();w.writerows(stages)
figures=p/'figures';figures.mkdir(exist_ok=True);os.environ['MPLCONFIGDIR']=str(p/'tmp/matplotlib-final')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
selected=[by_name[x] for x in ['final-v4-control-a','final-rgbd-a','final-rgbd-b','final-v4-control-b']]
phase_keys=['native_step_mean_ms','quality_and_export_mean_ms','snapshot_mean_ms','post_capture_mean_ms'];phase_labels=['Native physics / contact / cutting','Quality + surface export','State construction','Full RGB-D / CPU return']
fig,ax=plt.subplots(figsize=(10,5));bottom=[0.]*4
for key,label in zip(phase_keys,phase_labels):
 values=[x[key] for x in selected];ax.bar(range(4),values,bottom=bottom,label=label);bottom=[a+b for a,b in zip(bottom,values)]
other=[max(0,x['mean_ms']-b) for x,b in zip(selected,bottom)];ax.bar(range(4),other,bottom=bottom,label='Control + policy observation');ax.axhline(50,color='#aa2222',linestyle='--',label='20 Hz: 50 ms budget');ax.scatter(range(4),[x['p95_ms'] for x in selected],marker='_',s=220,color='black',label='P95 full frame');ax.set_xticks(range(4),['V4 control A','Final A','Final B','V4 control B']);ax.set_ylabel('Milliseconds per complete frame');ax.legend(fontsize=8);ax.set_title('1280 x 960 RGB-D, identical physics / checks / OpenCV threads');fig.tight_layout();fig.savefig(figures/'frame-budget.png',dpi=160);fig.savefig(figures/'frame-budget.svg');plt.close(fig)
analysis=json.loads((p/'results/analysis-final/summary.json').read_text());seventh=json.loads((p/'results/seventh-v2-selection.json').read_text());sixth=json.loads((p/'results/sixth-r8-v2-selection.json').read_text());reg=json.loads((p/'results/current-frame-final/result.json').read_text());new=[by_name['final-rgbd-a'],by_name['final-rgbd-b']];old=[by_name['final-v4-control-a'],by_name['final-v4-control-b']]
mean=lambda records,key:statistics.mean(x[key] for x in records)
lines=['# 实时性专项最终报告：瓶颈、激进优化与交互一致性','',f"最终候选在两次完整1410帧测试中为 **{new[0]['fps']:.2f} / {new[1]['fps']:.2f} Hz**，P95 **{new[0]['p95_ms']:.2f} / {new[1]['p95_ms']:.2f} ms**。稳定20Hz验收：**{'通过本固定脚本的性能门槛' if final['stable_20hz_passed'] else '未通过'}**。原正式代码未替换，既有物理域仍未批准。",'',f"同一负载时段、同线程设置的 V4/最终候选 ABBA 平均帧耗时：**{final['control_mean_frame_ms']:.2f} → {final['candidate_mean_frame_ms']:.2f} ms**，总体速度比 **{final['aggregate_wall_speedup']:.3f}×**。不将较早高CPU负载时的10.94Hz与后来的低负载成绩全差额归为代码收益。",'','## 到底哪里耗时最多','', '| 不重复阶段 | 同时段 V4 ms | 最终 ms | 最终占比 |','|---|---:|---:|---:|']
for title,key in [('原生物理、接触、切割及同步观测','native_step_mean_ms'),('质量检查及表面导出','quality_and_export_mean_ms'),('State构造与抓取证据','snapshot_mean_ms'),('网格、完整RGB-D、对齐和CPU读回','post_capture_mean_ms')]:lines.append(f"| {title} | {mean(old,key):.3f} | {mean(new,key):.3f} | {100*mean(new,key)/mean(new,'mean_ms'):.1f}% |")
new_other=mean(new,'mean_ms')-sum(mean(new,k) for k in phase_keys);old_other=mean(old,'mean_ms')-sum(mean(old,k) for k in phase_keys);lines+= [f"| 控制及策略观测复制等 | {old_other:.3f} | {new_other:.3f} | {100*new_other/mean(new,'mean_ms'):.1f}% |",'','质量阶段已经包含表面导出，不能再把surface_export加一次。State构造不等于每帧保存完整原生快照。原生CPU API范围包含GPU执行和同步，不能再与GPU核时间相加。','', '![完整帧预算](figures/frame-budget.png)','', '## 完整性能与资源','', '| 测试 | Hz | P50 ms | P95 ms | 最慢20帧窗口 Hz | >50ms帧 | GPU利用率均值 | 显存峰值 MiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in selected+[by_name['final-checked-physics']]:lines.append(f"| {r['name']} | {r['fps']:.2f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['rolling20_min_hz']:.2f} | {r['over_50ms']}/{r['frames']} | {r.get('gpu_utilization_mean_pct',0):.1f}% | {r.get('peak_vram_mib',0):.0f} |")
lines+=['','无图物理行仍包含原生物理、全部已启用检查、State及控制；纯原生阶段速度只是一帧内局部指标。RGB-D行包含每步新图像、原始/对齐深度、有效性及策略观测复制，已返回CPU；显示刷新、网络传输和磁盘录像未计入，不能据此宣称这些额外开销也已达标。初始化编译/重置不在稳态帧间隔内，运行中首次出现新几何的开销保留。','', '## CPU / GPU 时间线','',f"最终Nsight节点级捕获为切割期470–481的12帧：wall-clock **{analysis['mean']['wall_ms']:.3f} ms/帧**；实际CUDA核区间并集 **{analysis['mean']['kernel_union_ms']:.3f} ms**；GPU copy区间并集 **{analysis['mean']['memcpy_union_ms']:.3f} ms**；GPU memset **{analysis['mean']['memset_union_ms']:.3f} ms**；OpenGL设备工作区间 **{analysis['mean']['opengl_union_ms']:.3f} ms**；CUDA与GL总区间并集 **{analysis['mean']['gpu_union_ms']:.3f} ms**。这是有插桩的高负载切割窗口，不能当作1410帧平均FPS。",'', '| 最大CUDA核 | ms/帧 | 次/帧 |','|---|---:|---:|']
for x in analysis['kernels'][:10]:lines.append(f"| `{x['name'].split('(')[0]}` | {x['sum_ms_per_frame']:.3f} | {x['calls_per_frame']:.1f} |")
lines+=['','CPU与GPU时间线：[SVG](results/analysis-final/timeline.svg)、[Perfetto/Chrome trace JSON](results/analysis-final/timeline.json)；CPU线程wall/on-CPU/exclusive时间：[cpu-spans.json](results/final-profile/cpu-spans.json)，完整Python剖析：[python-profile.txt](results/final-profile/python-profile.txt)。CPU线程ID和CUDA stream/context分开，不把重叠层叠加。','', '| CPU范围（包含子范围） | wall ms/帧 | on-CPU ms/帧 | exclusive CPU ms/帧 |','|---|---:|---:|---:|']
for name in ['native.step','quality.all','quality.topology','state.snapshot','render.update_mesh','render.rgbd','render.gpu_post','vision.all','policy.observation']:
 x=analysis.get('cpu_spans_per_frame',{}).get(name)
 if x:lines.append(f"| {name} | {x['inclusive_wall_ms']:.3f} | {x['thread_cpu_ms']:.3f} | {x['exclusive_cpu_ms']:.3f} |")
lines+=['','on-CPU包含驱动忙等，不等于纯CPU数学运算。没有权限使用硬件性能计数器/CPU调度采样，不能凭寄存器数宣称已测得occupancy或显存带宽瓶颈。API期间设备覆盖率只是重叠证据，不是逐微秒等待原因的证明。','', '| CUDA CPU API | API wall ms/帧 | 同期间CUDA活动 ms/帧 | 无CUDA活动 ms/帧 |','|---|---:|---:|---:|']
for x in analysis['cuda_api_device_overlap'][:7]:lines.append(f"| {x['name']} | {x['wall_ms_per_frame']:.3f} | {x['cuda_active_during_api_ms_per_frame']:.3f} | {x['no_cuda_activity_during_api_ms_per_frame']:.3f} |")
lines+=['','## 已实施与取舍','', '1. 精确几何与抓取融合：保留全部候选、退化分支、证人节点和阈值；不使用fastmath。CPU随机/边界/真实状态对照见[结果](results/cpu-equivalence.json)。','2. 每帧组织位置、法线、UV、包围盒融合；21组属性逐位一致，7.47→0.63ms。全部属性仍每帧更新，未采用上一帧布局。','3. 顺序三角形汤使用持久Primitive/GL缓冲；动态数量变化更新绘制数量和几何中心。7组输出逐像素一致；17.40→6.82ms是同输入微基准，不能当完整帧收益。','4. GPU直接输出训练RGB，NumPy分配独立连续数组，Numba单遍填充。保留普通交互render默认BGR。首轮Numba所有权标志失败已保留并修正；4组RGB-D逐位对照通过。完整控制17.62→19.26Hz，视觉14.08→9.09ms。',f"5. 当前表面GPU拓扑摘要：112组合成边界和1410真实帧CPU/GPU指标一致。私有缓冲修正后的同条件复测CPU平均{sixth['cpu_mean_ms']:.3f}ms，GPU{sixth['gpu_mean_ms']:.3f}ms；{'保留' if sixth['topology_gpu_kept'] else '不保留'}。读取当前重建，不改接触缓存。",f"6. 最后一次均匀形状预计算：验证结果`{json.dumps(seventh.get('native_validation'),ensure_ascii=False)}`；{'经完整性能比较后保留' if seventh['native_precomputed_kept'] else '未纳入最终库'}。未使用候选压紧或工具形状假设。",'', 'GPU拓扑初版曾因新临时缓冲被自动纳入快照而失败。逐分配对照证明仅5个新增派生缓冲变化，旧物理分配和全部host字段不变；V2令新临时缓冲独立管理，未修改state_archive代码/ABI或排除任何旧物理字段，并重新通过严格完整快照对照及1410帧CPU/GPU对照。根因见[archive-v1-root-cause.json](results/archive-v1-root-cause.json)。','','同帧CPU线程重叠、GL nogil、PBO、固定背景更新未带来可靠完整收益，已从最终最小源码移除。抓钳特化/压紧无可靠收益；主要剪刀压紧版960次中201次严格接触对照失败，未采用。所有实验版本只留在归档目录，详见[实验记录](EXPERIMENTS.md)。','', '## 正确性与交互边界','', f"最终116项CPU接口回归通过；32帧真实闭合切割回归通过，cut_onset={reg.get('actual_cut_onset')}；4次当前状态RGB对照逐位一致，深度误差≤1e-6m、有效性一致；旧观测未被后续帧覆盖。两次完整1410帧保留原有检查且通过其局部范围。参数、动作空间和验收阈值逐项相同，见[参数对照](results/final-parameter-equivalence.json)。",'', 'GPU拓扑库保留原308个设备函数SASS逐位一致；最后查询实验的参考核也与原FindSurface SASS一致，其余307个原函数不变。完整ELF及host relocation散列不同，不声称整个库字节相同。',[f"实际最终库：`{final['dll']}`，SHA256 `{final['dll_sha256']}`。"][0],'', '这些验证不能替代尚未通过的物理域验收：新几何重复不稳定、纯摩擦准备失败、wall器械碰撞失败、真实撕裂分离正对照未触发、连续S25支撑绑定数量错位、随机化可见性失败、A/A未来轨迹不确定性仍未解除。`formal_physics_domain_approved=False`。原有失败见[原S18–S25报告](../../REPORT.md)与[上一轮Profiling报告](../REPORT.md)。','', '密集S25每像素支撑ID/权重仍单列：CPU展开每1280×960帧约252MB，且有连续绑定错位。当前RGB-D性能表没有启用该原基准未启用的特权标签；不能据此说完整密集标签达到20Hz。','', '## 剩余瓶颈及下一步','', '先看[按动作阶段的逐段耗时](results/final-stage-times.csv)和[逐帧原始数据](results/final-rgbd-a/result.json)。平均超过20Hz仍不能掩盖切割/密集接触期的慢帧。接下来的主要预算必须从原生接触与形变求解取得；保持原3子步、8次接触细化、材料参数、查询数量及切割规则，先解决导致严格对照失败的具体差异，再做接触查询数据布局和重复计算消除。现有CUDA Graph已经使用；不能把再次开启Graph当成新方案，图销毁/启动的API延迟也须先扣除重叠设备活动后评估。','', 'CPU负载在本轮从约60%降到约12%，无新RGB优化的同版本控制亦明显变快。已经用最终同线程ABBA重新对照；绝对帧率仍受共享主机影响，未测量到足够证据区分频率、缓存/内存竞争和调度各自贡献。本轮未改时钟、cgroup、MPS、系统权限或网络。','', '多worker不能直接把本单环境Hz乘以数量，也不自动缩短单帧延迟。本轮优先解决完整交互关键路径，仅使用一个有资源守卫的A100；没有进行新版多worker吞吐验收，不据此承诺并行环境数。','', '## 交付与复现','', '最终最小候选：[final-source](final-source/ASSEMBLY.json)。补丁与散列：[delivery/manifest.json](delivery/manifest.json)、[Python补丁](delivery/python-from-v4.patch)。补丁只做`git apply --check`，未应用到正式代码；其相对根为V4的endoscopic_tissue目录。原源码/库完整性复核见[source-integrity.json](results/source-integrity.json)。','', '使用现有环境执行`tools/reproduce_final.py --output results/一个新目录名`；脚本检查输出目录、源文件/库散列和GPU守卫。完整命令记录：[final-commands.jsonl](final-commands.jsonl)。所有完整、失败、资源中断、微基准、验证和插桩结果保留，完整性能表见[performance-table.md](results/performance-table.md)。未安装或升级依赖，未替换正式入口。','']
(p/'REPORT.md').write_text('\n'.join(lines))
repro='''"""Reproduce the validated configuration in a new in-project result directory."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parents[1];a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();out=(p/args.output).resolve() if not args.output.is_absolute() else args.output.resolve();assert out.is_relative_to(p),'Output must stay inside this independent experiment directory';assert not out.exists(),'Use a fresh output directory'
d=json.loads((p/'results/final-selection.json').read_text());manifest=json.loads((p/'delivery/manifest.json').read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();assert sha(d['dll'])==d['dll_sha256'],'Native binary changed'
for name,value in manifest['source_sha256'].items():assert sha(Path(d['source'])/name)==value,'Source changed: '+name
cmd=['/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python',str(p/'tools/profile_r7.py'),'--source',d['source'],'--dll',d['dll'],'--output',str(out),'--cpus','7,19,20,25']+d['flags'];raise SystemExit(subprocess.run(cmd).returncode)
'''
(p/'tools/reproduce_final.py').write_text(repro)
(package/'README.md').write_text('# 最终最小候选\n\n参见 ../REPORT.md 和 manifest.json。正式代码未替换，稳定20Hz状态按最终报告。\n\nPython补丁的相对根是 profiling/candidate-v4-source/endoscopic_tissue，原生补丁的相对根是 native-baseline/source。只进行了 git apply --check，未实际应用。编译原生候选时使用已有CMake、nvcc 12.5和g++-12，Release/sm80，与本目录编译日志相同。\n\n复现：用现有Python运行 ../tools/reproduce_final.py --output results/一个新目录。脚本检查源文件/库散列，仍受GPU资源守卫保护。\n')
(p/'README.md').write_text('# 实时性专项独立实验\n\n已完成结果见 [REPORT.md](REPORT.md)，实现与失败取舍见 [EXPERIMENTS.md](EXPERIMENTS.md)。\n\n最终最小源码位于 final-source，补丁与散列位于 delivery；原正式代码未替换。复现入口 tools/reproduce_final.py，必须选择本目录中的新输出目录。仅使用原Python环境，不安装依赖。\n\n历史候选candidate-r1到r8、原生特化/压紧/预计算及GPU只读诊断全部独立归档。失败实验不表示已采用；结果中profiled、topology_validation、native_validation及不完整场景不能计作完整性能验收。全部命令见各轮commands.jsonl，原始逐帧结果见results，日志见logs。\n\n不要在等待/执行中的队列外另起GPU基准。资源守卫会在外来任务出现时停止本轮，绝不终止外来作业。物理域原有失败和密集S25标签问题仍未解除，正式训练域未批准。\n')
print(json.dumps(dict(delivered=True,source=str(source),report=str(p/'REPORT.md'),changed_python_files=changed,changed_native_files=native_changed,stable_20hz_passed=final['stable_20hz_passed']),indent=2))
