"""Final measured report; keeps the previous two-optimization report as an archive."""
import json
from pathlib import Path
p=Path(__file__).resolve().parents[1]
read=lambda n:json.loads((p/'results'/n/'result.json').read_text())
d=json.loads((p/'results/final-summary.json').read_text());b,a=d['before'],d['after']
sel=json.loads((p/'results/all-queries-v2-selection.json').read_text());assert sel['qualified']
old=json.loads((p/'results/analysis-delivery-baseline/summary.json').read_text())
one=json.loads((p/'results/analysis-all-queries-v2-blade/summary.json').read_text())
final=json.loads((p/'results/analysis-all-queries-v2-candidate/summary.json').read_text())
diag=read('all-queries-v2-full-validation');q=diag['native_validation'];c=diag['coarse_bound_validation']
archive=p/'TWO_STAGE_REPORT.md'
if not archive.exists():
 archive.write_text((p/'REPORT.md').read_text().replace('主交付候选的完整验证和同库静默 ABBA 已完成；附加抓钳调度试验单列，仍在核验。','此文件归档剪刀合批加材料缓存的已完成数据。最终结论见 REPORT.md。'))
phase=[]
for title,k in [('原生物理、接触、切割及同步观测','native_step'),('质量检查与表面导出','quality_and_export'),('State构造','snapshot'),('完整当前帧RGB-D、深度对齐及CPU返回','vision'),('控制、策略观测复制等','other')]:
 phase.append(f"|{title}|{b['components_ms'][k]:.3f}|{a['components_ms'][k]:.3f}|{a['component_percent'][k]:.1f}%|")
rows=[]
for title,names in [('原核',['all-queries-v2-original-a','all-queries-v2-original-b']),('剪刀合批+材料缓存',['all-queries-v2-blade-a','all-queries-v2-blade-b']),('最终双器械合批+材料缓存',['all-queries-v2-candidate-a','all-queries-v2-candidate-b'])]:
 for i,name in enumerate(names):
  x=read(name)['performance'];rows.append(f"|{title} {'AB'[i]}|{x['fps']:.2f}|{x['mean_ms']:.2f}|{x['p50_ms']:.2f}|{x['p95_ms']:.2f}|{x['p99_ms']:.2f}|{x['rolling20_min_hz']:.2f}|{x['over_50ms']}|")
def ms(diag,terms):return sum(x['sum_ms_per_frame'] for x in diag['kernels'] if any(t in x['name'] for t in terms))
kr=[]
for title,terms in [('接触查询及保持器械分组的追加',['k_FindSurface(','k_FindSurfaceBatched(','k_AppendQueryKinds(']),('粗尺度材料累加',['k_CoarseAccumulate(','k_CoarseAccumulateBoundCache(']),('材料步长界限',['k_CoarseBound(','k_CoarseBoundFromCache(']),('接触/抓取装配',['k_AssembleUnifiedInteractions(']),('局部顶点求解',['k_UnifiedVertex<']),('接触面采样',['k_UnifiedManifold(']),('最小体积修复',['k_RepairMinimumVolume('])]:
 kr.append(f"|{title}|{ms(old,terms):.3f}|{ms(one,terms):.3f}|{ms(final,terms):.3f}|")
cpu=[]
for name in ['native.step','quality.all','state.snapshot','render.update_mesh','vision.all','policy.observation']:
 x=final['cpu_spans_per_frame'][name];cpu.append(f"|{name}|{x['inclusive_wall_ms']:.3f}|{x['thread_cpu_ms']:.3f}|{x['exclusive_cpu_ms']:.3f}|")
api=[]
for x in final['cuda_api_device_overlap'][:4]:
 api.append(f"|{x['name']}|{x['wall_ms_per_frame']:.3f}|{x['cuda_active_during_api_ms_per_frame']:.3f}|{x['no_cuda_activity_during_api_ms_per_frame']:.3f}|")
text=f"""# 原生物理、接触与切割：最终性能报告

**稳定20Hz仍未通过。** 最终两轮完整1280×960 RGB-D为 **23.00 / 23.06 Hz**，P95 **50.41 / 50.42ms**，最慢20帧窗口 **18.45Hz**。三项优化已实际实现和验证，正式入口未替换；S18–S25物理域、训练域仍未批准。

## 到底哪里耗时最多

同库、同线程设置的六轮交错测试，每轮1410帧。下表原核和最终各合并2820帧，单位是完整帧wall-clock ms，各行不重复。

|阶段|原核ms|最终ms|最终占比|
|---|---:|---:|---:|
"""+'\n'.join(phase)+f"""

原生阶段原占完整帧 **{b['component_percent']['native_step']:.1f}%**，最终仍占 **{a['component_percent']['native_step']:.1f}%**，是首要瓶颈。它包含GPU执行及同步观测，不能再加一次GPU核时间。质量行含surface_export；State构造不是每帧保存完整原生快照。

![完整帧预算与P95](figures/frame-budget.png)

## 实际GPU热点与时间线

以下为切割窗口470–481的12帧Nsight节点记录。原核列来自前一组静默原核捕获，两项/最终列来自最终库的剪刀/双器械合批捕获；原生产指令、脚本、参数和窗口相同，原有浮点原子次序仍使未来轨迹不确定。插桩窗口不用于整段FPS验收。

|CUDA工作，ms/帧|原核|剪刀合批+缓存|最终双器械合批+缓存|
|---|---:|---:|---:|
"""+'\n'.join(kr)+f"""

最终窗口wall-clock **{final['mean']['wall_ms']:.3f}ms/帧**；CUDA计算核并集 **{final['mean']['kernel_union_ms']:.3f}ms**，CUDA传输 **{final['mean']['memcpy_union_ms']:.3f}ms**，memset **{final['mean']['memset_union_ms']:.3f}ms**，全部CUDA并集 **{final['mean']['cuda_union_ms']:.3f}ms**。已记录GL工作 **{final['mean']['opengl_union_ms']:.3f}ms**，CUDA+GL并集 **{final['mean']['gpu_union_ms']:.3f}ms**。

直接剪切/重建代表核较小：前一窗口k_CutScissorsLevels约0.017ms、k_BuildCutTriangles约0.039ms、k_BindSurface约0.044ms。切割引发的重绑、体积修复与继续求解已经进入接触/材料/修复核，不能只计剪切函数或重复加总。

|最终CPU范围|wall ms/帧|thread CPU ms/帧|exclusive CPU ms/帧|
|---|---:|---:|---:|
"""+'\n'.join(cpu)+"""

thread CPU含驱动忙等，exclusive只扣除已插桩子范围，均不能直接解释成纯CPU数学计算。

|最终CPU CUDA API包络|API wall ms/帧|同时有CUDA活动 ms/帧|未记录CUDA活动 ms/帧|
|---|---:|---:|---:|
"""+'\n'.join(api)+f"""

API包络包含设备等待，不与GPU区间相加。Memcpy长包络大部分覆盖前序GPU计算，实际传输区间远小于API耗时；图销毁也存在等待重叠。未记录CUDA的时段可能含GL或CPU；GL记录有覆盖缺口。没有采集SM/DRAM硬件计数器及CPU调度采样，不能精确细分这些残差或宣称带宽饱和/occupancy已测得。

实际时间线：[原核SVG](results/analysis-delivery-baseline/timeline.svg)、[剪刀合批SVG](results/analysis-all-queries-v2-blade/timeline.svg)、[最终SVG](results/analysis-all-queries-v2-candidate/timeline.svg)、[最终Perfetto JSON](results/analysis-all-queries-v2-candidate/timeline.json)。CPU线程与GPU stream/context分别标注；完整统计和Python剖析位于对应results目录。

## 完整RGB-D、纯物理及资源

|配置|Hz|平均ms|P50ms|P95ms|P99ms|最慢20帧Hz|>50ms帧/1410|
|---|---:|---:|---:|---:|---:|---:|---:|
"""+'\n'.join(rows)+f"""

两轮合并原核 **{b['fps']:.2f}→{a['fps']:.2f}Hz**，平均 **{b['mean_ms']:.3f}→{a['mean_ms']:.3f}ms**，速度比 **{d['speedup']:.3f}×**；P95 **{b['p95_ms']:.3f}→{a['p95_ms']:.3f}ms**。仍有 **{a['over_50ms']}/{a['frames']}** 帧超过50ms，最慢20帧窗口 **{a['rolling20_min_hz']:.2f}Hz**，所以不判定稳定20Hz通过。

无图、保留原检查/State/控制的完整帧：原核 **{d['physics_before']['fps']:.2f}Hz**（前一组静默原核，生产指令及参数相同），最终 **{d['physics_after']['fps']:.2f}Hz**，最终最慢20帧 **{d['physics_after']['rolling20_min_hz']:.2f}Hz**。渲染和必要CPU返回使完整交互帧慢于纯物理；原生阶段倒数仅是阶段指标。

NVML测量时段利用率均值 **{b['gpu_percent']:.1f}%→{a['gpu_percent']:.1f}%**，显存采样峰值 **{b['sampled_peak_memory_mib']:.0f}→{a['sampled_peak_memory_mib']:.0f}MiB**。GPU工作减少后利用率不必升高。全机CPU均值 **{b['host_cpu_percent']:.2f}%/{a['host_cpu_percent']:.2f}%**，不能代替主线程分析。

![逐帧完整耗时](figures/frame-times.png)

同一A100 UUID、CPU亲和7/19/20/25、BLAS/OMP和OpenCV各1线程，seed0、settled/wall-traction、1410步。顺序为原核—剪刀—双器械—双器械—剪刀—原核，期间无并发编译。每步新1280×960 RGB、原始/对齐深度、有效性及独立CPU观测；原3子步、8次接触细化、材料/摩擦/切割参数和精度保留。运行中新拓扑/几何开销保留，初始化与重置不计稳态帧。

显示、网络、录像、策略网络和训练未计入；原基准未开启的密集S25支撑ID/权重也未计入，不能据本表声称这些负载已达标。历史11.60Hz时主机负载不同，不作为本轮代码收益分母。

## 三项实现及正确性

1. 剪刀同轮形状用二维网格并行查询，保留原数学体、缓存、预载和全部形状，各自活动表按原器械顺序拼接。
2. 抓钳三种形状复用同一合批核，覆盖五处分派；无body、超容量或不支持流使用原核回退。
3. 材料累加缓存本来已计算的J/gJ，紧随其后的本阶段步长界限复用。中间只累加块能量和求块步长，不改节点位置/权重/活性/组件。每次累加覆盖整个四面体池含死单元，不跨方向、迭代、子步、帧或拓扑更新复用旧值。

新增常驻GPU缓存约10MB，私有原始CUDA分配不进入物理快照；所有旧物理字段及state_archive/xpbd_state实现保留。诊断读取同一生产核真实输出，与原公式比较。完整1410帧中材料累加 **{c['accumulate_terms']:,}** 项、界限 **{c['bound_terms']:,}** 项，目标块、梯度/Hessian和界限尺度差异均0；其中非平凡受限界限 **{c['nontrivial_bound_scales']:,}** 个。接触有效记录 **{q['positive_contact_records']:,}**、接触面记录 **{q['manifold_records']:,}**，字节、权重、节点度数、活动集合、组序差异0，覆盖CUDA Graph，未跳过分支。

32步真实闭合切割触发cut_onset；4组当前RGB逐位一致、深度误差≤1e-6m、有效性一致，旧观测未被覆盖。完整原生快照逐位恢复，恢复后诊断及下一步通过。[图像回归](results/all-queries-v2-current-frame/result.json)、[真实切割图像](results/all-queries-v2-current-frame/current-rgb-step-31.png)、[快照回归](results/all-queries-v2-snapshot-roundtrip/result.json)、[完整生产核对照](results/all-queries-v2-full-validation/result.json)。这不承诺未来轨迹逐位确定；原有浮点原子累加次序不确定性仍在。

全部313个上一轮设备函数SASS不变。抓钳扩展保留329个父版函数，唯一新增核是诊断器。V1误按6槽解释抓钳3槽，仍记录为验证失败；V2保留原6槽检查器，新增按实际stride及声明kind次序的诊断。5个GPU正反例证明组内重排允许、跨组乱序拒绝，未放宽要求。[设备函数证明](results/all-queries-v2-device-binding.json)、[诊断根因](results/grasper-validator-stride-root-cause.json)、[GPU正反例](results/query-order-stride-probes/result.json)。

代码：[查询与对照](native-all-queries-v2/source/vendor/cuda/contact/surface_contact.cuh)、[材料缓存](native-all-queries-v2/source/vendor/cuda/contact/coarse_bound_cache.cuh)、[双器械分派与六处材料依赖链](native-all-queries-v2/source/vendor/cuda/xpbd/xpbd_solver.cu)。启动配置query_mode=3、query_threads=64、assembly_blocks=512、manifold_blocks=1024、bound_cache=true；初始化后不能切换。跨进程及快照复现须绑定执行配置、源码和库散列。

## 排除项、剩余限制与下一步

激进网格组合和已有宽相位组压紧在完整切割中出现位置非流形边，均按失败排除，CPU复核一致。零贡献快捷路径与包围盒预过滤对照通过但端到端变慢，也未采用。失败与调度变更的因果关系尚未建立，不用“原来不稳定”覆盖失败，不放宽原检查。[完整实验记录](EXPERIMENTS.md)、[两项优化阶段报告](TWO_STAGE_REPORT.md)、[失败CPU复核](results/rejected-physical-runs.json)。

新几何重复不稳定、纯摩擦准备失败、wall器械碰撞、真实撕裂正对照未触发、连续S25绑定错位、随机化可见性及未来轨迹不确定性仍未解除。**formal_physics_domain_approved=false；training_domain_approved=false。** 本轮局部通过不等于整个物理域通过。[原S18–S25报告](../../REPORT.md)。

剩余主要预算在接触窄相位/见证构造、粗尺度材料累加、接触装配，以及完整RGB-D。下一步先对子阶段测量，再验证warp协作/数据布局或装配与顶点求解数据流原型，继续用逐记录和生产核对照约束。其收益尚未验证。图生命周期和中间读回可分析，但API长包络大部分覆盖GPU工作，不能把全部等待当成可删除预算。减少迭代、降分辨率、使用上一帧状态不属于本方案。

新版多worker吞吐和并发训练没有在本轮验收，不能把单环境Hz乘worker数。后续采样须隔离失败轨迹并记录env/episode/tick、几何/材料/执行配置、源码/库散列；两张A100的吞吐和任务分配需单独验证。

## 交付与复现

仅用指定空闲A100及资源守卫，未终止其他任务；未安装依赖或改变系统/网络/时钟/MPS。正式入口未替换，旧源码和参数完整性见 [复核](results/source-integrity.json)。最终DLL SHA256：{d['choice']['dll_sha256']}。

[README](README.md)、[补丁与散列](delivery/manifest.json)、[全部性能表](results/performance-table.md)、[最终统计](results/final-summary.json)、[逐阶段CSV](results/final-stage-times.csv)、[交错命令](all-queries-v2-commands.jsonl)、[最终图像/快照/时间线命令](all-queries-v2-final-commands.jsonl)。补丁只执行git apply --check，未应用到正式版。
"""
(p/'REPORT.md').write_text(text)
print('Final report written',len(text),'characters')
