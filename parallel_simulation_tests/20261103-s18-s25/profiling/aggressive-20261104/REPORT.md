> GitHub 归档说明：本页结论保持原样。部分逐帧 JSON 链接已改为带原文件 SHA256 的指标摘要；原始 trace、编译库、环境和大型快照仅保留在服务器。详见 [归档范围](../../GITHUB_ARCHIVE.md)。

# 实时性专项最终报告：瓶颈、激进优化与交互一致性

最终候选在两次完整1410帧测试中为 **21.45 / 21.42 Hz**，P95 **57.15 / 57.05 ms**。稳定20Hz验收：**未通过**。原正式代码未替换，既有物理域仍未批准。

同一负载时段、同线程设置的 V4/最终候选 ABBA 平均帧耗时：**64.05 → 46.65 ms**，总体速度比 **1.373×**。不将较早高CPU负载时的10.94Hz与后来的低负载成绩全差额归为代码收益。

## 到底哪里耗时最多

| 不重复阶段 | 同时段 V4 ms | 最终 ms | 最终占比 |
|---|---:|---:|---:|
| 原生物理、接触、切割及同步观测 | 31.527 | 31.338 | 67.2% |
| 质量检查及表面导出 | 9.307 | 4.273 | 9.2% |
| State构造与抓取证据 | 2.215 | 1.328 | 2.8% |
| 网格、完整RGB-D、对齐和CPU读回 | 19.578 | 8.399 | 18.0% |
| 控制及策略观测复制等 | 1.420 | 1.314 | 2.8% |

质量阶段已经包含表面导出，不能再把surface_export加一次。State构造不等于每帧保存完整原生快照。原生CPU API范围包含GPU执行和同步，不能再与GPU核时间相加。

![完整帧预算](figures/frame-budget.png)

## 完整性能与资源

| 测试 | Hz | P50 ms | P95 ms | 最慢20帧窗口 Hz | >50ms帧 | GPU利用率均值 | 显存峰值 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| final-v4-control-a | 15.57 | 65.04 | 75.76 | 12.32 | 1207/1410 | 52.4% | 2114 |
| final-rgbd-a | 21.45 | 47.62 | 57.15 | 16.52 | 503/1410 | 70.3% | 2120 |
| final-rgbd-b | 21.42 | 47.70 | 57.05 | 16.37 | 511/1410 | 70.3% | 2120 |
| final-v4-control-b | 15.66 | 64.88 | 74.79 | 12.29 | 1194/1410 | 52.9% | 2114 |
| final-checked-physics | 26.85 | 38.24 | 48.45 | 19.04 | 48/1410 | 82.8% | 1929 |

无图物理行仍包含原生物理、全部已启用检查、State及控制；纯原生阶段速度只是一帧内局部指标。RGB-D行包含每步新图像、原始/对齐深度、有效性及策略观测复制，已返回CPU；显示刷新、网络传输和磁盘录像未计入，不能据此宣称这些额外开销也已达标。初始化编译/重置不在稳态帧间隔内，运行中首次出现新几何的开销保留。

## CPU / GPU 时间线

最终Nsight节点级捕获为切割期470–481的12帧：wall-clock **66.215 ms/帧**；实际CUDA核区间并集 **35.923 ms**；GPU copy区间并集 **0.450 ms**；GPU memset **0.719 ms**；OpenGL设备工作区间 **4.686 ms**；CUDA与GL总区间并集 **41.779 ms**。这是有插桩的高负载切割窗口，不能当作1410帧平均FPS。

| 最大CUDA核 | ms/帧 | 次/帧 |
|---|---:|---:|
| `k_FindSurface` | 9.394 | 162.0 |
| `k_CoarseAccumulate` | 3.643 | 36.0 |
| `k_AssembleUnifiedInteractions` | 2.375 | 192.0 |
| `k_UnifiedVertex` | 1.714 | 96.0 |
| `k_UnifiedManifold` | 1.698 | 45.0 |
| `k_CoarseBound` | 1.628 | 36.0 |
| `k_MaterialStencilAssemble` | 1.103 | 3.0 |
| `k_UnifiedTool` | 1.061 | 27.0 |
| `k_RepairMinimumVolume` | 0.913 | 372.0 |
| `k_MgGalerkin` | 0.797 | 9.0 |

调用点对照捕获确认：每帧162次FindSurface中，135次来自剪刀路径，27次来自抓钳；相关实现见[当前原生接触核](native-gpu-diagnostics-v2/source/vendor/cuda/contact/surface_contact.cuh)。单独提前计算器械形状未带来整帧收益，不能将其视为这个热点已经解决。

CPU与GPU时间线：[SVG](results/analysis-final/timeline.svg)、[Perfetto/Chrome trace JSON](results/analysis-final/timeline.json)；CPU线程wall/on-CPU/exclusive时间：[cpu-spans.json](results/final-profile/cpu-spans.json)，完整Python剖析：[python-profile.txt](results/final-profile/python-profile.txt)。CPU线程ID和CUDA stream/context分开，不把重叠层叠加。

| CPU范围（包含子范围） | wall ms/帧 | on-CPU ms/帧 | exclusive CPU ms/帧 |
|---|---:|---:|---:|
| native.step | 43.210 | 43.208 | 0.298 |
| quality.all | 6.578 | 6.578 | 0.105 |
| quality.topology | 0.266 | 0.267 | 0.267 |
| state.snapshot | 1.804 | 1.804 | 0.513 |
| render.update_mesh | 1.305 | 1.306 | 1.306 |
| render.rgbd | 8.675 | 8.671 | 0.162 |
| render.gpu_post | 3.814 | 3.813 | 3.813 |
| vision.all | 11.761 | 11.747 | 1.770 |
| policy.observation | 1.616 | 1.617 | 1.617 |

OpenGL工作记录存在未覆盖的帧区间，图中的GPU并集表示已记录活动的覆盖；不能据此推算精确硬件空闲时间或SM占用率。“无CUDA活动”也可能包含GL工作，不能直接称为CPU计算时间。

on-CPU包含驱动忙等，不等于纯CPU数学运算。本轮未采集硬件性能计数器/CPU调度采样，不能凭寄存器数宣称已测得occupancy或显存带宽瓶颈。API期间设备覆盖率只是重叠证据，不是逐微秒等待原因的证明。

| CUDA CPU API | API wall ms/帧 | 同期间CUDA活动 ms/帧 | 无CUDA活动 ms/帧 |
|---|---:|---:|---:|
| cudaMemcpy_v3020 | 29.049 | 26.663 | 2.386 |
| cudaGraphExecDestroy_v10000 | 5.075 | 4.631 | 0.444 |
| cudaGraphLaunch_v10000 | 4.074 | 3.471 | 0.602 |
| cudaLaunchKernel_v7000 | 2.126 | 1.324 | 0.802 |
| cudaGraphInstantiate_v12000 | 0.867 | 0.000 | 0.867 |
| cudaMemsetAsync_v3020 | 0.517 | 0.322 | 0.195 |
| cudaFree_v3020 | 0.216 | 0.086 | 0.130 |

## 已实施与取舍

1. 精确几何与抓取融合：保留全部候选、退化分支、证人节点和阈值；不使用fastmath。CPU随机/边界/真实状态对照见[结果](results/cpu-equivalence.json)。
2. 每帧组织位置、法线、UV、包围盒融合；21组属性逐位一致，7.47→0.63ms。全部属性仍每帧更新，未采用上一帧布局。
3. 顺序三角形汤使用持久Primitive/GL缓冲；动态数量变化更新绘制数量和几何中心。7组输出逐像素一致；17.40→6.82ms是同输入微基准，不能当完整帧收益。
4. GPU直接输出训练RGB，NumPy分配独立连续数组，Numba单遍填充。保留普通交互render默认BGR。首轮Numba所有权标志失败已保留并修正；4组RGB-D逐位对照通过。完整控制17.62→19.26Hz，视觉14.08→9.09ms。
5. 当前表面GPU拓扑摘要：112组合成边界和1410真实帧CPU/GPU指标一致。私有缓冲修正后的同条件复测CPU平均50.563ms，GPU46.556ms；保留。读取当前重建，不改接触缓存。
6. 最后一次均匀形状预计算：验证结果`{"paired_calls": 960, "mismatches": 0, "positive_contact_records": 73528, "capture_skips": 414, "passed": true}`；未纳入最终库。未使用候选压紧或工具形状假设。

GPU拓扑初版曾因新临时缓冲被自动纳入快照而失败。逐分配对照证明仅5个新增派生缓冲变化，旧物理分配和全部host字段不变；V2令新临时缓冲独立管理，未修改state_archive代码/ABI或排除任何旧物理字段，并重新通过严格完整快照对照及1410帧CPU/GPU对照。根因见[archive-v1-root-cause.json](results/archive-v1-root-cause.json)。

同帧CPU线程重叠、GL nogil、PBO、固定背景更新未带来可靠完整收益，已从最终最小源码移除。抓钳特化/压紧无可靠收益；主要剪刀压紧版960次中201次严格接触对照失败，未采用。所有实验版本只留在归档目录，详见[实验记录](EXPERIMENTS.md)。

## 正确性与交互边界

最终116项CPU接口回归通过；32帧真实闭合切割回归通过，cut_onset=True；4次当前状态RGB对照逐位一致，深度误差≤1e-6m、有效性一致；旧观测未被后续帧覆盖。两次完整1410帧保留原有检查且通过其局部范围。参数、动作空间和验收阈值逐项相同，见[参数对照](results/final-parameter-equivalence.json)。

真实切割状态的完整环境快照/恢复额外通过：重复GPU检查不改快照，继续执行后可逐位恢复原生状态，恢复后检查及下一步正常；新Python辅助模块均绑定散列。见[真实快照回归](results/snapshot-roundtrip-final/result.json)。这不证明未来轨迹逐位确定。

GPU拓扑库保留原308个设备函数SASS逐位一致；最后查询实验的参考核也与原FindSurface SASS一致，其余307个原函数不变。完整ELF及host relocation散列不同，不声称整个库字节相同。
实际最终库：`/srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-s18-s25/profiling/aggressive-20261104/native-gpu-diagnostics-v2/build/lib/libLiverCudaSim.so`，SHA256 `24a8c12cb8b47684558ac637e698e27ab19e443995f800ec2e2f01a1379c5a54`。

这些验证不能替代尚未通过的物理域验收：新几何重复不稳定、纯摩擦准备失败、wall器械碰撞失败、真实撕裂分离正对照未触发、连续S25支撑绑定数量错位、随机化可见性失败、A/A未来轨迹不确定性仍未解除。`formal_physics_domain_approved=False`。原有失败见[原S18–S25报告](../../REPORT.md)与[上一轮Profiling报告](../REPORT.md)。

密集S25每像素支撑ID/权重仍单列：CPU展开每1280×960帧约252MB，且有连续绑定错位。当前RGB-D性能表没有启用该原基准未启用的特权标签；不能据此说完整密集标签达到20Hz。

## 剩余瓶颈及下一步

先看[按动作阶段的逐段耗时](results/final-stage-times.csv)和[逐帧原始数据](results/final-rgbd-a/result-summary.json)。平均超过20Hz仍不能掩盖切割/密集接触期的慢帧。接下来的主要预算必须从原生接触与形变求解取得；保持原3子步、8次接触细化、材料参数、查询数量及切割规则，先解决导致严格对照失败的具体差异，再做接触查询数据布局和重复计算消除。现有CUDA Graph已经使用；不能把再次开启Graph当成新方案，图销毁/启动的API延迟也须结合重叠设备活动评估。

建议的下一轮顺序是：先对FindSurface的多种器械查询合并调度做最小原型，保留全部查询并继续逐记录对照；再分析CoarseAccumulate/UnifiedInteractions的数据访问及融合空间；最后评估动态表面数量导致的图重建。当前图销毁5.075ms中4.631ms与CUDA活动重叠，单看API时长会高估可省预算。以上尚未实施，不作为本轮收益。

CPU负载在本轮从约60%降到约12%，无新RGB优化的同版本控制亦明显变快。已经用最终同线程ABBA重新对照；绝对帧率仍受共享主机影响，未测量到足够证据区分频率、缓存/内存竞争和调度各自贡献。本轮未改时钟、cgroup、MPS、系统权限或网络。

多worker不能直接把本单环境Hz乘以数量，也不自动缩短单帧延迟。本轮优先解决完整交互关键路径，仅使用一个有资源守卫的A100；没有进行新版多worker吞吐验收，不据此承诺并行环境数。

## 交付与复现

最终最小候选：[final-source](final-source/ASSEMBLY.json)。补丁与散列：[delivery/manifest.json](delivery/manifest.json)、[Python补丁](delivery/python-from-v4.patch)。补丁只做`git apply --check`，未应用到正式代码；其相对根为V4的endoscopic_tissue目录。原源码/库完整性复核见[source-integrity.json](results/source-integrity.json)。

使用现有环境执行`tools/reproduce_final.py --output results/一个新目录名`；脚本检查输出目录、源文件/库散列和GPU守卫。完整命令记录：[final-commands.jsonl](final-commands.jsonl)。所有完整、失败、资源中断、微基准、验证和插桩结果保留，完整性能表见[performance-table.md](results/performance-table.md)。未安装或升级依赖，未替换正式入口。
