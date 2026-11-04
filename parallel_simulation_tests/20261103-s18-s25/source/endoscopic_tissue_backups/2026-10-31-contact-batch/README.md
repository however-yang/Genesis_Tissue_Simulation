# 剪刀批量接触实验与回归工具（2026-10-31）

所有候选独立基于 dev 39e5ebc655fde6f77259783fecaa43bda78de2d0，未叠加腕部解析或线程块共享补丁。默认DLL和生产CUDA源保持基线。**本目录是未采用的实验，不是默认运行配置。**

[完整结果报告](D:/Simulation/genesis-world/endoscopic_tissue/docs/contact-optimization-trials-2026-10-31.md)记录正常计时、交互差异、未解释异常和剩余工作。[evidence-summary.json](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-31-contact-batch/evidence-summary.json)由原始输出重新生成，含各次/各阶段均值、P95、最大值、质量失败项和视频元数据。

## 三个独立补丁

|补丁|实现|结果|
|---|---|---|
|batch-candidates.patch|一次准备器械界、一次三角面多部件粗筛；抽取查询体；128×32有界grid-stride消费全部候选|同输入接触权重出现差异，未进入正常性能或视频筛选|
|indexed-candidates.patch|相同批量前端，恢复原查询核函数体，只增加连续候选索引|精确输入及全流程密集抽样0差异；正常性能无采用依据，有一次异常慢基线|
|spread-candidates.patch|相同批量前端，以block+thread×gridDim的索引分散候选|精确输入及密集抽样0差异；核心43.238→42.745 ms，流水线54.320→54.574 ms；不采用|

两个修正版仍启动完整表面规模的窄阶段网格，不将它们称为有界工作队列。无候选数量截断，粗筛保留AABB外持续刀刃预载，缓存随每次调用重算。接触、材料和迭代参数没有调整。

## 数值诊断与保存的失败输入

batch-equivalence在step720发现2条记录差异；debug-dense-1在step714保存精确输入（冻结检查在该LCS_Step之后执行，流程report的最后完成步为713）。kind=5、triangle=30072、slot=180437、zero-based word26，即weight[0]：0.0504300036 vs0.0504295416。

失败输入位于D:/Simulation/genesis-world/endoscopic_tissue/outputs/contact-batch-20261031/debug-dense-1/failure-contact.bin，SHA256为78f001e5c7889b161c6f2722bb75e64200d7e5f962d50a485553d72e10661258。它包含原始GPU状态/缓存，只保存在本机，不提交大体积二进制或缓存未使用字节。snapshot-manifest.json保留来源及重放期望。没有该本机文件时，可以运行密集诊断重新采集；并行非确定性意味着不能保证再次在同一步遇到同一输入。

replay_contact_snapshot.py通过导出函数将同一输入分别送入普通路径、候选窄阶段和完整批处理。失败版三者各1条记录差异，接触数量一致；indexed/spread三者均0差异。由此定位到查询重构数值变化，未证明具体的FMA/指令原因。contact_equivalence.cuh还检查所有有效接触记录的完整字节、degree、活动成员与数量；不比较未使用记录的任意字节。

270个冻结状态覆盖step600–810逐步检查，其余每15步检查。indexed累计153,745,696个三角面查询，spread累计153,707,072个，均0差异。这些是同输入验证，不能宣称完整动力轨迹等价。

## 复现方式

使用现有genesis_world环境、CUDA12.5、MSVC及项目构建脚本。不要在有未保存修改的主工作目录叠加候选。建立基于上述commit的独立checkout/worktree，每次只应用一个补丁，并显式给build_native.cmd提供独立输出目录；省略该参数可能覆盖原单刀DLL。

1. 正常候选：git apply 对应补丁；运行该checkout中的endoscopic_tissue/native/build_native.cmd，输出到新的bin目录。正常性能DLL不添加任何诊断include。
2. indexed/spread接触诊断：在诊断checkout的xpbd_solver.cu末尾include本目录contact_equivalence.cuh（使用绝对路径或复制整个相邻头文件集合）。该文件包含reference_find_surface.cuh及contact_snapshot.cuh。另构建独立probe DLL。
3. 原有界版本重放诊断：使用同一contact_equivalence.cuh，但在独立诊断头文件目录将first_contact_snapshot.cuh复制为contact_snapshot.cuh，再include该目录的contact_equivalence.cuh。first_contact_equivalence.cuh是首次未增加快照的检查器，不包含重放接口。
4. graph-counter.patch独立添加已有图选择的CPU计数接口。spread密集检查使用了它并运行check_contact_graphs.py；其首行包括初始化3次选择（全程3333），不作为性能计时。只测基线慢段的bin-graph-baseline31则应用此补丁而不应用接触候选；measure_solver_graphs.py在初始化后清零（3330次）。
5. 对probe运行check_contact_snapshot_dense.py或check_contact_graphs.py；随后使用上一轮run_batch.py按ABBA清单运行正常DLL。计时、诊断、捕获、渲染串行分开。所有输出目录必须是新目录，修改清单中的output和DLL路径后复用。
6. 使用腕部目录的capture_motion.py记录实际状态，随后replay_single_blas.py、make_review.py和review_motion.py生成完整对照。run_indexed_screen.py/run_spread_screen.py保留本次确切调用顺序，但路径指向现有输出；重新运行前必须指定新的输出目录和清单，不能覆盖本次证据。

下列归档前核验可直接在本机已有产物上运行，第一项不运行物理模拟，第二项串行运行短CUDA回归：

~~~text
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-contact-batch\verify_artifacts.py
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-contact-batch\verify_gpu_regressions.py
~~~

verify_gpu_regressions.py明确要求失败版返回exit=1、修正版返回exit=0，避免“测试没有覆盖故障”的假通过。gpu-regression-verification.json为本轮重新运行结果；1698组腕部几何也重复验证。短回归不能代替完整交互录像和多场景采用级验收。

## 性能异常与交互限制

连续索引的最后一次基线后半程异常变慢，不能将99.572 ms均值混进加速比。原记录保留。图选择诊断复测1110步中recovery=0、最大64.46 ms，没有复现异常，也没有触发改变Graph路径的分段计时；先前原因仍未确认。

分散索引版录像第二剪肿瘤区域RMS速度峰值42.619 mm/s，基线26.775；夹爪开度和双刃载荷也有差异。本轮各实际录像的质量验收仍failed。完整1111帧录像、运动曲线和报告已保留，未因为冻结输入一致而宣布交互通过。

视频分别位于endoscopic_tissue/outputs/contact-indexed-20261031/indexed-review和contact-spread-20261031/spread-review。它们与本轮腕部目录中新录制、相同原DLL及设置的baseline-video比较。30 fps表示模拟时间播放，不表示执行速度。肾脏release阶段只是剪刀张开，夹爪并未松开，不能用该录像证明松夹后的自由回弹质量。

本轮没有采用候选，也没有启动更大范围的采用级多场景重复。联合粗模态没有实现，具体交叉项、三次体积界和夹持/剪刀能量检查缺口见主报告。
