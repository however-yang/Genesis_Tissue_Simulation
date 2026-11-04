# 隔离交互修复实验

本目录只构建实验 DLL，不修改默认 DLL、启动参数、材料参数或摩擦系数。release-r16 和 release-r17 均未通过普通阶段的性能验收；vertex-r18 的纯摩擦均值及尾部耗时也未通过。release-r19-volume-gate 的松夹穿入显著改善，但普通阶段核心均值与最慢原生路径仍未达到不退化要求。六个构建选项均不能作为默认版本。

## 构建

在仓库根目录运行：

~~~bat
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-interaction-fixes\package\build_experiment.py --variant release-r16
~~~

脚本从固定 Git 提交 38e6082edceef7898ab5ce8ca38a48ddcb175264 提取原始源码，检查覆盖文件的 SHA-256，然后在 artifacts/release-r16 内构建。它使用本机已安装的 MSVC 2022、CUDA 12.5，目标 GPU 架构为 86。构建日志、源文件清单与二进制哈希均保存在该目录。可先使用 --prepare-only 检查源码重建。目录内的 .gitattributes 保留覆盖源码和数值样本的原始字节，避免 Windows 换行转换破坏已记录的哈希。

可选版本：release-r19-volume-gate（基于R16的保守空体积查询门控，仍未通过验收）、release-r16（接触修复及空查询调度）、release-r17（额外压力能量专门化，性能仍未通过）、vertex-r18（基于R16的局部矩阵精度修复）、baseline-rim-trace2（原始源码的只读数值诊断，不能用于计时）。另可构建 interaction-r13-trace，以复现原生接触力和切割快照；它含只读诊断开销，不能用于性能比较。这些版本均使用同一构建命令的 --variant 参数。覆盖文件可引用另一个版本的相同文件，均单独校验哈希。

## 实验修改

游离组织接触进入原有隐式材料目标，向接触双方施加成对作用，保留体积保护。预测接触在同一子步内保留，材料校正后重新检查实际表面，避免“投影推出后又被体积约束拉回”的交替穿入。空间树与完全相同的嵌入点查询共享降低几何检测成本，每个原始接触的面积、刚度和阻尼保持独立。

R13 的空查询求解专门化经过七个原生状态、每个28项私有缓冲区逐字节核对。R14–R16只调整主机调度；与R13共301个GPU函数的指令哈希一致。R16仅在严格为空的不可变查询列表下跳过额外接触缓存初始化；非空查询仍执行全部原有接触检测与求解。

R17的70项私有缓冲区压力能量对照均保持相同的步长选择，能量差在指定浮点求和舍入范围内。R18只修改两个顶点求解CUDA内核并新增一个局部精确分解函数；其余299个GPU函数与R16指令哈希一致。R18没有采用R17的压力能量专门化。

以上检查不能替代完整交互验证。相同算例的并行累加与断裂触发仍有轨迹差异，不能将单个冻结状态的一致性宣称为整段轨迹逐位一致。

## 验证规则

- 材料、网格、时间步、迭代预算、摩擦系数与基线相同。
- 完整夹取、两次切除、侧拉、真正松夹和撤离；录像保留所有计算帧。
- 对松夹后的每帧做独立表面穿入检查；保留四面体、拓扑、刀刃重叠和未断连接的失败记录。
- 自由下落仍记录绝对位移；只有无固定点、无夹持且仍含完整内部材料格（八个角点、十二条双向边均存在）的独立块可按刚体运动分离后的形变另行检查。连接主体及孤立支撑节点继续使用原保护。
- 独立计时不录屏、不压缩、不运行几何审计；物理试验与独立计时串行，基线与实验版交错。全部物理试验结束后，已有状态的录像可并行导出，不运行求解器。
- 视频按仿真时间30帧/秒播放，不代表实际计算帧率。
- 只有效果与性能均通过才考虑替换默认版本；此包没有任何替换操作。

最终结果见[交互修复报告](D:/Simulation/genesis-world/endoscopic_tissue/docs/interaction-fixes-2026-10-31.md)；原始录像和大体积快照保存在 endoscopic_tissue/outputs/interaction-fixes-20261031，未加入 Git。

## 复现完整交互

仍在仓库根目录执行，输出目录必须尚不存在，避免覆盖已有录像和证据：

~~~bat
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-interaction-fixes\validate_strict.py --case kidney-reference --dll endoscopic_tissue_backups\2026-10-31-interaction-fixes\package\artifacts\release-r16\bin\LiverCudaSim.dll --output endoscopic_tissue\outputs\manual-r16-review --capture --allow-detached-flight
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-grasp-cut-validation\render_review.py endoscopic_tissue\outputs\manual-r16-review
~~~

薄壁完整补剪使用 record_wall_diagnostic.py --case wall-tip-completion-wide；纯摩擦中央咬合使用 record_friction_diagnostic.py --case grasp-central。均显式传入 --dll、--output 和 --capture。record_friction_diagnostic.py 遇到未包含只读导出函数的正式实验DLL时仍录制完整轨迹，但不生成力快照。

独立计时用 time_full_interaction.py；额外薄壁/夹取路径用 time_with_fixtures.py。不要与其他GPU试验、渲染或编译同时运行。run_*_gate.py 是本次会话的串行调度记录，含已结束的进程身份，不能当成通用启动脚本重新执行。

真实矩阵回归输入位于 evidence/numerics，另有896个固定随机种子的正定矩阵。test_solve_probe_actual.py 测试从R18实际头文件提取的求解结构；对应独立CUDA文件为 solve_probe_actual.cu。build-solve-probe-actual.cmd 使用本机CUDA/MSVC路径，仅构建测试DLL。

录像状态归档支持 replay.bin 和无损 replay.rdx。后者按前一完整网格帧建立字典，重建后的原始二进制长度与 SHA-256 均逐文件验证，全部帧保留；render_review.py 自动读取两种格式。读取字典归档需要本环境已有的 zstandard。归档仅节省存储，不计入求解性能。

release-r19-volume-gate 从R16独立分支，仅在已确认不存在未固定体积连通块时跳过 mode=2 的游离块查询读取；非空候选执行原检测。已有体积标记内核顺便维护一个保守标记，不减少材料或接触求解。只读导出可逐状态比较快路径与原路径排序后的完整查询集合。此版完整松夹的21次查询集合检查通过，77帧最大穿入0.0161毫米；性能筛选仍未通过，保持实验状态。使用 --capture --archive-capture 可直接保存逐字节可还原的全部交互帧，避免产生庞大的临时原始录像流。

## 结果核验

`evidence/interaction-evidence-index.json` 列出各段轨迹的参数、DLL哈希、原验收失败、几何审计和视频完整性。`evidence/visual-review.json` 单独记录实际看过的分镜和逐帧范围。完成录像或通过查询集合检查不等于通过效果验收。零摩擦负对照的运行时参数覆盖在索引中单列，原始设置快照仍保留。

`compare_timings.py` 检查四次ABBA运行的场景与设置一致，并保留GPU状态、原始报告及逐帧样本哈希；完整样本留在本机outputs目录。R18和R19的数值表包含均值、P95和最大帧，不把录像压缩过程的耗时当成求解性能。

已执行 `collect_evidence.py --require-complete --hash-videos`：30条记录对应录像均通过帧数及SHA校验，29条轨迹完成、1条基线失败中止保留；9项默认保护哈希全部一致。录像入口见[完整录像索引](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-31-interaction-fixes/package/video-index.md)。该命令可重复核对证据完整性，不会升级任何实验版本。
