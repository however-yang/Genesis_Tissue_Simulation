# 2026-10-31 物理/接触性能实验

本目录保存诊断工具、原始实现的独立参考、计时摘要及**没有进入默认运行路径的实验补丁**。本轮起点为 dev 的 `fafb85a75e43e84de04c7d79e92e38c22c9849c8`，默认剪刀库 SHA256 为 `0d09f56a1f4acb813424cd1dee3233d89ce7ef5c0d45e64f079c7d7387912225`。

结论与限制见 [实验报告](../../endoscopic_tissue/docs/performance-physics-contact-2026-10-31.md)。`results-summary.json` 是十二次完整正常 CUDA Graph 运行的摘要，保留真实质量失败。大体积录像、逐帧数据和实验 DLL 仅保存在本机 outputs / native 的对应目录。

## 工具

| 文件 | 用途 |
|---|---|
| `check_contact_equivalence.py` | 完整肾脏流程每 15 步调用冻结输入原生比较，差异即停止 |
| `contact_equivalence.cuh` | 只读实时状态，在独立 scratch 上查询原版和候选，比较有效记录、degree、列表成员 |
| `reference_find_surface.cuh` | 从起点冻结的原始 FindSurface 及其 triangleWitness / traceTriangleBox；不使用候选的同名 helper |
| `accumulate_equivalence.cuh` | 直接归约的整数精确和压力检查，6 类目标分布 |
| `partial_accumulate_equivalence.cuh` | 仅与补丁 05 一起使用，验证完整两阶段归约 |
| `check_accumulate.py` | 重复原生整数精确和验证 16 次，任一差异即失败 |
| `capture_motion.py` | 无 OpenGL 的完整状态记录，含每步三块材料区域的质心和运动速度；不用于性能计时 |
| `review_motion.py` | 检查 1111 个连续状态、相同阶段，绘制动态 A/B 曲线，不虚构“真实感阈值” |
| `summarize_experiments.py` | 汇总本轮 12 个正常计时目录，并检查 Settings 和步数一致 |
| `verify_evidence.py` | 收尾验证原生源码/默认库保护、完整计时、冻结输入检查和视频证据 |

完整回放渲染、并排视频合成和正常图执行计时沿用 [上一轮工具](../2026-10-30-perf-opt)。视频以模拟时间 30 fps 播放；运行速度必须读取计时文件。

## 补丁

所有补丁分别相对于本轮起点，不能叠加后冒充单项实验。`01` 跳过胶囊不使用的保留盒平面，`02` 禁止搜索 helper 内联，`03` 限制固定循环展开，`04` 合并归约地址选举，`05` 两阶段归约。它们均未采用。`diagnostic-fine-sections.patch` 仅增加细分事件标记，不是性能候选；该模式关闭正常图调度，只用于热点定位。

复现前在独立工作目录检查补丁。仅检查不会修改源码：

```powershell
git apply --check endoscopic_tissue_backups/2026-10-31-physics-opt/patches/05-two-stage-reduction.patch
```

需要构建时始终给 `build_native.cmd` 明确的独立输出目录。脚本省略参数会指向原始单刀库，禁止这样调用：

```powershell
& .\endoscopic_tissue\native\build_native.cmd D:\Simulation\genesis-world\endoscopic_tissue\native\bin-reproduction
```

接触 probe 构建仅在实验副本的 `xpbd_solver.cu` 末尾追加 `#include "D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-31-physics-opt/contact_equivalence.cuh"`；引用路径随实际目录调整。整数归约 probe 同理追加对应 `accumulate_equivalence.cuh`。诊断与正常候选必须编译到不同目录；正常计时前移除诊断 include，确保编译日志确实重编 `xpbd_solver.cu.obj`，而不只是重链接旧诊断对象。恢复文件后也要注意修改时间晚于构建对象。

## 运行

GPU 运行串行，不同时编译、渲染或运行其他 GPU 工作。以下路径中的输出目录必须是新目录。正常计时示例：

```powershell
& D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups/2026-10-30-perf-opt/measure_physics.py endoscopic_tissue/outputs/physics-opt-reproduction-baseline --scene kidney --tool scissors --dll endoscopic_tissue/native/bin-scissors/LiverCudaSim.dll
```

记录完整交互及动态数据：

```powershell
& D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups/2026-10-31-physics-opt/capture_motion.py endoscopic_tissue/outputs/physics-opt-reproduction-capture --scene kidney --tool scissors --dll endoscopic_tissue/native/bin-scissors/LiverCudaSim.dll
```

渲染每个真实求解状态：

```powershell
& D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups/2026-10-30-perf-opt/replay_single_blas.py endoscopic_tissue/outputs/physics-opt-reproduction-capture/replay.bin endoscopic_tissue/outputs/physics-opt-reproduction-video
```

这些完整肾脏流程沿用已有 `dev_run.py` 记录裂尖非流形边以完成运动，最终额外表皮切割等质量失败仍会导致 exit 1。必须同时读取 report 的步数、阶段、metrics、surface-audit；不能把所有 exit 1 都当作已知失败忽略。任何新候选只有通过完整性能和效果验证后，才进入采用讨论。
