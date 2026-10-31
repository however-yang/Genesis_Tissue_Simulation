# 材料加性多层与接触分组实验（2026-10-31）

独立起点：dev fbcaf71dc0eec30e00b55d46bf5ae0eddac271b3。此目录是隔离实验，不能据此替换默认 DLL。上轮联合粗模态两个版本不采用，本轮不叠加它们，也不把本轮两种算法叠加。

## 补丁与构建

- additive.patch：加性多层预条件器；保持原材料目标、Galerkin 层次、3/7 次 PCG、接触、工具、可行性和线搜索。additive_multigrid.cuh 中的细层 3×3 求逆使用 double 中间量。
- additive-probe.patch：同一算法加冻结层次导出与局部矩阵回归接口。正常计时不使用它。
- contact-groups.patch：每 64 个三角形的保守分组界；每次表面点更新后重建，夹爪/剪刀各次查询重新分类；原窄阶段仅增加可选早退。旧刀刃预载绕过分组早退，盒形器械界同时包括端点 AABB 与包围球。
- contact-groups-probe.patch：同一算法加独立参考、完整有效接触记录/degree/成员比较，以及与原逐三角形 AABB 的保守性检查。

使用现有 genesis_world 环境、项目 CUDA/MSVC 构建脚本。先在独立 checkout/worktree 中检出上述起点，并把本实验目录的头文件和补丁放到相同相对路径；每次只应用一个补丁。必须显式指定新的 DLL 输出目录，不能省略 build_native.cmd 的参数。

~~~text
git apply endoscopic_tissue_backups/2026-10-31-material-contact-next/additive.patch
endoscopic_tissue\native\build_native.cmd D:\Simulation\genesis-world\endoscopic_tissue\native\bin-additive-reproduction
~~~

在独立 checkout 中运行时，将输出路径改为该 checkout 下的新目录。正常版、探针版分别构建，构建后核对 git diff 和 DLL 哈希。build-manifest.json 记录本机实验 DLL/头文件/补丁哈希，二进制不提交 Git。

## 数值诊断与失败证据

初版使用原 float 细块求逆，在第 800 步的冻结层次检查超过原定 1e-4 相对误差阈值。initial_additive* 保存该实现；additive-error-localization.json 和 diagonal-frozen-inputs.json 保存误差定位及真实切口块。重建初版时，将 initial_additive_multigrid.cuh / initial_additive_probe.cuh 复制为对应非 initial 文件名，再应用 initial 补丁；不要在已经应用正常补丁的 checkout 上叠加。

check_additive_diagonal.py 调用 GPU 上的原求逆与新求逆，独立 NumPy double 求解作为参考，含 12 个冻结切口块、1024 个种子 SPD 块及两个退化/非正定块。其通过条件同时要求原求逆暴露误差、新求逆达到既定精度，没有放宽原检查阈值。不同编译上下文的原浮点误差不保证完全相同。

check_additive.py 在十个实际动态状态冻结层次和 CG 掩码，使用私有 scratch，不修改真实求解向量；CPU 独立递归检查输出、对称性、正二次型和固定节点。材料候选有限次 PCG 的轨迹不因此与基线等价。

check_contact_groups.py 在 270 个状态检查夹爪与剪刀、静止与移动两种查询，其中 600–810 步逐步检查，其余每 15 步检查。独立 reference_find_surface.cuh 保留原查询和原内联局部搜索；正常候选没有这些诊断开销。若失败，contact_snapshot.cuh 保存该查询，并追加 GRP1 尾部（完整组几何、组界、粗筛标志），原 batch 实验的重放器不能直接作为这份新尾部的完整验证器。

## 正常性能与录像

run_additive_trial.py / run_groups_trial.py 串行安排诊断、普通 CUDA Graph ABBA、实际状态捕获、渲染、完整并排录像和运动曲线。输出目录已存在时会拒绝覆盖。复现前改为新的输出根目录和清单；run_batch.py 不把 exit=1 的质量失败当成通过，外层还强制检查 1110 步完整性。

计时脚本沿用 2026-10-30-perf-opt/measure_physics.py；完整状态捕获沿用 2026-10-31-wrist-opt/capture_motion.py。录制包含初始态和每个步长，共 1111 帧，30 fps 是模拟时间播放速度，不是实测运行帧率。两项实验各有新录制的基线和候选。pipeline 包括控制、读回、记录提交及可能的背压；不含实时窗口显示、阶段边界检查或最后排空，不能宣称端到端 GUI 帧率。

run_sections.py 在全部正常计时结束后单独测三个版本的模块耗时。此模式关闭 Graph，含额外发射间隙，cg assembly 子项与外项重叠，不能全部相加或作为实时性能。

诊断入口沿用已有 kidney dev_run 的拓扑记录模式：原 non-manifold 计数另存 recorded_nonmanifold_edges，继续完成流程；不代表拓扑通过。最终 quality failed 保留。kidney 的 release 阶段是剪刀张开，夹爪仍保持夹持，不能用它证明真正松夹后的自由回弹质量。

## 核验

~~~text
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-material-contact-next\verify_artifacts.py --sources-only
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-material-contact-next\verify_artifacts.py
~~~

完整核验读取本机原始结果，检查默认源码/DLL、七个补丁、八次计时、四份实际捕获、两份完整对照、三次模块诊断及有界慢路径复现；逐帧解码全部六个视频。缺少本机输出时，需要先复现，不能凭 Git 中的小型汇总宣称测试通过。

最终结果见 [实验报告](D:/Simulation/genesis-world/endoscopic_tissue/docs/material-contact-experiments-2026-10-31.md) 与 evidence-summary.json。验证范围和未通过项均须保留，不用加速数字代替软组织交互真实性验收。

加性层次公式参考 [NGSolve 官方多重网格教程](https://docu.ngsolve.org/latest/i-tutorials/unit-2.1.3-multigrid/multigrid.html)。本项目含切割、非线性材料和接触，教程中的椭圆问题收敛结论不能直接转用。

contact-groups-graphs.patch 是额外的 CPU loaded/free/recovery 计数版；measure_solver_graphs.py 在第二剪后连续三步超过 100ms 时采集 GPU 状态，并临时开启四步 section profiling。run_slow_graph_diagnostics.py 最多复现三次，抓到连续慢段后停止追加。它们不属于正常 ABBA 性能，不删除原始 groups-b1 异常记录。

初版完整冻结层次 additive-0800.bin 仅保存在本机输出中，failure-snapshot-manifest.json 记录路径、大小、SHA256 和失败阈值。Git 中保留 12 个真实切口块用于短 GPU 回归；不同动态运行不保证在同一步触发同样误差。

make_playable_videos.py 额外导出两份 H.264 yuv420p 兼容播放副本，并核验全部 1111 帧；原始 comparison.mp4 和实际状态记录都保留。playback-videos.json 记录原文件与导出文件的哈希、帧数和编码设置；完整核验也会逐帧解码这两份副本。

复现顺序：完成两套 trial（含模块与慢路径诊断），再运行 make_playable_videos.py，随后运行 verify_artifacts.py 和 write_report.py。四个脚本的 DATA 根目录及各清单应一致指向新的输出目录。
