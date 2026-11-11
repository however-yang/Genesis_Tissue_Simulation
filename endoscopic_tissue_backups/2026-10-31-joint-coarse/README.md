# 联合粗模态实验（2026-10-31）

本目录保存独立实验及失败证据，**没有替换默认求解器**。起点为 dev `0403d319097ca77099edc5c33d51b2d22d4312a0`。原始 main、生产 CUDA 源、默认剪刀和单刀 DLL 保持原样。

完整结果见[实验报告](D:/Simulation/genesis-world/endoscopic_tissue/docs/joint-coarse-experiments-2026-10-31.md)。[evidence-summary.json](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-31-joint-coarse/evidence-summary.json)保存每次/每阶段计时、真实录像质量指标、输入与产物摘要及哈希。两项实验均未获得性能和交互的采用依据。

## 实现及文件

四个现有标量粗方向各有三个平移自由度，组成每个切断连通分量的 12 维系统；联合计算 12 个梯度和 78 个对称交叉曲率项。材料项使用原梯度与 GN 正半定搜索曲率，含客观黏性、压缩屏障和拉伸硬化；不是精确不定 Hessian。保留完整细层顶点、材料 CG、工具耦合、接触与切割。只替换 it=0 的四次粗修正，it=2 保持原修正。失败分量回退原四次修正。

|文件|用途|
|---|---|
|joint_coarse.cuh|当前带实际存储端点认证的联合求解器|
|joint_diagnostics.cuh|诊断专用，检查每次接受后的实际位置及局部接受条件|
|joint_probe.cuh|实际 GPU 代数、三次路径、浮点端点探针|
|joint_fallback_gate.cuh|第二项实验，跳过所有分量均接受时的旧粗层空遍历|
|initial_joint_*.cuh|未修复浮点端点问题的最初版本，用于重现失败，不能部署|
|initial-joint.patch / initial-joint-probe.patch|最初普通版/诊断版的独立集成补丁|
|joint-coarse.patch / joint-probe.patch|带存储端点认证的普通版/诊断版独立补丁|
|joint-gated.patch / joint-gated-probe.patch|第二项带回退调度的普通版/诊断版独立补丁|

六份补丁**分别**应用于同一个起点，不能叠加。包含路径指向本目录，头文件与补丁必须一起保留。第二版 16 个包装核函数去除名称、额外 gate 参数及入口条件后，与原函数逐字一致；`verify_artifacts.py`重新核验这一点。全部分量接受时，用 GateFinish 保持后续需要的 coarseMode/coarseWeight/globalCoarse 状态；跳过的安全分组 scratch 在下一次读取前重新初始化。

## 严格数值检查

- `check_joint_algebra.py`：180 个四面体，实际 CUDA 输出与独立 NumPy 双精度雅可比及原单模态对照；2004 个三次路径用例，包括端点安全但区间内部越界的案例。
- `check_joint_rounding.py`：12000 个可复现浮点端点用例；旧多项式路径接受 704 个实际存储端点低于下界的案例，新存储认证接受 0 个。首个完整输入保存在结果 JSON。
- `check_joint_dynamics_strict.py`：每步检查所有联合接受事件，负体积余量必须为 0。旧 DLL 在 step262 失败，新两版各完整 1110 步为 0。
- `check_joint_dynamics.py`：保留最初诊断器；其旧 volume_violation 容差不能作为当前通过标准。最初完整运行发现 70 个微小负余量，促成严格回归。

修复保留整段三次 det 路径检查，并额外物化一次真实浮点试探位置、检查实际存储 det 和接触闭合界，通过后复制同一位置写回。没有放宽体积下界。旧案例说明了多项式与浮点存储端点之间的数值缝隙，没有证明每个动态事件具体由哪一种编译器变换造成。

局部线搜索中的接触压力和法向保持当前线性化，摩擦为滞后耗散项；这不等于全系统保守势能，更不等于交互真实性验证。动力学诊断 DLL 会额外同步，不能用于性能结论。

## 复现构建

使用现有 genesis_world Python、CUDA 12.5、MSVC 2022。应在干净的独立 checkout 中恢复上述起点，再复制本目录所需文件；不要直接覆盖有用户修改的工作树。所有构建显式指定不同的输出目录。`build_native.cmd`省略参数会输出到原单刀 bin 目录，不能省略。

例如在独立 checkout 的根目录应用 `joint-gated.patch` 后，用该 checkout 的构建脚本和新输出路径构建：

~~~text
git apply endoscopic_tissue_backups/2026-10-31-joint-coarse/joint-gated.patch
endoscopic_tissue\native\build_native.cmd D:\YOUR_ISOLATED_CHECKOUT\endoscopic_tissue\native\bin-joint-gated-new
~~~

诊断版在另一个干净 checkout 应用 `joint-gated-probe.patch`，输出到新的 probe 目录。普通版不添加诊断代码。重现最初失败时，需将 `initial_joint_coarse.cuh`、`initial_joint_diagnostics.cuh`、`initial_joint_probe.cuh`分别复制为该独立实验目录的 `joint_coarse.cuh`、`joint_diagnostics.cuh`、`joint_probe.cuh`，再应用 initial 补丁。不能将 initial 补丁与当前修正头文件混合后称为原失败版。

`build-manifest.json`记录本机六个实验 DLL、两份保护 DLL 与实验源的 SHA256。源码哈希是本机原始字节哈希，Git 换行归一化后重新 checkout 可能不同；比较时应区分换行和实际代码修改。DLL 位于独立的 bin-joint-coarse31* 目录，不提交二进制。

## 实验顺序与输出

`run_trial.py`保留第一项的正常 Graph ABBA、基线/候选真实状态捕获、渲染与对照调用顺序；`run_gated_trial.py`保留第二项的代数、严格诊断、ABBA、候选捕获与对照。第二项录像使用同一轮新录制的基线录像。GPU 作业串行运行，计时与录像分开。

这些脚本和清单指向本轮已有输出。**重跑先修改为全新的输出目录和新清单，不能覆盖本次证据。** exit=1 可能来自完整流程最后的原有交互验收失败；脚本另行要求全部 1110 步完成。严格诊断还要求安全计数为 0，因此不能用允许 exit=1 跳过中途断言失败。

本机输出：

- `D:/Simulation/genesis-world/endoscopic_tissue/outputs/joint-coarse-20261031`：初始失败、严格红/绿回归、第一组 ABBA、基线及第一候选录像。
- `D:/Simulation/genesis-world/endoscopic_tissue/outputs/joint-gated-20261031`：第二版严格回归、第二组 ABBA、候选录像及并排对照。

每段视频 1111 帧、30 fps，以模拟时间播放，不是运行帧率。三个原视频 1280×960，两段并排视频 2560×1008。完整原始 replay、视频及大日志仅保存在本机；Git 保存源码、补丁、脚本、结果摘要与来源哈希。

`run_section_profiles.py`在上述序列结束后另行逐项诊断基线和第二版，关闭 CUDA Graph，定位模块成本。它不能代替普通 Graph 计时，也不能凭不同轨迹的两次数据独立归因到某一个内核。`inspect_kernel_resources.py`读取实际编译的寄存器与栈信息，亦不是运行时占用率测量。

现有本机证据可用以下命令核验，不会运行物理模拟：

~~~text
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-joint-coarse\verify_artifacts.py
~~~

## 采用限制

基线和两版候选的原质量验收均为 failed；表皮额外切边及记录的裂口非流形边没有被隐藏。夹持数量、钳口开度、双刃载荷及残端运动存在差异，不能由“两个断裂都完成”“能量下降”或“画面没有爆炸”推断效果保持。肾脏脚本 release 阶段仅剪刀张开，夹爪未松开；本轮没有验证自由松夹回弹。

本轮初筛没有性能收益，未进入肾脏≥6、墙面≥10、组织瓣≥10、单刀≥1的采用级重复，也没有临床视频匹配验证。不替换默认版本，不缩减细层自由度/子步/迭代，不提高阻尼来掩盖摆动，不放宽交互验收阈值。
