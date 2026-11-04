# 65,688 材料节点：正面肾脏与左右器械实验场景

本目录是独立场景实验，基于分支 `a100-parallel-validation-20261103` 的 `f1d454ca472b5647ce58186e7cc767b9e5a3fe8f`。可用于查看新的构图和减节点效果，**尚未通过交互无损验收，不替换原默认场景**。原生求解器源码、材料参数和原验收阈值未修改。

## Windows 实时窗口

在 PowerShell 中运行：

```powershell
& 'D:\Simulation\genesis-world\.worktrees\a100-parallel-validation-20261103\parallel_simulation_tests\kidney65-scene-20261104\run_scene.cmd'
```

沿用 `windows-live-20261104` 中单独编译的最新 Windows DLL，以及本机已有 `genesis_world` Python 环境。本轮仅改 Python 场景配置，无需重新编译 DLL。启动后执行完整的自动夹取、两次剪切和提离流程；不是新增手动双臂控制功能。空格暂停，任意键恢复，`Q` 或 `Esc` 停止；终端各阶段的 `display ... fps` 是窗口显示帧率。

原布局和仅减节点对照：

```powershell
& 'D:\Simulation\genesis-world\.worktrees\a100-parallel-validation-20261103\parallel_simulation_tests\kidney65-scene-20261104\run_scene.cmd' --preset baseline
& 'D:\Simulation\genesis-world\.worktrees\a100-parallel-validation-20261103\parallel_simulation_tests\kidney65-scene-20261104\run_scene.cmd' --preset mesh65
```

默认 `--preset frontal65` 为本轮新布局。`--inspect` 仅检查场景配置与节点数，不启动物理仿真。每次运行在本目录 `outputs/` 下生成唯一时间戳目录，已有输出不覆盖。

已测完整流程会在末尾保留 `Interaction quality failed` 及失败报告，这是误切/残连验收未通过，不能当成已验收版本。不要删除该检查来隐藏失败。

Windows 当前另有训练占用 RTX 3090，本轮没有在它运行时启动 GPU 对照。测本机帧率时需要在 GPU 空闲后运行。A100 的物理步速度不是 RTX 3090 的窗口显示帧率。

## 本轮修改

- 材料节点从 102,710 降为 **65,688**；活动节点 79,548，总格点槽位 236,313。后两个数字不能当成材料节点数。网格间距为 **1.25 × 1.00 × 1.25 mm**，保留切割法向的 1 mm 分辨率。
- 保持解剖形状和尺寸、杨氏模量 15 kPa、泊松比 0.25、步长 1/30 s 与 3 个子步。网格改变仍会改变离散质量与网格相关容差，不能仅据参数相同宣称形变等价。
- 镜头在肾脏中线正面、略俯视，并后移为上提组织留出空间；43° 视场。
- 夹爪从画面左侧水平进入，剪刀从右侧约 45° 进入；接触与显示使用同一真实工具位姿。夹爪尖端的预设目标轨迹保持，切割平面及阶段时长保持。
- 调低脂肪床遮挡、减弱过亮反光、加入较暗红褐色组织背景。背景是视觉近似，不是经过解剖标定的腹腔；没有增加碰撞物或烟雾遮挡。

## 检查与证据

见 [REPORT.md](REPORT.md)、[CPU 结构检查](review/final-cpu-check.json) 和各次 `outputs/<case>/`：

- `report.json`：原始交互验收结果，失败不会改成成功。
- `scene-config.json`：节点数、库和场景源码哈希。
- `timing.json`：服务器无窗口对照的逐帧物理、读回、渲染录像耗时。实时窗口使用另一条异步显示路径，不产生这份同步计时；请看终端 `display ... fps`。
- `endoscopic_demo.mp4`：完整交互录像。服务器无窗口录像按仿真时间 30 fps 编码，**其播放流畅度不代表实时计算帧率**；Windows 窗口录像则按原有实时显示逻辑采集。
- `review/<case>/`：录像解码检查与按仿真时间抽样的图板。

本轮保留已有误切失败，同时记录粗网格接触反力尖峰问题。最终镜头复测还报告材料分成 2 块、全格点连通分量仍为 1，属于未解决的残连告警。场景仍采用原演示的 `contact_model=2` 辅助夹持，不将其表述为已经验证的纯摩擦夹持。

## 服务器复现

已有授权的测试使用第二张 A100 UUID `GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa`，顺序执行。不要同时启动以下命令；`--guard` 会在 GPU 被其他计算占用时拒绝启动。

```powershell
ssh -o BatchMode=yes labserver '/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python -X utf8 -B /srv/disk2/hryang/genesis-exec/parallel_simulation_tests/kidney65-scene-20261104/run_scene.py --preset frontal65 --guard --headless'
```

环境、CUDA 库及模型均使用服务器已有版本，未安装依赖或修改其他科研任务。
