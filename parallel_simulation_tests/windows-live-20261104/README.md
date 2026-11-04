# Windows 本地实时演示

对应分支 `a100-parallel-validation-20261103`，提交 `f1d454ca472b5647ce58186e7cc767b9e5a3fe8f`。本目录是独立 Windows 构建与启动入口，没有替换原 `dev` 工作区、默认 DLL 或服务器版本。

在 PowerShell 执行：

```powershell
Set-Location 'D:\Simulation\genesis-world\.worktrees\a100-parallel-validation-20261103'
.\parallel_simulation_tests\windows-live-20261104\run_live.cmd --scene kidney
```

这是肾脏肿瘤夹取、牵拉、剪切和提起的实时窗口演示。流程由原有脚本控制器械；不是鼠标遥操作界面。空格暂停，任意键继续，Q 或 Esc 退出。演示结束后按任意键关闭。

薄壁夹取与连续剪切：

```powershell
.\parallel_simulation_tests\windows-live-20261104\run_live.cmd --scene wall-traction
```

额外打开 Genesis 自由观察窗口可追加 `--viewer`。`--help` 显示原有演示参数；本启动器固定新 DLL，并自动将结果写到本目录 `outputs/时间戳/`，因此不接受自定义 `--dll` 或 `--output`。`--check` 仅校验源码、加载库、设置主机端开关和导入 Python，不初始化物理会话。

## 构建与配置

- CUDA/C++：`native-core-runtime-20261104/native/source` 发布的 56 个文件。
- Python：发布清单绑定的 `native-core-25hz-20261104/python-gpu-check`，核对 333 个文件。
- Git 换行转换在校验时显式处理；不改发布清单来绕过 SHA 检查。
- 本机现有 CUDA 12.5.40、MSVC 14.29、Python `genesis_world` 环境；未安装或升级依赖。
- 原生构建副本只把 CMake 编译架构 `sm_80` 改为本机 RTX 3090 的 `sm_86`，没有修改求解器、材料参数、网格、子步或接触算法。
- 启动器在会话初始化前设置发布候选的 core mode `43264`、query mode `3`、64 query threads、512/1024 launch blocks 和 coarse bound cache。
- 使用现有实时演示循环及其全部检查；渲染显式启用发布 V8 的 GPU 后处理、深度对齐、fused mesh 和 persistent soup。没有关闭原演示的质量检查。
- `source-manifest.json`、`build-manifest.json` 和 `build.log` 记录本机构建来源；每次演示另存 `windows-launch.json`。

已准备好源码时可用 `build.cmd` 在此目录重新编译。它只写本目录 `build/` 和本机构建记录，不覆盖默认库。`prepare_build.py` 拒绝覆盖已有构建源码副本。

## 验证边界

Windows 本机 DLL 与服务器 Linux `.so` 是不同构建；A100 报告的 24.84 Hz 不能作为 RTX 3090 的实测帧率。这个窗口运行器也不等同于服务器每步同步完整 RGB-D 的训练基准。

本次编译和导入检查不构成完整夹取/切割无损验收，发布候选原有未通过项仍保留。只有实际运行后才能报告本机帧率和完整交互结果。GPU 被其他任务占用时会影响观察到的速度，不应据此判定性能回退。
