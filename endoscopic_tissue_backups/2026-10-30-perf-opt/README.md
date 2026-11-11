# 2026-10-30 性能与交互对照

结论与测量范围见 [实验报告](../../endoscopic_tissue/docs/performance-experiments-2026-10-30.md)。本目录的脚本用于本轮诊断，不替代生产质量检查。

## 保护的基线

- 提交：`c64c1a02ac0b6f004a36e370f14a6bfae7e76091`。
- 默认剪刀 DLL SHA256：`0d09f56a1f4acb813424cd1dee3233d89ce7ef5c0d45e64f079c7d7387912225`。
- `candidate-compaction.patch` 是已否决的碰撞候选实验，未应用于当前源文件；它不是待合入的修复。
- 冻结的源文件、实验 DLL、视频及逐步状态保留在本机；大型输出不提交 Git。

## 复现显示对照

从仓库根目录运行，使用已安装项目依赖和 GPU 驱动的 Python。GPU 作业须串行，输出目录必须不存在，避免覆盖已有证据。原生状态录制每条约 3.6 GB。

如需在新检出中恢复冻结的渲染文件：

```powershell
python -c "from pathlib import Path; import subprocess; Path('endoscopic_tissue_backups/2026-10-30-perf-opt/baseline_rendering.py').write_bytes(subprocess.check_output(['git','show','c64c1a02ac0b6f004a36e370f14a6bfae7e76091:endoscopic_tissue/rendering.py']))"
Copy-Item -LiteralPath endoscopic_tissue/rendering.py -Destination endoscopic_tissue_backups/2026-10-30-perf-opt/persistent_rendering_v2.py
```

先录制完整状态，再以冻结基线和当前渲染器读取同一份状态：

```powershell
python endoscopic_tissue_backups/2026-10-30-perf-opt/capture_replay.py endoscopic_tissue/outputs/new-capture --scene kidney --tool scissors
python endoscopic_tissue_backups/2026-10-30-perf-opt/replay_single_blas.py endoscopic_tissue/outputs/new-capture/replay.bin endoscopic_tissue/outputs/new-baseline-video endoscopic_tissue_backups/2026-10-30-perf-opt/baseline_rendering.py
python endoscopic_tissue_backups/2026-10-30-perf-opt/replay_single_blas.py endoscopic_tissue/outputs/new-capture/replay.bin endoscopic_tissue/outputs/new-candidate-video endoscopic_tissue/rendering.py
python endoscopic_tissue_backups/2026-10-30-perf-opt/compare_render.py endoscopic_tissue/outputs/new-baseline-video endoscopic_tissue/outputs/new-candidate-video
python endoscopic_tissue_backups/2026-10-30-perf-opt/make_review.py endoscopic_tissue/outputs/new-baseline-video/interactions.mp4 endoscopic_tissue/outputs/new-candidate-video/interactions.mp4 endoscopic_tissue/outputs/new-review "Persistent buffers"
```

`render_replay.py` 将更新与绘制分别计时，解压和视频编码不进入这些计时。完整视频按模拟时间 30 fps 播放，不能作为实际帧率。`compare_render.py` 输出全部原始图像哈希差异和采样 PNG 差异，它本身只报告，不代替验收判断。

`render_cases.py` 覆盖九项显示边界情况；`cases_single_blas.py` 在最终 BLAS 线程设置下运行它。最后追加 `knife` 参数使用单刀和无顶点色的脂肪材质路径，这属于显示检查，不是单刀物理验收。

## 集成运行和证据检查

```powershell
python endoscopic_tissue_backups/2026-10-30-perf-opt/integrated_display.py endoscopic_tissue/outputs/new-integrated
```

该入口保留实时物理、记录与显示线程，以及按墙钟时间记录的视频；只替换窗口贴图和键盘等待。报告中显示阶段的帧率不能等同于真实屏幕帧率。它还记录 BLAS 线程数量，检查运行前后设置是否恢复。

录制和集成脚本使用既有 `2026-10-28-kidney/dev_run.py`，保留非流形边的诊断记录以完成整条轨迹；基线的误切会使最终质量验收失败，正常写完 1110 步报告后仍可能返回 1。访问异常、缺失报告或未完成轨迹不是这个预期质量失败。生产代码没有放宽这些检查。

`verify_display.py --blas`、`blas-fixed-jobs.json` 和 `verify_artifacts.py` 使用本轮固定目录名，服务于本轮证据审核。`verify_artifacts.py` 检查图像、DLL、受保护源码与已完成运行的关键物理指标；输出成功只表示这些比较成立，不表示原有误切问题或手术真实性已通过验收。详见报告和原始 `report.json`。
