> GitHub 归档说明：本页结论保持原样。部分逐帧 JSON 链接已改为带原文件 SHA256 的指标摘要；原始 trace、编译库、环境和大型快照仅保留在服务器。详见 [归档范围](../GITHUB_ARCHIVE.md)。

# S18–S25 定量性能实验

本目录保存冻结基线、分阶段候选、Nsight 原始记录、CPU/GPU 时间线、正确性回归及未插桩性能对照。**尚未通过稳定 20 Hz 和完整物理域验收。** 正式代码、入口、原生 DLL、物理参数及验收阈值均未替换。结果及失败原因见 [REPORT.md](REPORT.md)。

## 版本与证据

- `baseline-source/`：原 S18–S25 版本的冻结副本，散列见 `baseline-manifest.json`。
- `candidate-v1-source/`：节点/连通检查融合、GPU RGB-D 后处理。
- `candidate-v2-source/`：增加器械筛选融合、CPU 密集像素组合；曾实验 UV 布局缓存。
- `candidate-v3-source/`：增加 GPU 深度对齐的阶段版本。
- `candidate-v4-source/`：保留 GPU 对齐，删除真实切割窗口没有命中的 UV 缓存。完整训练 RGB-D 对照使用此冻结版本。
- `candidate-v5-source/`：**未纳入最终候选的共享内存实验归档**。单 worker 有平均吞吐收益，但双 worker 未证明端到端收益且 P95 变差。仿真和渲染源码与 V4 相同；最终推荐仍为 V4。
- `candidate-source/`：早期编辑副本，不是最终候选或正式入口。
- `candidate-v4.patch` / `changes-v4.json`、`candidate-v5.patch` / `changes-v5.json`：相对基线的可审阅改动及散列。补丁只做 `git apply --check`，没有应用到正式目录。
- `*-plan.json`：实验命令和冻结输入散列。`results/*/result.json`：逐帧结果和失败记录。
- `results/performance-summary.{json,csv}`、`results/physical-regression-summary.json`：完整汇总，含失败/部分运行。
- `results/*.nsys-rep`、`*.sqlite`：原始 Nsight 记录；`results/analysis-*/timeline.json` 和 `results/pool-analysis/*timeline.json` 可用 Perfetto/Chrome trace 打开。
- `figures/`：可导出的 PNG/SVG 时间线。`tools/`：采集、回归及离线分析脚本。

V2/V3 的缓存仅归档实验，后续使用 V4/V5 时不要再传 `cache_surface_layout`。

## 重跑方式

先只读检查指定 A100 是否空闲，使用新的输出目录；脚本有资源守卫。不得同时启动两个独立基准，双进程由一个池管理。不要修改 MPS、时钟、网络或其他任务。项目现有环境已经够用，无需安装依赖。

```bash
P=/srv/disk2/hryang/genesis-exec
T="$P/parallel_simulation_tests/20261103-s18-s25/profiling"
PY="$P/envs/genesis-py311-cu125-v1/bin/python"
"$PY" "$T/tools/profile_v4.py" \
  --source "$T/candidate-v4-source" --output "$T/results/new-training-rgbd" \
  --cpus 7,19,20,25 --vision --gpu-postprocess --gpu-aligned-depth
```

这是 1280×960、supersample=1、每物理步新 RGB-D、CPU 图像读回、深度对齐、有效性掩码和策略观测复制的完整 1410 步固定脚本。不含桌面呈现、视频编码、网络、策略推理或学习器更新。

去掉 `--vision --gpu-aligned-depth` 是与历史 11.60 Hz 同口径的直接 RGB-D 路径。`--no-render` 仅用于明确标记的物理与全部检查测量。`--labels` 增加完整分辨率密集材料标签，目前持续切割有表面绑定数量错位和可见性检查失败，不能作为通过配置。绝不能用关标签后的帧率代表完整标签性能。

`--profile --start 470 --count 8 --max-frames 478 --render-warmup 4` 用于短窗口剖析。`--render-warmup` 只在剖析模式跳过窗口前图像，所有物理步仍执行；该结果不是整段 FPS。Nsight 原始命令保存在对应计划文件，节点级 kernel 归因必须保留 `--cuda-graph-trace=node`，否则会漏掉现有 CUDA Graph 内的设备计算。

实际池端到端测试：

```bash
PROFILE_POOL_SOURCE="$T/candidate-v4-source" "$PY" "$T/tools/pool_benchmark_final.py" \
  --output "$T/results/new-pool-shared" --workers 2 \
  --vision --optimized --gpu-aligned-depth
```

这包含 cut-ready 准备、8 步保持预热和每 worker 60 步切割目标变化。每 worker 一个 CPU 核，保留原检查，完整传递约 9.83 MB/帧的 RGB-D 观测。它是短窗口吞吐实验，不是完整 episode 或 20 Hz 验收。V5 仅供复查：改用 `candidate-v5-source`、`pool_benchmark_shared.py` 并加 `--shared-images`。该实验不改变公开观测数组的 dtype、内容或独立所有权，但还未证明双 worker 的端到端收益，因此未保留到 V4。

候选训练视觉配置：

```python
vision = dict(width=1280, height=960, every_steps=1, opencv_threads=4,
              gpu_postprocess=True, gpu_aligned_depth=True)

```

GPU 后处理只验证 supersample=1 的离屏内镜路径，不自动降低默认 UI 的 supersample=2，也不替换默认 UI。Numba 首次编译成本单独计入初始化；未通过的物理场景、A/A 轨迹非确定性及 S25 连续标签失败不能由 CPU 单元测试替代。
