> GitHub 归档说明：本页结论保持原样。部分逐帧 JSON 链接已改为带原文件 SHA256 的指标摘要；原始 trace、编译库、环境和大型快照仅保留在服务器。详见 [归档范围](../../GITHUB_ARCHIVE.md)。

# 原生核心性能实验，2026-11-04

本目录独立于正式版本，也不修改上一轮已保留代码。目标是完整 1280×960 RGB-D、控制、质量检查和必要 CPU 观测传输接近稳定 25 Hz；25 Hz 对应 40 ms/帧。原 dt=1/30、3 子步、8 次接触细化及材料/摩擦/切割参数保持原样。

本轮曾尝试查询生命周期、胶囊查询裁剪、寄存器约束、器械接触求值并行化、粗层归约和材料精确分支。部分局部数值对照通过的版本在完整切割复测出现非流形边，均不能作为稳定交互版本。代码和失败快照保留供复查。

当前隔离出的版本为 `native-readonly-v8` + `python-gpu-check`：物理求解与参考版一致，仅把当前三角形表面的非切割器械体诊断迁到 GPU。所有原采样点保留；临近原验收阈值时使用 CPU 原法。完整验收结果与性能数字见完成后的 `REPORT.md`，不得把早期单次成功或诊断帧率当成最终通过。

- `native-readonly-v8/source/`：精简候选源码；`build/lib/libLiverCudaSim.so`：本机 A100 的测试库。
- `python-gpu-check/`：独立 Python 适配，默认 CPU 诊断，测试时显式启用 GPU。
- `tools/`：编译、完整回归、同库对照、Nsight 分析与复现入口。
- `results/`、`logs/`：原始结果，含中止/失败结果，未过滤慢帧。
- `EXPERIMENTS.md`：保留、排除及失败原因。
- `delivery/manifest.json`：最终审计绑定，完成报告后生成；正式版本未替换。

最终审计完成后，可用现有环境复现，不安装依赖：

```bash
cd /srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104
/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python tools/reproduce.py --kind benchmark --variant readonly
/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python tools/reproduce.py --kind validate --variant readonly
/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python tools/reproduce.py --kind snapshot --variant readonly
```

使用 `--variant reference` 作同库原法对照，`--dry-run` 查看具体参数。复现仅申请 GPU 1 UUID，沿用已有 GPU 预约/资源上限检查，不修改网络、驱动、MPS、MIG 或时钟配置。物理域与训练域审批仍为 false；可选密集 S25 像素支撑节点标签不在 RGB-D 测速口径内。
