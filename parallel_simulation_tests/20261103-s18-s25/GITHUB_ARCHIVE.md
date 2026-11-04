# GitHub 归档范围（2026-11-04）

这是在既有公开归档上的增量更新；正式 `endoscopic_tissue/` 默认求解器没有被实验候选替换。

最新结论：只读 GPU 检查版本平均 23.65 Hz，P95 50.37 ms，最慢 20 帧窗口 18.42 Hz；稳定 25 Hz 未达标，物理域及训练域审批仍为 false。非流形边失败实验保留，不以单次成功结果冒充稳定通过。

- [最新性能与正确性报告](profiling/native-core-25hz-20261104/REPORT.md)
- [最新源码与审计清单](profiling/native-core-25hz-20261104/delivery/manifest.json)
- [CPU 静态非流形边最小复现](profiling/native-core-25hz-20261104/tools/reproduce_nonmanifold.py)
- [S18–S25 总报告](REPORT.md)

## Omitted raw artifacts

公开：源码、构建/测试脚本、参数、报告、统计汇总、时间线图片、小型四三角形复现及必要插图。逐帧 `result.json` 较大时，发布同目录 `result-summary.json`，保留完成状态、失败原因、性能、对照计数、首末帧，并记录原始文件字节数和 SHA256。它不是原始逐帧文件。

不公开到 Git：Python/CUDA 运行环境、编译库和对象、CMake 缓存、CUDA/CUPTI/Nsight 原始捕获、数据库、完整逐帧采样、视频、大型快照与运行日志、进程清单和未被既有仓库追踪的额外资源包。原报告里指向这些文件的链接仅在服务器原目录有效。

脚本按实验版本保存，含服务器绝对路径、GPU UUID、CPU 编号及固定二进制 SHA。全新克隆须先适配路径、准备既有依赖/资产、构建并重新验证库和源码绑定；本次上传没有重新运行 GPU 实验，不宣称全新克隆可一键复现实测。不要直接启动历史自动队列。

原始服务器目录：`/srv/disk2/hryang/genesis-exec/parallel_simulation_tests/20261103-s18-s25`。源码/发布文件清单见 `../publication-20261104-manifest.json`；此前归档清单与报告保留。

## 历史实验脚本状态

未采用的 CPU zero-body 实验生成器 `profiling/native-core-25hz-20261104/tools/prepare_zero_body_check.py` 原文件存在多余括号的语法错误，按原实验 SHA 原样保留，不能作为运行入口。最终 V8 源码、构建/复现及回归入口单独做了语法核对。归档不以历史失败脚本的存在宣称这些实验可运行或已经通过。
