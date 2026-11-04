# 当前保留代码：V8，只读 GPU 诊断版

**以后从这个目录查看当前源码。** 本目录只有版本说明与软链接；源码没有再复制。

|入口|内容|
|---|---|
|[python/endoscopic_tissue/](python/endoscopic_tissue/)|软组织仿真、双器械控制、渲染与训练接口|
|[python/genesis/](python/genesis/)|随冻结测试保留的 Genesis 依赖|
|[native/vendor/cuda/](native/vendor/cuda/)|原生形变、接触、抓取、切割和拓扑实现|
|[native/vendor/src/plugin_api.cpp](native/vendor/src/plugin_api.cpp)|Python/CUDA 的本机接口|
|`lib/libLiverCudaSim.so`|SHA256 已绑定的 Linux/A100 测试共享库|
|[VERSION.json](VERSION.json)|当前唯一版本标识、状态和实验路径|

这些入口指向 `parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/` 下的 `python-gpu-check` 与 `native-readonly-v8`；从软链接打开并修改文件，就是修改同一份文件。`python3 ../sim.py verify` 会检测它是否仍与冻结审计清单一致。

在项目根目录使用 `python3 sim.py` 查看状态，`python3 sim.py --help` 查看复测命令。当前入口直接委托原始 `tools/reproduce.py`，不重新编写仿真循环。

当前版本只新增保留的只读 GPU 检查，原生求解、抓取与切割使用参考实现。23.65 Hz 是已记录的完整 RGB-D 平均值；稳定 25 Hz、慢切割窗口 20 Hz 和完整物理域验收仍未通过。失败候选不再与本目录混放。

原完整 [README](../parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/README.md)、[性能报告](../parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/REPORT.md)、[交付审计](../parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/delivery/manifest.json) 和 [历史恢复说明](../archive/README.md) 保留供核对。
