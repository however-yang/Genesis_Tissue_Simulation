# Genesis 软组织双器械仿真

**最新代码：2026-11-04 原生核心优化候选。完整 1280×960 RGB-D 平均 24.84 Hz，P95 46.24 ms；稳定 25 Hz、所有慢窗口 20 Hz 和全域交互无损仍未通过。**

|用途|入口|
|---|---|
|最新候选及复测说明|[parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104/README.md](parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104/README.md)|
|最新 CUDA/C++ 源码|[候选 native/source/](parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104/native/source/)|
|本轮完整性能/正确性/失败报告|[REPORT.md](parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104/REPORT.md)|
|普通 ABBA 原始结果与汇总|[final-comparison.json](parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104/results/final-comparison.json)|
|源码、工具及本机库散列|[delivery/manifest.json](parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104/delivery/manifest.json)|
|保留交互参考 V8（未切换到候选）|[current/README.md](current/README.md)|
|机器可读的最新候选标识|[LATEST_VERSION.json](LATEST_VERSION.json)|
|需求与剩余工作|[docs/](docs/)|

本轮同条件 V8/候选交错复测：24.01→24.84 Hz，原生步 28.27→26.84 ms，P95 48.90→46.24 ms；实际切割阶段约 22 Hz。现有局部牵拉/实切、接触系数、RGB-D、快照和 136 项 CPU 回归通过。自由组织的严格跨轨迹复现仍失败，V8 自重跑也失败；新几何、纯摩擦、wall 碰撞和真实撕裂分离等既有失败继续保留。

## 使用哪一份代码

本轮只维护上表所指的一份候选原生源码，Python 借用已保留的 [V8 Python 源码](parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check/endoscopic_tissue/)。`current/` 仍标识未替换的 V8 参考实现。其它历史实验目录保留供旧报告与 Git 历史对照，不代表正在采用的代码。

这是源码和证据发布。Python 环境、编译 `.so/.dll`、构建缓存、Nsight 原始 trace 和大规模数组不放入 Git；保留源码、编译工具、SHA、最终原始 JSON、时间线导出和必要失败轨迹。新 clone 需要已有依赖和按说明构建的本机库才能执行 GPU 回归，缺少本机库时散列验证不会冒充通过。排除范围及服务器复测说明见 [PUBLICATION.md](parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104/PUBLICATION.md)。

`main` 保持原状，本版本位于 `a100-parallel-validation-20261103` 分支。服务器的正式部署与交互入口没有因本次上传而替换。

## Genesis 上游

项目基于 Genesis，保留原有许可证和上游代码。此前根目录的完整上游介绍保存在 [Genesis World README](docs/GENESIS_WORLD_UPSTREAM_README.md)。
