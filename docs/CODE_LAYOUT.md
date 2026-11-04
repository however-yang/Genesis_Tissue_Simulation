> 本文记录服务器本地目录整理。GitHub 此次追加最新候选与导航，已发布历史源码仍保留供对照；服务器 tar/bundle 未上传。最新候选见根 LATEST_VERSION.json。

# 代码目录整理与清理记录（2026-11-04）

## 现在应该看哪一份代码

**统一以 [current/README.md](../current/README.md) 和 [current/VERSION.json](../current/VERSION.json) 为准。当前保留的是 `native-readonly-v8-20261104`。**

|用途|统一位置|实际保留位置|
|---|---|---|
|Python 仿真、控制、渲染、训练接口|`current/python/endoscopic_tissue/`|`parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/python-gpu-check/endoscopic_tissue/`|
|Genesis 依赖|`current/python/genesis/`|同一冻结 Python 根目录下的 `genesis/`|
|原生 CUDA/C++ 源码|`current/native/`|该实验的 `native-readonly-v8/source/`|
|本机 A100 库|`current/lib/libLiverCudaSim.so`|该实验的 `native-readonly-v8/build/lib/libLiverCudaSim.so`|
|日常检查和复测|`sim.py`|委托该实验原有 `tools/reproduce.py`|

`current/` 的三个源码/库入口都是软链接，不是又复制一套项目。这样既能明确唯一入口，也不必改动已有模块的导入关系、资源查找、快照来源绑定或原始实验路径。原先的双器械、渲染、接触、切割和训练 API 文件内容均未修改。

当前只是保留的独立开发/复测版本，没有替换 `releases/0643e10991cf/`。完整 RGB-D 平均 23.65 Hz、P95 50.37 ms、最慢连续 20 帧 18.42 Hz 的原报告结论仍然有效；稳定 25 Hz、慢窗口 20 Hz 和完整物理域验收仍未通过。新增入口没有降低分辨率、减少子步或关闭检查。

## 清理了什么

|类别|数量|处理|
|---|---:|---|
|早期 profiling 中间/编辑副本|5 个源码目录|压缩归档后移除原重复目录；保留 baseline 与 V4 参考|
|上一轮渲染/调度中间候选|9 个源码目录|归档 `candidate-source` 和 R1–R8；保留真正采用的 `final-source`|
|本轮未保留候选|8 个目录|归档原生 V1–V7 与 CPU zero-check；保留 V8 与 Python GPU 检查版|
|只生成历史候选的一次性实验脚本|25 个文件|归档生成器、未采用版本构建器和历史批次编排；当前工具从 38 个缩为 13 个|
|已完成的一次性部署脚本|5 个文件|归档，避免误认为日常入口|
|GitHub 导出工作目录|2 个完整副本|确认无已跟踪改动或未跟踪/忽略文件，完整 Git bundle 验证成功后移除|
|旧增量发布 bundle|1 个|由包含全部祖先且独立可恢复的完整 Git bundle 替代|

淘汰候选及脚本共 **52 项、5,638 个普通文件**，归档含目录/链接共 **6,660 个条目**。归档逐成员校验，清理前再次检查原内容和文件集合未发生改变。

删除目标原占用约 **5.219 GiB**；新归档、完整 Git 历史和维护记录约 **0.606 GiB**，**净节省约 4.61 GiB**（按 `du` 分配块估算，四舍五入）。详见 [磁盘记录](../parallel_simulation_tests/maintenance-20261104/disk-summary.json)。

没有清理环境、资产、已部署 release、正常构建库、原始结果、失败快照、录像或 Nsight 记录。原有 baseline、已采用 V4、`aggressive/final-source`、原生 `native-all-queries-v2` 等必要参考继续保留，用于历史对照、补丁核查与机器码比较；它们不是当前开发入口。

## 目录结构

```text
genesis-exec/
  README.md                       # 从这里开始
  sim.py                          # 状态、校验、原命令复测
  Genesis_Tissue.code-workspace   # VS Code 只浏览 current 与 docs
  current/
    README.md
    VERSION.json
    python -> 唯一保留 Python 源码
    native -> 唯一保留 CUDA/C++ 源码
    lib    -> 已审计本机库
  docs/                           # 需求与整理说明
  archive/                        # 校验过的淘汰代码和发布历史
  parallel_simulation_tests/      # 结果、日志及必要历史参考
    maintenance-20261104/          # 本次清理的清单和核验
  releases/, builds/, envs/        # 原部署与运行依赖
  resources/, runs/, logs/        # 原资源与实验数据
```

根 `.ignore` 让默认文本搜索避开环境、归档和大规模运行数据；没有修改全局 IDE 设置。新的 VS Code workspace 是可选浏览入口。

## 验证结果

|检查|结果|
|---|---|
|保留版完整运行文件、资产与库|**1,421 个文件整理前后 SHA256 完全一致**，含原部署库及冻结交付清单|
|冻结交付源码清单|**53 个原生文件 + 333 个 Python 文件**及已审计 `.so` 匹配|
|现有 CPU 回归|**136 项全部通过**；显式隐藏 CUDA，没有 GPU 仿真|
|新旧入口参数对照|benchmark / validate / RGB-D / snapshot / profile **五类命令逐项一致**；清理前后均核对|
|拓扑静态最小复现|四例全部可从 `sim.py topology --case N` 运行；成功表示已知非流形错误被重现，不是物理验收通过|
|归档真实恢复|恢复一个淘汰原生候选，**119 个普通文件全部匹配**，链接正确，已有目标覆盖被拒绝；临时恢复副本随后移除|
|完整 Git 历史|在空 Git 仓库导入 bundle 并通过 `git fsck --full`，可恢复 `d201cfd` 及祖先，不依赖已删除副本|

原生 V8 共享库 SHA256：`38c0cd4ad5e9c5d1a3e43390246ad05797e1729a0b7e23f84c11cd76998ef5eb`。

本次没有运行 GPU 性能基准，没有产生新的 FPS 结论，也没有更改既有物理失败的验收状态。哈希和 CPU 回归支持“路径整理没有改写原运行实现”；它们不替代 GPU 物理、画质和人工交互验收。

## 恢复与后续维护

[archive/README.md](../archive/README.md) 提供按原路径恢复与完整 Git 克隆方法。历史计划/生成器引用了已归档的候选时，需要先恢复对应项；原 `delivery/manifest.json` 保留历史 harness 散列，没有为了目录清理改写这些证据。

后续实验优先保存补丁、配置和结果，避免再次复制整个 Genesis 项目。当前版本变更时同时更新 `current/VERSION.json` 和导航，失败候选不再占据“当前版本”入口。新构建应使用新的独立 build 目录，保留已审计库作对照。

这次是服务器本地目录整理。GitHub 上的 `d201cfd` 完整归档及其历史未改动；本次导航与清理记录尚未发布为新的 GitHub 提交。

机器可读清单、原 README 备份和完整日志位于 [maintenance-20261104](../parallel_simulation_tests/maintenance-20261104/)。
