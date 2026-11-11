# 本轮仿真侧实施与验证证据

本目录保存训练侧新增功能和隔离 R20 候选的复现材料。R20 未通过验收，不替换默认 DLL；默认入口、物理材料、摩擦、子步和迭代参数不变。

## 主要入口

- `run_cpu_checks.py`：使用工作区临时目录运行 CPU 契约/任务测试。
- `run_final_checks.py --tag NEW_TAG`：冻结训练源码后串行运行4个策略配置、同场景冻结重复评估、纯摩擦、RGBD和敏感性用例。新标签目录不可复用。
- `run_final_checks.py --tag NEW_TAG --batch`：按 `hundred-screen.json` 运行五组各20回合的局部筛查。每回合独立原生进程，全部结果包括失败均保留。
- `positive-hold-control.json`：纯摩擦保持任务的成功正对照。不能用时间上限或剪刀闭合代替任务成功。
- `audit_training_batch.py`：独立核对完整性、终止、自举与禁切对照，并报告旧 seed 分组中的重复动作。
- `test_supervision.py`：超时终止自己创建的子进程、磁盘预算阻止启动的实际故障注入。
- `verify_vision.py`：独立核对同步 RGBD 和深度映射。

## 原生候选

`package/release-r20-support-cache.json` 固定构建基线与所有覆盖源码哈希，`package/build_experiment.py` 构建独立 DLL。`audit_r20.py` 对完整1320步轨迹逐步比较缓存与原全量重算；`audit_surface_edges.py` 只读定位保存网格的异常边，不修改诊断阈值或物理。

1320次支撑比较相同，不等于完整物理候选通过。R20存在切后非流形边，独立ABBA计时未证明不退化；释放后主体未通过水密流形检查，顶点包含法的很小穿入值不能用作通过证据。

## 证据与存储

`final-v1/adapter-source`、`final-screen/adapter-source`、`final-positive/adapter-source` 是各批次启动时的精确源码。`source-hashes.json` 供复核，不能用后续源码覆盖。`final-input-validation/adapter-source` 保存最后收紧材料ID校验后的源码，66项CPU检查通过；对109个带目标的原生回合、6729条转移离线重算完全相同，见 `strict-id-offline-equivalence.json`。`machine.json` 是本轮修改前的保护文件哈希。

`evidence-summary.json` 汇总真实运行；`interaction-review` 保存关键阶段对照与全部77个释放帧的目视记录。实际数组、每回合状态和完整MP4位于 `endoscopic_tissue/outputs/training-completion-20261101`，不把多GB生成物提交到Git。

历史 r10/r11 生成回放为腾出记录空间转成无损 `replay.rdx`；只有完整重建的长度和SHA-256相等后才释放单个原 `replay.bin`。原视频不变，见两份 `storage-archive*.json`。这些归档脚本操作明确列出的历史产物，不属于常规采集流程，不需再次执行。

同场景不同 seed 的重复评价仅说明运行重复性，不代表独立几何泛化；数据集划分应使用按组织几何分组的索引。物理训练域仍未自动认证，所有探索数据读取需要显式承认范围。

百回合局部筛查已完整结束，6154条转移；60回合时限结束、40回合目标外损伤。该批为辅助接触模式2，纯摩擦保持正对照单独记录并在15步后达到持有成功条件。所有最终记录合计115回合，其中111个采集、4个同几何重复评估；另保留一次细网格准备失败。

详细结果见 `endoscopic_tissue/docs/training-completion-results-2026-11-01.md`，使用见 `training-policy-guide-2026-11-01.md`。真实机械臂按用户决定不在本轮范围。
