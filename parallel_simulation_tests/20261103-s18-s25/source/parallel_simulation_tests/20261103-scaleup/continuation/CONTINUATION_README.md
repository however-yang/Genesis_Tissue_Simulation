# 2026-11-03 隔离续做入口

默认 release 和默认原生库均不改动。先读 `CONTINUATION_REPORT.md`、`../SCALEUP_REPORT.md` 与用户清单 `../../../docs/remaining-work-2026-11-03.md`。所有实验候选均绑定独立库哈希；R9 已在第二对换抓分叉触发原 0.2 mm 门限，**不采用**。

## 已运行的队列

- `recovery-plan.json` → `recovery-run`：最后视频恢复、1320 步肾脏完整帧 CUDA 追踪。
- `repair-plan.json` → `repair-run`：R3 两份失败状态的原库精确恢复与额外接触修复诊断。
- `r9-plan.json` → `r9-run`：72 个 CUDA 代数测试通过；完整换抓矩阵完成 2 对，第 2 对拒绝并停止。不能称 20 对通过。
- `diagnostic-followup-plan.json` → `diagnostic-followup-run`：6 次原库精确恢复计时；16 组接触树冻结输入逐字节等价检查。树优化尚未整合进求解器。
- `../topology_mps.py --name t32 --workers 32 ...`：原 wall/三剪的 32 环境诊断重复；失败时保存完整网格。它增加失败 I/O，不能当受控性能对照。
- `training-adapter-plan.json` → `training-adapter-run`：显式热快照缓存 ABBA；2 环境微小牵拉采样、CPU 模仿更新、数值检查点读回、新回合回评。完成状态以 `state.json/result.json` 为准。不是完整切割策略或训练域认证。

`adapter_after_topology.py` 仅等待本轮 t32 私有 MPS 清理并确认 GPU 空闲，然后启动最后一个队列。它不是后台定时服务，也不修改服务器配置。

## 复用监督器

固定解释器：`/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python`。

```bash
python -B acceptance_queue.py --plan PLAN.json --directory NEW-RUN-DIR --gpu GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa
python -B acceptance_queue.py --plan PLAN.json --directory EXISTING-RUN-DIR --gpu GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa --resume
```

只有同一份计划和相同输入哈希可恢复。完成的产物先校验哈希；中断任务创建新 attempt，旧证据不覆盖；已有物理拒绝不会因 `--resume` 变为通过。GPU 保护只终止本监督器确认拥有的进程。不要同时运行相互独立的 GPU 队列。

## 代码范围

- `acceptance_queue.py`：子任务进度、恢复、结果验收、资源保护。`test_acceptance_queue.py` 的 5 项回归通过。
- `prepared_snapshot.py`：可选、显式内存限额的不可变热缓存，压缩快照文件和 SHA 不变；预热时间/约 1.26 GB 原生状态内存必须单独记账。4 项 CPU 合同测试通过，GPU 对照另有状态。
- `build_vertex_candidate.py`：从 R3 复制隔离源码，R9 仅实验重建修复的受限顶点方向；不是默认补丁。
- `topology_capture_worker.py`：在原质量判定之后采集首失败状态，不改变判定。
- `ccd-endpoint-witness/result.json`：线性运动三角形的代数反例；两个中间穿越事件被端点筛选跳过。没有宣称原肾脏轨迹实际出现该反例，也没有把候选检测器当作完整 CCD 响应。

## 仍需工作的边界

换抓穿入、纯摩擦进刀后的载荷稳定性、切口/辅助壳层拓扑、释放与连续碰撞、S16 两套各 900 次矩阵、同一最终候选整体验收、冷恢复 ≤1 秒、超过十万材料节点 33 Hz 尚未关闭。GPU0 上既有任务不动；S14 真实双机械臂仍按原决定排除。不能用并行放大被质量规则拒绝的数据，也不能把 R9 单对成功或代数正确记作物理修复完成。
