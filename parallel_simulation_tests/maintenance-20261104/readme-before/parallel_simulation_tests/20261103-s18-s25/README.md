# S18–S25 独立训练候选

本目录是隔离候选与证据目录。生产发布目录 `releases/0643e10991cf`、默认入口和默认库均未替换。当前候选**不能作为已通过交互质量及 20 Hz 验收的正式版本**；以 `REPORT.md` 和原始结果为准。

- `source/`：基于 `ebbfa5a` 的独立工作副本，分支 `s18-s25-training-candidate-20261103`。
- `native-s25/source/`、`native-s25/build/`：R3 基础上的独立原生标签 ABI 构建；物理求解核不变。
- `logs/`、`results/`：测试输出、原始失败和性能数据。
- `REQUIREMENTS.md`：用户需求单的冻结副本。
- `benchmark_interaction.py`、`evaluate_performance.py`：逐物理步完整渲染基准及禁止不合格候选上线的门槛。
- `probe_training_features.py`：真实 CUDA 快照恢复、像素标签重投影和状态不变性验证。
- `probe_pool.py`：两 worker 配置、快照和隔离最小验证。
- `probe_coupling_geometry.py`：S18 几何准备与正常轨迹筛查。

## 使用隔离 Python 包

```bash
P=/srv/disk2/hryang/genesis-exec
T="$P/parallel_simulation_tests/20261103-s18-s25"
PY="$P/envs/genesis-py311-cu125-v1/bin/python"
cd "$T/source"
export PYTHONPATH="$T/source"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$T/tmp"
```

以下开关均显式启用，默认合同和物理参数不变：

```python
EnvironmentConfig(
    dll_path=explicit_library, dll_sha256=explicit_sha,
    bounds=bounds, scene=scene,
    outcome_semantics="physical_outcome_v1",  # S19，要求训练遥测 ABI
    instrument_collision_checks=True,        # S22，保守碰撞包络；wall 尚未验收
    fast_topology_checks=True,               # 等价 CPU 算法候选，不改变检查频率和阈值
)
```

`tool_kinematics_v1` 只返回实际开度、朝向、限速后命令位置、时间及已声明视觉数据。实际求解位置和力不进入这个合同。`visual_tracking=True` 是显式假设，写入 manifest；默认关闭。`recorded_observation(reader, index, contract='tool_kinematics_v1')` 从录制元数据重建，旧数据缺少命令记录时拒绝臆造。

## S20 池协议

`endoscopic_tissue.training.parallel.ParallelNativePool` 使用显式 GPU UUID、CPU 列表、输出目录；GPU 忙则拒绝启动。最多 16 个 worker，未放宽之前 32/64 的稳定性限制。不自动启动 MPS。

配置是 `dataclasses.asdict(EnvironmentConfig(...))`，可加：

```python
config.update(
    observation_contract="tool_kinematics_v1",
    privileged_fields=["positions_m", "topology", "telemetry"],
    max_snapshot_bytes=2*1024**3,
    max_snapshot_capture_bytes=3*1024**3,
    max_response_bytes=64*1024**2,
)
```

内存数值是显式预算示例，不是容量承诺。每个 worker 的压缩快照和热缓存均计费；单次原生捕获另设预算和主机内存储备。快照操作池内串行，防止瞬时内存叠加。

```python
pool.send(0, "reset", seed=7, scene_config={
    "scene": "wall-traction", "preparation": "settled",
    "contact_model": 1, "physics": {"young_pa": 3000},
})
reply = pool.recv_ready()[0]
pool.send(0, "snapshot", hot=True)
key = pool.recv_ready()[0]["snapshot_id"]
pool.send(0, "restore", snapshot_id=key, branch_label="candidate-0")
branch = pool.recv_ready()[0]
pool.send(0, "drop_snapshot", snapshot_id=key)
pool.recv_ready()
```

每个 worker 只能有一个未消费请求。快照 ID 不能跨 worker 使用，reset 清空该 worker 的缓存。策略观测位于 `observation`，训练真值位于 `privileged`；每个分支单独保存 EpisodeWriter 数据和谱系。

## S23 服务器队列

不安装服务、不调整网络；通过已有 SSH 账号进入服务器后使用：

```bash
"$PY" -B -m endoscopic_tissue.training.server_queue submit \
  --queue "$T/queue" --spec "$T/examples/queue-smoke.json"
"$PY" -B -m endoscopic_tissue.training.server_queue status --queue "$T/queue"
"$PY" -B -m endoscopic_tissue.training.server_queue run-next \
  --queue "$T/queue" --gpu GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa
```

submit/status 不使用 GPU；run-next 仅在指定卡空闲时启动一个作业，监视其他任务和资源上限。源码、素材、规格和库哈希在执行前复核；修改候选后重新提交，不覆盖旧证据。`execution_complete` 只表示作业完整执行，`candidate_acceptance=not_certified` 不会自动变为正式验收通过。Windows 可通过现有 SSH 调用同一入口；本轮没有 Windows DLL 或本地/服务器分布等价结论。

## S25 视觉与材料标签

```python
vision = {
    "width": 320, "height": 240, "every_steps": 1,
    "privileged_labels": ["segmentation", "surface_supports", "pixel_supports"],
    # randomization 的字段范围参见 visual_randomization.py；省略则无扰动
}
```

必须显式指定独立标签 ABI v2 库。每个有效组织像素输出最多 24 个支撑节点 ID、权重和三维仿射偏移；同一个节点可在不同三角形顶点支撑中重复出现，重建时按槽累加。背景、器械或无有效材料支撑的像素输出 `valid=False`、ID=-1、权重=0。

重建公式（统一以米为单位）：

`x_pixel = Σ_slot weight[slot] * node_position[node_id[slot]] + offset_m`

完整皮面支撑遵循原生固定材料点插值；切面采用求解器的局部仿射绑定，不能称为精确全局材料形函数。可见三角形与透视插值来自独立浮点 OpenGL 通道，包含器械和背景的深度遮挡。标签与 RGB-D 同一状态；`visual_label_quality` 记录覆盖率及可见性不一致数。传感器噪声不扰动真值标签。

像素标签是额外的训练采集成本，未承诺开启后达到 20 Hz。320×240 的稠密 24 槽标签也可能超过默认 32 MiB IPC 预算，需要显式预算；优先留在 worker 数据文件内，按训练需求选取特权回传字段。默认交互路径不计算标签。
