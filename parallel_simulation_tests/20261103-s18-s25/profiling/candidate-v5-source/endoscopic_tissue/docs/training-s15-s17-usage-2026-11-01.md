# S15–S17 隔离实现与验收入口

2026-11-01。实现仍处于实验验收阶段。CPU 检查和原生编译已通过，真实 CUDA、完整交互视频、默认关闭时的耗时对照尚未完成；本文不代表训练域或新 DLL 已获准采用。默认入口和默认 DLL 保持原样。

## 配置与快照

入口位于 `endoscopic_tissue/training`。`SceneConfig.physics` 接收冻结的 `PhysicsOverrides`；`SceneConfig.geometry` 接收冻结的 `WallGeometry`。未指定时走现有场景与原生初始化路径。

```python
from pathlib import Path
from endoscopic_tissue.training.env import EnvironmentConfig, NativeEnvironment, SceneConfig
from endoscopic_tissue.training.collect import local_bounds
from endoscopic_tissue.training.provenance import sha256
from endoscopic_tissue.training.snapshot import EnvSnapshot

# 显式选择隔离 DLL。不能指向旧 DLL 然后假定具备状态 ABI。
dll = Path("endoscopic_tissue/native/bin-s15-training-snapshot1/LiverCudaSim.dll")
scene = SceneConfig("wall-traction", "cut_ready")
config = EnvironmentConfig(str(dll), sha256(dll), local_bounds(scene), scene,
                           cutting_enabled=True, max_steps=120)
with NativeEnvironment(config) as env:
    initial, context = env.reset(seed=0)
    frozen = env.snapshot()
    frozen.save("cut-ready.lcssnap")  # 已存在则拒绝覆盖
    loaded = EnvSnapshot.load("cut-ready.lcssnap", dll_sha256=context["dll_sha256"],
                              physics_config_sha256=frozen.metadata["physics_config_sha256"])
    branch_initial, branch_context = env.restore(loaded, branch_label="candidate-01")
    # 使用 env.step(action) 产生分支自己的 N 条转移、N+1 个状态。
```

快照只能在完整一步结束后操作。状态保存包含原生设备/主机缓冲、Python 监视器、动作限速器历史、RNG 和任务历史；恢复不会重放准备前缀。每次恢复生成新回合 ID，保留父回合、根谱系、分叉步与快照摘要。可以通过 `restore(..., goal=...)` 更换任务/奖励目标；不传 `goal` 继承任务历史，传 `None` 移除任务。不能改变材料、接触、切割许可或质量检查配置。

落盘载入检查 DLL SHA、状态 ABI、物理配置哈希；实际恢复另检查 Python 实现版本。跨版本、跨配置或损坏的数据被拒绝。同步视觉启用时必须附着相同配置的 `SynchronizedVision`，恢复后重新渲染。恢复中途出现原生复制失败或渲染失败，会关闭该会话；不得继续使用部分恢复的状态。

## 参数与可观测量

```python
from endoscopic_tissue.training.physics import PhysicsOverrides
from endoscopic_tissue.training.geometry import WallGeometry

physics = PhysicsOverrides(young_pa=8000, start_force_n=0.4)
scene = SceneConfig("wall-traction", "cut_ready", physics=physics,
                    geometry=WallGeometry(length_m=0.024))
```

| 字段 | 支持的实验输入 | 语义 |
|---|---|---|
| `young_pa` | 3000–24000 Pa | 材料刚度 |
| `start_force_n` | 0.2–0.6 N | 保留双刃各半起切力与额外 2 mm 挤压规则 |
| `blade_friction` | 0.05–0.4 | 刃摩擦 |
| `grasp_friction` | 0.3–1.2 | 只允许显式纯摩擦模式 1；同步调整真实静/动摩擦，保持原始比例 |
| `tear_stretch_ratio` | 0，或 1.3–2.0 | 0 关闭；其余值须分别完成名义与剂量验证 |
| `tear_exempt_rings` | 整数 0–2 | 原默认 1 圈 |
| `grasp_core_break_force_n` | 0.1–150 N | 原默认 150 N；用于辅助夹持模式的支持角点 EMA 释放 |

这些范围是输入边界，不是已验证安全域。没有开放挤压距离、接触内部常数、求解迭代预算。所有实际 `Settings` 字段与导入时冻结的规范默认值比较，覆盖内容进入上下文与哈希。旧数据审计入口为 `training/audit_physics.py`，只读，不改写原回合。

`EnvironmentConfig(failure_telemetry=True)` 显式启用只读损伤诊断。逐步结果位于 `State.metadata.failure_physics`；准备阶段也保留逐步 `failure_physics`，避免漏掉夹取、抬升峰值。撕裂原因由原生事件记录，分为刃切/撕裂、切口边缘/体内；夹持邻域是可与前两者重叠的空间标记。支持角点的力来自当前统一求解器实际材料夹持点的约束力，EMA 时间常数为 1 s；超过阈值时释放其支持的实际见证点，不添加新的固定点。切向载荷尚不能独立导出，字段明确为 `None`。

原 DLL 虽有旧版 `k_CoreBreakCheck`，当前统一求解路径并未调用它。因此新的活动失败律需要显式选择训练 ABI DLL；不能把旧 DLL 的传参成功当成锚点释放有效的证明。

几何族支持长度、高度、厚度、固定边、切面高度，切面仍轴对齐，网格至少 1 mm。固定边指支撑块的边界面，不会增加隐藏夹持。准备轨迹由同一个几何公式推导。几何组按实际形状划分；只改固定边或切面不会被伪装成新的留出解剖形状。

## 构建与证据

所有构建脚本位于 `endoscopic_tissue_backups/2026-11-01-s15-s17`。首次在新的检出中，依次运行：

```powershell
$py = "D:\develop\anaconda3\envs\genesis_world\python.exe"
$s15 = "D:\Simulation\genesis-world\endoscopic_tissue_backups\2026-11-01-s15-s17"
& $py -X utf8 "$s15\prepare_native_base.py"
& $py -X utf8 "$s15\prepare_training_native.py"
& $py -X utf8 "$s15\inventory_state.py"
& $py -X utf8 "$s15\prepare_snapshot_native.py"
& $py -X utf8 "$s15\build_native.py" --variant training-snapshot1
```

当前工作区已经生成源码，不要重复执行首次导出命令。脚本会拒绝覆盖已有首次导出目录。需要刷新隔离源码时，用生成脚本明确提供的 `--refresh`；不能把隔离产物复制到默认目录。CUDA/MSVC 路径在 `build_native.py` 内记录。构建清单保存源文件 SHA 和输出 DLL SHA。

验收脚本与证据位于 `endoscopic_tissue/outputs/s15-s17-20261101`：

| 检查 | 入口 | 必须看到的证据 |
|---|---|---|
| Python 契约 | `run_cpu_tests.py` | 严格参数、快照编码、分支数据、几何与任务检查 |
| C++ 异常恢复 | `run_host_state_tests.py` | CPU 故障注入，非法释放与遗留分配均为 0；不代表 CUDA 验收 |
| S15 原生往返 | `validate_native_snapshot.py` | T1 字节一致、T2 即刻观测一致、T3 单步 <0.01 mm |
| S15 环境与资源 | `validate_environment_snapshot.py` | 新会话恢复、分支隔离、拓扑回退/再剪、同步视觉、1000 次循环 |
| S15 分布 | `validate_snapshot_distribution.py` | 四类时刻各 20 对；同一个 cut_ready 缓存对 20 次新准备 |
| S15 汇总 | `summarize_snapshot.py` | T1–T8 证据、实际 DLL/Python 绑定、无并行干扰；缺失或失败不通过 |
| S16 名义/剂量 | `validate_damage.py`、`validate_failure_dose.py` | 完整原脚本与五方向 0–4 mm、0.5 mm 间隔、每档 10 次 |
| S16 汇总 | `summarize_failure.py` | 原生力峰值、重力伸长余量、起切/断边比较、首次失败剂量与位置 |
| S17 几何 | `prepare_geometry_jobs.py`、`summarize_geometry.py` | 三变体各五次全脚本；真实形状留出，无数据泄漏 |
| 默认耗时/视频 | `compare_default.py` 与 `validate_damage.py` | 相同脚本、ABBA 顺序、完整夹取/剪切/撤离；录像耗时与物理耗时分开 |

`gpu_queue.py` 只运行自身的串行测试，其他任务占用 GPU 时自动等待，不结束用户进程。`gpu-queue.json` 可追加独立任务；已有输出不覆盖，失败保留。快照的大样本任务以冒烟结果为前提。剂量试验使用快照缓存前，必须有同一 DLL 的 S15 验收文件；否则只能显式运行新准备的 `--fresh` 试验。

最终仍须单独核对完整交互视频和默认关闭时的帧率。静态核指令一致、CPU 测试通过或某次 `passed=true` 都不能单独批准替换默认版本。S16d 纯摩擦剂量实验仍由需求提出方 WAM 负责，本侧提供接口与现有问题记录。
