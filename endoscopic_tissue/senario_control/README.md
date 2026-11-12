# 场景控制与器械运动

[instrument_control](instrument_control/README.md) 专门计算夹爪和剪刀的运动。
模拟器、CUDA 适配、物理参数、剪切控制和肾脏场景保留在本层。
老师原有的 endoscopic_tissue 文件保持原样。

```text
senario_control/
├── instrument_control/   器械关节运动、位姿、开合和纯运动插值
├── instrument_driver/    外部输入衔接、夹爪和剪刀驱动
├── HYC_TEST/             纯器械运动测试、输入样例及结果
├── configs/kidney.toml   场景、器械初始配置及可选物理参数
├── config.py             场景配置加载：ScenarioConfig
├── commands.py           物理参数 Settings，并导出共享 Command
├── cutting.py            剪切开关与剪刀物理默认参数
├── motion_feedback.py    依据物理反馈调整退刀起点和驱动目标
├── cuda_adapter.py       器械运动目标到原生 CUDA 接口的适配
├── simulator.py          SimulationInterface 与组织仿真步进
├── kidney.py             肾脏几何和预设动作序列
├── example.py            场景查看及主动运行模拟的命令行入口
└── test_interface.py     接口检查代码
```

## 只计算器械运动

```python
from endoscopic_tissue.senario_control.instrument_control import Grasper, Scissors, JointTargets, load_config

config = load_config("endoscopic_tissue/senario_control/instrument_control/configs/instruments.toml")
grasper = Grasper(config.grasper)
scissors = Scissors(config.scissors)
```

器械目录的 `load_config()` 返回 `MotionConfig`，只包含两把器械的配置。
单步使用 `set_joint_targets()` → `advance(dt_s)` → `pose()`；
平滑动作使用 `instrument_control.smoothing.joint_transition()`，具体例子见器械目录说明。

外部字典或 JSON 输入可通过 [instrument_driver 的 MotionBridge](instrument_driver/README.md) 接入，
测试结果见 [测试报告](HYC_TEST/results/test_report.md)。
驱动入口为 `MotionBridge(grasper, scissors)`，接收已有实例；每步时间间隔由外部传入。

## 接入组织仿真

```python
from endoscopic_tissue.senario_control.simulator import SimulationInterface

with SimulationInterface(config="endoscopic_tissue/senario_control/configs/kidney.toml") as simulation:
    simulation.scissors.set_joint_targets(pitch_deg=15, yaw_deg=20, opening_deg=40)
    state = simulation.step_joints()
```

这里的 `SimulationInterface` 才会创建原生会话和执行物理步进。
它调用器械目录计算运动目标，再在本层执行预载限制、CUDA 求解和反馈更新。
`SimulationInterface` 也可以接收器械目录加载的 `MotionConfig`，场景默认使用 kidney。
本层 `config.load_config()` 或 `senario_control.load_config()` 返回带场景和物理参数的 `ScenarioConfig`。

关节实例模式使用 `step_joints()`、不带指令的 `step()` 和 `transition_joints()`；
原有末端动作模式使用 `step(command)`、`transition()` 和 `run_stages()`。
两种模式继续使用原有运动逻辑，关节平滑曲线由纯运动模块提供。

关节的当前控制角与物理求解结果分别保存。
`InstrumentPose` 是运动学目标，`state.command` 是驱动限制后发送的目标；
`state.observation` 和器械的 `feedback` 是实际求解反馈。
当前 CUDA 求解整体平移、开合、组织接触与断裂，Roll、Pitch、Yaw 仍为运动学驱动。
腕部各段的转动惯性、关节力矩、独立碰撞及渲染接入需要后续扩展。

## 肾脏与预设场景

`kidney.py` 维护肾脏与肿瘤几何，以及抓取、牵拉、剪切、退刀、取出的预设动作。
它从器械目录读取共享运动指令和动作类型。
`example.py` 保留场景查看入口：`python -m endoscopic_tissue.senario_control.example --scene kidney`。
