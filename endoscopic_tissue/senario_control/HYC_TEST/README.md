# HYC 单步驱动测试

本目录测试 [instrument_driver/motion_bridge.py](../instrument_driver/motion_bridge.py)。
测试代码显式创建器械并传给驱动；器械创建、配置加载和计时均不在驱动实现中。

```text
HYC_TEST/
├── test_motion_bridge.py  已有实例绑定、外部步长、空间运动、关节及异常输入测试
├── sample_inputs.json     外部目标输入样例
├── run_inputs.py          测试调用方：提供实例和步长，读取样例并保存结果
└── results/               测试报告和逐步结果
```

## 驱动调用

```python
from endoscopic_tissue.senario_control.instrument_driver import MotionBridge

# The caller supplies existing instruments.
driver = MotionBridge(grasper, scissors)
state = driver.step({
    "grasper": {"linear_velocity_cm_s": [1, 0, 0], "angular_velocity_deg_s": [0, 0, 30]},
    "scissors": {"opening_deg": 40},
}, dt_s=0.01)
```

完整输入格式、速度保持规则和坐标约定见 [驱动说明](../instrument_driver/README.md)。
`submit()` 只暂存输入；只有 `step()` 才更新状态。
驱动接收每一步的 `dt_s`，自身没有计时器、步数计数器或整段动作执行方法。

## 外部平滑采样

```python
from endoscopic_tissue.senario_control.instrument_control import JointTargets, joint_transition

targets = (
    (grasper, JointTargets(pitch_deg=30, opening_deg=10)),
    (scissors, JointTargets(yaw_deg=-30, opening_deg=40)),
)
for fraction in joint_transition(targets, duration_s=2.0, dt_s=0.01):
    state = driver.step(dt_s=0.01)
```

采样由外部循环控制；驱动只处理每次调用的一步。
测试包含真实关节位置上的转动、两瓣间隙、速度限制、外部平滑采样、输入原子性和输出独立性。
新增检查覆盖绑定已有实例、世界线速度、世界角速度、绝对位姿停止速度，以及错误帧保留已暂存指令。

## 运行

在 `C:\Simulation_Study` 下执行：

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest endoscopic_tissue.senario_control.HYC_TEST.test_motion_bridge -v
& '.\.venv\Scripts\python.exe' -B -m endoscopic_tissue.senario_control.HYC_TEST.run_inputs
```

样例由测试调用方运行 180 步，每步 1/60 秒，共 3 秒。
`run_inputs.py` 在输出中附加它自己计算的步数和时间；这些字段由调用方维护。
结果保存到 `results/sample_motion.json`，测试记录见 `results/test_report.md`。
本目录不执行 CUDA 或组织剪切仿真。
