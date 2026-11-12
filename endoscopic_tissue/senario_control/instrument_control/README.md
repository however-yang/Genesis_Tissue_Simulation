# 器械运动计算

这个文件夹用于计算夹爪和剪刀的关节运动、开合、各部件位姿及动作之间的平滑过渡。
输入是底座位姿、关节目标角、配置参数和时间步长；输出是关节状态、夹口间隙及世界变换矩阵。
计算依赖 NumPy、SciPy 和 Python 标准库，目录内代码只导入本目录的模块。

```text
instrument_control/
├── configs/instruments.toml  两把器械的尺寸、初始状态、转轴和速度限制
├── config.py                 JointConfig、InstrumentConfig、MotionConfig 与配置加载
├── articulation.py           单个关节实例、目标角和速度限制
├── instruments.py            夹爪、剪刀实例以及父子坐标变换
├── commands.py               末端运动目标数据
├── joints.py                 末端目标更新、校验和方向计算
├── smoothing.py              末端插值与关节平滑目标序列
└── __init__.py               对外导入入口
```

## 三处关节、四个自由度

夹爪和剪刀分别使用独立的 `Grasper`、`Scissors` 实例，每把器械都有自己的底座和四个关节状态。

```text
可平移底座
    └── Roll：绕杆轴旋转
          └── Pitch：腕部弯曲
                └── Yaw：末端整体转向
                      ├── 上瓣：+opening / 2
                      └── 下瓣：-opening / 2
```

Yaw 和 opening 共享最后一处关节位置，各占一个独立自由度。
底座还可以单独平移，并通过四元数设置整体方向。
改变 Pitch 或 Yaw 时，末端围绕对应关节位置运动；世界变换由父子矩阵依次相乘得到。

## 参数与状态

- 位置、连接偏移、杆长、夹口间隙使用厘米，世界 Y 轴向上。
- 关节角使用度，角速度使用度/秒，时间步长使用秒。
- `opening_deg` 是两瓣总夹角，每瓣相对中心线旋转其一半。
- `base_position_cm` 是最上方 Roll 关节的位置。
- 底座四元数顺序为 `(x, y, z, w)`；关节 `axis`、`origin_cm` 相对父节点定义。
- 零角度时，本地 X 朝向尖端、Y 朝向开合方向、Z 是开合轴。
- `target_deg` 是目标角，`angle_deg` 是当前控制角，`velocity_deg_s` 是本步控制角速度。

`load_config()` 返回 `MotionConfig`，只接受 `[grasper]` 和 `[scissors]` 两组运动配置。
初始数值沿用此前配置；场景和物理参数在上一级 `configs/kidney.toml` 中。

## 单步运动计算

```python
from endoscopic_tissue.senario_control.instrument_control import Grasper, Scissors, load_config

config = load_config("endoscopic_tissue/senario_control/instrument_control/configs/instruments.toml")
grasper = Grasper(config.grasper)
scissors = Scissors(config.scissors)

grasper.set_base_pose(position_cm=(-3.4, 8.8, 0.0))
grasper.set_joint_targets(roll_deg=20, pitch_deg=10, yaw_deg=-15, opening_deg=10)
scissors.set_joint_targets(roll_deg=-10, pitch_deg=15, yaw_deg=20, opening_deg=40)

dt_s = 1 / 30
grasper.advance(dt_s)
scissors.advance(dt_s)
grasper_pose = grasper.pose()
scissors_pose = scissors.pose()
roll_state = grasper.roll.state
```

`set_joint_targets()` 记录目标，`advance()` 按角速度限制推进，`pose()` 计算当前位姿。
`InstrumentPose` 包含杆体、Roll、Pitch、Yaw、上瓣和下瓣的 4×4 世界变换，以及夹口间隙和杆体端点。
外部渲染器或仿真程序可以读取这些结果。
`set_base_pose()` 直接设置底座位姿；底座轨迹由调用者逐步提供。

## 平滑关节运动

```python
from endoscopic_tissue.senario_control.instrument_control import JointTargets, joint_transition

targets = (
    (grasper, JointTargets(yaw_deg=15, opening_deg=10)),
    (scissors, JointTargets(pitch_deg=20, yaw_deg=-10, opening_deg=30)),
)
for fraction in joint_transition(targets, duration_s=2.0, dt_s=dt_s):
    grasper.advance(dt_s)
    scissors.advance(dt_s)
    grasper_pose = grasper.pose()
    scissors_pose = scissors.pose()
```

`joint_transition()` 沿用三次 smoothstep 曲线，将时长向上取整为完整时间步，并检查峰值角速度。
每次迭代只设置该步目标角，调用者随后执行 `advance(dt_s)`、读取 `pose()`，再进入下一次迭代。
未指定的坐标保留已有目标。原有末端指令可使用 `Command`、`interpolate()` 和 `Stage`。

可选的 `observe()` 只记录外部提供的数值，并把间隙换算为开合角；
`set_cutting()` 和指令中的剪切开关只记录供外部读取的标记。
物理求解、组织剪切和接触反馈处理由[上一级模块](../README.md)负责。

阅读顺序建议：`config.py` → `articulation.py` → `instruments.py` → `smoothing.py`。
