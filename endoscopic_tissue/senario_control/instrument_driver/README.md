# 夹爪和剪刀的单步驱动

驱动接收外部已有的夹爪、剪刀实例和每一步的时间间隔，更新空间位置、朝向和关节状态。
器械创建、配置加载、时钟及动作调度由外部负责。这里没有场景管理实现。

```text
instrument_driver/
├── motion_bridge.py  绑定已有实例、校验运动输入、计算单步更新并返回位姿
├── __init__.py       MotionBridge 导入入口
└── README.md         输入格式和职责说明
```

## 接口

```python
from endoscopic_tissue.senario_control.instrument_driver import MotionBridge

# grasper and scissors are existing instances supplied by the caller.
driver = MotionBridge(grasper=grasper, scissors=scissors)

# The caller invokes this once for each fixed external timestep.
state = driver.step({
    "grasper": {
        "linear_velocity_cm_s": [1.0, 0.0, 0.0],
        "angular_velocity_deg_s": [0.0, 0.0, 30.0],
        "opening_deg": 10.0,
    },
    "scissors": {"linear_velocity_cm_s": [0.0, -0.5, 0.0]},
}, dt_s=0.01)
```

上述一步使夹爪沿世界 X 轴移动 0.01 cm、绕世界 Z 轴旋转 0.3 度；
剪刀沿世界 Y 轴移动 -0.005 cm。开合关节同时按已有速度限制接近 10 度目标。

| 输入 | 含义 |
| --- | --- |
| `base_position_cm` | 本步指定的底座世界 XYZ 位置，厘米 |
| `base_quaternion_xyzw` | 本步指定的底座世界朝向，四元数 `(x,y,z,w)` |
| `linear_velocity_cm_s` | 世界 XYZ 方向线速度，厘米/秒 |
| `angular_velocity_deg_s` | 世界 XYZ 轴角速度向量，度/秒 |
| `roll_deg / pitch_deg / yaw_deg / opening_deg` | 关节目标角，度 |
| `dt_s` | 外部给定的本步时间间隔，秒；必须为正数 |

位置按 `p_next = p_current + v * dt_s` 更新。
角速度向量转换为本步旋转向量，再左乘当前朝向，因此角速度相对世界轴定义。
底座转动围绕底座自己的位置进行；各部件位姿随后由 `instrument_control` 的父子变换计算。

同一次输入中，绝对位置与线速度不能同时指定；绝对朝向与角速度也不能同时指定。
线速度、角速度会持续作用到后续 `step()`，可以用零向量停止。
传入绝对位置会停止已有线速度；传入绝对朝向会停止已有角速度。
绝对位姿和关节目标在下一步被应用一次，之后状态保存在外部器械实例中。

## 调用职责

- `MotionBridge(grasper, scissors)`：绑定已有实例，保留其当前状态。
- `submit(input)`：校验并暂存指令，不改变器械状态。
- `step(input, dt_s=...)`：每次只执行一个外部时间步。
- `step(dt_s=...)`：应用暂存指令，或继续已有速度和关节目标。
- `snapshot()`：返回关节状态、底座与部件位姿、间隙及杆体端点。

两把器械的输入和本步位置、旋转变化先完成校验，再写入实例。
返回的数据可序列化为 JSON，输出不包含内部计时或步数。
`is_cutting` 可作为外部剪切标记传入；本驱动不计算组织剪切。

整段轨迹可以在外部调用已有的 `instrument_control.smoothing.joint_transition()` 进行采样，
每取得一个关节目标样本，再调用一次驱动 `step()`。
测试和调用样例见 [HYC_TEST](../HYC_TEST/README.md)。
