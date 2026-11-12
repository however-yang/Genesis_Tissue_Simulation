"""把场景参数打包成 HYC CUDA 输入，不维护独立的器械默认值。

调用方先用 resolve_settings 解析场景，再传入实际网格间距。
几何沿用 NativeTissue 的模型加载器，打包过程不创建 CUDA 物理会话。
SceneInput 可上传 GPU，随后依次调用 k_LoadSceneInputs、k_InitializeInstruments。
"""
import ctypes as ct
from math import isfinite, sqrt

import numpy as np

from ..instrument_geometry import SCISSORS_WRIST_FRONT, load_grasper, load_scissors
from .config import ScenarioConfig
from .commands import Settings


# 传输协议使用 float32、int32 和连续标量数组，对应 scene_input.cuh 的字段布局。
F = ct.c_float
I = ct.c_int32
V3 = F * 3
V4 = F * 4
Pair3 = V3 * 2


class SceneJointInput(ct.LittleEndianStructure):
    """单关节的父坐标系转轴、铰点偏移、角度范围及速度限制。"""
    _fields_ = [("axis", V3), ("origin_cm", V3)]
    _fields_ += [(name, F) for name in ("lower_deg", "upper_deg", "initial_deg", "max_speed_deg_s")]


class SceneBackInput(ct.LittleEndianStructure):
    """刀背有向盒：局部中心、半尺寸和两个正交方向。"""
    _fields_ = [(name, V3) for name in ("center", "half_size", "axis_x", "axis_y")]


class SceneInstrumentInput(ct.LittleEndianStructure):
    """单把器械的完整参数包；字段顺序必须与 CUDA 结构一致。"""
    _fields_ = [("joints", SceneJointInput * 4), ("base_position_cm", V3), ("base_quaternion_xyzw", V4)]
    _fields_ += [(name, F) for name in ("mass_kg", "stiffness", "damping", "max_travel_cm")]
    _fields_ += [("guided", I)]
    _fields_ += [(name, F) for name in ("jaw_mass_kg", "jaw_stiffness", "jaw_damping", "jaw_speed_cm_s",
                                       "opening_speed_multiplier", "jaw_length_cm", "jaw_half_width_cm",
                                       "max_gap_cm", "shaft_length_cm", "shaft_radius_cm")]
    _fields_ += [("use_jaw_bounds", I), ("jaw_bounds_center", Pair3), ("jaw_bounds_half", Pair3), ("scissors", I)]
    _fields_ += [("edge_a", Pair3), ("edge_b", Pair3), ("scissors_radius_cm", F), ("back_contact", I)]
    _fields_ += [("backs", SceneBackInput * 2)]
    _fields_ += [(name, F) for name in ("wrist_front_cm", "wrist_length_cm", "wrist_radius_cm")]


class SceneInput(ct.LittleEndianStructure):
    """版本、帧时长、子步数，以及按夹爪/剪刀顺序排列的两个参数包。"""
    _fields_ = [("abi_version", ct.c_uint32), ("frame_dt_s", F), ("substeps", I),
                ("instruments", SceneInstrumentInput * 2)]


# 这些数字是传输结构的字节数，不是器械参数默认值。
assert ct.sizeof(SceneJointInput) == 40
assert ct.sizeof(SceneInstrumentInput) == 468
assert ct.sizeof(SceneInput) == 948


def _positive(value, name, *, zero=False):
    """检查数值有限且符合正数/非负数约束，避免布尔值被当作物理系数。"""
    if isinstance(value, (bool, np.bool_)) or not isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'non-negative' if zero else 'positive'}")


def _finite_transport(value):
    """检查转换为 float32 后的字段；整数协议字段不按浮点位模式检查。"""
    if isinstance(value, float):
        return isfinite(value)
    if isinstance(value, ct.Array):
        return all(_finite_transport(item) for item in value)
    if isinstance(value, ct.Structure):
        return all(_finite_transport(getattr(value, field[0])) for field in value._fields_)
    return True


def build_scene_input(config: ScenarioConfig, settings: Settings, *, grid_spacing_cm: float):
    """接收场景已解析的配置、物理参数和网格间距，输出可上传 GPU 的参数包。

    base_position_cm/shaft_length_cm/gap 使用厘米，关节角和速度使用度/秒制。
    质量使用千克，刚度和阻尼保持老师的原生单位；本函数不重新设置默认值。
    """
    _positive(grid_spacing_cm, "grid_spacing_cm")
    _positive(settings.dt, "dt")
    if isinstance(settings.substeps, bool) or not isinstance(settings.substeps, int) or not 0 < settings.substeps <= 2**31 - 1:
        raise ValueError("substeps must be a positive int32")
    if settings.cutting_tool != "scissors":
        raise ValueError("HYC scene inputs require cutting_tool='scissors'")
    for name in ("tool_mass_kg", "tool_drive_stiffness", "jaw_mass_kg", "jaw_drive_stiffness",
                 "jaw_speed_cm_s", "jaw_opening_speed_multiplier", "scissors_jaw_mass_kg",
                 "blade_mass_kg", "blade_drive_stiffness", "scissors_jaw_stiffness", "jaw_width"):
        _positive(getattr(settings, name), name)
    for name in ("tool_drive_damping", "tool_travel_fraction", "jaw_drive_damping", "contact_radius_cm"):
        _positive(getattr(settings, name), name, zero=True)
    for name in ("grasper_guided", "scissors_guided"):
        if not isinstance(getattr(settings, name), (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")

    # 使用实际网格资产计算接触代理，避免在 CUDA 头文件里再保存一份几何数值。
    geometry = load_grasper(config.grasper.jaw_length_cm, settings.jaw_width, settings.instrument_model)
    blades = load_scissors()
    if not np.isclose(config.scissors.jaw_length_cm, blades.jaws.length):
        raise ValueError("Scissors jaw length must match the loaded blade geometry")
    scene = SceneInput()
    scene.abi_version, scene.frame_dt_s, scene.substeps = 1, settings.dt, settings.substeps
    # 老师的驱动距离限制按网格间距和子步数缩放，这是每个子步的目标移动上限。
    travel = grid_spacing_cm * settings.tool_travel_fraction / settings.substeps
    for index, instrument in enumerate((config.grasper, config.scissors)):
        out = scene.instruments[index]
        # 关节配置来自场景；排列顺序与 CUDA 的 JointIndex 枚举一致。
        for joint_index, name in enumerate(("roll", "pitch", "yaw", "opening")):
            joint = getattr(instrument, name)
            out.joints[joint_index] = SceneJointInput(V3(*joint.axis), V3(*joint.origin_cm),
                joint.lower_deg, joint.upper_deg, joint.initial_deg, joint.max_speed_deg_s)
        out.base_position_cm = V3(*instrument.base_position_cm)
        out.base_quaternion_xyzw = V4(*instrument.base_quaternion_xyzw)
        out.jaw_length_cm, out.jaw_half_width_cm = instrument.jaw_length_cm, settings.jaw_width / 2
        out.max_gap_cm, out.shaft_length_cm = instrument.max_gap_cm, instrument.shaft_length_cm
        out.max_travel_cm = travel
        out.jaw_speed_cm_s = settings.jaw_speed_cm_s
        out.opening_speed_multiplier = settings.jaw_opening_speed_multiplier
        out.scissors = index
        if index == 0:
            # 夹爪读取场景的 tool_* 平移参数和 jaw_* 开合参数。
            # bounds 每瓣包含中心及半尺寸；不复制网格顶点到参数包。
            out.mass_kg, out.stiffness, out.damping = settings.tool_mass_kg, settings.tool_drive_stiffness, settings.tool_drive_damping
            out.guided = int(settings.grasper_guided)
            out.jaw_mass_kg, out.jaw_stiffness, out.jaw_damping = settings.jaw_mass_kg, settings.jaw_drive_stiffness, settings.jaw_drive_damping
            out.shaft_radius_cm = geometry.shaft_radius
            out.use_jaw_bounds = 1
            bounds = geometry.bounds.reshape(2, 2, 3)
            for side in range(2):
                out.jaw_bounds_center[side] = V3(*bounds[side, 0])
                out.jaw_bounds_half[side] = V3(*bounds[side, 1])
        else:
            # 剪刀读取 blade_* 与 scissors_jaw_* 参数，沿用老师的临界阻尼公式。
            # 两个阻尼分别按平移质量/刚度与开合质量/刚度计算。
            out.mass_kg, out.stiffness = settings.blade_mass_kg, settings.blade_drive_stiffness
            out.damping = 2 * sqrt(out.mass_kg * out.stiffness)
            out.guided = int(settings.scissors_guided)
            out.jaw_mass_kg, out.jaw_stiffness = settings.scissors_jaw_mass_kg, settings.scissors_jaw_stiffness
            out.jaw_damping = 2 * sqrt(out.jaw_mass_kg * out.jaw_stiffness)
            out.shaft_radius_cm = blades.jaws.shaft_radius
            # 刃口胶囊半径采用原接触规则，并随场景的 contact_radius_cm 更新。
            out.scissors_radius_cm = max(settings.contact_radius_cm, .005 * instrument.jaw_length_cm)
            out.back_contact = 1
            for side in range(2):
                out.edge_a[side] = V3(*blades.edges[side, 0])
                out.edge_b[side] = V3(*blades.edges[side, 1])
                values = blades.backs[side].reshape(4, 3)
                out.backs[side] = SceneBackInput(*(V3(*vector) for vector in values))
            out.wrist_front_cm = SCISSORS_WRIST_FRONT
            out.wrist_length_cm, out.wrist_radius_cm = instrument.shaft_length_cm, blades.jaws.shaft_radius
    # Python double 有限，不代表转换后的 float32 仍有限；子步时长还可能下溢为零。
    if not _finite_transport(scene) or ct.c_float(scene.frame_dt_s / scene.substeps).value <= 0:
        raise ValueError("Scene parameters must fit finite native float32 values")
    return scene
