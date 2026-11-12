"""Combine instrument motion configuration with scene selection and physical parameters.

Pure instrument geometry and joint-limit validation live in instrument_control.config.
This loader retains the scene/physics TOML format consumed by SimulationInterface.
"""
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

from .commands import Settings
from .cutting import scissors_settings
from .instrument_control.config import InstrumentConfig, grasper_config, scissors_config, update_instrument


@dataclass(frozen=True)
class ScenarioConfig:
    """Bind the two instrument configurations to a scene and optional physical settings."""
    scene: str
    grasper: InstrumentConfig
    scissors: InstrumentConfig
    physics: Settings | None = None


def load_config(path: str | Path):
    """Read a TOML file with scene, grasper, scissors and optional physical settings.

    Missing instrument fields retain the matching defaults. Physical settings retain native parameter conventions.
    A pure motion TOML is also accepted, using the default kidney scene and scene-specific physics.
    """
    with Path(path).open("rb") as source:
        values = tomllib.load(source)
    unknown = set(values) - {"scene", "grasper", "scissors", "physics"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    physics = None
    if "physics" in values:
        section = dict(values["physics"])
        unknown = set(section) - {item.name for item in fields(Settings)}
        if unknown:
            raise ValueError(f"Unknown physical settings: {sorted(unknown)}")
        for key in ("jaw_axis", "jaw_normal", "blade_axis"):
            if key in section:
                section[key] = tuple(section[key])
        physics = replace(scissors_settings(), **section)
    return ScenarioConfig(
        scene=values.get("scene", "kidney"),
        grasper=update_instrument(grasper_config(), values.get("grasper", {})),
        scissors=update_instrument(scissors_config(), values.get("scissors", {})),
        physics=physics,
    )


def resolve_settings(config: ScenarioConfig, *, scene=None, settings=None):
    """为场景模拟器和 HYC CUDA 使用同一套物理参数解析规则。"""
    # 显式 settings 与文件中的 [physics] 不能同时提供，避免参数来源歧义。
    if settings is not None and config.physics is not None:
        raise ValueError("Supply physical settings through either settings or config.physics")
    if scene is None:
        from ..scenario import SCISSORS_SCENES
        try:
            scene = SCISSORS_SCENES[config.scene]
        except KeyError as error:
            raise ValueError(f"Unknown scissors scene: {config.scene}") from error
    # 优先取唯一的显式输入；没有提供物理参数时，才采用该场景的已有默认设置。
    resolved = settings if settings is not None else config.physics
    if resolved is None:
        resolved = replace(scissors_settings(), young_pa=scene.young_pa, contact_model=scene.contact_model)
    # 夹口长度和最大间隙统一服从器械配置，避免物理设置与模型尺寸各取一份值。
    return replace(resolved, jaw_length=config.grasper.jaw_length_cm, jaw_open_gap=config.grasper.max_gap_cm)
