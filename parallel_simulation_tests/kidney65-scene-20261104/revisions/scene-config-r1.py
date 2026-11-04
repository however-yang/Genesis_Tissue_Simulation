"""Isolated 65k kidney scene; the published scene and solver stay unchanged."""
from dataclasses import dataclass, fields, replace

import numpy as np

from endoscopic_tissue import rendering, scenario
from endoscopic_tissue.grid import build_grid
from endoscopic_tissue.native_backend import scissors_frame


@dataclass(frozen=True)
class KidneyGridScene(scenario.ScissorsScene):
    lateral_spacing: float = .125

    def build_grid(self, density):
        grid = build_grid(self.lateral_spacing, density=density,
                          vertical_spacing=self.vertical_spacing, shape=self.shape)
        count = int(grid.inside.sum())
        if not 60000 <= count <= 70000:
            raise ValueError(f"Expected 60k-70k material nodes, got {count}")
        return grid


CAMERA = ((0., -.115, .085), (0., 0., .020))
GRASPER_AXIS = (.8, -.6, 0.)
SCISSORS_YAW_DEG = -45.
FAT_SURROUNDS = (
    ((0., -2.6, 0.), (7.1, 1.7, 3.0)),
    ((-6., -.7, -.1), (1.6, 1.5, 3.1)),
    ((6.1, -.65, .2), (1.7, 1.35, 3.0)),
    ((0., -.15, -4.6), (7.4, 2.4, 2.0)),
    ((0., 1., -15.), (18., 10., 4.)),
)


def configure(run, preset, lateral_mm=1.25):
    base = scenario.KIDNEY_SCENE
    original_settings = scenario.scissors_settings
    if preset == "baseline":
        return {"preset": preset, "material_nodes_expected": 102710,
                "spacing_mm": [1., 1., 1.], "camera": base.shot}, {}
    values = {field.name: getattr(base, field.name) for field in fields(scenario.ScissorsScene)}
    scene = KidneyGridScene(**values, lateral_spacing=lateral_mm / 10)
    view_options = {}
    metadata = {"preset": preset, "spacing_mm": [lateral_mm, 1., lateral_mm],
                "shape_dimensions_and_material_unchanged": True,
                "cut_normal_resolution_mm": 1., "physics_equivalence_claimed": False,
                "mesh_dependent_native_tolerances_follow_new_grid_spacing": True}
    if preset == "frontal65":
        # Preserve each prescribed grasper tip location while changing its
        # actual approach axis. Rendering and collision share this new axis.
        old = np.asarray(original_settings().jaw_axis)
        new = np.asarray(GRASPER_AXIS)
        hinge_offset = original_settings().jaw_length * (old - new)
        old_axis = np.asarray(scenario.KIDNEY_AXIS)
        new_axis = scissors_frame(0., 90., SCISSORS_YAW_DEG)[0]
        neck = np.array((.06, scenario.KIDNEY_PLANE_Y - scenario.SCISSORS_FACE_WIDTH, .01))
        stages = []
        for stage in base.stages:
            command = stage.command
            reach = float(np.dot(neck - np.asarray(command.blade_center), old_axis))
            updated = replace(command,
                              hinge=tuple(np.asarray(command.hinge) + hinge_offset),
                              blade_center=tuple(neck - reach * new_axis),
                              scissors_yaw_deg=SCISSORS_YAW_DEG)
            stages.append(replace(stage, command=updated))
        scene = replace(scene, stages=tuple(stages), advance=tuple(new_axis), shot=CAMERA,
                        closeup_offset=(0., -.055, .035))
        run.scissors_settings = lambda: replace(original_settings(), jaw_axis=GRASPER_AXIS)
        rendering.FAT_BACKDROP = FAT_SURROUNDS
        view_options = {"camera_fov": 43, "lamp_gain": .90, "roughness_scale": 1.18}
        metadata.update(grasper_axis=GRASPER_AXIS, scissors_yaw_deg=SCISSORS_YAW_DEG,
                        prescribed_grasper_tip_locations_preserved=True,
                        camera=CAMERA, visual_only_background=True, view_options=view_options,
                        note="New physical tool poses require their own interaction validation")
    else:
        metadata["camera"] = base.shot
    # The original CLI and cut-ribbon diagnostics must see the actual lateral
    # spacing. This changes only this process, never the frozen source files.
    scenario.SCISSORS_SPACING = scene.lateral_spacing
    run.SCISSORS_SPACING = scene.lateral_spacing
    run.SCISSORS_SCENES = {"kidney": scene}
    return metadata, view_options
