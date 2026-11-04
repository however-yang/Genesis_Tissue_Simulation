"""Lit Genesis surface adapter for independently simulated, changing topology.

The external-node API is isolated here because Genesis debug meshes are unlit.
Contact stays in CUDA. Visual tessellation never reconnects cut components.
"""
from pathlib import Path
import numpy as np
import trimesh
import genesis as gs
from genesis.ext import pyrender
from PIL import Image

from .grid import TO_GENESIS, to_genesis, METERS_PER_UNIT
from .native_backend import Observation, Settings, Surface
from .visual_surface import refine_surface
from .instrument_geometry import knife_frame, knife_geometry, load_grasper, load_davinci_wrist, load_scissors


# One repeat of the fat texture spans this many cm of rest tissue.
FAT_TEXTURE_CM = 1.5


def cylinder_between(a, b, radius):
    a, b = to_genesis(np.array((a, b), dtype=np.float32))
    return trimesh.creation.cylinder(radius=radius * METERS_PER_UNIT, segment=(a, b), sections=32)


def tissue_materials(kind="fat"):
    directory = Path(__file__).parent / "assets/textures"
    if kind == "fat":
        albedo = Image.open(directory / "fat/adipose_albedo.png").convert("RGB")
        height = np.array(albedo.convert("L"), dtype=np.float32) / 255
        dy, dx = np.gradient(height)
        normal = np.stack((-dx * 5, dy * 5, np.ones_like(height)), axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        # RGBA rows remain four-byte aligned for arbitrary source widths in
        # the OpenGL uploader; RGB rows require an unpack alignment override.
        normal_map = pyrender.Texture(
            source=Image.fromarray(((normal * .5 + .5) * 255).astype(np.uint8)).convert("RGBA"),
            source_channels="RGBA")
        skin = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=albedo, baseColorFactor=(1, .98, .94, 1),
            normalTexture=normal_map, metallicFactor=0, roughnessFactor=.29,
            doubleSided=True, alphaMode="OPAQUE")
        # Fresh cut faces read as wet, slightly bleeding fat, so a slit whose
        # walls still touch remains visible against the intact surface.
        interior = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=albedo, baseColorFactor=(.82, .44, .40, 1),
            normalTexture=normal_map, metallicFactor=0, roughnessFactor=.24,
            doubleSided=True, alphaMode="OPAQUE")
        return skin, interior
    if kind != "liver":
        raise ValueError(f"Unknown tissue appearance: {kind}")
    albedo = Image.open(directory / "liver2.png").convert("RGB")
    height = np.array(Image.open(directory / "liver2_height.png").convert("L"), dtype=np.float32) / 255
    spec = np.array(Image.open(directory / "liver2_spec.png").convert("L"), dtype=np.float32) / 255
    # Derive a tangent normal texture from the original height field at runtime.
    dy, dx = np.gradient(height)
    normal = np.stack((-dx * 3, dy * 3, np.ones_like(height)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    normal_map = Image.fromarray(((normal * .5 + .5) * 255).astype(np.uint8))
    roughness = np.zeros((*spec.shape, 3), dtype=np.uint8)
    roughness[..., 0] = 255
    roughness[..., 1] = np.clip((.32 - .12 * spec) * 255, 0, 255).astype(np.uint8)
    skin = pyrender.MetallicRoughnessMaterial(
        baseColorTexture=albedo, baseColorFactor=(.82, .44, .49, 1),
        normalTexture=normal_map, metallicRoughnessTexture=Image.fromarray(roughness),
        metallicFactor=0, roughnessFactor=1, doubleSided=True, alphaMode="OPAQUE")
    interior = pyrender.MetallicRoughnessMaterial(
        baseColorTexture=albedo, baseColorFactor=(.62, .20, .25, 1),
        normalTexture=normal_map, metallicFactor=0, roughnessFactor=.42,
        doubleSided=True, alphaMode="OPAQUE")
    return skin, interior


class EndoscopicView:
    def __init__(self, settings: Settings, width=1280, height=960, viewer=False):
        self.settings = settings
        self.width, self.height = width, height
        self.dynamic_nodes = []
        self.grasper_geometry = load_grasper(settings.jaw_length, settings.jaw_width, settings.instrument_model)
        self.knife_geometry = knife_geometry(settings.blade_length, settings.instrument_model)
        self.scissors_geometry = load_scissors() if settings.cutting_tool == "scissors" else None
        self.wrist = load_davinci_wrist() if settings.instrument_model == "davinci" else ()
        self.skin, self.interior = tissue_materials()
        background_skin, _ = tissue_materials("liver")
        self.steel = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(.72, .75, .78, 1), metallicFactor=.92, roughnessFactor=.19)
        self.shaft = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(.028, .032, .038, 1), metallicFactor=.15, roughnessFactor=.25)
        self.cartridge = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(.055, .21, .47, 1), metallicFactor=.05, roughnessFactor=.32)
        self.edge = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(.88, .88, .83, 1), metallicFactor=.9, roughnessFactor=.16)
        gs.init(backend=gs.cpu, logging_level="warning")
        camera = (0, -0.098, 0.070)
        target = (0, 0, 0.003)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=settings.dt, gravity=(0, 0, 0)),
            viewer_options=gs.options.ViewerOptions(camera_pos=camera, camera_lookat=target,
                camera_fov=43, res=(width, height), realtime_factor=None),
            vis_options=gs.options.VisOptions(show_world_frame=False, shadow=True,
                background_color=(.006, .002, .003), ambient_light=(.045, .040, .038),
                lights=[gs.options.vis.PointLight(pos=(-.004, -.090, .065),
                            color=(1, .96, .90), intensity=.012),
                        gs.options.vis.PointLight(pos=(.004, -.090, .065),
                            color=(1, .97, .94), intensity=.010),
                        gs.options.vis.DirectionalLight(dir=(.7, .3, -.7),
                            color=(.85, .91, 1), intensity=.55),
                        gs.options.vis.DirectionalLight(dir=(-.5, -.2, -.7),
                            color=(1, .94, .87), intensity=.30)]),
            show_viewer=viewer)
        self.scene.add_entity(gs.morphs.Box(size=(0.16, 0.13, 0.01), pos=(0, 0, -0.018),
                                           fixed=True, collision=False),
                              surface=gs.surfaces.Plastic(color=(0.16, 0.035, 0.045), roughness=.55))
        # Supersampling limits staircase edges in the endoscopic output.
        self.camera = self.scene.add_camera(res=(width * 2, height * 2), pos=camera, lookat=target,
                                            fov=43, near=0.001, far=0.5, debug=True, GUI=False)
        self.scene.build()
        for center, scale in (((0, -3, 0), (10, 2.1, 10)),
                              ((-6.0, -0.8, 0.4), (1.8, 1.2, 3.4)),
                              ((6.0, -1.2, 1.0), (2.1, 1.3, 3.8)),
                              ((0.0, -1.0, 5.5), (4.3, 1.0, 1.6)),
                              ((0.0, .2, -6.0), (8.0, 3.5, 2.8))):
            mesh = trimesh.creation.icosphere(subdivisions=4)
            native = mesh.vertices * scale + center
            mesh.vertices = to_genesis(native)
            directions = (native - center) / scale
            uv = np.column_stack((np.arctan2(directions[:, 2], directions[:, 0]) / (2 * np.pi),
                                  np.arcsin(directions[:, 1]) / np.pi)) * 4
            mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)
            self.add_mesh(mesh, background_skin)

    def add_mesh(self, mesh, material):
        node = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True,
                                         name="endo_" + str(gs.UID()), is_marker=False,
                                         poses=np.eye(4)[None])
        with self.scene.visualizer.viewer_lock:
            self.scene.visualizer.context.add_external_node(node)
        return node

    def update(self, surface: Surface, obs: Observation, *, divisions=4):
        for node in self.dynamic_nodes:
            self.scene.clear_debug_object(node)
        self.dynamic_nodes.clear()
        refined = refine_surface(surface, divisions=divisions) if divisions > 1 else surface
        wall = refined.anchors[refined.triangles[:, 0], 3] > .5
        rest = refined.anchors[refined.triangles, :3]
        rest_normal = np.cross(rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0])
        projection = np.abs(rest_normal).argmax(axis=1)
        for is_wall, material in ((False, self.skin), (True, self.interior)):
            for axis, uv_axes in enumerate(((1, 2), (0, 2), (0, 1))):
                selected = (wall == is_wall) & (projection == axis)
                if not np.any(selected):
                    continue
                ids, inverse = np.unique(refined.triangles[selected], return_inverse=True)
                mesh = trimesh.Trimesh(vertices=to_genesis(refined.vertices[ids]),
                                       faces=inverse.reshape(-1, 3), process=False)
                mesh.vertex_normals = refined.normals[ids] @ TO_GENESIS.T
                # Dominant rest-space projection prevents stretched textures on
                # vertical sides; the selected plane is stable during deformation.
                uv = refined.anchors[ids][:, uv_axes] / FAT_TEXTURE_CM
                mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)
                self.dynamic_nodes.append(self.add_mesh(mesh, material))

        h = obs.hinge
        axis = np.array(self.settings.jaw_axis)
        normal = np.array(self.settings.jaw_normal)
        geometry = self.grasper_geometry
        wrist_length = 1.82 * self.settings.jaw_width / .52 if self.wrist else 0
        self.dynamic_nodes.append(self.add_mesh(cylinder_between(h - geometry.shaft_length * axis,
                         h - wrist_length * axis, geometry.shaft_radius), self.shaft))
        jaw_frame = np.stack((axis, normal, np.cross(axis, normal)))
        for source in self.wrist:
            vertices = h + (source.vertices * (self.settings.jaw_width / .52)) @ jaw_frame
            mesh = trimesh.Trimesh(to_genesis(vertices), source.faces, process=False)
            self.dynamic_nodes.append(self.add_mesh(mesh, self.steel))
        for vertices, faces in zip(geometry.posed_jaws(h, axis, normal, obs.jaw_gap), geometry.faces):
            jaw = trimesh.Trimesh(to_genesis(vertices), faces, process=False)
            self.dynamic_nodes.append(self.add_mesh(jaw, self.steel))
        if self.scissors_geometry is not None:
            # A saved state without a scissors pose shows the tissue alone.
            if obs.scissors is None:
                return
            pose = obs.scissors
            geometry = self.scissors_geometry.jaws
            for vertices, faces in zip(geometry.posed_jaws(pose.hinge, pose.axis, pose.normal, pose.gap),
                                       geometry.faces):
                mesh = trimesh.Trimesh(to_genesis(vertices), faces, process=False)
                self.dynamic_nodes.append(self.add_mesh(mesh, self.edge))
            wrist_frame = np.stack((pose.axis, pose.normal, np.cross(pose.axis, pose.normal)))
            for source in self.wrist:
                mesh = trimesh.Trimesh(to_genesis(pose.hinge + source.vertices @ wrist_frame),
                                       source.faces, process=False)
                self.dynamic_nodes.append(self.add_mesh(mesh, self.steel))
            self.dynamic_nodes.append(self.add_mesh(cylinder_between(pose.hinge - 1.82 * pose.axis,
                pose.hinge - geometry.shaft_length * pose.axis, geometry.shaft_radius), self.shaft))
            return
        center = obs.blade[:3]
        half = knife_frame(self.settings.blade_axis)[0] * (self.settings.blade_length / 2)
        a, b = center - half, center + half
        self.dynamic_nodes.append(self.add_mesh(cylinder_between(a, b, self.settings.blade_radius), self.edge))
        knife = self.knife_geometry
        mesh = knife.blade_mesh(center, self.settings.blade_axis)
        mesh.vertices = to_genesis(mesh.vertices)
        self.dynamic_nodes.append(self.add_mesh(mesh, self.edge))
        handle_a, handle_b = knife.handle_points(center, self.settings.blade_axis)
        direction = (handle_a - handle_b) / np.linalg.norm(handle_a - handle_b)
        opening = np.array((0, 1, 0)) - direction * direction[1]
        opening /= np.linalg.norm(opening)
        wrist_frame = np.stack((direction, opening, np.cross(direction, opening)))
        for source in self.wrist:
            vertices = handle_a + source.vertices @ wrist_frame
            mesh = trimesh.Trimesh(to_genesis(vertices), source.faces, process=False)
            self.dynamic_nodes.append(self.add_mesh(mesh, self.steel))
        shaft_start = handle_a - 1.82 * direction if self.wrist else handle_a
        self.dynamic_nodes.append(self.add_mesh(cylinder_between(shaft_start, handle_b,
                                                                 knife.handle_radius), self.shaft))
        if self.wrist:
            mount = trimesh.creation.box(extents=(.22, .18, .05))
            mount.vertices = to_genesis((mount.vertices + (-knife.boxes[0, 3] + .12, .52, 0))
                                       @ knife_frame(self.settings.blade_axis) + center)
            self.dynamic_nodes.append(self.add_mesh(mount, self.cartridge))

    def capture(self, stage, sim_time, cut_edges, grip_points):
        rgb, depth, _, _ = self.camera.render(rgb=True, depth=True, force_render=True)
        image = Image.fromarray(rgb[..., :3]).resize((self.width, self.height), Image.Resampling.LANCZOS)
        yy, xx = np.mgrid[:self.height, :self.width]
        radius = np.sqrt(((xx - self.width / 2) / (self.height * .68)) ** 2
                         + ((yy - self.height / 2) / (self.height * .68)) ** 2)
        vignette = np.clip((1.05 - radius) / .2, 0, 1) * np.clip(1 - .18 * radius ** 2, 0, 1)
        frame = Image.fromarray((np.array(image) * vignette[..., None]).astype(np.uint8))
        # Depth is sampled, not color-filtered, and remains in metres.
        return frame, depth[::2, ::2]

    def close(self):
        gs.destroy()
