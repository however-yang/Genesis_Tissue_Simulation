"""Lit Genesis surface adapter for independently simulated, changing topology.

The external-node API is isolated here because Genesis debug meshes are unlit.
Contact stays in CUDA. Visual tessellation never reconnects cut components.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np

import cv2
from PIL import Image
import trimesh

import genesis as gs
from genesis.ext import pyrender

from .grid import METERS_PER_UNIT, TO_GENESIS, to_genesis
from .instrument_geometry import knife_frame, knife_geometry, load_davinci_wrist, load_grasper, load_scissors
from .native_backend import Observation, Settings, Surface
from .visual_surface import refine_surface


# One repeat of the fat texture spans this many cm of rest tissue.
FAT_TEXTURE_CM = 1.5
# Texture axes of a triangle, by the rest axis its rest normal is closest to.
UV_AXES = np.array(((1, 2), (0, 2), (0, 1)))
# Irradiance of the endoscope lamp at the point the camera looks at, in renderer units. The lamp sits on the lens, so
# tissue darkens with the square of its distance; its output follows the squared viewing distance, as the automatic
# light control of an endoscope keeps a close-up exposed like a wide shot.
LAMP_IRRADIANCE = 2.6
# Barrel distortion of the endoscope optics: the image at radius r, in units of the field stop radius, shows the scene
# at r (1 + BARREL r^2) / (1 + BARREL), which magnifies the centre by 1 + BARREL and keeps the edge of the view.
BARREL = .08


def cylinder_between(a, b, radius):
    """Return a cylinder from `a` to `b` with the given radius, all in cm, as a mesh in metres along the same axes."""
    return trimesh.creation.cylinder(radius=radius * METERS_PER_UNIT, segment=np.array((a, b)) * METERS_PER_UNIT,
                                     sections=32)


def part_pose(origin, frame):
    """Return the Genesis pose of a rigid part whose mesh is given in metres in `frame`, about `origin` in native cm.

    The rows of `frame` are the part's axes in native coordinates, a proper rotation.
    """
    pose = np.eye(4)
    pose[:3, :3] = TO_GENESIS @ np.transpose(frame)
    pose[:3, 3] = to_genesis(np.reshape(origin, (1, 3)))[0]
    return pose


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
        # Wet tissue: a smooth film over the fat sharpens the reflections of the lamp into glints.
        skin = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=albedo, baseColorFactor=(1, .98, .94, 1),
            normalTexture=normal_map, metallicFactor=0, roughnessFactor=.2,
            doubleSided=True, alphaMode="OPAQUE")
        # Fresh cut faces read as bleeding fat, darker and redder than the
        # intact surface, so a slit whose walls still touch remains visible.
        interior = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=albedo, baseColorFactor=(.6, .13, .11, 1),
            normalTexture=normal_map, metallicFactor=0, roughnessFactor=.15,
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
    """The endoscope camera on one Genesis scene: the tissue surface rebuilt per update, the instruments posed.

    With `supersample` 2 the camera renders twice the output resolution and filters it down, which keeps edges and the
    fat texture clean in recorded video but almost doubles the time per frame. With 1, for a live window, the
    renderer's own multisampling still smooths the edges, which then show a slight staircase.
    """

    def __init__(self, settings: Settings, width=1280, height=960, viewer=False, supersample=2):
        if supersample not in (1, 2):
            raise ValueError("supersample must be 1 or 2")
        self.settings = settings
        self.width, self.height = width, height
        self.supersample = supersample
        # Lanczos-3 weights over the 12 samples around each output pixel of a 2x supersampled render.
        offsets = (np.arange(12) - 5.5) / 2
        lanczos = np.sinc(offsets) * np.sinc(offsets / 3)
        self.lanczos = lanczos / lanczos.sum()
        self.tissue_meshes = []
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
        # The endoscope lamp lights the scene (see render); the faint fill keeps unlit tissue off pure black. Shadow
        # maps would cost every frame for nothing visible: the shadows of a lamp on the lens hide behind what casts
        # them, and the fill is too faint to cast any.
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=settings.dt, gravity=(0, 0, 0)),
            viewer_options=gs.options.ViewerOptions(camera_pos=camera, camera_lookat=target,
                camera_fov=43, res=(width, height), realtime_factor=None),
            vis_options=gs.options.VisOptions(show_world_frame=False, shadow=False,
                background_color=(.006, .002, .003), ambient_light=(.012, .010, .010),
                lights=[gs.options.vis.DirectionalLight(dir=(.7, .3, -.7),
                            color=(.85, .91, 1), intensity=.10),
                        gs.options.vis.DirectionalLight(dir=(-.5, -.2, -.7),
                            color=(1, .94, .87), intensity=.05)]),
            show_viewer=viewer)
        self.scene.add_entity(gs.morphs.Box(size=(0.16, 0.13, 0.01), pos=(0, 0, -0.018),
                                           fixed=True, collision=False),
                              surface=gs.surfaces.Plastic(color=(0.16, 0.035, 0.045), roughness=.55))
        self.camera = self.scene.add_camera(res=(width * supersample, height * supersample), pos=camera,
                                            lookat=target, fov=43, near=0.001, far=0.5, debug=True, GUI=False)
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
        self.lamp = self.scene.visualizer.context.add_node(pyrender.PointLight(color=(1, .97, .93), intensity=0))
        # Pixel offsets from the image centre, and their radius in units of 0.68 image heights: the vignette closes
        # the field stop at radius 1.05.
        offset = np.stack(np.meshgrid(np.arange(width) - (width - 1) / 2, np.arange(height) - (height - 1) / 2), axis=2)
        radius = np.linalg.norm(offset, axis=2) / (.68 * height)
        distortion = ((1 + BARREL * (radius / 1.05) ** 2) / (1 + BARREL))[..., None]
        # The map samples the renderer's buffer, which stores the image bottom-up, so the vertical offset flips.
        # OpenCV remaps fastest through a fixed-point map, built from a float32 one.
        self.lens_map = cv2.convertMaps((((width - 1) / 2, (height - 1) / 2) + offset * distortion * (1, -1))
                                        .astype(np.float32), None, cv2.CV_16SC2)
        vignette = np.clip((1.05 - radius) / .2, 0, 1) * np.clip(1 - .18 * radius ** 2, 0, 1)
        # OpenCV scales an image by a float32 array of its own shape.
        self.vignette = np.repeat(vignette[..., None], 3, axis=2).astype(np.float32)
        # The instruments are rigid parts, built once in their own frame and placed by their node poses (part_pose).
        axis, normal = np.array(settings.jaw_axis), np.array(settings.jaw_normal)
        self.jaw_frame = np.stack((axis, normal, np.cross(axis, normal)))
        geometry = self.grasper_geometry
        wrist_scale = settings.jaw_width / .52
        self.grasper_body = [self.add_part(cylinder_between((-geometry.shaft_length, 0, 0),
                                                            (-1.82 * wrist_scale if self.wrist else 0, 0, 0),
                                                            geometry.shaft_radius), self.shaft)]
        self.grasper_body += [self.add_part(trimesh.Trimesh(source.vertices * wrist_scale * METERS_PER_UNIT,
                                                            source.faces, process=False), self.steel)
                              for source in self.wrist]
        self.grasper_jaws = [self.add_part(trimesh.Trimesh(vertices * METERS_PER_UNIT, faces, process=False),
                                           self.steel) for vertices, faces in zip(geometry.vertices, geometry.faces)]
        # Posed unit vertices are the jaw frames, from the same arithmetic as the contact pads.
        self.jaw_frames = replace(geometry, vertices=(np.eye(3),) * 2)
        if self.scissors_geometry is not None:
            jaws = self.scissors_geometry.jaws
            self.scissors_blades = [self.add_part(trimesh.Trimesh(vertices * METERS_PER_UNIT, faces, process=False),
                                                  self.edge) for vertices, faces in zip(jaws.vertices, jaws.faces)]
            self.scissors_body = [self.add_part(trimesh.Trimesh(source.vertices * METERS_PER_UNIT, source.faces,
                                                                process=False), self.steel) for source in self.wrist]
            self.scissors_body.append(self.add_part(cylinder_between((-1.82, 0, 0), (-jaws.shaft_length, 0, 0),
                                                                     jaws.shaft_radius), self.shaft))
            self.blade_frames = replace(jaws, vertices=(np.eye(3),) * 2)
        else:
            # The knife parts keep their native axes and move with the blade centre.
            knife = self.knife_geometry
            half = knife_frame(settings.blade_axis)[0] * (settings.blade_length / 2)
            parts = [(cylinder_between(-half, half, settings.blade_radius), self.edge)]
            blade = knife.blade_mesh(np.zeros(3), settings.blade_axis)
            blade.vertices = blade.vertices * METERS_PER_UNIT
            parts.append((blade, self.edge))
            handle_a, handle_b = knife.handle_points(np.zeros(3), settings.blade_axis)
            direction = (handle_a - handle_b) / np.linalg.norm(handle_a - handle_b)
            opening = np.array((0, 1, 0)) - direction * direction[1]
            opening /= np.linalg.norm(opening)
            wrist_frame = np.stack((direction, opening, np.cross(direction, opening)))
            parts += [(trimesh.Trimesh((handle_a + source.vertices @ wrist_frame) * METERS_PER_UNIT, source.faces,
                                       process=False), self.steel) for source in self.wrist]
            shaft_start = handle_a - 1.82 * direction if self.wrist else handle_a
            parts.append((cylinder_between(shaft_start, handle_b, knife.handle_radius), self.shaft))
            if self.wrist:
                mount = trimesh.creation.box(extents=(.22, .18, .05))
                mount.vertices = ((mount.vertices + (-knife.boxes[0, 3] + .12, .52, 0))
                                  @ knife_frame(settings.blade_axis) * METERS_PER_UNIT)
                parts.append((mount, self.cartridge))
            self.knife_parts = [self.add_part(mesh, material) for mesh, material in parts]

    def add_mesh(self, mesh, material):
        node = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True,
                                         name="endo_" + str(gs.UID()), is_marker=False,
                                         poses=np.eye(4)[None])
        with self.scene.visualizer.viewer_lock:
            self.scene.visualizer.context.add_external_node(node)
        return node

    def add_part(self, mesh, material):
        """Add a rigid instrument part and return its scene node, which update() poses."""
        return self.scene.visualizer.context.external_nodes[self.add_mesh(mesh, material).name]

    def place(self, nodes, origin, frame):
        pose = part_pose(origin, frame)
        for node in nodes:
            self.scene.visualizer.context.set_node_pose(node, pose)

    def update(self, surface: Surface, obs: Observation, *, divisions=1):
        for mesh in self.tissue_meshes:
            self.scene.clear_debug_object(mesh)
        self.tissue_meshes.clear()
        refined = refine_surface(surface, divisions=divisions) if divisions > 1 else surface
        triangles = refined.triangles
        wall = refined.anchors[triangles[:, 0], 3] > .5
        rest = refined.anchors[triangles, :3]
        # Dominant rest-space projection prevents stretched textures on
        # vertical sides; the selected plane is stable during deformation.
        projection = np.abs(np.cross(rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0])).argmax(axis=1)
        positions = to_genesis(refined.vertices)
        normals = refined.normals @ TO_GENESIS.T
        for is_wall, material in ((False, self.skin), (True, self.interior)):
            selected = wall == is_wall
            if not np.any(selected):
                continue
            # One vertex per surface vertex and projection axis, since the texture coordinates of a vertex depend on
            # the axis of its triangle.
            keys = triangles[selected] * 3 + projection[selected, None]
            is_used = np.zeros(3 * len(refined.vertices), dtype=bool)
            is_used[keys] = True
            used = np.flatnonzero(is_used)
            ids = used // 3
            # Static vertex data keeps the surface normals of the reconstruction, which round the cut lips and the
            # edges where the lamp glints. Streamed positions would make the renderer replace them by the normals of
            # the flat triangles.
            primitive = pyrender.Primitive(
                positions=positions[ids], normals=normals[ids],
                texcoord_0=np.take_along_axis(refined.anchors[ids], UV_AXES[used % 3], axis=1) / FAT_TEXTURE_CM,
                indices=(np.cumsum(is_used) - 1)[keys], material=material, poses=np.eye(4)[None],
                stream_positions=False)
            mesh = pyrender.Mesh([primitive], name="endo_" + str(gs.UID()))
            with self.scene.visualizer.viewer_lock:
                self.scene.visualizer.context.add_external_node(mesh)
            self.tissue_meshes.append(mesh)

        axis, normal = np.array(self.settings.jaw_axis), np.array(self.settings.jaw_normal)
        self.place(self.grasper_body, obs.hinge, self.jaw_frame)
        for node, frame in zip(self.grasper_jaws, self.jaw_frames.posed_jaws(np.zeros(3), axis, normal, obs.jaw_gap)):
            self.place((node,), obs.hinge, frame)
        if self.scissors_geometry is not None:
            pose = obs.scissors
            # A saved state without a scissors pose shows the tissue alone. Visibility takes effect when the scene's
            # meshes change, which every update does by replacing the tissue.
            for node in (*self.scissors_blades, *self.scissors_body):
                node.mesh.is_visible = pose is not None
            if pose is None:
                return
            for node, frame in zip(self.scissors_blades,
                                   self.blade_frames.posed_jaws(np.zeros(3), pose.axis, pose.normal, pose.gap)):
                self.place((node,), pose.hinge, frame)
            wrist_frame = np.stack((pose.axis, pose.normal, np.cross(pose.axis, pose.normal)))
            self.place(self.scissors_body, pose.hinge, wrist_frame)
            return
        self.place(self.knife_parts, obs.blade[:3], np.eye(3))

    def render(self, depth=False):
        """Render the endoscopic image in OpenCV channel order (BGR), and with `depth` its depth map in metres.

        The image passes through the lens distortion (BARREL) and the vignette. The depth map is that of the pinhole
        camera, `camera`, without lens distortion.
        """
        pos, lookat = self.camera.pos, self.camera.lookat
        lamp_pose = np.eye(4)
        lamp_pose[:3, 3] = pos
        self.scene.visualizer.context.set_node_pose(self.lamp, lamp_pose)
        self.lamp.light.intensity = LAMP_IRRADIANCE * np.sum((lookat - pos) ** 2)
        rgb, depth_m, _, _ = self.camera.render(rgb=True, depth=depth, force_render=True)
        # The image is an upright view of the bottom-up GL buffer. The buffer itself needs no copy for OpenCV, and
        # the lens map samples it upside down.
        buffer = rgb[::-1, :, :3]
        if self.supersample > 1:
            # A box average of the samples would blur the fat texture visibly.
            buffer = cv2.sepFilter2D(buffer, cv2.CV_8U, self.lanczos, self.lanczos, anchor=(5, 5),
                                     borderType=cv2.BORDER_REFLECT)[::2, ::2]
            # Depth is sampled, not color-filtered, and remains in metres.
            depth_m = depth_m[::2, ::2] if depth else None
        frame = cv2.cvtColor(cv2.remap(buffer, *self.lens_map, cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR)
        return cv2.multiply(frame, self.vignette, dtype=cv2.CV_8U), depth_m

    def capture(self, stage, sim_time, cut_edges, grip_points):
        """Render the endoscopic image as an RGB PIL image, with its depth map in metres."""
        frame, depth = self.render(depth=True)
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), depth

    def close(self):
        gs.destroy()
