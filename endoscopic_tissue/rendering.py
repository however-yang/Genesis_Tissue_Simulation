"""Lit Genesis surface adapter for independently simulated, changing topology.

The external-node API is isolated here because Genesis debug meshes are unlit.
Contact stays in CUDA. Visual tessellation never reconnects cut components.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np

import cv2
from PIL import Image
from scipy import ndimage
import trimesh

import genesis as gs
from genesis.ext import pyrender

from .grid import METERS_PER_UNIT, TO_GENESIS, to_genesis
from .instrument_geometry import knife_frame, knife_geometry, load_davinci_wrist, load_grasper, load_scissors
from .kidney import Kidney
from .liver import (LIVER_PACKING, SCOPE_WALL, TORSO_CENTER_X, TORSO_RADIUS_X, TORSO_RADIUS_Y, Liver, cavity_mesh,
                    diaphragm_mesh)
from .native_backend import Observation, Settings, Surface
from .visual_surface import refine_surface


# One repeat of the fat texture spans this many cm of rest tissue.
FAT_TEXTURE_CM = 1.5
# One repeat of the kidney's detail textures spans this many cm of rest tissue.
KIDNEY_TEXTURE_CM = 2.4
# Organs around the block under the liver texture, as ellipsoids (native centre, half extents) in cm.
LIVER_BACKDROP = (((0, -3, 0), (10, 2.1, 10)), ((-6.0, -0.8, 0.4), (1.8, 1.2, 3.4)), ((6.0, -1.2, 1.0), (2.1, 1.3, 3.8)),
                  ((0.0, -1.0, 5.5), (4.3, 1.0, 1.6)), ((0.0, .2, -6.0), (8.0, 3.5, 2.8)))
# Perirenal fat around the kidney: the bed that hides its pinned lower part, lobules on either side and behind, and
# the fat over the posterior abdominal wall far behind. Its lobules are larger than those of the block's fat: one
# repeat of the fat texture spans BACKDROP_FAT_CM.
FAT_BACKDROP = (((0, -2.6, 0), (8.5, 3.0, 6.5)), ((-6.8, .2, 1.0), (2.2, 1.6, 2.6)), ((6.6, 0, .8), (2.0, 1.5, 2.8)),
                ((0, .8, -4.4), (7.0, 2.2, 2.2)), ((0, 2.0, -15.0), (18.0, 12.0, 4.0)))
BACKDROP_FAT_CM = 4.0
# Default endoscope (position, look-at) in Genesis metres, framing the block.
BLOCK_CAMERA = ((0, -0.098, 0.070), (0, 0, 0.003))
# Texture axes of a triangle, by the rest axis its rest normal is closest to.
UV_AXES = np.array(((1, 2), (0, 2), (0, 1)))
# Irradiance of the endoscope lamp at the point the camera looks at, in renderer units. The lamp sits on the lens, so
# tissue darkens with the square of its distance; its output follows the squared viewing distance, as the automatic
# light control of an endoscope keeps a close-up exposed like a wide shot.
LAMP_IRRADIANCE = 2.6
# The liver albedo is dark and the lobe curves away from the lens. This multiple of the endoscope lamp, with the warm
# fill below, keeps the lit face and the shaded face readable.
LIVER_LAMP_SCALE = 4.0
LIVER_AMBIENT = (.10, .04, .035)
PERITONEUM_TEXTURE = Path(__file__).parent / "assets/textures/peritoneum.jpg"
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


def tileable_noise(rng, size, scale):
    """Return smooth noise of unit variance on a square of `size` pixels that tiles, its features `scale` pixels across."""
    frequency = np.hypot(*np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij"))
    noise = np.fft.ifft2(np.fft.fft2(rng.normal(size=(size, size))) * np.exp(-(frequency * scale) ** 2)).real
    return noise / noise.std()


def vessel_tree(rng, size, roots, width):
    """Return the coverage, in [0, 1], of branching vessels on a square of `size` pixels that tiles.

    Each of `roots` trunks starts `width` pixels wide at a random point and heading, wanders and sprouts branches 0.72
    times as wide until they thin out below half a pixel.
    """
    # Drawn on the middle tile of a 3 x 3 canvas, then folded back onto one tile: vessels leaving it wrap around.
    canvas = np.zeros((3 * size, 3 * size), np.uint8)
    branches = [(rng.uniform(0, size, 2) + size, rng.uniform(0, 2 * np.pi), width, .6 * size) for _ in range(roots)]
    while branches:
        point, heading, branch_width, length = branches.pop()
        if branch_width < .5:
            continue
        for _ in range(int(length / 4)):
            heading += rng.normal(0, .08)
            step = 4 * np.array((np.cos(heading), np.sin(heading)))
            start = (point - size) % size + size
            # Sub-pixel endpoints in 1/16 pixel units.
            cv2.line(canvas, tuple(np.round(start * 16).astype(int)), tuple(np.round((start + step) * 16).astype(int)),
                     int(min(255, 120 + 80 * branch_width)), max(1, round(branch_width)), cv2.LINE_AA, shift=4)
            point = point + step
            if rng.random() < .02:
                branches.append((point, heading + rng.choice((-1, 1)) * rng.uniform(.5, 1.1), .72 * branch_width,
                                 .6 * length))
    return np.clip(canvas.reshape(3, size, 3, size).sum(axis=(0, 2)) / 255, 0, 1)


def wave_noise(points, scale, seed):
    """Return smooth noise of unit variance at `points`, shape (n, 3), its features about `scale` cm across."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(24, 3))
    directions *= 2 * np.pi / scale * rng.uniform(.7, 1.3, (24, 1)) / np.linalg.norm(directions, axis=1, keepdims=True)
    return np.sin(points @ directions.T + rng.uniform(0, 2 * np.pi, 24)).sum(axis=1) / np.sqrt(12)


def detail_material(albedo, roughness):
    """Return a wet tissue material of the RGB `albedo` array in [0, 1], its relief derived from its brightness."""
    dy, dx = np.gradient(albedo.mean(axis=2))
    normal = np.stack((-dx * 5, dy * 5, np.ones_like(dx)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    # RGBA rows remain four-byte aligned for arbitrary source widths in the OpenGL uploader.
    normal_map = pyrender.Texture(source=Image.fromarray(((normal * .5 + .5) * 255).astype(np.uint8)).convert("RGBA"),
                                  source_channels="RGBA")
    return pyrender.MetallicRoughnessMaterial(
        baseColorTexture=Image.fromarray((np.clip(albedo, 0, 1) * 255).astype(np.uint8)), normalTexture=normal_map,
        metallicFactor=0, roughnessFactor=roughness, doubleSided=True, alphaMode="OPAQUE")


class KidneyColors:
    """Vertex colours of the kidney and its tumor by rest position, on the surface and on cut faces.

    The capsule is maroon and the tumor golden yellow, with orange-red borders between its lobules and darker
    haemorrhagic patches, the two meeting across 1 mm at the surface of the tumor's ellipsoid. Cut faces show the
    bleeding cortex and the tumor's variegated yellow interior. Near the tumor the colours come from volumes of 0.4 mm cells
    computed once and interpolated per vertex. Elsewhere the surface takes the plain capsule colour: the scene cuts
    only near the tumor.
    """
    CAPSULE = np.array((.50, .16, .13))

    def __init__(self, kidney: Kidney):
        self.cell = .04
        self.low = kidney.tumor_center - (1.4, 1.1, 1.4)
        self.dims = np.round(np.array((2.8, 1.9, 2.8)) / self.cell).astype(int) + 1
        axes = [self.low[k] + np.arange(self.dims[k]) * self.cell for k in range(3)]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        tumor = kidney.tumor_weight(points, .1)[:, None]
        lobules = np.exp(-(wave_noise(points, .45, seed=1) / .3) ** 2)[:, None]
        haemorrhage = np.clip(wave_noise(points, .6, seed=2) - .9, 0, .6)[:, None] / .6
        mottling = 1 + .06 * wave_noise(points, 1.0, seed=3)[:, None]
        tumor_skin = np.array((.90, .62, .25)) * (1 - lobules) + np.array((.72, .32, .15)) * lobules
        tumor_skin = tumor_skin * (1 - .7 * haemorrhage) + np.array((.42, .12, .08)) * .7 * haemorrhage
        skin = self.CAPSULE * mottling * (1 - tumor) + tumor_skin * tumor
        necrosis = np.clip(wave_noise(points, .5, seed=4) - 1.0, 0, .5)[:, None] / .5
        tumor_cut = np.array((.92, .66, .28)) * (1 - necrosis) + np.array((.96, .86, .62)) * necrosis
        tumor_cut = tumor_cut * (1 - .6 * haemorrhage) + np.array((.55, .14, .10)) * .6 * haemorrhage
        cortex = np.array((.60, .17, .14)) * mottling
        self.skin = skin.reshape(*self.dims, 3)
        self.interior = (cortex * (1 - tumor) + tumor_cut * tumor).reshape(*self.dims, 3)

    def __call__(self, rest, is_wall):
        """Return the RGB colours of vertices at `rest` positions, on cut faces with `is_wall`."""
        coords = (rest - self.low) / self.cell
        colors = np.tile(self.CAPSULE, (len(rest), 1))
        near = np.all((coords >= 0) & (coords <= self.dims - 1), axis=1) | is_wall
        volume = self.interior if is_wall else self.skin
        colors[near] = np.stack([ndimage.map_coordinates(volume[..., c], coords[near].T, order=1, mode="nearest")
                                 for c in range(3)], axis=1)
        return colors


def tissue_materials(kind="fat"):
    directory = Path(__file__).parent / "assets/textures"
    if kind == "kidney":
        # Colours come per vertex (KidneyColors); the textures add the fine capsular vessels and the grain of cut
        # parenchyma. Both are generated, so they keep the repeat KIDNEY_TEXTURE_CM at any resolution.
        rng = np.random.default_rng(11)
        size = 1024
        vessels = np.maximum(vessel_tree(rng, size, 2, 4.0), .5 * vessel_tree(rng, size, 5, 1.8))
        capsule = (.92 + .035 * tileable_noise(rng, size, 3))[..., None] * (1 - vessels[..., None] * (.12, .32, .30))
        speckles = np.clip(tileable_noise(rng, size, 6) - 2.2, 0, .6) / .6
        cut = (.88 + .08 * tileable_noise(rng, size, 5))[..., None] * (1 - speckles[..., None] * (.3, .7, .7))
        # The wet capsule glints; fresh cut faces are wetter still.
        return detail_material(capsule / capsule.max(), .28), detail_material(cut / cut.max(), .18)
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

    The tissue is the fat block among liver-textured organs, with `kidney` that kidney and its tumor in perirenal fat,
    or with `liver` the posed liver inside the peritoneal cavity. `camera` is the endoscope's (position, look-at) in
    Genesis metres, by default BLOCK_CAMERA. `fov` is the vertical field of view in degrees.
    """

    def __init__(self, settings: Settings, width=1280, height=960, viewer=False, supersample=2,
                 kidney: Kidney | None = None, camera=BLOCK_CAMERA, liver: Liver | None = None, fov=43):
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
        self.lamp_irradiance = LAMP_IRRADIANCE
        self.lamp_pos = None
        self.overhead = None
        ambient = (.012, .010, .010)
        if liver is not None:
            self.skin, self.interior = tissue_materials("liver")
            self.colors, self.texture_cm, backdrop, floor = None, FAT_TEXTURE_CM, (), None
            self.lamp_irradiance = LAMP_IRRADIANCE * LIVER_LAMP_SCALE
            ambient = LIVER_AMBIENT
            # A second lamp sits directly above the lobe, clear of the cavity, so its light falls straight down.
            # lamp_reach is the squared distance that gives one full lamp's irradiance on the liver centre.
            center = (liver.low + liver.high) / 2
            overhead = center.copy()
            overhead[1] = liver.high[1] + 12.0
            self.lamp_pos = to_genesis(overhead.reshape(1, 3))[0]
            self.lamp_reach = np.sum((self.lamp_pos - to_genesis(center.reshape(1, 3))[0]) ** 2)
        elif kidney is None:
            self.skin, self.interior = tissue_materials()
            backdrop_skin, _ = tissue_materials("liver")
            self.colors, self.texture_cm, backdrop, floor = None, FAT_TEXTURE_CM, LIVER_BACKDROP, -0.018
        else:
            self.skin, self.interior = tissue_materials("kidney")
            backdrop_skin, _ = tissue_materials()
            # The floor sinks below the kidney's pinned base, inside the fat bed.
            self.colors, self.texture_cm, backdrop, floor = KidneyColors(kidney), KIDNEY_TEXTURE_CM, FAT_BACKDROP, -0.045
        self.steel = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(.72, .75, .78, 1), metallicFactor=.92, roughnessFactor=.19)
        self.shaft = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(.028, .032, .038, 1), metallicFactor=.15, roughnessFactor=.25)
        self.cartridge = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(.055, .21, .47, 1), metallicFactor=.05, roughnessFactor=.32)
        self.edge = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(.88, .88, .83, 1), metallicFactor=.9, roughnessFactor=.16)
        gs.init(backend=gs.cpu, logging_level="warning")
        camera, target = camera
        # The endoscope lamp lights the scene (see render). The fill keeps tissue the lamp misses off pure black.
        # Shadow maps would cost every frame for nothing visible: the shadows of a lamp on the lens hide behind
        # what casts them.
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=settings.dt, gravity=(0, 0, 0)),
            viewer_options=gs.options.ViewerOptions(camera_pos=camera, camera_lookat=target,
                camera_fov=fov, res=(width, height), realtime_factor=None),
            vis_options=gs.options.VisOptions(show_world_frame=False, shadow=False,
                background_color=(.006, .002, .003), ambient_light=ambient,
                lights=[gs.options.vis.DirectionalLight(dir=(.7, .3, -.7),
                            color=(.85, .91, 1), intensity=.10),
                        gs.options.vis.DirectionalLight(dir=(-.5, -.2, -.7),
                            color=(1, .94, .87), intensity=.05)]),
            show_viewer=viewer)
        if floor is not None:
            self.scene.add_entity(gs.morphs.Box(size=(0.16, 0.13, 0.01), pos=(0, 0, floor),
                                               fixed=True, collision=False),
                                  surface=gs.surfaces.Plastic(color=(0.16, 0.035, 0.045), roughness=.55))
        self.camera = self.scene.add_camera(res=(width * supersample, height * supersample), pos=camera,
                                            lookat=target, fov=fov, near=0.001, far=0.5, debug=True, GUI=False)
        self.scene.build()
        if liver is not None:
            shell, shell_faces = cavity_mesh()
            # The photograph runs along the body and around the shell, one copy to the full height, with the
            # circumferential repeat keeping its pixel aspect. The branch cut sits on the scope's side, behind the lens.
            theta = np.arctan2(shell[:, 1] / TORSO_RADIUS_Y, (shell[:, 0] - TORSO_CENTER_X) / TORSO_RADIUS_X)
            scope_theta = float(np.arctan2(SCOPE_WALL[1] / TORSO_RADIUS_Y,
                                           (SCOPE_WALL[0] - TORSO_CENTER_X) / TORSO_RADIUS_X))
            u = np.mod(theta - scope_theta, 2 * np.pi) / (2 * np.pi)
            shell_height = float(shell[:, 2].max() - shell[:, 2].min())
            v = (shell[:, 2] - float(shell[:, 2].min())) / shell_height
            corner_u = u[shell_faces]
            wrap = corner_u.max(axis=1) - corner_u.min(axis=1) > 0.5
            low = wrap[:, None] & (corner_u < 0.5)
            seam = np.unique(shell_faces[low])
            vertices = np.vstack((shell, shell[seam]))
            u = np.concatenate((u, u[seam] + 1))
            v = np.concatenate((v, v[seam]))
            remap = np.full(shell.shape[0], -1, dtype=np.int32)
            remap[seam] = np.arange(shell.shape[0], shell.shape[0] + seam.shape[0])
            faces = np.where(low, remap[shell_faces], shell_faces)
            segments = int(np.count_nonzero(np.abs(shell[:, 2] - shell[0, 2]) <= 1e-4))
            ring = np.vstack((shell[:segments, :2], shell[:1, :2]))
            circumference = float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum())
            image = Image.open(PERITONEUM_TEXTURE).convert("RGB")
            aspect = image.size[0] / image.size[1]
            u = u * (circumference / (aspect * shell_height))
            cavity = trimesh.Trimesh(to_genesis(vertices), faces, process=False)
            cavity.visual = trimesh.visual.texture.TextureVisuals(uv=np.column_stack((u, v)).astype(np.float32))
            peritoneum = pyrender.MetallicRoughnessMaterial(
                baseColorTexture=pyrender.Texture(
                    source=image, source_channels="RGB",
                    sampler=pyrender.Sampler(wrapS=pyrender.GLTF.MIRRORED_REPEAT, wrapT=pyrender.GLTF.MIRRORED_REPEAT)),
                # Integer components are read as 0-255, so the multiplier is written in floats.
                baseColorFactor=(1.0, 1.0, 1.0, 1.0), metallicFactor=0, roughnessFactor=.84, doubleSided=True,
                alphaMode="OPAQUE")
            self.add_mesh(cavity, peritoneum)
            wall, wall_faces = diaphragm_mesh(liver)
            # The diaphragm uses the shell's centimetres per repeat, so the photograph has the same size on both.
            wall_uv = np.column_stack((0.5 + (wall[:, 0] - TORSO_CENTER_X) / (aspect * shell_height),
                                       0.5 + wall[:, 1] / shell_height))
            disk = trimesh.Trimesh(to_genesis(wall), wall_faces, process=False)
            disk.visual = trimesh.visual.texture.TextureVisuals(uv=wall_uv.astype(np.float32))
            self.add_mesh(disk, peritoneum)
            # Packing is drawn only, so the grasp and the cuts are unchanged. It wears the liver surface.
            packing_skin, _ = tissue_materials("liver")
            for centre, scale in LIVER_PACKING:
                mesh = trimesh.creation.icosphere(subdivisions=4)
                directions = mesh.vertices
                native = directions * scale + centre
                native += directions * (0.12 * wave_noise(native, 3.5, seed=8))[:, None]
                mesh.vertices = to_genesis(native)
                uv = np.column_stack((native[:, 0], native[:, 2])) / FAT_TEXTURE_CM
                mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv.astype(np.float32))
                self.add_mesh(mesh, packing_skin)
        for center, scale in backdrop:
            mesh = trimesh.creation.icosphere(subdivisions=4 if kidney is None else 5)
            directions = mesh.vertices
            native = directions * scale + center
            if kidney is not None:
                # Fat bulges in lobules a few centimetres across, with smaller lumps on them.
                normals = directions / scale
                normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                lumps = .3 * wave_noise(native, 4.0, seed=5) + .08 * wave_noise(native, 1.6, seed=6)
                native += normals * lumps[:, None]
            mesh.vertices = to_genesis(native)
            # Four repeats of the liver texture around each organ; the fat keeps its scale in cm.
            repeats = 4 if kidney is None else 2 * np.pi * np.mean(scale) / BACKDROP_FAT_CM
            uv = np.column_stack((np.arctan2(directions[:, 2], directions[:, 0]) / (2 * np.pi),
                                  np.arcsin(directions[:, 1]) / np.pi)) * repeats
            mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)
            self.add_mesh(mesh, backdrop_skin)
        self.lamp = self.scene.visualizer.context.add_node(pyrender.PointLight(color=(1, .97, .93), intensity=0))
        if self.lamp_pos is not None:
            self.overhead = self.scene.visualizer.context.add_node(
                pyrender.PointLight(color=(1, .97, .93), intensity=0))
            overhead_pose = np.eye(4)
            overhead_pose[:3, 3] = self.lamp_pos
            self.scene.visualizer.context.set_node_pose(self.overhead, overhead_pose)
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
                texcoord_0=np.take_along_axis(refined.anchors[ids], UV_AXES[used % 3], axis=1) / self.texture_cm,
                color_0=None if self.colors is None else self.colors(refined.anchors[ids, :3], is_wall),
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
        self.lamp.light.intensity = self.lamp_irradiance * np.sum((lookat - pos) ** 2)
        # The liver keeps this lens lamp and the overhead lamp. Each is half of one full lamp.
        if self.overhead is not None:
            self.lamp.light.intensity *= 0.5
            self.overhead.light.intensity = 0.5 * self.lamp_irradiance * self.lamp_reach
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
