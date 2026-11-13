"""Surroundings drawn around a tissue: the cavity, the pads, and the light."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from PIL import Image
import trimesh

from genesis.ext import pyrender

from .grid import from_genesis, to_genesis
from .rendering import FAT_TEXTURE_CM, tissue_materials, wave_noise
from .tissue import TissueModel


PERITONEUM_TEXTURE = Path(__file__).parent / "assets" / "textures" / "peritoneum.jpg"


@runtime_checkable
class SurgicalEnvironment(Protocol):
    """Surroundings drawn around a tissue. The tissue itself is not part of them.

    `fov` is the vertical field of view in degrees. `ambient` is the RGB fill. `lamp_scale` multiplies the endoscope
    lamp before the view splits it. `overhead` returns the Genesis position of a fixed lamp and the squared distance
    that puts one full lamp on the tissue centre, or None when that lamp is absent. When it returns a lamp, the view
    gives half of the scaled output to the lens and half to the fixed lamp. `populate` adds the cavity and the other
    surroundings after the scene exists. The view provides `add_mesh` and `shot`, the fixed camera in Genesis metres.
    """

    fov: float
    ambient: tuple[float, float, float]
    lamp_scale: float

    def overhead(self, tissue: TissueModel):
        """Return the fixed lamp and its squared reach, or none when the lamp stays on the lens."""

    def populate(self, view, tissue: TissueModel):
        """Add the cavity, the pads, and the other surroundings to the view."""


@dataclass(frozen=True)
class Pad:
    """One drawn pad around the tissue, in centimetres. It does not enter the solver."""

    centre: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    is_cavity_coloured: bool


@dataclass(frozen=True)
class PeritonealCavity:
    """Supine peritoneal cavity: an elliptical shell, a diaphragm, packed tissue, and two lamps.

    The shell is open toward +Z. Pads are drawn only. The grasp and the cuts ignore them.
    """

    # Narrower than the block view, so the lobe fills more of the picture. The instrument tips stay in frame.
    fov: float = 56.0
    # Warm fill so the side of the lobe that faces away from both lamps stays readable.
    ambient: tuple[float, float, float] = (0.10, 0.04, 0.035)
    # The liver albedo is dark and the lobe curves away from the lens. The view then splits this across the two lamps.
    lamp_scale: float = 4.0
    # The fixed lamp sits this far above the tissue, in cm, clear of the shell.
    overhead_clearance: float = 12.0
    # The lobe already meets the lower-left wall, so the section keeps that side and draws in on the instrument side.
    center_x: float = -2.0
    radius_x: float = 16.0
    radius_y: float = 9.0
    z_min: float = -12.0
    z_max: float = 18.0
    # The diaphragm stands this far, in cm, past the surface point furthest along the sight line.
    diaphragm_inset: float = 3.0
    # Azimuth of the wall-texture join, on the lower wall under the liver so the lobe and the far opening cover it.
    scope_wall: tuple[float, float, float] = (-5.0, -9.0, 0.0)
    # The two upper pads overlap into one band. The lower pad reaches toward the scope and stays clear of the lobe.
    # A cavity-coloured pad wears the wall photograph. The others wear the liver surface.
    pads: tuple[Pad, ...] = (
        Pad(centre=(-5.5, -4.15, 11.5), half_extents=(8.0, 2.5, 6.2), is_cavity_coloured=True),
        Pad(centre=(-4.0, 6.9, 8.2), half_extents=(7.2, 1.05, 4.6), is_cavity_coloured=False),
        Pad(centre=(3.0, 6.7, 9.2), half_extents=(4.4, 1.05, 4.0), is_cavity_coloured=False),
        Pad(centre=(4.2, -3.6, 8.0), half_extents=(2.4, 1.8, 3.0), is_cavity_coloured=True),
    )

    def overhead(self, tissue):
        """Return the lamp above the tissue and the squared reach of one full lamp at its centre."""
        low = np.asarray(tissue.low)
        high = np.asarray(tissue.high)
        centre = (low + high) / 2
        position = centre.copy()
        position[1] = high[1] + self.overhead_clearance
        lamp = to_genesis(position.reshape(1, 3))[0]
        origin = to_genesis(centre.reshape(1, 3))[0]
        return lamp, np.sum((lamp - origin) ** 2)

    def cavity_mesh(self, segments=36, slices=24):
        """Return the supine torso shell, vertices in cm and triangle faces, with normals toward the cavity."""
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        heights = np.linspace(self.z_min, self.z_max, slices)
        circle = np.stack((np.cos(angles) * self.radius_x + self.center_x, np.sin(angles) * self.radius_y), axis=1)
        vertices = np.column_stack((np.tile(circle, (slices, 1)), np.repeat(heights, segments))).astype(np.float32)
        faces = []
        for slice_index in range(slices - 1):
            row = slice_index * segments
            nxt = row + segments
            for segment in range(segments):
                following = (segment + 1) % segments
                faces.append((row + segment, row + following, nxt + segment))
                faces.append((row + following, nxt + following, nxt + segment))
        # The increasing-angle order points outward. The opposite order points into the cavity.
        return vertices, np.array(faces, dtype=np.int32)[:, ::-1]

    def diaphragm_mesh(self, tissue, position, look, segments=36):
        """Return the elliptical diaphragm that closes the cavity beyond the tissue.

        The wall stands `diaphragm_inset` centimetres past the surface point furthest along the sight line, on the
        side facing away from the scope. Its normal points toward the scope.
        """
        direction = np.asarray(look) - np.asarray(position)
        direction /= np.linalg.norm(direction)
        surface = np.asarray(tissue.surface_points())
        far = surface[np.argmax((surface - position) @ direction)]
        # Inward is native -Z, so the wall is the vertical section of the cavity there.
        wall_z = far[2] - self.diaphragm_inset
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        rim = np.column_stack((np.cos(angles) * self.radius_x + self.center_x, np.sin(angles) * self.radius_y,
                               np.full(segments, wall_z)))
        vertices = np.vstack((np.array((self.center_x, 0.0, wall_z)), rim))
        triangles = [(0, index, index + 1) for index in range(1, segments)]
        triangles.append((0, segments, 1))
        return vertices.astype(np.float32), np.array(triangles, dtype=np.int32)

    def populate(self, view, tissue):
        """Add the shell, the diaphragm, and the pads to the view."""
        shell, shell_faces = self.cavity_mesh()
        # The photograph runs along the body and around the shell, one copy to the full height, with the
        # circumferential repeat keeping its pixel aspect. The branch cut sits on the scope's side, behind the lens.
        theta = np.arctan2(shell[:, 1] / self.radius_y, (shell[:, 0] - self.center_x) / self.radius_x)
        scope_theta = float(np.arctan2(
            self.scope_wall[1] / self.radius_y, (self.scope_wall[0] - self.center_x) / self.radius_x))
        u = np.mod(theta - scope_theta, 2 * np.pi) / (2 * np.pi)
        shell_height = float(shell[:, 2].max() - shell[:, 2].min())
        v = (shell[:, 2] - float(shell[:, 2].min())) / shell_height
        corner_u = u[shell_faces]
        is_wrapping = corner_u.max(axis=1) - corner_u.min(axis=1) > 0.5
        is_seam_corner = is_wrapping[:, None] & (corner_u < 0.5)
        seam = np.unique(shell_faces[is_seam_corner])
        vertices = np.vstack((shell, shell[seam]))
        u = np.concatenate((u, u[seam] + 1))
        v = np.concatenate((v, v[seam]))
        remap = np.full(shell.shape[0], -1, dtype=np.int32)
        remap[seam] = np.arange(shell.shape[0], shell.shape[0] + seam.shape[0])
        faces = np.where(is_seam_corner, remap[shell_faces], shell_faces)
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
                sampler=pyrender.Sampler(wrapS=pyrender.GLTF.MIRRORED_REPEAT,
                                         wrapT=pyrender.GLTF.MIRRORED_REPEAT)),
            # Integer components are read as 0-255, so the multiplier is written in floats.
            baseColorFactor=(1.0, 1.0, 1.0, 1.0), metallicFactor=0, roughnessFactor=.84, doubleSided=True,
            alphaMode="OPAQUE")
        view.add_mesh(cavity, peritoneum)
        position = from_genesis(view.shot[0])
        look = from_genesis(view.shot[1])
        wall, wall_faces = self.diaphragm_mesh(tissue, position, look)
        # The diaphragm uses the shell's centimetres per repeat, so the photograph has the same size on both.
        wall_uv = np.column_stack((0.5 + (wall[:, 0] - self.center_x) / (aspect * shell_height),
                                   0.5 + wall[:, 1] / shell_height))
        disk = trimesh.Trimesh(to_genesis(wall), wall_faces, process=False)
        disk.visual = trimesh.visual.texture.TextureVisuals(uv=wall_uv.astype(np.float32))
        view.add_mesh(disk, peritoneum)
        # Pads under the lobe use the cavity photograph with the wet roughness of the liver surface.
        # Pads above keep the liver surface.
        packing_skin, _ = tissue_materials("liver")
        pink_height = np.array(image.convert("L"), dtype=np.float32) / 255
        pink_dy, pink_dx = np.gradient(pink_height)
        pink_normal = np.stack((-pink_dx * 2, pink_dy * 2, np.ones_like(pink_height)), axis=-1)
        pink_normal /= np.linalg.norm(pink_normal, axis=-1, keepdims=True)
        lower_skin = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=pyrender.Texture(
                source=image, source_channels="RGB",
                sampler=pyrender.Sampler(wrapS=pyrender.GLTF.MIRRORED_REPEAT,
                                         wrapT=pyrender.GLTF.MIRRORED_REPEAT)),
            baseColorFactor=(1.0, 1.0, 1.0, 1.0),
            normalTexture=pyrender.Texture(
                source=Image.fromarray(((pink_normal * .5 + .5) * 255).astype(np.uint8)).convert("RGBA"),
                source_channels="RGBA"),
            metallicFactor=0, roughnessFactor=.26, doubleSided=True, alphaMode="OPAQUE")
        for pad in self.pads:
            mesh = trimesh.creation.icosphere(subdivisions=4)
            directions = mesh.vertices
            native = directions * pad.half_extents + pad.centre
            native += directions * (0.12 * wave_noise(native, 3.5, seed=8))[:, None]
            mesh.vertices = to_genesis(native)
            # A cavity-coloured pad shares the wall photograph's scale. The others keep the liver repeat.
            if pad.is_cavity_coloured:
                uv = np.column_stack((native[:, 0] / (aspect * shell_height), native[:, 2] / shell_height))
            else:
                uv = np.column_stack((native[:, 0], native[:, 2])) / FAT_TEXTURE_CM
            mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv.astype(np.float32))
            view.add_mesh(mesh, lower_skin if pad.is_cavity_coloured else packing_skin)


PERITONEAL_CAVITY = PeritonealCavity()
