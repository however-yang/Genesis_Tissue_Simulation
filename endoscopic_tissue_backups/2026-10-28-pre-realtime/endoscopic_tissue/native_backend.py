"""Strict C ABI adapter for the user's isolated CUDA solver snapshot."""
import ctypes as ct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grid import METERS_PER_UNIT, TissueGrid
from .instrument_geometry import (SCISSORS_WRIST_FRONT, knife_frame, knife_geometry, load_grasper, load_scissors,
                                  scissors_body_depths)

F = ct.c_float
I = ct.c_int32
P = ct.c_void_p
V3 = F * 3


class ReconDesc(ct.Structure):
    _fields_ = [(name, I) for name in (
        "voxels", "nodes", "intersections", "surface_edges", "tri_capacity", "nx", "ny", "nz")]
    _fields_ += [("spacing", F), ("qef_iterations", I)]


class PhysicsDesc(ct.Structure):
    _fields_ = [("nodes", I), ("bends", I)]
    _fields_ += [(name, F) for name in ("ks", "cs", "kb", "cb", "h_init", "h_max")]
    _fields_ += [("substeps", I)]
    _fields_ += [(name, F) for name in ("alpha", "young", "poisson", "strain_damping")]


class CutDesc(ct.Structure):
    _fields_ = [(name, I) for name in (
        "voxels", "nodes", "edges", "intersections", "tri_capacity", "nx", "ny", "nz")]
    _fields_ += [("spacing", F), ("origin", V3), ("point_capacity", I), ("interpolation", F), ("tear_ratio", F)]


class CutTool(ct.Structure):
    _fields_ = [(name, V3) for name in ("t1a", "t1b", "t1c", "t2a", "t2b", "t2c", "normal")]
    _fields_ += [("thickness", F), ("valid", I), ("low", V3), ("high", V3)]


class Grasper(ct.Structure):
    _fields_ = [("shaft_a", V3), ("shaft_b", V3), ("radius", F)]
    _fields_ += [(name, V3) for name in ("hinge", "axis", "mid", "normal")]
    _fields_ += [(name, F) for name in ("half_width", "length", "gap", "closure")]


class GraspParams(ct.Structure):
    _fields_ = [(name, F) for name in (
        "capture", "release", "closed_gap", "core_fraction", "friction", "contact_compliance",
        "skin", "ccd_velocity", "ccd_displacement", "ccd_omega", "grasp_omega",
        "core_compliance", "break_force", "grasp_displacement")]
    _fields_ += [("exempt_rings", I)]


class Blade(ct.Structure):
    _fields_ = [("a", V3), ("b", V3)]
    _fields_ += [(name, F) for name in ("radius", "start_force", "cut_force", "stiffness", "mass", "friction")]
    _fields_ += [("enabled", I)]


class ScissorsBack(ct.Structure):
    _fields_ = [(name, V3) for name in ("center", "half_size", "axis_x", "axis_y")]


class ScissorsControl(ct.Structure):
    _fields_ = [(name, V3) for name in ("hinge", "axis", "normal", "left_a", "left_b", "right_a", "right_b")]
    _fields_ += [(name, F) for name in ("length", "gap", "max_gap", "jaw_speed", "start_force", "cut_force",
                                      "stiffness", "mass", "friction", "jaw_mass", "jaw_stiffness")]
    _fields_ += [(name, I) for name in ("enabled", "cutting", "guided")]
    _fields_ += [("left_back", ScissorsBack), ("right_back", ScissorsBack), ("back_contact", I), ("squeeze", F)]


class ScissorsResult(ct.Structure):
    _fields_ = [(name, V3) for name in ("hinge", "axis", "normal", "left_a", "left_b", "right_a", "right_b")]
    _fields_ += [(name, F) for name in ("gap", "radius", "pressure", "mass", "jaw_mass")]
    _fields_ += [("force", V3), ("phase", I), ("contacts", I)]


@dataclass(frozen=True)
class Settings:
    dt: float = 1 / 30
    substeps: int = 3
    young_pa: float = 6000
    poisson: float = 0.25
    strain_damping_s: float = 0.04
    density_kg_m3: float = 1050
    jaw_length: float = 1.99
    jaw_width: float = 0.52
    jaw_open_gap: float = 1.0
    jaw_axis: tuple[float, float, float] = (0.0, -0.8, 0.6)
    jaw_normal: tuple[float, float, float] = (1.0, 0.0, 0.0)
    blade_length: float = 1.6
    blade_radius: float = 0.04
    blade_axis: tuple[float, float, float] = (2 ** -.5, 0.0, 2 ** -.5)
    start_force_n: float = 0.2
    cut_force_n: float = 0.12
    blade_drive_stiffness: float = 1200
    blade_mass_kg: float = 0.02
    blade_friction: float = 0.15
    blade_preload_n: float = 1.0
    contact_refine_passes: int = 8
    has_knife_body: bool = True
    instrument_model: str = "davinci"
    cutting_tool: str = "knife"
    # Spring from the commanded to the solved tip gap, in native force per cm.
    # It sets how hard the closing jaw presses against tissue resistance.
    scissors_jaw_stiffness: float = 150
    # Tip closure in cm over which both blades must load the tissue before it
    # can fracture. A smaller travel cuts tissue that the blades have barely
    # deformed; a larger one squeezes it more, and the blades may close fully
    # before the travel is reached. A fractured snip whose tips close this far
    # without touching tissue has left it, and must squeeze new tissue again.
    scissors_squeeze_cm: float = .2


@dataclass(frozen=True)
class Command:
    hinge: tuple[float, float, float] = (-1.4, 3.384, -2.288)
    closure: float = 0.0
    blade_center: tuple[float, float, float] = (0.2, 2.2, -0.6)
    blade_enabled: bool = True
    scissors_closure: float = 0.0
    scissors_cutting: bool = False
    # Scissors orientation in degrees, see scissors_frame.
    scissors_pitch_deg: float = 19.9
    scissors_roll_deg: float = 0.0
    scissors_yaw_deg: float = 0.0


def scissors_frame(pitch_deg, roll_deg=0.0, yaw_deg=0.0):
    """Return the scissors axis and opening normal in native coordinates.

    At zero angles the scissors point toward native -x and the first blade opens toward +y. The roll turns the opening
    normal about the axis toward +z, the pitch then tilts the axis down, and the yaw turns it about +y toward +z.
    """
    roll, pitch, yaw = np.radians((roll_deg, pitch_deg, yaw_deg))
    turn_roll = np.array(((1, 0, 0), (0, np.cos(roll), -np.sin(roll)), (0, np.sin(roll), np.cos(roll))))
    turn_pitch = np.array(((np.cos(pitch), -np.sin(pitch), 0), (np.sin(pitch), np.cos(pitch), 0), (0, 0, 1)))
    turn_yaw = np.array(((np.cos(yaw), 0, np.sin(yaw)), (0, 1, 0), (-np.sin(yaw), 0, np.cos(yaw))))
    rotation = turn_yaw @ turn_pitch @ turn_roll
    return rotation @ np.array((-1., 0, 0)), rotation @ np.array((0, 1., 0))


@dataclass(frozen=True)
class ScissorsLoads:
    """Compressive blade loads in newtons from the latest accepted substep, and at the latest fracture."""
    left_n: float
    right_n: float
    fracture_left_n: float
    fracture_right_n: float
    fracture_total_n: float
    fractures: int
    # Both blades have loaded the tissue in the current pressing gesture: the
    # native fracture law counts the squeeze from there.
    is_squeezing: bool


@dataclass(frozen=True)
class ScissorsPose:
    hinge: np.ndarray
    axis: np.ndarray
    normal: np.ndarray
    gap: float
    radius: float
    edges: np.ndarray
    pressure_n: float
    phase: int
    contacts: int
    loads: ScissorsLoads


@dataclass
class Observation:
    nodes: np.ndarray
    neighbors: np.ndarray
    grip_flags: np.ndarray
    hinge: np.ndarray
    jaw_gap: float
    grip_points: int
    grip_error_m: float
    blade: np.ndarray
    reaction_n: np.ndarray
    scissors: ScissorsPose | None = None


@dataclass
class Surface:
    vertices: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    anchors: np.ndarray


def pointer(array):
    assert array.flags.c_contiguous
    return array.ctypes.data_as(P)


class NativeTissue:
    """Owns one process-global native session; close before constructing another."""
    active_session = False

    def __init__(self, grid: TissueGrid, settings=Settings(), dll_path=None, initial_command=Command()):
        if NativeTissue.active_session:
            raise RuntimeError("CUDA solver is single-session; use separate processes for separate environments")
        self.grid, self.settings = grid, settings
        self.initial_command = initial_command
        if settings.cutting_tool not in ("knife", "scissors"):
            raise ValueError(f"Unknown cutting tool: {settings.cutting_tool}")
        self.scissors_geometry = load_scissors() if settings.cutting_tool == "scissors" else None
        self.grasper_geometry = load_grasper(settings.jaw_length, settings.jaw_width, settings.instrument_model)
        self.knife_geometry = knife_geometry(settings.blade_length, settings.instrument_model)
        self.blade_axis = knife_frame(settings.blade_axis)[0]
        self.closed = True
        # The scissors fracture law and its load export exist only in the scissors build.
        default = "native/bin/LiverCudaSim.dll" if self.scissors_geometry is None else "native/bin-scissors/LiverCudaSim.dll"
        self.dll_path = Path(dll_path) if dll_path else Path(__file__).parent / default
        self.dll = ct.CDLL(str(self.dll_path.resolve()))
        signatures = (
            ("LCS_Init", (P,) * 7, I), ("LCS_InitPhysics", (P,) * 8, I),
            ("LCS_InitCut", (P,) * 5, I), ("LCS_SetTool", (P,), None),
            ("LCS_PhysicsAbiVersion", (), I), ("LCS_GraspToolAbiVersion", (), I),
            ("LCS_PressureBladeAbiVersion", (), I), ("LCS_ContactModelVersion", (), I),
            ("LCS_SetGraspParams", (P,), None), ("LCS_SetGraspTool", (P,), I),
            ("LCS_SetPressureBlade", (P,), I), ("LCS_SetPressureCutMode", (I,), None),
            ("LCS_ScissorsAbiVersion", (), I), ("LCS_ScissorsResultAbiVersion", (), I),
            ("LCS_SetScissors", (P,), I), ("LCS_GetScissors", (P,), I),
            ("LCS_SetContactModel", (I, F, F, F, F), I),
            ("LCS_SetJawGeometry", (P,), I),
            ("LCS_SetKnifeGeometry", (I, P), I), ("LCS_KnifeBodyDepth", (P,), I),
            ("LCS_RefineContactSurface", (I,), I), ("LCS_SurfaceDepth", (P,), I),
            ("LCS_SetTissueGuard", (F, I, I, P, I), I),
            ("LCS_SetTissueMaterial", (F, F), I), ("LCS_SetGraspPatch", (F, F, F), I),
            ("LCS_SetToolDynamics", (F, F, F, F), I), ("LCS_SetJawDynamics", (F, F, F, F), I),
            ("LCS_SetInstrumentControl", (I,), I), ("LCS_SetGravity", (F, F, F), I),
            ("LCS_Step", (F,), I), ("LCS_Finalize", (), I),
            ("LCS_GetSurfaceCounts", (P, P), I), ("LCS_GetSurface", (P, P, P), I),
            ("LCS_GetSurfaceAux", (P,), I), ("LCS_GetCornerPos", (P,), I),
            ("LCS_GetNbrIdx", (P,), I), ("LCS_GetGraspCoreFlags", (I, P), I),
            ("LCS_GetSolvedTool", (P, P, P), I), ("LCS_GetSolvedJaw", (P,), I),
            ("LCS_GetPressureBlade", (P,), I), ("LCS_GetContactReaction", (P, P), I),
            ("LCS_GetMaterialQuality", (P,), I),
            ("LCS_GetCutDebug", (P, P), I), ("LCS_GetCutEdgeFractions", (P,), I),
            ("LCS_Shutdown", (), None))
        if self.scissors_geometry is not None:
            signatures += (("LCS_GetScissorsLoads", (P,), I), ("LCS_SetScissorsWrist", (F, F, F), I))
        self.functions = {}  # Name-to-function dispatch table; ctypes indexed lookup is not cached.
        for name, args, result in signatures:
            function = self.dll[name]
            function.argtypes, function.restype = args, result
            self.functions[name] = function
        abi = (("LCS_PhysicsAbiVersion", PhysicsDesc), ("LCS_GraspToolAbiVersion", Grasper),
               ("LCS_PressureBladeAbiVersion", Blade))
        # Only a scissors run sends scissors commands. The verified knife build
        # keeps the older scissors command layout.
        if self.scissors_geometry is not None:
            abi += (("LCS_ScissorsAbiVersion", ScissorsControl), ("LCS_ScissorsResultAbiVersion", ScissorsResult))
        for name, structure in abi:
            actual = self.functions[name]()
            if actual != ct.sizeof(structure):
                raise RuntimeError(f"{name}: DLL size {actual}, Python size {ct.sizeof(structure)}")
        if self.functions["LCS_ContactModelVersion"]() != 28:
            raise RuntimeError("This adapter requires the snapshotted contact model version 28")
        NativeTissue.active_session = True
        self.closed = False
        try:
            self.initialize()
        except (RuntimeError, ValueError, OSError):
            self.close()
            raise

    def check(self, name, *args):
        result = self.functions[name](*args)
        if result != 0:
            raise RuntimeError(f"{name} failed with native status {result}")

    def initialize(self):
        g, s = self.grid, self.settings
        n, v = len(g.positions), len(g.corners)
        # Fixed native topology storage, same capacity contract as the source plugin.
        self.tri_capacity = 6 * len(g.surface_edges) + 1024 + 2 * len(g.edges)
        rd = ReconDesc(v, n, len(g.intersections), len(g.surface_edges), self.tri_capacity,
                       *g.dims, g.spacing, 8)
        self.check("LCS_Init", ct.byref(rd), pointer(g.positions), pointer(g.corners),
                   pointer(g.offsets), pointer(g.counts), pointer(g.intersections), pointer(g.surface_edges))
        pd = PhysicsDesc(n, n * 15, 0, 0, 0, 0, 0.0001, 0.0015,
                         s.substeps, 0, s.young_pa * METERS_PER_UNIT, s.poisson, s.strain_damping_s)
        external = np.zeros_like(g.positions)
        self.check("LCS_InitPhysics", ct.byref(pd), pointer(g.mass), pointer(g.pinned),
                   pointer(g.active), pointer(external), pointer(g.neighbors),
                   pointer(g.rest_neighbors), pointer(g.bends))
        cd = CutDesc(v, n, len(g.edges), len(g.intersections), self.tri_capacity,
                     *g.dims, g.spacing, V3(*g.origin), 4 * len(g.edges), 0, 0)
        self.check("LCS_InitCut", ct.byref(cd), pointer(g.connectivity), pointer(g.edges),
                   pointer(g.occupied), pointer(g.inside))
        self.check("LCS_SetTissueGuard", 0.15, 1, n, pointer(g.inside), 1)
        self.check("LCS_SetTissueMaterial", 80, 0.01)
        self.check("LCS_SetGraspPatch", 0.35, 1 / 100000, 0.005)
        self.check("LCS_SetToolDynamics", 0.02, 1200, 10, g.spacing * 0.25)
        self.check("LCS_SetJawDynamics", 0.005, 150, 2, 1.2)
        self.check("LCS_SetInstrumentControl", 1)
        gp = GraspParams(0.5, 0.3, 0, 0.55, 0.6, 1e-8, g.spacing * 0.5,
                         0.5, g.spacing * 0.2, 0.25, 0.25, 1e-5, 15000,
                         g.spacing * 0.2, 1)
        self.functions["LCS_SetGraspParams"](ct.byref(gp))
        self.check("LCS_SetContactModel", 2, 0.6, 0.4, 0.1, 0.01)
        self.check("LCS_SetJawGeometry", pointer(self.grasper_geometry.bounds))
        self.check("LCS_SetKnifeGeometry", int(s.has_knife_body and self.scissors_geometry is None),
                   pointer(self.knife_geometry.boxes))
        self.functions["LCS_SetPressureCutMode"](1)
        tool = CutTool()
        tool.normal, tool.thickness = V3(1, 0, 0), 2 * s.blade_radius
        self.functions["LCS_SetTool"](ct.byref(tool))
        if self.scissors_geometry is not None:
            # The wrist pushes tissue aside as the scissors follow their cut; it
            # is the envelope scissors_body_depths measures.
            jaws = self.scissors_geometry.jaws
            self.check("LCS_SetScissorsWrist", SCISSORS_WRIST_FRONT, jaws.shaft_length, jaws.shaft_radius)
        self.command(self.initial_command)
        # The source allocates pressure-blade feedback on its first accepted step.
        # Initialize at rest with zero body force before exposing observations.
        self.check("LCS_Step", s.dt)
        self.check("LCS_Finalize")
        self.tick = 0

    def command(self, command: Command):
        s = self.settings
        if not np.isfinite((*command.hinge, *command.blade_center, command.closure)).all():
            raise ValueError("Tool command contains non-finite values")
        if not 0 <= command.closure <= 1 or not 0 <= command.scissors_closure <= 1:
            raise ValueError("Jaw closure must be in [0,1]")
        hinge = np.array(command.hinge)
        axis = np.array(s.jaw_axis)
        geometry = self.grasper_geometry
        tool = Grasper(V3(*(hinge - geometry.shaft_length * axis)), V3(*hinge), geometry.shaft_radius,
                       V3(*hinge), V3(*axis), V3(*(hinge + s.jaw_length / 2 * axis)),
                       V3(*s.jaw_normal), s.jaw_width / 2, s.jaw_length,
                       s.jaw_open_gap * (1 - command.closure), command.closure)
        self.check("LCS_SetGraspTool", ct.byref(tool))
        if self.scissors_geometry is not None:
            geometry = self.scissors_geometry
            backs = tuple(ScissorsBack(*(V3(*v) for v in box.reshape(4, 3))) for box in geometry.backs)
            axis, normal = scissors_frame(command.scissors_pitch_deg, command.scissors_roll_deg, command.scissors_yaw_deg)
            scissors = ScissorsControl(V3(*command.blade_center), V3(*axis), V3(*normal),
                *(V3(*v) for v in geometry.edges.reshape(4, 3)),
                geometry.jaws.length, geometry.max_gap * (1 - command.scissors_closure), geometry.max_gap,
                1.2, s.start_force_n / METERS_PER_UNIT, s.cut_force_n / METERS_PER_UNIT,
                s.blade_drive_stiffness, s.blade_mass_kg, s.blade_friction, .005, s.scissors_jaw_stiffness,
                1, int(command.scissors_cutting), 0, *backs, 1, s.scissors_squeeze_cm)
            self.check("LCS_SetScissors", ct.byref(scissors))
            return
        center = np.array(command.blade_center)
        offset = self.blade_axis * (s.blade_length / 2)
        blade = Blade(V3(*(center - offset)), V3(*(center + offset)),
                      s.blade_radius, s.start_force_n / METERS_PER_UNIT,
                      s.cut_force_n / METERS_PER_UNIT, s.blade_drive_stiffness,
                      s.blade_mass_kg, s.blade_friction, int(command.blade_enabled))
        self.check("LCS_SetPressureBlade", ct.byref(blade))

    def step(self, command: Command):
        if self.closed:
            raise RuntimeError("Native session is closed")
        self.command(command)
        ramp = min(1.0, (self.tick + 1) * self.settings.dt / 0.5)
        self.check("LCS_SetGravity", 0, -9.81 / METERS_PER_UNIT * ramp, 0)
        self.check("LCS_Step", self.settings.dt)
        self.check("LCS_Finalize")
        if self.settings.contact_refine_passes:
            self.check("LCS_RefineContactSurface", self.settings.contact_refine_passes)
        self.tick += 1
        return self.observe()

    def observe(self):
        n = len(self.grid.positions)
        nodes = np.empty((n, 3), dtype=np.float32)
        neighbors = np.empty((n, 6), dtype=np.int32)
        grip_flags = np.empty(n, dtype=np.int32)
        hinge, error, live, gap = V3(), F(), I(), F()
        blade = np.empty(10, dtype=np.float32)
        reaction, contacts = np.empty(3, dtype=np.float32), I()
        self.check("LCS_GetCornerPos", pointer(nodes))
        self.check("LCS_GetNbrIdx", pointer(neighbors))
        self.check("LCS_GetGraspCoreFlags", n, pointer(grip_flags))
        self.check("LCS_GetSolvedTool", ct.byref(hinge), ct.byref(error), ct.byref(live))
        self.check("LCS_GetSolvedJaw", ct.byref(gap))
        self.check("LCS_GetPressureBlade", pointer(blade))
        scissors = None
        if self.scissors_geometry is not None:
            result = ScissorsResult()
            self.check("LCS_GetScissors", ct.byref(result))
            loads = np.empty(7, dtype=np.float32)
            self.check("LCS_GetScissorsLoads", pointer(loads))
            scissors = ScissorsPose(np.array(result.hinge), np.array(result.axis), np.array(result.normal),
                                    result.gap, result.radius,
                                    np.array((result.left_a, result.left_b, result.right_a, result.right_b)).reshape(2, 2, 3),
                                    result.pressure * METERS_PER_UNIT, result.phase, result.contacts,
                                    ScissorsLoads(*(float(v) for v in loads[:5] * METERS_PER_UNIT), int(loads[5]),
                                                  bool(loads[6])))
            if not np.isfinite((*scissors.hinge, *scissors.axis, *scissors.normal, scissors.gap,
                                *scissors.edges.reshape(12), *loads)).all():
                raise RuntimeError("Non-finite scissors state; simulation stopped")
        self.check("LCS_GetContactReaction", pointer(reaction), ct.byref(contacts))
        if (not np.isfinite(nodes).all() or not np.isfinite(blade).all() or not np.isfinite(reaction).all()
                or not np.isfinite((*hinge, gap.value, error.value)).all()):
            raise RuntimeError("Non-finite native state; simulation stopped")
        return Observation(nodes, neighbors, grip_flags, np.array(hinge), gap.value, live.value,
                           error.value * METERS_PER_UNIT, blade, reaction * METERS_PER_UNIT, scissors)

    def surface(self):
        nv, ni = I(), I()
        self.check("LCS_GetSurfaceCounts", ct.byref(nv), ct.byref(ni))
        if nv.value <= 0 or ni.value % 3:
            raise RuntimeError("Native surface is empty or has an invalid triangle count")
        vertices = np.empty((nv.value, 3), dtype=np.float32)
        normals = np.empty_like(vertices)
        indices = np.empty(ni.value, dtype=np.int32)
        aux = np.empty((nv.value, 4), dtype=np.float32)
        self.check("LCS_GetSurface", pointer(vertices), pointer(normals), pointer(indices))
        self.check("LCS_GetSurfaceAux", pointer(aux))
        debug, interpolation = np.empty(18, dtype=np.uint32), F()
        self.check("LCS_GetCutDebug", pointer(debug), ct.byref(interpolation))
        if debug[16] > 4 * len(self.grid.edges) or debug[17] > 3 * self.tri_capacity:
            raise RuntimeError("Native cut/surface capacity exceeded")
        if not np.isfinite(vertices).all() or not np.isfinite(normals).all():
            raise RuntimeError("Non-finite reconstructed surface")
        return Surface(vertices, normals, indices.reshape(-1, 3), aux)

    def contact_depth_m(self):
        depth = F()
        self.check("LCS_SurfaceDepth", ct.byref(depth))
        return depth.value * METERS_PER_UNIT

    def cut_edge_fractions(self):
        fractions = np.empty((len(self.grid.positions), 6), dtype=np.float32)
        self.check("LCS_GetCutEdgeFractions", pointer(fractions))
        return fractions

    def material_quality(self):
        quality = np.empty(2, dtype=np.float32)
        self.check("LCS_GetMaterialQuality", pointer(quality))
        return quality

    def scissors_body_depths_m(self):
        """Return the tissue overlap with the blade bodies and with the wrist and shaft, in metres, from one surface read."""
        result = ScissorsResult()
        self.check("LCS_GetScissors", ct.byref(result))
        surface = self.surface()
        depths = scissors_body_depths(surface.vertices, surface.triangles, self.scissors_geometry,
                                      np.array(result.hinge), np.array(result.axis), np.array(result.normal), result.gap)
        return tuple(float(depth) * METERS_PER_UNIT for depth in depths)

    def knife_body_depth_m(self):
        depth = F()
        self.check("LCS_KnifeBodyDepth", ct.byref(depth))
        return depth.value * METERS_PER_UNIT

    def close(self):
        if not self.closed:
            self.functions["LCS_Shutdown"]()
            self.closed = True
            NativeTissue.active_session = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
