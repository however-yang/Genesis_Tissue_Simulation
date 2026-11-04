"""Conservative collection checks. These do not certify physical realism."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..instrument_geometry import scissors_body_depths
from ..surface_diagnostics import topology_statistics


def material_components(neighbors, material):
    ids = np.flatnonzero(material)
    mapping = np.full(len(material), -1, dtype=np.int32)
    mapping[ids] = np.arange(len(ids))
    targets = neighbors[ids, 1::2]
    owner, slot = np.where(targets >= 0)
    target = targets[owner, slot]
    keep = material[target]
    owner, target = owner[keep], mapping[target[keep]]
    graph = coo_matrix((np.ones(len(owner), np.uint8), (owner, target)), shape=(len(ids), len(ids))).tocsr()
    return int(connected_components(graph, directed=False, return_labels=False))


@dataclass(frozen=True)
class QualityLimits:
    max_displacement_m: float = .06
    max_body_overlap_m: float = .0002
    max_estimated_speed_m_s: float = 2.0

    def __post_init__(self):
        values = (self.max_displacement_m, self.max_body_overlap_m, self.max_estimated_speed_m_s)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Quality limits must be finite and positive")


class QualityMonitor:
    def __init__(self, grid, limits=QualityLimits(), *, topology_function=None):
        self.grid, self.limits = grid, limits
        self.topology_function = topology_function
        self.previous_neighbors = grid.neighbors
        self.components = material_components(grid.neighbors, grid.inside != 0)

    def assess(self, observed, previous, command, settings, material_quality, surface, geometry,
               *, permitted_tear_edges=0):
        failures, scope = [], []
        finite_quality = bool(np.isfinite(material_quality).all())
        values = {"min_volume_ratio": float(material_quality[0]) if finite_quality else None,
                  "invalid_tets": int(material_quality[1]) if finite_quality else None}
        if not np.isfinite(observed.nodes).all() or not finite_quality:
            failures.append("non_finite_state")
        if values["invalid_tets"]:
            failures.append("invalid_material_tetrahedra")
        active = self.grid.active != 0
        displacement = np.linalg.norm(observed.nodes[active] - self.grid.positions[active], axis=1) * .01
        max_displacement = float(displacement.max(initial=0))
        values["max_displacement_m"] = max_displacement if np.isfinite(max_displacement) else None
        if max_displacement > self.limits.max_displacement_m:
            failures.append("workspace_displacement_exceeded")
        if previous is not None:
            speed = np.linalg.norm(observed.nodes[active] - previous.nodes[active], axis=1) * (.01 / settings.dt)
            max_speed = float(speed.max(initial=0))
            values["max_estimated_speed_m_s"] = max_speed if np.isfinite(max_speed) else None
            if max_speed > self.limits.max_estimated_speed_m_s:
                failures.append("estimated_speed_limit_exceeded")
        changed = observed.neighbors is not self.previous_neighbors
        values.update(new_cut_edges=0, healed_edges=0, unexpected_neighbor_changes=0)
        if changed:
            before, after = self.previous_neighbors[:, 1::2], observed.neighbors[:, 1::2]
            values["new_cut_edges"] = int(np.count_nonzero((before >= 0) & (after < 0)))
            values["healed_edges"] = int(np.count_nonzero((before < 0) & (after >= 0)))
            values["unexpected_neighbor_changes"] = int(np.count_nonzero((before >= 0) & (after >= 0) & (before != after)))
            if values["healed_edges"] or values["unexpected_neighbor_changes"]:
                failures.append("topology_irreversibility_violation")
            if values["new_cut_edges"]>permitted_tear_edges and not command.scissors_cutting:
                failures.append("new_cut_while_disabled")
            self.components = material_components(observed.neighbors, self.grid.inside != 0)
        values["material_components"] = self.components
        if self.components != 1:
            scope.append("detached_material_outside_local_task_scope")
        self.previous_neighbors = observed.neighbors
        unchecked = ["true_curved_edge_pointwise_cut_correspondence", "internal_solver_residual",
                     "free_fragment_continuous_collision", "task_region_cut_completion",
                     "friction_grasp_task_success"]
        if surface is None:
            unchecked += ["tool_body_overlap", "surface_topology"]
        else:
            s = observed.scissors
            body, wrist = scissors_body_depths(surface.vertices, surface.triangles, geometry,
                                               s.hinge, s.axis, s.normal, s.gap)
            # Keep the geometry calculation and threshold comparison unchanged.
            # Positive depths may be numpy.float32; records require JSON scalars.
            values.update(blade_body_overlap_m=float(body * .01), wrist_overlap_m=float(wrist * .01))
            if max(body, wrist) * .01 > self.limits.max_body_overlap_m:
                failures.append("tool_body_overlap_exceeded")
            topo = (self.topology_function or topology_statistics)(surface)
            values["surface_topology"] = topo
            if (topo["positional_boundary_edges"] or topo["positional_nonmanifold_edges"]
                    or topo["duplicate_triangles"] or topo["degenerate_triangles"] or not topo["winding_consistent"]):
                failures.append("surface_topology_failure")
        return {"checked_metrics_passed": not failures, "numerical_failure_reasons": failures,
                "scope_failure_reasons": scope, "within_declared_local_scope": not scope,
                "unchecked": unchecked, "metrics": values, "training_domain_approved": False}
