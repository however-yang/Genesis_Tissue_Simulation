"""Bound validation to its existing local domain; never repair a failed state."""


def stop_reasons(quality):
    # These are the unchanged QualityMonitor decisions, not new tolerances.
    terminal = {'non_finite_state', 'invalid_material_tetrahedra',
                'workspace_displacement_exceeded', 'estimated_speed_limit_exceeded'}
    return ([reason for reason in quality['numerical_failure_reasons'] if reason in terminal]
            + list(quality['scope_failure_reasons']))
