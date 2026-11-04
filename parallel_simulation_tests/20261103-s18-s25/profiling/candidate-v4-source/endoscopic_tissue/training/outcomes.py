"""Opt-in outcome labels; preserve the underlying quality evidence verbatim."""

DETACHED = 'detached_material_outside_local_task_scope'


def classify(quality, before=None, damage=None, *, cutting_enabled=False):
    numerical = list(quality.get('numerical_failure_reasons', ()))
    scope = list(quality.get('scope_failure_reasons', ()))
    # A workspace limit describes the collection domain, not broken integration.
    if 'workspace_displacement_exceeded' in numerical:
        numerical.remove('workspace_displacement_exceeded')
        scope.append('workspace_displacement_exceeded')
    result = dict(schema='physical_outcome_v1', valid_outcome=True, terminal_reason=None,
                  material_separation=None, numerical_reasons=numerical, scope_reasons=scope)
    if numerical or (not quality['checked_metrics_passed'] and not scope):
        result.update(valid_outcome=False, terminal_reason='numerical:' + (numerical[0] if numerical else 'unspecified'))
        return result
    if DETACHED in scope:
        old = (before or {}).get('metrics', {}).get('material_components', 1)
        new = quality['metrics']['material_components']
        tears = (damage or {}).get('new_material_tear_edges', 0)
        blades = (damage or {}).get('new_material_blade_edges', 0)
        other_scope = [s for s in scope if s != DETACHED]
        # Simultaneous tearing and cutting is deliberately not attributed by guess.
        anchors = (damage or {}).get('anchors', {})
        detached_release = (new == old and new > 1 and not tears and not blades
                            and anchors.get('new_broken_material_points', 0) > 0
                            and anchors.get('remaining_material_points') == 0)
        if detached_release and not other_scope:
            result.update(terminal_reason='damage:fragment_detached', material_separation='previously_separated_fragment_released')
        elif new > old and tears > 0 and blades == 0 and not other_scope:
            result.update(terminal_reason='damage:tear_separation', material_separation='tear')
        elif new > old and blades > 0 and tears == 0 and cutting_enabled and not other_scope:
            # End at the current validated-domain boundary, but retain a valid cut
            # outcome for task evaluation instead of declaring material damage.
            result.update(terminal_reason='scope:blade_separation', material_separation='blade')
        else:
            result.update(valid_outcome=False, terminal_reason='scope:unattributed_material_separation',
                          material_separation='mixed' if tears and blades else 'unknown')
    elif scope:
        result.update(valid_outcome=False, terminal_reason='scope:' + scope[0])
    elif not quality['within_declared_local_scope']:
        result.update(valid_outcome=False, terminal_reason='scope:unspecified')
    return result


def state_valid(metadata):
    outcome = metadata.get('outcome')
    if outcome is not None:
        q = metadata['quality']
        numerical = [r for r in q.get('numerical_failure_reasons', ()) if r != 'workspace_displacement_exceeded']
        if numerical or (not q['checked_metrics_passed'] and not q.get('numerical_failure_reasons')):
            return False
        return bool(outcome['valid_outcome'])
    q = metadata['quality']
    return bool(q['checked_metrics_passed'] and q['within_declared_local_scope'])


def step_events(before, after, action, task=None, *, retained_threshold=.5):
    """Event history lives in State, so ordinary snapshots restore it exactly."""
    prior = before.metadata.get('event_history', {})
    history = dict(prior)
    events = []
    d = after.metadata.get('failure_physics', {})
    previous_damage = before.metadata.get('failure_physics', {})
    if d.get('new_material_tear_edges', 0) and not history.get('tear_seen',
            previous_damage.get('native_cumulative_counts', {}).get('materialTearEdges', 0) > 0):
        events.append('first_tear_edge')
        history['tear_seen'] = True
    anchors = d.get('anchors', {})
    if anchors.get('new_broken_material_points', 0) or anchors.get('new_broken_support_corners', 0):
        if not history.get('anchor_release_active', False):
            events.append('anchor_release_started')
        history['anchor_release_active'] = True
    elif after.metadata['grasp'].get('anchor_points', 0) > before.metadata['grasp'].get('anchor_points', 0):
        history['anchor_release_active'] = False
    old_grasp, new_grasp = before.metadata['grasp'], after.metadata['grasp']
    if (old_grasp.get('witness_ids') and old_grasp['witness_retained_fraction'] >= retained_threshold
            and new_grasp['witness_retained_fraction'] < retained_threshold):
        events.append('grasp_lost')
    command = action.get('requested_command_native', {})
    active = bool(command.get('scissors_cutting') and command.get('scissors_closure', 0) > 0)
    if active and not history.get('stroke_active', False):
        history['stroke_id'] = history.get('stroke_id', 0) + 1
        history['stroke_has_cut'] = False
    if d.get('new_material_blade_edges', 0) and active and not history.get('stroke_has_cut', False):
        events.append('cut_onset')
        history['stroke_has_cut'] = True
    if not active and history.get('stroke_active', False):
        events.append('cut_stroke_end')
    # Closing a tool is NOT proof of cutting through tissue.
    if after.metadata.get('outcome', {}).get('material_separation') == 'blade':
        events.append('cut_separation')
    if task and task.get('metrics', {}).get('cut_progress') == 1 and not history.get('goal_cut_complete', False):
        events.append('cut_goal_complete')
        history['goal_cut_complete'] = True
    history['stroke_active'] = active
    after.metadata['event_history'] = history
    after.metadata['events'] = [dict(kind=kind, step_id=after.metadata['step_id'],
        native_tick=after.metadata['native_tick'], t_sim_s=after.metadata['t_sim_s'],
        stroke_id=history.get('stroke_id'), witness_retained_threshold=retained_threshold) for kind in events]
