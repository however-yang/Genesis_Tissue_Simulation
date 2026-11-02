import unittest
from validation_guard import stop_reasons


class ContinuationGuardTests(unittest.TestCase):
    def test_unchanged_valid_state_continues(self):
        self.assertEqual(stop_reasons(dict(numerical_failure_reasons=[], scope_failure_reasons=[])), [])

    def test_existing_local_scope_failure_stops_without_reclassifying_it(self):
        reason = 'detached_material_outside_local_task_scope'
        quality = dict(numerical_failure_reasons=[], scope_failure_reasons=[reason])
        self.assertEqual(stop_reasons(quality), [reason])
        self.assertEqual(quality['scope_failure_reasons'], [reason])

    def test_every_existing_hard_failure_stops(self):
        for reason in ('non_finite_state', 'invalid_material_tetrahedra',
                       'workspace_displacement_exceeded', 'estimated_speed_limit_exceeded'):
            with self.subTest(reason=reason):
                self.assertEqual(stop_reasons(dict(numerical_failure_reasons=[reason],
                                                  scope_failure_reasons=[])), [reason])

    def test_nonterminal_failures_remain_available_for_full_diagnostic_recording(self):
        failures = ['surface_topology_failure', 'tool_body_overlap_exceeded']
        quality = dict(numerical_failure_reasons=failures.copy(), scope_failure_reasons=[])
        self.assertEqual(stop_reasons(quality), [])
        self.assertEqual(quality['numerical_failure_reasons'], failures)


if __name__ == '__main__':
    unittest.main()
