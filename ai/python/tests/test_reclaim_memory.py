import unittest

from fa_env.reclaim_memory import after_step


class ReclaimMemoryTest(unittest.TestCase):
    def setUp(self):
        self.old = [{"id": "804258102", "mass": 30, "x": 2, "z": 3}]
        self.scan = {"type": "reclaim", "pos": [0, 0], "radius": 50}

    def test_successful_empty_scan_forgets_consumed_wreck(self):
        self.assertEqual(after_step(self.old, [self.scan], [{"ok": True, "value": []}]), [])
        self.assertEqual(len(self.old), 1)  # previous decision's observation remains intact

    def test_no_scan_failed_scan_and_missing_reply_preserve_knowledge(self):
        for queries, answers in [([], []), ([self.scan], []), ([self.scan], [{"ok": False}])]:
            self.assertEqual(after_step(self.old, queries, answers), self.old)

    def test_partial_scan_merges_new_evidence_without_inventing_absence(self):
        new = {"id": "new", "mass": 9}
        queries = [{"type": "site"}, self.scan, self.scan]
        answers = [{"ok": True}, {"ok": True, "value": [new]}, {"ok": False}]
        self.assertEqual(after_step(self.old, queries, answers), self.old + [new])
        answers[-1] = {"ok": True, "value": []}
        self.assertEqual(after_step(self.old, queries, answers), [new])

    def test_unknown_reclaim_target_is_removed_before_next_decision(self):
        action = {"type": "reclaim", "target": 804258102}
        result = {"ok": False, "error": "unknown target 804258102"}
        self.assertEqual(after_step(self.old, [], [], [action], [result]), [])
        for other in [{"ok": True}, {"ok": False, "error": "unknown unit 804258102"},
                      {"ok": False, "error": "unknown target 99"}]:
            self.assertEqual(after_step(self.old, [], [], [action], [other]), self.old)

    def test_id_types_deduplicate_and_other_action_failure_is_not_reclaim_evidence(self):
        fresh = {"id": 804258102, "mass": 10}
        self.assertEqual(after_step(self.old, [self.scan, self.scan],
            [{"ok": True, "value": [fresh]}, {"ok": False}]), [fresh])
        self.assertEqual(after_step(self.old, [], [], [{"type": "attack", "target": 804258102}],
            [{"ok": False, "error": "unknown target 804258102"}]), self.old)

    def test_real_lua_traceback_message_removes_only_its_exact_target(self):
        action = {"type": "reclaim", "units": ["34"], "target": "804258102"}
        traceback = '\nstack traceback:\n\tagentbridge.lua:746: in function TargetEntity\n\t[C]: in function pcall'
        self.assertEqual(after_step(self.old, [], [], [action],
            [{"ok": False, "error": "unknown target 804258102" + traceback}]), [])
        for message in ["unknown target 80425810", "unknown unit 804258102", "target 804258102 is not visible"]:
            self.assertEqual(after_step(self.old, [], [], [action],
                [{"ok": False, "error": message + traceback}]), self.old)


if __name__ == "__main__":
    unittest.main()
