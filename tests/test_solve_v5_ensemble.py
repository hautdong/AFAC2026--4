import unittest

from afac2026_starter.solve_v5_ensemble import (
    disagreement_kind,
    policy_accepts,
    should_arbitrate,
)


class SolveV5EnsembleTests(unittest.TestCase):
    def test_classifies_three_way_answers(self):
        self.assertEqual("all_agree", disagreement_kind("A", "A", "A"))
        self.assertEqual("v4_eq_graph", disagreement_kind("A", "B", "B"))
        self.assertEqual("v4_eq_v2", disagreement_kind("A", "B", "A"))
        self.assertEqual("v2_eq_graph", disagreement_kind("A", "A", "B"))
        self.assertEqual("all_different", disagreement_kind("A", "B", "C"))

    def test_only_arbitrates_unresolved_risk_groups(self):
        self.assertTrue(should_arbitrate("v2_eq_graph"))
        self.assertTrue(should_arbitrate("all_different"))
        self.assertFalse(should_arbitrate("v4_eq_graph"))
        self.assertFalse(should_arbitrate("v4_eq_v2"))

    def test_graph_confirmed_policy_rejects_novel_answer(self):
        self.assertTrue(policy_accepts("graph-confirmed", True, "A", "A", "AD"))
        self.assertFalse(policy_accepts("graph-confirmed", True, "A", "ABD", "AB"))
        self.assertFalse(policy_accepts("graph-confirmed", False, "A", "A", "AD"))


if __name__ == "__main__":
    unittest.main()
