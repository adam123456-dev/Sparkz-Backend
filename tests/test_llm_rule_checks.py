import unittest

from app.checklists.llm_rule_checks import _filter_low_signal_checks, _normalize_rule_checks


class LlmRuleChecksTests(unittest.TestCase):
    def test_normalize_rule_checks(self) -> None:
        checks = _normalize_rule_checks(
            [
                {"check_id": "a", "label": "State registered office", "kind": "required"},
                {"check_id": "b", "label": "State registered office", "kind": "required"},
                {"label": "State company number", "kind": "supporting"},
            ],
            max_checks=5,
        )
        self.assertEqual(len(checks), 2)
        self.assertEqual(checks[0]["check_id"], "a")
        self.assertEqual(checks[0]["kind"], "required")
        self.assertEqual(checks[1]["check_id"], "c3")
        self.assertEqual(checks[1]["kind"], "supporting")

    def test_filters_generic_checks(self) -> None:
        checks = _filter_low_signal_checks(
            [
                {"check_id": "c1", "label": "Verify the presence of statement", "kind": "required"},
                {"check_id": "c2", "label": "Members have not required audit under s476", "kind": "required"},
                {"check_id": "c3", "label": "Ensure the correct financial year", "kind": "required"},
            ]
        )
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["check_id"], "c2")


if __name__ == "__main__":
    unittest.main()

