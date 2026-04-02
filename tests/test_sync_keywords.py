import unittest

from scripts.sync_checklists_to_supabase import _keywords_from_rule_checks


class SyncKeywordTests(unittest.TestCase):
    def test_filters_generic_reporting_keywords(self) -> None:
        checks = [
            {"check_id": "c1", "label": "Date at end of reporting period is stated", "kind": "required"},
            {"check_id": "c2", "label": "Financial statements covered are stated", "kind": "required"},
        ]
        keywords = _keywords_from_rule_checks(checks)
        self.assertNotIn("date", keywords)
        self.assertNotIn("period", keywords)
        self.assertNotIn("financial", keywords)
        self.assertNotIn("statements", keywords)
        self.assertEqual(keywords, [])

    def test_keeps_useful_multiword_phrase(self) -> None:
        checks = [
            {"check_id": "c1", "label": "Revenue recognition accounting policy is disclosed", "kind": "required"},
        ]
        keywords = _keywords_from_rule_checks(checks)
        self.assertIn("revenue recognition accounting policy", keywords)
        self.assertIn("revenue", keywords)
        self.assertIn("recognition", keywords)
        self.assertIn("accounting", keywords)
        self.assertIn("policy", keywords)


if __name__ == "__main__":
    unittest.main()
