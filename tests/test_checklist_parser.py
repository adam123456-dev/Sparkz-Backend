from pathlib import Path
import unittest

from app.checklists import parse_workbook
from app.checklists.parser import ChecklistWorkbookParser


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ChecklistParserTests(unittest.TestCase):
    def test_normalizes_requirement_id_float_artifact(self) -> None:
        parser = ChecklistWorkbookParser()
        self.assertEqual(
            parser._normalize_requirement_id_candidate("4.0199999999999996"),
            "4.02",
        )
        self.assertEqual(parser._normalize_requirement_id_candidate("4.02"), "4.02")

    def _require_file(self, path: Path) -> None:
        if not path.exists():
            self.skipTest(f"Workbook not found for test: {path}")

    def test_parses_frs102_1a_workbook(self) -> None:
        workbook = PROJECT_ROOT / "Disclosure Checklists" / "FRS1021A_DC_2025.xlsx"
        self._require_file(workbook)
        items = parse_workbook(workbook)

        self.assertGreater(len(items), 120)
        ids = {item.requirement_id for item in items}
        self.assertIn("1.01", ids)
        self.assertIn("A2.01", ids)
        self.assertIn("A3.01", ids)

        first = next(item for item in items if item.requirement_id == "1.01")
        self.assertEqual(first.framework, "FRS102_1A")
        self.assertTrue(first.requirement_text.lower().startswith("the directors"))
        self.assertTrue(first.reference_text.lower().startswith("s416"))

    def test_parses_charities_workbook(self) -> None:
        workbook = PROJECT_ROOT / "drive-download-20260326T003740Z-3-001" / "Charities_FRS102_DC_2025.xlsx"
        self._require_file(workbook)
        items = parse_workbook(workbook)

        self.assertGreater(len(items), 450)
        ids = {item.requirement_id for item in items}
        self.assertIn("1.01", ids)
        self.assertIn("A3.01", ids)
        self.assertIn("A14.01", ids)

        charity_item = next(item for item in items if item.requirement_id == "A14.01")
        self.assertEqual(charity_item.framework, "CHARITIES_FRS102")
        self.assertIn("service concession arrangements", charity_item.requirement_text.lower())

    def test_builds_embedding_text(self) -> None:
        workbook = PROJECT_ROOT / "Disclosure Checklists" / "FRS1021A_DC_2025.xlsx"
        self._require_file(workbook)
        items = parse_workbook(workbook)
        sample = items[0]
        text = sample.embedding_text
        self.assertIn("Framework:", text)
        self.assertIn("Requirement ID:", text)
        self.assertIn("Requirement:", text)


if __name__ == "__main__":
    unittest.main()

