from pathlib import Path
import unittest

from app.checklists import parse_workbook
from app.checklists.parser import ChecklistWorkbookParser, RawRow


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ChecklistParserTests(unittest.TestCase):
    def test_normalizes_requirement_id_float_artifact(self) -> None:
        parser = ChecklistWorkbookParser()
        self.assertEqual(
            parser._normalize_requirement_id_candidate("4.0199999999999996"),
            "4.02",
        )
        self.assertEqual(parser._normalize_requirement_id_candidate("4.02"), "4.02")

    def test_pads_excel_numeric_id_6_10(self) -> None:
        parser = ChecklistWorkbookParser()
        self.assertEqual(parser._pad_id_decimal_segments("6.1"), "6.10")
        self.assertEqual(parser._pad_id_decimal_segments("6.10"), "6.10")
        self.assertEqual(parser._pad_id_decimal_segments("5.01"), "5.01")

    def test_extracts_sibling_alpha_clauses_6_01(self) -> None:
        parser = ChecklistWorkbookParser()
        rows = [
            RawRow(
                first="6.01",
                second="Disclose, in a note at the foot of the balance sheet, the total amount of each of the following that are not included in the balance sheet:",
                third="",
                all_cells=("6.01", "Disclose...", ""),
            ),
            RawRow(first="(a)", second="financial commitments;", third="", all_cells=("(a)", "financial commitments;", "")),
            RawRow(first="(b)", second="guarantees;", third="", all_cells=("(b)", "guarantees;", "")),
            RawRow(first="(c)", second="contingencies", third="", all_cells=("(c)", "contingencies", "")),
        ]
        items = parser._extract_items_from_rows(
            workbook_name="x.xlsx",
            framework="FRS105",
            sheet_name="Sheet1",
            rows=rows,
        )
        ids = [it.requirement_id for it in items]
        self.assertEqual(
            set(ids),
            {"6.01(a)", "6.01(b)", "6.01(c)"},
        )
        a = next(it for it in items if it.requirement_id == "6.01(a)")
        self.assertIn("financial commitments", a.requirement_text.lower())
        self.assertIn("disclose", a.requirement_text.lower())

    def _require_file(self, path: Path) -> None:
        if not path.exists():
            self.skipTest(f"Workbook not found for test: {path}")

    def test_extracts_atomic_subclauses_with_context(self) -> None:
        parser = ChecklistWorkbookParser()
        rows = [
            RawRow(
                first="6.01",
                second="Details of advances and credits granted by a micro-entity...",
                third="",
                all_cells=("6.01", "Details of advances and credits granted by a micro-entity...", ""),
            ),
            RawRow(first="(a)", second="The details required of an advance or credit are:", third="", all_cells=("(a)", "The details required of an advance or credit are:", "")),
            RawRow(first="(i)", second="its amount", third="", all_cells=("(i)", "its amount", "")),
            RawRow(first="(ii)", second="any indication of the interest rate", third="", all_cells=("(ii)", "any indication of the interest rate", "")),
            RawRow(first="Note 1", second="This is guidance only.", third="", all_cells=("Note 1", "This is guidance only.", "")),
        ]
        items = parser._extract_items_from_rows(
            workbook_name="x.xlsx",
            framework="FRS105",
            sheet_name="Sheet1",
            rows=rows,
        )
        ids = [it.requirement_id for it in items]
        self.assertIn("6.01(a)(i)", ids)
        self.assertIn("6.01(a)(ii)", ids)
        self.assertNotIn("6.01", ids)
        first = next(it for it in items if it.requirement_id == "6.01(a)(i)")
        self.assertIn("The details required of an advance or credit are", first.requirement_text)
        self.assertIn("its amount", first.requirement_text)

    def test_splits_plain_list_lines_under_alpha_into_atomic_rules(self) -> None:
        parser = ChecklistWorkbookParser()
        rows = [
            RawRow(
                first="3.01",
                second="A micro-entity must present:",
                third="",
                all_cells=("3.01", "A micro-entity must present:", ""),
            ),
            RawRow(
                first="(a)",
                second="Statement of financial position format 1.",
                third="",
                all_cells=("(a)", "Statement of financial position format 1.", ""),
            ),
            RawRow(first="", second="Called up share capital not paid;", third="", all_cells=("", "Called up share capital not paid;", "")),
            RawRow(first="", second="Fixed assets;", third="", all_cells=("", "Fixed assets;", "")),
            RawRow(first="", second="Capital and reserves.", third="", all_cells=("", "Capital and reserves.", "")),
        ]
        items = parser._extract_items_from_rows(
            workbook_name="x.xlsx",
            framework="FRS105",
            sheet_name="Sheet1",
            rows=rows,
        )
        ids = [it.requirement_id for it in items]
        self.assertEqual(ids, ["3.01(a)(1)", "3.01(a)(2)", "3.01(a)(3)"])
        self.assertIn("Called up share capital not paid", items[0].requirement_text)
        self.assertIn("Fixed assets", items[1].requirement_text)
        self.assertIn("Capital and reserves", items[2].requirement_text)

    def test_ignores_guidance_rows_below_alpha_rule(self) -> None:
        parser = ChecklistWorkbookParser()
        rows = [
            RawRow(
                first="4.01",
                second="The financial statements of a micro-entity shall include:",
                third="",
                all_cells=("4.01", "The financial statements of a micro-entity shall include:", ""),
            ),
            RawRow(
                first="(b)",
                second="an income statement for the reporting period.",
                third="",
                all_cells=("(b)", "an income statement for the reporting period.", ""),
            ),
            RawRow(
                first="",
                second="A micro-entity may use titles for the financial statements other than those used in FRS 105 as long as they are not misleading.",
                third="",
                all_cells=("", "A micro-entity may use titles for the financial statements other than those used in FRS 105 as long as they are not misleading.", ""),
            ),
            RawRow(
                first="",
                second="[Guidance] Table of Equivalence for Company Law terminology",
                third="",
                all_cells=("", "[Guidance] Table of Equivalence for Company Law terminology", ""),
            ),
            RawRow(
                first="",
                second='"Per the above, the standard allows for the usage of other terminology."',
                third="",
                all_cells=("", '"Per the above, the standard allows for the usage of other terminology."', ""),
            ),
        ]
        items = parser._extract_items_from_rows(
            workbook_name="x.xlsx",
            framework="FRS105",
            sheet_name="Sheet1",
            rows=rows,
        )
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.requirement_id, "4.01(b)")
        self.assertEqual(item.requirement_text, "The financial statements of a micro-entity shall include: an income statement for the reporting period.")
        self.assertNotIn("Table of Equivalence", item.requirement_text)
        self.assertNotIn("other terminology", item.requirement_text)

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