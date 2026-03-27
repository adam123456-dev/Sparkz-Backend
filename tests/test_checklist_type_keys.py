import unittest
from pathlib import Path

from app.core.checklist_type_keys import (
    resolve_framework_form_value,
    type_key_from_workbook_path,
)


class ChecklistTypeKeysTests(unittest.TestCase):
    def test_resolve_ui_labels(self) -> None:
        self.assertEqual(resolve_framework_form_value("IFRS"), "ifrs")
        self.assertEqual(resolve_framework_form_value("FRS 102"), "frs102")
        self.assertEqual(resolve_framework_form_value("FRS 105"), "frs105")

    def test_resolve_canonical_keys(self) -> None:
        self.assertEqual(resolve_framework_form_value("ifrs"), "ifrs")
        self.assertEqual(resolve_framework_form_value("frs102"), "frs102")

    def test_workbook_mapping(self) -> None:
        self.assertEqual(type_key_from_workbook_path("IFRS_DC_2025.xlsx"), "ifrs")
        self.assertEqual(type_key_from_workbook_path("FRS1021A_DC_2025.xlsx"), "frs102")
        self.assertEqual(type_key_from_workbook_path(Path("FRS105_DC_2025.xlsx")), "frs105")

    def test_unknown_framework_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_framework_form_value("US GAAP")


if __name__ == "__main__":
    unittest.main()
