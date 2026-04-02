import unittest

from app.checklists.models import ChecklistItem
from scripts.sync_checklists_to_supabase import (
    _build_rule_checks_for_item,
    _fallback_keywords_for_item,
    _fallback_section_hints_for_item,
)


class SyncExtractionLogicTests(unittest.TestCase):
    def test_atomic_leaf_row_uses_single_deterministic_rule_check(self) -> None:
        item = ChecklistItem(
            source_workbook="FRS105.xlsx",
            framework="FRS105",
            sheet_name="Micro DC",
            section_path="",
            requirement_id="5.01(a)(iv)",
            requirement_text=(
                "Details of advances and credits granted by a micro-entity to its directors "
                "must be shown in the notes to the financial statements. any amounts repaid;"
            ),
            reference_text="s472(1A)",
            requirement_base_id="5.01",
            clause_path="(a)(iv)",
            requirement_text_leaf="any amounts repaid;",
        )
        settings = type("Settings", (), {"openai_api_key": "", "openai_chat_model": "gpt-4o-mini"})()
        checks = _build_rule_checks_for_item(item, settings)
        self.assertEqual(
            checks,
            [{"check_id": "c1", "label": "amounts repaid for advances and credits to directors", "kind": "required"}],
        )

    def test_fallback_keywords_prefer_atomic_leaf_scope(self) -> None:
        item = ChecklistItem(
            source_workbook="FRS105.xlsx",
            framework="FRS105",
            sheet_name="Micro DC",
            section_path="",
            requirement_id="5.01(a)(iv)",
            requirement_text=(
                "Details of advances and credits granted by a micro-entity to its directors "
                "must be shown in the notes to the financial statements. any amounts repaid;"
            ),
            reference_text="s472(1A)",
            requirement_base_id="5.01",
            clause_path="(a)(iv)",
            requirement_text_leaf="any amounts repaid;",
        )
        keywords = _fallback_keywords_for_item(item)
        self.assertIn("amounts repaid for advances and credits to directors", keywords)
        self.assertNotIn("financial statements", keywords)
        self.assertNotIn("details", keywords)

    def test_resolves_pronoun_leaf_with_parent_context(self) -> None:
        item = ChecklistItem(
            source_workbook="FRS105.xlsx",
            framework="FRS105",
            sheet_name="Micro DC",
            section_path="",
            requirement_id="5.01(a)(iii)",
            requirement_text=(
                "Details of advances and credits granted by a micro-entity to its directors and guarantees "
                "of any kind entered into by a micro-entity on behalf of its directors must be shown in the "
                "notes to the financial statements. its main conditions;"
            ),
            reference_text="s472(1A)",
            requirement_base_id="5.01",
            clause_path="(a)(iii)",
            requirement_text_leaf="its main conditions;",
        )
        settings = type("Settings", (), {"openai_api_key": "", "openai_chat_model": "gpt-4o-mini"})()
        checks = _build_rule_checks_for_item(item, settings)
        self.assertEqual(
            checks,
            [{"check_id": "c1", "label": "main conditions of advances and credits to directors", "kind": "required"}],
        )
        keywords = _fallback_keywords_for_item(item)
        self.assertIn("main conditions of advances and credits to directors", keywords)
        self.assertNotIn("guarantees", keywords)

    def test_fallback_section_hints_not_empty_for_note_style_rule(self) -> None:
        item = ChecklistItem(
            source_workbook="FRS105.xlsx",
            framework="FRS105",
            sheet_name="Micro DC",
            section_path="",
            requirement_id="5.01",
            requirement_text="Details must be shown in the notes to the financial statements for directors' advances.",
            reference_text="s472(1A)",
            requirement_base_id="5.01",
            clause_path="",
            requirement_text_leaf="",
        )
        hints = _fallback_section_hints_for_item(item)
        self.assertIn("notes", hints)
        self.assertIn("directors", hints)


if __name__ == "__main__":
    unittest.main()
