from dataclasses import dataclass


@dataclass(slots=True)
class ChecklistItem:
    source_workbook: str
    framework: str
    sheet_name: str
    section_path: str
    requirement_id: str
    requirement_text: str
    reference_text: str
    requirement_base_id: str = ""
    clause_path: str = ""
    requirement_text_leaf: str = ""
    notes_text: str = ""
    item_kind: str = "rule"

    @property
    def embedding_text(self) -> str:
        parts = [
            f"Framework: {self.framework}",
            f"Sheet: {self.sheet_name}",
            f"Section: {self.section_path}" if self.section_path else "",
            f"Requirement ID: {self.requirement_id}",
            f"Base Requirement ID: {self.requirement_base_id}" if self.requirement_base_id else "",
            f"Clause Path: {self.clause_path}" if self.clause_path else "",
            f"Requirement (full): {self.requirement_text}",
            f"Requirement (leaf): {self.requirement_text_leaf}"
            if self.requirement_text_leaf and self.requirement_text_leaf != self.requirement_text
            else "",
            f"Notes: {self.notes_text}" if self.notes_text else "",
            f"Reference: {self.reference_text}" if self.reference_text else "",
        ]
        return "\n".join(part for part in parts if part).strip()

