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

    @property
    def embedding_text(self) -> str:
        parts = [
            f"Framework: {self.framework}",
            f"Sheet: {self.sheet_name}",
            f"Section: {self.section_path}" if self.section_path else "",
            f"Requirement ID: {self.requirement_id}",
            f"Requirement: {self.requirement_text}",
            f"Reference: {self.reference_text}" if self.reference_text else "",
        ]
        return "\n".join(part for part in parts if part).strip()

