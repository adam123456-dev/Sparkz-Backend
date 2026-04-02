from dataclasses import dataclass


@dataclass(slots=True)
class RedactedChunk:
    chunk_index: int
    page_number: int
    text_redacted: str
    text_hash: str
    heading_guess: str = ""
    section_title: str = ""
    statement_area: str = ""
    chunk_type: str = ""
    note_number: str = ""

    @property
    def embedding_text(self) -> str:
        parts = [
            f"Statement area: {self.statement_area}" if self.statement_area else "",
            f"Section title: {self.section_title}" if self.section_title else "",
            f"Note number: {self.note_number}" if self.note_number else "",
            f"Chunk type: {self.chunk_type}" if self.chunk_type else "",
            f"Heading hint: {self.heading_guess}" if self.heading_guess and self.heading_guess != self.section_title else "",
            f"Content: {self.text_redacted}",
        ]
        return "\n".join(part for part in parts if part).strip()

