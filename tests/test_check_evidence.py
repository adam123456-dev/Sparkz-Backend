import unittest
from dataclasses import dataclass

from app.evaluation.check_evidence import select_chunks_for_check, select_evidence_for_check


@dataclass
class FakeChunk:
    chunk_id: str
    page_number: int
    text_redacted: str
    heading_guess: str
    similarity: float
    section_title: str = ""
    statement_area: str = ""
    chunk_type: str = ""
    note_number: str = ""


class CheckEvidenceTests(unittest.TestCase):
    def test_selects_best_chunk_for_atomic_check(self) -> None:
        chunks = [
            FakeChunk(
                chunk_id="a",
                page_number=12,
                text_redacted="The average number of employees was 40.",
                heading_guess="Employees",
                similarity=0.78,
                section_title="Employees",
                statement_area="notes",
                chunk_type="narrative",
            ),
            FakeChunk(
                chunk_id="b",
                page_number=7,
                text_redacted="Revenue is recognised when control passes.",
                heading_guess="Revenue recognition",
                similarity=0.72,
                section_title="Revenue recognition",
                statement_area="notes",
                chunk_type="narrative",
            ),
        ]
        selected = select_chunks_for_check(
            check_label="average number of employees",
            chunks=chunks,
            max_chunks=1,
        )
        self.assertEqual([chunk.chunk_id for chunk in selected], ["a"])
        text = select_evidence_for_check(
            check_label="average number of employees",
            chunks=chunks,
            max_chunks=1,
            max_chars=220,
        )
        self.assertIn("employees", text.lower())
        self.assertNotIn("revenue is recognised", text.lower())


if __name__ == "__main__":
    unittest.main()
