import unittest

from app.pipeline.chunking import _heading_guess, build_chunks_from_redacted_pages


class ChunkingTests(unittest.TestCase):
    def test_heading_guess_removes_numeric_noise(self) -> None:
        text = (
            "Group underlying operating profit 7.[PHONE_4] £m £m Group net assets "
            "Balance sheet 6,868 7,253 Less: Pension obligations and related tax balances."
        )
        heading = _heading_guess(text)
        self.assertIn("Group underlying operating profit", heading)
        self.assertNotIn("[PHONE_4]", heading)
        self.assertNotIn("6,868", heading)

    def test_heading_guess_trims_to_short_hint(self) -> None:
        text = (
            "Revenue recognition accounting policies for retail sales online orders "
            "gift cards and commissions apply across the group and related undertakings."
        )
        heading = _heading_guess(text)
        self.assertLessEqual(len(heading.split()), 12)

    def test_build_chunks_respects_headings(self) -> None:
        pages = [
            "Note 1 Accounting policies\nRevenue recognition\nRevenue is recognised when control passes.\n\n"
            "Note 2 Employees\nThe average number of employees was 40."
        ]
        chunks = build_chunks_from_redacted_pages(pages, chunk_word_target=40, chunk_word_overlap=5)
        headings = [chunk.heading_guess for chunk in chunks]
        self.assertTrue(any("Revenue recognition" in heading for heading in headings))
        self.assertTrue(any("Employees" in heading for heading in headings))
        self.assertTrue(any(chunk.statement_area == "notes" for chunk in chunks))
        self.assertTrue(any("Revenue recognition" in chunk.section_title for chunk in chunks))
        self.assertTrue(any(chunk.note_number == "1" for chunk in chunks))
        self.assertTrue(any("Statement area: notes" in chunk.embedding_text for chunk in chunks))
        self.assertTrue(any("Content: Revenue is recognised when control passes." in chunk.embedding_text for chunk in chunks))

    def test_drops_footer_page_number_chunks(self) -> None:
        pages = [
            "Demo Micro-entity (FRS 105) Limited\nOfficers and Professional Advisers\nYear ended 31 December 2016\n"
            "Directors Mrs Joan Micro\nRegistered office 16 Micro Street\n1"
        ]
        chunks = build_chunks_from_redacted_pages(pages)
        texts = [chunk.text_redacted for chunk in chunks]
        self.assertFalse(any(text == "1" for text in texts))

    def test_splits_notes_by_numbered_note_heading(self) -> None:
        pages = [
            "Statement of Financial Position\n31 December 2016\n"
            "NOTES TO THE FINANCIAL STATEMENTS\n"
            "1 General information\nThe company is a private company limited by shares.\n"
            "2 Guarantees and other financial commitments\nThe company had capital commitments of 8,000.\n"
            "3 Directors advances, credit and guarantees\nThe credit amount is interest free and will be repaid within 2 months.\n"
            "5"
        ]
        chunks = build_chunks_from_redacted_pages(pages)
        note_titles = [chunk.section_title for chunk in chunks if chunk.statement_area == "notes"]
        note_numbers = [chunk.note_number for chunk in chunks if chunk.statement_area == "notes"]
        self.assertTrue(any(title.startswith("1 General information") for title in note_titles))
        self.assertTrue(any(title.startswith("2 Guarantees and other financial commitments") for title in note_titles))
        self.assertTrue(any(title.startswith("3 Directors advances, credit and guarantees") for title in note_titles))
        self.assertEqual(sorted(set(note_numbers)), ["1", "2", "3"])

    def test_keeps_related_table_rows_together_in_larger_chunk(self) -> None:
        pages = [
            "Detailed Income Statement (continued)\nYear ended 31 December 2016\n"
            "STAFF COSTS\n2016 2015\n£ £\nCost of sales\nWages and salaries 50,000 20,000\n"
            "Employers NI 1,200 1,060\n51,200 21,060\nAdministrative costs\n"
            "Directors remuneration 50,000 50,000\nDirectors pension costs 5,000 5,000\n55,000 55,000\n9"
        ]
        chunks = build_chunks_from_redacted_pages(pages)
        combined = "\n".join(chunk.text_redacted for chunk in chunks)
        self.assertIn("Wages and salaries 50,000 20,000", combined)
        self.assertIn("Employers NI 1,200 1,060", combined)
        self.assertIn("Directors remuneration 50,000 50,000", combined)
        self.assertFalse(any(chunk.text_redacted == "9" for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
