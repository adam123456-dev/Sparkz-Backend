import unittest

from app.evaluation.retrieval_rerank import (
    final_rank_score,
    heading_match_score,
    keyword_overlap_score,
    section_hint_score,
)


class RetrievalRerankTests(unittest.TestCase):
    def test_keyword_overlap_score_counts_phrase_token_coverage(self) -> None:
        score = keyword_overlap_score(
            ["revenue", "revenue recognition", "accounting policy"],
            {"revenue", "recognition", "policy"},
        )
        self.assertAlmostEqual(score, 2 / 3)

    def test_heading_match_score_prefers_heading_tokens(self) -> None:
        score = heading_match_score(
            ["revenue", "recognition", "turnover"],
            "Accounting policies revenue recognition",
        )
        self.assertAlmostEqual(score, 2 / 3)

    def test_final_rank_score_can_beat_raw_semantic_with_better_matches(self) -> None:
        weak_semantic = final_rank_score(
            semantic_similarity=0.76,
            keyword_overlap=1.0,
            heading_match=1.0,
        )
        strong_semantic_only = final_rank_score(
            semantic_similarity=0.88,
            keyword_overlap=0.0,
            heading_match=0.0,
        )
        self.assertGreater(weak_semantic, strong_semantic_only)

    def test_section_hint_score_uses_heading_and_text(self) -> None:
        score = section_hint_score(
            ["notes", "directors", "related party"],
            "Directors and related party note",
            "Details of directors' transactions are shown in the notes.",
        )
        self.assertGreaterEqual(score, 2 / 3)


if __name__ == "__main__":
    unittest.main()
