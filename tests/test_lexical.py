import unittest

from app.evaluation.lexical import build_inverted_index, candidate_indices_for_keywords


class LexicalIndexTests(unittest.TestCase):
    def test_union_candidates(self) -> None:
        texts = [
            "micro entity revenue note",
            "balance sheet assets",
            "revenue recognition policy",
        ]
        inv = build_inverted_index(texts)
        cand = candidate_indices_for_keywords(["revenue", "missingtoken"], inv)
        self.assertEqual(cand, {0, 2})

    def test_empty_keywords(self) -> None:
        inv = build_inverted_index(["a b c"])
        self.assertEqual(candidate_indices_for_keywords([], inv), set())


if __name__ == "__main__":
    unittest.main()
