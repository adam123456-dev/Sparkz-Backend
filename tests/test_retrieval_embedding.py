import unittest

from app.checklists.retrieval_embedding import retrieval_embedding_source_text


class RetrievalEmbeddingTextTests(unittest.TestCase):
    def test_prefers_leaf(self) -> None:
        self.assertEqual(
            retrieval_embedding_source_text(
                requirement_text="The accounts must state: the registered office;",
                requirement_text_leaf="the registered office address;",
            ),
            "Requirement context: The accounts must state: the registered office;\nFocus: the registered office address;",
        )

    def test_falls_back_to_full(self) -> None:
        self.assertEqual(
            retrieval_embedding_source_text(
                requirement_text="Full requirement only.",
                requirement_text_leaf="",
            ),
            "Full requirement only.",
        )

    def test_keeps_short_leaf_with_parent_context(self) -> None:
        self.assertEqual(
            retrieval_embedding_source_text(
                requirement_text="Statement of financial position format 1 must include fixed assets.",
                requirement_text_leaf="Fixed assets;",
            ),
            "Requirement context: Statement of financial position format 1 must include fixed assets.\nFocus: Fixed assets;",
        )

    def test_prefers_specific_leaf_when_substantive(self) -> None:
        self.assertEqual(
            retrieval_embedding_source_text(
                requirement_text="The entity must disclose the accounting policy for revenue recognition.",
                requirement_text_leaf="disclose the accounting policy for revenue recognition",
            ),
            "disclose the accounting policy for revenue recognition",
        )


if __name__ == "__main__":
    unittest.main()
