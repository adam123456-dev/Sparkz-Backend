import unittest

from app.checklists.retrieval_embedding import retrieval_embedding_source_text


class RetrievalEmbeddingTextTests(unittest.TestCase):
    def test_prefers_leaf(self) -> None:
        self.assertEqual(
            retrieval_embedding_source_text(
                requirement_text="The accounts must state: the registered office;",
                requirement_text_leaf="the registered office address;",
            ),
            "the registered office address;",
        )

    def test_falls_back_to_full(self) -> None:
        self.assertEqual(
            retrieval_embedding_source_text(
                requirement_text="Full requirement only.",
                requirement_text_leaf="",
            ),
            "Full requirement only.",
        )


if __name__ == "__main__":
    unittest.main()
