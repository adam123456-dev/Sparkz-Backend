import unittest

import numpy as np

from app.evaluation.embedding_vector import embedding_to_float_vector


class TestEmbeddingParse(unittest.TestCase):
    def test_json_string_array(self) -> None:
        raw = "[-0.5,0.25,1.0]"
        v = embedding_to_float_vector(raw)
        self.assertEqual(v.dtype, np.float32)
        np.testing.assert_array_almost_equal(v, np.array([-0.5, 0.25, 1.0], dtype=np.float32))

    def test_list(self) -> None:
        v = embedding_to_float_vector([0.0, 2.0])
        self.assertEqual(v.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
