import unittest

from app.core.requirement_order import requirement_id_sort_key


class RequirementOrderTests(unittest.TestCase):
    def test_orders_numeric_and_clauses(self) -> None:
        ids = ["6.01(b)", "1.01", "6.01(a)", "6.01(a)(i)", "6.01(a)(ii)", "4.01"]
        ordered = sorted(ids, key=requirement_id_sort_key)
        self.assertEqual(
            ordered,
            ["1.01", "4.01", "6.01(a)", "6.01(a)(i)", "6.01(a)(ii)", "6.01(b)"],
        )

    def test_orders_segments_numerically(self) -> None:
        self.assertLess(
            requirement_id_sort_key("1.01"),
            requirement_id_sort_key("6.01"),
        )
        self.assertLess(
            requirement_id_sort_key("6.01"),
            requirement_id_sort_key("6.02"),
        )


if __name__ == "__main__":
    unittest.main()
