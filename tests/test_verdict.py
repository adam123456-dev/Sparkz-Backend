import unittest

from app.evaluation.verdict import parse_judge_response


class TestVerdict(unittest.TestCase):
    def test_valid_json(self) -> None:
        s, w = parse_judge_response(
            '{"status":"FULLY"}',
            explanation_max_chars=200,
        )
        self.assertEqual(s, "fully_met")
        self.assertIsNone(w)

    def test_partial_and_none(self) -> None:
        self.assertEqual(
            parse_judge_response('{"status":"PARTIAL"}')[0],
            "partially_met",
        )
        self.assertEqual(parse_judge_response('{"status":"NONE"}')[0], "missing")

    def test_invalid(self) -> None:
        self.assertEqual(parse_judge_response(""), (None, None))
        self.assertEqual(parse_judge_response("FULLY"), (None, None))
        self.assertEqual(parse_judge_response('{"status":"MAYBE"}'), (None, None))


if __name__ == "__main__":
    unittest.main()
