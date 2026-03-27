import unittest

from app.evaluation.verdict import parse_judge_response


class TestVerdict(unittest.TestCase):
    def test_valid_json(self) -> None:
        s, w = parse_judge_response(
            '{"status":"FULLY","why":"The note states revenue recognition policy."}',
            explanation_max_chars=200,
        )
        self.assertEqual(s, "fully_met")
        self.assertIn("revenue", w or "")

    def test_partial_and_none(self) -> None:
        self.assertEqual(
            parse_judge_response('{"status":"PARTIAL","why":"x"}')[0],
            "partially_met",
        )
        self.assertEqual(parse_judge_response('{"status":"NONE","why":""}')[0], "missing")

    def test_invalid(self) -> None:
        self.assertEqual(parse_judge_response(""), (None, None))
        self.assertEqual(parse_judge_response("FULLY"), (None, None))
        self.assertEqual(parse_judge_response('{"status":"MAYBE"}'), (None, None))


if __name__ == "__main__":
    unittest.main()
