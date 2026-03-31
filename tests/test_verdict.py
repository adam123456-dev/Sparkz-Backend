import unittest

from app.evaluation.verdict import parse_judge_response


class TestVerdict(unittest.TestCase):
    def test_valid_json_status_only(self) -> None:
        v = parse_judge_response(
            '{"status":"FULLY"}',
            explanation_max_chars=200,
        )
        self.assertEqual(v.status, "fully_met")
        self.assertIsNone(v.reason)
        self.assertIsNone(v.confidence)

    def test_partial_and_none(self) -> None:
        self.assertEqual(parse_judge_response('{"status":"PARTIAL"}').status, "partially_met")
        self.assertEqual(parse_judge_response('{"status":"NONE"}').status, "missing")

    def test_reason_truncation(self) -> None:
        long_reason = "x" * 500
        raw = f'{{"status":"FULLY","reason":"{long_reason}","confidence":0.9}}'
        v = parse_judge_response(raw, explanation_max_chars=50)
        self.assertEqual(v.status, "fully_met")
        self.assertIsNotNone(v.reason)
        self.assertLessEqual(len(v.reason), 51)

    def test_confidence_clamped(self) -> None:
        v = parse_judge_response('{"status":"FULLY","confidence":1.5}')
        self.assertEqual(v.confidence, 1.0)
        v2 = parse_judge_response('{"status":"PARTIAL","confidence":-2}')
        self.assertEqual(v2.confidence, 0.0)

    def test_invalid(self) -> None:
        v0 = parse_judge_response("")
        self.assertIsNone(v0.status)
        self.assertIsNone(v0.reason)
        self.assertIsNone(v0.confidence)
        self.assertIsNone(parse_judge_response("FULLY").status)
        self.assertIsNone(parse_judge_response('{"status":"MAYBE"}').status)


if __name__ == "__main__":
    unittest.main()
