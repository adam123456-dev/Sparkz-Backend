import unittest

from app.pipeline.pii import redact_pii, redact_pii_with_audit


class PiiTests(unittest.TestCase):
    def test_preserves_financial_amounts_and_company_numbers(self) -> None:
        text = (
            "Company registration number: 12345678\n"
            "Turnover 422,560 383,200\n"
            "Tax (1,500) (100)\n"
            "Net assets 238,858 204,538"
        )
        redacted = redact_pii(text)
        self.assertIn("12345678", redacted)
        self.assertIn("422,560", redacted)
        self.assertIn("(1,500)", redacted)
        self.assertNotIn("[PHONE_", redacted)

    def test_redacts_explicit_contact_field_only(self) -> None:
        text = "Telephone: +44 20 1234 5678\nEmail: person@example.com"
        redacted, audits = redact_pii_with_audit(text)
        self.assertIn("Telephone: [PHONE_1]", redacted)
        self.assertIn("Email: [EMAIL_1]", redacted)
        self.assertEqual({entry.entity_type for entry in audits}, {"phone", "email"})


if __name__ == "__main__":
    unittest.main()
