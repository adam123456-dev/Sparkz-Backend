import unittest

try:
    from app.core.config import Settings
except ModuleNotFoundError:  # pragma: no cover - dependency not installed yet
    Settings = None  # type: ignore[assignment]


class SettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        if Settings is None:
            self.skipTest("pydantic dependencies are not installed yet")

    def test_cors_parses_comma_separated_values(self) -> None:
        settings = Settings(
            APP_CORS_ORIGINS="http://localhost:5173,https://sparkz.app",
            SUPABASE_URL="https://example.supabase.co",
            SUPABASE_SERVICE_ROLE_KEY="service-role-key",
        )
        self.assertEqual(
            settings.cors_origins,
            ["http://localhost:5173", "https://sparkz.app"],
        )

    def test_validate_external_services_raises_when_missing(self) -> None:
        settings = Settings()
        with self.assertRaises(RuntimeError):
            settings.validate_external_services()


if __name__ == "__main__":
    unittest.main()

