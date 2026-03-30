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
        # ``model_validate`` avoids local .env overriding explicit test values.
        settings = Settings.model_validate(
            {
                "app_cors_origins": "http://localhost:5173,https://sparkz.app",
                "supabase_url": "https://example.supabase.co",
                "supabase_service_role_key": "service-role-key",
            }
        )
        self.assertEqual(
            settings.cors_origins,
            ["http://localhost:5173", "https://sparkz.app"],
        )

    def test_validate_external_services_raises_when_missing(self) -> None:
        settings = Settings.model_validate(
            {"supabase_url": "", "supabase_service_role_key": ""},
        )
        with self.assertRaises(RuntimeError):
            settings.validate_external_services()


if __name__ == "__main__":
    unittest.main()

