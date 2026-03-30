from __future__ import annotations

from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    app_name: str = "Sparkz Backend"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"
    app_cors_origins: str = "http://localhost:5173"

    supabase_url: str = ""
    supabase_service_role_key: str = ""

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    evaluation_top_k: int = 3
    evaluation_evidence_max_chars: int = 2800
    evaluation_requirement_max_chars: int = 1200
    evaluation_explanation_max_chars: int = 320
    evaluation_keyword_prefilter: bool = True
    enable_ocr: bool = False
    upload_dir: str = "./tmp/uploads"

    @field_validator("app_log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper().strip()
        if normalized not in allowed:
            raise ValueError(f"Invalid APP_LOG_LEVEL '{value}'. Allowed: {sorted(allowed)}")
        return normalized

    @field_validator("app_cors_origins")
    @classmethod
    def validate_cors_string(cls, value: str) -> str:
        if not value.strip():
            return "http://localhost:5173"
        return value

    @field_validator("evaluation_top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        return max(1, int(value))

    @field_validator("evaluation_evidence_max_chars", "evaluation_requirement_max_chars")
    @classmethod
    def validate_prompt_caps(cls, value: int) -> int:
        return max(256, int(value))

    @field_validator("evaluation_keyword_prefilter", mode="before")
    @classmethod
    def coerce_keyword_prefilter(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v == "":
                return True
            if v in ("0", "false", "no", "off"):
                return False
            if v in ("1", "true", "yes", "on"):
                return True
        return bool(value)

    @field_validator("evaluation_explanation_max_chars")
    @classmethod
    def validate_explanation_cap(cls, value: int) -> int:
        return max(64, int(value))

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.app_cors_origins.split(",") if item.strip()]

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"

    def validate_external_services(self) -> None:
        missing = []
        if not self.supabase_url:
            missing.append("SUPABASE_URL")
        if not self.supabase_service_role_key:
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if missing:
            missing_text = ", ".join(missing)
            raise RuntimeError(f"Missing required environment variables: {missing_text}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

