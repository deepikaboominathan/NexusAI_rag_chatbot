
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NexusAI API"
    app_version: str = "1.0.0"
    environment: str = "development"

    # Database settings
    database_url: str

    # API authentication
    nexusai_api_token: str

    # Hugging Face settings (kept for now)
    hf_token: str | None = None
    hf_model: str = "openai/gpt-oss-120b:fastest"

    # Gemini settings
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    # Environment configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
