import json
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str = ""          # override for OpenRouter / other providers
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.2
    openai_timeout: int = 30
    openai_use_json_schema: bool = True  # set False for OpenRouter / non-OpenAI providers
    cors_origins: list[str] = ["http://localhost:5173"]

    # Live data ingestion
    fred_api_key: str = ""                    # required for FRED — get free key at fred.stlouisfed.org
    live_cache_ttl_seconds: int = 900         # 15 min default
    live_allow_stale_cache: bool = True       # serve stale on provider error
    http_timeout_seconds: int = 20

    # Valuation provider
    fmp_api_key: str = ""                     # Financial Modeling Prep — for Mag 7 Forward P/E basket
    valuation_provider: str = "fmp"           # "fmp" (primary) | "yahoo" (fallback / override)

    # Policy support heuristic defaults (used when live political signals aren't API-sourced)
    default_fed_put: bool = False
    default_treasury_put: bool = False
    default_political_put: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return json.loads(v)
        return v  # type: ignore[return-value]


settings = Settings()
