from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://ahody:ahody@localhost:5432/ahody"
    voyage_api_key: str = ""
    anthropic_api_key: str = ""

    embedding_model: str = "voyage-4"

    # DECISIONS.md §18.1 — Anthropic cookbook split: Haiku for high-volume
    # schema-constrained extraction, Sonnet for resolution arbitrating
    # conflicting evidence. Aliases (not dated snapshots) per §9 convention.
    extraction_model: str = "claude-haiku-4-5"
    resolution_model: str = "claude-sonnet-4-6"


settings = Settings()
