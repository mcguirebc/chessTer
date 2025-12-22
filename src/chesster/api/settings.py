from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHESSTER_")

    idempotency_db_path: str = "data/idempotency.sqlite"

    stockfish_path: str | None = None
    stockfish_default_depth: int = 10

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_default_model: str = "gpt-oss-20b"
    ollama_timeout_s: float = 30.0




