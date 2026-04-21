"""Runtime configuration loaded from environment variables.

Single source of truth for which provider/model to use, where credentials
live, etc. Kept deliberately small — ``.env.example`` documents every setting.

Provider keys are read opportunistically: you only need the key for the
provider you actually use. The comparison eval auto-detects which keys are
set and only runs those providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Root of the project (two levels up from this file: src/sift/config.py -> <root>)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # Which provider to use by default ("anthropic", "openai", "google", "groq").
    llm_provider: str

    # Per-provider API keys. Empty string means "not configured".
    anthropic_api_key: str
    openai_api_key: str
    google_api_key: str
    groq_api_key: str

    # Optional explicit model override. If unset each provider falls back to its
    # own ``default_model`` attribute, so you don't need to care which provider
    # is active when picking a model.
    model: str | None

    # Gmail + local DB paths.
    google_credentials_path: Path
    google_token_path: Path
    db_path: Path

    @classmethod
    def from_env(cls) -> "Config":
        # Empty model string from env means "use the provider default".
        model_env = os.getenv("SIFT_MODEL", "").strip()
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "anthropic").lower(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            model=model_env or None,
            google_credentials_path=Path(os.getenv("GOOGLE_CREDENTIALS_PATH", "./credentials.json")),
            google_token_path=Path(os.getenv("GOOGLE_TOKEN_PATH", "./token.json")),
            db_path=Path(os.getenv("SIFT_DB", "./sift.db")),
        )


CONFIG = Config.from_env()
