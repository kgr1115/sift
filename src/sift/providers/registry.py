"""Provider registry + factory.

Usage:
    from sift.providers import get_provider
    provider = get_provider("anthropic")  # or None for the default

The default is controlled by the ``LLM_PROVIDER`` env var and falls back to
"anthropic". Providers are lazily instantiated — no API client is constructed
until you actually call into the provider.
"""

from __future__ import annotations

import os
from functools import lru_cache

from .base import LLMProvider


class ProviderRegistry:
    """Name -> provider class. Instantiated lazily."""

    def __init__(self) -> None:
        self._classes: dict[str, type[LLMProvider]] = {}

    def register(self, name: str, cls: type[LLMProvider]) -> None:
        self._classes[name] = cls

    def names(self) -> list[str]:
        return sorted(self._classes)

    def create(self, name: str, *, model: str | None = None) -> LLMProvider:
        if name not in self._classes:
            raise KeyError(
                f"Unknown provider {name!r}. Known: {', '.join(self.names()) or '(none)'}"
            )
        return self._classes[name](model=model)


REGISTRY = ProviderRegistry()


# -------- Registration -------- #
# We register classes here rather than with decorators so that the registry
# has an explicit list of every supported provider in one place.
from .anthropic import AnthropicProvider  # noqa: E402
from .google import GoogleProvider  # noqa: E402
from .openai_compat import GroqProvider, OpenAIProvider  # noqa: E402

REGISTRY.register("anthropic", AnthropicProvider)
REGISTRY.register("openai", OpenAIProvider)
REGISTRY.register("google", GoogleProvider)
REGISTRY.register("groq", GroqProvider)


# -------- Public helpers -------- #

def list_providers() -> list[str]:
    return REGISTRY.names()


@lru_cache(maxsize=8)
def get_provider(name: str, model: str | None = None) -> LLMProvider:
    return REGISTRY.create(name, model=model)


def get_default_provider() -> LLMProvider:
    """Return the provider selected by ``LLM_PROVIDER`` env var (default: anthropic).

    Also honors ``SIFT_MODEL`` if set — blank means "use the provider
    default model".
    """
    name = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model_env = os.getenv("SIFT_MODEL", "").strip() or None
    return get_provider(name, model_env)


def list_available_providers() -> list[str]:
    """Return providers whose API key is actually set in the environment.

    Used by the comparison eval to auto-detect which providers to score.
    """
    available: list[str] = []
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    for name, env_var in env_map.items():
        if os.getenv(env_var):
            available.append(name)
    return available
