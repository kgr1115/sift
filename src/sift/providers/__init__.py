"""LLM provider implementations.

Each provider subclasses ``LLMProvider`` from base.py and registers itself via
the registry. The public entry points in ``sift.llm`` dispatch through
the registry — nothing outside this subpackage should import providers directly.

Adding a new provider:
  1. Create a file like ``my_provider.py`` subclassing ``LLMProvider``.
  2. Implement ``structured_call`` and ``free_text_call``.
  3. Import & register in this ``__init__.py``.

That's it. No caller code changes.
"""

from .anthropic import AnthropicProvider
from .base import LLMProvider, LLMResult, UsageInfo
from .google import GoogleProvider
from .openai_compat import GroqProvider, OpenAIProvider
from .registry import ProviderRegistry, get_default_provider, get_provider, list_providers

__all__ = [
    "AnthropicProvider",
    "GoogleProvider",
    "GroqProvider",
    "LLMProvider",
    "LLMResult",
    "OpenAIProvider",
    "ProviderRegistry",
    "UsageInfo",
    "get_default_provider",
    "get_provider",
    "list_providers",
]
