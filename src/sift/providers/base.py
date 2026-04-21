"""Provider abstraction.

Every LLM backend implements ``LLMProvider`` with two methods: ``structured_call``
(schema-enforced JSON output) and ``free_text_call`` (plain text). Both return
an ``LLMResult`` carrying token usage + latency so the provider-comparison
eval can build a cost/performance table without the provider leaking impl details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UsageInfo:
    """Accounting for a single LLM call."""

    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str  # Exact model string used, for reproducibility.

    def estimated_cost_usd(self, input_per_mtok: float, output_per_mtok: float) -> float:
        return (
            self.input_tokens / 1_000_000 * input_per_mtok
            + self.output_tokens / 1_000_000 * output_per_mtok
        )


@dataclass
class LLMResult:
    """Container for a provider's response."""

    data: dict[str, Any] | None = None   # Set on structured_call.
    text: str | None = None              # Set on free_text_call.
    usage: UsageInfo | None = None
    provider: str = ""                   # e.g. "anthropic", "openai", "google", "groq"
    extra: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses must:
      * set ``name`` (short id like "anthropic")
      * set ``default_model``
      * populate ``pricing`` with per-model (input_per_mtok, output_per_mtok)
      * implement ``structured_call`` and ``free_text_call``
    """

    name: str = ""
    default_model: str = ""
    pricing: dict[str, tuple[float, float]] = {}

    def __init__(self, *, model: str | None = None) -> None:
        self.model = model or self.default_model

    @abstractmethod
    def structured_call(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        max_tokens: int = 1024,
    ) -> LLMResult:
        """Return structured JSON conforming to ``input_schema``."""
        ...

    @abstractmethod
    def free_text_call(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> LLMResult:
        """Return free-form text."""
        ...

    def get_pricing(self, model: str | None = None) -> tuple[float, float]:
        """Return (input_per_mtok, output_per_mtok). Returns (0, 0) for unknown models."""
        m = model or self.model
        return self.pricing.get(m, (0.0, 0.0))

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(model={self.model!r})"
