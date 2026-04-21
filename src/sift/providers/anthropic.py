"""Anthropic provider — Claude models via the anthropic SDK.

Structured output uses Claude's tool-use API: we declare a single "tool" whose
input_schema IS the desired output schema, then force tool_choice so the model
has to call it exactly once. The tool inputs come back as our structured result.

This is the cleanest structured-output path in the industry — Anthropic's
schema enforcement happens at the API layer, not via post-hoc JSON parsing.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

import anthropic

from .base import LLMProvider, LLMResult, UsageInfo


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    default_model = "claude-sonnet-4-6"

    # Per-MTok prices (input, output). Approximate — update when Anthropic moves pricing.
    pricing = {
        "claude-opus-4-6": (15.00, 75.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-haiku-4-5": (1.00, 5.00),
        "claude-haiku-4-5-20251001": (1.00, 5.00),
    }

    @lru_cache(maxsize=1)
    def _client(self) -> anthropic.Anthropic:  # type: ignore[override]
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return anthropic.Anthropic(api_key=key)

    def structured_call(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict,
        max_tokens: int = 1024,
    ) -> LLMResult:
        t0 = time.perf_counter()
        resp = self._client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=[{"name": tool_name, "description": tool_description, "input_schema": input_schema}],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        tool_use = next((b for b in resp.content if b.type == "tool_use"), None)
        if tool_use is None:
            raise RuntimeError(
                f"Claude did not emit a {tool_name} tool-use block. "
                f"stop_reason={resp.stop_reason}"
            )

        return LLMResult(
            data=dict(tool_use.input),  # type: ignore[arg-type]
            usage=UsageInfo(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                latency_ms=latency_ms,
                model=self.model,
            ),
            provider=self.name,
        )

    def free_text_call(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> LLMResult:
        t0 = time.perf_counter()
        resp = self._client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = "\n\n".join(b.text for b in resp.content if b.type == "text").strip()

        return LLMResult(
            text=text,
            usage=UsageInfo(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                latency_ms=latency_ms,
                model=self.model,
            ),
            provider=self.name,
        )
