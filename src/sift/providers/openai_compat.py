"""OpenAI + Groq providers.

Both use the OpenAI Python SDK because Groq exposes an OpenAI-compatible API
at ``https://api.groq.com/openai/v1``. The only differences between the two
are the base URL, the API key env var, and the model list — so they share a
base implementation and differ by class attributes.

Structured output uses OpenAI's strict JSON-schema ``response_format`` where
available (GPT-4o family). For Groq/Llama we fall back to JSON-mode plus a
schema reminder in the user message; this is less strict than Anthropic
tool-use but works in practice for schemas we use.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache

from openai import OpenAI

from .base import LLMProvider, LLMResult, UsageInfo


class _OpenAICompatProvider(LLMProvider):
    """Shared base for OpenAI + Groq. Subclasses set name/default_model/pricing."""

    # Subclasses override these.
    env_var: str = ""
    base_url: str | None = None  # None means use the library default (OpenAI's endpoint)
    supports_strict_schema: bool = True  # False for models that only have JSON-mode

    @lru_cache(maxsize=1)
    def _client(self) -> OpenAI:  # type: ignore[override]
        key = os.getenv(self.env_var, "")
        if not key:
            raise RuntimeError(f"{self.env_var} is not set")
        if self.base_url:
            return OpenAI(api_key=key, base_url=self.base_url)
        return OpenAI(api_key=key)

    # ---------------- structured_call ----------------

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

        if self.supports_strict_schema:
            # OpenAI's strict structured-output API. The schema must have additionalProperties: false
            # at every nesting level and required: [...] listing every property for strict mode.
            strict_schema = _make_strict(input_schema)
            resp = self._client().chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": tool_name,
                        "description": tool_description,
                        "schema": strict_schema,
                        "strict": True,
                    },
                },
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
        else:
            # JSON mode fallback: tell the model to output JSON and include the schema in the prompt.
            schema_hint = (
                "Respond with a JSON object that matches this schema exactly:\n"
                f"```json\n{json.dumps(input_schema, indent=2)}\n```"
            )
            resp = self._client().chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system + "\n\n" + schema_hint},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)

        return LLMResult(
            data=data,
            usage=UsageInfo(
                input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                latency_ms=latency_ms,
                model=self.model,
            ),
            provider=self.name,
        )

    # ---------------- free_text_call ----------------

    def free_text_call(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> LLMResult:
        t0 = time.perf_counter()
        resp = self._client().chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = (resp.choices[0].message.content or "").strip()
        return LLMResult(
            text=text,
            usage=UsageInfo(
                input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                latency_ms=latency_ms,
                model=self.model,
            ),
            provider=self.name,
        )


def _make_strict(schema: dict) -> dict:
    """Transform a permissive JSON schema into OpenAI's strict mode shape.

    Strict mode requires ``additionalProperties: false`` on every object node and
    every property listed in ``required``. We walk the schema once and apply both.
    """
    def _walk(node: dict) -> dict:
        if not isinstance(node, dict):
            return node  # type: ignore[unreachable]
        out = dict(node)
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
            props = out.get("properties", {})
            if props:
                out["properties"] = {k: _walk(v) for k, v in props.items()}
                # Strict mode requires every property in `required`.
                out["required"] = list(props.keys())
        elif out.get("type") == "array" and isinstance(out.get("items"), dict):
            out["items"] = _walk(out["items"])
        return out

    return _walk(schema)


class OpenAIProvider(_OpenAICompatProvider):
    name = "openai"
    default_model = "gpt-4o-mini"
    env_var = "OPENAI_API_KEY"
    supports_strict_schema = True
    pricing = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
    }


class GroqProvider(_OpenAICompatProvider):
    """Groq runs open-weight models (Llama 3.3, Mixtral) at very low latency.

    Uses the OpenAI-compatible endpoint. JSON mode is supported; strict JSON
    schema is not universally available so we stay in JSON-mode-plus-hint land.
    """

    name = "groq"
    default_model = "llama-3.3-70b-versatile"
    env_var = "GROQ_API_KEY"
    base_url = "https://api.groq.com/openai/v1"
    supports_strict_schema = False
    pricing = {
        # Groq's posted per-MTok pricing as of early 2026. Treat as approximate.
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "llama-3.1-8b-instant": (0.05, 0.08),
        "mixtral-8x7b-32768": (0.24, 0.24),
    }
