"""Google Gemini provider via the google-genai SDK.

Structured output uses Gemini's ``response_schema`` + ``response_mime_type=
"application/json"`` config. Gemini's schema format is slightly different from
OpenAPI/JSON-Schema (no ``$ref``, limited ``oneOf``, etc.), but for our simple
schemas we can pass through almost unchanged.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache

from google import genai
from google.genai import types

from .base import LLMProvider, LLMResult, UsageInfo


class GoogleProvider(LLMProvider):
    name = "google"
    # gemini-2.0-flash was retired for new users in 2026; 2.5-flash is the
    # current flash-tier model and keeps the comparison apples-to-apples with
    # gpt-4o-mini and llama-3.3-70b.
    default_model = "gemini-2.5-flash"

    pricing = {
        # Google's posted per-MTok pricing (non-thinking mode for 2.5-flash).
        # Flash pricing tiers at input-token-count boundaries which we don't
        # model here; numbers are the <128K tier.
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.5-flash-lite": (0.10, 0.40),
        "gemini-2.5-pro": (1.25, 10.00),
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-2.0-flash-lite": (0.075, 0.30),
    }

    @lru_cache(maxsize=1)
    def _client(self) -> "genai.Client":  # type: ignore[override]
        key = os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        return genai.Client(api_key=key)

    def structured_call(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,  # Not used by Gemini — kept for interface parity
        tool_description: str,  # Not used
        input_schema: dict,
        max_tokens: int = 1024,
    ) -> LLMResult:
        t0 = time.perf_counter()
        resp = self._client().models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=input_schema,
                # Disable thinking for structured calls. 2.5-flash is a
                # thinking model by default, which burns the output-token
                # budget on internal reasoning and returns an empty payload
                # for modest max_tokens. Classification doesn't benefit from
                # thinking and we want an apples-to-apples latency comparison.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        if not resp.text:
            raise RuntimeError(
                f"Gemini returned no text (finish_reason="
                f"{getattr(resp.candidates[0], 'finish_reason', '?') if resp.candidates else '?'}). "
                "Likely hit max_output_tokens before producing JSON."
            )
        data = json.loads(resp.text)
        usage = resp.usage_metadata
        return LLMResult(
            data=data,
            usage=UsageInfo(
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
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
        resp = self._client().models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                # See structured_call for rationale.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = (resp.text or "").strip()
        usage = resp.usage_metadata
        return LLMResult(
            text=text,
            usage=UsageInfo(
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                latency_ms=latency_ms,
                model=self.model,
            ),
            provider=self.name,
        )
