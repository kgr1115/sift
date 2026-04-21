"""Quick diagnostic: try a trivial structured call against each provider
whose API key is set, print whatever happens.

Usage:
    python scripts/check_providers.py

Intended for debugging auth / schema / SDK issues when the provider-comparison
eval shows 40/40 errors for a provider. The classifier schema is reasonably
simple but some providers are fussier than others about required fields,
description lengths, nullable types, etc. — this reproduces a minimal call so
the actual exception surfaces.
"""

from __future__ import annotations

from sift.llm import structured_call_full
from sift.providers.registry import list_available_providers

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["urgent", "needs_reply", "fyi", "newsletter", "trash"],
        },
        "confidence": {"type": "number"},
        "one_line_summary": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["category", "confidence", "one_line_summary", "reason"],
}


def main() -> None:
    providers = list_available_providers()
    print(f"Providers with API key set: {providers}\n")

    for name in providers:
        print(f"--- {name} ---")
        try:
            result = structured_call_full(
                system="You classify one email. Respond only with the tool call.",
                user="From: test@example.com\nSubject: Quick question\nBody: Can you review this by Friday?",
                tool_name="classify_thread",
                tool_description="Record the classification.",
                input_schema=SCHEMA,
                provider_name=name,
                max_tokens=200,
                log_tag=f"diag_{name}",
            )
            print(f"OK: data={result.data}")
            if result.usage:
                print(
                    f"    tokens in/out={result.usage.input_tokens}/"
                    f"{result.usage.output_tokens} latency={result.usage.latency_ms:.0f}ms"
                )
        except Exception as e:  # noqa: BLE001 — we want to see the exception type
            print(f"FAIL: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
