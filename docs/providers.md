# Multi-provider LLM layer

Sift can talk to four LLM providers behind a single interface. Which one runs at any given moment is controlled by the `LLM_PROVIDER` environment variable — everything else stays the same.

## Why this exists

Two reasons:

1. **Empirical cost/quality decisions.** Model pricing pages are not a substitute for measurement on your own task. By the time you've written a real classifier prompt and a labeled eval set, you're one change (`LLM_PROVIDER=openai`) away from a real answer.
2. **Portfolio demonstration.** The abstraction is small and well-factored enough to read in a sitting; it shows up as one `LLMProvider` ABC, four tiny subclasses, and a registry.

## Supported providers

| Provider  | `LLM_PROVIDER` value | Default model              | Structured-output mechanism              |
|-----------|----------------------|----------------------------|------------------------------------------|
| Anthropic | `anthropic`          | `claude-sonnet-4-6`        | Tool-use with enforced `input_schema`    |
| OpenAI    | `openai`             | `gpt-4o-mini`              | `response_format: json_schema` (strict)  |
| Google    | `google`             | `gemini-2.5-flash`         | `response_schema` + JSON MIME type       |
| Groq      | `groq`               | `llama-3.3-70b-versatile`  | JSON-mode + schema hint in the prompt    |

Anthropic's tool-use pattern is the cleanest — the schema is enforced at the API layer and Claude can't return invalid JSON without erroring. OpenAI's strict JSON-schema is the next best; it requires `additionalProperties: false` on every object and every property listed in `required`, so we have a small `_make_strict()` helper that walks a permissive schema and tightens it. Google's Gemini accepts the schema directly on the request. Groq's Llama models don't have universal JSON-schema support, so we fall back to JSON mode plus a schema hint in the system prompt — strictly weaker, but workable.

## Picking a provider

Edit `.env`:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
# leave the others blank
```

Or override per-call in Python:

```python
from sift.llm import structured_call_full

result = structured_call_full(
    system=...,
    user=...,
    tool_name="classify_thread",
    tool_description="...",
    input_schema=...,
    provider_name="groq",   # override
    model="llama-3.1-8b-instant",
)
print(result.provider, result.usage.model, result.usage.latency_ms)
```

Override precedence for the model string, highest first:

1. Explicit `model=` argument to the call
2. `SIFT_MODEL` env var
3. The provider's own `default_model` attribute

## The provider-comparison eval

```bash
pytest evals/test_provider_comparison.py -v -s
```

The eval auto-detects which providers have API keys set in the environment, runs the classifier through each, and writes:

- `evals/last_provider_comparison.md` — a human-friendly table (accuracy, errors, total cost, $/1k threads, average latency, per-category recall)
- `evals/last_provider_comparison.json` — the same data for downstream tooling

Typical cost on the 40-fixture set across all four providers: **well under $0.05**. This eval is the thing I reach for when someone asks "can we use a cheaper model for triage?" — the answer is a markdown table, not vibes.

## Adding a new provider

One file under `src/sift/providers/`. Subclass `LLMProvider`, implement `structured_call` and `free_text_call`, declare `name` / `default_model` / `pricing`, then register in `providers/registry.py`. That's it — no caller-side changes. See `providers/anthropic.py` for the canonical example.

## What I'd change with more time

- **Retries and backoff.** Every provider has its own rate-limit error class; a small shared retry wrapper would live in `base.py`.
- **Streaming.** The `LLMResult` type has room for partial responses but nothing produces them today.
- **Semantic-cache layer.** Identical classifier prompts on identical threads don't need to hit the network twice; a small SQLite-backed cache keyed on `(provider, model, prompt_hash)` would drop eval cost by another order of magnitude.
- **Per-provider strict-mode schema transforms.** OpenAI has `_make_strict`; Google's schema dialect has its own quirks (no `$ref`, limited `oneOf`) that would benefit from a matching transform for more complex schemas.
