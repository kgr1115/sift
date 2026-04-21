"""Provider-agnostic facade for LLM calls.

The real per-provider implementations live in ``sift.providers``.
This module exposes the call-site API used by the rest of the codebase:

  * ``structured_call``  — schema-enforced JSON (classification, drafts, …)
  * ``free_text_call``   — plain text (morning brief narrative, etc.)
  * ``load_prompt``      — read a prompt template from ./prompts/

Dispatch rules:
  * The caller can pass ``provider_name=`` to force a specific backend. This is
    how the provider-comparison eval fans the same prompt out to every LLM.
  * Otherwise we use ``LLM_PROVIDER`` from the environment (default: anthropic).
  * The caller can pass ``model=`` to override the provider's default model.

Logs every call to ``logs/<tag>.jsonl`` so mis-classifications and weird drafts
are debuggable without re-running. This used to live in a big Anthropic-specific
wrapper; keeping the logging here (rather than inside each provider) means the
log format is identical across providers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .config import CONFIG, PROJECT_ROOT
from .providers import LLMResult, get_default_provider, get_provider

logger = logging.getLogger(__name__)

# Log directory — gitignored. Useful for post-hoc prompt debugging.
LOG_DIR = PROJECT_ROOT / "logs"


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Heuristic match for 429/rate-limit across provider SDKs.

    Each provider SDK raises a different concrete class (anthropic.RateLimitError,
    openai.RateLimitError, google.api_core.exceptions.ResourceExhausted, etc.).
    Rather than import them all, we match on class name + HTTP status attribute,
    which is good enough to catch every provider we ship with.
    """
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "resourceexhausted" in name:
        return True
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return status == 429


def _retry_on_rate_limit(call, *, log_tag: str, max_attempts: int = 5):
    """Call ``call()`` with exponential backoff on 429s.

    Backoff: 2, 4, 8, 16, 32 seconds. Honors ``Retry-After`` header when the
    provider SDK exposes it (anthropic sets it on the underlying httpx response).
    """
    for attempt in range(max_attempts):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 — we re-raise non-rate-limit errors
            if not _is_rate_limit_error(exc) or attempt == max_attempts - 1:
                raise
            # Prefer server-supplied Retry-After if we can find it.
            retry_after = None
            resp = getattr(exc, "response", None)
            if resp is not None:
                headers = getattr(resp, "headers", {}) or {}
                try:
                    retry_after = float(headers.get("retry-after") or headers.get("Retry-After") or 0) or None
                except (TypeError, ValueError):
                    retry_after = None
            delay = retry_after if retry_after else (2 ** (attempt + 1))
            logger.warning(
                "[%s] rate-limited (attempt %d/%d), sleeping %.1fs",
                log_tag, attempt + 1, max_attempts, delay,
            )
            time.sleep(delay)
    # Unreachable — the loop either returns or raises.
    raise RuntimeError("unreachable")


def _resolve_provider(provider_name: str | None, model: str | None):
    """Pick a provider instance. ``provider_name=None`` -> configured default.

    Precedence for the model string, highest first:
      1. explicit ``model=`` argument (call-site override, e.g. comparison eval)
      2. ``SIFT_MODEL`` env var (via CONFIG.model) — but ONLY when the resolved
         provider matches the configured default. Otherwise we'd try to run
         e.g. ``claude-sonnet-4-6`` on OpenAI, which 404s. The comparison eval
         passes ``provider_name=`` explicitly and expects each provider to fall
         back to its own ``default_model``.
      3. each provider's ``default_model`` attribute
    """
    effective_provider = provider_name or CONFIG.llm_provider
    if model is not None:
        effective_model: str | None = model
    elif effective_provider == CONFIG.llm_provider:
        effective_model = CONFIG.model
    else:
        effective_model = None
    if provider_name is None and effective_model is None:
        # Preserve the original "no overrides" path so get_default_provider
        # stays the single source of truth when nothing's customized.
        return get_default_provider()
    return get_provider(effective_provider, effective_model)


def _log_interaction(tag: str, payload: dict[str, Any]) -> None:
    """Append a single JSON line to logs/<tag>.jsonl. Failures here are ignored."""
    try:
        LOG_DIR.mkdir(exist_ok=True)
        line = json.dumps({"ts": time.time(), **payload}, default=str)
        (LOG_DIR / f"{tag}.jsonl").open("a").write(line + "\n")
    except Exception:  # noqa: BLE001 — logging must never break the main path
        logger.exception("Failed to log LLM interaction")


def _log_result(tag: str, system: str, user: str, result: LLMResult) -> None:
    usage = result.usage
    _log_interaction(
        tag,
        {
            "provider": result.provider,
            "model": usage.model if usage else None,
            "system_chars": len(system),
            "user_chars": len(user),
            "data": result.data,
            "text_chars": len(result.text) if result.text else None,
            "usage": {
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "latency_ms": usage.latency_ms if usage else None,
            },
        },
    )


def structured_call(
    *,
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    input_schema: dict[str, Any],
    model: str | None = None,
    provider_name: str | None = None,
    max_tokens: int = 1024,
    log_tag: str = "llm",
) -> dict[str, Any]:
    """Run a schema-enforced structured call against the chosen provider.

    Returns just the validated ``data`` dict for backwards-compatibility with
    the original Anthropic-only helper. Use ``structured_call_full`` when you
    need usage / latency / provider metadata (e.g. for the comparison eval).
    """
    return structured_call_full(
        system=system,
        user=user,
        tool_name=tool_name,
        tool_description=tool_description,
        input_schema=input_schema,
        model=model,
        provider_name=provider_name,
        max_tokens=max_tokens,
        log_tag=log_tag,
    ).data or {}


def structured_call_full(
    *,
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    input_schema: dict[str, Any],
    model: str | None = None,
    provider_name: str | None = None,
    max_tokens: int = 1024,
    log_tag: str = "llm",
) -> LLMResult:
    """Like ``structured_call`` but returns the full ``LLMResult``.

    Preferred for any code that needs to reason about cost, latency, or which
    provider actually answered (notably ``evals/test_provider_comparison.py``).
    """
    provider = _resolve_provider(provider_name, model)
    result = _retry_on_rate_limit(
        lambda: provider.structured_call(
            system=system,
            user=user,
            tool_name=tool_name,
            tool_description=tool_description,
            input_schema=input_schema,
            max_tokens=max_tokens,
        ),
        log_tag=log_tag,
    )
    _log_result(log_tag, system, user, result)
    return result


def free_text_call(
    *,
    system: str,
    user: str,
    model: str | None = None,
    provider_name: str | None = None,
    max_tokens: int = 1024,
    log_tag: str = "llm_freetext",
) -> str:
    """Plain text completion. Returns just the text for backwards compat."""
    provider = _resolve_provider(provider_name, model)
    result = _retry_on_rate_limit(
        lambda: provider.free_text_call(system=system, user=user, max_tokens=max_tokens),
        log_tag=log_tag,
    )
    _log_result(log_tag, system, user, result)
    return result.text or ""


def load_prompt(name: str) -> str:
    """Load a prompt template from src/sift/prompts/<name>.md.

    Prompts live on disk in markdown so they diff cleanly in PRs and can be
    reviewed by non-engineers. Use ``str.format`` at call sites for slot-filling.
    """
    path = Path(__file__).parent / "prompts" / f"{name}.md"
    return path.read_text(encoding="utf-8")
