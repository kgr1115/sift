"""Triage classifier.

Given a Thread, produce a Classification: one of 5 categories, a confidence
score, a one-line summary, and a short reason. All via Claude tool-use so the
schema is enforced.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import cache
from .config import CONFIG
from .llm import load_prompt, structured_call
from .models import CATEGORY_VALUES, Classification, Thread

logger = logging.getLogger(__name__)

_CLASSIFY_SYSTEM = load_prompt("classify")

_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": CATEGORY_VALUES,
            "description": "The single triage category for this thread.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "How confident you are in this classification (0.0-1.0).",
        },
        "one_line_summary": {
            "type": "string",
            "description": "Under 20 words. What Kyle needs to know at a glance.",
        },
        "reason": {
            "type": "string",
            "description": "Under 30 words. Why this category?",
        },
    },
    "required": ["category", "confidence", "one_line_summary", "reason"],
}


def _render_thread(thread: Thread) -> str:
    """Serialize a thread for the LLM. Keep it simple and readable."""
    return (
        f"From: {thread.from_name} <{thread.from_}>\n"
        f"To: {thread.to}\n"
        f"Received: {thread.received_at.isoformat()}\n"
        f"Subject: {thread.subject}\n"
        f"\n"
        f"{thread.body}"
    )


def classify_thread(
    thread: Thread,
    *,
    model: str | None = None,
    provider_name: str | None = None,
) -> Classification:
    """Classify a single thread. One LLM call.

    ``provider_name`` lets callers (notably the comparison eval) target a
    specific backend; leaving it ``None`` uses the configured default.
    """
    user = f"Classify the following email thread:\n\n---\n{_render_thread(thread)}\n---"

    result = structured_call(
        system=_CLASSIFY_SYSTEM,
        user=user,
        tool_name="classify_thread",
        tool_description="Record the triage classification for an email thread.",
        input_schema=_CLASSIFY_SCHEMA,
        model=model,
        provider_name=provider_name,
        max_tokens=400,
        log_tag="classify",
    )

    return Classification(thread_id=thread.id, **result)


def classify_threads(
    threads: list[Thread],
    *,
    model: str | None = None,
    provider_name: str | None = None,
    max_workers: int = 3,
    use_cache: bool = True,
) -> list[Classification]:
    """Classify many threads concurrently.

    We run up to ``max_workers`` classifications in parallel. 3 is a
    conservative default that stays under Anthropic's 30k input-tokens/min
    rate limit on free/starter tiers while still cutting runtime ~3x. Bump
    it via the argument if you have a higher-tier key.

    Transient 429s from the provider are retried with exponential backoff
    inside ``llm.structured_call``; only exhausted retries land here.

    When ``use_cache`` is true (default) we consult :mod:`sift.cache` first
    and only make LLM calls for threads we haven't classified before. Same
    ``thread_id`` → same classification; if you change the prompt or switch
    models you'll want to ``sift cache-clear classifications`` first.
    """
    results: dict[str, Classification] = {}
    to_classify: list[Thread] = []

    if use_cache:
        for t in threads:
            hit = cache.get_cached_classification(t.id)
            if hit is not None:
                results[t.id] = hit
            else:
                to_classify.append(t)
        if results:
            logger.info(
                "Classifier cache: %d/%d hit, %d to run",
                len(results), len(threads), len(to_classify),
            )
    else:
        to_classify = list(threads)

    cache_model = model or CONFIG.model
    cache_provider = provider_name or CONFIG.llm_provider

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(classify_thread, t, model=model, provider_name=provider_name): t
            for t in to_classify
        }
        for fut in as_completed(futures):
            thread = futures[fut]
            try:
                classification = fut.result()
            except Exception:
                logger.exception("Classification failed for thread %s", thread.id)
                # We still want to produce *something* so the brief isn't empty.
                classification = Classification(
                    thread_id=thread.id,
                    category="fyi",  # safe-ish default — won't raise alarms or hide urgent items wrongly
                    confidence=0.0,
                    one_line_summary=f"[classification failed] {thread.subject}",
                    reason="Classifier errored; falling back to FYI.",
                )
            results[thread.id] = classification
            # Persist only real (non-fallback) classifications so transient
            # errors don't poison the cache.
            if use_cache and classification.confidence > 0.0:
                try:
                    cache.cache_classification(
                        classification, model=cache_model, provider=cache_provider
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to cache classification for %s", thread.id)
    # Preserve original order.
    return [results[t.id] for t in threads]
