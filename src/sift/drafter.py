"""Reply drafter.

Given a Thread (and a VoiceProfile), produce a Draft. One LLM call per draft.
We only draft for threads classified as ``urgent`` or ``needs_reply``.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import cache
from .config import CONFIG
from .llm import load_prompt, structured_call
from .models import Category, Classification, Draft, Thread
from .voice import VoiceProfile, current_voice_profile

logger = logging.getLogger(__name__)

# Automated senders that don't accept replies. Matched case-insensitively
# against the local-part of the From: address. Kept conservative on purpose:
# a false positive here means we *don't* draft a reply the user wanted, which
# is worse than drafting a reply they'll ignore. We intentionally do NOT match
# "support@", "billing@", "info@", etc. — those often accept human replies.
_NO_REPLY_LOCAL_PART_PATTERNS = [
    re.compile(r"^no[-_.]?reply(\+.*)?$", re.IGNORECASE),       # noreply, no-reply, no_reply
    re.compile(r"^do[-_.]?not[-_.]?reply(\+.*)?$", re.IGNORECASE),  # donotreply, do-not-reply
    re.compile(r"^notifications?(\+.*)?$", re.IGNORECASE),      # notification, notifications
    re.compile(r"^mailer[-_.]?daemon$", re.IGNORECASE),         # bounce messages
    re.compile(r"^bounces?(\+.*)?$", re.IGNORECASE),            # bounce, bounces
    re.compile(r"^postmaster$", re.IGNORECASE),
    re.compile(r"^auto[-_.]?(reply|confirm|responder)(\+.*)?$", re.IGNORECASE),
]


def is_no_reply_sender(email_address: str) -> bool:
    """True if the given email address looks like an automated/no-reply sender.

    Matches against the local-part of the address (before the ``@``). Used by
    the drafter to skip thread that aren't worth generating a reply for — the
    classifier still runs, so the thread still shows up in the brief, just
    without a draft attached.
    """
    if not email_address or "@" not in email_address:
        return False
    local = email_address.split("@", 1)[0].strip()
    return any(p.match(local) for p in _NO_REPLY_LOCAL_PART_PATTERNS)


_DRAFT_SYSTEM_TEMPLATE = load_prompt("draft")

_DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "body": {
            "type": "string",
            "description": "The reply body, ready to paste into Gmail. Start with salutation if appropriate.",
        },
        "subject": {
            "type": "string",
            "description": "The subject line for the reply. Usually 'Re: <original subject>'.",
        },
        "tone_notes": {
            "type": "string",
            "description": "Under 20 words. Why you chose this register/tone.",
        },
    },
    "required": ["body", "subject", "tone_notes"],
}

# Which categories warrant a draft.
DRAFT_CATEGORIES = {Category.URGENT, Category.NEEDS_REPLY}


def _render_thread(thread: Thread, *, recipient_email: str | None = None) -> str:
    """Render the incoming thread for the drafter prompt.

    We deliberately label Kyle's address (when known) as the ``To:`` so the
    LLM can't get confused about which side of the exchange it's writing for.
    Without this, threads where the sender is a company ("From: Anthropic...
    your card was declined") sometimes elicit drafts written *as* the company,
    not as Kyle replying to it.
    """
    to_line = recipient_email or thread.to
    return (
        f"From: {thread.from_name} <{thread.from_}>\n"
        f"To: {to_line}  (this is you — you are Kyle, replying TO the sender above)\n"
        f"Subject: {thread.subject}\n"
        f"\n"
        f"{thread.body}"
    )


def draft_reply(
    thread: Thread,
    *,
    voice: VoiceProfile | None = None,
    user_email: str | None = None,
    model: str | None = None,
) -> Draft:
    """Draft a single reply. One LLM call."""
    v = voice or current_voice_profile(user_email=user_email)
    system = _DRAFT_SYSTEM_TEMPLATE.format(voice_profile=v.render_for_prompt())
    rendered = _render_thread(thread, recipient_email=user_email)
    user = (
        "Draft Kyle's reply to the email thread below. "
        "Kyle is the recipient (To:), and he is writing back to the sender (From:).\n\n"
        f"---\n{rendered}\n---"
    )

    result = structured_call(
        system=system,
        user=user,
        tool_name="submit_draft",
        tool_description="Submit a reply draft for Kyle to review and send.",
        input_schema=_DRAFT_SCHEMA,
        model=model,
        max_tokens=600,
        log_tag="draft",
    )

    return Draft(thread_id=thread.id, **result)


def draft_replies(
    threads: list[Thread],
    classifications: list[Classification],
    *,
    voice: VoiceProfile | None = None,
    user_email: str | None = None,
    model: str | None = None,
    max_workers: int = 3,
    use_cache: bool = True,
) -> dict[str, Draft]:
    """Draft replies for every thread whose classification warrants one.

    Returns a dict from thread_id -> Draft (threads that didn't warrant a draft
    are omitted). Concurrency identical to the classifier's approach — kept at
    3 to stay under free/starter-tier rate limits; retry with backoff handles
    any 429s that slip through.

    When ``use_cache`` is true (default) we consult :mod:`sift.cache` first
    and only draft for threads that don't already have a cached draft. Drafts
    are deterministic-ish and expensive, so reusing the previous draft on a
    rerun is almost always what you want. Clear via ``sift cache-clear drafts``
    to force fresh drafts after a prompt/model change.
    """
    class_by_id = {c.thread_id: c for c in classifications}
    # Two filters: (1) classification must warrant a draft;
    # (2) sender must not be an automated/no-reply address — drafting replies
    # to noreply@ wastes tokens and confuses the user about what's actionable.
    skipped_no_reply = 0
    targets: list[Thread] = []
    for t in threads:
        c = class_by_id.get(t.id)
        if c is None or c.category not in DRAFT_CATEGORIES:
            continue
        if is_no_reply_sender(t.from_):
            skipped_no_reply += 1
            continue
        targets.append(t)
    if skipped_no_reply:
        logger.info("Drafter: skipped %d no-reply sender(s)", skipped_no_reply)

    # Resolve voice once: a single cache lookup beats one per draft call.
    # If caller gave us an explicit profile, use it; otherwise consult cache
    # via current_voice_profile (which falls back to DEFAULT_VOICE on miss).
    resolved_voice = voice or current_voice_profile(user_email=user_email)

    drafts: dict[str, Draft] = {}
    to_draft: list[Thread] = []

    if use_cache:
        for t in targets:
            hit = cache.get_cached_draft(t.id)
            if hit is not None:
                drafts[t.id] = hit
            else:
                to_draft.append(t)
        if drafts:
            logger.info(
                "Drafter cache: %d/%d hit, %d to run",
                len(drafts), len(targets), len(to_draft),
            )
    else:
        to_draft = list(targets)

    cache_model = model or CONFIG.model
    cache_provider = CONFIG.llm_provider

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                draft_reply, t, voice=resolved_voice, user_email=user_email, model=model
            ): t
            for t in to_draft
        }
        for fut in as_completed(futures):
            thread = futures[fut]
            try:
                draft = fut.result()
            except Exception:
                logger.exception("Drafting failed for thread %s", thread.id)
                continue
            drafts[thread.id] = draft
            if use_cache:
                try:
                    cache.cache_draft(draft, model=cache_model, provider=cache_provider)
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to cache draft for %s", thread.id)
    return drafts
