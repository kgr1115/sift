"""Learn the user's writing voice from their sent mail.

Two modes:

* :data:`DEFAULT_VOICE` — a hand-written best-guess profile used on first run
  or when Gmail isn't connected. Lets the drafter work in fixtures mode out of
  the box, and lets the whole pipeline run end-to-end without a network call.

* :func:`learn_voice_profile` — one Claude call over a batch of recent sent
  mail. Produces a :class:`VoiceProfile` with a compressed style summary and
  three verbatim example replies. Cached per-user-email in the SQLite layer
  with a weeklong TTL; writing styles drift over months, not minutes.

:func:`current_voice_profile` is the seam every other module uses. It consults
the cache first, falls back to :data:`DEFAULT_VOICE`, and optionally triggers
a live learn when Gmail access is available and no fresh profile is cached.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from . import cache
from .llm import load_prompt, structured_call
from .models import VoiceProfile

logger = logging.getLogger(__name__)

# Re-export so callers can `from sift.voice import VoiceProfile`.
__all__ = [
    "VoiceProfile",
    "DEFAULT_VOICE",
    "current_voice_profile",
    "learn_voice_profile",
    "VOICE_CACHE_TTL_SECONDS",
]

# One week. Writing voice changes on the order of months, but we re-learn
# weekly so new idioms and relationships (e.g. a new employer, a new contact
# the user is regularly writing to) get picked up without manual action.
VOICE_CACHE_TTL_SECONDS: float = 7 * 24 * 60 * 60


# Default profile used when we haven't yet ingested the user's sent mail.
# This is a best-guess voice for a senior PM / engineer who is job-searching.
DEFAULT_VOICE = VoiceProfile(
    summary=(
        "Kyle writes in a warm-but-efficient register. He's comfortable with lowercase for "
        "casual threads with friends and family, and switches to sentence-case with a "
        "'Best,' sign-off for professional outreach like recruiters, hiring managers, and "
        "VCs. His replies are typically 2-5 sentences — he does not write long emails. "
        "He never opens with 'I hope this finds you well.' He uses em-dashes sparingly. "
        "He signs personal emails with just 'Kyle' and professional ones with 'Best,\\nKyle'."
    ),
    style_examples=[],  # Will be populated once we've pulled real sent mail.
)


_VOICE_SYSTEM = load_prompt("voice")

_VOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "A compressed voice description (~150-300 words). Written as guidance "
                "a drafter could follow directly: register rules, opener/closer patterns, "
                "length expectations, notable quirks."
            ),
        },
        "style_examples": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Three verbatim example replies from the input spanning the user's "
                "register range. Copy body text exactly; do not edit or fabricate."
            ),
        },
    },
    "required": ["summary", "style_examples"],
}


def _render_sent_batch(messages: list[dict[str, str]]) -> str:
    """Format sent messages for the voice-learner prompt.

    We truncate each body at ~1500 chars to keep the prompt under a reasonable
    input-token budget. That's enough for the LLM to grok style on a short
    reply; signatures and long email threads aren't what we're analyzing.
    """
    chunks: list[str] = []
    for i, m in enumerate(messages, 1):
        body = (m.get("body") or "").strip()
        if len(body) > 1500:
            body = body[:1500].rstrip() + "\n[...truncated]"
        chunks.append(
            f"--- Sent message {i} ---\n"
            f"To: {m.get('to', '')}\n"
            f"Subject: {m.get('subject', '')}\n"
            f"\n{body}"
        )
    return "\n\n".join(chunks)


def learn_voice_profile(
    sent_messages: list[dict[str, str]],
    user_email: str,
    *,
    model: str | None = None,
) -> VoiceProfile:
    """Summarize a batch of sent mail into a VoiceProfile via one LLM call.

    Inputs must already be fetched (via :func:`gmail_client.fetch_sent_messages`
    or equivalent) so this function is testable without a network dependency
    — pass a list of ``{"subject": ..., "to": ..., "body": ...}`` dicts.
    """
    if not sent_messages:
        # Fall back rather than sending an empty prompt to the LLM.
        logger.warning("learn_voice_profile: no sent messages; returning DEFAULT_VOICE.")
        profile = DEFAULT_VOICE.model_copy(update={"user_email": user_email, "learned_at": datetime.now(timezone.utc)})
        return profile

    user = (
        f"Analyze the following {len(sent_messages)} emails sent by {user_email} "
        "and produce a voice profile per the system instructions.\n\n"
        f"{_render_sent_batch(sent_messages)}"
    )

    result = structured_call(
        system=_VOICE_SYSTEM,
        user=user,
        tool_name="submit_voice_profile",
        tool_description="Submit a compressed voice profile for the user.",
        input_schema=_VOICE_SCHEMA,
        model=model,
        max_tokens=1200,
        log_tag="voice",
    )

    return VoiceProfile(
        summary=result["summary"],
        style_examples=result.get("style_examples", []),
        user_email=user_email,
        learned_at=datetime.now(timezone.utc),
    )


def current_voice_profile(
    *,
    user_email: str | None = None,
    use_cache: bool = True,
    max_age_seconds: float | None = VOICE_CACHE_TTL_SECONDS,
) -> VoiceProfile:
    """Return the best available voice profile.

    Resolution order:
      1. If ``user_email`` is given and ``use_cache`` is true, look it up in
         the voice-profile cache. Hits older than ``max_age_seconds`` are
         treated as a miss.
      2. Otherwise (no email, no cache hit), return :data:`DEFAULT_VOICE`.

    We deliberately do *not* learn-on-miss here — that would turn every
    ``sift brief`` first-run into an extra-expensive call that also requires
    network. The explicit ``sift learn-voice`` command exists for that.
    """
    if use_cache and user_email:
        try:
            hit = cache.get_cached_voice_profile(user_email, max_age_seconds=max_age_seconds)
        except Exception:  # noqa: BLE001
            logger.exception("Voice cache lookup failed; falling back to DEFAULT_VOICE.")
            hit = None
        if hit is not None:
            return hit
    return DEFAULT_VOICE
