"""Unit tests for voice learning and profile resolution.

No Gmail, no LLM — we mock sift.voice.structured_call so the learner's code
path is exercised without a network dependency. Goal: be confident that:

* current_voice_profile returns DEFAULT_VOICE when no email / no cache hit.
* current_voice_profile returns the cached profile when one exists.
* learn_voice_profile round-trips a fake LLM response into a VoiceProfile.
* An empty batch short-circuits to DEFAULT_VOICE (no prompt send).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from sift import cache, voice
from sift.models import VoiceProfile


@pytest.fixture(autouse=True)
def _fresh_cache():
    cache.close_all()
    yield
    cache.close_all()


@pytest.fixture
def tmp_cache_db(tmp_path, monkeypatch):
    """Point the cache module at a tmp-path SQLite DB.

    ``CONFIG`` is a ``frozen=True`` dataclass so we can't poke its fields; we
    swap the module-level reference for a copy with the right db_path instead.
    ``_resolve_db`` reads ``CONFIG.db_path`` at call time, so this works.
    """
    db = tmp_path / "v.db"
    monkeypatch.setattr(cache, "CONFIG", replace(cache.CONFIG, db_path=db))
    return db


class TestCurrentProfile:
    def test_no_email_returns_default(self):
        # With no user_email the resolver can't check the cache at all.
        assert voice.current_voice_profile() is voice.DEFAULT_VOICE

    def test_cache_miss_returns_default(self, tmp_cache_db):
        got = voice.current_voice_profile(user_email="nobody@example.com")
        assert got is voice.DEFAULT_VOICE

    def test_cache_hit_returns_cached(self, tmp_cache_db):
        cached = VoiceProfile(
            summary="Cached summary.",
            style_examples=["hi!"],
            user_email="kyle@example.com",
            learned_at=datetime.now(timezone.utc),
        )
        cache.cache_voice_profile(cached)

        got = voice.current_voice_profile(user_email="kyle@example.com")
        assert got.summary == "Cached summary."
        assert got.user_email == "kyle@example.com"

    def test_use_cache_false_bypasses_cache(self, tmp_cache_db):
        cache.cache_voice_profile(
            VoiceProfile(
                summary="Should be ignored.",
                style_examples=[],
                user_email="kyle@example.com",
                learned_at=datetime.now(timezone.utc),
            )
        )
        got = voice.current_voice_profile(user_email="kyle@example.com", use_cache=False)
        assert got is voice.DEFAULT_VOICE


class TestLearnProfile:
    def test_empty_batch_short_circuits(self):
        # Should return DEFAULT_VOICE without invoking the LLM at all.
        with patch("sift.voice.structured_call") as mock_call:
            got = voice.learn_voice_profile([], user_email="kyle@example.com")
            mock_call.assert_not_called()
        assert got.summary == voice.DEFAULT_VOICE.summary
        assert got.user_email == "kyle@example.com"
        assert got.learned_at is not None

    def test_round_trips_llm_response(self):
        fake_response = {
            "summary": "Short, lowercase, no salutation. Signs 'Kyle'.",
            "style_examples": ["thanks!", "on it.", "will do."],
        }
        with patch("sift.voice.structured_call", return_value=fake_response) as mock_call:
            profile = voice.learn_voice_profile(
                [{"subject": "Re: X", "to": "a@b.com", "body": "thanks!"}],
                user_email="kyle@example.com",
            )
            assert mock_call.called
        assert profile.summary == fake_response["summary"]
        assert profile.style_examples == fake_response["style_examples"]
        assert profile.user_email == "kyle@example.com"
        assert profile.learned_at is not None

    def test_rendered_prompt_includes_subject_and_body(self):
        """Sanity check that the batch renderer reaches the LLM call."""
        captured: dict[str, str] = {}

        def fake_call(*, system, user, **_kw):  # noqa: ARG001
            captured["user"] = user
            return {"summary": "ok", "style_examples": []}

        with patch("sift.voice.structured_call", side_effect=fake_call):
            voice.learn_voice_profile(
                [
                    {"subject": "Lunch Friday?", "to": "a@b.com", "body": "sounds good"},
                    {"subject": "Re: Q2", "to": "c@d.com", "body": "will look tomorrow"},
                ],
                user_email="kyle@example.com",
            )

        assert "Lunch Friday?" in captured["user"]
        assert "sounds good" in captured["user"]
        assert "Re: Q2" in captured["user"]
