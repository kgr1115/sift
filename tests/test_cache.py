"""Unit tests for the SQLite cache layer.

No network, no LLM, no Gmail — tmp_path gives each test a fresh DB file and we
exercise the three entity tables independently. Things we want to be confident in:

* Round-trip: what you put in, you get back (Pydantic-equal).
* History-id invalidation: mismatched history_id returns a miss, not stale data.
* Last-write-wins: upsert with a new payload overwrites the old one.
* Admin ops: ``clear(table)`` and ``stats()`` do the right thing.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sift import cache
from sift.models import Category, Classification, Draft, Thread, VoiceProfile


@pytest.fixture(autouse=True)
def _isolate_connections():
    """Each test gets a fresh connection pool so tmp_path DBs don't bleed."""
    cache.close_all()
    yield
    cache.close_all()


def _make_thread(thread_id: str = "t1", subject: str = "Hello") -> Thread:
    return Thread(
        id=thread_id,
        from_="alice@example.com",
        from_name="Alice",
        to="kyle@example.com",
        subject=subject,
        received_at=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        body="Hi Kyle, quick question...",
    )


def _make_classification(thread_id: str = "t1", category: Category = Category.NEEDS_REPLY) -> Classification:
    return Classification(
        thread_id=thread_id,
        category=category,
        confidence=0.92,
        one_line_summary="Alice asks a quick question.",
        reason="Direct question addressed to Kyle.",
    )


def _make_draft(thread_id: str = "t1") -> Draft:
    return Draft(
        thread_id=thread_id,
        subject="Re: Hello",
        body="Hey Alice — yep, happy to help. Give me until Friday.",
        tone_notes="Casual, direct; matches Kyle's usual short-reply register.",
    )


class TestThreads:
    def test_miss_on_empty_db(self, tmp_path):
        assert cache.get_cached_thread("nope", db_path=tmp_path / "c.db") is None

    def test_round_trip(self, tmp_path):
        db = tmp_path / "c.db"
        original = _make_thread()
        cache.cache_thread(original, history_id="h1", db_path=db)

        restored = cache.get_cached_thread("t1", history_id="h1", db_path=db)
        assert restored == original

    def test_history_id_mismatch_is_miss(self, tmp_path):
        db = tmp_path / "c.db"
        cache.cache_thread(_make_thread(), history_id="h1", db_path=db)
        # Same thread_id, different history_id — treat as stale.
        assert cache.get_cached_thread("t1", history_id="h2", db_path=db) is None

    def test_history_id_unspecified_returns_regardless(self, tmp_path):
        db = tmp_path / "c.db"
        cache.cache_thread(_make_thread(), history_id="h1", db_path=db)
        assert cache.get_cached_thread("t1", db_path=db) is not None

    def test_upsert_last_write_wins(self, tmp_path):
        db = tmp_path / "c.db"
        cache.cache_thread(_make_thread(subject="First"), history_id="h1", db_path=db)
        cache.cache_thread(_make_thread(subject="Second"), history_id="h2", db_path=db)

        got = cache.get_cached_thread("t1", db_path=db)
        assert got is not None
        assert got.subject == "Second"


class TestClassifications:
    def test_round_trip(self, tmp_path):
        db = tmp_path / "c.db"
        original = _make_classification()
        cache.cache_classification(original, history_id="h1", model="claude-sonnet-4-6", provider="anthropic", db_path=db)

        restored = cache.get_cached_classification("t1", history_id="h1", db_path=db)
        assert restored == original

    def test_history_id_mismatch_is_miss(self, tmp_path):
        db = tmp_path / "c.db"
        cache.cache_classification(_make_classification(), history_id="h1", db_path=db)
        assert cache.get_cached_classification("t1", history_id="h2", db_path=db) is None


class TestDrafts:
    def test_round_trip(self, tmp_path):
        db = tmp_path / "c.db"
        original = _make_draft()
        cache.cache_draft(original, history_id="h1", model="claude-sonnet-4-6", provider="anthropic", db_path=db)

        restored = cache.get_cached_draft("t1", history_id="h1", db_path=db)
        assert restored == original


class TestVoiceProfiles:
    def _make_profile(self, email: str = "kyle@example.com") -> VoiceProfile:
        return VoiceProfile(
            summary="Short, lowercase replies. Signs 'Kyle' for personal, 'Best, Kyle' for work.",
            style_examples=["thanks!", "sounds good — shipping today."],
            user_email=email,
            learned_at=datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc),
        )

    def test_round_trip(self, tmp_path):
        db = tmp_path / "c.db"
        original = self._make_profile()
        cache.cache_voice_profile(original, db_path=db)

        restored = cache.get_cached_voice_profile("kyle@example.com", db_path=db)
        assert restored is not None
        assert restored.summary == original.summary
        assert restored.style_examples == original.style_examples
        assert restored.user_email == original.user_email

    def test_miss_returns_none(self, tmp_path):
        db = tmp_path / "c.db"
        assert cache.get_cached_voice_profile("nobody@example.com", db_path=db) is None

    def test_ttl_expiry_returns_none(self, tmp_path):
        db = tmp_path / "c.db"
        cache.cache_voice_profile(self._make_profile(), db_path=db)
        # The learned_at timestamp is set by the cache layer (_now_iso) at
        # write time, so asking for "< 0 seconds old" is guaranteed to miss.
        got = cache.get_cached_voice_profile(
            "kyle@example.com", max_age_seconds=-1.0, db_path=db
        )
        assert got is None

    def test_ttl_none_means_always_hit(self, tmp_path):
        db = tmp_path / "c.db"
        cache.cache_voice_profile(self._make_profile(), db_path=db)
        got = cache.get_cached_voice_profile("kyle@example.com", db_path=db)
        assert got is not None

    def test_cache_rejects_missing_email(self, tmp_path):
        db = tmp_path / "c.db"
        bad = VoiceProfile(summary="no email here", style_examples=[])
        with pytest.raises(ValueError, match="user_email"):
            cache.cache_voice_profile(bad, db_path=db)

    def test_upsert_last_write_wins(self, tmp_path):
        db = tmp_path / "c.db"
        first = self._make_profile()
        cache.cache_voice_profile(first, db_path=db)
        second = first.model_copy(update={"summary": "Replaced summary."})
        cache.cache_voice_profile(second, db_path=db)

        got = cache.get_cached_voice_profile("kyle@example.com", db_path=db)
        assert got is not None
        assert got.summary == "Replaced summary."


class TestAdmin:
    def test_stats_counts_each_table(self, tmp_path):
        db = tmp_path / "c.db"
        cache.cache_thread(_make_thread("t1"), db_path=db)
        cache.cache_thread(_make_thread("t2"), db_path=db)
        cache.cache_classification(_make_classification("t1"), db_path=db)
        cache.cache_draft(_make_draft("t1"), db_path=db)

        s = cache.stats(db_path=db)
        assert s == {"threads": 2, "classifications": 1, "drafts": 1, "voice_profiles": 0}

    def test_clear_single_table(self, tmp_path):
        db = tmp_path / "c.db"
        cache.cache_thread(_make_thread(), db_path=db)
        cache.cache_classification(_make_classification(), db_path=db)

        removed = cache.clear("threads", db_path=db)
        assert removed == 1
        assert cache.stats(db_path=db) == {
            "threads": 0, "classifications": 1, "drafts": 0, "voice_profiles": 0,
        }

    def test_clear_all_tables(self, tmp_path):
        db = tmp_path / "c.db"
        cache.cache_thread(_make_thread(), db_path=db)
        cache.cache_classification(_make_classification(), db_path=db)
        cache.cache_draft(_make_draft(), db_path=db)

        removed = cache.clear(db_path=db)
        assert removed == 3
        assert cache.stats(db_path=db) == {
            "threads": 0, "classifications": 0, "drafts": 0, "voice_profiles": 0,
        }

    def test_clear_unknown_table_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown cache table"):
            cache.clear("not_a_table", db_path=tmp_path / "c.db")


class TestInit:
    def test_init_db_creates_file(self, tmp_path):
        db = tmp_path / "subdir" / "sift.db"
        assert not db.exists()
        resolved = cache.init_db(db)
        assert resolved == db.resolve()
        assert db.exists()
        # Tables should exist.
        assert cache.stats(db_path=db) == {
            "threads": 0, "classifications": 0, "drafts": 0, "voice_profiles": 0,
        }
