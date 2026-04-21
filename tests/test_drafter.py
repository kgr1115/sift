"""Unit tests for the drafter — no LLM, no Gmail.

Scope:
* ``is_no_reply_sender`` catches the common automated-sender patterns and
  doesn't false-positive on real human/service addresses that might be worth
  replying to.
* ``draft_replies`` skips no-reply senders entirely (no LLM call, no draft in
  the returned dict) while still passing through normal threads.
* ``_render_thread`` labels Kyle's address as ``To:`` so the prompt can't
  confuse the recipient with the sender — this is the prompt-side half of
  the sender/recipient fix we shipped for the Anthropic-billing-thread bug.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from sift import drafter
from sift.drafter import is_no_reply_sender
from sift.models import Category, Classification, Draft, Thread


def _make_thread(
    thread_id: str = "t1",
    from_: str = "alice@example.com",
    from_name: str = "Alice",
    subject: str = "Quick question",
) -> Thread:
    return Thread(
        id=thread_id,
        from_=from_,
        from_name=from_name,
        to="kyle@example.com",
        subject=subject,
        received_at=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        body="Hi Kyle, can we move our 1:1 to Thursday?",
    )


def _cls(thread_id: str, category: Category = Category.NEEDS_REPLY) -> Classification:
    return Classification(
        thread_id=thread_id,
        category=category,
        confidence=0.9,
        one_line_summary="They want to move the 1:1.",
        reason="Direct ask with a question mark.",
    )


class TestIsNoReplySender:
    @pytest.mark.parametrize(
        "address",
        [
            "noreply@example.com",
            "no-reply@example.com",
            "no_reply@example.com",
            "No-Reply@Example.Com",  # case-insensitive
            "donotreply@example.com",
            "do-not-reply@example.com",
            "notifications@github.com",
            "notification@service.io",
            "mailer-daemon@mail.example.com",
            "bounce@lists.example.com",
            "bounces@lists.example.com",
            "postmaster@example.com",
            "auto-reply@vendor.com",
            "autoconfirm@vendor.com",
            "noreply+tag@example.com",  # plus-addressed variants
        ],
    )
    def test_matches_automated_senders(self, address):
        assert is_no_reply_sender(address) is True

    @pytest.mark.parametrize(
        "address",
        [
            "alice@example.com",
            "kyle.g.rauch@gmail.com",
            "recruiter@bigco.com",
            "support@vendor.com",   # support can accept human replies
            "billing@vendor.com",   # billing too
            "info@startup.io",
            "hello@newsletter.com",
            "team@startup.io",
            "",                     # empty string is not a match
            "not-an-email",         # no @ → not a match
        ],
    )
    def test_does_not_match_normal_senders(self, address):
        assert is_no_reply_sender(address) is False


class TestDraftRepliesSkipsNoReply:
    def test_skips_no_reply_sender(self):
        # Two threads: one from a real person, one from noreply@. Classifier
        # says both need_reply. Drafter should only draft for the real person.
        real = _make_thread(thread_id="real", from_="alice@example.com")
        noreply = _make_thread(
            thread_id="auto",
            from_="no-reply@anthropic.com",
            from_name="Anthropic",
            subject="Payment failed",
        )
        classifications = [_cls("real"), _cls("auto", category=Category.URGENT)]

        fake_draft = Draft(
            thread_id="real", subject="Re: Quick question",
            body="sounds good", tone_notes="casual",
        )

        # Bypass the cache entirely; patch draft_reply so we can assert it
        # was only invoked for the human-sender thread.
        with patch("sift.drafter.draft_reply", return_value=fake_draft) as mock_draft:
            drafts = drafter.draft_replies(
                [real, noreply], classifications, use_cache=False,
            )

        # Only one LLM call — for the human sender.
        assert mock_draft.call_count == 1
        assert mock_draft.call_args.args[0].id == "real"
        assert "real" in drafts
        assert "auto" not in drafts

    def test_classifier_misses_are_ignored(self):
        # If no classification exists for a thread, it shouldn't get drafted.
        t = _make_thread()
        with patch("sift.drafter.draft_reply") as mock_draft:
            drafts = drafter.draft_replies([t], [], use_cache=False)
            mock_draft.assert_not_called()
        assert drafts == {}

    def test_non_draft_categories_are_skipped(self):
        # FYI / newsletter / trash shouldn't produce drafts.
        t = _make_thread()
        classifications = [_cls(t.id, category=Category.FYI)]
        with patch("sift.drafter.draft_reply") as mock_draft:
            drafts = drafter.draft_replies([t], classifications, use_cache=False)
            mock_draft.assert_not_called()
        assert drafts == {}


class TestRenderThread:
    def test_labels_recipient_explicitly(self):
        """The To: line should call out Kyle as the reply sender.

        This is the prompt-side half of the sender/recipient fix. Without
        this, drafts against company emails sometimes come back written AS
        the company (the Anthropic-billing bug).
        """
        t = _make_thread(from_="billing@anthropic.com", from_name="Anthropic")
        rendered = drafter._render_thread(t, recipient_email="kyle.g.rauch@gmail.com")
        # Kyle's email appears as To:
        assert "kyle.g.rauch@gmail.com" in rendered
        # And we explicitly flag it
        assert "you are Kyle, replying TO the sender" in rendered
        # Sender appears as From:
        assert "From: Anthropic <billing@anthropic.com>" in rendered

    def test_falls_back_to_thread_to_when_no_email(self):
        t = _make_thread()
        rendered = drafter._render_thread(t, recipient_email=None)
        # Thread.to is "kyle@example.com" in the fixture
        assert "kyle@example.com" in rendered
