"""Pure-unit tests for the Gmail client.

Everything in this file is offline: we exercise the MIME parser, header
helpers, and reply-MIME builder without touching the Gmail API. An end-to-end
OAuth test lives separately and is gated on a cached ``token.json``.
"""

from __future__ import annotations

import base64
import email
import email.policy
import sys
from pathlib import Path

import pytest

# Make the package importable without a pip install, matching evals/conftest.py.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sift import gmail_client  # noqa: E402
from sift.models import Draft  # noqa: E402


def _b64url(s: str) -> str:
    """Encode a string the way Gmail does (URL-safe base64, no padding stripped)."""
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii")


# ---------------------------------------------------------------------------
# _decode_b64url
# ---------------------------------------------------------------------------
class TestDecodeB64Url:
    def test_roundtrip_plain(self) -> None:
        assert gmail_client._decode_b64url(_b64url("hello world")) == "hello world"

    def test_handles_missing_padding(self) -> None:
        # Gmail sometimes strips the trailing '=' padding.
        raw = _b64url("hi").rstrip("=")
        assert gmail_client._decode_b64url(raw) == "hi"

    def test_empty_input(self) -> None:
        assert gmail_client._decode_b64url("") == ""

    def test_utf8(self) -> None:
        assert gmail_client._decode_b64url(_b64url("café — naïve")) == "café — naïve"


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------
class TestStripHtml:
    def test_basic_tags(self) -> None:
        got = gmail_client._strip_html("<p>Hi <b>there</b></p>")
        assert got == "Hi there"

    def test_br_becomes_newline(self) -> None:
        got = gmail_client._strip_html("Line one<br>Line two<br/>Line three")
        assert "Line one" in got and "Line two" in got and "Line three" in got
        assert "<br" not in got

    def test_script_and_style_dropped(self) -> None:
        raw = "<style>.x{color:red}</style><script>alert(1)</script><p>body</p>"
        got = gmail_client._strip_html(raw)
        assert "alert" not in got
        assert "color" not in got
        assert "body" in got

    def test_entities_unescaped(self) -> None:
        assert gmail_client._strip_html("A &amp; B &lt;3") == "A & B <3"


# ---------------------------------------------------------------------------
# _extract_body — prefers text/plain, falls back to HTML
# ---------------------------------------------------------------------------
class TestExtractBody:
    def test_simple_plain(self) -> None:
        payload = {"mimeType": "text/plain", "body": {"data": _b64url("just text")}}
        assert gmail_client._extract_body(payload) == "just text"

    def test_multipart_prefers_plain_over_html(self) -> None:
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": _b64url("plain version")}},
                {"mimeType": "text/html", "body": {"data": _b64url("<p>html version</p>")}},
            ],
        }
        assert gmail_client._extract_body(payload) == "plain version"

    def test_html_fallback_when_no_plain(self) -> None:
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/html", "body": {"data": _b64url("<p>hello <b>world</b></p>")}},
            ],
        }
        got = gmail_client._extract_body(payload)
        assert "hello" in got and "world" in got
        assert "<" not in got

    def test_nested_multipart(self) -> None:
        # Real Gmail threads often have multipart/mixed wrapping multipart/alternative.
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": _b64url("nested plain")}},
                        {"mimeType": "text/html", "body": {"data": _b64url("<p>nested html</p>")}},
                    ],
                },
                # Attachment part — no inline body.
                {"mimeType": "application/pdf", "body": {"attachmentId": "abc"}},
            ],
        }
        assert gmail_client._extract_body(payload) == "nested plain"

    def test_empty_payload(self) -> None:
        assert gmail_client._extract_body({}) == ""


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------
class TestHeaderHelpers:
    def test_header_case_insensitive(self) -> None:
        msg = {"payload": {"headers": [{"name": "Subject", "value": "Hi"}]}}
        assert gmail_client._header(msg, "subject") == "Hi"
        assert gmail_client._header(msg, "SUBJECT") == "Hi"

    def test_header_missing(self) -> None:
        assert gmail_client._header({"payload": {}}, "From") == ""

    def test_split_from_with_display_name(self) -> None:
        name, addr = gmail_client._split_from('"Alice Smith" <alice@example.com>')
        assert name == "Alice Smith"
        assert addr == "alice@example.com"

    def test_split_from_bare_address(self) -> None:
        name, addr = gmail_client._split_from("bob@example.com")
        assert addr == "bob@example.com"
        # Fallback uses the local-part as a display name.
        assert name == "bob"

    def test_parse_date_rfc2822(self) -> None:
        dt = gmail_client._parse_date("Mon, 20 Apr 2026 09:15:30 -0400")
        assert dt.year == 2026 and dt.month == 4 and dt.day == 20
        assert dt.tzinfo is not None  # always aware

    def test_parse_date_bad_falls_back_to_now(self) -> None:
        dt = gmail_client._parse_date("not a date")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_parse_date_minus_zero_zero_is_aware(self) -> None:
        # -0000 means "UTC but origin unknown"; parsedate_to_datetime returns
        # a *naive* datetime here. We must coerce to aware so that sorting a
        # mixed batch doesn't blow up on "can't compare naive and aware".
        dt = gmail_client._parse_date("Mon, 20 Apr 2026 09:15:30 -0000")
        assert dt.tzinfo is not None

    def test_parse_date_sort_mixed_batch(self) -> None:
        # Regression: the bug that crashed push-drafts on a 20-thread batch.
        dts = [
            gmail_client._parse_date("Mon, 20 Apr 2026 09:15:30 -0400"),  # aware
            gmail_client._parse_date("Mon, 20 Apr 2026 09:15:30 -0000"),  # previously naive
            gmail_client._parse_date("not a date"),  # fallback -> aware
        ]
        # Must not raise.
        sorted(dts)


# ---------------------------------------------------------------------------
# _thread_to_model
# ---------------------------------------------------------------------------
class TestThreadToModel:
    def _make_msg(self, *, from_: str, to: str, subject: str, date: str, body: str) -> dict:
        return {
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": from_},
                    {"name": "To", "value": to},
                    {"name": "Subject", "value": subject},
                    {"name": "Date", "value": date},
                ],
                "body": {"data": _b64url(body)},
            }
        }

    def test_single_inbound_message(self) -> None:
        thread = {
            "id": "T1",
            "messages": [
                self._make_msg(
                    from_="Alice <alice@example.com>",
                    to="me@example.com",
                    subject="Question",
                    date="Mon, 20 Apr 2026 09:15:30 -0400",
                    body="Can you review this?",
                )
            ],
        }
        model = gmail_client._thread_to_model(thread, self_email="me@example.com")
        assert model is not None
        assert model.id == "T1"
        assert model.from_name == "Alice"
        assert model.from_ == "alice@example.com"
        assert model.subject == "Question"
        assert "Can you review this?" in model.body

    def test_outbound_only_thread_is_skipped(self) -> None:
        thread = {
            "id": "T2",
            "messages": [
                self._make_msg(
                    from_="Me <me@example.com>",
                    to="alice@example.com",
                    subject="Following up",
                    date="Mon, 20 Apr 2026 09:15:30 -0400",
                    body="Hey, following up.",
                )
            ],
        }
        assert gmail_client._thread_to_model(thread, self_email="me@example.com") is None

    def test_multi_message_joins_bodies(self) -> None:
        thread = {
            "id": "T3",
            "messages": [
                self._make_msg(
                    from_="Alice <alice@example.com>",
                    to="me@example.com",
                    subject="Plan",
                    date="Mon, 20 Apr 2026 09:00:00 -0400",
                    body="First ask.",
                ),
                self._make_msg(
                    from_="Me <me@example.com>",
                    to="alice@example.com",
                    subject="Re: Plan",
                    date="Mon, 20 Apr 2026 09:30:00 -0400",
                    body="My thoughts.",
                ),
                self._make_msg(
                    from_="Alice <alice@example.com>",
                    to="me@example.com",
                    subject="Re: Plan",
                    date="Mon, 20 Apr 2026 10:00:00 -0400",
                    body="Follow-up question.",
                ),
            ],
        }
        model = gmail_client._thread_to_model(thread, self_email="me@example.com")
        assert model is not None
        # Last inbound message drives top-level subject/sender.
        assert model.from_ == "alice@example.com"
        # Body contains every message, in order, with sender tags.
        assert "First ask." in model.body
        assert "My thoughts." in model.body
        assert "Follow-up question." in model.body
        assert model.body.index("First ask.") < model.body.index("Follow-up question.")


# ---------------------------------------------------------------------------
# _build_reply_mime
# ---------------------------------------------------------------------------
class TestBuildReplyMime:
    def test_preserves_reply_headers(self) -> None:
        raw_b64 = gmail_client._build_reply_mime(
            to_addr="alice@example.com",
            from_addr="me@example.com",
            subject="Budget question",
            body="Thanks — here's what I think.",
            in_reply_to="<abc123@mail.example.com>",
            references="<prev1@mail.example.com>",
        )
        # Decode and parse the resulting RFC 2822 message. policy.default
        # gives us an EmailMessage (which has get_content()) rather than the
        # legacy Message type (which doesn't, especially on Py 3.13+).
        raw = base64.urlsafe_b64decode(raw_b64.encode("ascii"))
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        assert msg["To"] == "alice@example.com"
        assert msg["From"] == "me@example.com"
        assert msg["Subject"] == "Re: Budget question"  # auto-prefixed
        assert msg["In-Reply-To"] == "<abc123@mail.example.com>"
        # References should accumulate: prior refs + in-reply-to.
        assert "<prev1@mail.example.com>" in msg["References"]
        assert "<abc123@mail.example.com>" in msg["References"]
        body = msg.get_content()
        assert "Thanks" in body

    def test_doesnt_double_prefix_re(self) -> None:
        raw_b64 = gmail_client._build_reply_mime(
            to_addr="a@x.com",
            from_addr="b@x.com",
            subject="Re: already replied",
            body="hi",
            in_reply_to="",
            references="",
        )
        raw = base64.urlsafe_b64decode(raw_b64.encode("ascii"))
        msg = email.message_from_bytes(raw)
        assert msg["Subject"] == "Re: already replied"


# ---------------------------------------------------------------------------
# Integration smoke test — only runs when the user has completed OAuth
# ---------------------------------------------------------------------------
_TOKEN_EXISTS = gmail_client.token_file().exists()


@pytest.mark.skipif(not _TOKEN_EXISTS, reason="No cached Gmail token.json — run `sift auth` to enable.")
def test_gmail_integration_whoami() -> None:
    """Sanity check: cached token still works and returns a Gmail address."""
    email_addr = gmail_client.whoami()
    assert "@" in email_addr


@pytest.mark.skipif(not _TOKEN_EXISTS, reason="No cached Gmail token.json — run `sift auth` to enable.")
def test_gmail_integration_fetch_smoke() -> None:
    """Fetch 1 thread end-to-end; confirm it parses into our Thread model."""
    threads = gmail_client.fetch_recent_threads(limit=1)
    # Might be 0 if the inbox is empty, but never a crash.
    for t in threads:
        assert t.id
        assert t.from_
        assert t.subject is not None


# ---------------------------------------------------------------------------
# Draft model compatibility — pure unit, no network
# ---------------------------------------------------------------------------
def test_draft_model_builds_mime_without_error() -> None:
    """Regression: a realistic Draft instance must pass through the MIME builder."""
    draft = Draft(
        thread_id="T1",
        subject="Re: Question",
        body="Sure, here's my take.\n\n— Kyle",
        tone_notes="warm, concise",
    )
    raw_b64 = gmail_client._build_reply_mime(
        to_addr="alice@example.com",
        from_addr="kyle@example.com",
        subject=draft.subject,
        body=draft.body,
        in_reply_to="<abc@x>",
        references="",
    )
    # Valid url-safe base64.
    base64.urlsafe_b64decode(raw_b64.encode("ascii"))
