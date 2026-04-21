"""Gmail connector: OAuth 2.0 flow, thread fetching, and draft pushing.

Design notes
------------
* **Scopes**: ``gmail.readonly`` + ``gmail.compose``. Read covers fetching
  threads; compose lets us create Drafts that show up in the user's Gmail
  Drafts folder. We deliberately do *not* request ``gmail.send`` — the user
  always sends the reply themselves after reviewing it.

* **Token storage**: credentials.json (OAuth client secret) sits at the repo
  root and is loaded once. token.json (user-granted access/refresh tokens)
  is written next to it after the first browser flow and reused thereafter.
  Both are in ``.gitignore``.

* **Refresh vs. re-auth**: google-auth handles silent refresh for expired
  access tokens. If the refresh token itself has been revoked or expired
  (testing-mode apps expire refresh tokens after 7 days), we fall back to
  re-running the browser flow.

* **Thread → Thread mapping**: Gmail's thread shape is deeply nested MIME.
  We flatten it to :class:`sift.models.Thread` by taking the last inbound
  message's sender/subject/date and joining the full message bodies into
  one text blob. That mirrors the shape of our fixture inbox and keeps the
  classifier/drafter code identical for real and synthetic inputs.

* **Relative paths**: ``credentials.json`` / ``token.json`` are resolved
  relative to :data:`sift.config.PROJECT_ROOT` if not absolute, so the
  ``sift`` CLI works regardless of the user's cwd.
"""

from __future__ import annotations

import base64
import html
import logging
import re
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .config import CONFIG, PROJECT_ROOT
from .models import Draft, Thread

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------
# Keep these narrow: read for triage/brief, compose so drafts land in the
# user's Gmail Drafts folder. No send, no modify, no delete.
SCOPES: list[str] = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _resolve(path: Path) -> Path:
    """Resolve a configured path against PROJECT_ROOT if it's relative."""
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def credentials_file() -> Path:
    return _resolve(CONFIG.google_credentials_path)


def token_file() -> Path:
    return _resolve(CONFIG.google_token_path)


class GmailAuthError(RuntimeError):
    """Raised when OAuth setup is incomplete or credentials are invalid."""


# ---------------------------------------------------------------------------
# OAuth flow
# ---------------------------------------------------------------------------
def _load_cached_creds() -> Credentials | None:
    tok = token_file()
    if not tok.exists():
        return None
    try:
        return Credentials.from_authorized_user_file(str(tok), SCOPES)
    except ValueError:
        # Scopes changed or file is malformed — force a fresh auth run.
        logger.warning("Cached token.json is stale (scopes changed?); re-authenticating.")
        return None


def _run_browser_flow() -> Credentials:
    """Kick off the desktop OAuth flow. Opens the user's browser."""
    creds_path = credentials_file()
    if not creds_path.exists():
        raise GmailAuthError(
            f"credentials.json not found at {creds_path}. "
            "See docs/gmail_setup.md to create a Google Cloud OAuth client "
            "and download the client secret."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
    # port=0 picks a free local port; the flow spins up a tiny HTTP server
    # to receive the OAuth redirect.
    return flow.run_local_server(port=0, prompt="consent")


def get_credentials(*, force_refresh: bool = False) -> Credentials:
    """Return valid OAuth credentials, running the browser flow if needed.

    Order of operations:
      1. Load token.json if present.
      2. If it's valid, return it (unless force_refresh).
      3. If it has a refresh token and is expired, try refreshing silently.
      4. Otherwise run the full browser flow.

    The resulting token is always written to token.json so subsequent calls
    don't need the browser.
    """
    creds = _load_cached_creds()

    if creds and creds.valid and not force_refresh:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _write_token(creds)
            return creds
        except Exception as e:  # noqa: BLE001 — any refresh failure means re-auth.
            logger.warning("Silent token refresh failed (%s); re-running browser flow.", e)

    creds = _run_browser_flow()
    _write_token(creds)
    return creds


def _write_token(creds: Credentials) -> None:
    tok = token_file()
    tok.parent.mkdir(parents=True, exist_ok=True)
    tok.write_text(creds.to_json(), encoding="utf-8")
    logger.info("Wrote Gmail token cache to %s", tok)


def get_service(*, creds: Credentials | None = None) -> Any:
    """Return a Gmail API service client (v1)."""
    c = creds or get_credentials()
    # cache_discovery=False silences a noisy warning on fresh envs.
    return build("gmail", "v1", credentials=c, cache_discovery=False)


def whoami(service: Any | None = None) -> str:
    """Return the authenticated user's primary email address."""
    svc = service or get_service()
    profile = svc.users().getProfile(userId="me").execute()
    return profile["emailAddress"]


# ---------------------------------------------------------------------------
# MIME body extraction
# ---------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _decode_b64url(data: str) -> str:
    """Gmail returns message bodies as base64url-encoded UTF-8."""
    if not data:
        return ""
    # Gmail uses URL-safe base64 without padding; pad before decoding.
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to decode base64url body")
        return ""


def _strip_html(raw: str) -> str:
    """Naive HTML → text. Good enough for classifier input.

    We deliberately avoid BeautifulSoup to keep deps small; the classifier
    doesn't need pixel-perfect text, just enough to read.
    """
    # Drop <script> / <style> blocks entirely.
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    # Turn <br>, </p>, </div> into newlines so paragraphs survive.
    raw = re.sub(r"<(br\s*/?|/p|/div|/li)>", "\n", raw, flags=re.IGNORECASE)
    # Strip remaining tags.
    raw = _HTML_TAG_RE.sub("", raw)
    # Unescape &amp; etc.
    raw = html.unescape(raw)
    # Collapse runs of whitespace.
    raw = _WHITESPACE_RE.sub(" ", raw)
    raw = _BLANK_LINES_RE.sub("\n\n", raw)
    return raw.strip()


def _extract_body(payload: dict[str, Any]) -> str:
    """Walk a Gmail payload tree and pull out the best plain-text body.

    Preference order: text/plain > text/html (stripped) > empty.
    Visits every part; a sibling ``text/plain`` beats an ancestor's HTML.
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def walk(part: dict[str, Any]) -> None:
        mime = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")
        if mime == "text/plain" and data:
            plain_parts.append(_decode_b64url(data))
        elif mime == "text/html" and data:
            html_parts.append(_decode_b64url(data))
        for child in part.get("parts", []) or []:
            walk(child)

    walk(payload)

    if plain_parts:
        return "\n\n".join(p.strip() for p in plain_parts if p.strip())
    if html_parts:
        return _strip_html("\n\n".join(html_parts))
    return ""


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------
def _header(msg: dict[str, Any], name: str) -> str:
    """Case-insensitive header lookup on a Gmail message resource."""
    for h in msg.get("payload", {}).get("headers", []) or []:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _parse_date(raw: str) -> datetime:
    """Parse an RFC 2822 Date header into an aware UTC datetime.

    ``email.utils.parsedate_to_datetime`` returns a *naive* datetime for
    headers with a ``-0000`` timezone (spec: "UTC but origin unknown") and
    for malformed/missing timezones. Sorting a batch that mixes aware and
    naive datetimes raises ``TypeError: can't compare offset-naive and
    offset-aware datetimes`` — so we always coerce to aware UTC here and
    keep the caller simple.
    """
    if not raw:
        return datetime.now(timezone.utc)
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        # Naive -> treat as UTC. parsedate_to_datetime already converts the
        # timestamp to UTC wall-clock for -0000 inputs, so this is faithful.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _split_from(raw: str) -> tuple[str, str]:
    """Return (display_name, email_addr) from a "Name <addr@x>" header."""
    name, addr = parseaddr(raw)
    return (name or addr.split("@")[0], addr)


# ---------------------------------------------------------------------------
# Thread fetching
# ---------------------------------------------------------------------------
def _thread_to_model(thread_resource: dict[str, Any], self_email: str) -> Thread | None:
    """Convert a Gmail thread resource into our Thread model.

    Returns None if the thread is effectively empty or contains only messages
    the user sent (nothing inbound to classify).
    """
    messages = thread_resource.get("messages", []) or []
    if not messages:
        return None

    # Find the most recent *inbound* message — one where the user isn't the
    # primary sender. If every message is from the user (rare but possible
    # for INBOX threads via filters), we skip it.
    inbound = [m for m in messages if self_email.lower() not in _header(m, "From").lower()]
    primary = inbound[-1] if inbound else messages[-1]
    if not inbound:
        # Nothing to classify — this thread only contains outbound messages.
        return None

    from_name, from_addr = _split_from(_header(primary, "From"))
    to_addr = _header(primary, "To") or self_email
    subject = _header(primary, "Subject") or "(no subject)"
    date = _parse_date(_header(primary, "Date"))

    # Body: join all messages in chronological order so Claude sees full
    # context for reply-y threads. Prefix each with its sender so the LLM
    # can tell who said what.
    body_chunks: list[str] = []
    for m in messages:
        who, _ = _split_from(_header(m, "From"))
        body = _extract_body(m.get("payload", {}))
        if not body:
            continue
        body_chunks.append(f"[{who}]\n{body}")
    body = "\n\n---\n\n".join(body_chunks) if body_chunks else thread_resource.get("snippet", "")

    return Thread(
        id=thread_resource["id"],
        **{"from": from_addr},
        from_name=from_name,
        to=to_addr,
        subject=subject,
        received_at=date,
        body=body,
    )


def fetch_recent_threads(
    limit: int = 25,
    *,
    query: str | None = None,
    label_ids: list[str] | None = None,
    service: Any | None = None,
) -> list[Thread]:
    """Fetch up to ``limit`` recent threads from the authenticated inbox.

    Parameters
    ----------
    limit :
        Upper bound on threads returned. Defaults to 25 to keep demos cheap.
    query :
        Optional Gmail search query (e.g. ``"is:unread newer_than:2d"``).
        Takes precedence over ``label_ids`` when both are set.
    label_ids :
        Gmail label ids to filter by; defaults to ``["INBOX"]`` which hides
        sent-only threads and spam. Ignored if ``query`` is supplied.
    """
    svc = service or get_service()
    self_email = whoami(svc)

    list_kwargs: dict[str, Any] = {"userId": "me", "maxResults": limit}
    if query:
        list_kwargs["q"] = query
    else:
        list_kwargs["labelIds"] = label_ids or ["INBOX"]

    try:
        resp = svc.users().threads().list(**list_kwargs).execute()
    except HttpError as e:
        raise GmailAuthError(f"Gmail API error while listing threads: {e}") from e

    thread_stubs = resp.get("threads", []) or []
    logger.info("Gmail returned %d thread stubs (limit=%d)", len(thread_stubs), limit)

    threads: list[Thread] = []
    for stub in thread_stubs:
        tid = stub["id"]
        try:
            full = svc.users().threads().get(userId="me", id=tid, format="full").execute()
        except HttpError as e:
            logger.warning("Skipping thread %s (fetch failed: %s)", tid, e)
            continue
        model = _thread_to_model(full, self_email)
        if model is not None:
            threads.append(model)

    # Newest first, matching what users expect in a morning brief.
    threads.sort(key=lambda t: t.received_at, reverse=True)
    return threads


# ---------------------------------------------------------------------------
# Draft push
# ---------------------------------------------------------------------------
def _get_thread_raw(service: Any, thread_id: str) -> dict[str, Any]:
    return service.users().threads().get(userId="me", id=thread_id, format="metadata",
                                         metadataHeaders=["From", "To", "Cc", "Subject",
                                                          "Message-ID", "References",
                                                          "In-Reply-To"]).execute()


def _build_reply_mime(
    *,
    to_addr: str,
    from_addr: str,
    subject: str,
    body: str,
    in_reply_to: str,
    references: str,
) -> str:
    """Construct an RFC 2822 reply and return it base64url-encoded.

    Gmail's drafts.create wants ``raw`` = url-safe base64 of the full
    message (headers + body). We use email.message.EmailMessage so quoting,
    line folding and charset handling are correct.
    """
    msg = EmailMessage()
    msg["To"] = to_addr
    msg["From"] = from_addr
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    msg["Subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    # References should accumulate: existing refs + parent Message-ID.
    refs = " ".join(s for s in [references, in_reply_to] if s).strip()
    if refs:
        msg["References"] = refs
    msg.set_content(body)

    raw_bytes = msg.as_bytes()
    return base64.urlsafe_b64encode(raw_bytes).decode("ascii")


def push_draft(
    draft: Draft,
    *,
    service: Any | None = None,
) -> str:
    """Push a drafted reply into the user's Gmail Drafts folder.

    Looks up the original thread's most recent message to pull the correct
    reply headers (To, In-Reply-To, References) and Gmail threadId, so the
    draft appears as a reply inside the original thread.

    Returns the draft's Gmail id.
    """
    svc = service or get_service()
    me = whoami(svc)

    # Fetch the thread's headers so we know who to reply to and with what
    # threading headers. Using format=metadata is lighter than 'full'.
    thread = _get_thread_raw(svc, draft.thread_id)
    messages = thread.get("messages", []) or []
    if not messages:
        raise GmailAuthError(f"Thread {draft.thread_id} has no messages; can't draft reply.")
    # Reply to the most recent inbound message (the one that's waiting on us).
    inbound = [m for m in messages if me.lower() not in _header(m, "From").lower()]
    parent = inbound[-1] if inbound else messages[-1]

    _from_name, from_addr = _split_from(_header(parent, "From"))
    message_id = _header(parent, "Message-ID")
    references = _header(parent, "References")

    raw_b64 = _build_reply_mime(
        to_addr=from_addr,
        from_addr=me,
        subject=draft.subject,
        body=draft.body,
        in_reply_to=message_id,
        references=references,
    )

    body = {"message": {"raw": raw_b64, "threadId": draft.thread_id}}
    created = svc.users().drafts().create(userId="me", body=body).execute()
    draft_id = created.get("id", "")
    logger.info("Created Gmail draft %s for thread %s", draft_id, draft.thread_id)
    return draft_id


# ---------------------------------------------------------------------------
# Sent-message fetching (for voice learning)
# ---------------------------------------------------------------------------
def fetch_sent_messages(
    limit: int = 50,
    *,
    service: Any | None = None,
) -> list[dict[str, str]]:
    """Fetch up to ``limit`` recent messages from the user's Sent folder.

    Returns a list of dicts with ``subject``, ``to``, ``body`` keys. Messages
    with empty bodies or that fail to fetch are skipped; a short batch is
    better than a hard failure for voice learning.

    Unlike :func:`fetch_recent_threads` this operates on individual messages
    (``users.messages.list`` with ``labelIds=["SENT"]``) because for voice
    learning we want *just* the user's replies, not the inbound messages that
    preceded them in a thread.
    """
    svc = service or get_service()
    try:
        resp = svc.users().messages().list(
            userId="me", labelIds=["SENT"], maxResults=limit
        ).execute()
    except HttpError as e:
        raise GmailAuthError(f"Gmail API error while listing sent messages: {e}") from e

    stubs = resp.get("messages", []) or []
    logger.info("Gmail returned %d sent-message stubs (limit=%d)", len(stubs), limit)

    out: list[dict[str, str]] = []
    for stub in stubs:
        mid = stub["id"]
        try:
            full = svc.users().messages().get(userId="me", id=mid, format="full").execute()
        except HttpError as e:
            logger.warning("Skipping sent message %s (fetch failed: %s)", mid, e)
            continue
        body = _extract_body(full.get("payload", {}))
        if not body.strip():
            continue
        out.append(
            {
                "subject": _header(full, "Subject"),
                "to": _header(full, "To"),
                "body": body,
            }
        )
    return out


def push_drafts(drafts: list[Draft], *, service: Any | None = None) -> dict[str, str]:
    """Push many drafts; returns {thread_id: gmail_draft_id}.

    Errors on individual drafts are logged but don't abort the batch — a
    failed push for one thread shouldn't lose the other drafts.
    """
    svc = service or get_service()
    out: dict[str, str] = {}
    for d in drafts:
        try:
            out[d.thread_id] = push_draft(d, service=svc)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to push draft for thread %s", d.thread_id)
    return out
