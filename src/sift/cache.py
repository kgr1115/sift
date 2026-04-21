"""SQLite cache for fetched threads, classifications, and drafts.

Why a cache
-----------
Gmail's API is fast, but LLM calls aren't. Classifying 20 threads costs
~10–30 seconds of wall-clock and hits rate limits hard on tier-1 Anthropic
accounts (we observed ~50% of requests 429'ing on a 20-thread cold run).
Caching by thread id lets reruns skip all work for threads we've already
seen — the typical morning workflow is "run again 5 minutes later after
some new mail arrived", which should only re-classify the *delta*.

Invalidation
------------
Keyed by ``(thread_id, history_id)``. Gmail's ``historyId`` bumps on *any*
change to a thread — new message, label change, mark-read, archive — which
is exactly the semantics we want. If the caller doesn't know the historyId
(e.g. running against fixtures, or the initial fetch path), we key by
``thread_id`` alone and cache indefinitely. ``sift cache-clear`` wipes the
whole thing for a clean slate.

Schema
------
Separate tables per entity so we can evict one without nuking the others.
Payloads are Pydantic-serialized JSON — cheap to deserialize and
self-describing when inspecting the DB by hand via ``sqlite3 sift.db``.

Connection model
----------------
One cached ``sqlite3.Connection`` per resolved DB path, stored in a
module-level dict. SQLite opens are cheap but the schema bootstrap isn't
free, so keeping the connection alive avoids repeating that on every call.
``check_same_thread=False`` + SQLite's own locking lets the classifier's
``ThreadPoolExecutor`` write safely in parallel.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from .config import CONFIG, PROJECT_ROOT
from .models import Classification, Draft, Thread, VoiceProfile

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id    TEXT PRIMARY KEY,
    history_id   TEXT,
    payload_json TEXT NOT NULL,
    fetched_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classifications (
    thread_id     TEXT PRIMARY KEY,
    history_id    TEXT,
    payload_json  TEXT NOT NULL,
    classified_at TEXT NOT NULL,
    model         TEXT,
    provider      TEXT
);

CREATE TABLE IF NOT EXISTS drafts (
    thread_id    TEXT PRIMARY KEY,
    history_id   TEXT,
    payload_json TEXT NOT NULL,
    drafted_at   TEXT NOT NULL,
    model        TEXT,
    provider     TEXT
);

CREATE TABLE IF NOT EXISTS voice_profiles (
    user_email   TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    learned_at   TEXT NOT NULL,
    model        TEXT,
    provider     TEXT
);
"""

_VALID_TABLES = ("threads", "classifications", "drafts", "voice_profiles")

# Connection pool keyed by resolved DB path. Protected by _lock for dict ops;
# SQLite itself serializes writes on the underlying file.
_conn_cache: dict[str, sqlite3.Connection] = {}
_lock = Lock()


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------
def _resolve_db(path: Path | None = None) -> Path:
    p = path or CONFIG.db_path
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _conn(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a cached SQLite connection, initializing schema on first use."""
    resolved = _resolve_db(db_path)
    key = str(resolved)
    with _lock:
        conn = _conn_cache.get(key)
        if conn is None:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(key, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA)
            conn.commit()
            _conn_cache[key] = conn
            logger.debug("Initialized cache DB at %s", resolved)
        return conn


def init_db(db_path: Path | None = None) -> Path:
    """Explicitly initialize the cache DB. Returns the resolved path.

    Exposed mostly for the ``sift cache-clear`` command and for tests that
    want a fresh DB at a specific path. Every other function in this module
    implicitly initializes on first call.
    """
    _conn(db_path)
    return _resolve_db(db_path)


def close_all() -> None:
    """Close all cached connections. Tests call this between cases."""
    with _lock:
        for c in _conn_cache.values():
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        _conn_cache.clear()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------
def get_cached_thread(
    thread_id: str,
    *,
    history_id: str | None = None,
    db_path: Path | None = None,
) -> Thread | None:
    """Return a cached Thread, or None on miss / history_id mismatch."""
    row = _conn(db_path).execute(
        "SELECT history_id, payload_json FROM threads WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    if history_id is not None and row["history_id"] != history_id:
        return None
    return Thread.model_validate_json(row["payload_json"])


def cache_thread(
    thread: Thread,
    *,
    history_id: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Upsert a Thread by thread_id."""
    _conn(db_path).execute(
        """
        INSERT INTO threads (thread_id, history_id, payload_json, fetched_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            history_id   = excluded.history_id,
            payload_json = excluded.payload_json,
            fetched_at   = excluded.fetched_at
        """,
        (thread.id, history_id, thread.model_dump_json(by_alias=True), _now_iso()),
    )
    _conn(db_path).commit()


# ---------------------------------------------------------------------------
# Classifications
# ---------------------------------------------------------------------------
def get_cached_classification(
    thread_id: str,
    *,
    history_id: str | None = None,
    db_path: Path | None = None,
) -> Classification | None:
    row = _conn(db_path).execute(
        "SELECT history_id, payload_json FROM classifications WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    if history_id is not None and row["history_id"] != history_id:
        return None
    return Classification.model_validate_json(row["payload_json"])


def cache_classification(
    classification: Classification,
    *,
    history_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    db_path: Path | None = None,
) -> None:
    _conn(db_path).execute(
        """
        INSERT INTO classifications (thread_id, history_id, payload_json, classified_at, model, provider)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            history_id    = excluded.history_id,
            payload_json  = excluded.payload_json,
            classified_at = excluded.classified_at,
            model         = excluded.model,
            provider      = excluded.provider
        """,
        (
            classification.thread_id,
            history_id,
            classification.model_dump_json(),
            _now_iso(),
            model,
            provider,
        ),
    )
    _conn(db_path).commit()


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------
def get_cached_draft(
    thread_id: str,
    *,
    history_id: str | None = None,
    db_path: Path | None = None,
) -> Draft | None:
    row = _conn(db_path).execute(
        "SELECT history_id, payload_json FROM drafts WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    if history_id is not None and row["history_id"] != history_id:
        return None
    return Draft.model_validate_json(row["payload_json"])


def cache_draft(
    draft: Draft,
    *,
    history_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    db_path: Path | None = None,
) -> None:
    _conn(db_path).execute(
        """
        INSERT INTO drafts (thread_id, history_id, payload_json, drafted_at, model, provider)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            history_id   = excluded.history_id,
            payload_json = excluded.payload_json,
            drafted_at   = excluded.drafted_at,
            model        = excluded.model,
            provider     = excluded.provider
        """,
        (draft.thread_id, history_id, draft.model_dump_json(), _now_iso(), model, provider),
    )
    _conn(db_path).commit()


# ---------------------------------------------------------------------------
# Voice profiles
# ---------------------------------------------------------------------------
def get_cached_voice_profile(
    user_email: str,
    *,
    max_age_seconds: float | None = None,
    db_path: Path | None = None,
) -> VoiceProfile | None:
    """Return a cached voice profile for ``user_email``, or None.

    If ``max_age_seconds`` is provided and the stored profile is older than
    that, returns None (caller can treat as a miss and re-learn). This is the
    TTL knob for voice learning — writing styles drift over months, not
    minutes, so a weeklong default is fine.
    """
    row = _conn(db_path).execute(
        "SELECT payload_json, learned_at FROM voice_profiles WHERE user_email = ?",
        (user_email,),
    ).fetchone()
    if row is None:
        return None
    if max_age_seconds is not None:
        try:
            learned_at = datetime.fromisoformat(row["learned_at"])
        except ValueError:
            return None
        age = (datetime.now(timezone.utc) - learned_at).total_seconds()
        if age > max_age_seconds:
            return None
    return VoiceProfile.model_validate_json(row["payload_json"])


def cache_voice_profile(
    profile: VoiceProfile,
    *,
    model: str | None = None,
    provider: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Upsert a voice profile. ``profile.user_email`` must be set."""
    if not profile.user_email:
        raise ValueError("VoiceProfile.user_email is required to cache a profile.")
    _conn(db_path).execute(
        """
        INSERT INTO voice_profiles (user_email, payload_json, learned_at, model, provider)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_email) DO UPDATE SET
            payload_json = excluded.payload_json,
            learned_at   = excluded.learned_at,
            model        = excluded.model,
            provider     = excluded.provider
        """,
        (
            profile.user_email,
            profile.model_dump_json(),
            _now_iso(),
            model,
            provider,
        ),
    )
    _conn(db_path).commit()


# ---------------------------------------------------------------------------
# Admin: clear + stats
# ---------------------------------------------------------------------------
def clear(table: str | None = None, *, db_path: Path | None = None) -> int:
    """Delete cache entries; return the number of rows removed.

    If ``table`` is one of ``threads``, ``classifications``, ``drafts``,
    ``voice_profiles``, only that table is truncated. If ``None``, all are cleared.
    """
    conn = _conn(db_path)
    tables = list(_VALID_TABLES) if table is None else [table]
    total = 0
    for t in tables:
        if t not in _VALID_TABLES:
            raise ValueError(
                f"Unknown cache table: {t!r}. Expected one of {_VALID_TABLES}."
            )
        cur = conn.execute(f"DELETE FROM {t}")  # noqa: S608 — table name is validated above
        total += cur.rowcount
    conn.commit()
    return total


def stats(*, db_path: Path | None = None) -> dict[str, int]:
    """Return ``{table_name: row_count}`` for every cache table."""
    conn = _conn(db_path)
    out: dict[str, int] = {}
    for t in _VALID_TABLES:
        cur = conn.execute(f"SELECT COUNT(*) AS n FROM {t}")  # noqa: S608
        out[t] = cur.fetchone()["n"]
    return out
