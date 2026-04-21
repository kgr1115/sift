"""Command-line interface for Sift.

Usage:
  sift brief --source fixtures         # run against the synthetic inbox
  sift brief --source gmail            # run against real Gmail (requires OAuth)
  sift draft <thread_id>               # draft a single reply

Built on Typer for nice help output. Stays thin on purpose — real logic lives
in the library modules so it's easy to eval and unit-test.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown

from . import cache
from .brief import build_brief, render_brief, render_brief_llm
from .classifier import classify_threads
from .drafter import draft_replies, draft_reply
from .fixtures import load_labeled_threads
from .models import Thread

# gmail_client pulls in google-* libraries; import inside handlers to keep
# `sift brief --source fixtures` fast and network-free for people who haven't
# set up OAuth yet.

# Windows terminals default to cp1252 for stdout, which can't encode the emoji
# our brief renderer uses (🔥, ↩️, etc.) and rich's legacy_windows renderer
# calls SetConsoleTextAttribute + a cp1252 encode that raises UnicodeEncodeError
# on those codepoints. Two belt-and-suspenders fixes:
#   1. Reconfigure stdout/stderr to UTF-8 with errors='replace' so worst-case
#      we get ? instead of a crash.
#   2. Force rich to the ANSI renderer path (legacy_windows=False) so it
#      bypasses the SetConsoleTextAttribute write entirely.
# Modern Windows Terminal / ConPTY handles ANSI + UTF-8 out of the box; older
# cmd.exe with a non-65001 codepage may show mojibake but will not crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and _stream.encoding.lower() != "utf-8":
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

app = typer.Typer(help="Sift — AI inbox triage, drafts, and a morning brief.", no_args_is_help=True)
console = Console(legacy_windows=False) if sys.platform == "win32" else Console()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class Source(str, Enum):
    fixtures = "fixtures"
    gmail = "gmail"


def _load_threads(
    source: Source,
    *,
    limit: int = 25,
    query: str | None = None,
) -> list[Thread]:
    if source == Source.fixtures:
        # LabeledThread is a subclass of Thread, so this is safe.
        return list(load_labeled_threads())
    # Gmail: lazy import so google-* deps never load on the fixtures path.
    from . import gmail_client

    return gmail_client.fetch_recent_threads(limit=limit, query=query)


def _gmail_whoami_safe() -> str | None:
    """Best-effort authenticated email lookup. Returns None on any failure
    so callers can fall through to a default profile without crashing the run."""
    try:
        from . import gmail_client

        return gmail_client.whoami()
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).debug("whoami lookup failed; continuing without user_email")
        return None


@app.command()
def brief(
    source: Annotated[Source, typer.Option(help="Where to pull threads from.")] = Source.fixtures,
    draft: Annotated[bool, typer.Option(help="Also draft replies for urgent/needs_reply threads.")] = True,
    llm_brief: Annotated[bool, typer.Option(help="Use Claude for the final brief rendering (slower, costs tokens).")] = False,
    limit: Annotated[int, typer.Option(help="Max threads to fetch (Gmail source only).")] = 25,
    query: Annotated[str | None, typer.Option(help="Gmail search query, e.g. 'is:unread newer_than:2d' (Gmail source only).")] = None,
    push: Annotated[bool, typer.Option(help="After drafting, push drafts to Gmail Drafts folder (Gmail source only).")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass the SQLite cache; force fresh classifications/drafts.")] = False,
) -> None:
    """Classify the inbox and print a morning brief."""
    threads = _load_threads(source, limit=limit, query=query)
    console.print(f"[cyan]Classifying {len(threads)} threads...[/cyan]")
    classifications = classify_threads(threads, use_cache=not no_cache)

    drafts = {}
    if draft:
        console.print("[cyan]Drafting replies for urgent + needs_reply...[/cyan]")
        # Pass user_email so the drafter can pick up a cached voice profile
        # (if Kyle has run `sift learn-voice`). Gmail-source only; fixtures
        # don't have an authenticated user.
        user_email = _gmail_whoami_safe() if source == Source.gmail else None
        drafts = draft_replies(
            threads, classifications, use_cache=not no_cache, user_email=user_email
        )

    brief_data = build_brief(threads, classifications, drafts)
    md = render_brief_llm(brief_data) if llm_brief else render_brief(brief_data)

    console.print()
    console.print(Markdown(md))
    console.print()

    # Also print a separate "drafts" section since the brief itself stays terse.
    if drafts:
        console.rule("[bold]Drafted replies[/bold]")
        for item in brief_data.items:
            if item.draft is None:
                continue
            console.print(f"\n[bold]{item.thread.from_name}[/bold] — {item.thread.subject}")
            console.print(f"[dim]{item.draft.tone_notes}[/dim]")
            console.print(item.draft.body)

    # Optionally push drafts back to Gmail. Gmail-source only; fixtures have
    # made-up thread ids that wouldn't resolve against the real API.
    if push and drafts:
        if source != Source.gmail:
            console.print("\n[yellow]--push ignored: only valid with --source gmail.[/yellow]")
        else:
            from . import gmail_client

            console.print(f"\n[cyan]Pushing {len(drafts)} drafts to Gmail...[/cyan]")
            results = gmail_client.push_drafts(list(drafts.values()))
            console.print(f"[green]Created {len(results)} Gmail draft(s).[/green]")


@app.command()
def classify(
    source: Annotated[Source, typer.Option(help="Where to pull threads from.")] = Source.fixtures,
    limit: Annotated[int, typer.Option(help="Max threads to fetch (Gmail source only).")] = 25,
    query: Annotated[str | None, typer.Option(help="Gmail search query (Gmail source only).")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass the SQLite cache; force fresh classifications.")] = False,
) -> None:
    """Run the classifier only and print per-thread results as a table."""
    from rich.table import Table

    threads = _load_threads(source, limit=limit, query=query)
    classifications = classify_threads(threads, use_cache=not no_cache)

    table = Table(title=f"Classified {len(threads)} threads")
    table.add_column("ID", style="dim", width=6)
    table.add_column("From", style="cyan")
    table.add_column("Subject", style="white")
    table.add_column("Category", style="yellow")
    table.add_column("Conf", justify="right")
    class_by_id = {c.thread_id: c for c in classifications}
    for t in threads:
        c = class_by_id[t.id]
        table.add_row(
            t.id,
            t.from_name[:20],
            t.subject[:45],
            c.category.value,
            f"{c.confidence:.2f}",
        )
    console.print(table)


@app.command("draft")
def draft_cmd(
    thread_id: str,
    source: Annotated[Source, typer.Option(help="Where to pull threads from.")] = Source.fixtures,
    push: Annotated[bool, typer.Option(help="Also push the draft to Gmail Drafts (Gmail source only).")] = False,
) -> None:
    """Draft a reply for a single thread by ID."""
    threads = _load_threads(source)
    match = next((t for t in threads if t.id == thread_id), None)
    if match is None:
        raise typer.BadParameter(f"No thread with id {thread_id!r}")

    console.print(f"[cyan]Drafting reply for {thread_id}...[/cyan]")
    d = draft_reply(match)
    console.rule(f"[bold]Re: {match.subject}[/bold]")
    console.print(f"[dim]{d.tone_notes}[/dim]\n")
    console.print(d.body)

    if push:
        if source != Source.gmail:
            console.print("\n[yellow]--push ignored: only valid with --source gmail.[/yellow]")
        else:
            from . import gmail_client

            draft_id = gmail_client.push_draft(d)
            console.print(f"\n[green]Pushed to Gmail Drafts (id={draft_id}).[/green]")


@app.command("auth")
def auth_cmd(
    force: Annotated[bool, typer.Option(help="Force re-running the browser flow even if a cached token exists.")] = False,
) -> None:
    """Run (or re-run) the Gmail OAuth flow and cache the token.

    Normally you don't need this — the first `sift brief --source gmail` will
    open the browser on your behalf. This command is handy for:
      * First-time setup, to verify credentials.json is wired up before a demo.
      * Forcing a re-auth after you revoke the app or change scopes.
    """
    # Lazy import so sift-auth failures happen only when you actually invoke it.
    from . import gmail_client

    try:
        creds = gmail_client.get_credentials(force_refresh=force)
        svc = gmail_client.get_service(creds=creds)
        email = gmail_client.whoami(svc)
    except gmail_client.GmailAuthError as e:
        console.print(f"[red]Gmail auth failed:[/red] {e}")
        raise typer.Exit(code=1) from e

    console.print(f"[green]Authenticated as {email}[/green]")
    console.print(f"Token cached at [dim]{gmail_client.token_file()}[/dim]")


@app.command("push-drafts")
def push_drafts_cmd(
    limit: Annotated[int, typer.Option(help="Max threads to consider from Gmail.")] = 25,
    query: Annotated[str | None, typer.Option(help="Gmail search query.")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass the SQLite cache; force fresh classifications/drafts.")] = False,
) -> None:
    """Fetch recent Gmail threads, classify, draft replies, and push them to Gmail Drafts.

    End-to-end command: this is what you'd run in the morning to land a set of
    AI-written drafts in your Gmail Drafts folder for review before you send.
    """
    from . import gmail_client

    threads = _load_threads(Source.gmail, limit=limit, query=query)
    if not threads:
        console.print("[yellow]No threads found.[/yellow]")
        return

    console.print(f"[cyan]Classifying {len(threads)} threads...[/cyan]")
    classifications = classify_threads(threads, use_cache=not no_cache)

    console.print("[cyan]Drafting replies for urgent + needs_reply...[/cyan]")
    user_email = _gmail_whoami_safe()
    drafts = draft_replies(
        threads, classifications, use_cache=not no_cache, user_email=user_email
    )
    if not drafts:
        console.print("[yellow]Nothing warranted a draft.[/yellow]")
        return

    console.print(f"[cyan]Pushing {len(drafts)} drafts to Gmail...[/cyan]")
    results = gmail_client.push_drafts(list(drafts.values()))
    console.print(f"[green]Created {len(results)} Gmail draft(s).[/green]")


@app.command("learn-voice")
def learn_voice_cmd(
    limit: Annotated[int, typer.Option(help="Max sent messages to analyze.")] = 50,
    force: Annotated[bool, typer.Option(help="Re-learn even if a fresh cached profile exists.")] = False,
) -> None:
    """Learn the user's writing voice from recent sent mail.

    Fetches up to ``limit`` messages from the user's Sent folder, summarizes
    their style with a single LLM call, and caches the resulting VoiceProfile
    by user email. The drafter will pick this up automatically on subsequent
    `sift brief --source gmail` / `sift push-drafts` runs.
    """
    from . import cache, gmail_client, voice

    user_email = gmail_client.whoami()
    if not force:
        existing = cache.get_cached_voice_profile(
            user_email, max_age_seconds=voice.VOICE_CACHE_TTL_SECONDS
        )
        if existing is not None:
            console.print(
                f"[yellow]Fresh voice profile already cached for {user_email} "
                f"(learned {existing.learned_at}). Pass --force to re-learn.[/yellow]"
            )
            return

    console.print(f"[cyan]Fetching up to {limit} sent messages for {user_email}...[/cyan]")
    messages = gmail_client.fetch_sent_messages(limit=limit)
    if not messages:
        console.print("[yellow]No sent messages found; nothing to learn from.[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Analyzing {len(messages)} messages with Claude...[/cyan]")
    profile = voice.learn_voice_profile(messages, user_email=user_email)
    cache.cache_voice_profile(profile)

    console.rule("[bold]Learned voice profile[/bold]")
    console.print(profile.summary)
    if profile.style_examples:
        console.print(f"\n[dim]Captured {len(profile.style_examples)} verbatim style example(s).[/dim]")
    console.print(f"\n[green]Cached for {user_email}.[/green]")


@app.command("cache-stats")
def cache_stats_cmd() -> None:
    """Show row counts for each cache table."""
    from rich.table import Table

    counts = cache.stats()
    db_path = cache.init_db()
    table = Table(title=f"Cache ({db_path})")
    table.add_column("Table")
    table.add_column("Rows", justify="right")
    for name, n in counts.items():
        table.add_row(name, str(n))
    console.print(table)


@app.command("cache-clear")
def cache_clear_cmd(
    table: Annotated[
        str | None,
        typer.Argument(help="Which table to clear ('threads', 'classifications', 'drafts'). Omit to clear all."),
    ] = None,
) -> None:
    """Wipe cache entries. No-op if the DB is already empty."""
    try:
        n = cache.clear(table)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    label = table or "all tables"
    console.print(f"[green]Cleared {n} rows from {label}.[/green]")


if __name__ == "__main__":
    app()
