"""Morning brief: turn classified threads (+ drafts) into a crisp markdown digest.

Two paths:

  * `render_brief()` — the fast, deterministic path. Builds the markdown
    digest from the structured classifications directly (no LLM call). This is
    what we actually ship to the user.

  * `render_brief_llm()` — optionally pass the structured data through Claude
    to get a more human-feeling headline and punchier phrasing. Slower,
    costs tokens. Off by default.

Keeping the deterministic path first means the brief works offline if the API
is down or the user is deliberately avoiding extra token spend.
"""

from __future__ import annotations

from datetime import datetime

from .llm import free_text_call, load_prompt
from .models import Brief, BriefItem, Category, Classification, Draft, Thread

_BRIEF_SYSTEM = load_prompt("brief")


def build_brief(
    threads: list[Thread],
    classifications: list[Classification],
    drafts: dict[str, Draft] | None = None,
) -> Brief:
    """Assemble the structured Brief. Pure Python, no LLM call."""
    drafts = drafts or {}
    class_by_id = {c.thread_id: c for c in classifications}
    items = [
        BriefItem(
            thread=t,
            classification=class_by_id[t.id],
            draft=drafts.get(t.id),
        )
        for t in threads
        if t.id in class_by_id
    ]
    return Brief(generated_at=datetime.utcnow(), items=items)


def render_brief(brief: Brief) -> str:
    """Deterministic markdown renderer."""
    urgent = brief.by_category(Category.URGENT)
    reply = brief.by_category(Category.NEEDS_REPLY)
    fyi = brief.by_category(Category.FYI)
    news = brief.by_category(Category.NEWSLETTER)
    trash = brief.by_category(Category.TRASH)

    date_str = brief.generated_at.strftime("%A, %B %d, %Y")
    lines: list[str] = [f"# Morning Brief — {date_str}", ""]

    # Headline
    if urgent:
        top = urgent[0]
        lines.append(f"**Headline:** {top.classification.one_line_summary}")
    elif reply:
        top = reply[0]
        lines.append(f"**Headline:** {len(reply)} threads waiting on you. Top: {top.classification.one_line_summary}")
    else:
        lines.append("**Headline:** Quiet inbox today \u2014 nothing pressing.")
    lines.append("")

    # Urgent
    lines.append(f"## 🔥 Urgent ({len(urgent)})")
    if not urgent:
        lines.append("_none_")
    else:
        for item in urgent:
            lines.append(
                f"- **{item.thread.from_name}** — {item.classification.one_line_summary}"
            )
    lines.append("")

    # Needs reply
    lines.append(f"## ↩️  Awaiting your reply ({len(reply)})")
    if not reply:
        lines.append("_none_")
    else:
        for item in reply:
            lines.append(
                f"- **{item.thread.from_name}** — {item.classification.one_line_summary}"
            )
    lines.append("")

    # Drafts — inline preview of the top drafts so the UI/README can show
    # "AI wrote a reply in your voice" at a glance, instead of hiding it
    # in a per-thread expander. Cap at 3 so the brief stays scannable.
    drafted = [i for i in urgent + reply if i.draft is not None]
    if drafted:
        lines.append(f"## \u270d\ufe0f Drafts ready for review ({len(drafted)})")
        for item in drafted[:3]:
            tone = f" — _{item.draft.tone_notes}_" if item.draft.tone_notes else ""
            lines.append(
                f"**To: {item.thread.from_name}** · _Re: {item.thread.subject}_{tone}"
            )
            lines.append("")
            # Blockquote the draft body so it visually sits inside the brief.
            for body_line in item.draft.body.splitlines() or [""]:
                lines.append(f"> {body_line}" if body_line else ">")
            lines.append("")
        if len(drafted) > 3:
            lines.append(f"_(+ {len(drafted) - 3} more drafts below)_")
        lines.append("")

    # FYI — cap at 5 most-recent
    lines.append(f"## 📬 FYI ({len(fyi)})")
    if not fyi:
        lines.append("_none_")
    else:
        recent_fyi = sorted(fyi, key=lambda i: i.thread.received_at, reverse=True)[:5]
        for item in recent_fyi:
            lines.append(
                f"- **{item.thread.from_name}** — {item.classification.one_line_summary}"
            )
        if len(fyi) > 5:
            lines.append(f"- _(+ {len(fyi) - 5} more)_")
    lines.append("")

    # Noise summary
    lines.append("## 🗞️ Noise")
    lines.append(
        f"{len(news)} newsletter{'s' if len(news) != 1 else ''}, "
        f"{len(trash)} promotional/spam — no action."
    )

    return "\n".join(lines)


def render_brief_llm(brief: Brief) -> str:
    """Pass the structured brief through Claude for a more human-feeling headline.

    Not used by default; opt-in via ``--llm-brief`` on the CLI.
    """
    data = brief.model_dump_json(indent=2)
    user = (
        "Here is today's classified inbox data as JSON. Produce the morning brief "
        "following the structure in your system prompt.\n\n"
        f"```json\n{data}\n```"
    )
    return free_text_call(
        system=_BRIEF_SYSTEM,
        user=user,
        max_tokens=800,
        log_tag="brief",
    )
