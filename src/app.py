"""Streamlit UI for Sift.

Run:
    streamlit run src/app.py

What this is for: a clean demo surface that makes for good screenshots in
the README. Not a full inbox app — we leave that to real email clients.

Layout:
  Sidebar   — source selector, run controls, cost estimate
  Main      — the morning brief at the top, expandable per-thread cards below
              with classification + draft side-by-side
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Make the package importable when running `streamlit run src/app.py` without an install.
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402

from sift.brief import build_brief, render_brief  # noqa: E402
from sift.classifier import classify_threads  # noqa: E402
from sift.drafter import draft_replies  # noqa: E402
from sift.fixtures import load_labeled_threads  # noqa: E402
from sift.models import Category, Thread  # noqa: E402

st.set_page_config(page_title="Sift", page_icon="📬", layout="wide")

# ---------- CSS polish (minimal, mostly spacing) ----------
st.markdown(
    """
    <style>
    .small-muted { color: #888; font-size: 0.85rem; }
    .cat-chip {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .cat-urgent { background: #ffe4e4; color: #b10000; }
    .cat-needs_reply { background: #e4f0ff; color: #0b4d8a; }
    .cat-fyi { background: #eaeaea; color: #444; }
    .cat-newsletter { background: #fff4d6; color: #805a00; }
    .cat-trash { background: #f0f0f0; color: #999; text-decoration: line-through; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📬 Sift")
    st.caption("AI triage, drafts, and a morning brief.")

    source = st.selectbox(
        "Source",
        options=["Synthetic fixture inbox", "Gmail (coming soon)"],
        index=0,
    )
    do_drafts = st.checkbox("Draft replies for urgent + needs_reply", value=True)

    run = st.button("Run", type="primary", use_container_width=True)

    st.divider()
    st.markdown(
        "<span class='small-muted'>Built by <a href='mailto:kyle.g.rauch@gmail.com'>Kyle Rauch</a> "
        "as a portfolio project. "
        "<a href='https://github.com/'>Source</a></span>",
        unsafe_allow_html=True,
    )


# ---------- Helpers ----------
def cat_chip(cat: Category) -> str:
    return f"<span class='cat-chip cat-{cat.value}'>{cat.value}</span>"


def cached_fixture_threads() -> list[Thread]:
    # No @st.cache_* on purpose: the fixture load is milliseconds, and caching
    # Pydantic model instances across hot-reloads causes stale-class validation
    # errors (cached instance is of the old Thread class, not the reloaded one).
    labeled = load_labeled_threads()
    return [Thread.model_validate(t.model_dump(by_alias=True)) for t in labeled]


# ---------- Main ----------
st.title("Morning Brief")

if not run:
    st.info(
        "Pick a source in the sidebar and hit **Run** to classify the inbox and draft replies. "
        "The synthetic fixture inbox runs against 40 hand-labeled threads — no API key needed to see the UI, "
        "but classification/drafting will call Claude."
    )
    st.stop()

if "Gmail" in source:
    st.error("Gmail integration is queued for the next phase of the build. Use the synthetic inbox for now.")
    st.stop()

# --- Run the pipeline ---
with st.status("Running pipeline...", expanded=True) as status:
    status.write("Loading threads from the fixture inbox...")
    threads = cached_fixture_threads()
    status.write(f"Loaded {len(threads)} threads.")

    status.write(f"Classifying with Claude (concurrent, up to 8 in-flight)...")
    classifications = classify_threads(threads)
    status.write(f"✓ Classified {len(classifications)} threads.")

    drafts = {}
    if do_drafts:
        status.write("Drafting replies for urgent + needs_reply...")
        drafts = draft_replies(threads, classifications)
        status.write(f"✓ Drafted {len(drafts)} replies.")

    brief_data = build_brief(threads, classifications, drafts)
    status.update(label="Done.", state="complete")

# --- Brief ---
st.markdown(render_brief(brief_data))

st.divider()

# --- Per-thread cards ---
st.header("All threads")

# Filter controls
col_filter_1, col_filter_2 = st.columns([1, 3])
with col_filter_1:
    cat_filter = st.multiselect(
        "Filter by category",
        options=[c.value for c in Category],
        default=[c.value for c in Category],
    )

# Group & display
visible = [
    item for item in brief_data.items
    if item.classification.category.value in cat_filter
]
# Sort by urgency order then by recency
CAT_ORDER = [Category.URGENT, Category.NEEDS_REPLY, Category.FYI, Category.NEWSLETTER, Category.TRASH]
visible.sort(
    key=lambda i: (CAT_ORDER.index(i.classification.category), -i.thread.received_at.timestamp())
)

for item in visible:
    thread = item.thread
    cls = item.classification
    header = f"{cat_chip(cls.category)} **{thread.from_name}** — {thread.subject}"
    with st.expander(header, expanded=False):
        # Using markdown with unsafe_allow_html for the chip
        st.markdown(
            f"{cat_chip(cls.category)} <span class='small-muted'>"
            f"from {thread.from_} · received {thread.received_at.strftime('%b %d, %H:%M')} · "
            f"confidence {cls.confidence:.2f}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"🧠 {cls.reason}")
        st.write("**Summary:** " + cls.one_line_summary)

        col_body, col_draft = st.columns(2)
        with col_body:
            st.markdown("**Original**")
            st.text(thread.body)

        with col_draft:
            if item.draft:
                st.markdown("**Drafted reply** — " + f"_{item.draft.tone_notes}_")
                st.code(item.draft.body, language="")
            else:
                st.markdown("**Drafted reply**")
                st.caption("_No draft generated for this category._")

# --- Footer ---
st.divider()
st.caption(
    f"Generated at {brief_data.generated_at.strftime('%Y-%m-%d %H:%M UTC')} · "
    f"{len(threads)} threads processed · "
    f"{len(drafts)} drafts written"
)
