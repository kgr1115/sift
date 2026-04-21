"""Pydantic data models shared across the pipeline.

Why Pydantic: we're passing JSON back and forth between Claude's tool-use outputs,
the cache, the UI, and the evals. A single source-of-truth schema means the
classifier can't quietly return an unexpected shape and ripple breakage downstream.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    """The five triage buckets. Order is display-priority (urgent first)."""

    URGENT = "urgent"
    NEEDS_REPLY = "needs_reply"
    FYI = "fyi"
    NEWSLETTER = "newsletter"
    TRASH = "trash"


# Helpful for schemas / prompts that need the literal strings.
CATEGORY_VALUES = [c.value for c in Category]


class Thread(BaseModel):
    """A normalized email thread — the input to the classifier and drafter.

    This is deliberately a subset of Gmail's full thread schema; we don't want
    Claude sifting through MIME headers when a few clean fields will do.
    """

    id: str
    from_: str = Field(alias="from")
    from_name: str
    to: str
    subject: str
    received_at: datetime
    body: str

    model_config = {"populate_by_name": True}


class LabeledThread(Thread):
    """A thread with a ground-truth label. Used for evals and the synthetic fixture inbox."""

    label: Category
    notes: str = ""


class Classification(BaseModel):
    """The classifier's structured output for a single thread."""

    thread_id: str
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    one_line_summary: str
    reason: str


class Draft(BaseModel):
    """A draft reply for a single thread."""

    thread_id: str
    subject: str
    body: str
    tone_notes: str = ""  # Why Claude chose this tone — useful for debugging and writeups.


class BriefItem(BaseModel):
    """One row in the morning brief — a classified thread, with an optional draft."""

    thread: Thread
    classification: Classification
    draft: Draft | None = None


class Brief(BaseModel):
    """The rendered morning brief."""

    generated_at: datetime
    items: list[BriefItem]

    def by_category(self, cat: Category) -> list[BriefItem]:
        return [i for i in self.items if i.classification.category == cat]


class VoiceProfile(BaseModel):
    """A compressed description of how the user writes.

    Populated either from a hand-written default or by :mod:`sift.voice` after
    ingesting the user's Gmail Sent folder. Cached per user email so a freshly
    learned profile persists across runs.
    """

    summary: str
    style_examples: list[str] = Field(default_factory=list)

    # Populated when the profile is learned (not present on the hardcoded default).
    user_email: str | None = None
    learned_at: datetime | None = None

    def render_for_prompt(self) -> str:
        """Format this profile for injection into the drafter's system prompt."""
        lines = [self.summary]
        if self.style_examples:
            lines.append("")
            lines.append("Example replies the user has actually sent:")
            for i, ex in enumerate(self.style_examples, 1):
                lines.append(f"\n--- Example {i} ---\n{ex}\n")
        return "\n".join(lines)
