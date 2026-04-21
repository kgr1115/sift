"""Drafter evals using LLM-as-judge.

Scoring a generated email for quality is inherently subjective. We lean on
Claude-as-judge — a separate model call with a rubric — to grade each draft
on five dimensions (addresses the ask, matches register, factuality, length
appropriateness, no-AI-tells) on a 1-5 scale.

This is deliberately not a perfect methodology. It is, however, a *repeatable*
one: when you change the drafter prompt, this suite tells you whether the
average quality score went up or down across the fixture set. That's what
matters for iterative prompt work.

We run judge-scoring on a small subset (5 threads) by default to keep token
spend reasonable. Crank via ``DRAFTER_EVAL_SAMPLES`` env var.
"""

from __future__ import annotations

import os
from pathlib import Path
from statistics import mean

import pytest

from sift.drafter import draft_reply
from sift.fixtures import load_labeled_threads
from sift.llm import structured_call
from sift.models import Category, Thread

JUDGE_SYSTEM = """You are grading an AI-drafted email reply for quality.

Grade each of the five criteria on a 1-5 scale. Be honest — the point of this
eval is to catch regressions.

1. **addresses_ask** (1-5): Does the draft address every question or ask in the original email?
2. **register_match** (1-5): Is the tone appropriate for who the sender is and how they wrote?
3. **factuality** (1-5): Does the draft avoid inventing facts (specific dates, dollar amounts, promises Kyle didn't make)? Use of [bracket placeholders] for unknowns is correct; stating made-up specifics is wrong.
4. **length_appropriate** (1-5): Is the draft the right length for the thread? Short threads should get short replies.
5. **no_ai_tells** (1-5): Does the draft avoid cliched AI openers like 'I hope this finds you well' or phrases like 'I'd be happy to'?

Also produce a one-sentence `overall_comment` summarizing the verdict.

Any score below 3 on any dimension is a red flag — explain in the comment."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "addresses_ask": {"type": "integer", "minimum": 1, "maximum": 5},
        "register_match": {"type": "integer", "minimum": 1, "maximum": 5},
        "factuality": {"type": "integer", "minimum": 1, "maximum": 5},
        "length_appropriate": {"type": "integer", "minimum": 1, "maximum": 5},
        "no_ai_tells": {"type": "integer", "minimum": 1, "maximum": 5},
        "overall_comment": {"type": "string"},
    },
    "required": [
        "addresses_ask",
        "register_match",
        "factuality",
        "length_appropriate",
        "no_ai_tells",
        "overall_comment",
    ],
}


DIMENSIONS = ["addresses_ask", "register_match", "factuality", "length_appropriate", "no_ai_tells"]

# Floor we expect each dimension to clear on average across the sample.
DIMENSION_FLOOR = 3.5


def _judge_draft(thread: Thread, draft_body: str) -> dict[str, int | str]:
    user = (
        "Here is the original email:\n\n"
        f"---\nFrom: {thread.from_name} <{thread.from_}>\n"
        f"Subject: {thread.subject}\n\n{thread.body}\n---\n\n"
        "Here is the AI-drafted reply to be graded:\n\n"
        f"---\n{draft_body}\n---\n\n"
        "Grade the draft."
    )
    return structured_call(
        system=JUDGE_SYSTEM,
        user=user,
        tool_name="grade_draft",
        tool_description="Submit the rubric grades for this draft.",
        input_schema=JUDGE_SCHEMA,
        max_tokens=400,
        log_tag="judge",
    )


@pytest.fixture(scope="session")
def draftable_sample():
    """Pick the first N threads that warrant a draft (urgent + needs_reply)."""
    n = int(os.getenv("DRAFTER_EVAL_SAMPLES", "5"))
    all_threads = load_labeled_threads()
    draftable = [t for t in all_threads if t.label in {Category.URGENT, Category.NEEDS_REPLY}]
    return draftable[:n]


@pytest.fixture(scope="session")
def graded_drafts(draftable_sample):
    """Draft + grade. One run, reused across dimension assertions."""
    plain = [Thread.model_validate(t.model_dump(by_alias=True)) for t in draftable_sample]
    drafts = [draft_reply(t) for t in plain]
    grades = [_judge_draft(t, d.body) for t, d in zip(plain, drafts)]
    return plain, drafts, grades


@pytest.mark.llm
@pytest.mark.parametrize("dim", DIMENSIONS)
def test_dimension_average(graded_drafts, dim):
    _, _, grades = graded_drafts
    avg = mean(g[dim] for g in grades)  # type: ignore[arg-type]
    assert avg >= DIMENSION_FLOOR, (
        f"Average '{dim}' score {avg:.2f} fell below floor {DIMENSION_FLOOR}. "
        f"Comments: {[g['overall_comment'] for g in grades]}"
    )


@pytest.mark.llm
def test_write_drafter_artifact(graded_drafts):
    threads, drafts, grades = graded_drafts

    # Summary line
    avg_by_dim = {dim: mean(g[dim] for g in grades) for dim in DIMENSIONS}  # type: ignore[arg-type]
    overall = mean(avg_by_dim.values())

    lines = [
        "# Latest Drafter Eval Run",
        "",
        f"**Overall mean score:** {overall:.2f} / 5  (n={len(grades)})",
        "",
        "| Dimension | Avg |",
        "|-----------|-----|",
    ]
    for dim, v in avg_by_dim.items():
        lines.append(f"| {dim} | {v:.2f} |")

    lines.append("\n## Individual drafts\n")
    for t, d, g in zip(threads, drafts, grades):
        lines.append(f"### {t.from_name} — {t.subject}\n")
        lines.append(f"_{g['overall_comment']}_\n")
        lines.append(
            "Scores: "
            + ", ".join(f"{dim}={g[dim]}" for dim in DIMENSIONS)
            + "\n"
        )
        lines.append(f"```\n{d.body}\n```\n")

    (Path(__file__).parent / "last_run_drafter.md").write_text("\n".join(lines), encoding="utf-8")
