"""Classifier evals: measure per-category precision/recall on the labeled fixture set.

Run:
    pytest evals/test_classifier.py -v

With ``ANTHROPIC_API_KEY`` set, this makes ~40 parallel Claude calls and takes
about 15-30 seconds. Without it, the LLM tests are skipped automatically.

We also write the latest metrics to ``evals/last_run.md`` so the README can
be regenerated from real numbers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sift.classifier import classify_threads
from sift.fixtures import load_labeled_threads
from sift.models import Thread

from .metrics import format_metrics_table, overall_accuracy, per_category_metrics

# Floors we expect the classifier to clear. If a prompt change pushes any of
# these down, pytest fails — which is the point. These numbers are calibrated
# to be ambitious-but-achievable; tighten once we have a real baseline.
ACCURACY_FLOOR = 0.85
PER_CATEGORY_RECALL_FLOORS: dict[str, float] = {
    "urgent": 0.80,  # Missing an urgent thread is the most costly error.
    "trash": 0.80,   # Misrouting spam to needs_reply is the second-most-costly.
    "needs_reply": 0.70,
    "fyi": 0.65,
    "newsletter": 0.65,
}


@pytest.fixture(scope="session")
def labeled_threads():
    return load_labeled_threads()


@pytest.fixture(scope="session")
def classification_run(labeled_threads):
    """One real LLM run, reused across every assertion below."""
    # Cast LabeledThread -> plain Thread so the classifier doesn't see the label.
    plain = [Thread.model_validate(t.model_dump(by_alias=True)) for t in labeled_threads]
    predictions = classify_threads(plain)
    assert len(predictions) == len(labeled_threads)
    return labeled_threads, predictions


@pytest.mark.llm
def test_overall_accuracy(classification_run):
    labeled, predictions = classification_run
    truth = [t.label.value for t in labeled]
    preds = [p.category.value for p in predictions]
    acc = overall_accuracy(preds, truth)
    assert acc >= ACCURACY_FLOOR, (
        f"Overall accuracy {acc:.2%} fell below floor {ACCURACY_FLOOR:.0%}. "
        f"Prompt regression? Run `pytest evals/ -v` locally and inspect misclassifications."
    )


@pytest.mark.llm
@pytest.mark.parametrize("category,recall_floor", list(PER_CATEGORY_RECALL_FLOORS.items()))
def test_per_category_recall(classification_run, category, recall_floor):
    """Each category must clear its recall floor.

    Recall matters more than precision for this product: a false negative on
    'urgent' means an important email gets buried, which is the failure mode
    users hate most.
    """
    labeled, predictions = classification_run
    truth = [t.label.value for t in labeled]
    preds = [p.category.value for p in predictions]
    metrics = per_category_metrics(preds, truth)
    m = metrics[category]
    assert m.recall >= recall_floor, (
        f"Recall for '{category}' was {m.recall:.2%} "
        f"(tp={m.true_positive}, fn={m.false_negative}, support={m.support}), "
        f"below floor {recall_floor:.0%}."
    )


@pytest.mark.llm
def test_write_metrics_artifact(classification_run, tmp_path_factory):
    """Side-effect test: dump the latest metrics to evals/last_run.md."""
    labeled, predictions = classification_run
    truth = [t.label.value for t in labeled]
    preds = [p.category.value for p in predictions]
    metrics = per_category_metrics(preds, truth)
    acc = overall_accuracy(preds, truth)

    out = Path(__file__).parent / "last_run.md"
    out.write_text(
        "# Latest Classifier Eval Run\n\n"
        f"**Overall accuracy:** {acc:.2%} ({sum(1 for p, g in zip(preds, truth) if p == g)}/{len(truth)})\n\n"
        f"{format_metrics_table(metrics)}\n\n"
        f"## Misclassifications\n\n"
        + "\n".join(
            f"- **{t.id}** `{t.label.value} → {p.category.value}` "
            f"(conf {p.confidence:.2f}) — {t.subject}"
            for t, p in zip(labeled, predictions)
            if t.label.value != p.category.value
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Key-free sanity tests (always run, no API key required)
# ---------------------------------------------------------------------------

def test_fixture_parses():
    threads = load_labeled_threads()
    assert len(threads) >= 30, "Expected at least 30 labeled fixtures"
    cats = {t.label.value for t in threads}
    assert cats == {"urgent", "needs_reply", "fyi", "newsletter", "trash"}


def test_metrics_math():
    """Sanity-check the metrics helper against a known case."""
    preds = ["a", "a", "b", "b", "a"]
    truth = ["a", "b", "b", "a", "a"]
    # 'a': tp=2, fp=1, fn=1  -> P=2/3, R=2/3, F=2/3
    # 'b': tp=1, fp=1, fn=1  -> P=1/2, R=1/2, F=1/2
    m = per_category_metrics(preds, truth)
    assert round(m["a"].precision, 3) == 0.667
    assert round(m["a"].recall, 3) == 0.667
    assert round(m["b"].precision, 3) == 0.5
    assert round(m["b"].recall, 3) == 0.5
    assert overall_accuracy(preds, truth) == 0.6
