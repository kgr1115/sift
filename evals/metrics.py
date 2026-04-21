"""Precision/recall/F1 helpers for the classifier eval.

Small, dependency-free so the eval suite stays fast. If we ever want more
sophistication (confusion matrices across many runs, bootstrap CIs, etc.) it's
worth pulling in scikit-learn, but for a 5-class, 40-example dataset this is
plenty.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CategoryMetrics:
    category: str
    true_positive: int
    false_positive: int
    false_negative: int
    support: int  # total examples with this label as ground truth

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def per_category_metrics(
    predictions: list[str], ground_truth: list[str]
) -> dict[str, CategoryMetrics]:
    """Compute precision/recall/F1 for each category present in the ground truth."""
    assert len(predictions) == len(ground_truth), "Length mismatch"

    all_cats = set(ground_truth) | set(predictions)
    results: dict[str, CategoryMetrics] = {}
    for cat in all_cats:
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == cat and g == cat)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == cat and g != cat)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != cat and g == cat)
        support = sum(1 for g in ground_truth if g == cat)
        results[cat] = CategoryMetrics(
            category=cat,
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            support=support,
        )
    return results


def overall_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    if not ground_truth:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(ground_truth)


def format_metrics_table(metrics: dict[str, CategoryMetrics]) -> str:
    """Render a markdown-friendly metrics table. Useful for README updates."""
    order = ["urgent", "needs_reply", "fyi", "newsletter", "trash"]
    rows = [m for cat in order if (m := metrics.get(cat))]
    # Append any extra categories not in the canonical order
    for cat, m in metrics.items():
        if cat not in order:
            rows.append(m)

    header = "| Category     | Precision | Recall | F1    | Support |"
    sep =    "|--------------|-----------|--------|-------|---------|"
    lines = [header, sep]
    for m in rows:
        lines.append(
            f"| {m.category:<12} | {m.precision:.2f}      | {m.recall:.2f}   | {m.f1:.2f}  | {m.support:>7} |"
        )
    return "\n".join(lines)
