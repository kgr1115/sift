"""Provider-comparison eval.

Runs the classifier against every provider whose API key is set in the
environment, records accuracy + tokens + latency + estimated cost, and writes
a markdown comparison table to ``evals/last_provider_comparison.md``.

This is the marquee eval for the multi-provider feature: it's how you
empirically answer "which provider is cheapest for good-enough quality on
*this* task?" rather than guessing from marketing pages.

Run:
    pytest evals/test_provider_comparison.py -v -s

Typical cost on 40 fixtures with all four providers: well under $0.05.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sift.classifier import _CLASSIFY_SCHEMA, _CLASSIFY_SYSTEM, _render_thread
from sift.fixtures import load_labeled_threads
from sift.llm import structured_call_full
from sift.models import Classification
from sift.providers import get_provider
from sift.providers.registry import list_available_providers

from .metrics import overall_accuracy, per_category_metrics


@dataclass
class ProviderRun:
    """One full run of the classifier for a single provider."""

    provider: str
    model: str
    accuracy: float
    per_category_recall: dict[str, float]
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    estimated_cost_usd: float = 0.0
    misclassifications: list[tuple[str, str, str]] = field(default_factory=list)
    # misclassifications: [(thread_id, truth, predicted), ...]


def _run_provider(provider_name: str, threads) -> ProviderRun:
    """Classify every fixture with the given provider. One sequential pass.

    We deliberately run serially here (not concurrent like ``classify_threads``)
    so the latency numbers per call are clean per-call numbers, not
    contaminated by network-level concurrency effects. Total wall time doesn't
    matter for this eval.
    """
    provider_obj = get_provider(provider_name)
    model = provider_obj.model
    input_per_mtok, output_per_mtok = provider_obj.get_pricing(model)

    predictions: list[Classification] = []
    input_tokens = 0
    output_tokens = 0
    total_latency_ms = 0.0
    errors = 0

    for t in threads:
        user = f"Classify the following email thread:\n\n---\n{_render_thread(t)}\n---"
        try:
            result = structured_call_full(
                system=_CLASSIFY_SYSTEM,
                user=user,
                tool_name="classify_thread",
                tool_description="Record the triage classification for an email thread.",
                input_schema=_CLASSIFY_SCHEMA,
                provider_name=provider_name,
                max_tokens=400,
                log_tag=f"compare_{provider_name}",
            )
            data = result.data or {}
            predictions.append(Classification(thread_id=t.id, **data))
            if result.usage:
                input_tokens += result.usage.input_tokens
                output_tokens += result.usage.output_tokens
                total_latency_ms += result.usage.latency_ms
        except Exception as e:  # noqa: BLE001 — one provider erroring shouldn't kill the whole run
            errors += 1
            predictions.append(
                Classification(
                    thread_id=t.id,
                    category="fyi",
                    confidence=0.0,
                    one_line_summary=f"[error: {type(e).__name__}]",
                    reason=str(e)[:200],
                )
            )

    truth = [t.label.value for t in threads]
    preds = [p.category.value for p in predictions]
    metrics = per_category_metrics(preds, truth)
    acc = overall_accuracy(preds, truth)

    misclassifications = [
        (t.id, truth[i], preds[i]) for i, t in enumerate(threads) if truth[i] != preds[i]
    ]

    cost = (
        input_tokens / 1_000_000 * input_per_mtok
        + output_tokens / 1_000_000 * output_per_mtok
    )

    return ProviderRun(
        provider=provider_name,
        model=model,
        accuracy=acc,
        per_category_recall={cat: m.recall for cat, m in metrics.items()},
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_latency_ms=total_latency_ms,
        errors=errors,
        estimated_cost_usd=cost,
        misclassifications=misclassifications,
    )


def _format_comparison(runs: list[ProviderRun], n_fixtures: int) -> str:
    """Render the comparison table as a markdown file."""
    if not runs:
        return "# Provider Comparison\n\nNo providers had API keys set.\n"

    # Sort by accuracy desc, then cost asc — easy to scan
    runs = sorted(runs, key=lambda r: (-r.accuracy, r.estimated_cost_usd))

    lines = [
        "# Provider Comparison",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. {n_fixtures} labeled fixtures._",
        "",
        "| Provider | Model | Accuracy | Errors | Total cost | $/1k threads | Avg latency | In / Out tokens |",
        "|----------|-------|---------:|-------:|-----------:|-------------:|------------:|---------------:|",
    ]
    for r in runs:
        per_thread = r.estimated_cost_usd / max(n_fixtures, 1)
        per_1k = per_thread * 1000
        avg_latency = r.total_latency_ms / max(n_fixtures, 1)
        lines.append(
            f"| {r.provider} | `{r.model}` | {r.accuracy:.1%} | {r.errors} | "
            f"${r.estimated_cost_usd:.4f} | ${per_1k:.3f} | "
            f"{avg_latency:.0f} ms | {r.input_tokens:,} / {r.output_tokens:,} |"
        )

    lines += ["", "## Per-category recall", ""]
    cats = sorted({cat for r in runs for cat in r.per_category_recall})
    header = "| Provider | " + " | ".join(cats) + " |"
    sep = "|----------|" + "|".join(["---:"] * len(cats)) + "|"
    lines += [header, sep]
    for r in runs:
        row = "| " + r.provider + " | " + " | ".join(
            f"{r.per_category_recall.get(c, 0.0):.0%}" for c in cats
        ) + " |"
        lines.append(row)

    lines += ["", "## Notes", ""]
    lines += [
        "- **Accuracy** is raw classification accuracy over all labeled fixtures.",
        "- **Errors** are calls where the provider raised or returned an invalid payload "
        "(counted as a miss against the `fyi` fallback).",
        "- **Cost** uses the per-MTok prices declared on each provider class "
        "(`pricing` attribute). Update those if pricing moves.",
        "- **Avg latency** is wall-clock per-call time and includes network round-trip.",
        "- Pricing is approximate. These numbers are a *decision aid*, not an audit.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Key-free structural tests (always run)
# ---------------------------------------------------------------------------

def test_format_comparison_with_no_runs():
    """Empty runs list produces a valid, non-crashing report."""
    out = _format_comparison([], n_fixtures=0)
    assert "# Provider Comparison" in out


def test_format_comparison_sorts_by_accuracy():
    runs = [
        ProviderRun(
            provider="b", model="b-1", accuracy=0.70, per_category_recall={"x": 0.7},
            input_tokens=100, output_tokens=50, total_latency_ms=1000,
            estimated_cost_usd=0.01,
        ),
        ProviderRun(
            provider="a", model="a-1", accuracy=0.90, per_category_recall={"x": 0.9},
            input_tokens=100, output_tokens=50, total_latency_ms=2000,
            estimated_cost_usd=0.02,
        ),
    ]
    out = _format_comparison(runs, n_fixtures=10)
    # The higher-accuracy provider should appear first in the table.
    assert out.index("| a |") < out.index("| b |")


# ---------------------------------------------------------------------------
# The real eval — only runs when at least one provider key is set
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_provider_comparison():
    """Fan the classifier out to every configured provider; write a report.

    No hard accuracy floors here — we just want the table. The per-provider
    floors belong in ``test_classifier.py`` against the *default* provider.
    """
    available = list_available_providers()
    if not available:
        pytest.skip("No provider API keys configured")

    threads = load_labeled_threads()
    runs: list[ProviderRun] = []
    for name in available:
        print(f"\n--- Running {name} on {len(threads)} fixtures ---")
        runs.append(_run_provider(name, threads))

    report = _format_comparison(runs, n_fixtures=len(threads))
    out_md = Path(__file__).parent / "last_provider_comparison.md"
    out_md.write_text(report, encoding="utf-8")

    # Also dump a machine-readable JSON so downstream tooling doesn't have to
    # parse markdown. Skip the misclassifications list to keep it small.
    summary = [
        {
            "provider": r.provider,
            "model": r.model,
            "accuracy": r.accuracy,
            "errors": r.errors,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "total_latency_ms": r.total_latency_ms,
            "estimated_cost_usd": r.estimated_cost_usd,
            "per_category_recall": r.per_category_recall,
        }
        for r in runs
    ]
    (Path(__file__).parent / "last_provider_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Sanity: every provider ran at least *something*.
    for r in runs:
        assert r.input_tokens + r.errors > 0, f"{r.provider} produced no calls"
