"""Load the hand-labeled fixture inbox.

These fixtures play two roles:

  1. A synthetic inbox the CLI and UI can run against without Gmail being connected.
  2. The ground-truth dataset for the classifier evals.

Keeping the loader here (and not in evals/) means the same file works for
both demo purposes and testing, with no duplication.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import LabeledThread

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "evals" / "fixtures" / "labeled_threads.json"


def load_labeled_threads(path: Path | None = None) -> list[LabeledThread]:
    """Read the fixture JSON and return parsed LabeledThread models."""
    target = path or FIXTURE_PATH
    with target.open(encoding="utf-8") as f:
        raw = json.load(f)
    return [LabeledThread.model_validate(item) for item in raw]
