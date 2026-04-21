"""Pytest configuration for the evals suite.

Two important behaviors:

  1. If ``ANTHROPIC_API_KEY`` is not set in the environment, every eval that
     needs a real LLM call is skipped rather than failed. This lets the test
     suite run in CI without secrets and on forks that can't access the key.

  2. Real-LLM evals are marked with ``@pytest.mark.llm`` so you can run
     ``pytest -m 'not llm'`` for a fast, key-free sanity check.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the package importable without a pip install (convenient for the eval harness)
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "llm: requires a real Anthropic API call")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if os.getenv("ANTHROPIC_API_KEY"):
        return
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set; skipping real-LLM evals")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)
