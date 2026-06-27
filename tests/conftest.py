"""Shared pytest fixtures. Tests are CPU-only and never download models/data:
they use the rule normalizer + the placeholder TTS backend.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SAMPLE_DATA = REPO / "sample_data"


@pytest.fixture(autouse=True, scope="session")
def _artifacts_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("abk_artifacts")
    os.environ["AUDIOBOOK_AI_ARTIFACTS_DIR"] = str(d)
    os.environ.setdefault("AUDIOBOOK_AI_LOG_LEVEL", "WARNING")
    yield


@pytest.fixture
def cfg():
    from audiobook_ai.config import AppConfig
    c = AppConfig()
    c.data.synthetic_train_size = 800
    c.data.synthetic_val_size = 200
    c.data.synthetic_test_size = 200
    c.data.hard_slice_size = 150
    return c


@pytest.fixture
def sample_book():
    return str(SAMPLE_DATA / "sample_book.txt")


@pytest.fixture
def sample_md():
    return str(SAMPLE_DATA / "sample_article.md")
