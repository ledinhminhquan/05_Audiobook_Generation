"""End-to-end agent pipeline (rule normalizer + placeholder TTS, fully offline)."""

from __future__ import annotations

from audiobook_ai.agent.narrator_agent import NarratorAgent
from audiobook_ai.agent.policy import audio_qa_checks
from audiobook_ai.config import AppConfig


def _agent():
    return NarratorAgent(AppConfig(), load_model=False, tts_backend="placeholder")


def test_full_pipeline_completes(sample_book):
    job = _agent().process(path=sample_book, title="Test Book", synth=True)
    sd = job.to_dict()
    assert sd["status"] in ("completed", "needs_review")
    assert sd["n_chapters"] >= 2
    assert sd["n_spoken_segments"] >= 4
    # all four decision points fired
    assert {d["id"] for d in sd["decisions"]} == {"D1", "D2", "D3", "D4"}
    # audio + subtitles + manifest produced
    assert "wav" in sd["outputs"]
    assert "srt" in sd["outputs"]
    assert sd["metrics"]["audio_duration"] > 0
    # every traced step succeeded
    assert all(t["ok"] for t in sd["trace"])


def test_normalize_preview_no_audio():
    job = _agent().normalize_preview("He paid $5.2M in 1984.")
    assert job.status.value == "normalized"
    assert job.segments
    assert "million dollars" in " ".join(s.normalized for s in job.segments)
    assert not job.outputs            # no audio in preview mode


def test_audio_qa_checks_detect_empty():
    import numpy as np
    cfg = AppConfig().agent
    empty = audio_qa_checks(np.zeros(0, dtype=np.float32), 16000, "hello", cfg)
    assert empty["checks"]["empty"] is True
    assert empty["pass"] is False
    good = audio_qa_checks(np.random.default_rng(0).standard_normal(16000).astype("float32") * 0.05,
                           16000, "a short clip here", cfg)
    assert "duration_ratio" in good["checks"]


def test_decisions_have_branches(sample_book):
    job = _agent().process(path=sample_book, title="Test", synth=True)
    by_id = {d.id: d for d in job.decisions}
    assert by_id["D1"].branch in ("structured", "assisted", "degraded")
    assert by_id["D4"].branch in ("single", "multi")
