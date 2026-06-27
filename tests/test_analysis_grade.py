"""Leaf modules: error analysis, audio-quality report, monitoring, grading."""

from __future__ import annotations

from pathlib import Path

from audiobook_ai.analysis.audio_quality import audio_quality_report
from audiobook_ai.analysis.error_analysis import error_analysis
from audiobook_ai.grading.checklist import build_checklist
from audiobook_ai.monitoring.drift_report import monitoring_report


def test_error_analysis_structure(cfg):
    res = error_analysis(cfg, save=False)
    assert "per_class" in res["test"]
    assert res["hard"]["n"] > 0
    assert 0.0 <= res["hard"]["unrecoverable"]["rate"] <= 1.0


def test_audio_quality_report(cfg):
    res = audio_quality_report(cfg, backend="placeholder", save=False)
    assert res["n_segments"] >= 1
    assert 0.0 <= res["pass_rate"] <= 1.0


def test_monitoring_handles_empty_logs(cfg):
    res = monitoring_report(cfg, log_path="/nonexistent/jobs.jsonl", save=False)
    assert res["n_jobs"] == 0
    assert res["overall"]["n"] == 0


def test_grade_repo():
    repo = Path(__file__).resolve().parents[1]
    res = build_checklist(repo)
    assert res["summary"]["FAIL"] == 0, [i for i in res["items"] if i["status"] == "FAIL"]
    assert res["ok"] is True
