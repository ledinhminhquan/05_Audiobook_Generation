"""One-button autopilot: data -> train -> eval -> analysis -> report + slides + bundle.

Each stage is isolated in try/except and never aborts the run; the autopilot
always returns a per-stage summary and writes a submission bundle to
``artifacts/submission/submission-<stamp>/``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config import AppConfig, artifacts_dir, ensure_dirs
from ..logging_utils import get_logger, utc_now_iso, utc_stamp

logger = get_logger(__name__)


def _step(steps: List[Dict], name: str, fn: Callable[[], Any], skip: bool = False) -> Optional[Any]:
    if skip:
        steps.append({"step": name, "status": "skipped"})
        return None
    try:
        out = fn()
        steps.append({"step": name, "status": "ok"})
        return out
    except Exception as exc:
        logger.warning("autopilot step %s failed: %s", name, exc)
        steps.append({"step": name, "status": "error", "error": str(exc)})
        return None


def run_autopilot(cfg: AppConfig, title: str = None, author: str = None,
                  train: bool = True, limit: Optional[int] = None) -> Dict:
    ensure_dirs()
    title = title or cfg.project_title
    author = author or cfg.author
    steps: List[Dict] = []

    _step(steps, "build_corpus", lambda: __import__(
        "audiobook_ai.data.download_dataset", fromlist=["prepare_corpus"]).prepare_corpus(cfg))
    if train:
        _step(steps, "train_normalizer", lambda: __import__(
            "audiobook_ai.training.train_normalizer", fromlist=["train_normalizer"]).train_normalizer(cfg, limit=limit))
    _step(steps, "evaluate", lambda: __import__(
        "audiobook_ai.training.evaluate", fromlist=["evaluate"]).evaluate(cfg))
    _step(steps, "benchmark", lambda: __import__(
        "audiobook_ai.analysis.latency", fromlist=["benchmark"]).benchmark(cfg, n=20, warmup=2))
    _step(steps, "error_analysis", lambda: __import__(
        "audiobook_ai.analysis.error_analysis", fromlist=["error_analysis"]).error_analysis(cfg))
    _step(steps, "audio_quality", lambda: __import__(
        "audiobook_ai.analysis.audio_quality", fromlist=["audio_quality_report"]).audio_quality_report(cfg))

    stamp = utc_stamp()
    sub = artifacts_dir() / "submission" / f"submission-{stamp}"
    sub.mkdir(parents=True, exist_ok=True)

    report = _step(steps, "report", lambda: __import__(
        "audiobook_ai.autoreport.report_pdf", fromlist=["generate_report"]).generate_report(
        cfg, title=title, author=author, out_path=sub / "report.pdf"))
    slides = _step(steps, "slides", lambda: __import__(
        "audiobook_ai.autoreport.slides_pptx", fromlist=["generate_slides"]).generate_slides(
        cfg, title=title, author=author, out_path=sub / "slides.pptx"))
    checklist = _step(steps, "grading", lambda: __import__(
        "audiobook_ai.grading.checklist", fromlist=["build_checklist"]).build_checklist(
        Path(__file__).resolve().parents[3]))

    manifest = {"generated_at": utc_now_iso(), "title": title, "author": author,
                "steps": steps, "grading_checklist": checklist}
    (sub / "submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    try:
        with zipfile.ZipFile(sub / "submission_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in sub.iterdir():
                if f.is_file() and f.name != "submission_bundle.zip":
                    z.write(f, f.name)
    except Exception as exc:
        logger.warning("bundle zip failed: %s", exc)

    logger.info("Autopilot done -> %s", sub)
    return {"submission_dir": str(sub), "steps": steps,
            "grading": (checklist or {}).get("summary"),
            "report": str(report) if report else None, "slides": str(slides) if slides else None}


__all__ = ["run_autopilot"]
