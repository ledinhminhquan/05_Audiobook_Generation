"""The narrator agent — a deterministic FSM that turns a document into an audiobook.

    parse (D1) -> normalize (D2, optional LLM) -> [synth path:] route-voices (D4)
        -> synthesize -> audio-QA (D3, bounded re-synth) -> stitch+master+export

Runs fully offline (rule normalizer + placeholder TTS) and upgrades automatically
when a fine-tuned model / neural TTS are present. Every step is timed and traced;
a manifest.json captures the full reproducible record.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable, Optional

from ..config import AppConfig, ensure_dirs, output_dir
from ..logging_utils import JsonlLogger, get_logger, utc_stamp
from ..models.normalizer import load_normalizer
from . import tools
from .llm_orchestrator import LLMBrain
from .state import Decision, JobState, JobStatus, ToolTrace

logger = get_logger(__name__)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()[:48] or "audiobook"


class NarratorAgent:
    def __init__(self, cfg: Optional[AppConfig] = None, *, load_model: bool = True,
                 tts_backend: Optional[str] = None):
        self.cfg = cfg or AppConfig()
        self.normalizer = load_normalizer(self.cfg.model, prefer="neural" if load_model else "rule")
        self.brain = LLMBrain(self.cfg.agent)
        self._tts = None
        self._tts_backend = tts_backend
        ensure_dirs()
        self._log = JsonlLogger(self.cfg.serving.job_log_path) if self.cfg.serving.log_jobs else None

    # ---- lazy TTS engine ---------------------------------------------------
    def _tts_engine(self):
        if self._tts is None:
            from ..synthesis.tts_backend import load_tts_backend
            self._tts = load_tts_backend(self.cfg.tts, backend=self._tts_backend)
        return self._tts

    # ---- timed step --------------------------------------------------------
    def _step(self, job: JobState, name: str, fn: Callable[[], JobState], summary: str = "") -> JobState:
        t0 = time.perf_counter()
        try:
            job = fn()
            ok, err = True, None
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            ok, err = False, str(exc)
        job.add_trace(ToolTrace(tool=name, ok=ok, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                summary=summary or name, error=err))
        return job

    # ---- main entrypoint ---------------------------------------------------
    def process(self, *, path: Optional[str] = None, text: Optional[str] = None,
                title: Optional[str] = None, author: Optional[str] = None,
                synth: bool = True, out_dir: Optional[str] = None) -> JobState:
        filename = Path(path).name if path else (title or "pasted_text")
        job = JobState(filename=filename, title=title or "Audiobook", author=author or self.cfg.author)

        job = self._step(job, "parse", lambda: tools.tool_parse(job, self.cfg, path=path, text=text,
                                                                title=title, author=author),
                         summary="parse + segment + classify")
        if not job.segments:
            job.review_reasons.append("no readable text extracted from the document")
            return self._finish(job, JobStatus.FAILED, out_dir, synth)

        job = self._step(job, "normalize", lambda: tools.tool_normalize(job, self.cfg, normalizer=self.normalizer,
                                                                        brain=self.brain),
                         summary="text normalization (D2)")

        if not synth:
            return self._finish(job, JobStatus.NORMALIZED, out_dir, synth=False)

        base = _slug(job.title)
        odir = out_dir or str(output_dir() / f"{base}-{utc_stamp()}")
        job = self._step(job, "route_voices", lambda: tools.tool_route_voices(job, self.cfg), summary="voice routing (D4)")
        tts = self._tts_engine()
        job.model_versions["tts_backend"] = tts.name
        job = self._step(job, "synthesize", lambda: tools.tool_synthesize(job, self.cfg, tts=tts),
                         summary=f"TTS ({tts.name})")
        job = self._step(job, "audio_qa", lambda: tools.tool_audio_qa(job, self.cfg, tts=tts), summary="audio QA (D3)")
        job = self._step(job, "stitch", lambda: tools.tool_stitch(job, self.cfg, out_dir=odir, base_name=base),
                         summary="stitch + master + export")

        flagged = sum(s.flagged for s in job.segments)
        status = JobStatus.NEEDS_REVIEW if flagged else JobStatus.COMPLETED
        return self._finish(job, status, odir, synth=True)

    def normalize_preview(self, text: str, title: str = "Preview") -> JobState:
        """Run only parse + normalize (no audio) — used by the API /normalize + demo."""
        return self.process(text=text, title=title, synth=False)

    # ---- finalize ----------------------------------------------------------
    def _finish(self, job: JobState, status: JobStatus, out_dir: Optional[str], synth: bool) -> JobState:
        job.status = status
        job.model_versions.setdefault("model_version", self.cfg.serving.model_version)
        if synth and out_dir:
            try:
                manifest = {"job": job.to_dict(), "config": {
                    "tts_backend": job.model_versions.get("tts_backend"),
                    "target_lufs": self.cfg.audio.target_lufs, "peak_dbfs": self.cfg.audio.peak_dbfs,
                    "model_version": self.cfg.serving.model_version}}
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
                job.outputs["manifest"] = str(Path(out_dir) / "manifest.json")
            except Exception as exc:
                logger.warning("manifest write failed: %s", exc)
        if self._log is not None:
            try:
                self._log.log("job", filename=job.filename, status=status.value,
                              n_segments=len(job.segments), n_flagged=sum(s.flagged for s in job.segments),
                              metrics=job.metrics)
            except Exception:
                pass
        return job


_AGENT: Optional[NarratorAgent] = None


def get_agent(cfg: Optional[AppConfig] = None, **kwargs) -> NarratorAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = NarratorAgent(cfg, **kwargs)
    return _AGENT


__all__ = ["NarratorAgent", "get_agent"]
