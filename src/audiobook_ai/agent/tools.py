"""Agent tools — each operates on the JobState and returns it.

Tools never assume a GPU or a trained model: the normalizer can be the rule
baseline and the TTS backend can be the placeholder synth, so the whole pipeline
runs offline for tests/CI. The orchestrator wraps each call with timing/trace.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import AppConfig
from ..data.document import Document, document_from_text, parse_document
from ..logging_utils import get_logger
from ..models.baseline_rules import RuleNormalizer
from . import policy
from .state import Decision, JobState, NormSegment

logger = get_logger(__name__)


# ── parse + segment + classify (S0–S3) ──────────────────────────────────────
def tool_parse(job: JobState, cfg: AppConfig, *, path: Optional[str] = None,
               text: Optional[str] = None, title: Optional[str] = None,
               author: Optional[str] = None) -> JobState:
    if path:
        doc = parse_document(path, cfg.parse, title=title, author=author)
    else:
        doc = document_from_text(text or "", cfg.parse, title=title or "Pasted Text")
    job.title = title or doc.title
    job.author = author or doc.author
    job.source_format = doc.source_format
    job.n_chapters = doc.n_chapters
    job.metrics["chapter_titles"] = {c.index: c.title for c in doc.chapters}
    job.segments = [NormSegment(text=s.text, kind=s.kind, chapter_index=s.chapter_index, order=s.order)
                    for s in doc.iter_segments() if s.kind != "skippable"]
    job.parse_score = policy.compute_parse_score(doc)
    route = policy.parse_route(job.parse_score)
    job.add_decision(Decision(id="D1", name="parse_quality_routing", branch=route,
                              score=job.parse_score,
                              detail=f"{doc.n_chapters} chapters, {len(job.segments)} segments, format={doc.source_format}"))
    return job


# ── normalize (S4, D2) ───────────────────────────────────────────────────────
def tool_normalize(job: JobState, cfg: AppConfig, *, normalizer, brain=None) -> JobState:
    segs = job.segments
    if not segs:
        return job
    texts = [s.text for s in segs]
    baseline = RuleNormalizer()
    base_out = baseline.normalize_batch(texts)

    is_neural = hasattr(normalizer, "normalize_batch_with_conf")
    if is_neural:
        neural_out, confs = normalizer.normalize_batch_with_conf(texts)
        job.model_versions["normalizer"] = getattr(normalizer, "version", "neural")
    else:
        neural_out, confs = [None] * len(texts), [1.0] * len(texts)
        job.model_versions["normalizer"] = getattr(normalizer, "version", "rule_baseline")

    n_flagged = n_escalated = n_disagree = 0
    for i, s in enumerate(segs):
        d = policy.decide_normalization(neural_out[i], confs[i], base_out[i], cfg.agent)
        if d["escalate"] and brain is not None and brain.available():
            adv = brain.disambiguate_normalization(s.text, neural_out[i] or base_out[i], base_out[i])
            if adv is not None:
                d["text"], d["source"], d["flagged"] = adv["text"], "llm", False
                d["reason"] = f"llm: {adv.get('rationale','')[:60]}"
                n_escalated += 1
        s.normalized = d["text"]
        s.confidence = float(d["confidence"])
        s.source = d["source"]
        s.flagged = bool(d["flagged"])
        s.flag_reason = d["reason"]
        n_flagged += int(s.flagged)
        n_disagree += int(d["disagree"])

    job.add_decision(Decision(id="D2", name="normalization_confidence", branch="escalated" if n_escalated else ("flagged" if n_flagged else "confident"),
                              score=round(float(np.mean(confs)), 4) if confs else 1.0,
                              detail=f"flagged={n_flagged}, escalated={n_escalated}, baseline_disagree={n_disagree}/{len(segs)}",
                              llm_used=n_escalated > 0))
    job.metrics["normalization"] = {"source": "neural" if is_neural else "rule",
                                    "flagged": n_flagged, "escalated": n_escalated,
                                    "baseline_disagree": n_disagree, "n": len(segs)}
    return job


# ── voice routing (S5, D4) ───────────────────────────────────────────────────
def tool_route_voices(job: JobState, cfg: AppConfig) -> JobState:
    counts: Dict[str, int] = {}
    for s in job.segments:
        s.voice = policy.route_voice(s.kind, cfg.tts)
        counts[s.voice] = counts.get(s.voice, 0) + 1
    job.add_decision(Decision(id="D4", name="voice_routing", branch="multi" if len(counts) > 1 else "single",
                              detail=", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))))
    job.metrics["voices"] = counts
    return job


# ── synthesize (S6) ──────────────────────────────────────────────────────────
def tool_synthesize(job: JobState, cfg: AppConfig, *, tts) -> JobState:
    job._sample_rate = tts.sample_rate
    job.model_versions["tts"] = tts.name
    t0 = time.perf_counter()
    for i, s in enumerate(job.segments):
        res = tts.synthesize(s.normalized or s.text, voice=s.voice)
        job._audio[i] = res.audio
        s.audio_duration = res.duration
    job.metrics["synth_seconds"] = round(time.perf_counter() - t0, 3)
    return job


# ── audio QA (S7, D3) ────────────────────────────────────────────────────────
def tool_audio_qa(job: JobState, cfg: AppConfig, *, tts) -> JobState:
    n_pass = n_fail = n_resynth = 0
    for i, s in enumerate(job.segments):
        audio = job._audio.get(i, np.zeros(0, dtype=np.float32))
        qa = policy.audio_qa_checks(audio, job._sample_rate, s.normalized or s.text, cfg.agent)
        attempt = 0
        while not qa["pass"] and attempt < cfg.agent.max_resynth_attempts:
            strategy = policy.resynth_strategy(attempt)
            try:
                res = tts.synthesize(s.normalized or s.text, voice=s.voice)
                cand = res.audio
                cand_qa = policy.audio_qa_checks(cand, job._sample_rate, s.normalized or s.text, cfg.agent)
                # keep the better of the two by pass-then-duration sanity
                if cand_qa["pass"] or abs(cand_qa["checks"]["duration_ratio"] - 1) < abs(qa["checks"]["duration_ratio"] - 1):
                    audio, qa = cand, cand_qa
                    job._audio[i] = cand
                    s.audio_duration = res.duration
            except Exception as exc:
                logger.info("resynth (%s) failed: %s", strategy, exc)
            attempt += 1
            n_resynth += 1
        s.resynth_count = attempt
        s.qa = qa["checks"]
        s.qa["pass"] = qa["pass"]
        if qa["pass"]:
            n_pass += 1
        else:
            n_fail += 1
            s.flagged = True
            s.flag_reason = (s.flag_reason + "; audio_qa_failed").strip("; ")
    job.add_decision(Decision(id="D3", name="audio_qa_gate",
                              branch="all_pass" if n_fail == 0 else "some_failed",
                              score=round(n_pass / max(1, n_pass + n_fail), 4),
                              detail=f"pass={n_pass}, fail={n_fail}, resynth_attempts={n_resynth}"))
    job.metrics["audio_qa"] = {"pass": n_pass, "fail": n_fail, "resynth_attempts": n_resynth}
    return job


# ── stitch + master + export (S8–S9) ─────────────────────────────────────────
def tool_stitch(job: JobState, cfg: AppConfig, *, out_dir: str, base_name: str) -> JobState:
    from ..synthesis import stitch as st
    from ..synthesis.subtitles import write_subtitles

    titles = job.metrics.get("chapter_titles", {})
    chapters_audio = job.chapters_audio()
    for ch in chapters_audio:
        ch["title"] = titles.get(ch["index"], f"Chapter {ch['index'] + 1}")
    timeline = st.assemble(chapters_audio, job._sample_rate, cfg.audio)
    timeline.audio = st.normalize_loudness(timeline.audio, job._sample_rate,
                                           cfg.audio.target_lufs, cfg.audio.peak_dbfs)
    out_dir_p = Path(out_dir)
    paths = st.export_all(timeline, out_dir_p, base_name, cfg.audio, title=job.title, author=job.author)
    if cfg.audio.subtitles:
        sub = write_subtitles(timeline.segments, out_dir_p / f"{base_name}.{cfg.audio.subtitle_format}",
                              cfg.audio.subtitle_format)
        paths[cfg.audio.subtitle_format] = str(sub)
    job.outputs.update(paths)
    lufs = st.measure_lufs(timeline.audio, job._sample_rate)
    job.metrics["audio_duration"] = round(timeline.duration, 2)
    job.metrics["measured_lufs"] = round(lufs, 2) if lufs is not None else None
    job.metrics["sample_rate"] = job._sample_rate
    synth_s = job.metrics.get("synth_seconds", 0.0)
    job.metrics["rtf"] = round(synth_s / timeline.duration, 4) if timeline.duration else None
    return job


__all__ = ["tool_parse", "tool_normalize", "tool_route_voices", "tool_synthesize",
           "tool_audio_qa", "tool_stitch"]
