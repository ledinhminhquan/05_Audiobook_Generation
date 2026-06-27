"""Shared state types for the narrator agent (FSM context + audit records)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    PARSED = "parsed"
    NORMALIZED = "normalized"
    SYNTHESIZED = "synthesized"
    QA_DONE = "qa_done"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


@dataclass
class ToolTrace:
    tool: str
    ok: bool
    latency_ms: float
    summary: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "ok": self.ok, "latency_ms": self.latency_ms,
                "summary": self.summary, "error": self.error}


@dataclass
class Decision:
    id: str                       # "D1".."D4"
    name: str
    branch: str
    score: Optional[float] = None
    detail: str = ""
    llm_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "branch": self.branch,
                "score": self.score, "detail": self.detail, "llm_used": self.llm_used}


@dataclass
class NormSegment:
    """A spoken segment carried through normalize -> voice -> synth -> QA."""
    text: str
    kind: str = "narration"
    chapter_index: int = 0
    order: int = 0
    normalized: str = ""
    confidence: float = 1.0
    source: str = "rule"          # "neural" | "rule" | "llm"
    flagged: bool = False
    flag_reason: str = ""
    voice: str = "narrator"
    audio_duration: float = 0.0
    resynth_count: int = 0
    qa: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "kind": self.kind, "chapter_index": self.chapter_index,
                "order": self.order, "normalized": self.normalized, "confidence": round(self.confidence, 4),
                "source": self.source, "flagged": self.flagged, "flag_reason": self.flag_reason,
                "voice": self.voice, "audio_duration": round(self.audio_duration, 3),
                "resynth_count": self.resynth_count, "qa": self.qa}


@dataclass
class JobState:
    filename: str = "document"
    title: str = "Audiobook"
    author: str = "Unknown"
    status: JobStatus = JobStatus.PENDING
    source_format: str = "txt"
    n_chapters: int = 0
    parse_score: float = 0.0
    segments: List[NormSegment] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    trace: List[ToolTrace] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    review_reasons: List[str] = field(default_factory=list)
    model_versions: Dict[str, str] = field(default_factory=dict)
    # transient (not serialized): raw per-segment audio + sample rate
    _audio: Dict[int, Any] = field(default_factory=dict, repr=False)
    _sample_rate: int = 16000

    def add_trace(self, t: ToolTrace) -> None:
        self.trace.append(t)

    def add_decision(self, d: Decision) -> None:
        self.decisions.append(d)

    def spoken(self) -> List[NormSegment]:
        return [s for s in self.segments if s.kind != "skippable"]

    def chapters_audio(self) -> List[Dict[str, Any]]:
        """Group per-segment audio by chapter for the stitcher."""
        groups: Dict[int, Dict[str, Any]] = {}
        for i, s in enumerate(self.segments):
            if s.kind == "skippable":
                continue
            g = groups.setdefault(s.chapter_index, {"title": "", "index": s.chapter_index, "segments": []})
            g["segments"].append({"text": s.normalized or s.text, "kind": s.kind,
                                  "audio": self._audio.get(i)})
        return [groups[k] for k in sorted(groups)]

    def to_dict(self) -> Dict[str, Any]:
        n_flagged = sum(s.flagged for s in self.segments)
        return {
            "filename": self.filename, "title": self.title, "author": self.author,
            "status": self.status.value, "source_format": self.source_format,
            "n_chapters": self.n_chapters, "n_segments": len(self.segments),
            "n_spoken_segments": len(self.spoken()), "n_flagged": n_flagged,
            "parse_score": round(self.parse_score, 4),
            "metrics": self.metrics, "outputs": self.outputs,
            "decisions": [d.to_dict() for d in self.decisions],
            "trace": [t.to_dict() for t in self.trace],
            "review_reasons": self.review_reasons,
            "model_versions": self.model_versions,
            "segments": [s.to_dict() for s in self.segments],
        }


__all__ = ["JobStatus", "ToolTrace", "Decision", "NormSegment", "JobState"]
