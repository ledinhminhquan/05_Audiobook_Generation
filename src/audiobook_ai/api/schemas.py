"""Pydantic request/response schemas for the Audiobook Generation API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NormalizeRequest(BaseModel):
    text: str = Field(..., description="Written text to normalize for TTS")


class SegmentView(BaseModel):
    text: str
    normalized: str
    kind: str
    voice: str = "narrator"
    confidence: float = 1.0
    source: str = "rule"
    flagged: bool = False


class NormalizeResponse(BaseModel):
    normalized: str
    n_segments: int
    source: str
    segments: List[SegmentView]


class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Document text to convert into an audiobook")
    title: str = "Audiobook"
    author: str = "Unknown"
    voice: Optional[str] = None
    backend: Optional[str] = Field(None, description="TTS backend: speecht5|kokoro|parler|pyttsx3")


class JobResponse(BaseModel):
    status: str
    title: str
    source_format: str
    n_chapters: int
    n_segments: int
    n_flagged: int
    parse_score: float
    metrics: Dict[str, Any]
    outputs: Dict[str, str]
    decisions: List[Dict[str, Any]]
    model_versions: Dict[str, str]


class HealthResponse(BaseModel):
    status: str
    normalizer: str
    version: str


__all__ = ["NormalizeRequest", "NormalizeResponse", "SegmentView",
           "SynthesizeRequest", "JobResponse", "HealthResponse"]
