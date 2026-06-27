"""Read-along subtitle generation (SRT / VTT) from the audio timeline."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .stitch import SegmentSpan


def _ts(seconds: float, vtt: bool = False) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    sep = "." if vtt else ","
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: List[SegmentSpan]) -> str:
    lines: List[str] = []
    idx = 1
    for seg in segments:
        if seg.kind == "skippable" or not seg.text.strip():
            continue
        lines.append(str(idx))
        lines.append(f"{_ts(seg.start)} --> {_ts(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
        idx += 1
    return "\n".join(lines)


def to_vtt(segments: List[SegmentSpan]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        if seg.kind == "skippable" or not seg.text.strip():
            continue
        lines.append(f"{_ts(seg.start, vtt=True)} --> {_ts(seg.end, vtt=True)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def write_subtitles(segments: List[SegmentSpan], path: str | Path, fmt: str = "srt") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = to_vtt(segments) if fmt == "vtt" else to_srt(segments)
    path.write_text(content, encoding="utf-8")
    return path


__all__ = ["to_srt", "to_vtt", "write_subtitles"]
