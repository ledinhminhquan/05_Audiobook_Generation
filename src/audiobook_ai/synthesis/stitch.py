"""Assemble per-segment audio into a mastered, chaptered audiobook.

Concatenates segments with sentence/paragraph/chapter silence gaps, tracks the
timeline (for subtitles + chapter markers), normalizes loudness toward the ACX
target (−18 LUFS / −3 dBTP), and exports WAV / MP3 / M4B (+ chapters). Heavy libs
(soundfile, pydub, pyloudnorm, ffmpeg) are optional — WAV always works via the
stdlib ``wave`` module; MP3/M4B are skipped gracefully when ffmpeg is absent.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import AudioConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentSpan:
    text: str
    kind: str
    chapter_index: int
    start: float
    end: float


@dataclass
class ChapterSpan:
    title: str
    index: int
    start: float
    end: float


@dataclass
class Timeline:
    audio: np.ndarray
    sample_rate: int
    segments: List[SegmentSpan] = field(default_factory=list)
    chapters: List[ChapterSpan] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return len(self.audio) / float(self.sample_rate) if self.sample_rate else 0.0


def _silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(max(0.0, seconds) * sr), dtype=np.float32)


def assemble(chapters_audio: List[Dict], sr: int, cfg: AudioConfig) -> Timeline:
    """``chapters_audio`` = ``[{title, index, segments:[{text,kind,audio}]}]`` -> Timeline."""
    book: List[np.ndarray] = []
    seg_spans: List[SegmentSpan] = []
    ch_spans: List[ChapterSpan] = []
    t = 0.0
    sent_gap = cfg.sentence_gap_ms / 1000.0
    para_gap = cfg.paragraph_gap_ms / 1000.0
    chap_gap = cfg.chapter_gap_ms / 1000.0

    for ci, ch in enumerate(chapters_audio):
        ch_start = t
        segs = ch.get("segments", [])
        for si, seg in enumerate(segs):
            audio = np.asarray(seg.get("audio", np.zeros(0, dtype=np.float32)), dtype=np.float32)
            dur = len(audio) / float(sr) if sr else 0.0
            book.append(audio)
            seg_spans.append(SegmentSpan(text=seg.get("text", ""), kind=seg.get("kind", "narration"),
                                         chapter_index=ci, start=t, end=t + dur))
            t += dur
            if si < len(segs) - 1:
                gap = para_gap if seg.get("kind") == "heading" else sent_gap
                book.append(_silence(gap, sr))
                t += gap
        ch_spans.append(ChapterSpan(title=ch.get("title", f"Chapter {ci + 1}"), index=ci,
                                    start=ch_start, end=t))
        if ci < len(chapters_audio) - 1:
            book.append(_silence(chap_gap, sr))
            t += chap_gap

    audio = np.concatenate(book) if book else np.zeros(0, dtype=np.float32)
    return Timeline(audio=audio, sample_rate=sr, segments=seg_spans, chapters=ch_spans)


# ─────────────────────────────────────────────────────────────────────────────
# Loudness / peak normalization
# ─────────────────────────────────────────────────────────────────────────────
def measure_lufs(audio: np.ndarray, sr: int) -> Optional[float]:
    try:
        import pyloudnorm as pyln  # lazy
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(audio.astype(np.float64)))
    except Exception:
        return None


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float, peak_dbfs: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    out = audio.astype(np.float32)
    lufs = measure_lufs(out, sr)
    if lufs is not None and np.isfinite(lufs):
        gain = 10 ** ((target_lufs - lufs) / 20.0)
        out = out * gain
    else:  # RMS fallback toward an approximate LUFS-equivalent RMS target
        rms = float(np.sqrt(np.mean(out ** 2))) or 1e-9
        target_rms = 10 ** ((target_lufs + 0.0) / 20.0)
        out = out * (target_rms / rms)
    peak = float(np.max(np.abs(out))) or 1e-9
    ceiling = 10 ** (peak_dbfs / 20.0)
    if peak > ceiling:
        out = out * (ceiling / peak)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Encoding / export
# ─────────────────────────────────────────────────────────────────────────────
def _to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def write_wav(audio: np.ndarray, sr: int, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf  # lazy, preferred
        sf.write(str(path), audio.astype(np.float32), sr)
        return path
    except Exception:
        pass
    with wave.open(str(path), "wb") as w:        # stdlib fallback
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(_to_int16(audio).tobytes())
    return path


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def export_mp3(wav_path: str | Path, mp3_path: str | Path) -> Optional[Path]:
    mp3_path = Path(mp3_path)
    try:
        from pydub import AudioSegment  # lazy (uses ffmpeg)
        AudioSegment.from_wav(str(wav_path)).export(str(mp3_path), format="mp3", bitrate="128k")
        return mp3_path
    except Exception as exc:
        if _has_ffmpeg():
            try:
                subprocess.run(["ffmpeg", "-y", "-i", str(wav_path), "-b:a", "128k", str(mp3_path)],
                               check=True, capture_output=True)
                return mp3_path
            except Exception as exc2:
                logger.info("mp3 export via ffmpeg failed (%s)", exc2)
        else:
            logger.info("mp3 export skipped (no pydub/ffmpeg): %s", exc)
    return None


def _ffmetadata(chapters: List[ChapterSpan], title: str, author: str) -> str:
    lines = [";FFMETADATA1", f"title={title}", f"artist={author}", f"album={title}"]
    for ch in chapters:
        lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                  f"START={int(ch.start * 1000)}", f"END={int(ch.end * 1000)}",
                  f"title={ch.title}"]
    return "\n".join(lines) + "\n"


def export_m4b(wav_path: str | Path, m4b_path: str | Path, chapters: List[ChapterSpan],
               title: str, author: str) -> Optional[Path]:
    if not _has_ffmpeg():
        logger.info("m4b export skipped (ffmpeg not found)")
        return None
    m4b_path = Path(m4b_path)
    meta = Path(tempfile.mktemp(suffix=".txt"))
    meta.write_text(_ffmetadata(chapters, title, author), encoding="utf-8")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_path), "-i", str(meta),
                        "-map_metadata", "1", "-c:a", "aac", "-b:a", "96k", str(m4b_path)],
                       check=True, capture_output=True)
        return m4b_path
    except Exception as exc:
        logger.info("m4b export failed (%s)", exc)
        return None
    finally:
        try:
            meta.unlink()
        except OSError:
            pass


def export_all(timeline: Timeline, out_dir: str | Path, base_name: str, cfg: AudioConfig,
               title: str = "Audiobook", author: str = "Unknown") -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    wav = out_dir / f"{base_name}.wav"
    write_wav(timeline.audio, timeline.sample_rate, wav)
    paths["wav"] = str(wav)
    if "mp3" in cfg.output_formats:
        mp3 = export_mp3(wav, out_dir / f"{base_name}.mp3")
        if mp3:
            paths["mp3"] = str(mp3)
    if cfg.make_m4b:
        m4b = export_m4b(wav, out_dir / f"{base_name}.m4b", timeline.chapters, title, author)
        if m4b:
            paths["m4b"] = str(m4b)
    return paths


__all__ = ["Timeline", "SegmentSpan", "ChapterSpan", "assemble", "normalize_loudness",
           "measure_lufs", "write_wav", "export_mp3", "export_m4b", "export_all"]
