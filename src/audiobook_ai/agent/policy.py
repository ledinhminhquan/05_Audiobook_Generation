"""Decision-point logic for the narrator agent (pure, testable, no model deps).

Four explicit decision points act on intermediate outputs:
* **D1** parse-quality / format routing,
* **D2** normalization-confidence escalation,
* **D3** audio-QA re-synthesis gate,
* **D4** per-segment voice routing.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

import numpy as np

from ..config import AgentConfig

_WS = re.compile(r"\s+")


def _ws(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


# ── D1 ───────────────────────────────────────────────────────────────────────
def compute_parse_score(document) -> float:
    segs = [s for s in document.iter_segments() if s.kind != "skippable"]
    if not segs:
        return 0.0
    total = sum(len(s.text) for s in segs)
    if total < 20:
        return 0.15
    alpha = sum(1 for s in segs for c in s.text if c.isalpha() or c.isspace())
    alpha_ratio = alpha / max(1, total)
    has_struct = 1.0 if (document.n_chapters > 1 or any(s.kind == "heading" for s in document.iter_segments())) else 0.65
    avg_len = total / len(segs)
    len_ok = 1.0 if 10 <= avg_len <= 2000 else 0.5
    return round(min(1.0, 0.5 * alpha_ratio + 0.3 * has_struct + 0.2 * len_ok), 4)


def parse_route(score: float) -> str:
    if score >= 0.85:
        return "structured"
    if score >= 0.5:
        return "assisted"
    return "degraded"


# ── D2 ───────────────────────────────────────────────────────────────────────
def decide_normalization(neural_text: Optional[str], neural_conf: float,
                         baseline_text: str, cfg: AgentConfig) -> Dict[str, Any]:
    """Pick the normalized text + decide whether to flag/escalate (D2)."""
    if neural_text is None:
        return {"text": baseline_text, "source": "rule", "flagged": False,
                "reason": "", "confidence": 1.0, "disagree": False, "escalate": False}
    disagree = _ws(neural_text) != _ws(baseline_text)
    low_conf = neural_conf < cfg.norm_confidence_min
    flagged = low_conf
    reason = f"low confidence {neural_conf:.2f}" if low_conf else ""
    return {"text": neural_text, "source": "neural", "flagged": flagged, "reason": reason,
            "confidence": neural_conf, "disagree": disagree, "escalate": low_conf}


# ── D3 ───────────────────────────────────────────────────────────────────────
def audio_qa_checks(audio: np.ndarray, sr: int, text: str, cfg: AgentConfig) -> Dict[str, Any]:
    n = int(len(audio))
    dur = n / sr if sr else 0.0
    expected = max(0.3, len(text) * 0.06)
    ratio = dur / expected if expected else 0.0
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if n else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if n else 0.0
    floor = 10 ** (cfg.audio_silence_dbfs / 20.0)
    silence_frac = float(np.mean(np.abs(audio) < floor)) if n else 1.0
    clip_ceiling = 10 ** (cfg.audio_clip_peak_dbfs / 20.0)
    checks = {
        "empty": bool(n == 0 or rms < 1e-5),
        "duration": round(dur, 3),
        "duration_ratio": round(ratio, 3),
        "duration_ok": bool(cfg.audio_min_dur_ratio <= ratio <= cfg.audio_max_dur_ratio),
        "peak_dbfs": round(20 * math.log10(peak + 1e-9), 2),
        "clipped": bool(peak >= clip_ceiling),
        "rms_dbfs": round(20 * math.log10(rms + 1e-9), 2),
        "silence_frac": round(silence_frac, 3),
        "too_silent": bool(silence_frac > cfg.audio_max_silence_ratio),
    }
    passed = (not checks["empty"]) and checks["duration_ok"] and (not checks["clipped"]) and (not checks["too_silent"])
    action = "accept" if passed else "resynth"
    return {"pass": bool(passed), "checks": checks, "action": action}


def resynth_strategy(attempt: int) -> str:
    """Escalating re-synthesis strategy for D3 retries."""
    return {0: "reseed_jitter", 1: "split_and_pad"}.get(attempt, "fallback_backend")


# ── D4 ───────────────────────────────────────────────────────────────────────
def route_voice(kind: str, cfg_tts) -> str:
    if kind == "heading":
        return cfg_tts.heading_voice
    if kind == "dialogue":
        return cfg_tts.dialogue_voice
    return cfg_tts.narrator_voice


__all__ = ["compute_parse_score", "parse_route", "decide_normalization",
           "audio_qa_checks", "resynth_strategy", "route_voice"]
