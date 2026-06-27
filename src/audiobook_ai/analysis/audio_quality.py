"""Audio-quality report: run the agent on the sample book and aggregate QA.

Summarizes the per-segment audio-QA gate (D3) — pass-rate, peak/RMS, duration
ratios, silence fraction, re-synthesis attempts — plus the final mastered
loudness. Defaults to the deterministic placeholder backend so the report is
fast and reproducible without a TTS model; pass ``backend`` to use a real one.
"""

from __future__ import annotations

import json
from statistics import mean
from typing import Dict, List, Optional

from ..config import AppConfig, data_dir, run_dir
from ..data.samples import write_sample_book
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def audio_quality_report(cfg: AppConfig, backend: str = "placeholder",
                         save: bool = True) -> Dict:
    from ..agent.narrator_agent import NarratorAgent

    book = write_sample_book(data_dir() / "samples")
    agent = NarratorAgent(cfg, load_model=False, tts_backend=backend)
    job = agent.process(path=str(book), title="QA Sample", synth=True)
    sd = job.to_dict()

    qa_rows: List[Dict] = [s["qa"] for s in sd["segments"] if s.get("qa")]
    def _avg(key):
        vals = [q[key] for q in qa_rows if isinstance(q.get(key), (int, float))]
        return round(mean(vals), 3) if vals else None

    n = len(qa_rows)
    n_pass = sum(1 for q in qa_rows if q.get("pass"))
    result = {
        "backend": backend,
        "n_segments": n,
        "pass_rate": round(n_pass / max(1, n), 4),
        "fail": n - n_pass,
        "resynth_attempts": sd["metrics"].get("audio_qa", {}).get("resynth_attempts", 0),
        "avg_duration_ratio": _avg("duration_ratio"),
        "avg_peak_dbfs": _avg("peak_dbfs"),
        "avg_rms_dbfs": _avg("rms_dbfs"),
        "avg_silence_frac": _avg("silence_frac"),
        "measured_lufs": sd["metrics"].get("measured_lufs"),
        "target_lufs": cfg.audio.target_lufs,
        "audio_duration": sd["metrics"].get("audio_duration"),
        "rtf": sd["metrics"].get("rtf"),
    }
    if save:
        d = run_dir() / "audio_quality"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"audio-{utc_stamp()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = ["audio_quality_report"]
