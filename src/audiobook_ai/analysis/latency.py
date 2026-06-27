"""Latency / throughput benchmark: TN normalize latency + TTS real-time-factor."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from ..config import AppConfig, run_dir
from ..data.samples import SAMPLE_TN_PAIRS
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def _percentiles(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    return {"p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2),
            "mean": round(float(a.mean()), 2)}


def benchmark(cfg: AppConfig, n: int = 30, warmup: int = 3,
              measure_synth: bool = True, save: bool = True) -> Dict:
    from ..models.normalizer import load_normalizer

    sents = [b for b, _ in SAMPLE_TN_PAIRS]
    sents = (sents * ((n // len(sents)) + 1))[:n]

    normalizer = load_normalizer(cfg.model, prefer="neural")
    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    for s in sents[:warmup]:
        normalizer.normalize(s)
    norm_lat: List[float] = []
    for s in sents:
        t0 = time.perf_counter()
        normalizer.normalize(s)
        norm_lat.append((time.perf_counter() - t0) * 1000.0)

    out: Dict = {
        "device": device,
        "normalizer": getattr(normalizer, "name", "rule"),
        "normalize_ms": _percentiles(norm_lat),
        "normalize_throughput_per_s": round(1000.0 / max(0.01, np.mean(norm_lat)), 1),
        "n": n,
    }

    if measure_synth:
        try:
            from ..synthesis.tts_backend import load_tts_backend
            tts = load_tts_backend(cfg.tts)
            text = "He paid five point two million dollars in nineteen eighty four."
            tts.synthesize(text, voice="narrator")  # warmup
            t0 = time.perf_counter()
            res = tts.synthesize(text, voice="narrator")
            wall = time.perf_counter() - t0
            out["synth"] = {"backend": tts.name, "sample_rate": tts.sample_rate,
                            "audio_seconds": round(res.duration, 3),
                            "wall_seconds": round(wall, 3),
                            "rtf": round(wall / res.duration, 4) if res.duration else None}
        except Exception as exc:
            logger.info("synth benchmark skipped (%s)", exc)
            out["synth"] = {"error": str(exc)}

    if save:
        d = run_dir() / "benchmark"
        d.mkdir(parents=True, exist_ok=True)
        import json
        (d / f"benchmark-{utc_stamp()}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


__all__ = ["benchmark"]
