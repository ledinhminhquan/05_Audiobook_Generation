"""Data preparation entrypoint.

For this project the primary corpus is *generated*, not downloaded: this builds
and caches the synthetic TN splits, writes the sample book, and (optionally)
pre-fetches the speaker-embedding dataset + a small real eval set. Network steps
degrade gracefully so ``data`` works fully offline.
"""

from __future__ import annotations

from typing import Dict

from ..config import AppConfig
from ..logging_utils import get_logger
from .dataset import build_corpus, corpus_dir
from .samples import write_sample_book

logger = get_logger(__name__)


def prepare_corpus(cfg: AppConfig) -> Dict:
    splits = build_corpus(cfg, save=True)
    sample = write_sample_book(corpus_dir().parent / "samples")
    return {"task": "tn_corpus", "dir": str(corpus_dir()),
            "counts": {k: len(v) for k, v in splits.items()},
            "sample_book": str(sample)}


def prefetch_speaker_embeddings(cfg: AppConfig) -> Dict:
    try:
        from datasets import load_dataset  # lazy
        ds = load_dataset(cfg.tts.speaker_xvectors, split="validation")
        return {"task": "speaker_xvectors", "id": cfg.tts.speaker_xvectors, "rows": len(ds)}
    except Exception as exc:
        logger.warning("Could not prefetch speaker embeddings (%s)", exc)
        return {"task": "speaker_xvectors", "id": cfg.tts.speaker_xvectors, "error": str(exc)}


def download_all(cfg: AppConfig) -> Dict:
    out = {"corpus": prepare_corpus(cfg)}
    out["speaker_embeddings"] = prefetch_speaker_embeddings(cfg)
    return out


def download_task(task: str, cfg: AppConfig) -> Dict:
    if task == "corpus":
        return prepare_corpus(cfg)
    if task == "speakers":
        return prefetch_speaker_embeddings(cfg)
    raise ValueError(f"Unknown data task: {task}")


__all__ = ["prepare_corpus", "prefetch_speaker_embeddings", "download_all", "download_task"]
