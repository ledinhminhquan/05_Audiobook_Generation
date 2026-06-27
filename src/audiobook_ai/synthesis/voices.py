"""Speaker / voice bank for the TTS backends.

Maps named voices ("narrator", "dialogue", "heading", ...) to SpeechT5 x-vector
speaker embeddings loaded from ``Matthijs/cmu-arctic-xvectors``. When the dataset
or torch is unavailable it returns a deterministic pseudo-embedding so synthesis
still proceeds (a generic, stable voice). All heavy imports are lazy.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

from ..config import TtsConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class SpeakerBank:
    def __init__(self, cfg: TtsConfig):
        self.cfg = cfg
        self._ds = None
        self._cache: Dict[str, "object"] = {}
        self._loaded = False

    def _ensure_dataset(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from datasets import load_dataset  # lazy
            self._ds = load_dataset(self.cfg.speaker_xvectors, split="validation")
            logger.info("Loaded %d speaker x-vectors from %s", len(self._ds), self.cfg.speaker_xvectors)
        except Exception as exc:
            logger.info("Speaker x-vectors unavailable (%s); using deterministic fallback voices.", exc)
            self._ds = None

    def _pseudo_embedding(self, voice: str):
        """Deterministic 512-d unit-ish vector derived from the voice name (offline fallback)."""
        import torch  # lazy
        seed = int(hashlib.sha256(voice.encode()).hexdigest(), 16) % (2 ** 31)
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(512, generator=g) * 0.05
        return v.unsqueeze(0)

    def get(self, voice: str):
        """Return a ``[1, 512]`` speaker-embedding tensor for ``voice``."""
        if voice in self._cache:
            return self._cache[voice]
        self._ensure_dataset()
        emb = None
        if self._ds is not None:
            try:
                import torch  # lazy
                idx = self.cfg.voice_indices.get(voice, self.cfg.voice_indices.get(self.cfg.narrator_voice, 7306))
                idx = min(idx, len(self._ds) - 1)
                emb = torch.tensor(self._ds[idx]["xvector"]).unsqueeze(0)
            except Exception as exc:
                logger.warning("x-vector lookup failed for %s (%s); using fallback", voice, exc)
                emb = None
        if emb is None:
            emb = self._pseudo_embedding(voice)
        self._cache[voice] = emb
        return emb


__all__ = ["SpeakerBank"]
