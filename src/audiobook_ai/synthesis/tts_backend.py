"""Text-to-speech backends behind a uniform interface.

``BaseTTS.synthesize(text, voice) -> TTSResult`` chunks long text, synthesizes
each chunk and concatenates. Backends, in fallback order:

* ``SpeechT5TTS``  – PRIMARY: microsoft/speecht5_tts + speecht5_hifigan + x-vectors (MIT)
* ``KokoroTTS``    – optional Apache-2.0 quality upgrade (24 kHz)
* ``ParlerTTS``    – optional Apache-2.0 prompt-styled voices
* ``Pyttsx3TTS``   – offline OS TTS (no model download)
* ``PlaceholderTTS`` – ultimate floor: deterministic low-noise audio so the
  pipeline (and tests/CI) always produce a valid waveform with no model at all.

All heavy imports are lazy; ``load_tts_backend`` never raises.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import TtsConfig
from ..logging_utils import get_logger
from .voices import SpeakerBank

logger = get_logger(__name__)

_SEC_PER_CHAR = 0.06


@dataclass
class TTSResult:
    audio: np.ndarray          # float32 mono in [-1, 1]
    sample_rate: int
    backend: str
    voice: str

    @property
    def duration(self) -> float:
        return len(self.audio) / float(self.sample_rate) if self.sample_rate else 0.0


def _chunk_text(text: str, max_chars: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sents = re.findall(r"[^.!?]*[.!?]+|\S[^.!?]*$", text)
    chunks: List[str] = []
    cur = ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(s) > max_chars:                 # split a very long clause on commas/spaces
            if cur:
                chunks.append(cur); cur = ""
            words, w = s.split(), ""
            for tok in words:
                if len(w) + len(tok) + 1 <= max_chars:
                    w = (w + " " + tok).strip()
                else:
                    chunks.append(w); w = tok
            if w:
                chunks.append(w)
        elif not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur += " " + s
        else:
            chunks.append(cur); cur = s
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


class BaseTTS:
    name = "base"

    def __init__(self, cfg: TtsConfig, sample_rate: int, max_chars: int):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.max_chars = max_chars

    def _synth_one(self, text: str, voice: str) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def synthesize(self, text: str, voice: str = "narrator") -> TTSResult:
        chunks = _chunk_text(text, self.max_chars)
        if not chunks:
            return TTSResult(np.zeros(0, dtype=np.float32), self.sample_rate, self.name, voice)
        gap = np.zeros(int(0.12 * self.sample_rate), dtype=np.float32)
        pieces: List[np.ndarray] = []
        for i, c in enumerate(chunks):
            try:
                wav = self._synth_one(c, voice).astype(np.float32)
            except Exception as exc:
                logger.warning("%s failed on a chunk (%s); inserting silence", self.name, exc)
                wav = np.zeros(int(len(c) * _SEC_PER_CHAR * self.sample_rate), dtype=np.float32)
            pieces.append(wav)
            if i < len(chunks) - 1:
                pieces.append(gap)
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        return TTSResult(audio, self.sample_rate, self.name, voice)


class PlaceholderTTS(BaseTTS):
    """Deterministic low-amplitude noise sized to the text (no model required)."""
    name = "placeholder"

    def __init__(self, cfg: TtsConfig):
        super().__init__(cfg, sample_rate=cfg.sample_rate, max_chars=cfg.max_chars_per_chunk)

    def _synth_one(self, text: str, voice: str) -> np.ndarray:
        n = max(int(len(text) * _SEC_PER_CHAR * self.sample_rate), int(0.3 * self.sample_rate))
        seed = int(hashlib.sha256((voice + "|" + text).encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        env = np.linspace(0.6, 1.0, n)
        return (rng.standard_normal(n) * 0.02 * env).astype(np.float32)


class Pyttsx3TTS(BaseTTS):
    name = "pyttsx3"

    def __init__(self, cfg: TtsConfig):
        super().__init__(cfg, sample_rate=22050, max_chars=cfg.max_chars_per_chunk)
        import pyttsx3  # lazy; raises if unavailable
        self._engine_factory = pyttsx3.init
        self._engine_factory().stop()  # smoke-test init

    def _synth_one(self, text: str, voice: str) -> np.ndarray:
        import tempfile, wave, os
        eng = self._engine_factory()
        path = tempfile.mktemp(suffix=".wav")
        eng.save_to_file(text, path)
        eng.runAndWait()
        try:
            with wave.open(path, "rb") as w:
                self.sample_rate = w.getframerate()
                frames = w.readframes(w.getnframes())
            arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return arr
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class SpeechT5TTS(BaseTTS):
    name = "speecht5"

    def __init__(self, cfg: TtsConfig, device: Optional[str] = None):
        super().__init__(cfg, sample_rate=16000, max_chars=min(cfg.max_chars_per_chunk, 480))
        import torch  # lazy
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SpeechT5Processor.from_pretrained(cfg.speecht5_processor)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(cfg.speecht5_model).to(self.device).eval()
        self.vocoder = SpeechT5HifiGan.from_pretrained(cfg.speecht5_vocoder).to(self.device).eval()
        self.bank = SpeakerBank(cfg)

    def _synth_one(self, text: str, voice: str) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        spk = self.bank.get(voice).to(self.device)
        with torch.no_grad():
            speech = self.model.generate_speech(inputs["input_ids"], spk, vocoder=self.vocoder)
        return speech.detach().cpu().numpy().astype(np.float32)


class KokoroTTS(BaseTTS):
    name = "kokoro"

    def __init__(self, cfg: TtsConfig):
        super().__init__(cfg, sample_rate=24000, max_chars=cfg.max_chars_per_chunk)
        from kokoro import KPipeline  # lazy
        self._pipe = KPipeline(lang_code="a")

    def _synth_one(self, text: str, voice: str) -> np.ndarray:
        chosen = self.cfg.kokoro_voice
        out = []
        for _, _, audio in self._pipe(text, voice=chosen):
            out.append(np.asarray(audio, dtype=np.float32))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class ParlerTTS(BaseTTS):
    name = "parler"

    def __init__(self, cfg: TtsConfig, device: Optional[str] = None):
        super().__init__(cfg, sample_rate=44100, max_chars=cfg.max_chars_per_chunk)
        import torch  # lazy
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(cfg.parler_model).to(self.device)
        self.tok = AutoTokenizer.from_pretrained(cfg.parler_model)
        self.sample_rate = self.model.config.sampling_rate

    def _synth_one(self, text: str, voice: str) -> np.ndarray:
        torch = self._torch
        desc = self.tok(self.cfg.parler_description, return_tensors="pt").to(self.device)
        prompt = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            gen = self.model.generate(input_ids=desc.input_ids, prompt_input_ids=prompt.input_ids)
        return gen.cpu().numpy().squeeze().astype(np.float32)


_BACKENDS = {"speecht5": SpeechT5TTS, "kokoro": KokoroTTS, "parler": ParlerTTS, "pyttsx3": Pyttsx3TTS}


def load_tts_backend(cfg: TtsConfig, backend: Optional[str] = None,
                     device: Optional[str] = None) -> BaseTTS:
    """Return a working TTS backend, falling back until one initializes."""
    requested = backend or cfg.backend
    if requested == "placeholder":
        return PlaceholderTTS(cfg)
    order: List[str] = []
    if requested and requested != "auto":
        order.append(requested)
    for b in ("speecht5", "pyttsx3"):
        if b not in order:
            order.append(b)
    for name in order:
        cls = _BACKENDS.get(name)
        if cls is None:
            continue
        try:
            inst = cls(cfg, device=device) if name in ("speecht5", "parler") else cls(cfg)
            logger.info("TTS backend: %s (%d Hz)", inst.name, inst.sample_rate)
            return inst
        except Exception as exc:
            logger.info("TTS backend %s unavailable (%s); trying next", name, exc)
    logger.info("Falling back to placeholder TTS (no model).")
    return PlaceholderTTS(cfg)


__all__ = ["TTSResult", "BaseTTS", "SpeechT5TTS", "KokoroTTS", "ParlerTTS",
           "Pyttsx3TTS", "PlaceholderTTS", "load_tts_backend"]
