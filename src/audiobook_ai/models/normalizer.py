"""Neural Text-Normalization model wrapper (the TRAINED core).

Wraps a fine-tuned ByT5 / T5 seq2seq checkpoint behind the same ``normalize`` /
``normalize_batch`` interface as the rule baseline, and exposes a per-example
**confidence** (length-normalized sequence probability) used by the agent's D2
decision point. All heavy imports (torch / transformers) are lazy so the package
stays importable on core deps; ``load_normalizer`` degrades gracefully to the
rule baseline when transformers or a checkpoint is unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import ModelConfig
from ..logging_utils import get_logger
from .baseline_rules import RuleNormalizer

logger = get_logger(__name__)


class NeuralNormalizer:
    """Fine-tuned seq2seq normalizer (ByT5/T5)."""

    name = "neural_tn"

    def __init__(self, model, tokenizer, *, prefix: str, device: str,
                 max_source_length: int, max_target_length: int, version: str, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.device = device
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.version = version
        self.model_id = model_id

    # ---- factory -----------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_path: str, cfg: ModelConfig,
                        device: Optional[str] = None) -> "NeuralNormalizer":
        import torch  # lazy
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(dev)
        model.eval()
        version = _read_version(model_path)
        logger.info("Loaded neural normalizer from %s (device=%s)", model_path, dev)
        return cls(model, tok, prefix=cfg.task_prefix, device=dev,
                   max_source_length=cfg.max_source_length,
                   max_target_length=cfg.max_target_length,
                   version=version, model_id=str(model_path))

    # ---- inference ---------------------------------------------------------
    def _generate(self, texts: List[str], with_conf: bool) -> Tuple[List[str], List[float]]:
        import torch  # lazy
        inputs = [self.prefix + t for t in texts]
        enc = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True,
                             max_length=self.max_source_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        gen_kwargs = dict(max_new_tokens=self.max_target_length, num_beams=1, do_sample=False)
        if with_conf:
            gen_kwargs.update(output_scores=True, return_dict_in_generate=True)
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        if with_conf:
            seqs = out.sequences
            confs = self._sequence_confidence(out, seqs)
        else:
            seqs = out
            confs = [1.0] * len(texts)
        decoded = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        return [d.strip() for d in decoded], confs

    def _sequence_confidence(self, out, seqs) -> List[float]:
        import torch  # lazy
        try:
            trans = self.model.compute_transition_scores(seqs, out.scores, normalize_logits=True)
            confs: List[float] = []
            for row in trans:
                vals = [v.item() for v in row if not math.isinf(v.item()) and not math.isnan(v.item())]
                confs.append(float(math.exp(sum(vals) / len(vals))) if vals else 0.0)
            return confs
        except Exception:
            return [1.0] * seqs.shape[0]

    def normalize(self, text: str) -> str:
        return self._generate([text], with_conf=False)[0][0]

    def normalize_with_conf(self, text: str) -> Tuple[str, float]:
        outs, confs = self._generate([text], with_conf=True)
        return outs[0], confs[0]

    def normalize_batch(self, texts: List[str], batch_size: int = 32) -> List[str]:
        res: List[str] = []
        for i in range(0, len(texts), batch_size):
            res.extend(self._generate(texts[i:i + batch_size], with_conf=False)[0])
        return res

    def normalize_batch_with_conf(self, texts: List[str], batch_size: int = 32
                                  ) -> Tuple[List[str], List[float]]:
        outs: List[str] = []
        confs: List[float] = []
        for i in range(0, len(texts), batch_size):
            o, c = self._generate(texts[i:i + batch_size], with_conf=True)
            outs.extend(o)
            confs.extend(c)
        return outs, confs


def _read_version(model_path: str) -> str:
    meta = Path(model_path) / "tn_meta.json"
    if meta.exists():
        try:
            import json
            return json.loads(meta.read_text(encoding="utf-8")).get("version", "neural-1.0")
        except Exception:
            pass
    return "neural-1.0"


def default_model_path(cfg: ModelConfig) -> Path:
    """``model_dir/tn_normalizer/latest`` if present, else the output dir itself."""
    latest = cfg.output_dir / "latest"
    return latest if latest.exists() else cfg.output_dir


def load_normalizer(cfg: ModelConfig, *, prefer: str = "neural", device: Optional[str] = None):
    """Return the best available normalizer (neural if possible, else rule baseline).

    Always succeeds — the rule baseline has no dependencies and never raises.
    """
    if prefer == "rule":
        return RuleNormalizer()
    path = default_model_path(cfg)
    if not Path(path).exists():
        logger.info("No fine-tuned TN checkpoint at %s; using rule baseline.", path)
        return RuleNormalizer()
    try:
        return NeuralNormalizer.from_pretrained(str(path), cfg, device=device)
    except Exception as exc:
        logger.warning("Could not load neural normalizer (%s); using rule baseline.", exc)
        return RuleNormalizer()


__all__ = ["NeuralNormalizer", "RuleNormalizer", "load_normalizer", "default_model_path"]
