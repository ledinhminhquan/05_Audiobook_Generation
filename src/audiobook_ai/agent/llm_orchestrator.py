"""Optional LLM reasoning brain for the agent (anthropic), with rule fallback.

Used only at escalation hooks (the D2 normalization decision). It is **never in
the control path**: disabled by default, validates its own output, and on any
problem (missing key, timeout, invalid JSON, exception) the caller keeps the
rule/neural result. Default deployment makes zero paid API calls.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from ..config import AgentConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class LLMBrain:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._client = None
        self._tried = False

    def available(self) -> bool:
        if not self.cfg.llm_fallback_enabled:
            return False
        return bool(os.environ.get(self.cfg.llm_api_key_env))

    def _get_client(self):
        if self._tried:
            return self._client
        self._tried = True
        try:
            import anthropic  # lazy
            key = os.environ.get(self.cfg.llm_api_key_env)
            if not key:
                return None
            self._client = anthropic.Anthropic(api_key=key)
        except Exception as exc:
            logger.info("anthropic client unavailable (%s)", exc)
            self._client = None
        return self._client

    def disambiguate_normalization(self, sentence: str, neural_text: str,
                                   baseline_text: str) -> Optional[Dict[str, Any]]:
        """Return ``{text, confidence, rationale}`` or ``None`` to keep the existing result."""
        if not self.available():
            return None
        client = self._get_client()
        if client is None:
            return None
        prompt = (
            "You normalize written English into spoken form for a text-to-speech audiobook "
            "(expand numbers, money, dates, abbreviations; resolve ambiguous tokens like 'St.' "
            "= Saint vs Street using context).\n\n"
            f"Sentence: {sentence}\n"
            f"Candidate A (model): {neural_text}\n"
            f"Candidate B (rules): {baseline_text}\n\n"
            "Reply with ONLY a JSON object: "
            '{"text": "<best spoken form of the full sentence>", "confidence": <0..1>, '
            '"rationale": "<short>"}.'
        )
        try:
            msg = client.messages.create(
                model=self.cfg.llm_model, max_tokens=400, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(b, "text", "") for b in msg.content)
            data = _extract_json(text)
            if data and isinstance(data.get("text"), str) and data["text"].strip():
                return {"text": data["text"].strip(),
                        "confidence": float(data.get("confidence", 0.7)),
                        "rationale": str(data.get("rationale", ""))[:200]}
        except Exception as exc:
            logger.info("LLM disambiguation failed (%s); keeping existing result", exc)
        return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


__all__ = ["LLMBrain"]
