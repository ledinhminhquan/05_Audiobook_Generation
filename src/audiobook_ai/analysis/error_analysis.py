"""Per-semiotic-class error analysis: where the normalizer fails and how badly.

Compares the model under test (neural if available, else the rule baseline)
against gold on the test + hard slices, breaks errors down by semiotic class,
collects example mistakes, and tallies the Sproat-style "unrecoverable" errors
(numeric/quantity mis-reads that change meaning for a listener).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..models.baseline_rules import RuleNormalizer
from ..models.normalizer import load_normalizer
from ..data.dataset import load_or_build

logger = get_logger(__name__)

_NUMERIC = {"CARDINAL", "DECIMAL", "MONEY", "MEASURE", "DATE", "TIME", "TELEPHONE", "ORDINAL", "FRACTION", "ROMAN"}


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def _analyze(normalizer, rows: List[Dict], max_examples: int = 12) -> Dict:
    befores = [r["before"] for r in rows]
    golds = [_ws(r["after"]) for r in rows]
    try:
        preds = [_ws(p) for p in normalizer.normalize_batch(befores)]
    except Exception:
        preds = [_ws(normalizer.normalize(b)) for b in befores]

    per_class = defaultdict(lambda: {"n": 0, "errors": 0})
    examples: List[Dict] = []
    unrecoverable: List[Dict] = []
    for r, g, p in zip(rows, golds, preds):
        ok = g == p
        for c in (r.get("classes") or ["PLAIN"]):
            per_class[c]["n"] += 1
            if not ok:
                per_class[c]["errors"] += 1
        if not ok:
            if len(examples) < max_examples:
                examples.append({"before": r["before"], "gold": g, "pred": p,
                                 "classes": r.get("classes", []), "hard": r.get("hard", False)})
            # unrecoverable heuristic: numeric class AND the spoken digits/quantity changed
            if any(c in _NUMERIC for c in r.get("classes", [])):
                unrecoverable.append({"before": r["before"], "gold": g, "pred": p})

    per_class_out = {c: {"n": v["n"], "errors": v["errors"],
                         "accuracy": round(1 - v["errors"] / max(1, v["n"]), 4)}
                     for c, v in sorted(per_class.items())}
    n_err = sum(v["errors"] for v in per_class.values())
    return {"n": len(rows), "errors": n_err,
            "per_class": per_class_out, "examples": examples,
            "unrecoverable": {"count": len(unrecoverable), "rate": round(len(unrecoverable) / max(1, len(rows)), 4),
                              "examples": unrecoverable[:max_examples]}}


def error_analysis(cfg: AppConfig, limit: Optional[int] = None, save: bool = True) -> Dict:
    splits = load_or_build(cfg)
    test = splits.get("test", [])
    hard = splits.get("hard", [])
    if limit:
        test, hard = test[:limit], hard[:limit]

    rule = RuleNormalizer()
    neural = load_normalizer(cfg.model, prefer="neural")
    neural_available = not isinstance(neural, RuleNormalizer)
    model = neural if neural_available else rule

    result = {
        "model": getattr(model, "name", "rule"),
        "neural_available": neural_available,
        "test": _analyze(model, test),
        "hard": _analyze(model, hard),
        "baseline_hard": _analyze(rule, hard) if neural_available else None,
    }
    if save:
        d = run_dir() / "error_analysis"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"errors-{utc_stamp()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("error analysis saved (test errors=%d, hard errors=%d)",
                    result["test"]["errors"], result["hard"]["errors"])
    return result


__all__ = ["error_analysis"]
