"""Evaluate the TN normalizer: neural vs rule baseline, overall + per semiotic class.

Produces the headline "model beats baseline" comparison on the held-out test set
and on the ambiguous ``hard`` slice, with per-class exact-match accuracy. Runs
baseline-only (no torch needed) when a trained model is unavailable.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..models.baseline_rules import RuleNormalizer
from ..models.normalizer import load_normalizer
from ..data.dataset import load_or_build

logger = get_logger(__name__)


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _score(normalizer, rows: List[Dict]) -> Dict:
    befores = [r["before"] for r in rows]
    golds = [_ws(r["after"]) for r in rows]
    try:
        preds = [_ws(p) for p in normalizer.normalize_batch(befores)]
    except Exception as exc:
        logger.warning("normalize_batch failed (%s); falling back per-item", exc)
        preds = [_ws(normalizer.normalize(b)) for b in befores]
    correct = [p == g for p, g in zip(preds, golds)]
    n = len(rows) or 1
    per_class_hits: Dict[str, List[int]] = defaultdict(list)
    for r, ok in zip(rows, correct):
        for c in (r.get("classes") or ["PLAIN"]):
            per_class_hits[c].append(1 if ok else 0)
    per_class = {c: round(sum(v) / len(v), 4) for c, v in sorted(per_class_hits.items())}
    macro = round(sum(per_class.values()) / len(per_class), 4) if per_class else 0.0
    return {"n": len(rows), "sentence_accuracy": round(sum(correct) / n, 4),
            "macro_class_accuracy": macro, "per_class": per_class}


def evaluate(cfg: AppConfig, which: str = "test", limit: Optional[int] = None,
             save: bool = True) -> Dict:
    splits = load_or_build(cfg)
    rows = splits.get(which, [])
    if limit:
        rows = rows[:limit]
    hard = splits.get("hard", [])
    if limit:
        hard = hard[:limit]

    rule = RuleNormalizer()
    neural = load_normalizer(cfg.model, prefer="neural")
    neural_available = not isinstance(neural, RuleNormalizer)

    result: Dict = {
        "which": which, "neural_available": neural_available,
        "baseline": {"test": _score(rule, rows), "hard": _score(rule, hard)},
    }
    if neural_available:
        n_test, n_hard = _score(neural, rows), _score(neural, hard)
        result["neural"] = {"test": n_test, "hard": n_hard,
                            "version": getattr(neural, "version", "?")}
        result["delta"] = {
            "test_sentence_accuracy": round(n_test["sentence_accuracy"] - result["baseline"]["test"]["sentence_accuracy"], 4),
            "hard_sentence_accuracy": round(n_hard["sentence_accuracy"] - result["baseline"]["hard"]["sentence_accuracy"], 4),
        }
        result["summary"] = {
            "neural_test_acc": n_test["sentence_accuracy"],
            "baseline_test_acc": result["baseline"]["test"]["sentence_accuracy"],
            "neural_hard_acc": n_hard["sentence_accuracy"],
            "baseline_hard_acc": result["baseline"]["hard"]["sentence_accuracy"],
            "beats_baseline": n_test["sentence_accuracy"] >= result["baseline"]["test"]["sentence_accuracy"],
        }
    else:
        result["summary"] = {
            "baseline_test_acc": result["baseline"]["test"]["sentence_accuracy"],
            "baseline_hard_acc": result["baseline"]["hard"]["sentence_accuracy"],
            "note": "no trained model found; reporting rule baseline only",
        }

    if save:
        out = run_dir() / "eval"
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"eval-{which}-{utc_stamp()}.json"
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")
        (out / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Eval saved -> %s", p)
    return result


__all__ = ["evaluate"]
