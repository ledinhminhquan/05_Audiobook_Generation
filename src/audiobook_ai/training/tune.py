"""Lightweight hyperparameter search for the TN model.

Sweeps a small learning-rate grid on a reduced data/epoch budget and keeps the
setting with the best validation sentence-accuracy. Deliberately cheap (the
assignment asks for *basic* tuning); the full run then uses the winning LR.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)

_DEFAULT_GRID: List[float] = [3.0e-4, 5.0e-4, 1.0e-3]


def tune_normalizer(cfg: AppConfig, n_trials: int = 3, limit: int = 4000,
                    epochs: int = 1, grid: Optional[List[float]] = None) -> Dict:
    from .train_normalizer import train_normalizer

    lrs = (grid or _DEFAULT_GRID)[:n_trials]
    trials: List[Dict] = []
    best = None
    for lr in lrs:
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg.model.learning_rate = lr
        trial_cfg.model.num_train_epochs = epochs
        trial_cfg.model.output_subdir = f"tn_tune/lr_{lr:.0e}"
        trial_cfg.model.eval_steps = max(100, trial_cfg.model.eval_steps // 2)
        trial_cfg.model.save_steps = trial_cfg.model.eval_steps
        try:
            res = train_normalizer(trial_cfg, limit=limit, resume=False)
            acc = res["metrics"].get("eval_sentence_accuracy", 0.0)
        except Exception as exc:
            logger.warning("trial lr=%s failed: %s", lr, exc)
            acc = 0.0
            res = {"error": str(exc)}
        rec = {"learning_rate": lr, "eval_sentence_accuracy": acc, "result": res}
        trials.append(rec)
        if best is None or acc > best["eval_sentence_accuracy"]:
            best = rec
        logger.info("trial lr=%.0e -> sentence_accuracy=%.4f", lr, acc)

    out = {"best": {"learning_rate": best["learning_rate"],
                    "eval_sentence_accuracy": best["eval_sentence_accuracy"]} if best else None,
           "trials": [{"learning_rate": t["learning_rate"],
                       "eval_sentence_accuracy": t["eval_sentence_accuracy"]} for t in trials]}
    d = run_dir() / "tune"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"tune-{utc_stamp()}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (d / "best.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


__all__ = ["tune_normalizer"]
