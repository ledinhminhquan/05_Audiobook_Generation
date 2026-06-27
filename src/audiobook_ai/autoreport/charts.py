"""Matplotlib charts for the report/slides. Returns saved PNG paths; if matplotlib
is unavailable every function degrades to returning ``None`` (no chart)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import get_logger

logger = get_logger(__name__)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def baseline_vs_neural_chart(eval_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not eval_art:
        return None
    try:
        plt = _mpl()
        b = eval_art.get("baseline", {})
        n = eval_art.get("neural", {})
        cats = ["test", "hard"]
        base_vals = [b.get(c, {}).get("sentence_accuracy", 0) for c in cats]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        x = range(len(cats))
        if n:
            neu_vals = [n.get(c, {}).get("sentence_accuracy", 0) for c in cats]
            ax.bar([i - 0.2 for i in x], base_vals, width=0.4, label="rule baseline", color="#9aa7b4")
            ax.bar([i + 0.2 for i in x], neu_vals, width=0.4, label="ByT5 (trained)", color="#2b6cb0")
        else:
            ax.bar(list(x), base_vals, width=0.5, label="rule baseline", color="#9aa7b4")
        ax.set_xticks(list(x)); ax.set_xticklabels([c.upper() for c in cats])
        ax.set_ylabel("sentence exact-match acc"); ax.set_ylim(0, 1)
        ax.set_title("Text Normalization: model vs baseline"); ax.legend()
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("baseline_vs_neural_chart skipped (%s)", exc)
        return None


def per_class_chart(error_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not error_art:
        return None
    try:
        plt = _mpl()
        pc = (error_art.get("hard", {}) or error_art.get("test", {})).get("per_class", {})
        if not pc:
            return None
        items = sorted(pc.items(), key=lambda kv: kv[1].get("accuracy", 0))
        labels = [k for k, _ in items]
        vals = [v.get("accuracy", 0) for _, v in items]
        fig, ax = plt.subplots(figsize=(6.5, max(3.2, 0.32 * len(labels))))
        ax.barh(labels, vals, color="#2b6cb0")
        ax.set_xlim(0, 1); ax.set_xlabel("exact-match accuracy")
        ax.set_title("Per-semiotic-class accuracy")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("per_class_chart skipped (%s)", exc)
        return None


def qa_chart(audio_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not audio_art:
        return None
    try:
        plt = _mpl()
        keys = ["pass_rate", "avg_duration_ratio"]
        vals = [audio_art.get(k) or 0 for k in keys]
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.bar(["QA pass-rate", "avg dur ratio"], vals, color=["#2f855a", "#dd6b20"])
        ax.set_title("Audio QA"); fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("qa_chart skipped (%s)", exc)
        return None


def build_all(arts: Dict[str, Any], out_dir: Path) -> List[Tuple[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    charts: List[Tuple[str, Path]] = []
    for name, fn, key in [("baseline_vs_neural", baseline_vs_neural_chart, "eval"),
                          ("per_class", per_class_chart, "error_analysis"),
                          ("qa", qa_chart, "audio_quality")]:
        p = fn(arts.get(key) or {}, out_dir / f"{name}.png")
        if p:
            charts.append((name, p))
    return charts


__all__ = ["baseline_vs_neural_chart", "per_class_chart", "qa_chart", "build_all"]
