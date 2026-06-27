"""Build, cache and load the Text-Normalization dataset splits.

Primary source = the deterministic synthetic generator (``tn_corpus``). Produces
leakage-free ``train / val / test`` splits plus a separate ambiguous ``hard``
evaluation slice, cached as JSONL under ``data_dir/tn_corpus``. Optionally mixes
in a real HF (written, spoken) dataset. ``datasets`` is imported lazily so the
package stays importable on core deps.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, DataConfig, data_dir
from ..logging_utils import get_logger
from .tn_corpus import TNCorpusGenerator

logger = get_logger(__name__)

_SPLITS = ("train", "val", "test", "hard")


def corpus_dir() -> Path:
    return data_dir() / "tn_corpus"


def corpus_signature(dc: DataConfig) -> Dict[str, Any]:
    return {
        "source": "synthetic+real" if dc.hf_tn_dataset else "synthetic",
        "hf_tn_dataset": dc.hf_tn_dataset or None,
        "train": dc.synthetic_train_size, "val": dc.synthetic_val_size,
        "test": dc.synthetic_test_size, "hard": dc.hard_slice_size, "seed": dc.seed,
    }


def _write_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_real_tn(dc: DataConfig, limit: Optional[int] = None) -> List[Dict]:
    """Optionally load a real HF (written, spoken) TN dataset. Returns [] on any failure."""
    if not dc.hf_tn_dataset:
        return []
    try:
        from datasets import load_dataset  # lazy
        ds = load_dataset(dc.hf_tn_dataset, dc.hf_tn_config or None, split="train")
        rows: List[Dict] = []
        for i, ex in enumerate(ds):
            if limit and i >= limit:
                break
            before = str(ex.get(dc.hf_tn_text_col, "")).strip()
            after = str(ex.get(dc.hf_tn_target_col, "")).strip()
            if before and after:
                rows.append({"before": before, "after": after, "classes": ["REAL"], "hard": False})
        logger.info("Loaded %d rows from real TN dataset %s", len(rows), dc.hf_tn_dataset)
        return rows
    except Exception as exc:
        logger.warning("Could not load real TN dataset %s (%s); synthetic only.", dc.hf_tn_dataset, exc)
        return []


def build_corpus(cfg: AppConfig, save: bool = True) -> Dict[str, List[Dict]]:
    """Generate the synthetic corpus, build leakage-free splits, optionally cache."""
    dc = cfg.data
    gen = TNCorpusGenerator(seed=dc.seed)
    total = dc.synthetic_train_size + dc.synthetic_val_size + dc.synthetic_test_size
    pool = [e.as_dict() for e in gen.generate(total, seed=dc.seed)]
    rng = random.Random(dc.seed)
    rng.shuffle(pool)

    real = load_real_tn(dc)
    n_tr, n_va = dc.synthetic_train_size, dc.synthetic_val_size
    train = pool[:n_tr] + real
    rng.shuffle(train)
    val = pool[n_tr:n_tr + n_va]
    test = pool[n_tr + n_va:]

    # prevent leakage: no val/test input may appear among training inputs
    train_before = {r["before"] for r in train}
    val = [r for r in val if r["before"] not in train_before]
    test = [r for r in test if r["before"] not in train_before]

    hard = [e.as_dict() for e in gen.generate_hard(dc.hard_slice_size, seed=dc.seed)]
    hard = [r for r in hard if r["before"] not in train_before]

    splits = {"train": train, "val": val, "test": test, "hard": hard}
    logger.info("Corpus built: %s", {k: len(v) for k, v in splits.items()})
    if save:
        for name, rows in splits.items():
            _write_jsonl(rows, corpus_dir() / f"{name}.jsonl")
        (corpus_dir() / "signature.json").write_text(
            json.dumps(corpus_signature(dc), indent=2), encoding="utf-8")
    return splits


def load_or_build(cfg: AppConfig, force: bool = False) -> Dict[str, List[Dict]]:
    cdir = corpus_dir()
    if not force and all((cdir / f"{s}.jsonl").exists() for s in _SPLITS):
        try:
            sig = json.loads((cdir / "signature.json").read_text(encoding="utf-8"))
            if sig.get("seed") == cfg.data.seed and sig.get("train") == cfg.data.synthetic_train_size:
                return {s: _read_jsonl(cdir / f"{s}.jsonl") for s in _SPLITS}
        except Exception:
            pass
    return build_corpus(cfg)


def to_hf_datasets(splits: Dict[str, List[Dict]]):
    """Convert split dicts to a ``datasets.DatasetDict`` (lazy import)."""
    from datasets import Dataset, DatasetDict  # lazy
    out = {}
    for name, rows in splits.items():
        if not rows:
            continue
        out[name] = Dataset.from_list([{"before": r["before"], "after": r["after"],
                                        "classes": r.get("classes", []), "hard": bool(r.get("hard", False))}
                                       for r in rows])
    return DatasetDict(out)


__all__ = ["build_corpus", "load_or_build", "to_hf_datasets", "load_real_tn",
           "corpus_signature", "corpus_dir"]
